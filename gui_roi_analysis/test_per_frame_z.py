"""Headless tests for per-frame z correction (gui_roi_fast_simple.py).

Unit tests use synthetic data. The optional integration test (--real PKL) rebuilds the
GUI TIFF of a few sets into a temporary folder, quantifies the same sets from the FLIM
files and checks that every pre/post frame uses identical planes in both: the ROI sum on
the GUI TIFF and the FLIM quantification must differ only by one constant scale
(TIFF decode normalisation), frame by frame. Session files are only read.

Run:
    python test_per_frame_z.py
    python test_per_frame_z.py --real G:\\ImagingData\\Tetsuya\\20260925\\auto1\\combined_df_respan.pkl [--sets 4]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gui_roi_fast_simple as g  # noqa: E402


def _set_df(shift_z, z_rel=7):
    phases = ["pre", "pre", "pre", "unc", "post", "post"]
    fps = [f"C:/x/f_{i:03d}.flim" for i in range(len(phases))]
    return pd.DataFrame({"phase": phases, "file_path": fps, "nth_omit_induction": [0, 1, 2, -1, 3, 4],
                         "z_relative_step_nth": [np.nan] * 3 + [z_rel] + [np.nan] * 2,
                         "shift_z": shift_z})


def test_centers_follow_shift_z():
    sz = [0.0, 1.2, 2.0, 2.0, 3.6, -0.4]
    df = _set_df(sz)
    shift_of = {r.file_path: r.shift_z for r in df.itertuples() if r.phase != "unc"}
    c, mode = g.per_frame_z_centers(df, shift_of, fallback_center=99)
    assert mode == g.Z_MODE_PER_FRAME
    # last pre (f_002) is read exactly at the uncaging plane
    assert c["C:/x/f_002.flim"] == 7
    # aligned = raw + shift: a frame with larger shift_z sees the tissue at a lower raw index
    assert c["C:/x/f_000.flim"] == 7 + round(2.0 - 0.0) == 9
    assert c["C:/x/f_001.flim"] == 7 + round(2.0 - 1.2) == 8
    assert c["C:/x/f_004.flim"] == 7 + round(2.0 - 3.6) == 5
    assert c["C:/x/f_005.flim"] == 7 + round(2.0 + 0.4) == 9


def test_same_plane_on_synthetic_volumes():
    """A bright plane at aligned z=A is found inside every per-frame raw window."""
    rng = np.random.default_rng(1)
    A = 10  # aligned plane of the spine
    sz = [0.0, -2.0, 1.0, 1.0, 3.0, -1.0]
    df = _set_df(sz, z_rel=A - 1)  # raw index in last pre = A - shift_z(last pre) = 9
    shift_of = {r.file_path: r.shift_z for r in df.itertuples() if r.phase != "unc"}
    c, _ = g.per_frame_z_centers(df, shift_of, 0)
    for fp, s in shift_of.items():
        raw_plane = A - int(s)  # raw = aligned - shift
        vol = rng.random((20, 4, 4)) * 0.1
        vol[raw_plane] += 1.0
        zf, zt = g.z_window(c[fp], 0, 20)
        assert zf <= raw_plane < zt, (fp, raw_plane, zf, zt)


def test_fallbacks():
    df = _set_df([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    shift_of = {r.file_path: r.shift_z for r in df.itertuples() if r.phase != "unc"}
    no_zrel = df.copy()
    no_zrel["z_relative_step_nth"] = np.nan
    c, mode = g.per_frame_z_centers(no_zrel, shift_of, 5)
    assert mode == g.Z_MODE_FIXED and set(c.values()) == {5}
    partial = dict(shift_of)
    partial.pop("C:/x/f_004.flim")
    assert g.per_frame_z_centers(df, partial, 5)[1] == g.Z_MODE_FIXED
    os.environ["FLIM_ROI_PER_FRAME_Z"] = "0"
    try:
        assert g.per_frame_z_centers(df, shift_of, 5)[1] == g.Z_MODE_FIXED
    finally:
        os.environ.pop("FLIM_ROI_PER_FRAME_Z")


def test_frame_info_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x_frame_info.csv")
        assert g._frame_info_z_windows(p) is None
        pd.DataFrame({"phase": ["pre", "uncaging", "post"], "filename": ["A_001.flim", "u.flim", "A_003.flim"],
                      "z_from": [3, np.nan, 5], "z_to": [6, np.nan, 8], "shift_y": [0, 0, 0]}).to_csv(p, index=False)
        assert g._frame_info_z_windows(p) is None  # old file without z_mode
        assert not g._frame_info_z_matches(p, [])
        fi = pd.read_csv(p)
        fi["z_mode"] = g.Z_MODE_PER_FRAME
        fi.to_csv(p, index=False)
        assert g._frame_info_z_windows(p) == {("pre", "a_001.flim"): (3, 6), ("post", "a_003.flim"): (5, 8)}
        assert g._frame_info_z_matches(p, [("A_001.flim", (3, 6, 4, 0.0)), ("A_003.flim", (5, 8, 6, 0.0))])
        assert not g._frame_info_z_matches(p, [("A_001.flim", (3, 6, 4, 0.0)), ("A_003.flim", (4, 7, 5, 0.0))])


def real_integration(pkl: str, n_sets: int) -> None:
    import tifffile

    import flim_fast_io as ffi

    ffi._save_cached_intensity = lambda *a, **k: None  # never write into the session folder
    df_all = pd.read_pickle(pkl)
    keys = ["filepath_without_number", "group", "nth_set_label"]
    sets = [k for k, d in df_all[df_all.nth_set_label != -1].groupby(keys, sort=False)
            if "z_relative_step_nth" in d.columns and (d.phase == "unc").any()]
    # prefer sets with the largest z drift within the set
    def spread(k):
        d = df_all.set_index(keys).loc[k]
        s = pd.to_numeric(d[d.phase.isin(["pre", "post"])].shift_z, errors="coerce")
        return float(s.max() - s.min())
    def rejected(k):
        d = df_all.set_index(keys).loc[k]
        return "reject" in d.columns and bool((d["reject"] == 1).any() or (d["reject"] == True).any())
    # non-rejected sets with a real (1-6 slice) z drift inside the set, largest first
    sets = [k for k in sets if not rejected(k) and 1.0 <= spread(k) <= 6.0]
    sets = sorted(sets, key=spread, reverse=True)[:n_sets]
    print("sets:", [(k[1], k[2], round(spread(k), 2)) for k in sets])
    with tempfile.TemporaryDirectory() as td:
        cdf = df_all.copy()
        for k in sets:
            sdf = cdf[(cdf[keys[0]] == k[0]) & (cdf.group == k[1]) & (cdf.nth_set_label == k[2])]
            fl = sdf[sdf.phase.isin(["pre", "post"])].file_path.tolist()
            shp = ffi.load_flim_intensity(fl[0], use_cache=True)[0].shape
            Z, Y, X = int(shp[0]), int(shp[-2]), int(shp[-1])
            g._rebuild_one_set_full_size(
                combined_df=cdf, each_set_df=sdf, each_set_label=k[2], each_group=k[1], ch=2, z_plus_minus=1,
                Z_full=Z, Y_full=Y, X_full=X, Aligned_4d_array=np.empty((0, Z, 1, 1)), file_path_to_array_idx={},
                n_aligned=0,
                runtime_shift_map={r.file_path: (float(r.shift_y), float(r.shift_x)) for r in sdf.itertuples()
                                   if r.phase in ("pre", "post")},
                tif_savefolder=td, skip_tiff_if_exists=False, error_log=[], fast_mode=True, intensity_cache={},
                runtime_shift_z_map={r.file_path: float(r.shift_z) for r in sdf.itertuples()
                                     if r.phase in ("pre", "post")},
            )
        sub = cdf.set_index(keys).loc[sets].reset_index()
        sub["after_align_save_path"] = sub["after_align_full_save_path"]
        sub["reject"] = 0  # test only: rejected sets are fine for the plane check
        # Spine ROI: a 9x9 square at the uncaging point in every GUI frame (Type A), then raw (Type B)
        for k in sets:
            sdf = sub[(sub[keys[0]] == k[0]) & (sub.group == k[1]) & (sub.nth_set_label == k[2])]
            tiff = sdf.after_align_save_path.iloc[0]
            stack = tifffile.imread(tiff)
            cy, cx = int(sdf.corrected_uncaging_y.iloc[0]), int(sdf.corrected_uncaging_x.iloc[0])
            m = np.zeros(stack.shape, np.uint8)
            m[:, max(cy - 4, 0):cy + 5, max(cx - 4, 0):cx + 5] = 1
            base = os.path.splitext(tiff)[0]
            for t in g.ROI_TYPES:
                tifffile.imwrite(f"{base}_{t}_roi_mask.tif", m)
        g.save_drift_corrected_roi_masks(sub)
        out_csv = os.path.join(td, "quant.csv")
        g.quantify_intensity_from_flim(sub, 2, 1, out_csv, skip_lifetime_analysis=True)
        q = pd.read_csv(out_csv)
        n_ok = 0
        for k in sets:
            sdf = sub[(sub[keys[0]] == k[0]) & (sub.group == k[1]) & (sub.nth_set_label == k[2])]
            tiff = sdf.after_align_save_path.iloc[0]
            stack = tifffile.imread(tiff).astype(np.float64)
            m = tifffile.imread(os.path.splitext(tiff)[0] + "_Spine_roi_mask.tif") > 0
            fi = pd.read_csv(os.path.splitext(tiff)[0] + "_frame_info.csv")
            qq = q[(q.group == k[1]) & (q.set_label == k[2])].reset_index(drop=True)
            ratios, zs = [], []
            for i, r in fi.iterrows():
                if r.phase not in ("pre", "post"):
                    continue
                qi = qq[qq.file_path.map(lambda p: os.path.basename(str(p)).lower()) == r.filename.lower()]
                assert len(qi) == 1, r.filename
                assert int(qi.z_from.iloc[0]) == int(r.z_from) and int(qi.z_to.iloc[0]) == int(r.z_to), \
                    (r.filename, qi.z_from.iloc[0], r.z_from)
                tiff_sum = stack[i][m[i]].sum()
                ratios.append(float(qi.Spine_Ch2_intensity.iloc[0]) / tiff_sum)
                zs.append(f"{int(r.z_from)}-{int(r.z_to) - 1}")
                n_ok += 1
            ratios = np.array(ratios)
            rel = ratios.max() / ratios.min() - 1
            print(f"  {k[1]} set {k[2]}: z mode {fi.z_mode.iloc[0]}, windows {zs}, "
                  f"quant/TIFF ratio {ratios.mean():.6g} (max rel spread {rel:.2e})")
            assert rel < 1e-5, "GUI TIFF and FLIM quantification use different pixels/planes"
        print(f"real integration: {n_ok} pre/post frames identical in GUI TIFF and FLIM quantification")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None)
    ap.add_argument("--sets", type=int, default=4)
    a = ap.parse_args()
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
            n += 1
    print(f"all {n} unit tests passed")
    if a.real:
        real_integration(a.real, a.sets)
