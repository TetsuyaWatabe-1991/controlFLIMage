"""Headless tests for alignment_correlation.py (and its display in the set review panel).

Unit: pearson_overlap is 1 for an exactly shifted copy (overlap only) and lower for a
wrong shift. --real SET_FOLDER PKL: a prepared set folder (combined_df_one_set.pkl + tif/,
copied to a temporary folder) with the session pkl; r_high (vs the set's Pre 1, as shown in
the TIFF) is recomputed independently from the raw FLIM files with numpy (no filtering) and
must match; the numbers must be drawn on the panel. Writing the intensity cache is disabled
(session files are only read).

Run:
    python test_alignment_correlation.py
    python test_alignment_correlation.py --real <set folder> <session combined_df_respan.pkl>
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.dirname(HERE))
import alignment_correlation as ac  # noqa: E402


def test_pearson_overlap():
    rng = np.random.default_rng(0)
    ref = rng.random((64, 64))
    for dy, dx in ((0, 0), (3, -5), (-7, 2)):
        mov = np.zeros_like(ref)
        # mov moved by (+dy, +dx) equals ref: mov[y, x] = ref[y + dy, x + dx]
        h, w = ref.shape
        mov[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)] = ref[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
        assert abs(ac.pearson_overlap(ref, mov, dy, dx) - 1.0) < 1e-12, (dy, dx)
        assert ac.pearson_overlap(ref, mov, dy + 2, dx) < 0.5
    assert np.isnan(ac.pearson_overlap(np.ones((10, 10)), np.ones((10, 10)), 0, 0))


def real_check(set_folder: str, session_pkl: str) -> None:
    import flim_fast_io as ffi
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import set_review_panel as P

    ffi._save_cached_intensity = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "set")
        shutil.copytree(set_folder, dst)
        sdf = pd.read_pickle(os.path.join(dst, "combined_df_one_set.pkl"))
        tiff = os.path.join(dst, "tif", os.path.basename(str(sdf.after_align_full_save_path.dropna().iloc[0])))
        full = pd.read_pickle(session_pkl)
        g = sdf.group.iloc[0]
        gdf = full[(full.group == g) & (full.filepath_without_number == sdf.filepath_without_number.iloc[0])]
        r = ac.set_alignment_r(sdf, gdf, tiff, ch=2)
        assert len(r) and r.r_high.between(-1, 1).all(), r
        assert r.r_low.notna().all() and r.r_low.between(-1, 1).all() and (r.lowmag != "").all(), r
        assert os.path.exists(os.path.splitext(tiff)[0] + ac.R_CSV_SUFFIX)
        # independent recomputation of one highmag value from the raw FLIM files (numpy only,
        # no filter): Pre 1 and the last post frame, each the max projection over its z window
        # moved by its integer shift (frame_info), r over the pixels both cover
        fi = pd.read_csv(os.path.splitext(tiff)[0] + "_frame_info.csv")
        by_name = {os.path.basename(str(p)).lower(): str(p) for p in sdf.file_path}

        def moved_mip(fr):
            v = ffi.load_flim_intensity(by_name[fr.filename.lower()], use_cache=True)[0][:, 0, 1].astype(float)
            img = v[int(fr.z_from):int(fr.z_to)].max(0)
            dy, dx = int(fr.shift_y), int(fr.shift_x)
            out = np.full_like(img, np.nan)
            h, w = img.shape
            out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
                img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            return out

        p1 = fi[fi.phase == "pre"].sort_values("frame").iloc[0]
        fr = fi[fi.phase == "post"].iloc[-1]
        a, b = moved_mip(p1), moved_mip(fr)
        ok = np.isfinite(a) & np.isfinite(b)
        r_np = round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3)
        got = float(r[r.filename == fr.filename.lower()].r_high.iloc[0])
        assert got == r_np, (got, r_np)
        assert float(r[r.filename == p1.filename.lower()].r_high.iloc[0]) == 1.0
        assert (r.reference == "pre1").all()
        # drawn on the panel
        q = pd.read_csv(os.path.join(dst, "quant.csv"))
        review = P.load_set_review(tiff, q, corr=r)
        fig = plt.figure(figsize=(12, 5))
        axes = P.draw_set_review(fig, review)
        texts = [t.get_text() for ax, f in axes for t in ax.texts]
        n_pp = sum(f.phase in ("pre", "post") for _, f in axes)
        assert sum(t.startswith("H ") and "  L " in t and "\n" not in t for t in texts) == n_pp, texts
        plt.close(fig)
        # whole session (first 2 sets, no cache beside the TIFFs, CSV into the temp folder)
        out = ac.session_alignment_r(session_pkl, ch=2, out_csv=os.path.join(td, "session_r.csv"),
                                     write_cache=False, max_sets=2)
        sr = pd.read_csv(out)
        assert sr.groupby(["group", "set_label"]).ngroups == 2 and sr.r_high.notna().all() \
            and sr.r_low.notna().all(), sr
        print(f"session check passed: {len(sr)} frames in 2 sets")
        print(r.to_string(index=False))
        print(f"real check passed: {len(r)} frames, r_high recomputed with numpy = {r_np:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs=2, default=None)
    a = ap.parse_args()
    test_pearson_overlap()
    print("PASS test_pearson_overlap")
    if a.real:
        real_check(*a.real)
