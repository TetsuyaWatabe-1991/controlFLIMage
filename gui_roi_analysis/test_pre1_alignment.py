"""Headless tests for pre1_alignment.py and the ROI-mask move of remap_roi_mask_stacks.

Unit:
  1. realign_sets_to_pre1 on synthetic volumes: a frame that is Pre 1 moved by (dz, dy, dx)
     gets shift = Pre 1 shift - (dz, dy, dx) (sign of the global alignment); the global
     values are kept in shift_*_002 and reference "002" restores them; a second call does
     not change anything.
  2. remap_roi_mask_stacks moves a TIFF-coordinate mask by the change of the applied shift
     and leaves a *_raw mask in place.

--real SESSION GROUP SET [CH]: the FLIM files of one group and its existing TIFF folder
files are copied to a temporary folder; the group's sets are registered to Pre 1, the
stacks rebuilt (fast_mode, as in the analysis) and the ROI masks carried over. For the
given set: correlation of every frame with Pre 1 before (global 002 registration) and
after; every Spine mask frame equals the old one moved by the shift change. Nothing is
written into the session folder.

Run:
    python test_pre1_alignment.py
    python test_pre1_alignment.py --real G:\\ImagingData\\Tetsuya\\20261002\\auto2 CaMKII_3_pos1__highmag_4_ 2 1
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.dirname(HERE))
import pre1_alignment as pa  # noqa: E402
from gui_roi_respan_seg_masks import _translate, remap_roi_mask_stacks  # noqa: E402


def _blobs(shape=(9, 96, 96), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape)
    v = np.zeros(shape)
    for _ in range(25):
        cz, cy, cx = rng.uniform(2, shape[0] - 2), rng.uniform(10, shape[1] - 10), rng.uniform(10, shape[2] - 10)
        v += np.exp(-(((z - cz) / 1.5) ** 2 + ((y - cy) / 3) ** 2 + ((x - cx) / 3) ** 2))
    return v * 100 + rng.poisson(2, shape)


def test_realign_sign_and_restore():
    ref = _blobs()
    d = (1, -4, 3)
    mov = np.roll(ref, d, axis=(0, 1, 2))  # mov = Pre 1 moved by +d
    vols = {"p1.flim": ref, "p2.flim": mov, "post.flim": ref.copy()}
    df = pd.DataFrame({
        "group": "g", "nth_set_label": 0.0, "nth_omit_induction": [0, 1, 2, 3],
        "phase": ["pre", "pre", "unc", "post"], "file_path": ["p1.flim", "p2.flim", "unc.flim", "post.flim"],
        "shift_z": [0.5, 9.0, 9.0, 9.0], "shift_y": [10.0, 50.0, 50.0, 50.0], "shift_x": [-2.0, 7.0, 7.0, 7.0],
    })
    orig = pa._volume
    pa._volume = lambda p, ch: vols[p]
    try:
        out, changed = pa.realign_sets_to_pre1(df, 1, log=lambda *a: None)
        assert changed
        p2 = out[out.file_path == "p2.flim"].iloc[0]
        assert np.allclose([p2.shift_z, p2.shift_y, p2.shift_x], [0.5 - 1, 10 + 4, -2 - 3], atol=0.3), p2
        post = out[out.file_path == "post.flim"].iloc[0]
        assert np.allclose([post.shift_z, post.shift_y, post.shift_x], [0.5, 10, -2], atol=0.3), post
        unc = out[out.file_path == "unc.flim"].iloc[0]
        assert (unc.shift_z, unc.shift_y, unc.shift_x) == (9.0, 50.0, 7.0)  # uncaging row untouched
        assert (out.align_reference == "pre1").all() and (out.shift_y_002 == df.shift_y).all()
        again, changed = pa.realign_sets_to_pre1(out, 1, log=lambda *a: None)
        assert not changed
        back, changed = pa.realign_sets_to_pre1(out, 1, reference="002", log=lambda *a: None)
        assert changed and np.array_equal(back.shift_y.values, df.shift_y.values)
    finally:
        pa._volume = orig


def test_remap_moves_tiff_masks_only():
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "g_0.0_after_align_full")
        mask = np.zeros((2, 20, 20), np.uint8)
        mask[1, 8:11, 8:11] = 1
        tifffile.imwrite(base + "_Spine_roi_mask.tif", mask)
        tifffile.imwrite(base + "_Spine_roi_mask_raw.tif", mask)
        old = pd.DataFrame({"frame": [0, 1], "filename": ["a.flim", "b.flim"], "shift_y": [0, 5], "shift_x": [0, -1]})
        new = old.assign(shift_y=[0, 2], shift_x=[0, 3])
        new.to_csv(base + "_frame_info.csv", index=False)
        assert remap_roi_mask_stacks({base + ".tif": old}) == 2
        m = tifffile.imread(base + "_Spine_roi_mask.tif")
        assert np.array_equal(m[1], _translate(mask[1], -3, 4)) and m[1, 5:8, 12:15].all()
        assert np.array_equal(tifffile.imread(base + "_Spine_roi_mask_raw.tif"), mask)


def real_check(session: str, group: str, set_label: float, ch: int) -> None:
    import flim_fast_io as ffi

    import alignment_correlation as ac
    from gui_roi_fast_simple import rebuild_tiff_full_size_for_roi
    from gui_roi_respan_seg_masks import snapshot_frame_info

    ffi._save_cached_intensity = lambda *a, **k: None
    full = pd.read_pickle(os.path.join(session, "combined_df_respan.pkl"))
    gdf = full[full.group == group].copy()
    with tempfile.TemporaryDirectory() as td:
        for p in gdf.file_path.unique():
            shutil.copy2(p, td)
        gdf["file_path"] = [os.path.join(td, os.path.basename(p)) for p in gdf.file_path]
        gdf["filepath_without_number"] = [os.path.join(td, os.path.basename(str(p).replace("\\", "/")))
                                          for p in gdf.filepath_without_number]
        tif_td = os.path.join(td, "tif")
        os.makedirs(tif_td)
        for t in gdf.after_align_full_save_path.dropna().astype(str).unique():
            for f in glob.glob(glob.escape(os.path.splitext(t)[0]) + "*"):
                shutil.copy2(f, tif_td)
        for c in ("after_align_full_save_path", "before_align_full_save_path", "after_align_save_path"):
            gdf[c] = [os.path.join(tif_td, os.path.basename(str(p).replace("\\", "/"))) if isinstance(p, str) else p
                      for p in gdf[c]]
        tiff = str(gdf[gdf.nth_set_label == set_label].after_align_full_save_path.dropna().iloc[0])
        assert tiff.startswith(td)
        fi_old = pd.read_csv(os.path.splitext(tiff)[0] + "_frame_info.csv")
        r_old = ac.highmag_r(tiff, fi_old)
        old_fi = snapshot_frame_info(gdf)
        old_mask = tifffile.imread(os.path.splitext(tiff)[0] + "_Spine_roi_mask.tif")

        gdf, changed = pa.realign_sets_to_pre1(gdf.reset_index(drop=True), ch)
        out = rebuild_tiff_full_size_for_roi(gdf, ch, 1, fast_mode=True, intensity_cache={})
        assert all(str(p).startswith(td) for p in out.after_align_full_save_path.dropna())
        remap_roi_mask_stacks(old_fi)
        fi_new = pd.read_csv(os.path.splitext(tiff)[0] + "_frame_info.csv")
        r_new = ac.highmag_r(tiff, fi_new)
        new_mask = tifffile.imread(os.path.splitext(tiff)[0] + "_Spine_roi_mask.tif")
        assert new_mask.shape == old_mask.shape
        for j in range(len(fi_new)):
            dy = int(fi_new.shift_y[j]) - int(fi_old.shift_y[j])
            dx = int(fi_new.shift_x[j]) - int(fi_old.shift_x[j])
            assert np.array_equal(new_mask[j], _translate(old_mask[j], dy, dx)), j
        print(f"{group} set {set_label:g} (ch{ch}), shifts changed: {changed}")
        print("  frame  applied shift 002 -> pre1    r vs Pre 1: 002 -> pre1")
        for (_, a), (_, b) in zip(fi_old.iterrows(), fi_new.iterrows()):
            if str(a.phase) not in ("pre", "post"):
                continue
            n = str(a.filename).lower()
            print(f"  {n[-8:-5]} {a.phase:4s}  ({int(a.shift_y):+4d},{int(a.shift_x):+4d}) -> "
                  f"({int(b.shift_y):+4d},{int(b.shift_x):+4d})    {r_old[n]:+.3f} -> {r_new[n]:+.3f}")
        print("real check passed (masks moved with the frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="+", default=None, help="SESSION GROUP SET [CH]")
    a = ap.parse_args()
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    if a.real:
        real_check(a.real[0], a.real[1], float(a.real[2]), int(a.real[3]) if len(a.real) > 3 else 1)
