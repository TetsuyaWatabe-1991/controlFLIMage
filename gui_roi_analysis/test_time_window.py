"""Headless tests for the time window of the respan analysis (apply_time_window).

Unit: frames outside [lo, hi] minutes leave their set (nth_set_label -1, phase "None"),
the uncaging and in-window frames stay; a set without post frames left is removed; a
second call with another window (or None) starts from the original sets; "changed" is
only True when the sets differ.

--real PKL [GROUP]: on a real combined_df (read only), list the removed frames; then copy
the FLIM files of one group to a temporary folder and rebuild its full-size TIFF stacks
with the window (fast_mode, as in the analysis); its existing ROI masks are
re-ordered (remap_roi_mask_stacks) and must keep the same mask per FLIM file. Each set's TIFF must have exactly the
pre/unc/post frames left in the set, and frame_info.csv must not list removed files.
Nothing is written into the session folder.

Run:
    python test_time_window.py
    python test_time_window.py --real G:\\ImagingData\\Tetsuya\\20260929\\auto1\\combined_df_respan.pkl
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.dirname(HERE))
from gui_roi_respan_seg_masks import (  # noqa: E402
    _frame_keys,
    apply_time_window,
    remap_roi_mask_stacks,
    snapshot_frame_info,
)


def _toy() -> pd.DataFrame:
    rows = []
    # group a, set 0: pre at -45/-20/-5, unc at 0, post at 10/49/51/300
    for t, ph in [(-45, "pre"), (-20, "pre"), (-5, "pre"), (0, "unc"), (10, "post"), (49, "post"),
                  (51, "post"), (300, "post")]:
        rows.append(dict(group="a", nth_set_label=0.0, phase=ph, relative_time_min=float(t)))
    # group b, set 0: every post frame is late -> whole set removed
    for t, ph in [(-10, "pre"), (0, "unc"), (60, "post"), (70, "post")]:
        rows.append(dict(group="b", nth_set_label=0.0, phase=ph, relative_time_min=float(t)))
    rows.append(dict(group="a", nth_set_label=-1.0, phase="None", relative_time_min=np.nan))
    return pd.DataFrame(rows)


def test_window_removes_frames():
    df, changed = apply_time_window(_toy(), (-40, 50))
    assert changed
    a = df[(df.group == "a") & (df.nth_set_label == 0)]
    assert sorted(a.relative_time_min) == [-20, -5, 0, 10, 49], a
    assert (df[(df.group == "a") & df.relative_time_min.isin([-45, 51, 300])].phase == "None").all()
    assert not (df[df.group == "b"].nth_set_label >= 0).any()
    assert int(df.excluded_time_window.sum()) == 3 + 4


def test_window_is_reversible_and_idempotent():
    df, _ = apply_time_window(_toy(), (-40, 50))
    again, changed = apply_time_window(df, (-40, 50))
    assert not changed
    wide, changed = apply_time_window(df, None)
    assert changed
    orig = _toy()
    assert np.array_equal(wide.nth_set_label.values, orig.nth_set_label.values)
    assert (wide.phase.values == orig.phase.values).all()
    narrow, _ = apply_time_window(df, (-10, 20))
    assert sorted(narrow[(narrow.group == "a") & (narrow.nth_set_label == 0)].relative_time_min) == [-5, 0, 10]


def real_check(pkl: str, group: str | None) -> None:
    import tifffile

    import flim_fast_io as ffi
    from gui_roi_fast_simple import rebuild_tiff_full_size_for_roi

    full = pd.read_pickle(pkl)
    df, changed = apply_time_window(full, (-40, 50))
    removed = df[df.excluded_time_window]
    print(f"{len(removed)} frames removed, changed={changed}")
    print(removed[["group", "nth_set_label_before_time_window", "phase_before_time_window",
                   "relative_time_min"]].to_string())
    group = group or str(removed.group.iloc[0])
    gdf = df[df.group == group].copy()
    ffi._save_cached_intensity = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as td:
        for p in gdf.file_path.unique():
            shutil.copy2(p, td)
        gdf["file_path"] = [os.path.join(td, os.path.basename(p)) for p in gdf.file_path]
        # the stacks go to dirname(filepath_without_number)/tif: point it at the temp folder too
        gdf["filepath_without_number"] = [os.path.join(td, os.path.basename(str(p).replace("\\", "/")))
                                          for p in gdf.filepath_without_number]
        assert all(str(p).startswith(td) for p in gdf.filepath_without_number)
        # existing stacks, frame_info and ROI masks of the group, copied as they are
        tif_td = os.path.join(td, "tif")
        os.makedirs(tif_td)
        src_tiffs = gdf.after_align_full_save_path.dropna().astype(str).unique()
        for t in src_tiffs:
            base = os.path.splitext(t)[0]
            for f in glob.glob(glob.escape(base) + "*"):
                shutil.copy2(f, tif_td)
        for c in ("after_align_full_save_path", "before_align_full_save_path", "after_align_save_path"):
            if c in gdf.columns:
                gdf[c] = [os.path.join(tif_td, os.path.basename(str(p).replace("\\", "/")))
                          if isinstance(p, str) else p for p in gdf[c]]
        old_fi = snapshot_frame_info(gdf)
        old_masks = {t: tifffile.imread(os.path.splitext(t)[0] + "_Spine_roi_mask.tif") for t in old_fi}
        out = rebuild_tiff_full_size_for_roi(gdf.reset_index(drop=True), 2, 1, fast_mode=True,
                                             intensity_cache={})
        n_rewritten = remap_roi_mask_stacks(old_fi)
        for t, old in old_fi.items():
            new = pd.read_csv(os.path.splitext(t)[0] + "_frame_info.csv")
            m_new = tifffile.imread(os.path.splitext(t)[0] + "_Spine_roi_mask.tif")
            assert m_new.shape[0] == len(new), (t, m_new.shape, len(new))
            old_idx = {k: i for i, k in enumerate(_frame_keys(old))}
            for j, k in enumerate(_frame_keys(new)):
                assert np.array_equal(m_new[j], old_masks[t][old_idx[k]]), (t, k)
            print(f"  {os.path.basename(t)}: Spine mask {len(old)} -> {len(new)} frames, same mask per file")
        print(f"  {n_rewritten} mask files re-ordered")
        for s, sdf in out[out.nth_set_label >= 0].groupby("nth_set_label"):
            tiff = str(sdf.after_align_full_save_path.dropna().iloc[0])
            assert tiff.startswith(td), tiff
            n_tif = tifffile.imread(tiff).shape[0]
            n_pre, n_post = int((sdf.phase == "pre").sum()), int((sdf.phase == "post").sum())
            fi = pd.read_csv(os.path.splitext(tiff)[0] + "_frame_info.csv")
            ph = fi.phase.str.lower()
            n_unc = int((~ph.isin(["pre", "post"])).sum())  # uncaging FLIM frames
            assert (ph == "pre").sum() == n_pre and (ph == "post").sum() == n_post, (s, ph.value_counts())
            assert n_tif == len(fi) == n_pre + n_unc + n_post, (s, n_tif, len(fi), n_pre, n_unc, n_post)
            gone = {os.path.basename(p).lower() for p in removed[removed.group == group].file_path}
            assert not (set(fi.filename.str.lower()) & gone), fi.filename.tolist()
            t = sdf[sdf.phase.isin(["pre", "post"])].relative_time_min
            assert t.between(-40, 50).all(), t.tolist()
            print(f"  {group} set {s:g}: TIFF {n_tif} frames = pre {n_pre} + unc {n_unc} + post {n_post}, "
                  f"time {t.min():.1f} to {t.max():.1f} min")
    print("real check passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None)
    ap.add_argument("--group", default=None)
    a = ap.parse_args()
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    if a.real:
        real_check(a.real, a.group)
