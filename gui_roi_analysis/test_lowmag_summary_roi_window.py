"""Headless check: the summary figure shows the quantified image under the quantified ROI.

For every set of a combined_df pkl, the Pre / Post panels of
lowmag_img_save_with_GCdata_tiff_roi_surface_depth.py are reproduced with the module's
own functions (_flim_zproj_from_row + _load_roi_mask_from_tiff). The Spine mean inside
the drawn ROI on the drawn image must equal Spine_Ch2_intensity of the all-frames CSV
for that file. Nothing is written.

Run:
    python test_lowmag_summary_roi_window.py --real <combined_df_respan.pkl> [--max-sets N]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.dirname(HERE))
import lowmag_img_save_with_GCdata_tiff_roi_surface_depth as lm  # noqa: E402


def real_check(pkl: str, max_sets: int | None) -> None:
    df = pd.read_pickle(pkl)
    q = pd.read_csv(pkl.replace(".pkl", "_intensity_lifetime_all_frames.csv"))
    q["base"] = q.file_path.map(lambda p: os.path.basename(str(p).replace("\\", "/")).lower())
    n_ok = n_bad = n_sets = 0
    bad = []
    for (grp, sl), sdf in df[df.nth_set_label != -1].groupby(["group", "nth_set_label"], sort=False):
        if max_sets and n_sets >= max_sets:
            break
        n_sets += 1
        qs = q[(q.group.astype(str) == str(grp)) & (q.set_label.astype(float) == float(sl))]
        cache = {}
        panels = [sdf[sdf.phase == "pre"].sort_values("nth_omit_induction").iloc[-1],
                  sdf[sdf.phase == "post"].sort_values("nth_omit_induction").iloc[0]]
        for row in panels:
            img = lm._flim_zproj_from_row(row, ch_1or2=2)
            mask = lm._load_roi_mask_from_tiff(row, "Spine", cache, set_df=sdf)
            ref = qs[(qs.base == os.path.basename(str(row.file_path).replace("\\", "/")).lower()) & (qs.phase == row.phase)]
            if mask is None or not mask.any() or not len(ref):
                continue
            val, exp = float(img[mask].mean()), float(ref.Spine_Ch2_intensity.iloc[0])
            if np.isclose(val, exp, rtol=1e-6):
                n_ok += 1
            else:
                n_bad += 1
                bad.append((f"{grp}{sl:g}", row.phase, os.path.basename(str(row.file_path)), round(val, 3), round(exp, 3)))
    print(f"summary-figure panels: {n_ok} match the quantification, {n_bad} differ ({n_sets} sets)")
    for b in bad[:15]:
        print("  differs:", *b)
    assert n_bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--max-sets", type=int, default=None)
    a = ap.parse_args()
    real_check(a.real, a.max_sets)
