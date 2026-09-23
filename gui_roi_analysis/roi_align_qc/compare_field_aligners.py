"""Score field-level aligners against the spine position the human kept.

The target drift of each frame is where the spine sat in the raw projection,
reconstructed from the manual ROI on the aligned TIFF and the shift that was
actually applied. A useful aligner matches that drift using the dendrite and
the rest of the field.

Usage:
    python compare_field_aligners.py --df-path G:/.../combined_df_respan.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from diagnose_alignment import _before_path, _load_stack  # noqa: E402
from field_align import FIELD_ALIGNERS  # noqa: E402
from tracking import (  # noqa: E402
    best_sign_error,
    centroid_trajectory,
    mean_trajectory_error,
    pairwise_content_shifts,
)


def _mean_radius(shift_yx: np.ndarray) -> float:
    dist = np.hypot(shift_yx[:, 0], shift_yx[:, 1])
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _one_set(
    set_df: pd.DataFrame,
    group: str,
    set_label: int,
    session: str,
    aligners: dict | None = None,
) -> dict | None:
    after_path = set_df["after_align_full_save_path"].iloc[0] if "after_align_full_save_path" in set_df else None
    if not isinstance(after_path, str) or not os.path.exists(after_path):
        after_path = set_df["after_align_save_path"].iloc[0]
    if not isinstance(after_path, str) or not os.path.exists(after_path):
        return None
    before_path = _before_path(after_path, set_df)
    if before_path is None:
        return None
    before = _load_stack(before_path)
    after = _load_stack(after_path)
    if before is None or after is None:
        return None

    tiff_dir = os.path.dirname(after_path)
    base = os.path.splitext(os.path.basename(after_path))[0]
    manual_path = os.path.join(tiff_dir, f"{base}_Spine_roi_mask.tif")
    info_path = os.path.join(tiff_dir, f"{base}_frame_info.csv")
    if not os.path.exists(manual_path) or not os.path.exists(info_path):
        return None
    manual = tifffile.imread(manual_path)
    if manual.ndim == 2:
        manual = np.repeat(manual[None, ...], len(after), axis=0)
    info = pd.read_csv(info_path)
    n_frames = min(len(before), len(after), len(manual), len(info))
    before = before[:n_frames]
    after = after[:n_frames]
    manual = np.asarray(manual[:n_frames] > 0)
    info = info.iloc[:n_frames].reset_index(drop=True)
    if "shift_y" not in info.columns:
        return None

    manual_traj = centroid_trajectory(manual)
    if not np.isfinite(manual_traj).all():
        return None
    stored = np.column_stack(
        [
            pd.to_numeric(info["shift_y"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(info["shift_x"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    stored = stored - stored[0]
    tiff_shift = pairwise_content_shifts(before, after)
    _, stored_sign = best_sign_error(stored, tiff_shift)
    applied = stored_sign * stored
    # Spine motion in the raw stack. applied is the content shift already baked
    # into the aligned TIFF, so subtracting it puts the manual ROI back.
    target = manual_traj - applied

    row = {
        "session": session,
        "group": group,
        "set_label": int(set_label),
        "manual_span_px": float(np.nanmax(np.hypot(manual_traj[:, 0], manual_traj[:, 1]))),
        "current_residual_px": _mean_radius(manual_traj),
        "target_span_px": float(np.nanmax(np.hypot(target[:, 0], target[:, 1]))),
    }
    if aligners is None:
        aligners = FIELD_ALIGNERS
    for name, aligner in aligners.items():
        estimate = aligner(before)
        row[f"{name}_residual_px"] = mean_trajectory_error(estimate, target)
    return row


def _summarize(table: pd.DataFrame, methods: list[str] | None = None) -> str:
    lines = [f"sets {len(table)}"]
    if methods is None:
        methods = list(FIELD_ALIGNERS)
    large = table[table["manual_span_px"] >= 3.0]
    for label, frame in (("all", table), ("large_manual", large)):
        if len(frame) == 0:
            continue
        lines.append(f"[{label}] n={len(frame)} current residual median={frame['current_residual_px'].median():.2f} px")
        for name in methods:
            col = f"{name}_residual_px"
            better = frame[col] + 0.25 < frame["current_residual_px"]
            lines.append(
                f"  {name}: median residual {frame[col].median():.2f} px, "
                f"frac better than current {float(better.mean()):.2f}"
            )
    return "\n".join(lines)


def run_session(
    df_path: str,
    out_dir: str | None,
    aligners: dict | None = None,
    file_stem: str = "field_aligner",
) -> pd.DataFrame:
    combined = pd.read_pickle(df_path)
    session = Path(df_path).parent.name + "_" + Path(df_path).parents[1].name
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(df_path), "roi_align_qc")
    os.makedirs(out_dir, exist_ok=True)
    rows: list[dict] = []
    for _filepath, filegroup in combined.groupby("filepath_without_number", sort=False):
        for group, group_df in filegroup.groupby("group", sort=False):
            for set_label, set_df in group_df.groupby("nth_set_label", sort=False):
                if int(set_label) < 0:
                    continue
                row = _one_set(set_df, str(group), int(set_label), session, aligners)
                if row is not None:
                    rows.append(row)
                print(f"  {group} set {int(set_label)}: {'ok' if row else 'skip'}", flush=True)
    table = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"{file_stem}_comparison.csv")
    table.to_csv(csv_path, index=False)
    method_names = list(FIELD_ALIGNERS if aligners is None else aligners)
    summary = _summarize(table, method_names)
    with open(os.path.join(out_dir, f"{file_stem}_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(summary + "\n")
    print(summary)
    print(f"wrote {csv_path}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare field-level aligners to human spine positions")
    parser.add_argument("--df-path", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_session(args.df_path, args.out_dir)


if __name__ == "__main__":
    main()
