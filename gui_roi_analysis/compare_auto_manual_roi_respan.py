# -*- coding: utf-8 -*-
"""
Compare auto (seg_masks) vs manual (saved ROI GUI) masks and quantification.

Usage:
    python compare_auto_manual_roi_respan.py --df-path "G:/.../combined_df_respan.pkl"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ongoing" / "ASIcontroller"))

from gui_roi_fast_simple import (  # noqa: E402
    ROI_MASK_RAW_SUFFIX,
    ROI_TYPES,
    quantify_intensity_from_flim,
    save_drift_corrected_roi_masks,
)
from gui_roi_respan_seg_masks import (  # noqa: E402
    _load_mask_2d,
    create_roi_masks_from_seg_masks,
    highmag_savefolder_from_filepath_without_number,
    match_uncaging_record_for_set,
    seg_mask_paths,
)
from respan_uncaging_log import parse_uncaging_records  # noqa: E402


def _mask_iou(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    a_bin = a > 0
    b_bin = b > 0
    inter = int(np.logical_and(a_bin, b_bin).sum())
    union = int(np.logical_or(a_bin, b_bin).sum())
    iou = inter / union if union else 1.0
    return float(iou), int(a_bin.sum()), int(b_bin.sum())


def compare_masks(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Compare Type-A ROI masks (middle frame) to seg_masks per set."""
    rows: list[dict] = []
    for filepath_wo in combined_df["filepath_without_number"].unique():
        filegroup = combined_df[combined_df["filepath_without_number"] == filepath_wo]
        highmag_folder = highmag_savefolder_from_filepath_without_number(filepath_wo)
        records = parse_uncaging_records(highmag_folder)

        for group in filegroup["group"].unique():
            group_df = filegroup[filegroup["group"] == group]
            for set_label in group_df["nth_set_label"].unique():
                if set_label == -1:
                    continue
                set_df = group_df[group_df["nth_set_label"] == set_label]
                tiff_path = set_df["after_align_full_save_path"].iloc[0]
                if pd.isna(tiff_path) or not os.path.exists(str(tiff_path)):
                    tiff_path = set_df["after_align_save_path"].iloc[0]
                if pd.isna(tiff_path) or not os.path.exists(str(tiff_path)):
                    continue

                record = match_uncaging_record_for_set(set_df, records)
                if record is None:
                    continue
                mask_paths = seg_mask_paths(highmag_folder, record.spine_stem)

                tiff_dir = os.path.dirname(str(tiff_path))
                base_name = os.path.splitext(os.path.basename(str(tiff_path)))[0]

                for roi_type in ROI_TYPES:
                    cur_path = os.path.join(tiff_dir, f"{base_name}_{roi_type}_roi_mask.tif")
                    if not os.path.exists(cur_path) or roi_type not in mask_paths:
                        continue
                    cur_stack = tifffile.imread(cur_path)
                    mid = cur_stack.shape[0] // 2 if cur_stack.ndim == 3 else 0
                    cur2d = cur_stack[mid] if cur_stack.ndim == 3 else cur_stack
                    auto2d = _load_mask_2d(mask_paths[roi_type])
                    iou, cur_px, auto_px = _mask_iou(cur2d, auto2d)
                    rows.append(
                        {
                            "group": group,
                            "set_label": int(set_label),
                            "spine_stem": record.spine_stem,
                            "roi_type": roi_type,
                            "manual_px": cur_px,
                            "auto_px": auto_px,
                            "iou": iou,
                            "changed": (iou < 0.99) or (cur_px != auto_px),
                        }
                    )
    return pd.DataFrame(rows)


def _backup_manual_masks(combined_df: pd.DataFrame, backup_dir: str) -> list[str]:
    os.makedirs(backup_dir, exist_ok=True)
    backed: list[str] = []
    for filepath_wo in combined_df["filepath_without_number"].unique():
        filegroup = combined_df[combined_df["filepath_without_number"] == filepath_wo]
        for group in filegroup["group"].unique():
            group_df = filegroup[filegroup["group"] == group]
            for set_label in group_df["nth_set_label"].unique():
                if set_label == -1:
                    continue
                set_df = group_df[group_df["nth_set_label"] == set_label]
                tiff_path = set_df["after_align_save_path"].iloc[0]
                if pd.isna(tiff_path):
                    continue
                tiff_dir = os.path.dirname(str(tiff_path))
                base_name = os.path.splitext(os.path.basename(str(tiff_path)))[0]
                for roi_type in ROI_TYPES:
                    for suffix in ("_roi_mask.tif", f"{ROI_MASK_RAW_SUFFIX}.tif"):
                        src = os.path.join(tiff_dir, f"{base_name}_{roi_type}{suffix}")
                        if not os.path.exists(src):
                            continue
                        rel = os.path.relpath(src, os.path.dirname(backup_dir))
                        dst = os.path.join(backup_dir, rel.replace(os.sep, "__"))
                        shutil.copy2(src, dst)
                        backed.append(dst)
    return backed


def _restore_manual_masks(backup_dir: str, combined_df: pd.DataFrame) -> None:
    if not os.path.isdir(backup_dir):
        return
    for filepath_wo in combined_df["filepath_without_number"].unique():
        filegroup = combined_df[combined_df["filepath_without_number"] == filepath_wo]
        for group in filegroup["group"].unique():
            group_df = filegroup[filegroup["group"] == group]
            for set_label in group_df["nth_set_label"].unique():
                if set_label == -1:
                    continue
                set_df = group_df[group_df["nth_set_label"] == set_label]
                tiff_path = set_df["after_align_save_path"].iloc[0]
                if pd.isna(tiff_path):
                    continue
                tiff_dir = os.path.dirname(str(tiff_path))
                base_name = os.path.splitext(os.path.basename(str(tiff_path)))[0]
                for roi_type in ROI_TYPES:
                    for suffix in ("_roi_mask.tif", f"{ROI_MASK_RAW_SUFFIX}.tif"):
                        dst = os.path.join(tiff_dir, f"{base_name}_{roi_type}{suffix}")
                        rel = os.path.relpath(dst, os.path.dirname(backup_dir))
                        src = os.path.join(backup_dir, rel.replace(os.sep, "__"))
                        if os.path.exists(src):
                            shutil.copy2(src, dst)


def quantify_with_auto_masks(
    combined_df: pd.DataFrame,
    *,
    ch_1or2: int,
    z_plus_minus: int,
    out_csv: str,
    skip_lifetime: bool,
) -> pd.DataFrame:
    backup_dir = os.path.join(os.path.dirname(out_csv), "_roi_manual_backup_for_compare")
    print(f"Backing up manual ROI masks to {backup_dir}")
    _backup_manual_masks(combined_df, backup_dir)

    try:
        print("Writing auto ROI masks from seg_masks...")
        create_roi_masks_from_seg_masks(
            combined_df,
            skip_if_roi_mask_exists=False,
        )
        print("Building drift-corrected auto masks (Type B)...")
        save_drift_corrected_roi_masks(combined_df)
        quantify_intensity_from_flim(
            combined_df,
            ch_1or2,
            z_plus_minus,
            out_csv,
            skip_lifetime_analysis=skip_lifetime,
        )
        return pd.read_csv(out_csv)
    finally:
        print("Restoring manual ROI masks...")
        _restore_manual_masks(backup_dir, combined_df)


def _summary_metrics(df: pd.DataFrame, ch: int) -> pd.DataFrame:
    """Per-set pre mean and post mean Spine intensity (Ch2 default for tdTom)."""
    ch_name = f"Ch{ch}"
    int_col = f"Spine_{ch_name}_intensity"
    if int_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for (group, set_label), g in df.groupby(["group", "set_label"], sort=True):
        pre = g[g["phase"] == "pre"][int_col]
        post = g[g["phase"] == "post"][int_col]
        if pre.empty or post.empty:
            continue
        pre_mean = float(pre.mean())
        post_mean = float(post.mean())
        delta_ff0 = post_mean / pre_mean - 1.0 if pre_mean else np.nan
        rows.append(
            {
                "group": group,
                "set_label": set_label,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "delta_FF0": delta_ff0,
            }
        )
    return pd.DataFrame(rows)


def compare_quant_csv(manual_csv: str, auto_csv: str, ch_1or2: int) -> pd.DataFrame:
    manual = pd.read_csv(manual_csv)
    auto = pd.read_csv(auto_csv)
    ch_name = f"Ch{ch_1or2}"
    key_cols = ["group", "set_label", "phase", "slice"]
    value_cols = [
        f"Spine_{ch_name}_intensity",
        f"DendriticShaft_{ch_name}_intensity",
        f"Spine_{ch_name}_lifetime",
    ]
    value_cols = [c for c in value_cols if c in manual.columns and c in auto.columns]

    merged = manual[key_cols + value_cols].merge(
        auto[key_cols + value_cols],
        on=key_cols,
        suffixes=("_manual", "_auto"),
        how="inner",
    )
    for col in value_cols:
        mcol, acol = f"{col}_manual", f"{col}_auto"
        merged[f"{col}_abs_diff"] = merged[mcol] - merged[acol]
        with np.errstate(divide="ignore", invalid="ignore"):
            merged[f"{col}_pct_diff"] = 100.0 * (merged[mcol] / merged[acol] - 1.0)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare auto vs manual respan ROIs")
    parser.add_argument(
        "--df-path",
        default=r"G:\ImagingData\Tetsuya\20260701\auto1\combined_df_respan.pkl",
    )
    parser.add_argument("--ch", type=int, default=2)
    parser.add_argument("--z-plus-minus", type=int, default=2)
    parser.add_argument(
        "--skip-lifetime",
        action="store_true",
        help="Skip lifetime fitting during auto re-quantification (faster)",
    )
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="Only compare mask geometry, skip re-quantification",
    )
    args = parser.parse_args()

    df_path = args.df_path
    out_dir = os.path.dirname(df_path)
    combined_df = pd.read_pickle(df_path)

    mask_cmp = compare_masks(combined_df)
    mask_out = os.path.join(out_dir, "roi_auto_vs_manual_mask_compare.csv")
    mask_cmp.to_csv(mask_out, index=False)
    print(f"Wrote mask comparison: {mask_out}")
    if len(mask_cmp):
        n_sets = mask_cmp[["group", "set_label"]].drop_duplicates().shape[0]
        changed_sets = (
            mask_cmp[mask_cmp["changed"]][["group", "set_label"]].drop_duplicates().shape[0]
        )
        print(f"Sets compared: {n_sets}")
        print(f"Sets with any ROI change vs seg_masks: {changed_sets}")
        print(mask_cmp.groupby("roi_type")["iou"].describe().round(3))
        spine = mask_cmp[mask_cmp["roi_type"] == "Spine"]
        if len(spine):
            print("Spine IoU (lowest 10):")
            print(
                spine.nsmallest(10, "iou")[
                    ["group", "set_label", "spine_stem", "manual_px", "auto_px", "iou"]
                ].to_string(index=False)
            )

    manual_csv = df_path.replace(".pkl", "_intensity_lifetime_all_frames.csv")
    if args.mask_only:
        return

    auto_csv = df_path.replace(".pkl", "_intensity_lifetime_all_frames_AUTO_roi.csv")
    if not os.path.exists(manual_csv):
        print(f"Manual quant CSV not found: {manual_csv}")
        return

    print("Re-quantifying with auto (seg_mask) ROIs...")
    quantify_with_auto_masks(
        combined_df,
        ch_1or2=args.ch,
        z_plus_minus=args.z_plus_minus,
        out_csv=auto_csv,
        skip_lifetime=args.skip_lifetime,
    )

    frame_cmp = compare_quant_csv(manual_csv, auto_csv, args.ch)
    frame_out = os.path.join(out_dir, "roi_auto_vs_manual_frame_compare.csv")
    frame_cmp.to_csv(frame_out, index=False)
    print(f"Wrote frame-level comparison: {frame_out}")

    int_col = f"Spine_Ch{args.ch}_intensity_abs_diff"
    if int_col in frame_cmp.columns:
        abs_diff = frame_cmp[int_col].abs()
        print(
            f"Spine Ch{args.ch} intensity |abs diff|: "
            f"median={abs_diff.median():.4f}, mean={abs_diff.mean():.4f}, "
            f"max={abs_diff.max():.4f}"
        )

    manual_sum = _summary_metrics(pd.read_csv(manual_csv), args.ch)
    auto_sum = _summary_metrics(pd.read_csv(auto_csv), args.ch)
    if not manual_sum.empty and not auto_sum.empty:
        summary = manual_sum.merge(
            auto_sum,
            on=["group", "set_label"],
            suffixes=("_manual", "_auto"),
        )
        summary["delta_FF0_diff"] = summary["delta_FF0_manual"] - summary["delta_FF0_auto"]
        summary_out = os.path.join(out_dir, "roi_auto_vs_manual_set_summary.csv")
        summary.to_csv(summary_out, index=False)
        print(f"Wrote per-set summary: {summary_out}")
        print("delta_FF0 (manual - auto) per set:")
        print(
            summary[
                ["group", "set_label", "delta_FF0_manual", "delta_FF0_auto", "delta_FF0_diff"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        print(
            f"delta_FF0_diff: median={summary['delta_FF0_diff'].median():.4f}, "
            f"mean={summary['delta_FF0_diff'].mean():.4f}, "
            f"max|diff|={summary['delta_FF0_diff'].abs().max():.4f}"
        )


if __name__ == "__main__":
    main()
