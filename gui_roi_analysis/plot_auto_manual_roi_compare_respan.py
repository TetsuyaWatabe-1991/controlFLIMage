# -*- coding: utf-8 -*-
"""
Per-spine figure: auto vs manual ROI comparison.

Layout (1 row x 5 columns):
  [Ch2 intensity time series] | [Pre auto] | [Pre manual] | [Post 25-35min auto] | [Post 25-35min manual]

Usage:
    python plot_auto_manual_roi_compare_respan.py --all
    python plot_auto_manual_roi_compare_respan.py --group pos1__highmag_1_ --set-label 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.ticker import MaxNLocator
from skimage.segmentation import find_boundaries

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ongoing" / "ASIcontroller"))

from gui_roi_respan_seg_masks import (  # noqa: E402
    _load_mask_2d,
    highmag_savefolder_from_filepath_without_number,
    match_uncaging_record_for_set,
    seg_mask_paths,
)
from respan_uncaging_log import parse_uncaging_records  # noqa: E402

UNC_FIRST_FRAME_DICT = {33: 2, 55: 5}
LTP_POST_MIN_WINDOW = (25, 35)
CH = 2
ROI_COLOR = {"Spine": "red", "DendriticShaft": "blue", "Background": "green"}
ROI_TYPES_PLOT = ["Spine", "DendriticShaft"]


def _highmag_side_length_um(statedict: dict) -> float:
    zoom = float(statedict.get("State.Acq.zoom", 15))
    fov = statedict.get("State.Acq.FOV_default", [128.0 * zoom, 128.0 * zoom])
    return float(fov[0]) / zoom


def add_intensity_normalized(ts_df: pd.DataFrame, ch: int = CH) -> pd.DataFrame:
    """
    Normalize Spine intensity per set (same as 20260701_tdTom_LTPanalysis_10h_bin.py):
      intensity_div_by_nAve = intensity / nAveFrame
      intensity_normalized = intensity_div_by_nAve / mean(pre) - 1
    """
    out = ts_df.copy()
    int_col = f"Spine_Ch{ch}_intensity"
    out["intensity_div_by_nAve"] = out[int_col] / out["nAveFrame"]
    pre_mean = out.loc[out["phase"] == "pre", "intensity_div_by_nAve"].mean()
    if not np.isfinite(pre_mean) or pre_mean <= 0:
        out["intensity_normalized"] = np.nan
    else:
        out["intensity_normalized"] = out["intensity_div_by_nAve"] / pre_mean - 1.0
    return out


def add_aligned_time_sec(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Align elapsed_time_sec so uncaging trigger is 0 (same as LTP analysis script)."""
    out = ts_df.copy()
    out["aligned_time_sec"] = np.nan
    for group in out["group"].unique():
        for set_label in out[out["group"] == group]["set_label"].unique():
            mask = (out["group"] == group) & (out["set_label"] == set_label)
            each = out.loc[mask]
            unc = each[each["phase"] == "unc"]
            n_unc = len(unc)
            if n_unc not in UNC_FIRST_FRAME_DICT:
                raise ValueError(f"Unsupported unc frame count {n_unc} for {group} set {set_label}")
            trigger = unc["elapsed_time_sec"].iloc[UNC_FIRST_FRAME_DICT[n_unc] - 1]
            out.loc[mask, "aligned_time_sec"] = each["elapsed_time_sec"].values - trigger
    return out


def _stack_frame_idx_from_csv_row(
    row: pd.Series, set_df: pd.DataFrame, n_pre: int, n_unc: int
) -> int | None:
    phase = str(row.get("phase", ""))
    if phase == "pre":
        pre_df = set_df[set_df["phase"] == "pre"].sort_values("slice")
        order = list(pre_df.index)
        if row.name in order:
            return order.index(row.name)
    elif phase == "post":
        post_df = set_df[set_df["phase"] == "post"].sort_values("slice")
        order = list(post_df.index)
        if row.name in order:
            return n_pre + n_unc + order.index(row.name)
    elif phase == "unc":
        sl = int(row.get("slice", 0))
        return n_pre + sl
    return None


def _draw_mask_boundary(
    ax: plt.Axes,
    mask: np.ndarray,
    color: str,
    side_length_um: float,
    linewidth: float = 0.8,
) -> None:
    if mask is None or mask.size == 0 or not np.any(mask > 0):
        return
    mask_bool = mask > 0
    boundaries = find_boundaries(mask_bool, mode="thick")
    height, width = mask_bool.shape
    inc_x = side_length_um / width
    inc_y = side_length_um / height
    for y in range(height):
        for x in range(width):
            if not boundaries[y, x]:
                continue
            px = x * inc_x
            py = y * inc_y
            if y == 0 or mask_bool[y - 1, x] != mask_bool[y, x]:
                ax.plot([px, px + inc_x], [py, py], color=color, linewidth=linewidth)
            if y == height - 1 or mask_bool[y + 1, x] != mask_bool[y, x]:
                ax.plot([px, px + inc_x], [py + inc_y, py + inc_y], color=color, linewidth=linewidth)
            if x == 0 or mask_bool[y, x - 1] != mask_bool[y, x]:
                ax.plot([px, px], [py, py + inc_y], color=color, linewidth=linewidth)
            if x == width - 1 or mask_bool[y, x + 1] != mask_bool[y, x]:
                ax.plot([px + inc_x, px + inc_x], [py, py + inc_y], color=color, linewidth=linewidth)


def _plot_image_with_rois(
    ax: plt.Axes,
    image_2d: np.ndarray,
    masks: dict[str, np.ndarray],
    side_length_um: float,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    ax.imshow(
        image_2d,
        cmap="gray",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
        extent=(0, side_length_um, side_length_um, 0),
    )
    for roi_type in ROI_TYPES_PLOT:
        if roi_type in masks:
            _draw_mask_boundary(ax, masks[roi_type], ROI_COLOR[roi_type], side_length_um)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _load_manual_masks(
    tiff_dir: str, base_name: str, frame_idx: int, roi_types: list[str]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for roi_type in roi_types:
        path = os.path.join(tiff_dir, f"{base_name}_{roi_type}_roi_mask.tif")
        if not os.path.exists(path):
            continue
        stack = tifffile.imread(path)
        if stack.ndim == 2:
            out[roi_type] = stack
        elif 0 <= frame_idx < stack.shape[0]:
            out[roi_type] = stack[frame_idx]
    return out


def _load_auto_masks(seg_paths: dict[str, Path], roi_types: list[str]) -> dict[str, np.ndarray]:
    key_map = {
        "Spine": "Spine",
        "DendriticShaft": "DendriticShaft",
        "Background": "Background",
    }
    out: dict[str, np.ndarray] = {}
    for roi_type in roi_types:
        if key_map[roi_type] in seg_paths:
            out[roi_type] = _load_mask_2d(seg_paths[key_map[roi_type]])
    return out


def plot_one_spine(
    *,
    group: str,
    set_label: float,
    combined_df: pd.DataFrame,
    manual_csv: pd.DataFrame,
    auto_csv: pd.DataFrame,
    out_path: str,
    ch: int = CH,
) -> str:
    set_df = combined_df[
        (combined_df["group"] == group) & (combined_df["nth_set_label"] == set_label)
    ]
    if set_df.empty:
        raise ValueError(f"No combined_df rows for {group} set {set_label}")

    filepath_wo = set_df["filepath_without_number"].iloc[0]
    highmag_folder = highmag_savefolder_from_filepath_without_number(filepath_wo)
    records = parse_uncaging_records(highmag_folder)
    record = match_uncaging_record_for_set(set_df, records)
    if record is None:
        raise ValueError(f"No uncaging record for {group} set {set_label}")

    seg_paths = seg_mask_paths(highmag_folder, record.spine_stem)
    tiff_path = str(set_df["after_align_full_save_path"].iloc[0])
    tiff_dir = os.path.dirname(tiff_path)
    base_name = os.path.splitext(os.path.basename(tiff_path))[0]
    stack = tifffile.imread(tiff_path)
    side_length_um = _highmag_side_length_um(set_df.iloc[0]["statedict"])

    n_pre = int(set_df["n_pre_frames"].iloc[0])
    n_unc = int(set_df["n_unc_frames"].iloc[0])

    manual_ts = manual_csv[
        (manual_csv["group"] == group) & (manual_csv["set_label"] == set_label)
    ].copy()
    auto_ts = auto_csv[(auto_csv["group"] == group) & (auto_csv["set_label"] == set_label)].copy()
    manual_ts = add_aligned_time_sec(manual_ts)
    auto_ts = add_aligned_time_sec(auto_ts)
    manual_ts = add_intensity_normalized(manual_ts, ch=ch)
    auto_ts = add_intensity_normalized(auto_ts, ch=ch)

    pre_rows = manual_ts[manual_ts["phase"] == "pre"].sort_values("slice")
    if pre_rows.empty:
        raise ValueError(f"No pre frames for {group} set {set_label}")
    pre_row = pre_rows.iloc[-1]
    pre_idx = _stack_frame_idx_from_csv_row(pre_row, manual_ts, n_pre, n_unc)
    if pre_idx is None:
        pre_idx = len(pre_rows) - 1

    post_window = manual_ts[
        (manual_ts["phase"] == "post")
        & (manual_ts["aligned_time_sec"] >= LTP_POST_MIN_WINDOW[0] * 60)
        & (manual_ts["aligned_time_sec"] <= LTP_POST_MIN_WINDOW[1] * 60)
    ].sort_values("aligned_time_sec")
    if post_window.empty:
        post_window = manual_ts[manual_ts["phase"] == "post"].sort_values("aligned_time_sec")
    post_row = post_window.iloc[len(post_window) // 2]
    post_idx = _stack_frame_idx_from_csv_row(post_row, manual_ts, n_pre, n_unc)
    if post_idx is None:
        post_idx = n_pre + n_unc

    pre_img = stack[pre_idx]
    post_img = stack[post_idx]
    vmin = float(np.percentile(stack, 2))
    vmax = float(np.percentile(stack, 98))

    manual_pre_masks = _load_manual_masks(tiff_dir, base_name, pre_idx, ROI_TYPES_PLOT)
    manual_post_masks = _load_manual_masks(tiff_dir, base_name, post_idx, ROI_TYPES_PLOT)
    auto_masks = _load_auto_masks(seg_paths, ROI_TYPES_PLOT)

    group_set_id = f"{group}_{int(set_label)}"
    fig = plt.figure(figsize=(22, 4.5))
    fig.suptitle(
        f"{group_set_id}  |  spine: {record.spine_stem}",
        fontsize=12,
        fontweight="bold",
    )
    gs = fig.add_gridspec(1, 5, width_ratios=[2.2, 1, 1, 1, 1], wspace=0.35)

    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.plot(
        manual_ts["aligned_time_sec"],
        manual_ts["intensity_normalized"],
        color="C0",
        linewidth=1.2,
        alpha=0.85,
        label="Manual ROI",
    )
    ax_ts.plot(
        auto_ts["aligned_time_sec"],
        auto_ts["intensity_normalized"],
        color="C1",
        linewidth=1.2,
        alpha=0.85,
        linestyle="--",
        label="Auto ROI",
    )
    ax_ts.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax_ts.axvspan(
        LTP_POST_MIN_WINDOW[0] * 60,
        LTP_POST_MIN_WINDOW[1] * 60,
        color="lightgray",
        alpha=0.25,
    )
    xlim = ax_ts.get_xlim()
    ax_ts.plot([xlim[0], xlim[1]], [0, 0], "--", color="gray", linewidth=0.5)
    ax_ts.set_xlabel("Time (sec)")
    ax_ts.set_ylabel(r"$\Delta$spine volume (a.u.)")
    ax_ts.set_title(f"Spine Ch{ch} intensity (pre-normalized)")
    ax_ts.legend(loc="best", fontsize=9)
    ax_ts.grid(True, alpha=0.2)

    ax_pre_auto = fig.add_subplot(gs[0, 1])
    _plot_image_with_rois(
        ax_pre_auto,
        pre_img,
        auto_masks,
        side_length_um,
        "Pre (auto ROI)",
        vmin,
        vmax,
    )

    ax_pre_man = fig.add_subplot(gs[0, 2])
    _plot_image_with_rois(
        ax_pre_man,
        pre_img,
        manual_pre_masks,
        side_length_um,
        "Pre (manual ROI)",
        vmin,
        vmax,
    )

    post_title_suffix = f"{LTP_POST_MIN_WINDOW[0]}-{LTP_POST_MIN_WINDOW[1]} min"
    ax_post_auto = fig.add_subplot(gs[0, 3])
    _plot_image_with_rois(
        ax_post_auto,
        post_img,
        auto_masks,
        side_length_um,
        f"Post {post_title_suffix} (auto ROI)",
        vmin,
        vmax,
    )

    ax_post_man = fig.add_subplot(gs[0, 4])
    _plot_image_with_rois(
        ax_post_man,
        post_img,
        manual_post_masks,
        side_length_um,
        f"Post {post_title_suffix} (manual ROI)",
        vmin,
        vmax,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def iter_spine_sets(combined_df: pd.DataFrame) -> list[tuple[str, float]]:
    """Return (group, set_label) for every quantified spine set."""
    sets: list[tuple[str, float]] = []
    sub = combined_df[combined_df["nth_set_label"] >= 0]
    for group in sorted(sub["group"].unique()):
        for set_label in sorted(sub[sub["group"] == group]["nth_set_label"].unique()):
            sets.append((group, float(set_label)))
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot auto vs manual ROI comparison per spine")
    parser.add_argument(
        "--df-path",
        default=r"G:\ImagingData\Tetsuya\20260701\auto1\combined_df_respan.pkl",
    )
    parser.add_argument(
        "--manual-csv",
        default=None,
        help="Manual quant CSV (default: df-path with _intensity_lifetime_all_frames.csv)",
    )
    parser.add_argument(
        "--auto-csv",
        default=None,
        help="Auto quant CSV (default: ..._AUTO_roi.csv)",
    )
    parser.add_argument("--group", default="pos1__highmag_1_")
    parser.add_argument("--set-label", type=float, default=0.0)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for every spine set in combined_df",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output folder (default: <auto1>/compare)",
    )
    args = parser.parse_args()

    df_path = args.df_path
    base_dir = os.path.dirname(df_path)
    manual_csv = args.manual_csv or df_path.replace(".pkl", "_intensity_lifetime_all_frames.csv")
    auto_csv = args.auto_csv or df_path.replace(
        ".pkl", "_intensity_lifetime_all_frames_AUTO_roi.csv"
    )
    out_dir = args.out_dir or os.path.join(base_dir, "compare")
    os.makedirs(out_dir, exist_ok=True)

    combined_df = pd.read_pickle(df_path)
    manual_df = pd.read_csv(manual_csv)
    auto_df = pd.read_csv(auto_csv)

    if args.all:
        spine_sets = iter_spine_sets(combined_df)
        print(f"Generating {len(spine_sets)} comparison plots -> {out_dir}")
        ok, failed = 0, []
        for group, set_label in spine_sets:
            tag = f"{group.rstrip('_')}_set{int(set_label)}"
            out_path = os.path.join(out_dir, f"{tag}_auto_vs_manual_roi.png")
            try:
                plot_one_spine(
                    group=group,
                    set_label=set_label,
                    combined_df=combined_df,
                    manual_csv=manual_df,
                    auto_csv=auto_df,
                    out_path=out_path,
                )
                ok += 1
                print(f"  OK  {tag}")
            except Exception as exc:
                failed.append((tag, str(exc)))
                print(f"  FAIL {tag}: {exc}")
        print(f"Done: {ok}/{len(spine_sets)} saved to {out_dir}")
        if failed:
            print("Failed sets:")
            for tag, msg in failed:
                print(f"  {tag}: {msg}")
        return

    tag = f"{args.group.rstrip('_')}_set{int(args.set_label)}"
    out_path = os.path.join(out_dir, f"{tag}_auto_vs_manual_roi.png")
    saved = plot_one_spine(
        group=args.group,
        set_label=args.set_label,
        combined_df=combined_df,
        manual_csv=manual_df,
        auto_csv=auto_df,
        out_path=out_path,
    )
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
