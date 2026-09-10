# -*- coding: utf-8 -*-
"""
Evaluate candidate shift-correction algorithms for auto (RESPAN seg_mask) Spine ROIs.

Background: comparing manually corrected Spine ROI masks against the raw RESPAN seg_mask
ROI across 5 imaging sessions showed that ~96% of manual edits are a pure integer-pixel
translation of the exact same mask shape (0 shape change, 0-20 px offset). This script
therefore treats "auto ROI correction" as a small local shift-prediction problem instead
of a segmentation problem, and benchmarks several shift-prediction algorithms against the
manual edits as ground truth, using leave-one-session-out cross-validation.

This script only reads existing FLIM/TIFF/mask files and never modifies production code or
existing output files. All results are written to a new output directory.

Usage:
    python evaluate_auto_roi_shift_algorithms.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import center_of_mass
from skimage.morphology import binary_dilation as _binary_dilation
from skimage.morphology import disk


def binary_dilation(mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    return _binary_dilation(mask, footprint=footprint)
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

SESSION_DF_PATHS: list[str] = [
    r"G:\ImagingData\Tetsuya\20260610\auto3\combined_df_respan.pkl",
    r"G:\ImagingData\Tetsuya\20260611\auto1\combined_df_respan.pkl",
    r"G:\ImagingData\Tetsuya\20260622\auto1\combined_df_respan.pkl",
    r"G:\ImagingData\Tetsuya\20260623\auto1\combined_df_respan.pkl",
    r"G:\ImagingData\Tetsuya\20260701\auto1\combined_df_respan.pkl",
]
OUTPUT_DIR = r"G:\ImagingData\Tetsuya\20260701\auto1\auto_roi_shift_algo_eval"
ROI_TYPE = "Spine"
SEARCH_RADIUS = 6
RING_WIDTH = 3
DILATE_PX = 4
N_WORST_EXAMPLES = 10

# Grid search candidates for the combined (contrast + shaft-repulsion + regularization) algorithm.
W_OVERLAP_GRID = [0.0, 0.5, 1.0, 2.0, 4.0]
W_REG_GRID = [0.0, 0.05, 0.1, 0.2, 0.4]

# Grid search candidates for the confidence-gated contrast algorithm: only move away from the
# zero-shift auto mask when the contrast improvement clears this margin, otherwise stay put.
MARGIN_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]


@dataclass
class SpineShiftSample:
    """One Spine ROI set with its auto (seg_mask) and manual ground-truth mask."""

    session: str
    group: str
    set_label: float
    spine_stem: str
    auto_mask: np.ndarray
    shaft_mask: np.ndarray
    manual_mask: np.ndarray
    ref_image: np.ndarray
    post_ref_image: np.ndarray | None
    gt_shift: tuple[int, int]
    is_pure_shift: bool
    sample_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.sample_id = f"{self.session}|{self.group}|{int(self.set_label)}"


def translate_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer-pixel translate a 2D mask, filling the vacated border with False."""
    height, width = mask.shape
    out = np.zeros_like(mask)
    ys0, ys1 = max(0, dy), min(height, height + dy)
    yd0, yd1 = max(0, -dy), min(height, height - dy)
    xs0, xs1 = max(0, dx), min(width, width + dx)
    xd0, xd1 = max(0, -dx), min(width, width - dx)
    out[ys0:ys1, xs0:xs1] = mask[yd0:yd1, xd0:xd1]
    return out


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a_bin, b_bin = mask_a > 0, mask_b > 0
    union = int(np.logical_or(a_bin, b_bin).sum())
    if union == 0:
        return 1.0
    inter = int(np.logical_and(a_bin, b_bin).sum())
    return inter / union


def nearest_shaft_point(mask: np.ndarray, shaft_mask: np.ndarray) -> tuple[float, float] | None:
    """Shaft pixel closest to the mask centroid (used as an away-from-shaft reference point)."""
    if not shaft_mask.any():
        return None
    ys, xs = np.nonzero(shaft_mask)
    cy, cx = center_of_mass(mask)
    d2 = (ys - cy) ** 2 + (xs - cx) ** 2
    idx = int(np.argmin(d2))
    return float(ys[idx]), float(xs[idx])


def _resolve_tiff_path(set_df: pd.DataFrame) -> str | None:
    tiff_path = set_df["after_align_full_save_path"].iloc[0] if "after_align_full_save_path" in set_df else None
    if pd.isna(tiff_path) or not os.path.exists(str(tiff_path)):
        tiff_path = set_df["after_align_save_path"].iloc[0]
    if pd.isna(tiff_path) or not os.path.exists(str(tiff_path)):
        return None
    return str(tiff_path)


def _pure_shift(manual2d: np.ndarray, auto2d: np.ndarray) -> tuple[bool, tuple[int, int]]:
    """Check whether manual2d equals auto2d translated by a whole-pixel offset."""
    if manual2d.shape != auto2d.shape:
        return False, (0, 0)
    cy_m, cx_m = center_of_mass(manual2d > 0)
    cy_a, cx_a = center_of_mass(auto2d > 0)
    dy, dx = cy_m - cy_a, cx_m - cx_a
    dyi, dxi = int(round(dy)), int(round(dx))
    if abs(dy - dyi) > 1e-6 or abs(dx - dxi) > 1e-6:
        return False, (dyi, dxi)
    shifted = translate_mask(auto2d > 0, dyi, dxi)
    return bool(np.array_equal(shifted, manual2d > 0)), (dyi, dxi)


def build_dataset(df_paths: list[str]) -> list[SpineShiftSample]:
    """Collect one SpineShiftSample per (session, group, set_label) with both masks present."""
    samples: list[SpineShiftSample] = []
    for df_path in df_paths:
        if not os.path.exists(df_path):
            print(f"  session missing, skip: {df_path}")
            continue
        session = Path(df_path).parent.name + "_" + Path(df_path).parents[1].name
        combined_df = pd.read_pickle(df_path)
        n_before = len(samples)
        for filepath_wo in combined_df["filepath_without_number"].unique():
            filegroup = combined_df[combined_df["filepath_without_number"] == filepath_wo]
            highmag_folder = highmag_savefolder_from_filepath_without_number(filepath_wo)
            try:
                records = parse_uncaging_records(highmag_folder)
            except Exception:
                records = []
            if not records:
                continue

            for group in filegroup["group"].unique():
                group_df = filegroup[filegroup["group"] == group]
                for set_label in group_df["nth_set_label"].unique():
                    if set_label == -1:
                        continue
                    set_df = group_df[group_df["nth_set_label"] == set_label]
                    if "reject" in set_df.columns and (set_df["reject"] == 1).any():
                        continue
                    tiff_path = _resolve_tiff_path(set_df)
                    if tiff_path is None:
                        continue
                    record = match_uncaging_record_for_set(set_df, records)
                    if record is None:
                        continue
                    mask_paths = seg_mask_paths(highmag_folder, record.spine_stem)
                    if ROI_TYPE not in mask_paths:
                        continue

                    tiff_dir = os.path.dirname(tiff_path)
                    base_name = os.path.splitext(os.path.basename(tiff_path))[0]
                    manual_path = os.path.join(tiff_dir, f"{base_name}_{ROI_TYPE}_roi_mask.tif")
                    if not os.path.exists(manual_path):
                        continue

                    n_pre = int(set_df["n_pre_frames"].iloc[0])
                    if n_pre <= 0:
                        continue

                    manual_stack = tifffile.imread(manual_path)
                    mid = manual_stack.shape[0] // 2 if manual_stack.ndim == 3 else 0
                    manual2d = manual_stack[mid] if manual_stack.ndim == 3 else manual_stack
                    auto2d = _load_mask_2d(mask_paths[ROI_TYPE])
                    if manual2d.sum() == 0 or auto2d.sum() == 0:
                        continue

                    is_pure, gt_shift = _pure_shift(manual2d, auto2d)

                    shaft_mask = np.zeros_like(auto2d, dtype=bool)
                    if "DendriticShaft" in mask_paths:
                        shaft_mask = _load_mask_2d(mask_paths["DendriticShaft"]) > 0

                    try:
                        full_stack = tifffile.imread(tiff_path)
                    except Exception:
                        continue
                    if full_stack.ndim != 3 or full_stack.shape[0] < n_pre:
                        continue
                    ref_image = full_stack[:n_pre].mean(axis=0).astype(np.float32)

                    n_unc = int(set_df["n_unc_frames"].iloc[0])
                    post_start = n_pre + n_unc
                    post_ref_image = None
                    if full_stack.shape[0] > post_start:
                        post_ref_image = full_stack[post_start:].mean(axis=0).astype(np.float32)

                    samples.append(
                        SpineShiftSample(
                            session=session,
                            group=group,
                            set_label=float(set_label),
                            spine_stem=record.spine_stem,
                            auto_mask=auto2d > 0,
                            shaft_mask=shaft_mask,
                            manual_mask=manual2d > 0,
                            ref_image=ref_image,
                            post_ref_image=post_ref_image,
                            gt_shift=gt_shift,
                            is_pure_shift=is_pure,
                        )
                    )
        print(f"  {df_path}: {len(samples) - n_before} Spine samples")
    return samples


class LocalContext:
    """Small cropped working copy of a sample's arrays for fast shift grid search."""

    def __init__(self, sample: SpineShiftSample, pad: int) -> None:
        ys, xs = np.nonzero(sample.auto_mask)
        y0, y1 = max(0, ys.min() - pad), min(sample.auto_mask.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(sample.auto_mask.shape[1], xs.max() + pad + 1)
        self.y0, self.x0 = y0, x0
        self.auto_mask = sample.auto_mask[y0:y1, x0:x1]
        self.shaft_mask = sample.shaft_mask[y0:y1, x0:x1]
        self.image = np.clip(sample.ref_image[y0:y1, x0:x1], 0, None)
        scale = float(np.percentile(self.image, 90))
        self.scale = scale if scale > 1e-6 else 1.0
        self.auto_centroid = center_of_mass(self.auto_mask)


def _contrast(ctx: LocalContext, shifted: np.ndarray, ring_width: int) -> float:
    if shifted.sum() == 0:
        return -np.inf
    ring = binary_dilation(shifted, disk(ring_width)) & ~shifted
    inside = float(ctx.image[shifted].mean())
    outside_vals = ctx.image[ring]
    outside = float(outside_vals.mean()) if outside_vals.size else 0.0
    return (inside - outside) / ctx.scale


def alg_baseline(_sample: SpineShiftSample) -> tuple[int, int]:
    return (0, 0)


def alg_centroid_snap(sample: SpineShiftSample, dilate_px: int = DILATE_PX, radius: int = SEARCH_RADIUS) -> tuple[int, int]:
    dilated = binary_dilation(sample.auto_mask, disk(dilate_px))
    img = np.clip(sample.ref_image, 0, None)
    ys, xs = np.nonzero(dilated)
    if len(ys) == 0:
        return (0, 0)
    weights = img[ys, xs]
    if weights.sum() <= 0:
        return (0, 0)
    cy = float(np.average(ys, weights=weights))
    cx = float(np.average(xs, weights=weights))
    auto_cy, auto_cx = center_of_mass(sample.auto_mask)
    dy = int(np.clip(round(cy - auto_cy), -radius, radius))
    dx = int(np.clip(round(cx - auto_cx), -radius, radius))
    return (dy, dx)


def alg_template_contrast(sample: SpineShiftSample, radius: int = SEARCH_RADIUS, ring_width: int = RING_WIDTH) -> tuple[int, int]:
    ctx = LocalContext(sample, pad=radius + ring_width + 2)
    best_shift, best_score = (0, 0), -np.inf
    for dy, dx in product(range(-radius, radius + 1), range(-radius, radius + 1)):
        shifted = translate_mask(ctx.auto_mask, dy, dx)
        score = _contrast(ctx, shifted, ring_width)
        if score > best_score:
            best_score, best_shift = score, (dy, dx)
    return best_shift


def alg_shaft_repulsion(sample: SpineShiftSample, radius: int = SEARCH_RADIUS) -> tuple[int, int]:
    if not sample.shaft_mask.any() or not (sample.auto_mask & sample.shaft_mask).any():
        return (0, 0)
    ctx = LocalContext(sample, pad=radius + 2)
    best_shift, best_overlap, best_dist = (0, 0), None, None
    for dy, dx in product(range(-radius, radius + 1), range(-radius, radius + 1)):
        shifted = translate_mask(ctx.auto_mask, dy, dx)
        overlap = int((shifted & ctx.shaft_mask).sum())
        dist = dy * dy + dx * dx
        if best_overlap is None or overlap < best_overlap or (overlap == best_overlap and dist < best_dist):
            best_overlap, best_dist, best_shift = overlap, dist, (dy, dx)
    return best_shift


def alg_combined(
    sample: SpineShiftSample,
    radius: int = SEARCH_RADIUS,
    ring_width: int = RING_WIDTH,
    w_overlap: float = 2.0,
    w_reg: float = 0.1,
) -> tuple[int, int]:
    ctx = LocalContext(sample, pad=radius + ring_width + 2)
    best_shift, best_score = (0, 0), -np.inf
    for dy, dx in product(range(-radius, radius + 1), range(-radius, radius + 1)):
        shifted = translate_mask(ctx.auto_mask, dy, dx)
        if shifted.sum() == 0:
            continue
        contrast = _contrast(ctx, shifted, ring_width)
        overlap_frac = float((shifted & ctx.shaft_mask).sum()) / float(shifted.sum())
        reg = (dy * dy + dx * dx) / float(radius * radius)
        score = contrast - w_overlap * overlap_frac - w_reg * reg
        if score > best_score:
            best_score, best_shift = score, (dy, dx)
    return best_shift


def alg_confidence_gated(
    sample: SpineShiftSample,
    radius: int = SEARCH_RADIUS,
    ring_width: int = RING_WIDTH,
    margin: float = 0.1,
) -> tuple[int, int]:
    """Only move away from the auto mask if the contrast gain clears `margin`; else stay put."""
    ctx = LocalContext(sample, pad=radius + ring_width + 2)
    zero_score = _contrast(ctx, ctx.auto_mask, ring_width)
    best_shift, best_score = (0, 0), zero_score
    for dy, dx in product(range(-radius, radius + 1), range(-radius, radius + 1)):
        if dy == 0 and dx == 0:
            continue
        shifted = translate_mask(ctx.auto_mask, dy, dx)
        score = _contrast(ctx, shifted, ring_width)
        if score > best_score:
            best_score, best_shift = score, (dy, dx)
    if best_shift != (0, 0) and (best_score - zero_score) < margin:
        return (0, 0)
    return best_shift


def tune_confidence_gated_leave_one_session_out(samples: list[SpineShiftSample]) -> tuple[dict, pd.DataFrame]:
    """Grid-search `margin` per held-out session using the other sessions' mean IoU."""
    sessions = sorted({s.session for s in samples})
    fold_rows = []
    per_fold_margin: dict[str, float] = {}

    for held_out in sessions:
        train = [s for s in samples if s.session != held_out]
        test = [s for s in samples if s.session == held_out]
        best_margin, best_mean_iou = 0.1, -1.0
        for margin in MARGIN_GRID:
            ious = [
                mask_iou(translate_mask(s.auto_mask, *alg_confidence_gated(s, margin=margin)), s.manual_mask)
                for s in train
            ]
            mean_iou = float(np.mean(ious)) if ious else -1.0
            if mean_iou > best_mean_iou:
                best_mean_iou, best_margin = mean_iou, margin
        per_fold_margin[held_out] = best_margin
        test_df = evaluate_algorithm(test, alg_confidence_gated, margin=best_margin)
        test_df["cv_margin"] = best_margin
        fold_rows.append(test_df)
        print(f"  fold held_out={held_out}: margin={best_margin}, train_mean_iou={best_mean_iou:.3f}, "
              f"test_mean_iou={test_df['iou'].mean():.3f} (n={len(test)})")

    cv_results = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()

    best_global_margin, best_global_iou = 0.1, -1.0
    for margin in MARGIN_GRID:
        ious = [
            mask_iou(translate_mask(s.auto_mask, *alg_confidence_gated(s, margin=margin)), s.manual_mask)
            for s in samples
        ]
        mean_iou = float(np.mean(ious)) if ious else -1.0
        if mean_iou > best_global_iou:
            best_global_iou, best_global_margin = mean_iou, margin

    summary = {
        "per_fold_margin": per_fold_margin,
        "global_margin": best_global_margin,
        "global_mean_iou": best_global_iou,
    }
    return summary, cv_results


# Candidate fixed push magnitudes (px) for the shaft-away-only algorithm. 0 lets CV fall back to
# "never move" if even a small conservative push does not pay off on the training sessions.
PUSH_PX_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def alg_shaft_away_push(sample: SpineShiftSample, push_px: float = 1.0, radius: int = SEARCH_RADIUS) -> tuple[int, int]:
    """Push the auto mask a small fixed distance directly away from the nearest shaft pixel.

    This uses only the shaft-avoidance cue (the one diagnostic signal found to correlate, weakly
    but genuinely, with the real manual shift direction) and ignores local image intensity, which
    was shown to carry no usable signal for this task.
    """
    if push_px <= 0 or not sample.shaft_mask.any():
        return (0, 0)
    auto_cy, auto_cx = center_of_mass(sample.auto_mask)
    nearest = nearest_shaft_point(sample.auto_mask, sample.shaft_mask)
    if nearest is None:
        return (0, 0)
    away = np.array([auto_cy - nearest[0], auto_cx - nearest[1]])
    norm = float(np.linalg.norm(away))
    if norm < 1e-6:
        return (0, 0)
    unit = away / norm
    dy = int(np.clip(round(unit[0] * push_px), -radius, radius))
    dx = int(np.clip(round(unit[1] * push_px), -radius, radius))
    return (dy, dx)


def tune_shaft_away_push_leave_one_session_out(samples: list[SpineShiftSample]) -> tuple[dict, pd.DataFrame]:
    """Grid-search the fixed push magnitude per held-out session using the other sessions' mean IoU."""
    sessions = sorted({s.session for s in samples})
    fold_rows = []
    per_fold_push: dict[str, float] = {}

    for held_out in sessions:
        train = [s for s in samples if s.session != held_out]
        test = [s for s in samples if s.session == held_out]
        best_push, best_mean_iou = 0.0, -1.0
        for push_px in PUSH_PX_GRID:
            ious = [
                mask_iou(translate_mask(s.auto_mask, *alg_shaft_away_push(s, push_px=push_px)), s.manual_mask)
                for s in train
            ]
            mean_iou = float(np.mean(ious)) if ious else -1.0
            if mean_iou > best_mean_iou:
                best_mean_iou, best_push = mean_iou, push_px
        per_fold_push[held_out] = best_push
        test_df = evaluate_algorithm(test, alg_shaft_away_push, push_px=best_push)
        test_df["cv_push_px"] = best_push
        fold_rows.append(test_df)
        print(f"  fold held_out={held_out}: push_px={best_push}, train_mean_iou={best_mean_iou:.3f}, "
              f"test_mean_iou={test_df['iou'].mean():.3f} (n={len(test)})")

    cv_results = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()

    best_global_push, best_global_iou = 0.0, -1.0
    for push_px in PUSH_PX_GRID:
        ious = [
            mask_iou(translate_mask(s.auto_mask, *alg_shaft_away_push(s, push_px=push_px)), s.manual_mask)
            for s in samples
        ]
        mean_iou = float(np.mean(ious)) if ious else -1.0
        if mean_iou > best_global_iou:
            best_global_iou, best_global_push = mean_iou, push_px

    summary = {
        "per_fold_push_px": per_fold_push,
        "global_push_px": best_global_push,
        "global_mean_iou": best_global_iou,
    }
    return summary, cv_results


def evaluate_algorithm(samples: list[SpineShiftSample], predict_fn, **kwargs) -> pd.DataFrame:
    rows = []
    for sample in samples:
        dy, dx = predict_fn(sample, **kwargs)
        shifted = translate_mask(sample.auto_mask, dy, dx)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "session": sample.session,
                "group": sample.group,
                "set_label": sample.set_label,
                "is_pure_shift": sample.is_pure_shift,
                "gt_dy": sample.gt_shift[0],
                "gt_dx": sample.gt_shift[1],
                "pred_dy": dy,
                "pred_dx": dx,
                "shift_error": float(np.hypot(dy - sample.gt_shift[0], dx - sample.gt_shift[1])),
                "iou": mask_iou(shifted, sample.manual_mask),
                "baseline_iou": mask_iou(sample.auto_mask, sample.manual_mask),
            }
        )
    return pd.DataFrame(rows)


def tune_combined_leave_one_session_out(samples: list[SpineShiftSample]) -> tuple[dict, pd.DataFrame]:
    """Grid-search (w_overlap, w_reg) per held-out session using the other sessions' mean IoU."""
    sessions = sorted({s.session for s in samples})
    fold_rows = []
    per_fold_params: dict[str, tuple[float, float]] = {}

    for held_out in sessions:
        train = [s for s in samples if s.session != held_out]
        test = [s for s in samples if s.session == held_out]
        best_params, best_mean_iou = (2.0, 0.1), -1.0
        for w_overlap, w_reg in product(W_OVERLAP_GRID, W_REG_GRID):
            ious = [
                mask_iou(translate_mask(s.auto_mask, *alg_combined(s, w_overlap=w_overlap, w_reg=w_reg)), s.manual_mask)
                for s in train
            ]
            mean_iou = float(np.mean(ious)) if ious else -1.0
            if mean_iou > best_mean_iou:
                best_mean_iou, best_params = mean_iou, (w_overlap, w_reg)
        per_fold_params[held_out] = best_params
        test_df = evaluate_algorithm(test, alg_combined, w_overlap=best_params[0], w_reg=best_params[1])
        test_df["cv_w_overlap"], test_df["cv_w_reg"] = best_params
        fold_rows.append(test_df)
        print(f"  fold held_out={held_out}: params={best_params}, train_mean_iou={best_mean_iou:.3f}, "
              f"test_mean_iou={test_df['iou'].mean():.3f} (n={len(test)})")

    cv_results = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()

    best_global_params, best_global_iou = (2.0, 0.1), -1.0
    for w_overlap, w_reg in product(W_OVERLAP_GRID, W_REG_GRID):
        ious = [
            mask_iou(translate_mask(s.auto_mask, *alg_combined(s, w_overlap=w_overlap, w_reg=w_reg)), s.manual_mask)
            for s in samples
        ]
        mean_iou = float(np.mean(ious)) if ious else -1.0
        if mean_iou > best_global_iou:
            best_global_iou, best_global_params = mean_iou, (w_overlap, w_reg)

    summary = {
        "per_fold_params": per_fold_params,
        "global_params": best_global_params,
        "global_mean_iou": best_global_iou,
    }
    return summary, cv_results


def _draw_boundary(ax: plt.Axes, mask: np.ndarray, color: str) -> None:
    if not np.any(mask):
        return
    boundaries = find_boundaries(mask, mode="thick")
    ys, xs = np.nonzero(boundaries)
    ax.scatter(xs, ys, s=1.2, c=color)


def save_worst_case_examples(
    samples: list[SpineShiftSample],
    baseline_df: pd.DataFrame,
    best_shift_fn,
    best_kwargs: dict,
    out_dir: str,
    n_examples: int = N_WORST_EXAMPLES,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    worst_ids = baseline_df.sort_values("baseline_iou").head(n_examples)["sample_id"].tolist()
    sample_by_id = {s.sample_id: s for s in samples}
    for sample_id in worst_ids:
        sample = sample_by_id[sample_id]
        dy, dx = best_shift_fn(sample, **best_kwargs)
        predicted = translate_mask(sample.auto_mask, dy, dx)
        vmin, vmax = np.percentile(sample.ref_image, [2, 98])

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        titles = ["Auto (no shift)", f"Predicted (dy={dy}, dx={dx})", "Manual (ground truth)"]
        masks = [sample.auto_mask, predicted, sample.manual_mask]
        for ax, title, mask in zip(axes, titles, masks):
            ax.imshow(sample.ref_image, cmap="gray", vmin=vmin, vmax=vmax)
            _draw_boundary(ax, mask, "red")
            _draw_boundary(ax, sample.manual_mask, "lime")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{sample.sample_id}  spine={sample.spine_stem}", fontsize=9)
        fig.tight_layout()
        safe_name = sample_id.replace("|", "_").replace(os.sep, "-").replace(":", "")
        fig.savefig(os.path.join(out_dir, f"{safe_name}.png"), dpi=130)
        plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Building dataset from all sessions...")
    samples = build_dataset(SESSION_DF_PATHS)
    print(f"Total Spine samples: {len(samples)}")
    n_pure = sum(s.is_pure_shift for s in samples)
    print(f"Pure-integer-shift samples: {n_pure}/{len(samples)}")

    pd.DataFrame(
        [
            {
                "sample_id": s.sample_id,
                "session": s.session,
                "group": s.group,
                "set_label": s.set_label,
                "spine_stem": s.spine_stem,
                "is_pure_shift": s.is_pure_shift,
                "gt_dy": s.gt_shift[0],
                "gt_dx": s.gt_shift[1],
            }
            for s in samples
        ]
    ).to_csv(os.path.join(OUTPUT_DIR, "dataset_index.csv"), index=False)

    print("\nEvaluating simple algorithms (no tuning)...")
    results: dict[str, pd.DataFrame] = {}
    results["baseline"] = evaluate_algorithm(samples, alg_baseline)
    results["centroid_snap"] = evaluate_algorithm(samples, alg_centroid_snap)
    results["template_contrast"] = evaluate_algorithm(samples, alg_template_contrast)
    results["shaft_repulsion"] = evaluate_algorithm(samples, alg_shaft_repulsion)

    print("\nTuning combined algorithm via leave-one-session-out CV...")
    cv_summary, combined_cv_df = tune_combined_leave_one_session_out(samples)
    results["combined_cv"] = combined_cv_df
    print(f"Combined algorithm global params (fit on all sessions): {cv_summary['global_params']}, "
          f"global mean IoU={cv_summary['global_mean_iou']:.3f}")
    print(f"Per-fold chosen params: {cv_summary['per_fold_params']}")

    print("\nTuning confidence-gated algorithm via leave-one-session-out CV...")
    gated_summary, gated_cv_df = tune_confidence_gated_leave_one_session_out(samples)
    results["confidence_gated_cv"] = gated_cv_df
    print(f"Confidence-gated global margin (fit on all sessions): {gated_summary['global_margin']}, "
          f"global mean IoU={gated_summary['global_mean_iou']:.3f}")
    print(f"Per-fold chosen margins: {gated_summary['per_fold_margin']}")

    print("\nTuning shaft-away-only push algorithm via leave-one-session-out CV...")
    push_summary, push_cv_df = tune_shaft_away_push_leave_one_session_out(samples)
    results["shaft_away_push_cv"] = push_cv_df
    print(f"Shaft-away-push global push_px (fit on all sessions): {push_summary['global_push_px']}, "
          f"global mean IoU={push_summary['global_mean_iou']:.3f}")
    print(f"Per-fold chosen push_px: {push_summary['per_fold_push_px']}")

    for name, df in results.items():
        df.to_csv(os.path.join(OUTPUT_DIR, f"per_sample_results_{name}.csv"), index=False)

    summary_rows = []
    for name, df in results.items():
        summary_rows.append(
            {
                "algorithm": name,
                "n_samples": len(df),
                "mean_iou": df["iou"].mean(),
                "median_iou": df["iou"].median(),
                "mean_baseline_iou": df["baseline_iou"].mean(),
                "mean_iou_gain": (df["iou"] - df["baseline_iou"]).mean(),
                "frac_regressed": float((df["iou"] < df["baseline_iou"] - 1e-9).mean()),
                "frac_improved": float((df["iou"] > df["baseline_iou"] + 1e-9).mean()),
                "exact_shift_match_frac": float(
                    ((df["pred_dy"] == df["gt_dy"]) & (df["pred_dx"] == df["gt_dx"])).mean()
                ),
                "mean_shift_error": df["shift_error"].mean(),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_iou", ascending=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "algorithm_comparison_summary.csv"), index=False)
    print("\n=== Algorithm comparison (overall) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    per_session_rows = []
    for name, df in results.items():
        for session, group_df in df.groupby("session"):
            per_session_rows.append(
                {
                    "algorithm": name,
                    "session": session,
                    "n_samples": len(group_df),
                    "mean_iou": group_df["iou"].mean(),
                    "mean_baseline_iou": group_df["baseline_iou"].mean(),
                    "mean_iou_gain": (group_df["iou"] - group_df["baseline_iou"]).mean(),
                }
            )
    per_session_df = pd.DataFrame(per_session_rows)
    per_session_df.to_csv(os.path.join(OUTPUT_DIR, "algorithm_comparison_per_session.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    order = summary_df["algorithm"].tolist()
    data = [results[name]["iou"].values for name in order]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel("IoU vs manual Spine ROI")
    ax.set_title(f"Auto ROI shift-correction algorithm comparison (n={len(samples)} sets, 5 sessions)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "algorithm_iou_boxplot.png"), dpi=150)
    plt.close(fig)

    print("\nSaving worst-baseline-case example figures using the best-performing algorithm...")
    best_name = summary_df.iloc[0]["algorithm"]
    if best_name == "combined_cv":
        best_fn, best_kwargs = alg_combined, {
            "w_overlap": cv_summary["global_params"][0],
            "w_reg": cv_summary["global_params"][1],
        }
    elif best_name == "confidence_gated_cv":
        best_fn, best_kwargs = alg_confidence_gated, {"margin": gated_summary["global_margin"]}
    elif best_name == "shaft_away_push_cv":
        best_fn, best_kwargs = alg_shaft_away_push, {"push_px": push_summary["global_push_px"]}
    else:
        name_to_fn = {
            "baseline": alg_baseline,
            "centroid_snap": alg_centroid_snap,
            "template_contrast": alg_template_contrast,
            "shaft_repulsion": alg_shaft_repulsion,
        }
        best_fn, best_kwargs = name_to_fn[best_name], {}
    save_worst_case_examples(
        samples,
        results["baseline"],
        best_fn,
        best_kwargs,
        os.path.join(OUTPUT_DIR, "worst_case_examples"),
    )

    print(f"\nBest algorithm overall: {best_name}")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
