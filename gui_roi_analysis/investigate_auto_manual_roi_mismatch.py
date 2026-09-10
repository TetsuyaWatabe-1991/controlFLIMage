# -*- coding: utf-8 -*-
"""
Investigate *why* manual Spine ROI corrections diverge from the RESPAN auto seg_mask ROI.

Reuses the labeled dataset built in evaluate_auto_roi_shift_algorithms.py (auto mask, manual
mask, reference image, ground-truth shift, is_pure_shift flag) and adds diagnostic measurements
instead of another shift-prediction algorithm:

- Does the human shift move the ROI away from the DendriticShaft seg_mask (distance-based, not
  overlap-based)?
- Is the manual mask centroid closer than the auto mask centroid to the local smoothed intensity
  peak (i.e. would a less noise-sensitive peak finder have found the same spot)?
- How many mismatches are large detection failures (RESPAN found a different spine/location)
  versus small local nudges?

Read-only: does not modify production code or existing outputs. Writes new files into a new
subfolder under the existing auto_roi_shift_algo_eval output directory.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, distance_transform_edt, gaussian_filter
from skimage.segmentation import find_boundaries

sys.path.append(os.path.dirname(__file__))

from evaluate_auto_roi_shift_algorithms import (  # noqa: E402
    OUTPUT_DIR,
    SESSION_DF_PATHS,
    SpineShiftSample,
    build_dataset,
    nearest_shaft_point,
)

INVESTIGATION_DIR = os.path.join(OUTPUT_DIR, "mismatch_investigation")
PEAK_SMOOTH_SIGMA_PX = 1.5
SEARCH_PAD = 12
N_EXAMPLES_PER_CATEGORY = 6


def _shaft_distance_map(shaft_mask: np.ndarray) -> np.ndarray | None:
    """Per-pixel distance (px) to the nearest DendriticShaft seg_mask pixel."""
    if shaft_mask is None or not shaft_mask.any():
        return None
    return distance_transform_edt(~shaft_mask)


def _mask_min_distance(mask: np.ndarray, dist_map: np.ndarray | None) -> float | None:
    if dist_map is None or not mask.any():
        return None
    return float(dist_map[mask].min())


def _local_smoothed_peak(
    sample: SpineShiftSample, image: np.ndarray, pad: int = SEARCH_PAD
) -> tuple[float, float]:
    """Location of the smoothed intensity peak in a neighborhood around the auto mask."""
    ys, xs = np.nonzero(sample.auto_mask)
    h, w = sample.auto_mask.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = np.clip(image[y0:y1, x0:x1], 0, None)
    smoothed = gaussian_filter(crop, sigma=PEAK_SMOOTH_SIGMA_PX)
    peak_y, peak_x = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    return (float(peak_y + y0), float(peak_x + x0))


def build_diagnostics_table(samples: list[SpineShiftSample]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        auto_cy, auto_cx = center_of_mass(sample.auto_mask)
        manual_cy, manual_cx = center_of_mass(sample.manual_mask)
        shift_mag = float(np.hypot(*sample.gt_shift))

        dist_map = _shaft_distance_map(sample.shaft_mask)
        shaft_dist_auto = _mask_min_distance(sample.auto_mask, dist_map)
        shaft_dist_manual = _mask_min_distance(sample.manual_mask, dist_map)

        peak_y, peak_x = _local_smoothed_peak(sample, sample.ref_image)
        auto_to_peak = float(np.hypot(auto_cy - peak_y, auto_cx - peak_x))
        manual_to_peak = float(np.hypot(manual_cy - peak_y, manual_cx - peak_x))

        post_peak_y = post_peak_x = auto_to_post_peak = manual_to_post_peak = np.nan
        if sample.post_ref_image is not None:
            post_peak_y, post_peak_x = _local_smoothed_peak(sample, sample.post_ref_image)
            auto_to_post_peak = float(np.hypot(auto_cy - post_peak_y, auto_cx - post_peak_x))
            manual_to_post_peak = float(np.hypot(manual_cy - post_peak_y, manual_cx - post_peak_x))

        away_from_shaft_cosine = None
        nearest_shaft = nearest_shaft_point(sample.auto_mask, sample.shaft_mask)
        if nearest_shaft is not None and shift_mag > 0:
            away_vec = np.array([auto_cy - nearest_shaft[0], auto_cx - nearest_shaft[1]])
            if np.linalg.norm(away_vec) > 1e-6:
                away_unit = away_vec / np.linalg.norm(away_vec)
                shift_vec = np.array([sample.gt_shift[0], sample.gt_shift[1]])
                away_from_shaft_cosine = float(
                    np.dot(shift_vec / shift_mag, away_unit)
                )

        if shift_mag == 0:
            category = "unchanged"
        elif not sample.is_pure_shift or shift_mag >= 8:
            category = "large_mismatch_likely_wrong_spine"
        elif shaft_dist_auto is not None and shaft_dist_auto <= 2.0:
            category = "near_shaft_nudge"
        else:
            category = "small_nudge_other"

        rows.append(
            {
                "sample_id": sample.sample_id,
                "session": sample.session,
                "group": sample.group,
                "set_label": sample.set_label,
                "spine_stem": sample.spine_stem,
                "is_pure_shift": sample.is_pure_shift,
                "gt_dy": sample.gt_shift[0],
                "gt_dx": sample.gt_shift[1],
                "shift_magnitude_px": shift_mag,
                "category": category,
                "shaft_dist_auto_px": shaft_dist_auto,
                "shaft_dist_manual_px": shaft_dist_manual,
                "shaft_dist_delta": (
                    None if shaft_dist_auto is None else shaft_dist_manual - shaft_dist_auto
                ),
                "auto_to_smoothed_peak_px": auto_to_peak,
                "manual_to_smoothed_peak_px": manual_to_peak,
                "manual_closer_to_peak": manual_to_peak < auto_to_peak - 1e-6,
                "has_post_image": sample.post_ref_image is not None,
                "auto_to_post_peak_px": auto_to_post_peak,
                "manual_to_post_peak_px": manual_to_post_peak,
                "manual_closer_to_post_peak": (
                    manual_to_post_peak < auto_to_post_peak - 1e-6
                    if sample.post_ref_image is not None
                    else None
                ),
                "away_from_shaft_cosine": away_from_shaft_cosine,
                "auto_mask_area_px": int(sample.auto_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def print_summary(diag: pd.DataFrame) -> None:
    print("\n=== Category counts ===")
    print(diag["category"].value_counts().to_string())

    changed = diag[diag["shift_magnitude_px"] > 0]
    print(f"\n=== Changed sets only (n={len(changed)}) ===")

    with_shaft = changed.dropna(subset=["shaft_dist_delta"])
    print(f"\nSets with a DendriticShaft seg_mask available: {len(with_shaft)}/{len(changed)}")
    if len(with_shaft):
        moved_away = (with_shaft["shaft_dist_delta"] > 0.5).mean()
        moved_closer = (with_shaft["shaft_dist_delta"] < -0.5).mean()
        print(f"  fraction where manual mask is FARTHER from shaft than auto: {moved_away:.3f}")
        print(f"  fraction where manual mask is CLOSER to shaft than auto:    {moved_closer:.3f}")
        print(f"  mean shaft_dist_auto (px):   {with_shaft['shaft_dist_auto_px'].mean():.2f}")
        print(f"  mean shaft_dist_manual (px): {with_shaft['shaft_dist_manual_px'].mean():.2f}")
        near = with_shaft[with_shaft["shaft_dist_auto_px"] <= 2.0]
        print(f"  of these, auto mask already within 2px of shaft: {len(near)}/{len(with_shaft)}")
        if len(near):
            print(f"    -> fraction that moved farther from shaft after manual edit: "
                  f"{(near['shaft_dist_delta'] > 0.5).mean():.3f}")

    print(f"\nmanual centroid closer to smoothed local intensity peak (pre-phase) than auto centroid: "
          f"{changed['manual_closer_to_peak'].mean():.3f} of changed sets")
    print(f"  mean auto->peak distance (px):   {changed['auto_to_smoothed_peak_px'].mean():.2f}")
    print(f"  mean manual->peak distance (px): {changed['manual_to_smoothed_peak_px'].mean():.2f}")

    have_post = changed[changed["has_post_image"]]
    if len(have_post):
        print(f"\nSame test using the POST-uncaging phase image (n={len(have_post)}):")
        print(f"  manual closer to post-phase peak: {have_post['manual_closer_to_post_peak'].mean():.3f}")
        print(f"  mean auto->post_peak distance (px):   {have_post['auto_to_post_peak_px'].mean():.2f}")
        print(f"  mean manual->post_peak distance (px): {have_post['manual_to_post_peak_px'].mean():.2f}")

    cosine = changed["away_from_shaft_cosine"].dropna()
    if len(cosine):
        print(f"\nShift-direction vs away-from-shaft direction (cosine similarity, n={len(cosine)}):")
        print(f"  mean cosine:   {cosine.mean():.3f}  (>0 means shift tends to point away from shaft)")
        print(f"  median cosine: {cosine.median():.3f}")
        print(f"  fraction with cosine > 0 (any away component): {(cosine > 0).mean():.3f}")
        print(f"  fraction with cosine > 0.5 (mostly away):      {(cosine > 0.5).mean():.3f}")

    corr = changed[["shift_magnitude_px", "auto_mask_area_px"]].corr().iloc[0, 1]
    print(f"\nCorrelation(shift magnitude, auto mask area): {corr:.3f}")
    print("  mask area by shift-magnitude bucket:")
    bucket = pd.cut(changed["shift_magnitude_px"], [0, 1.5, 3, 100], labels=["1-1px", "2-3px", "4px+"])
    print(changed.groupby(bucket, observed=True)["auto_mask_area_px"].agg(["mean", "median", "count"]))


def _draw_boundary(ax: plt.Axes, mask: np.ndarray, color: str) -> None:
    if not np.any(mask):
        return
    boundaries = find_boundaries(mask, mode="thick")
    ys, xs = np.nonzero(boundaries)
    ax.scatter(xs, ys, s=1.2, c=color)


def save_category_examples(
    samples: list[SpineShiftSample],
    diag: pd.DataFrame,
    out_dir: str,
    n_per_category: int = N_EXAMPLES_PER_CATEGORY,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sample_by_id = {s.sample_id: s for s in samples}

    for category, group_df in diag.groupby("category"):
        if category == "unchanged":
            continue
        pick = group_df.sort_values("shift_magnitude_px", ascending=False).head(n_per_category)
        for _, row in pick.iterrows():
            sample = sample_by_id[row["sample_id"]]
            vmin, vmax = np.percentile(sample.ref_image, [2, 98])

            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.imshow(sample.ref_image, cmap="gray", vmin=vmin, vmax=vmax)
            _draw_boundary(ax, sample.auto_mask, "red")
            _draw_boundary(ax, sample.manual_mask, "lime")
            if sample.shaft_mask.any():
                _draw_boundary(ax, sample.shaft_mask, "cyan")
            ax.set_title(
                f"{sample.sample_id}\nshift=({row['gt_dy']:.0f},{row['gt_dx']:.0f}) "
                f"shaft_dist auto={row['shaft_dist_auto_px']} manual={row['shaft_dist_manual_px']}",
                fontsize=7,
            )
            ax.axis("off")
            fig.tight_layout()
            safe_name = row["sample_id"].replace("|", "_").replace(os.sep, "-").replace(":", "")
            cat_dir = os.path.join(out_dir, category)
            os.makedirs(cat_dir, exist_ok=True)
            fig.savefig(os.path.join(cat_dir, f"{safe_name}.png"), dpi=140)
            plt.close(fig)
    print(f"Saved category example panels (red=auto, lime=manual, cyan=DendriticShaft) to {out_dir}")


def main() -> None:
    print("Building dataset from all sessions (reusing evaluate_auto_roi_shift_algorithms)...")
    samples = build_dataset(SESSION_DF_PATHS)
    print(f"Total Spine samples: {len(samples)}")

    diag = build_diagnostics_table(samples)
    os.makedirs(INVESTIGATION_DIR, exist_ok=True)
    diag.to_csv(os.path.join(INVESTIGATION_DIR, "mismatch_diagnostics.csv"), index=False)

    print_summary(diag)
    save_category_examples(samples, diag, os.path.join(INVESTIGATION_DIR, "examples"))
    print(f"\nAll investigation outputs saved to: {INVESTIGATION_DIR}")


if __name__ == "__main__":
    main()
