# -*- coding: utf-8 -*-
"""Debug plot: raw / binarized shaft / filtered shaft for one mushroom spine."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from respan_mushroom_core import (  # noqa: E402
    SHAFT_FIT_RADIUS_UM,
    SHAFT_Z_HALF_WINDOW,
    _compute_spine_geometry,
    _dendrite_line_xy,
    _z_window_indices,
    base_name_from_flim_path,
    ensure_respan_analysis,
    estimate_dendrite_line,
    filter_mushroom_spines_respan,
    load_pixel_um,
    load_respan_volumes,
    load_spine_rows,
    raw_local_z_mip,
    respan_run_dir,
    savefolder_from_flim_path,
    shaft_mask_mip_near_z,
)

FLIM_PATH = (
    r"G:\ImagingData\Tetsuya\20260608\mushroom_1dend - Copy\pos1__highmag_1_002.flim"
)
# pos1__highmag_1_000 -> RESPAN spine 14; pos1__highmag_1_001 -> spine 1
TARGET_SPINE_ID = 14
OUT_STEM = "pos1__highmag_1_000"
CHANNEL = 2


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Plot shaft binarization/filter steps for one spine.")
    parser.add_argument("--flim", default=FLIM_PATH)
    parser.add_argument("--spine-id", type=int, default=TARGET_SPINE_ID)
    parser.add_argument("--out-stem", default=OUT_STEM, help="Output filename stem, e.g. pos1__highmag_1_000")
    parser.add_argument("--channel", type=int, default=CHANNEL, choices=[1, 2])
    return parser.parse_args()


def _shaft_pixels_in_radius(
    shaft_mask_2d: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    xy_pixel_um: float,
    radius_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(np.asarray(shaft_mask_2d, dtype=bool))
    if len(xs) == 0:
        return np.array([]), np.array([]), np.array([])
    dist_um = np.hypot((xs - anchor_x) * xy_pixel_um, (ys - anchor_y) * xy_pixel_um)
    nearby = dist_um <= radius_um
    return xs[nearby], ys[nearby], dist_um[nearby]


def _draw_markers(
    ax,
    *,
    raw_mip: np.ndarray,
    shaft_mask: np.ndarray,
    circle_x: np.ndarray,
    circle_y: np.ndarray,
    shaft_x: float,
    shaft_y: float,
    head_x: float,
    head_y: float,
    show_shaft: bool,
    show_fit: bool = False,
    xs_fit: np.ndarray | None = None,
    ys_fit: np.ndarray | None = None,
    dend_slope: float | None = None,
    dend_intercept: float | None = None,
) -> None:
    ax.imshow(raw_mip, cmap="gray", origin="upper")
    if show_shaft and shaft_mask.any():
        overlay = np.zeros((*shaft_mask.shape, 4), dtype=np.float32)
        overlay[shaft_mask] = (0.0, 1.0, 0.0, 0.45)
        ax.imshow(overlay, origin="upper")
    ax.plot(circle_x, circle_y, color="white", linewidth=1.0, linestyle="--")
    ax.plot(shaft_x, shaft_y, "+", color="magenta", markersize=12, markeredgewidth=1.5)
    ax.plot(head_x, head_y, "o", color="yellow", markersize=6, markeredgecolor="black")
    if show_fit and xs_fit is not None and len(xs_fit):
        ax.scatter(xs_fit, ys_fit, s=10, c="red", alpha=0.95)
        if dend_slope is not None and dend_intercept is not None:
            dend_x, dend_y = _dendrite_line_xy(raw_mip.shape, dend_slope, dend_intercept)
            ax.plot(dend_x, dend_y, color="cyan", linewidth=1.5)
    ax.axis("off")


def main() -> None:
    from pathlib import Path

    args = _parse_args()
    flim_path = Path(args.flim)
    target_spine_id = args.spine_id
    out_stem = args.out_stem
    channel = args.channel

    tiff_path, json_path = ensure_respan_analysis(flim_path, channel=channel, rerun=False)
    run_dir = respan_run_dir(flim_path)
    csv_path = run_dir / "Tables" / f"{tiff_path.stem}_detected_spines.csv"
    label_path = run_dir / "Validation_Data" / "Segmentation_Labels" / tiff_path.name

    xy_pixel_um, _, _ = load_pixel_um(json_path)
    rows = load_spine_rows(csv_path)
    row_by_id = {int(float(r["spine_id"])): r for r in rows}
    if target_spine_id not in row_by_id:
        raise KeyError(f"spine_id {target_spine_id} not found in {csv_path}")

    mushrooms = filter_mushroom_spines_respan(rows)
    if target_spine_id not in mushrooms:
        print(f"warning: spine {target_spine_id} is not in mushroom filter set")

    zyx = tf.imread(tiff_path).astype(np.float32)
    class_labels_zyx = tf.imread(label_path)
    respan_volumes = load_respan_volumes(run_dir, tiff_path.stem)

    row = row_by_id[target_spine_id]
    geom = _compute_spine_geometry(
        row, zyx, class_labels_zyx, xy_pixel_um, respan_volumes, target_spine_id
    )
    head = geom["head_zyx"]
    shaft_y, shaft_x = geom["shaft_anchor_yx"]
    dend_slope = geom["dend_slope"]
    dend_intercept = geom["dend_intercept"]

    raw_mip, z0_raw, z1_raw = raw_local_z_mip(zyx, head[0])
    z_idx = int(np.clip(round(head[0]), 0, class_labels_zyx.shape[0] - 1))
    z0_bin, z1_bin = _z_window_indices(z_idx, class_labels_zyx.shape[0], SHAFT_Z_HALF_WINDOW)

    shaft_binary = shaft_mask_mip_near_z(class_labels_zyx, head[0])

    xs_fit, ys_fit, dists_fit = _shaft_pixels_in_radius(
        shaft_binary, shaft_x, shaft_y, xy_pixel_um, SHAFT_FIT_RADIUS_UM
    )
    dend_slope_bin, dend_intercept_bin = estimate_dendrite_line(
        shaft_binary, shaft_y, shaft_x, xy_pixel_um=xy_pixel_um
    )

    radius_px = SHAFT_FIT_RADIUS_UM / xy_pixel_um
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = shaft_x + radius_px * np.cos(theta)
    circle_y = shaft_y + radius_px * np.sin(theta)

    save_dir = Path(savefolder_from_flim_path(str(flim_path)))
    out_path = save_dir / f"{out_stem}_shaft_binary_filter_debug.png"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    _draw_markers(
        axes[0],
        raw_mip=raw_mip,
        shaft_mask=np.zeros_like(shaft_binary),
        circle_x=circle_x,
        circle_y=circle_y,
        shaft_x=shaft_x,
        shaft_y=shaft_y,
        head_x=head[2],
        head_y=head[1],
        show_shaft=False,
    )
    axes[0].set_title(f"Raw Z-MIP (Z{z0_raw}-{z1_raw - 1})", fontsize=9)

    _draw_markers(
        axes[1],
        raw_mip=raw_mip,
        shaft_mask=shaft_binary,
        circle_x=circle_x,
        circle_y=circle_y,
        shaft_x=shaft_x,
        shaft_y=shaft_y,
        head_x=head[2],
        head_y=head[1],
        show_shaft=True,
    )
    axes[1].set_title(
        f"Binarized: dendrite class, Z{z0_bin}-{z1_bin - 1} OR (n={int(shaft_binary.sum())})",
        fontsize=9,
    )

    _draw_markers(
        axes[2],
        raw_mip=raw_mip,
        shaft_mask=shaft_binary,
        circle_x=circle_x,
        circle_y=circle_y,
        shaft_x=shaft_x,
        shaft_y=shaft_y,
        head_x=head[2],
        head_y=head[1],
        show_shaft=True,
        show_fit=True,
        xs_fit=xs_fit,
        ys_fit=ys_fit,
        dend_slope=dend_slope_bin,
        dend_intercept=dend_intercept_bin,
    )
    axes[2].set_title(
        f"Fit on binarized mask (in {SHAFT_FIT_RADIUS_UM}um={len(xs_fit)}, "
        f"slope={dend_slope_bin:.3f})",
        fontsize=9,
    )

    base_name = base_name_from_flim_path(str(flim_path))
    fig.suptitle(
        f"{out_stem}.png | RESPAN spine {target_spine_id} | "
        f"anchor=({shaft_x:.1f},{shaft_y:.1f}) head=({head[2]:.1f},{head[1]:.1f}) Z={head[0]:.1f} | "
        f"fit slope={dend_slope_bin:.3f}",
        fontsize=10,
    )
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"saved: {out_path}")
    print(f"  binarized Z range: Z{z0_bin}-{z1_bin - 1} (head Z idx={z_idx}, +/-{SHAFT_Z_HALF_WINDOW})")
    print(f"  anchor: ({shaft_x:.2f}, {shaft_y:.2f})")
    print(f"  head:   ({head[2]:.2f}, {head[1]:.2f}, Z={head[0]:.2f})")
    print(f"  binarized pixels: {int(shaft_binary.sum())}")
    print(f"  shaft pixels in {SHAFT_FIT_RADIUS_UM} um (on binarized): {len(xs_fit)}")
    if len(dists_fit):
        print(f"  dist_um range: {dists_fit.min():.3f} - {dists_fit.max():.3f}")


if __name__ == "__main__":
    main()
