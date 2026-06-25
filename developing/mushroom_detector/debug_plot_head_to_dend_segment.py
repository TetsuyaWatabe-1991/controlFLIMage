# -*- coding: utf-8 -*-
"""
Plot head-to-dendrite distance segments for manual debugging.

Shows two distance definitions on the same Z-local MIP:
  - RESPAN bandpass metric (head_euclidean_dist_to_dend via labeled_dendrites EDT)
  - Mushroom shaft_to_head_um (nearest nnU-Net dendrite class pixel in XY)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from respan_mushroom_core import (  # noqa: E402
    DENDRITE_LABEL_VALUE,
    SHAFT_Z_HALF_WINDOW,
    _z_window_indices,
    nearest_shaft_anchor_xy,
    respan_export_paths,
    respan_run_dir,
)

VOL_CH_LABELED_DENDRITES = 4


def load_row_from_png(png_path: str) -> pd.Series:
    png = Path(png_path)
    savefolder = png.parent
    base_name = png.stem.rsplit("_", 1)[0]
    spine_idx = int(png.stem.rsplit("_", 1)[1])
    feat_path = savefolder / f"{base_name}_respan_mushroom_features.csv"
    feat = pd.read_csv(feat_path)
    row = feat.loc[feat["spine_index"] == spine_idx]
    if row.empty:
        raise ValueError(f"spine_index {spine_idx} not found in {feat_path}")
    return row.iloc[0]


def load_respan_arrays(
    flim_path: Path,
    *,
    xy_um: float,
    z_um: float,
    channel: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path, str]:
    run_dir = respan_run_dir(flim_path)
    tiff_path, _ = respan_export_paths(flim_path, channel=channel)
    tiff_stem = tiff_path.stem
    raw = tf.imread(tiff_path)
    if raw.ndim == 4:
        raw = raw[:, 0]
    labels = tf.imread(run_dir / "Validation_Data" / "Segmentation_Labels" / f"{tiff_stem}.tif")
    vol = tf.imread(run_dir / "Validation_Data" / "Validation_Vols" / f"{tiff_stem}.tif")
    dd = tf.imread(
        run_dir / "Validation_Data" / "Validation_Vols" / f"{tiff_stem}_dendrite_distance.tif"
    ).astype(np.float64)
    labeled_dendrites = vol[:, VOL_CH_LABELED_DENDRITES, :, :]
    return raw, labels.astype(np.int16), labeled_dendrites, dd, run_dir, tiff_stem


def raw_local_z_mip(
    raw_zyx: np.ndarray,
    z_pix: float,
    *,
    z_half_window: int = SHAFT_Z_HALF_WINDOW,
) -> tuple[np.ndarray, int, int]:
    z_idx = int(np.clip(round(z_pix), 0, raw_zyx.shape[0] - 1))
    z0, z1 = _z_window_indices(z_idx, raw_zyx.shape[0], z_half_window)
    return raw_zyx[z0:z1].max(axis=0), z0, z1


def nearest_labeled_dendrite_point(
    labeled_dendrites_zyx: np.ndarray,
    z_pix: float,
    y_pix: float,
    x_pix: float,
    xy_um: float,
    z_um: float,
) -> tuple[np.ndarray, float]:
    mask = labeled_dendrites_zyx > 0
    pts = np.argwhere(mask)
    if len(pts) == 0:
        raise ValueError("No labeled dendrite voxels found.")
    query = np.array([round(z_pix), round(y_pix), round(x_pix)], dtype=float)
    dists = np.sqrt(
        ((pts[:, 0] - query[0]) * z_um) ** 2
        + ((pts[:, 1] - query[1]) * xy_um) ** 2
        + ((pts[:, 2] - query[2]) * xy_um) ** 2
    )
    idx = int(dists.argmin())
    return pts[idx], float(dists[idx])


def nearest_class_dendrite_point(
    class_labels_zyx: np.ndarray,
    z_pix: float,
    y_pix: float,
    x_pix: float,
    xy_um: float,
    z_um: float,
) -> tuple[np.ndarray, float]:
    mask = class_labels_zyx == DENDRITE_LABEL_VALUE
    pts = np.argwhere(mask)
    if len(pts) == 0:
        raise ValueError("No class dendrite voxels found.")
    query = np.array([round(z_pix), round(y_pix), round(x_pix)], dtype=float)
    dists = np.sqrt(
        ((pts[:, 0] - query[0]) * z_um) ** 2
        + ((pts[:, 1] - query[1]) * xy_um) ** 2
        + ((pts[:, 2] - query[2]) * xy_um) ** 2
    )
    idx = int(dists.argmin())
    return pts[idx], float(dists[idx])


def map_head_crop_to_global(
    head_crop: np.ndarray,
    volume_shape: tuple[int, int, int],
    head_z: float,
    head_y: float,
    head_x: float,
    *,
    z0: int,
    y0: int,
    x0: int,
) -> np.ndarray:
    head_global = np.zeros(volume_shape, dtype=bool)
    for p in np.argwhere(head_crop):
        gz = int(p[0] + z0)
        gy = int(p[1] + y0)
        gx = int(p[2] + x0)
        if (
            0 <= gz < volume_shape[0]
            and 0 <= gy < volume_shape[1]
            and 0 <= gx < volume_shape[2]
        ):
            head_global[gz, gy, gx] = True
    return head_global


def load_head_mask_from_spine_vols(
    run_dir: Path,
    tiff_stem: str,
    respan_spine_id: int,
    head_z: float,
    head_y: float,
    head_x: float,
    class_labels_zyx: np.ndarray,
    dendrite_distance_vox: np.ndarray,
    xy_um: float,
) -> np.ndarray:
    """Map RESPAN Spine_vols head channel back into full-image ZYX coordinates."""
    vol_paths = sorted(run_dir.glob(f"Spine_Arrays/Spine_vols_{tiff_stem}_b*.tif"))
    if not vol_paths:
        raise FileNotFoundError(f"No Spine_vols under {run_dir / 'Spine_Arrays'}")
    spine_vols = tf.imread(vol_paths[0])
    det_csv = run_dir / "Tables" / f"{tiff_stem}_detected_spines.csv"
    det = pd.read_csv(det_csv)
    spine_ids = det["spine_id"].astype(int).tolist()
    if respan_spine_id not in spine_ids:
        raise ValueError(f"spine_id {respan_spine_id} not in {det_csv}")
    sv_idx = spine_ids.index(respan_spine_id)
    head_crop = spine_vols[sv_idx, :, 1, :, :] > 0
    cz = int(round(head_z))
    cy = int(round(head_y))
    cx = int(round(head_x))
    best_mask: np.ndarray | None = None
    best_score = -1.0
    search = range(-8, 9)
    for dz in search:
        for dy in search:
            for dx in search:
                z0 = cz - head_crop.shape[0] // 2 + dz
                y0 = cy - head_crop.shape[1] // 2 + dy
                x0 = cx - head_crop.shape[2] // 2 + dx
                head_global = map_head_crop_to_global(
                    head_crop,
                    class_labels_zyx.shape,
                    head_z,
                    head_y,
                    head_x,
                    z0=z0,
                    y0=y0,
                    x0=x0,
                )
                if not head_global.any():
                    continue
                overlap = np.count_nonzero(
                    head_global & (class_labels_zyx == 1)
                )
                min_dd = float((dendrite_distance_vox[head_global] * xy_um).min())
                score = overlap - 0.01 * min_dd
                if score > best_score:
                    best_score = score
                    best_mask = head_global
    if best_mask is None or not best_mask.any():
        raise ValueError("Could not align Spine_vols head crop to full-image coordinates.")
    return best_mask


def respan_metric_point_on_head(
    head_mask_3d: np.ndarray,
    dendrite_distance_vox: np.ndarray,
    xy_um: float,
) -> tuple[np.ndarray, float]:
    if not head_mask_3d.any():
        raise ValueError("Empty head mask for RESPAN metric point search.")
    dd_um = dendrite_distance_vox * xy_um
    vals = np.where(head_mask_3d, dd_um, np.inf)
    idx = int(np.argmin(vals))
    pt = np.array(np.unravel_index(idx, head_mask_3d.shape))
    return pt, float(vals.reshape(-1)[idx])


def plot_distance_segments(
    png_path: str,
    output_path: str | None = None,
    *,
    channel: int = 2,
) -> str:
    row = load_row_from_png(png_path)
    flim_path = Path(str(row["flim_path"]))
    head_z = float(row["head_z_pix"])
    head_y = float(row["head_y_pix"])
    head_x = float(row["head_x_pix"])
    spine_id = int(row["respan_spine_id"])
    respan_htd = float(row["respan_head_euclidean_dist_to_dend"])
    shaft_to_head = float(row["shaft_to_head_um"])
    xy_um = float(row["xy_pixel_um"])
    z_um = float(row["z_pixel_um"])

    raw, class_labels, labeled_dendrites, dd_vox, run_dir, tiff_stem = load_respan_arrays(
        flim_path,
        xy_um=xy_um,
        z_um=z_um,
        channel=channel,
    )
    head_mask_3d = load_head_mask_from_spine_vols(
        run_dir,
        tiff_stem,
        spine_id,
        head_z,
        head_y,
        head_x,
        class_labels,
        dd_vox,
        xy_um,
    )
    z_idx = int(np.clip(round(head_z), 0, class_labels.shape[0] - 1))
    spine_mask_2d = class_labels[z_idx] == 1

    mip, z0, z1 = raw_local_z_mip(raw, head_z)
    class_dend_mip = np.any(class_labels[z0:z1] == DENDRITE_LABEL_VALUE, axis=0)
    labeled_dend_mip = np.any(labeled_dendrites[z0:z1] > 0, axis=0)

    shaft_dist, shaft_y, shaft_x = nearest_shaft_anchor_xy(
        class_labels,
        head_z,
        head_y,
        head_x,
        spine_mask_2d,
        xy_um,
    )
    class_pt, class_dist = nearest_class_dendrite_point(
        class_labels, head_z, head_y, head_x, xy_um, z_um
    )
    labeled_pt, labeled_dist = nearest_labeled_dendrite_point(
        labeled_dendrites, head_z, head_y, head_x, xy_um, z_um
    )
    metric_pt, metric_dist = respan_metric_point_on_head(head_mask_3d, dd_vox, xy_um)

    if output_path is None:
        out_dir = Path(png_path).parents[1] / "distance_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{Path(png_path).stem}_htd_segments.png")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(mip, cmap="gray", origin="upper")
    ax.plot(head_x, head_y, "o", color="yellow", markersize=9, label="Head centroid")

    ax.plot(
        [head_x, shaft_x],
        [head_y, shaft_y],
        color="lime",
        linewidth=2.5,
        label=f"shaft_to_head (class dend) {shaft_dist:.2f} um",
    )
    ax.plot(shaft_x, shaft_y, "s", color="lime", markersize=7)

    ax.plot(
        [head_x, labeled_pt[2]],
        [head_y, labeled_pt[1]],
        color="red",
        linewidth=2.5,
        linestyle="--",
        label=f"nearest labeled_dendrite {labeled_dist:.2f} um",
    )
    ax.plot(labeled_pt[2], labeled_pt[1], "s", color="red", markersize=7)

    metric_labeled_pt, _ = nearest_labeled_dendrite_point(
        labeled_dendrites,
        metric_pt[0],
        metric_pt[1],
        metric_pt[2],
        xy_um,
        z_um,
    )
    ax.plot(
        [metric_pt[2], metric_labeled_pt[2]],
        [metric_pt[1], metric_labeled_pt[1]],
        color="magenta",
        linewidth=2.5,
        label=(
            f"RESPAN bandpass head->labeled_dend {metric_dist:.2f} um "
            f"(Z={metric_pt[0]})"
        ),
    )
    ax.plot(metric_pt[2], metric_pt[1], "x", color="magenta", markersize=10)
    ax.plot(metric_labeled_pt[2], metric_labeled_pt[1], "D", color="magenta", markersize=6)

    ax.contour(class_dend_mip, levels=[0.5], colors=["cyan"], linewidths=1.0)
    ax.contour(labeled_dend_mip, levels=[0.5], colors=["orange"], linewidths=1.0, linestyles="--")

    note = (
        f"spine_id={spine_id}  RESPAN head_euclidean_dist_to_dend={respan_htd:.2f} um\n"
        f"min over RESPAN head mask x labeled_dendrites EDT = {metric_dist:.2f} um\n"
        f"shaft_to_head_um (feature CSV) = {shaft_to_head:.2f} um\n"
        f"cyan = nnU-Net dendrite class | orange dashed = labeled_dendrites\n"
        f"RESPAN EDT ignores nnU-Net dendrite where labeled_dendrites has a gap."
    )
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.75, "pad": 5},
    )
    ax.set_title(f"Head-to-dend distance debug: {Path(png_path).name}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote: {output_path}")
    print(note.replace("\n", " | "))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot head-to-dend distance segments.")
    parser.add_argument("--png", required=True, help="Per-spine review or source PNG path.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--channel", type=int, default=2)
    args = parser.parse_args()
    plot_distance_segments(args.png, args.output, channel=args.channel)


if __name__ == "__main__":
    main()
