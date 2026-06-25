# -*- coding: utf-8 -*-
"""
Pick spaced high-mag ROI centers from RESPAN low-mag spine detections.

Clusters RESPAN spine heads in 3D (physical um), scores each cluster by spine count,
then uses lowmag_mushroom_branch_finder.select_spaced_positions for mutual spacing.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _SCRIPT_DIR.parents[1]
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from after_click_image_func import (  # noqa: E402
    get_abs_um_pos_from_center_3d,
    save_image_with_assigned_pos_3d,
    save_pix_pos_from_click_list,
    save_um_pos_from_click_list,
)
from FLIMageAlignment import get_xyz_pixel_um  # noqa: E402
from FLIMageFileReader2 import FileReader  # noqa: E402
from respan_mushroom_core import (  # noqa: E402
    _normalize_to_float,
    base_name_from_flim_path,
    build_class_overlay_mip,
    respan_export_paths,
    savefolder_from_flim_path,
)

DEFAULT_FLIM = r"G:\ImagingData\Tetsuya\20260608\pos1_001.flim"


def physical_distance_um(p1_zyx, p2_zyx, xy_um: float, z_um: float) -> float:
    """Same as lowmag_mushroom_branch_finder (3D physical distance)."""
    dz = (p1_zyx[0] - p2_zyx[0]) * z_um
    dy = (p1_zyx[1] - p2_zyx[1]) * xy_um
    dx = (p1_zyx[2] - p2_zyx[2]) * xy_um
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def select_spaced_positions(
    candidates: list[dict],
    max_num: int,
    min_spacing_um: float,
    xy_um: float,
    z_um: float,
) -> list[dict]:
    """Greedy pick by spiny_score with min 3D spacing (from lowmag_mushroom_branch_finder)."""
    ordered = sorted(candidates, key=lambda c: c["spiny_score"], reverse=True)
    selected: list[dict] = []
    for cand in ordered:
        if len(selected) >= max_num:
            break
        if all(
            physical_distance_um(cand["zyx"], s["zyx"], xy_um, z_um) >= min_spacing_um
            for s in selected
        ):
            selected.append(cand)
    return selected


def export_path_from_flim(flim_path: str | Path) -> Path:
    flim_path = Path(flim_path)
    return flim_path.parent / flim_path.stem


def cluster_spines_connected(
    heads_zyx: np.ndarray,
    xy_um: float,
    z_um: float,
    eps_um: float,
) -> np.ndarray:
    """Connected components: link spines within eps_um (3D physical distance)."""
    n = len(heads_zyx)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    phys = heads_zyx.astype(np.float64) * np.array([z_um, xy_um, xy_um], dtype=np.float64)
    cluster_id = 0
    for seed in range(n):
        if labels[seed] >= 0:
            continue
        queue = [seed]
        labels[seed] = cluster_id
        while queue:
            u = queue.pop()
            dists = np.linalg.norm(phys - phys[u], axis=1)
            for v in np.where((dists <= eps_um) & (labels < 0))[0]:
                labels[v] = cluster_id
                queue.append(int(v))
        cluster_id += 1
    return labels


def build_region_candidates_from_spines(
    feature_df: pd.DataFrame,
    *,
    xy_um: float,
    z_um: float,
    cluster_eps_um: float = 10.0,
    min_spines_per_region: int = 3,
    max_shaft_to_head_um: float = 5.0,
) -> list[dict]:
    """Group RESPAN spines into local shaft regions; score = spine count."""
    if feature_df.empty:
        return []

    df = feature_df.copy()
    if max_shaft_to_head_um > 0 and "shaft_to_head_um" in df.columns:
        df = df[df["shaft_to_head_um"].astype(float) <= max_shaft_to_head_um]
    if df.empty:
        return []

    heads = df[["head_z_pix", "head_y_pix", "head_x_pix"]].astype(float).values
    labels = cluster_spines_connected(heads, xy_um, z_um, cluster_eps_um)

    candidates: list[dict] = []
    for cluster_id in sorted(set(labels)):
        if cluster_id < 0:
            continue
        mask = labels == cluster_id
        n_spines = int(mask.sum())
        if n_spines < min_spines_per_region:
            continue
        cluster_heads = heads[mask]
        zyx = np.median(cluster_heads, axis=0)
        cluster_df = df.iloc[np.where(mask)[0]]
        candidates.append(
            {
                "cluster_id": int(cluster_id),
                "zyx": zyx.astype(int),
                "spiny_score": float(n_spines),
                "n_spines": n_spines,
                "cluster_heads_zyx": cluster_heads.astype(np.float64),
                "respan_spine_ids": cluster_df["respan_spine_id"].astype(int).tolist(),
                "mean_shaft_to_head_um": float(cluster_df["shaft_to_head_um"].mean())
                if "shaft_to_head_um" in cluster_df.columns
                else np.nan,
            }
        )
    return candidates


def _z_mip_window(zyx: np.ndarray, z_center: int, z_half: int) -> np.ndarray:
    z_center = int(np.clip(z_center, 0, zyx.shape[0] - 1))
    z0 = max(0, z_center - z_half)
    z1 = min(zyx.shape[0], z_center + z_half + 1)
    return zyx[z0:z1].max(axis=0).astype(np.float32)


def _cluster_crop_box_yx(
    cluster_heads_zyx: np.ndarray,
    image_shape_yx: tuple[int, int],
    *,
    xy_um: float,
    padding_um: float = 8.0,
    min_crop_um: float = 28.0,
) -> tuple[int, int, int, int]:
    """Square crop (y0, y1, x0, x1) around cluster spine heads with padding."""
    pad_px = padding_um / xy_um
    min_half_px = (min_crop_um / xy_um) / 2.0
    ys = cluster_heads_zyx[:, 1]
    xs = cluster_heads_zyx[:, 2]
    cy = float(np.median(ys))
    cx = float(np.median(xs))
    half_y = max(float(ys.max() - ys.min()) / 2.0 + pad_px, min_half_px)
    half_x = max(float(xs.max() - xs.min()) / 2.0 + pad_px, min_half_px)
    half = max(half_y, half_x)
    height, width = image_shape_yx
    y0 = int(np.floor(cy - half))
    y1 = int(np.ceil(cy + half))
    x0 = int(np.floor(cx - half))
    x1 = int(np.ceil(cx + half))
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(height, y1)
    x1 = min(width, x1)
    return y0, y1, x0, x1


def _pad_to_square(image: np.ndarray, fill: float = 0.0) -> np.ndarray:
    if image.ndim == 2:
        h, w = image.shape
        side = max(h, w)
        out = np.full((side, side), fill, dtype=image.dtype)
        oy = (side - h) // 2
        ox = (side - w) // 2
        out[oy : oy + h, ox : ox + w] = image
        return out
    h, w = image.shape[:2]
    side = max(h, w)
    out = np.full((side, side, image.shape[2]), fill, dtype=image.dtype)
    oy = (side - h) // 2
    ox = (side - w) // 2
    out[oy : oy + h, ox : ox + w] = image
    return out


def _draw_crop_rect_rgb(
    rgb: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    *,
    color: tuple[float, float, float] = (1.0, 1.0, 0.0),
    linewidth: int = 2,
) -> np.ndarray:
    out = rgb.copy()
    yy = [y0, y0, y1 - 1, y1 - 1, y0]
    xx = [x0, x1 - 1, x1 - 1, x0, x0]
    for t in range(linewidth):
        for i in range(4):
            y_a, y_b = yy[i], yy[i + 1]
            x_a, x_b = xx[i], xx[i + 1]
            if y_a == y_b:
                y_lo = max(0, y_a - t)
                y_hi = min(out.shape[0], y_a + t + 1)
                x_lo = max(0, min(x_a, x_b))
                x_hi = min(out.shape[1], max(x_a, x_b) + 1)
                out[y_lo:y_hi, x_lo:x_hi] = color
            else:
                x_lo = max(0, x_a - t)
                x_hi = min(out.shape[1], x_a + t + 1)
                y_lo = max(0, min(y_a, y_b))
                y_hi = min(out.shape[0], max(y_a, y_b) + 1)
                out[y_lo:y_hi, x_lo:x_hi] = color
    return out


def save_region_panel_png(
    zyx_array: np.ndarray,
    region: dict,
    region_index: int,
    savepath: Path,
    *,
    z_half_window: int = 3,
    crop_padding_um: float = 8.0,
    min_crop_um: float = 28.0,
    xy_um: float = 0.355,
    panel_size_px: int = 512,
) -> None:
    """
    Save side-by-side panel: left = full-FOV Z-MIP with crop box; right = cropped Z±window MIP.
    """
    z_center = int(region["zyx"][0])
    heads = region["cluster_heads_zyx"]
    y0, y1, x0, x1 = _cluster_crop_box_yx(
        heads,
        zyx_array.shape[1:],
        xy_um=xy_um,
        padding_um=crop_padding_um,
        min_crop_um=min_crop_um,
    )

    full_mip = _normalize_to_float(zyx_array.max(axis=0))
    local_mip = _normalize_to_float(_z_mip_window(zyx_array, z_center, z_half_window))

    full_rgb = np.stack([full_mip, full_mip, full_mip], axis=-1)
    full_rgb = _draw_crop_rect_rgb(full_rgb, y0, y1, x0, x1)
    crop_rgb = np.stack([local_mip[y0:y1, x0:x1]] * 3, axis=-1)

    left_sq = _pad_to_square(full_rgb, fill=0.0)
    right_sq = _pad_to_square(crop_rgb, fill=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(left_sq, origin="upper")
    axes[0].set_title(
        f"Region {region_index:02d} | full Z-MIP | crop box",
        fontsize=9,
    )
    axes[0].axis("off")
    z0 = max(0, z_center - z_half_window)
    z1 = min(zyx_array.shape[0] - 1, z_center + z_half_window)
    axes[1].imshow(right_sq, origin="upper")
    axes[1].set_title(
        f"cropped Z{z0}-{z1} MIP | n_spines={region['n_spines']}",
        fontsize=9,
    )
    axes[1].axis("off")
    fig.suptitle(
        f"cluster {region['cluster_id']} center ZYX="
        f"({region['zyx'][0]},{region['zyx'][1]},{region['zyx'][2]})",
        fontsize=10,
    )
    fig.savefig(savepath, bbox_inches="tight", pad_inches=0.08, dpi=panel_size_px // 5)
    plt.close(fig)


def export_region_panels(
    zyx_array: np.ndarray,
    selected: list[dict],
    output_dir: Path,
    *,
    xy_um: float,
    z_half_window: int = 3,
    crop_padding_um: float = 8.0,
    min_crop_um: float = 28.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, region in enumerate(selected):
        out_path = output_dir / (
            f"region_{idx:02d}_cluster{region['cluster_id']}_panel.png"
        )
        save_region_panel_png(
            zyx_array,
            region,
            idx,
            out_path,
            z_half_window=z_half_window,
            crop_padding_um=crop_padding_um,
            min_crop_um=min_crop_um,
            xy_um=xy_um,
        )
    print(f"  saved {len(selected)} region panel(s): {output_dir}")
    return output_dir


def save_region_overview_png(
    overlay_mip_rgb: np.ndarray,
    candidates: list[dict],
    selected: list[dict],
    savepath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(overlay_mip_rgb, origin="upper")
    selected_ids = {c["cluster_id"] for c in selected}
    for cand in candidates:
        z, y, x = cand["zyx"]
        is_sel = cand["cluster_id"] in selected_ids
        color = "yellow" if is_sel else "cyan"
        ms = 120 if is_sel else 50
        ax.scatter(x, y, s=ms, facecolors="none", edgecolors=color, linewidths=2 if is_sel else 1)
        ax.text(
            x + 2,
            y + 2,
            f"c{cand['cluster_id']} n={cand['n_spines']}",
            color=color,
            fontsize=7 if is_sel else 6,
        )
    ax.set_title(f"RESPAN clusters (yellow=selected, n={len(selected)})")
    ax.axis("off")
    fig.savefig(savepath, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)


def save_region_scores_csv(candidates: list[dict], selected: list[dict], csv_path: Path) -> None:
    selected_ids = {c["cluster_id"] for c in selected}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id",
                "z_pix",
                "y_pix",
                "x_pix",
                "n_spines",
                "mean_shaft_to_head_um",
                "selected",
                "respan_spine_ids",
            ],
        )
        writer.writeheader()
        for cand in candidates:
            z, y, x = cand["zyx"]
            writer.writerow(
                {
                    "cluster_id": cand["cluster_id"],
                    "z_pix": int(z),
                    "y_pix": int(y),
                    "x_pix": int(x),
                    "n_spines": cand["n_spines"],
                    "mean_shaft_to_head_um": cand.get("mean_shaft_to_head_um", np.nan),
                    "selected": int(cand["cluster_id"] in selected_ids),
                    "respan_spine_ids": ";".join(str(i) for i in cand["respan_spine_ids"]),
                }
            )


def pick_spaced_regions_from_respan(
    flim_path: str | Path,
    *,
    max_pos_cand_num: int = 12,
    min_spacing_um: float = 15.0,
    cluster_eps_um: float = 6.0,
    min_spines_per_region: int = 2,
    max_shaft_to_head_um: float = 5.0,
    skip_if_defined: bool = False,
    export_panels: bool = True,
    z_half_window: int = 3,
    crop_padding_um: float = 8.0,
    min_crop_um: float = 28.0,
) -> list[dict]:
    """
    Load RESPAN spine features, cluster spines, pick spaced region centers.

    Returns selected candidate dicts (same shape as lowmag branch finder for spacing).
    """
    flim_path = Path(flim_path)
    if not flim_path.is_file():
        raise FileNotFoundError(f"FLIM not found: {flim_path}")

    savefolder = Path(savefolder_from_flim_path(str(flim_path)))
    base_name = base_name_from_flim_path(str(flim_path))
    feature_csv = savefolder / f"{base_name}_respan_spine_features.csv"
    if not feature_csv.is_file():
        raise FileNotFoundError(
            f"RESPAN feature CSV not found: {feature_csv}\n"
            "Run detect_spines_respan_single.py on this FLIM first."
        )

    export_path = export_path_from_flim(flim_path)
    export_path.mkdir(parents=True, exist_ok=True)
    pos_pix_csv = export_path / "assigned_pixel_pos.csv"
    pos_um_csv = export_path / "assigned_relative_um_pos.csv"
    if skip_if_defined and pos_pix_csv.is_file():
        print(f"Skipping (already defined): {pos_pix_csv}")
        return []

    print(f"\n=== RESPAN spaced regions: {flim_path} ===")
    feature_df = pd.read_csv(feature_csv)
    print(f"  RESPAN spines in CSV: {len(feature_df)}")

    iminfo = FileReader()
    iminfo.read_imageFile(str(flim_path), True)
    zyx_array = np.array(iminfo.image).sum(axis=tuple([1, 2, 5]))
    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
    xy_um = float((x_um + y_um) / 2.0)
    print(f"  stack Z,Y,X = {zyx_array.shape}, pixel um xy={xy_um:.3f} z={z_um:.3f}")

    candidates = build_region_candidates_from_spines(
        feature_df,
        xy_um=xy_um,
        z_um=z_um,
        cluster_eps_um=cluster_eps_um,
        min_spines_per_region=min_spines_per_region,
        max_shaft_to_head_um=max_shaft_to_head_um,
    )
    print(
        f"  spine clusters (>={min_spines_per_region} spines, "
        f"eps={cluster_eps_um} um): {len(candidates)}"
    )

    if not candidates:
        print("  no region candidates; nothing saved")
        return []

    selected = select_spaced_positions(
        candidates, max_pos_cand_num, min_spacing_um, xy_um, z_um
    )
    print(f"  selected {len(selected)} spaced region(s) (min_spacing={min_spacing_um} um)")

    selected_zyx = [c["zyx"].tolist() for c in selected]
    save_pix_pos_from_click_list(selected_zyx, csv_savepath=str(pos_pix_csv))
    zyx_um_dict = get_abs_um_pos_from_center_3d(iminfo.statedict, selected_zyx)
    save_um_pos_from_click_list(zyx_um_dict, csv_savepath=str(pos_um_csv))

    tiff_path, _ = respan_export_paths(flim_path, channel=2)
    overlay_mip = None
    if tiff_path.is_file():
        import tifffile as tf

        from respan_mushroom_core import respan_run_dir  # noqa: E402

        label_path = (
            respan_run_dir(flim_path)
            / "Validation_Data"
            / "Segmentation_Labels"
            / tiff_path.name
        )
        if label_path.is_file():
            raw = tf.imread(tiff_path)
            labels = tf.imread(label_path)
            overlay_mip = build_class_overlay_mip(raw, labels)

    if overlay_mip is not None:
        overview_path = export_path / "respan_spaced_regions_overview.png"
        save_region_overview_png(overlay_mip, candidates, selected, overview_path)
        print(f"  saved overview: {overview_path}")

    scores_path = export_path / "respan_spaced_region_scores.csv"
    save_region_scores_csv(candidates, selected, scores_path)
    print(f"  saved scores: {scores_path}")

    save_image_with_assigned_pos_3d(
        tif_path="",
        pix_pos_csv_path=str(pos_pix_csv),
        png_savefolder=str(export_path),
        input_arr=True,
        array=zyx_array,
    )

    if export_panels:
        panels_dir = export_path / "region_panels"
        export_region_panels(
            zyx_array,
            selected,
            panels_dir,
            xy_um=xy_um,
            z_half_window=z_half_window,
            crop_padding_um=crop_padding_um,
            min_crop_um=min_crop_um,
        )

    print(f"  saved {len(selected_zyx)} positions -> {pos_pix_csv}")
    for cand in selected:
        z, y, x = cand["zyx"]
        print(
            f"    cluster {cand['cluster_id']}: zyx=({z},{y},{x}), "
            f"n_spines={cand['n_spines']}, "
            f"mean_shaft_to_head={cand.get('mean_shaft_to_head_um', float('nan')):.2f} um"
        )
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick spaced shaft regions from RESPAN spine detections (low-mag)."
    )
    parser.add_argument("--flim", default=DEFAULT_FLIM)
    parser.add_argument("--max-pos", type=int, default=12)
    parser.add_argument("--min-spacing-um", type=float, default=15.0)
    parser.add_argument("--cluster-eps-um", type=float, default=6.0)
    parser.add_argument("--min-spines-per-region", type=int, default=2)
    parser.add_argument("--max-shaft-to-head-um", type=float, default=5.0)
    parser.add_argument("--z-half-window", type=int, default=3)
    parser.add_argument("--crop-padding-um", type=float, default=8.0)
    parser.add_argument("--min-crop-um", type=float, default=28.0)
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()
    pick_spaced_regions_from_respan(
        args.flim,
        max_pos_cand_num=args.max_pos,
        min_spacing_um=args.min_spacing_um,
        cluster_eps_um=args.cluster_eps_um,
        min_spines_per_region=args.min_spines_per_region,
        max_shaft_to_head_um=args.max_shaft_to_head_um,
        skip_if_defined=not args.force,
        export_panels=not args.no_panels,
        z_half_window=args.z_half_window,
        crop_padding_um=args.crop_padding_um,
        min_crop_um=args.min_crop_um,
    )
    print(f"\n=========== done in {time.perf_counter() - t0:.2f} s ============")


if __name__ == "__main__":
    main()
