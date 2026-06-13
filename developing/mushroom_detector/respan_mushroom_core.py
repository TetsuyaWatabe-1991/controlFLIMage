# -*- coding: utf-8 -*-
"""Filter mushroom spines from RESPAN outputs and save per-spine markers."""

from __future__ import annotations

import configparser
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from skimage.draw import polygon as draw_polygon
from skimage.measure import find_contours, regionprops

_CONTROLFLIMAGE = Path(__file__).resolve().parents[2]
_RESPAN_ROOT = _CONTROLFLIMAGE.parent / "ongoing" / "RESPAN"
for _path in (str(_CONTROLFLIMAGE), str(_RESPAN_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scipy import ndimage  # noqa: E402

# Legacy one-sided defaults (used when use_bandpass=False).
MIN_SHAFT_TO_HEAD_UM = 0.5
MAX_HEAD_TO_DEND_UM = 1.1
MIN_HEAD_VOL_UM3 = 0.25
MAX_HEAD_VOL_UM3 = 1.00
# Keep one mushroom per XY cluster; 1-2 um spacing target for uncaging positions.
DEDUPE_MUSHROOM_XY_SEP_UM = 1.0
SPINE_LABEL_VALUE = 1
DENDRITE_LABEL_VALUE = 2
CONTOUR_Z_HALF_WINDOW = 2
SHAFT_Z_HALF_WINDOW = 2
SHAFT_FIT_RADIUS_UM = 3.0
MIN_SHAFT_FIT_PIXELS_IN_RADIUS = 2
SHAFT_MORPH_OPEN_RADIUS_UM = 0.2
SHAFT_MORPH_CLOSE_RADIUS_UM = 0.5
COLOR_SHAFT = np.array([0.0, 1.0, 0.0], dtype=np.float32)
COLOR_SPINE_CYAN = np.array([0.0, 1.0, 1.0], dtype=np.float32)
COLOR_UNCAGING = np.array([1.0, 0.0, 0.0], dtype=np.float32)
COLOR_CLASS_SPINE = np.array([1.0, 0.2, 0.2], dtype=np.float32)
COLOR_CLASS_DENDRITE = np.array([0.0, 1.0, 0.0], dtype=np.float32)
SHAFT_ROI_RECT_LENGTH_PX = 10
SHAFT_ROI_RECT_HEIGHT_PX = 2
OVERLAY_ALPHA = 0.55
UNCAGING_WALK_MAX_STEPS = 100
UNCAGING_WALK_STEP_PX = 0.1
UNCAGING_SPINE_INTENSITY_FRAC = 0.5
UNCAGING_MAX_DIST_EXTRA_UM = 2.0
SPINE_OUTLINE_DILATION_PX = 4
SHAFT_OUTLINE_DILATION_PX = 2
SEG_MASK_SUBFOLDER = "seg_masks"
EDGE_EXCLUDE_PERCENT = 10.0
BG_INTENSITY_PERCENTILE = 50.0
BG_MEDIAN_FILTER_SIZE = 3
BG_SEARCH_RADIUS_SCALE = 2.0
BG_MASK_RADIUS_SCALE = 1.0
BG_EDGE_EXCLUDE_PERCENT = 15.0
BG_EXCLUSION_RADIUS_SCALE = 2.0
MIN_BG_CIRCLE_RADIUS_PX = 1.0
BG_RADIUS_SHRINK_STEP_PX = 1.0
MUSHROOM_ASSIGN_SUMMARY_FILENAME = "mushroom_spine_assign_summary.csv"
# RESPAN Validation_Vols channel order (Z, C, Y, X).
VOL_CH_NEURON = 0
VOL_CH_SPINES_BINARY = 1
VOL_CH_SPINES_FILTERED = 2
VOL_CH_CONNECTED_NECKS = 3
VOL_CH_LABELED_DENDRITES = 4
VOL_CH_SKELETON = 5


def base_name_from_flim_path(flim_path: str) -> str:
    return os.path.basename(flim_path[:-9])


def savefolder_from_flim_path(flim_path: str) -> str:
    """Same convention as multi_spine / DeepD3 mushroom scripts."""
    legacy = flim_path[:-9]
    if os.path.isdir(legacy):
        return legacy
    try:
        os.makedirs(legacy, exist_ok=True)
        if os.path.isdir(legacy):
            return legacy
    except OSError:
        pass
    stem = os.path.splitext(flim_path)[0]
    os.makedirs(stem, exist_ok=True)
    return stem


def clear_existing_mushroom_spine_outputs(savefolder: str | Path, base_name: str) -> int:
    """Remove prior per-spine ini/png/seg-mask outputs before re-detection."""
    savefolder = Path(savefolder)
    removed = 0
    legacy_pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{3}})$")
    id_pattern = re.compile(rf"^{re.escape(base_name)}_id(\d+)$")

    for ini_path in savefolder.glob(f"{base_name}_*.ini"):
        stem = ini_path.stem
        if not (legacy_pattern.match(stem) or id_pattern.match(stem)):
            continue
        targets = [
            ini_path,
            savefolder / f"{stem}.png",
        ]
        seg_dir = savefolder / SEG_MASK_SUBFOLDER
        if seg_dir.is_dir():
            targets.extend(seg_dir.glob(f"{stem}_*.tif"))
        for target in targets:
            if target.is_file():
                target.unlink()
                removed += 1
    return removed


def ini_stem_for_respan_spine_id(base_name: str, respan_spine_id: int) -> str:
    """Canonical stem tied to RESPAN instance id, e.g. pos2__highmag_1_id014."""
    return f"{base_name}_id{int(respan_spine_id):03d}"


def parse_respan_spine_id_from_stem(stem: str) -> int | None:
    """Parse RESPAN id from canonical stem suffix _idNNN."""
    match = re.search(r"_id(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None


def remove_spine_stem_outputs(savefolder: str | Path, stem: str) -> int:
    """Delete ini/png/seg files for one spine stem (used when renaming to id-based stem)."""
    savefolder = Path(savefolder)
    removed = 0
    targets = [
        savefolder / f"{stem}.ini",
        savefolder / f"{stem}.png",
    ]
    seg_dir = savefolder / SEG_MASK_SUBFOLDER
    if seg_dir.is_dir():
        targets.extend(seg_dir.glob(f"{stem}_*.tif"))
    for target in targets:
        if target.is_file():
            target.unlink()
            removed += 1
    return removed


def get_next_ini_indices(savefolder: str, base_name: str, count: int) -> list[int]:
    """Return consecutive 000-based indices (call clear_existing_mushroom_spine_outputs first)."""
    return list(range(count))


def save_spine_dend_info(
    spine_zyx: list[int],
    dend_slope: float,
    dend_intercept: float,
    inipath: str | Path,
    excluded: int = 0,
) -> None:
    config = configparser.ConfigParser()
    config["uncaging_settings"] = {
        "spine_z": str(spine_zyx[0]),
        "spine_y": str(spine_zyx[1]),
        "spine_x": str(spine_zyx[2]),
        "dend_slope": str(dend_slope),
        "dend_intercept": str(dend_intercept),
        "excluded": str(excluded),
    }
    with open(inipath, "w", encoding="utf-8") as configfile:
        config.write(configfile)


def _flim_export_stem(flim_path: Path, channel: int) -> str:
    return f"{flim_path.stem}_ch{channel}_zyx"


def respan_run_dir(flim_path: Path) -> Path:
    return flim_path.parent / "respan_runs" / flim_path.stem


def respan_export_paths(flim_path: Path, channel: int = 2) -> tuple[Path, Path]:
    export_dir = flim_path.parent / "deepd3_annotation_stacks"
    stem = _flim_export_stem(flim_path, channel)
    return export_dir / f"{stem}.tif", export_dir / f"{stem}.json"


def validation_vols_path(run_dir: Path, tiff_stem: str) -> Path:
    return run_dir / "Validation_Data" / "Validation_Vols" / f"{tiff_stem}.tif"


def load_respan_volumes(run_dir: Path, tiff_stem: str) -> dict[str, np.ndarray] | None:
    """Load per-spine instance volumes saved by RESPAN (requires save_intermediate_data)."""
    vol_path = validation_vols_path(run_dir, tiff_stem)
    if not vol_path.exists():
        return None
    vol = tf.imread(vol_path)
    if vol.ndim != 4:
        raise ValueError(f"Unexpected Validation_Vols shape {vol.shape} in {vol_path}")
    return {
        "neuron": vol[:, VOL_CH_NEURON],
        "spines_binary": vol[:, VOL_CH_SPINES_BINARY],
        "spines_filtered": vol[:, VOL_CH_SPINES_FILTERED],
        "connected_necks": vol[:, VOL_CH_CONNECTED_NECKS],
        "labeled_dendrites": vol[:, VOL_CH_LABELED_DENDRITES],
        "skeleton": vol[:, VOL_CH_SKELETON],
    }


def respan_outputs_ready(flim_path: Path, channel: int = 2) -> bool:
    """True when RESPAN segmentation export exists (CSV may be absent if no spines)."""
    run_dir = respan_run_dir(flim_path)
    tiff_path, json_path = respan_export_paths(flim_path, channel)
    label_path = run_dir / "Validation_Data" / "Segmentation_Labels" / tiff_path.name
    return (
        tiff_path.exists()
        and json_path.exists()
        and label_path.exists()
    )


def ensure_respan_analysis(
    flim_path: Path,
    *,
    channel: int = 2,
    rerun: bool = False,
    nnunet_fold: str | int = "all",
) -> tuple[Path, Path]:
    """Run RESPAN from FLIM if outputs are missing."""
    from run_batch_stacks import _resolve_nnunet_fold  # noqa: E402
    from run_from_flim import run_from_flim  # noqa: E402

    resolved_fold = _resolve_nnunet_fold(nnunet_fold)
    if str(nnunet_fold) != resolved_fold:
        print(
            f"  nnUNet fold {nnunet_fold} unavailable; "
            f"using fold_{resolved_fold} checkpoint"
        )
    print(f"  nnUNet fold: {resolved_fold}")

    tiff_path, json_path = respan_export_paths(flim_path, channel)
    if rerun or not respan_outputs_ready(flim_path, channel):
        run_from_flim(
            flim_path,
            channel=channel,
            rerun=rerun,
            skip_overlays=True,
            nnunet_fold=nnunet_fold,
        )
    if not tiff_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"RESPAN export missing for {flim_path}")
    return tiff_path, json_path


def load_pixel_um(json_path: Path) -> tuple[float, float, float]:
    with json_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    xy_um = (float(meta["x_pixel_um"]) + float(meta["y_pixel_um"])) / 2.0
    z_um = float(meta["z_pixel_um"])
    return xy_um, xy_um, z_um


def load_spine_rows(csv_path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        if missing_ok:
            return []
        raise FileNotFoundError(f"detected_spines CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def centroid_in_edge_band(
    x_pix: float,
    y_pix: float,
    image_shape_yx: tuple[int, int],
    edge_percent: float = EDGE_EXCLUDE_PERCENT,
) -> bool:
    """True when the centroid lies in the outer edge_percent margin of the image."""
    height, width = image_shape_yx
    margin_x = width * edge_percent / 100.0
    margin_y = height * edge_percent / 100.0
    return (
        x_pix < margin_x
        or x_pix > (width - 1) - margin_x
        or y_pix < margin_y
        or y_pix > (height - 1) - margin_y
    )


def filter_mushroom_spines_respan(
    rows: list[dict[str, Any]],
    *,
    use_bandpass: bool = True,
    min_shaft_to_head_um: float = MIN_SHAFT_TO_HEAD_UM,
    max_head_to_dend_um: float = MAX_HEAD_TO_DEND_UM,
    min_head_vol_um3: float = MIN_HEAD_VOL_UM3,
    max_head_vol_um3: float = MAX_HEAD_VOL_UM3,
    image_shape_yx: tuple[int, int] | None = None,
    edge_exclude_percent: float = EDGE_EXCLUDE_PERCENT,
) -> dict[int, dict[str, Any]]:
    """Return mushroom candidates keyed by RESPAN spine_id."""
    mushrooms: dict[int, dict[str, Any]] = {}
    for row in rows:
        spine_id = int(float(row["spine_id"]))
        head_x = float(row["x"])
        head_y = float(row["y"])
        head_to_dend_um = float(row["head_euclidean_dist_to_dend"])
        head_vol_um3 = float(row["head_vol"])
        if image_shape_yx is not None and centroid_in_edge_band(
            head_x, head_y, image_shape_yx, edge_exclude_percent
        ):
            print(
                f"  skip spine {spine_id}: centroid ({head_x:.1f}, {head_y:.1f}) "
                f"in outer {edge_exclude_percent:g}% image margin"
            )
            continue

        if not use_bandpass:
            if head_to_dend_um < min_shaft_to_head_um:
                print(
                    f"  skip spine {spine_id}: head-to-dendrite {head_to_dend_um:.3f} um "
                    f"(< {min_shaft_to_head_um} um)"
                )
                continue
            if max_head_to_dend_um > 0 and head_to_dend_um > max_head_to_dend_um:
                print(
                    f"  skip spine {spine_id}: head-to-dendrite {head_to_dend_um:.3f} um "
                    f"(> {max_head_to_dend_um} um)"
                )
                continue
            if head_vol_um3 < min_head_vol_um3:
                print(
                    f"  skip spine {spine_id}: head volume {head_vol_um3:.3f} um^3 "
                    f"(< {min_head_vol_um3} um^3)"
                )
                continue
            if max_head_vol_um3 > 0 and head_vol_um3 > max_head_vol_um3:
                print(
                    f"  skip spine {spine_id}: head volume {head_vol_um3:.3f} um^3 "
                    f"(> {max_head_vol_um3} um^3)"
                )
                continue

        mushrooms[spine_id] = {
            "row": row,
            "shaft_to_head_um": head_to_dend_um,
            "head_vol_um3": head_vol_um3,
        }
    return mushrooms


def filter_mushrooms_by_shaft_to_head_bandpass(
    mushrooms: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Drop mushrooms whose nnU-Net shaft-to-head distance lies outside the band."""
    from mushroom_bandpass import passes_shaft_to_head_bandpass

    kept: dict[int, dict[str, Any]] = {}
    for spine_id, info in mushrooms.items():
        geom = info.get("geometry")
        if geom is None:
            print(f"  skip spine {spine_id}: missing geometry for shaft-to-head bandpass")
            continue
        shaft_to_head_um = float(geom["shaft_to_head_um"])
        passed, reason = passes_shaft_to_head_bandpass(shaft_to_head_um)
        if not passed:
            print(f"  skip spine {spine_id}: {reason}")
            continue
        kept[spine_id] = info
    return kept


def filter_mushrooms_by_seg_area_bandpass(
    mushrooms: dict[int, dict[str, Any]],
    xy_pixel_um: float,
) -> dict[int, dict[str, Any]]:
    """Drop mushrooms whose 2D seg mask area lies outside the acceptable band."""
    from mushroom_bandpass import passes_seg_area_bandpass

    kept: dict[int, dict[str, Any]] = {}
    for spine_id, info in mushrooms.items():
        geom = info.get("geometry")
        if geom is None:
            print(f"  skip spine {spine_id}: missing geometry for seg-area bandpass")
            continue
        seg_metrics = _regionprops_metrics(geom["spine_mask_2d"], xy_pixel_um, prefix="seg")
        seg_area_um2 = seg_metrics.get("seg_area_um2")
        if seg_area_um2 is None:
            print(f"  skip spine {spine_id}: missing seg area")
            continue
        passed, reason = passes_seg_area_bandpass(float(seg_area_um2))
        if not passed:
            print(f"  skip spine {spine_id}: {reason}")
            continue
        kept[spine_id] = info
    return kept


def dedupe_mushrooms_by_xy_respan(
    mushrooms: dict[int, dict[str, Any]],
    xy_pixel_um: float,
    min_sep_um: float,
    *,
    all_rows: list[dict[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Drop mushrooms whose head lies within min_sep_um of any RESPAN candidate.

    Separation is measured against all pre-filter CSV rows (all_rows), not only
    other mushroom-filter survivors.
    """
    if min_sep_um <= 0 or not mushrooms:
        return mushrooms

    reference_rows = all_rows if all_rows is not None else [
        info["row"] for info in mushrooms.values()
    ]
    reference_xy_um = np.array(
        [
            [float(row["x"]) * xy_pixel_um, float(row["y"]) * xy_pixel_um]
            for row in reference_rows
        ],
        dtype=np.float64,
    )
    reference_ids = [int(float(row["spine_id"])) for row in reference_rows]

    kept: dict[int, dict[str, Any]] = {}
    for spine_id, info in mushrooms.items():
        row = info["row"]
        head_xy_um = np.array(
            [float(row["x"]) * xy_pixel_um, float(row["y"]) * xy_pixel_um],
            dtype=np.float64,
        )
        other_mask = np.array(
            [ref_id != spine_id for ref_id in reference_ids], dtype=bool
        )
        if not other_mask.any():
            kept[spine_id] = info
            continue

        dists_um = np.hypot(
            reference_xy_um[other_mask, 0] - head_xy_um[0],
            reference_xy_um[other_mask, 1] - head_xy_um[1],
        )
        nearest_um = float(np.min(dists_um))
        if nearest_um < min_sep_um:
            nearest_idx = int(np.argmin(dists_um))
            other_ids = [ref_id for ref_id, keep in zip(reference_ids, other_mask) if keep]
            nearest_id = other_ids[nearest_idx]
            print(
                f"  dedupe skip spine {spine_id}: nearest RESPAN candidate "
                f"{nearest_id} is {nearest_um:.3f} um away (< {min_sep_um} um)"
            )
            continue
        kept[spine_id] = info
    return kept


RATING_Z_HALF_WINDOW = 3


def filter_mushroom_rows_by_z_xy_proximity(
    rows: list[dict[str, Any]],
    *,
    min_xy_sep_um: float = DEDUPE_MUSHROOM_XY_SEP_UM,
    z_half_window: int = RATING_Z_HALF_WINDOW,
    group_key: str = "base_name",
) -> list[dict[str, Any]]:
    """Drop rows whose head lies within min_xy_sep_um of another row in Z±z_half_window."""
    if min_xy_sep_um <= 0 or not rows:
        return list(rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_key, ""))
        grouped.setdefault(key, []).append(row)

    kept_rows: list[dict[str, Any]] = []
    dropped = 0
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=lambda row: int(row.get("spine_index", 0)))
        kept_group: list[dict[str, Any]] = []
        for row in ordered:
            try:
                z_pix = float(row["head_z_pix"])
                x_pix = float(row["head_x_pix"])
                y_pix = float(row["head_y_pix"])
                xy_um = float(row["xy_pixel_um"])
            except (KeyError, TypeError, ValueError):
                kept_group.append(row)
                continue

            conflict = False
            for kept in kept_group:
                kz = float(kept["head_z_pix"])
                if abs(z_pix - kz) > z_half_window:
                    continue
                kx = float(kept["head_x_pix"])
                ky = float(kept["head_y_pix"])
                kxy = float(kept["xy_pixel_um"])
                xy_sep_um = math.hypot(
                    (x_pix - kx) * xy_um,
                    (y_pix - ky) * kxy,
                )
                if xy_sep_um <= min_xy_sep_um:
                    dropped += 1
                    conflict = True
                    break
            if not conflict:
                kept_group.append(row)
        kept_rows.extend(kept_group)

    if dropped:
        print(
            f"  rating filter: {len(rows)} -> {len(kept_rows)} "
            f"(dropped {dropped} with XY<={min_xy_sep_um} um within Z±{z_half_window})"
        )
    return kept_rows


def filter_mushrooms_by_shaft_fit_radius(
    mushrooms: dict[int, dict[str, Any]],
    xy_pixel_um: float,
    *,
    radius_um: float = SHAFT_FIT_RADIUS_UM,
    min_pixels: int = MIN_SHAFT_FIT_PIXELS_IN_RADIUS,
) -> dict[int, dict[str, Any]]:
    """Drop mushrooms with too few dendrite pixels within radius_um of the shaft anchor."""
    kept: dict[int, dict[str, Any]] = {}
    for spine_id, info in mushrooms.items():
        geom = info.get("geometry")
        if geom is None:
            print(f"  skip spine {spine_id}: missing geometry for shaft-fit filter")
            continue

        shaft_y, shaft_x = geom["shaft_anchor_yx"]
        pixel_count = shaft_pixel_count_within_radius_2d(
            geom["shaft_mip_binary"],
            shaft_x,
            shaft_y,
            xy_pixel_um,
            radius_um=radius_um,
        )
        if pixel_count < min_pixels:
            print(
                f"  skip spine {spine_id}: {pixel_count} shaft pixel(s) within "
                f"{radius_um:g} um of anchor (< {min_pixels})"
            )
            continue
        kept[spine_id] = info
    return kept


def calc_point_on_line_close_to_xy(
    x: float,
    y: float,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
    """Foot of perpendicular from (x, y) onto y = slope*x + intercept."""
    x_c = (x + slope * (y - intercept)) / (slope**2 + 1)
    y_c = slope * x_c + intercept
    return float(y_c), float(x_c)


def shaft_to_head_xy_um(
    head_x: float,
    head_y: float,
    dend_slope: float,
    dend_intercept: float,
    xy_pixel_um: float,
) -> tuple[float, float, float]:
    """Return shaft anchor (y, x) and XY shaft-to-head distance in um."""
    shaft_y, shaft_x = calc_point_on_line_close_to_xy(
        head_x, head_y, dend_slope, dend_intercept
    )
    dist_um = float(
        np.hypot((head_x - shaft_x) * xy_pixel_um, (head_y - shaft_y) * xy_pixel_um)
    )
    return dist_um, shaft_y, shaft_x


def _shaft_rectangle_mask_2d(
    head_x: float,
    head_y: float,
    dend_slope: float,
    dend_intercept: float,
    image_shape: tuple[int, int],
    rect_length: float = SHAFT_ROI_RECT_LENGTH_PX,
    rect_height: float = SHAFT_ROI_RECT_HEIGHT_PX,
) -> tuple[np.ndarray, float, float]:
    """Dendritic-shaft ROI rectangle on the fitted dendrite line."""
    shaft_y, shaft_x = calc_point_on_line_close_to_xy(
        head_x, head_y, dend_slope, dend_intercept
    )
    theta = np.arctan(dend_slope)
    dx = (rect_length / 2) * np.cos(theta)
    dy = (rect_length / 2) * np.sin(theta)
    px = (rect_height / 2) * -np.sin(theta)
    py = (rect_height / 2) * np.cos(theta)
    corners_x = [shaft_x - dx - px, shaft_x - dx + px, shaft_x + dx + px, shaft_x + dx - px]
    corners_y = [shaft_y - dy - py, shaft_y - dy + py, shaft_y + dy + py, shaft_y - dy - py]
    rr_rect, cc_rect = draw_polygon(corners_y, corners_x, shape=image_shape)
    mask = np.zeros(image_shape, dtype=bool)
    mask[rr_rect, cc_rect] = True
    return mask, shaft_y, shaft_x


def _regionprops_metrics(mask_2d: np.ndarray, xy_pixel_um: float, prefix: str = "seg") -> dict:
    if not mask_2d.any():
        return {}
    props = regionprops(mask_2d.astype(np.uint8))
    if not props:
        return {}
    prop = props[0]
    xy2 = xy_pixel_um**2
    metrics = {
        f"{prefix}_area_um2": float(prop.area * xy2),
        f"{prefix}_perimeter_um": float(prop.perimeter * xy_pixel_um),
    }
    for attr, key in (
        ("major_axis_length", f"{prefix}_major_axis_um"),
        ("minor_axis_length", f"{prefix}_minor_axis_um"),
        ("equivalent_diameter_area", f"{prefix}_equivalent_diameter_um"),
    ):
        try:
            metrics[key] = float(getattr(prop, attr) * xy_pixel_um)
        except (IndexError, AttributeError, TypeError, ValueError):
            metrics[key] = np.nan
    return metrics


def _morph_disk_radius_px(xy_pixel_um: float, radius_um: float) -> int:
    return max(1, int(round(radius_um / max(xy_pixel_um, 1e-6))))


def _disk_structure(radius_px: int) -> np.ndarray:
    size = 2 * radius_px + 1
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (xx * xx + yy * yy) <= radius_px * radius_px


def seg_mask_save_dir(savefolder: str | Path) -> Path:
    """Subfolder for per-spine segmentation TIFFs (keeps PNG/ini folder clean)."""
    path = Path(savefolder) / SEG_MASK_SUBFOLDER
    path.mkdir(parents=True, exist_ok=True)
    return path


def dilate_mask_2d(mask_2d: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary dilation with a disk structuring element."""
    mask = np.asarray(mask_2d, dtype=bool)
    if radius_px <= 0 or not mask.any():
        return mask
    return ndimage.binary_dilation(mask, structure=_disk_structure(radius_px))


def save_label_z_mip_volume_preview(
    raw_zyx: np.ndarray,
    label_zyx: np.ndarray,
    *,
    xy_pixel_um: float,
    z_pixel_um: float,
    out_path: Path,
    min_dendrite_vol_um: float = 1.0,
) -> Path:
    """Save raw Z-MIP with nnUNet labels and per-CC 3D volume annotations."""
    raw = np.asarray(raw_zyx, dtype=np.float32)
    lab = np.asarray(label_zyx)
    voxel_um3 = float(xy_pixel_um) * float(xy_pixel_um) * float(z_pixel_um)
    min_dend_vox = round(min_dendrite_vol_um / voxel_um3) if voxel_um3 > 0 else 0

    raw_mip = raw.max(axis=0)
    vmax = float(np.percentile(raw_mip, 99.5)) if raw_mip.size else 1.0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(raw_mip, cmap="gray", vmin=0, vmax=vmax)

    class_styles = (
        (SPINE_LABEL_VALUE, "spine", "autumn"),
        (DENDRITE_LABEL_VALUE, "dendrite", "winter"),
    )
    legend_handles: list[Any] = []
    for label_val, class_name, cmap_name in class_styles:
        mask3d = lab == label_val
        if not mask3d.any():
            continue
        cc, n_cc = ndimage.label(mask3d)
        mip = mask3d.max(axis=0)
        shown = np.ma.masked_where(~mip, mip)
        ax.imshow(shown, cmap=cmap_name, alpha=0.85, vmin=label_val, vmax=label_val)
        for cc_id in range(1, n_cc + 1):
            n_vox = int((cc == cc_id).sum())
            vol_um3 = n_vox * voxel_um3
            ys, xs = np.where((cc == cc_id).max(axis=0))
            if len(ys) == 0:
                continue
            cy = float(np.mean(ys))
            cx = float(np.mean(xs))
            passes = ""
            if class_name == "dendrite":
                passes = " OK" if n_vox >= min_dend_vox else " NG"
            ax.text(
                cx,
                cy,
                f"{vol_um3:.2f}\n({n_vox}v){passes}",
                color="white",
                fontsize=7,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.55},
            )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=plt.cm.get_cmap(cmap_name)(0.6),
                lw=4,
                label=class_name,
            )
        )

    ax.set_title(
        f"Label Z-MIP volumes (voxel={voxel_um3:.4f} um^3, "
        f"min dendrite>={min_dendrite_vol_um:g} um^3 / {min_dend_vox} vox)"
    )
    ax.axis("off")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved label Z-MIP volume preview: {out_path}")
    return out_path


def build_spine_outline_mask_2d(
    spine_mask_2d: np.ndarray,
    shaft_mask_2d: np.ndarray,
    *,
    spine_dilation_px: int = SPINE_OUTLINE_DILATION_PX,
    shaft_dilation_px: int | None = None,
) -> np.ndarray:
    """
    Cyan outline region: dilated spine minus overlap with dilated shaft MIP.

    Spine and shaft use independent disk dilation radii (px). Default shaft
    dilation is SHAFT_OUTLINE_DILATION_PX (spine uses SPINE_OUTLINE_DILATION_PX).
    """
    if shaft_dilation_px is None:
        shaft_dilation_px = SHAFT_OUTLINE_DILATION_PX
    spine_dilated = dilate_mask_2d(spine_mask_2d, spine_dilation_px)
    shaft_dilated = dilate_mask_2d(shaft_mask_2d, shaft_dilation_px)
    return spine_dilated & ~shaft_dilated


def shaft_pixel_count_within_radius_2d(
    shaft_mask_2d: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    xy_pixel_um: float,
    *,
    radius_um: float = SHAFT_FIT_RADIUS_UM,
) -> int:
    """Count shaft pixels strictly within radius_um of the anchor (no radius fallback)."""
    mask = np.asarray(shaft_mask_2d, dtype=bool)
    if not mask.any():
        return 0

    ys, xs = np.where(mask)
    dist_um = np.hypot((xs - anchor_x) * xy_pixel_um, (ys - anchor_y) * xy_pixel_um)
    return int((dist_um <= radius_um).sum())


def shaft_mask_within_radius_2d(
    shaft_mask_2d: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    xy_pixel_um: float,
    *,
    radius_um: float = SHAFT_FIT_RADIUS_UM,
) -> np.ndarray:
    """Binarized shaft pixels within radius_um of the anchor (no dilation)."""
    mask = np.asarray(shaft_mask_2d, dtype=bool)
    out = np.zeros(mask.shape, dtype=bool)
    if not mask.any():
        return out

    ys, xs = np.where(mask)
    dist_um = np.hypot((xs - anchor_x) * xy_pixel_um, (ys - anchor_y) * xy_pixel_um)
    nearby = dist_um <= radius_um
    if int(nearby.sum()) < MIN_SHAFT_FIT_PIXELS_IN_RADIUS:
        nearby = dist_um <= radius_um * 1.5
    out[ys[nearby], xs[nearby]] = True
    return out


def save_mask_2d_tiff(mask_2d: np.ndarray, savepath: str | Path) -> None:
    tf.imwrite(
        savepath,
        np.asarray(mask_2d, dtype=np.uint8),
        photometric="minisblack",
    )


def build_low_intensity_mask_2d(
    zyx: np.ndarray,
    *,
    percentile: float = BG_INTENSITY_PERCENTILE,
    median_size: int = BG_MEDIAN_FILTER_SIZE,
) -> np.ndarray:
    """Lower-intensity half of the full-Z max projection after median smoothing."""
    mip = np.asarray(zyx, dtype=np.float64).max(axis=0)
    if median_size >= 3:
        mip = ndimage.median_filter(mip, size=median_size)
    threshold = float(np.percentile(mip.ravel(), percentile))
    return mip <= threshold


def mask_equivalent_circle_radius_px(mask_2d: np.ndarray) -> float:
    """Equivalent circle radius for a binary ROI mask."""
    mask = np.asarray(mask_2d, dtype=bool)
    if not mask.any():
        return 0.0
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return 0.0
    return float(props[0].equivalent_diameter_area) / 2.0


def build_circle_mask_2d(
    image_shape: tuple[int, int],
    center_y: float,
    center_x: float,
    radius_px: float,
) -> np.ndarray:
    """Filled circle mask centered at (center_y, center_x)."""
    mask = np.zeros(image_shape, dtype=bool)
    if radius_px <= 0:
        return mask
    yy, xx = np.ogrid[: image_shape[0], : image_shape[1]]
    mask[(xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius_px**2] = True
    return mask


def circle_fully_inside_mask(
    center_y: float,
    center_x: float,
    radius_px: float,
    allowed_mask: np.ndarray,
) -> bool:
    """True when every pixel in the circle lies inside allowed_mask."""
    if radius_px <= 0:
        return False
    allowed = np.asarray(allowed_mask, dtype=bool)
    circle = build_circle_mask_2d(allowed.shape, center_y, center_x, radius_px)
    return bool(np.all(allowed[circle]))


def mask_centroid_yx(mask_2d: np.ndarray) -> tuple[float, float]:
    """Mask centroid as (y, x)."""
    ys, xs = np.where(np.asarray(mask_2d, dtype=bool))
    if len(xs) == 0:
        return 0.0, 0.0
    return float(np.mean(ys)), float(np.mean(xs))


def build_double_radius_exclusion_zone_2d(
    roi_mask_2d: np.ndarray,
    *,
    radius_scale: float = BG_EXCLUSION_RADIUS_SCALE,
) -> np.ndarray:
    """Exclusion disk: radius_scale x equivalent-circle radius at ROI centroid."""
    mask = np.asarray(roi_mask_2d, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    cy, cx = mask_centroid_yx(mask)
    radius_px = mask_equivalent_circle_radius_px(mask) * radius_scale
    return build_circle_mask_2d(mask.shape, cy, cx, radius_px)


def build_bg_exclusion_mask_2d(
    spine_outline_mask_2d: np.ndarray,
    shaft_roi_mask_2d: np.ndarray,
    *,
    radius_scale: float = BG_EXCLUSION_RADIUS_SCALE,
) -> np.ndarray:
    """Forbidden region around spine outline and shaft ROI (2x equivalent radius each)."""
    exclusion = build_double_radius_exclusion_zone_2d(
        spine_outline_mask_2d, radius_scale=radius_scale
    )
    exclusion |= build_double_radius_exclusion_zone_2d(
        shaft_roi_mask_2d, radius_scale=radius_scale
    )
    return exclusion


def circle_overlaps_mask(
    center_y: float,
    center_x: float,
    radius_px: float,
    forbidden_mask: np.ndarray,
) -> bool:
    """True when a circle shares any pixels with forbidden_mask."""
    if radius_px <= 0:
        return False
    circle = build_circle_mask_2d(
        forbidden_mask.shape, center_y, center_x, radius_px
    )
    return bool((circle & np.asarray(forbidden_mask, dtype=bool)).any())


def build_image_interior_mask_2d(
    image_shape: tuple[int, int],
    edge_exclude_percent: float = BG_EDGE_EXCLUDE_PERCENT,
) -> np.ndarray:
    """True for pixels inside the image, excluding an outer edge_exclude_percent margin."""
    height, width = image_shape
    interior = np.zeros((height, width), dtype=bool)
    margin_y = height * edge_exclude_percent / 100.0
    margin_x = width * edge_exclude_percent / 100.0
    y0 = int(np.ceil(margin_y))
    x0 = int(np.ceil(margin_x))
    y1 = int(np.floor(height - margin_y))
    x1 = int(np.floor(width - margin_x))
    if y0 < y1 and x0 < x1:
        interior[y0:y1, x0:x1] = True
    return interior


def _iter_bg_trial_radii_px(base_radius_px: float) -> list[float]:
    """Descending candidate BG radii from the reference size down to the minimum."""
    start_radius = max(base_radius_px, MIN_BG_CIRCLE_RADIUS_PX)
    radii: list[float] = []
    radius_px = start_radius
    while radius_px >= MIN_BG_CIRCLE_RADIUS_PX - 1e-6:
        radii.append(float(radius_px))
        radius_px -= BG_RADIUS_SHRINK_STEP_PX
    return radii


def _bg_circle_candidate_offsets(n_directions: int = 8) -> np.ndarray:
    """Unit vectors on a ring (y, x) for approximate BG placement."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)
    return np.stack([np.sin(angles), np.cos(angles)], axis=1)


def _bg_circle_valid_center(
    center_y: float,
    center_x: float,
    required_radius_px: float,
    bg_radius_px: float,
    placement_mask: np.ndarray,
    exclusion_mask: np.ndarray,
) -> bool:
    return circle_fully_inside_mask(
        center_y, center_x, required_radius_px, placement_mask
    ) and not circle_overlaps_mask(center_y, center_x, bg_radius_px, exclusion_mask)


def _find_bg_center_approximate(
    ref_cy: float,
    ref_cx: float,
    base_radius_px: float,
    placement_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    image_shape: tuple[int, int],
    *,
    search_radius_scale: float,
    mask_radius_scale: float,
) -> tuple[tuple[float, float] | None, float]:
    """
    Pick the first valid BG circle near the spine/shaft reference point.

    Tries a few offset directions instead of scanning every image pixel.
    """
    offsets = _bg_circle_candidate_offsets()
    height, width = image_shape
    for trial_radius_px in _iter_bg_trial_radii_px(base_radius_px):
        mask_radius_px = trial_radius_px * mask_radius_scale
        search_radius_px = max(trial_radius_px * search_radius_scale, mask_radius_px * 1.5)
        for scale in (1.0, 1.5, 2.0, 2.5):
            offset_px = search_radius_px * scale
            for dy, dx in offsets:
                cy = ref_cy + dy * offset_px
                cx = ref_cx + dx * offset_px
                if not (0 <= cy < height and 0 <= cx < width):
                    continue
                if _bg_circle_valid_center(
                    cy,
                    cx,
                    mask_radius_px,
                    mask_radius_px,
                    placement_mask,
                    exclusion_mask,
                ):
                    return (float(cy), float(cx)), mask_radius_px
    return None, base_radius_px * mask_radius_scale


def _find_bg_center_exhaustive(
    ref_ys: np.ndarray,
    ref_xs: np.ndarray,
    base_radius_px: float,
    placement_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    image_shape: tuple[int, int],
    *,
    search_radius_scale: float,
    mask_radius_scale: float,
) -> tuple[tuple[float, float] | None, float]:
    """Legacy full-image search: nearest valid BG circle to spine/shaft pixels."""
    height, width = image_shape

    def _find_nearest_valid_center(
        required_radius_px: float,
        *,
        bg_radius_px: float,
    ) -> tuple[tuple[float, float] | None, float]:
        best: tuple[float, float] | None = None
        best_dist = np.inf
        for cy in range(height):
            for cx in range(width):
                if not _bg_circle_valid_center(
                    cy, cx, required_radius_px, bg_radius_px, placement_mask, exclusion_mask
                ):
                    continue
                dist = float(np.min(np.hypot(ref_xs - cx, ref_ys - cy)))
                if dist < best_dist:
                    best_dist = dist
                    best = (float(cy), float(cx))
        return best, best_dist

    for trial_radius_px in _iter_bg_trial_radii_px(base_radius_px):
        mask_radius_px = trial_radius_px * mask_radius_scale
        search_radius_px = trial_radius_px * search_radius_scale
        best_center, _ = _find_nearest_valid_center(
            search_radius_px, bg_radius_px=mask_radius_px
        )
        if best_center is None and search_radius_px > mask_radius_px:
            best_center, _ = _find_nearest_valid_center(
                mask_radius_px, bg_radius_px=mask_radius_px
            )
        if best_center is not None:
            return best_center, mask_radius_px
    return None, base_radius_px * mask_radius_scale


def compute_background_circle_roi(
    low_int_mask_2d: np.ndarray,
    spine_outline_mask_2d: np.ndarray,
    shaft_fit_radius_mask_2d: np.ndarray,
    shaft_rect_mask: np.ndarray,
    *,
    search_radius_scale: float = BG_SEARCH_RADIUS_SCALE,
    mask_radius_scale: float = BG_MASK_RADIUS_SCALE,
    edge_exclude_percent: float = BG_EDGE_EXCLUDE_PERCENT,
    exclusion_radius_scale: float = BG_EXCLUSION_RADIUS_SCALE,
    approximate: bool = True,
) -> tuple[np.ndarray, tuple[float, float] | None, float]:
    """
    Place a background circle near the spine/shaft ROIs inside the low-int mask.

    Default (approximate=True): try a handful of offsets from the ROI centroid.
    approximate=False: legacy exhaustive nearest-pixel search (slow on large FOVs).
    """
    image_shape = low_int_mask_2d.shape
    spine_radius = mask_equivalent_circle_radius_px(spine_outline_mask_2d)
    shaft_radius = mask_equivalent_circle_radius_px(shaft_fit_radius_mask_2d)
    base_radius_px = max(spine_radius, shaft_radius, MIN_BG_CIRCLE_RADIUS_PX)
    placement_mask = np.asarray(low_int_mask_2d, dtype=bool) & build_image_interior_mask_2d(
        image_shape, edge_exclude_percent
    )
    exclusion_mask = build_bg_exclusion_mask_2d(
        spine_outline_mask_2d,
        shaft_rect_mask,
        radius_scale=exclusion_radius_scale,
    )

    reference_masks = [
        m
        for m in (spine_outline_mask_2d, shaft_fit_radius_mask_2d, shaft_rect_mask)
        if m is not None and np.asarray(m, dtype=bool).any()
    ]
    empty_mask = np.zeros(image_shape, dtype=bool)
    if not reference_masks:
        return empty_mask, None, base_radius_px * mask_radius_scale

    ref_cy, ref_cx = mask_centroid_yx(shaft_rect_mask)
    if not np.asarray(shaft_rect_mask, dtype=bool).any():
        ref_cy, ref_cx = mask_centroid_yx(spine_outline_mask_2d)

    if approximate:
        best_center, mask_radius_px = _find_bg_center_approximate(
            ref_cy,
            ref_cx,
            base_radius_px,
            placement_mask,
            exclusion_mask,
            image_shape,
            search_radius_scale=search_radius_scale,
            mask_radius_scale=mask_radius_scale,
        )
    else:
        ref_ys = np.concatenate([np.where(m)[0] for m in reference_masks]).astype(np.float64)
        ref_xs = np.concatenate([np.where(m)[1] for m in reference_masks]).astype(np.float64)
        best_center, mask_radius_px = _find_bg_center_exhaustive(
            ref_ys,
            ref_xs,
            base_radius_px,
            placement_mask,
            exclusion_mask,
            image_shape,
            search_radius_scale=search_radius_scale,
            mask_radius_scale=mask_radius_scale,
        )

    if best_center is not None:
        bg_mask = build_circle_mask_2d(
            image_shape, best_center[0], best_center[1], mask_radius_px
        )
        return bg_mask, best_center, mask_radius_px

    return empty_mask, None, base_radius_px * mask_radius_scale


def plot_mask_contour(
    ax,
    mask_2d: np.ndarray,
    *,
    color: str,
    linewidth: float = 1.2,
    linestyle: str = "-",
) -> None:
    """Draw mask boundary on a matplotlib axis."""
    if not np.asarray(mask_2d, dtype=bool).any():
        return
    for contour in find_contours(mask_2d.astype(np.float32), 0.5):
        ax.plot(
            contour[:, 1],
            contour[:, 0],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )


def plot_uncaging_marker(ax, x_pos: float, y_pos: float, *, markersize: float = 8) -> None:
    """Yellow X marker for uncaging position."""
    ax.plot(
        x_pos,
        y_pos,
        "x",
        color="yellow",
        markersize=markersize,
        markeredgewidth=1.2,
        markeredgecolor="yellow",
    )


def shaft_mask_mip_near_z(
    class_labels_zyx: np.ndarray,
    z_pix: float,
    *,
    z_half_window: int = SHAFT_Z_HALF_WINDOW,
) -> np.ndarray:
    """Dendrite/shaft MIP from class labels over Z +/- z_half_window."""
    z_idx = int(np.clip(round(z_pix), 0, class_labels_zyx.shape[0] - 1))
    z0, z1 = _z_window_indices(z_idx, class_labels_zyx.shape[0], z_half_window)
    return np.any(class_labels_zyx[z0:z1] == DENDRITE_LABEL_VALUE, axis=0)


def fill_shaft_mask_holes(
    shaft_mask_2d: np.ndarray,
    xy_pixel_um: float,
    *,
    open_radius_um: float = SHAFT_MORPH_OPEN_RADIUS_UM,
    close_radius_um: float = SHAFT_MORPH_CLOSE_RADIUS_UM,
) -> np.ndarray:
    """Smooth shaft mask with opening, then close gaps/holes."""
    mask = np.asarray(shaft_mask_2d, dtype=bool)
    if not mask.any():
        return mask

    open_px = _morph_disk_radius_px(xy_pixel_um, open_radius_um)
    close_px = _morph_disk_radius_px(xy_pixel_um, close_radius_um)
    opened = ndimage.binary_opening(mask, structure=_disk_structure(open_px))
    closed = ndimage.binary_closing(opened, structure=_disk_structure(close_px))
    return closed


def estimate_dendrite_line(
    shaft_mask_2d: np.ndarray,
    anchor_y: float,
    anchor_x: float,
    *,
    xy_pixel_um: float,
    fit_radius_um: float = SHAFT_FIT_RADIUS_UM,
) -> tuple[float, float]:
    """Fit y = slope*x + intercept using shaft pixels within fit_radius_um of anchor."""
    ys, xs = np.where(np.asarray(shaft_mask_2d, dtype=bool))
    if len(xs) < 2:
        return 0.0, float(anchor_y)

    dist_um = np.hypot((xs - anchor_x) * xy_pixel_um, (ys - anchor_y) * xy_pixel_um)
    nearby = dist_um <= fit_radius_um
    if int(nearby.sum()) < MIN_SHAFT_FIT_PIXELS_IN_RADIUS:
        nearby = dist_um <= fit_radius_um * 1.5

    xs_fit = xs[nearby].astype(np.float64)
    ys_fit = ys[nearby].astype(np.float64)
    slope, intercept = np.polyfit(xs_fit, ys_fit, 1)
    return float(slope), float(intercept)


def _normalize_to_float(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    low, high = np.percentile(arr, (1.0, 99.5))
    if high <= low:
        high = low + 1.0
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _z_window_indices(z_idx: int, n_z: int, half: int = SHAFT_Z_HALF_WINDOW) -> tuple[int, int]:
    z0 = max(0, z_idx - half)
    z1 = min(n_z, z_idx + half + 1)
    return z0, z1


def shaft_mask_at_z(class_labels_zyx: np.ndarray, z_idx: int) -> np.ndarray:
    """Dendrite/shaft mask using the target Z plus +/- SHAFT_Z_HALF_WINDOW slices."""
    z0, z1 = _z_window_indices(z_idx, class_labels_zyx.shape[0], SHAFT_Z_HALF_WINDOW)
    return np.any(class_labels_zyx[z0:z1] == DENDRITE_LABEL_VALUE, axis=0)


def mask_window_near_z(mask_zyx: np.ndarray, z_pix: float) -> np.ndarray:
    """Return a 3D mask substack spanning Z +/- SHAFT_Z_HALF_WINDOW around z_pix."""
    z_idx = int(np.clip(round(z_pix), 0, mask_zyx.shape[0] - 1))
    z0, z1 = _z_window_indices(z_idx, mask_zyx.shape[0], SHAFT_Z_HALF_WINDOW)
    return mask_zyx[z0:z1]


def build_class_overlay_slice(
    raw_slice: np.ndarray,
    label_slice: np.ndarray,
    *,
    overlay_alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """Single-Z RGB overlay matching export_z_overlays (red=spine class, green=dendrite)."""
    gray = _normalize_to_float(raw_slice)
    rgb = np.stack([gray, gray, gray], axis=-1)
    for label_value, color in (
        (SPINE_LABEL_VALUE, COLOR_CLASS_SPINE),
        (DENDRITE_LABEL_VALUE, COLOR_CLASS_DENDRITE),
    ):
        mask = label_slice == label_value
        if np.any(mask):
            rgb[mask] = (1.0 - overlay_alpha) * rgb[mask] + overlay_alpha * color
    return np.clip(rgb, 0.0, 1.0)


def build_class_overlay_mip(
    raw_zyx: np.ndarray,
    class_labels_zyx: np.ndarray,
    *,
    overlay_alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """RGB MIP overlay using nnU-Net class labels (same style as export_z_overlays)."""
    gray = _normalize_to_float(raw_zyx.max(axis=0))
    rgb = np.stack([gray, gray, gray], axis=-1)
    spine_mip = np.any(class_labels_zyx == SPINE_LABEL_VALUE, axis=0)
    dend_mip = np.any(class_labels_zyx == DENDRITE_LABEL_VALUE, axis=0)
    rgb[spine_mip] = (1.0 - overlay_alpha) * rgb[spine_mip] + overlay_alpha * COLOR_CLASS_SPINE
    rgb[dend_mip] = (1.0 - overlay_alpha) * rgb[dend_mip] + overlay_alpha * COLOR_CLASS_DENDRITE
    return np.clip(rgb, 0.0, 1.0)


def save_lowmag_class_overlay_pngs(
    raw_zyx: np.ndarray,
    class_labels_zyx: np.ndarray,
    savefolder: str | Path,
    base_name: str,
    *,
    z_slices_subdir: str = "class_overlay_z_slices",
    zproj_subdir: str = "class_overlay_zproj",
    save_z_montage: bool = True,
) -> tuple[Path, Path]:
    """
    Low-mag outputs: per-Z class overlay PNGs plus raw/class Z-projection PNGs.

    Skips per-spine ROI plots; uses plt.imsave for speed.
    """
    savefolder = Path(savefolder)
    z_dir = savefolder / z_slices_subdir
    zproj_dir = savefolder / zproj_subdir
    z_dir.mkdir(parents=True, exist_ok=True)
    zproj_dir.mkdir(parents=True, exist_ok=True)

    n_z = raw_zyx.shape[0]
    slice_rgbs: list[np.ndarray] = []
    for z_idx in range(n_z):
        rgb = build_class_overlay_slice(raw_zyx[z_idx], class_labels_zyx[z_idx])
        out_path = z_dir / f"{base_name}_z{z_idx:03d}_class_overlay.png"
        plt.imsave(out_path, rgb)
        slice_rgbs.append(rgb)

    raw_mip = _normalize_to_float(raw_zyx.max(axis=0))
    overlay_mip = build_class_overlay_mip(raw_zyx, class_labels_zyx)
    raw_zproj_path = zproj_dir / f"{base_name}_raw_zproj.png"
    overlay_zproj_path = zproj_dir / f"{base_name}_class_overlay_zproj.png"
    plt.imsave(raw_zproj_path, raw_mip, cmap="gray")
    plt.imsave(overlay_zproj_path, overlay_mip)

    if save_z_montage and n_z > 1:
        montage_path = z_dir / f"{base_name}_all_z_class_overlay_montage.png"
        fig, axes = plt.subplots(1, n_z, figsize=(2.0 * n_z, 2.2))
        if n_z == 1:
            axes = [axes]
        for z_idx, ax in enumerate(axes):
            ax.imshow(slice_rgbs[z_idx], origin="upper")
            ax.set_title(f"Z{z_idx}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{base_name} class overlay (red=spine, green=dendrite)", fontsize=10)
        fig.savefig(montage_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"  saved Z montage: {montage_path}")

    print(f"  saved {n_z} Z-slice class overlays: {z_dir}")
    print(f"  saved Z projection PNGs: {raw_zproj_path.name}, {overlay_zproj_path.name}")
    return z_dir, zproj_dir


def save_spine_centroids_overview_png(
    overlay_mip_rgb: np.ndarray,
    spines: dict[int, dict[str, Any]],
    savefolder: str | Path,
    base_name: str,
    *,
    overview_title: str | None = None,
) -> None:
    """Overview PNG with RESPAN spine head centroids on the class-overlay Z projection."""
    roi_path = Path(savefolder) / f"{base_name}_respan_spine_centroids.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay_mip_rgb, origin="upper")
    for spine_id, info in spines.items():
        row = info["row"]
        head_x = float(row["x"])
        head_y = float(row["y"])
        ax.plot(head_x, head_y, "o", color="yellow", markersize=4, markeredgecolor="black")
        ax.text(
            head_x + 1.5,
            head_y - 1.5,
            str(spine_id),
            color="yellow",
            fontsize=6,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 0.8, "edgecolor": "none"},
        )
    ax.set_title(overview_title or f"RESPAN spines (n={len(spines)})")
    ax.axis("off")
    fig.savefig(roi_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  saved spine centroid overview: {roi_path}")


def build_minimal_spine_feature_row(
    *,
    flim_path: str,
    base_name: str,
    spine_index: int,
    respan_spine_id: int,
    spine_info: dict[str, Any],
    xy_pixel_um: float,
    z_pixel_um: float,
    dedupe_sep_um: float,
    n_candidates: int,
    n_before_dedupe: int,
    n_after_dedupe: int,
) -> dict[str, Any]:
    """Feature row from RESPAN CSV only (no per-spine geometry or ROI files)."""
    row = spine_info["row"]
    return {
        "flim_path": flim_path,
        "base_name": base_name,
        "spine_index": spine_index,
        "respan_spine_id": respan_spine_id,
        "png_path": "",
        "ini_path": "",
        "spine_mask_tif": "",
        "shaft_mask_tif": "",
        "bg_mask_tif": "",
        "xy_pixel_um": xy_pixel_um,
        "z_pixel_um": z_pixel_um,
        "min_shaft_to_head_um": 0.0,
        "dedupe_mushroom_xy_sep_um": dedupe_sep_um,
        "n_candidates": n_candidates,
        "n_mushroom_before_dedupe": n_before_dedupe,
        "n_mushroom_after_dedupe": n_after_dedupe,
        "head_z_pix": float(row["z"]),
        "head_y_pix": float(row["y"]),
        "head_x_pix": float(row["x"]),
        "uncaging_z_pix": np.nan,
        "uncaging_y_pix": np.nan,
        "uncaging_x_pix": np.nan,
        "shaft_anchor_y_pix": np.nan,
        "shaft_anchor_x_pix": np.nan,
        "shaft_to_head_um": float(spine_info["shaft_to_head_um"]),
        "head_to_uncaging_um": np.nan,
        "head_vol_um3": float(spine_info["head_vol_um3"]),
        "head_area_um2": float(row["head_area"]),
        "spine_type": row.get("spine_type", ""),
        "dend_slope": np.nan,
        "dend_intercept": np.nan,
        "nearest_neighbor_xy_um": float(row.get("nearest_neighbor_xy_um", np.nan))
        if row.get("nearest_neighbor_xy_um")
        else np.nan,
        "nearest_neighbor_respan_id": int(row["nearest_neighbor_spine_id"])
        if row.get("nearest_neighbor_spine_id") not in (None, "")
        else np.nan,
        "respan_head_euclidean_dist_to_dend": float(row["head_euclidean_dist_to_dend"]),
        "bg_center_y_pix": np.nan,
        "bg_center_x_pix": np.nan,
        "bg_radius_px": np.nan,
        "bg_radius_um": np.nan,
        "param_dedupe_mushroom_xy_sep_um": dedupe_sep_um,
        "detection_pipeline": "respan_lowmag",
    }


def build_custom_overlay_slice(
    raw_slice: np.ndarray,
    shaft_mask: np.ndarray,
    spine_mask: np.ndarray,
    *,
    overlay_alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """Single-Z overlay: shaft=green, spine=cyan on grayscale background."""
    gray = _normalize_to_float(raw_slice)
    rgb = np.stack([gray, gray, gray], axis=-1)
    if shaft_mask.any():
        rgb[shaft_mask] = (1.0 - overlay_alpha) * rgb[shaft_mask] + overlay_alpha * COLOR_SHAFT
    if spine_mask.any():
        rgb[spine_mask] = (1.0 - overlay_alpha) * rgb[spine_mask] + overlay_alpha * COLOR_SPINE_CYAN
    return np.clip(rgb, 0.0, 1.0)


def raw_local_z_mip(
    raw_zyx: np.ndarray,
    z_center: float,
    *,
    z_half_window: int = SHAFT_Z_HALF_WINDOW,
) -> tuple[np.ndarray, int, int]:
    """Max-intensity projection over Z center +/- z_half_window slices."""
    z_idx = int(np.clip(round(z_center), 0, raw_zyx.shape[0] - 1))
    z0, z1 = _z_window_indices(z_idx, raw_zyx.shape[0], z_half_window)
    mip = raw_zyx[z0:z1].max(axis=0).astype(np.float32)
    return mip, z0, z1


def build_per_spine_class_overlay_mip(
    raw_mip: np.ndarray,
    class_labels_zyx: np.ndarray,
    z0: int,
    z1: int,
    target_spine_mip: np.ndarray,
    *,
    overlay_alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """
    Local-Z MIP overlay: dendrite=green, other spines=red, target spine=cyan.
    """
    gray = _normalize_to_float(raw_mip)
    rgb = np.stack([gray, gray, gray], axis=-1)
    label_stack = class_labels_zyx[z0:z1]
    shaft_mask = np.any(label_stack == DENDRITE_LABEL_VALUE, axis=0)
    all_spine_mask = np.any(label_stack == SPINE_LABEL_VALUE, axis=0)
    target_mask = np.asarray(target_spine_mip, dtype=bool)
    other_spine_mask = all_spine_mask & ~target_mask

    if shaft_mask.any():
        rgb[shaft_mask] = (1.0 - overlay_alpha) * rgb[shaft_mask] + overlay_alpha * COLOR_CLASS_DENDRITE
    if other_spine_mask.any():
        rgb[other_spine_mask] = (
            (1.0 - overlay_alpha) * rgb[other_spine_mask] + overlay_alpha * COLOR_CLASS_SPINE
        )
    if target_mask.any():
        rgb[target_mask] = (1.0 - overlay_alpha) * rgb[target_mask] + overlay_alpha * COLOR_SPINE_CYAN
    return np.clip(rgb, 0.0, 1.0)


def _uncaging_points_at_z(
    mushrooms: dict[int, dict[str, Any]],
    z_idx: int,
) -> list[tuple[int, float, float]]:
    points: list[tuple[int, float, float]] = []
    for spine_id, info in mushrooms.items():
        unc = info["geometry"]["uncaging_zyx"]
        if int(round(unc[0])) == z_idx:
            points.append((spine_id, float(unc[2]), float(unc[1])))
    return points


def save_z_triplet_overlays(
    raw_zyx: np.ndarray,
    class_labels_zyx: np.ndarray,
    mushrooms: dict[int, dict[str, Any]],
    output_dir: Path,
    stem: str,
    *,
    respan_volumes: dict[str, np.ndarray] | None = None,
    dpi: int = 200,
) -> Path:
    """
    Save one 3-panel PNG per Z:
      raw | class-label overlay | shaft(green)+spine(cyan)+uncaging(red)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_z = raw_zyx.shape[0]

    for z_idx in range(n_z):
        raw_slice = raw_zyx[z_idx]
        label_slice = class_labels_zyx[z_idx]
        class_rgb = build_class_overlay_slice(raw_slice, label_slice)
        shaft_mask = shaft_mask_at_z(class_labels_zyx, z_idx)

        if respan_volumes is not None:
            spine_mask = respan_volumes["spines_filtered"][z_idx] > 0
        else:
            spine_mask = label_slice == SPINE_LABEL_VALUE

        custom_rgb = build_custom_overlay_slice(raw_slice, shaft_mask, spine_mask)
        unc_points = _uncaging_points_at_z(mushrooms, z_idx)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=dpi)
        axes[0].imshow(raw_slice, cmap="gray", origin="upper")
        axes[0].set_title(f"Raw Z={z_idx}", fontsize=9)
        axes[1].imshow(class_rgb, origin="upper")
        axes[1].set_title("Class overlay", fontsize=9)
        axes[2].imshow(custom_rgb, origin="upper")
        axes[2].set_title("Shaft+spine+uncaging", fontsize=9)
        for spine_id, x_pos, y_pos in unc_points:
            plot_uncaging_marker(axes[2], x_pos, y_pos, markersize=6)
            axes[2].text(
                x_pos + 1.2,
                y_pos - 1.2,
                str(spine_id),
                color="yellow",
                fontsize=6,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.5, "edgecolor": "none"},
            )
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"{stem} | Z={z_idx} | L:class overlay | R:green=shaft(Z±{SHAFT_Z_HALF_WINDOW}), "
            f"cyan=spine, yellow X=uncaging",
            fontsize=9,
        )
        out_path = output_dir / f"{stem}_z{z_idx:02d}_triplet.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    montage_path = output_dir / f"{stem}_all_z_triplet_montage.png"
    _save_triplet_montage(raw_zyx, class_labels_zyx, n_z, montage_path, dpi=dpi)
    print(f"  saved {n_z} Z triplet overlays to: {output_dir}")
    print(f"  montage: {montage_path}")
    return output_dir


def _save_triplet_montage(
    raw_zyx: np.ndarray,
    class_labels_zyx: np.ndarray,
    n_z: int,
    montage_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """Horizontal montage of class-overlay panels (middle column of each Z triplet)."""
    fig, axes = plt.subplots(1, n_z, figsize=(2.0 * n_z, 2.5), dpi=dpi)
    if n_z == 1:
        axes = [axes]
    for z_idx, ax in enumerate(axes):
        class_rgb = build_class_overlay_slice(raw_zyx[z_idx], class_labels_zyx[z_idx])
        ax.imshow(class_rgb, origin="upper")
        ax.set_title(f"Z{z_idx}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Class overlay per Z (center panel of each triplet)", fontsize=10)
    fig.savefig(montage_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_mushroom_z_triplet_overlays(
    spine_id: int,
    mushroom_info: dict[str, Any],
    raw_zyx: np.ndarray,
    class_labels_zyx: np.ndarray,
    output_dir: Path,
    stem: str,
    *,
    respan_volumes: dict[str, np.ndarray] | None = None,
    dpi: int = 200,
) -> None:
    """Per-mushroom Z triplets for slices where that spine instance exists."""
    geom = mushroom_info["geometry"]
    spine_mask_3d = geom["spine_mask_3d"]
    z_indices = sorted(int(z) for z in np.unique(np.where(spine_mask_3d)[0]))
    if not z_indices:
        return

    mush_dir = output_dir / f"{stem}_spine{spine_id}_z_triplets"
    mush_dir.mkdir(parents=True, exist_ok=True)

    for z_idx in z_indices:
        raw_slice = raw_zyx[z_idx]
        label_slice = class_labels_zyx[z_idx]
        class_rgb = build_class_overlay_slice(raw_slice, label_slice)
        shaft_mask = shaft_mask_at_z(class_labels_zyx, z_idx)
        spine_mask = spine_mask_3d[z_idx]

        custom_rgb = build_custom_overlay_slice(raw_slice, shaft_mask, spine_mask)
        unc = geom["uncaging_zyx"]

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=dpi)
        axes[0].imshow(raw_slice, cmap="gray", origin="upper")
        axes[0].set_title(f"Raw Z={z_idx}", fontsize=9)
        axes[1].imshow(class_rgb, origin="upper")
        axes[1].set_title("Class overlay", fontsize=9)
        axes[2].imshow(custom_rgb, origin="upper")
        axes[2].set_title("This mushroom", fontsize=9)
        if int(round(unc[0])) == z_idx:
            plot_uncaging_marker(axes[2], unc[2], unc[1], markersize=7)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{stem} spine {spine_id} | Z={z_idx}", fontsize=9)
        fig.savefig(mush_dir / f"{stem}_spine{spine_id}_z{z_idx:02d}_triplet.png", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


def spine_instance_masks(
    respan_volumes: dict[str, np.ndarray],
    spine_id: int,
    z_pix: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-spine 3D/2D masks from RESPAN spines_filtered instance labels."""
    sf = respan_volumes["spines_filtered"]
    mask_3d = sf == spine_id
    z_idx = int(np.clip(round(z_pix), 0, sf.shape[0] - 1))
    mask_2d = sf[z_idx] == spine_id
    return mask_3d, mask_2d


def dendrite_line_mask_zyx(
    respan_volumes: dict[str, np.ndarray] | None,
    class_labels_zyx: np.ndarray,
    dendrite_id: int,
) -> np.ndarray:
    """Pick the best available dendrite geometry for line fitting."""
    if respan_volumes is not None:
        skel = respan_volumes["skeleton"] > 0
        if skel.any():
            return skel
        ld = respan_volumes["labeled_dendrites"] == dendrite_id
        if ld.any():
            return ld
    return class_labels_zyx == DENDRITE_LABEL_VALUE


def _dendrite_line_xy(mip_shape: tuple[int, int], slope: float, intercept: float):
    dend_x, dend_y = [], []
    height, width = mip_shape
    for x in range(1, width - 1):
        y = slope * x + intercept
        if 1 < y < height - 1:
            dend_x.append(x)
            dend_y.append(y)
    return dend_x, dend_y


def _spine_mask_2d(
    labels_zyx: np.ndarray,
    spine_id: int,
    z_pix: float,
    head_y: float,
    head_x: float,
) -> np.ndarray:
    """Build a 2D spine mask near the head using RESPAN class labels."""
    z_idx = int(round(z_pix))
    z0 = max(0, z_idx - CONTOUR_Z_HALF_WINDOW)
    z1 = min(labels_zyx.shape[0], z_idx + CONTOUR_Z_HALF_WINDOW + 1)
    spine_mip = np.any(labels_zyx[z0:z1] == SPINE_LABEL_VALUE, axis=0)

    labeled = spine_mip.astype(np.uint8)
    if not labeled.any():
        return labeled.astype(bool)

    structure = np.ones((3, 3), dtype=np.uint8)
    components, n_comp = ndimage.label(spine_mip, structure=structure)
    if n_comp == 0:
        return spine_mip

    yi = int(np.clip(round(head_y), 0, spine_mip.shape[0] - 1))
    xi = int(np.clip(round(head_x), 0, spine_mip.shape[1] - 1))
    target_label = components[yi, xi]
    if target_label == 0:
        dist = np.hypot(
            np.arange(spine_mip.shape[1])[None, :] - head_x,
            np.arange(spine_mip.shape[0])[:, None] - head_y,
        )
        mask_indices = np.where(spine_mip)
        nearest = int(np.argmin(dist[mask_indices]))
        target_label = components[mask_indices[0][nearest], mask_indices[1][nearest]]
    return components == target_label


def _spine_mask_3d(
    labels_zyx: np.ndarray,
    z_pix: float,
    head_y: float,
    head_x: float,
) -> np.ndarray:
    """Return a 3D binary mask for the spine connected component at the head."""
    z_idx = int(round(z_pix))
    z0 = max(0, z_idx - CONTOUR_Z_HALF_WINDOW)
    z1 = min(labels_zyx.shape[0], z_idx + CONTOUR_Z_HALF_WINDOW + 1)
    sub = labels_zyx[z0:z1] == SPINE_LABEL_VALUE
    if not sub.any():
        return np.zeros_like(labels_zyx, dtype=bool)

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    components, n_comp = ndimage.label(sub, structure=structure)
    if n_comp == 0:
        return np.zeros_like(labels_zyx, dtype=bool)

    local_z = int(np.clip(z_idx - z0, 0, sub.shape[0] - 1))
    yi = int(np.clip(round(head_y), 0, sub.shape[1] - 1))
    xi = int(np.clip(round(head_x), 0, sub.shape[2] - 1))
    target = components[local_z, yi, xi]
    if target == 0:
        mask_idx = np.where(sub)
        nearest = int(
            np.argmin(
                np.hypot(mask_idx[2] - head_x, mask_idx[1] - head_y)
            )
        )
        target = components[mask_idx[0][nearest], mask_idx[1][nearest], mask_idx[2][nearest]]

    full_mask = np.zeros_like(labels_zyx, dtype=bool)
    full_mask[z0:z1] = components == target
    return full_mask


def _shaft_binary_mask_from_labels(
    labels_zyx: np.ndarray,
    shaft_rect_mask: np.ndarray,
    z_pix: int,
    *,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    z0 = max(0, z_pix - CONTOUR_Z_HALF_WINDOW)
    z1 = min(labels_zyx.shape[0], z_pix + CONTOUR_Z_HALF_WINDOW + 1)
    dend_mip = np.any(labels_zyx[z0:z1] == DENDRITE_LABEL_VALUE, axis=0)
    mask = dend_mip & shaft_rect_mask
    if exclude_mask is not None:
        mask &= ~np.asarray(exclude_mask, dtype=bool)
    if not mask.any():
        return mask

    labeled, n_labels = ndimage.label(mask)
    if n_labels <= 1:
        return mask
    best_label = 1 + int(
        np.argmax([np.sum(labeled == i) for i in range(1, n_labels + 1)])
    )
    return labeled == best_label


def _dendrite_orientation_from_slope(dend_slope: float) -> float:
    """Map fitted dendrite slope to the DeepD3-style orientation range."""
    orientation = math.atan(dend_slope)
    if orientation < -math.pi / 2:
        orientation += math.pi
    elif orientation > math.pi / 2:
        orientation -= math.pi
    return orientation


def _uncaging_walk_step_sign(
    head_x: float,
    head_y: float,
    shaft_x: float,
    shaft_y: float,
    orientation: float,
) -> float:
    """Choose walk direction so the path moves away from the shaft anchor."""
    x_moved = head_x - shaft_x
    y_moved = head_y - shaft_y
    x_rotated = x_moved * math.cos(orientation) - y_moved * math.sin(orientation)
    if x_rotated <= 0:
        return UNCAGING_WALK_STEP_PX
    return -UNCAGING_WALK_STEP_PX


def _intensity_spine_bin_2d(
    slab: np.ndarray,
    head_y: float,
    head_x: float,
    spine_mask_2d: np.ndarray | None = None,
    *,
    intensity_frac: float = UNCAGING_SPINE_INTENSITY_FRAC,
) -> np.ndarray:
    """Bright-spine blob used for uncaging walk (DeepD3-style threshold)."""
    height, width = slab.shape
    yi = int(np.clip(round(head_y), 0, height - 1))
    xi = int(np.clip(round(head_x), 0, width - 1))
    ref_intensity = float(slab[yi, xi])
    if ref_intensity > 0:
        return slab > ref_intensity * intensity_frac
    if spine_mask_2d is not None and spine_mask_2d.any():
        return np.asarray(spine_mask_2d, dtype=bool)
    return np.zeros(slab.shape, dtype=bool)


def estimate_uncaging_point(
    zyx: np.ndarray,
    spine_mask_3d: np.ndarray,
    head_z: float,
    head_y: float,
    head_x: float,
    *,
    dend_slope: float,
    shaft_y: float,
    shaft_x: float,
    xy_pixel_um: float,
    spine_mask_2d: np.ndarray | None = None,
    shaft_to_head_um: float | None = None,
    max_dist_extra_um: float = UNCAGING_MAX_DIST_EXTRA_UM,
) -> tuple[float, float, float]:
    """
    Walk away from the shaft along the fitted dendrite axis until leaving the
    bright spine blob (DeepD3-style uncaging placement).
    """
    head_z_f = float(head_z)
    head_y_f = float(head_y)
    head_x_f = float(head_x)

    if shaft_to_head_um is None:
        shaft_to_head_um = float(
            np.hypot((head_x_f - shaft_x) * xy_pixel_um, (head_y_f - shaft_y) * xy_pixel_um)
        )
    max_dist_um = max(float(shaft_to_head_um) + max_dist_extra_um, 2.0)

    z_idx = int(np.clip(round(head_z_f), 0, zyx.shape[0] - 1))
    if spine_mask_3d.any():
        z0, z1 = _z_window_indices(z_idx, spine_mask_3d.shape[0], CONTOUR_Z_HALF_WINDOW)
        z_sums = spine_mask_3d[z0:z1].sum(axis=(1, 2))
        if z_sums.any():
            z_idx = z0 + int(np.argmax(z_sums))

    slab = zyx[z_idx].astype(np.float64)
    mask2d = spine_mask_3d[z_idx] if spine_mask_3d.any() else None
    spine_bin = _intensity_spine_bin_2d(
        slab,
        head_y_f,
        head_x_f,
        spine_mask_2d if spine_mask_2d is not None else mask2d,
    )
    if not spine_bin.any():
        return head_z_f, head_y_f, head_x_f

    orientation = _dendrite_orientation_from_slope(dend_slope)
    step_sign = _uncaging_walk_step_sign(head_x_f, head_y_f, shaft_x, shaft_y, orientation)
    candi_x = head_x_f
    candi_y = head_y_f

    height, width = slab.shape
    min_y, max_y = 1, height - 2
    min_x, max_x = 1, width - 2

    for _ in range(UNCAGING_WALK_MAX_STEPS):
        if not (min_y < candi_y < max_y and min_x < candi_x < max_x):
            break

        dist_um = float(
            np.hypot((shaft_x - candi_x) * xy_pixel_um, (shaft_y - candi_y) * xy_pixel_um)
        )
        if dist_um > max_dist_um:
            break

        yi_i = int(candi_y)
        xi_i = int(candi_x)
        if spine_bin[yi_i, xi_i]:
            candi_x -= math.cos(orientation) * step_sign
            candi_y += math.sin(orientation) * step_sign
        else:
            return float(z_idx), float(candi_y), float(candi_x)

    return head_z_f, head_y_f, head_x_f


def save_spine_segmentation_tiff(mask_3d: np.ndarray, savepath: str | Path) -> None:
    tf.imwrite(
        savepath,
        mask_3d.astype(np.uint8),
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )


def save_mushroom_overlay_png_respan(
    raw_zyx: np.ndarray,
    class_labels_zyx: np.ndarray,
    head_zyx: list[float],
    uncaging_zyx: list[float],
    dend_slope: float,
    dend_intercept: float,
    savepath: str | Path,
    *,
    spine_mask_3d: np.ndarray,
    shaft_rect_mask: np.ndarray | None = None,
    bg_circle_mask: np.ndarray | None = None,
    spine_id: int | None = None,
    auto_rating: int | None = None,
    auto_rating_label: str = "",
    pix_size: int = 512,
    z_half_window: int = SHAFT_Z_HALF_WINDOW,
) -> None:
    """
    Save one 3-panel PNG per mushroom spine:
      L: raw local Z-MIP (head Z +/- z_half_window)
      C: same MIP + class overlay (shaft=green, other spines=red, target=cyan)
      R: same MIP + spine outline (cyan), shaft/BG ROI contours, shaft line, uncaging X
    """
    savepath = Path(savepath)
    raw_mip, z0, z1 = raw_local_z_mip(raw_zyx, head_zyx[0], z_half_window=z_half_window)
    target_spine_mip = np.any(spine_mask_3d[z0:z1], axis=0)
    shaft_mip_binary = shaft_mask_mip_near_z(
        class_labels_zyx, head_zyx[0], z_half_window=z_half_window
    )
    outline_mask = build_spine_outline_mask_2d(target_spine_mip, shaft_mip_binary)
    center_rgb = build_per_spine_class_overlay_mip(
        raw_mip,
        class_labels_zyx,
        z0,
        z1,
        target_spine_mip,
    )

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(raw_mip, cmap="gray", origin="upper")
    axs[0].set_title(f"Raw Z-MIP (Z{z0}-{z1 - 1})", fontsize=9)
    axs[1].imshow(center_rgb, origin="upper")
    axs[1].set_title("Class overlay (target=cyan)", fontsize=9)
    axs[2].imshow(raw_mip, cmap="gray", origin="upper")
    dend_x, dend_y = _dendrite_line_xy(raw_mip.shape, dend_slope, dend_intercept)
    axs[2].plot(dend_x, dend_y, color="lime", linewidth=1.5)
    plot_mask_contour(axs[2], outline_mask, color="cyan", linewidth=1.2)
    if shaft_rect_mask is not None:
        plot_mask_contour(
            axs[2],
            shaft_rect_mask,
            color="lime",
            linewidth=1.4,
            linestyle="--",
        )
    if bg_circle_mask is not None:
        plot_mask_contour(
            axs[2],
            bg_circle_mask,
            color="orange",
            linewidth=1.4,
            linestyle="--",
        )
    plot_uncaging_marker(axs[2], uncaging_zyx[2], uncaging_zyx[1])
    axs[2].set_title("Target + shaft/BG ROI + uncaging", fontsize=9)

    for ax in axs:
        ax.axis("off")

    title = f"spine {spine_id}" if spine_id is not None else "mushroom spine"
    fig.suptitle(
        f"{title} | head Z={int(round(head_zyx[0]))} | "
        f"shaft Z±{z_half_window}, target spine=cyan",
        fontsize=10,
    )

    if auto_rating is not None:
        from mushroom_auto_rating import AUTO_RATING_COLORS, AUTO_RATING_SHORT_LABELS

        banner = AUTO_RATING_SHORT_LABELS.get(auto_rating, f"{auto_rating}")
        color = AUTO_RATING_COLORS.get(auto_rating, "white")
        label_text = auto_rating_label.replace("_", " ") if auto_rating_label else banner
        fig.text(
            0.5,
            0.02,
            f"AUTO {banner}  ({label_text})",
            ha="center",
            va="bottom",
            fontsize=22,
            fontweight="bold",
            color="white",
            bbox={
                "facecolor": color,
                "alpha": 0.92,
                "pad": 8,
                "edgecolor": "white",
                "linewidth": 2,
            },
        )
        for ax in axs:
            ax.text(
                0.02,
                0.98,
                str(auto_rating),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=28,
                fontweight="bold",
                color="white",
                bbox={
                    "facecolor": color,
                    "alpha": 0.85,
                    "pad": 4,
                    "edgecolor": "none",
                },
            )

    fig.savefig(
        savepath,
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=int(pix_size / 4 * 1.3264),
    )
    plt.close(fig)


def save_respan_overview_pngs(
    mip_img: np.ndarray,
    mushrooms: dict[int, dict[str, Any]],
    savefolder: str,
    base_name: str,
    *,
    overview_title: str | None = None,
) -> None:
    """Save overview MIP and ROI PNGs (DeepD3-style filenames)."""
    mip_path = os.path.join(savefolder, f"{base_name}_respan_mip.png")
    roi_path = os.path.join(savefolder, f"{base_name}_respan_roi.png")

    plt.imsave(mip_path, mip_img, cmap="gray")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mip_img, cmap="gray")
    for spine_id, info in mushrooms.items():
        geom = info.get("geometry", {})
        head = geom.get("head_zyx", [info["row"]["z"], info["row"]["y"], info["row"]["x"]])
        unc = geom.get(
            "uncaging_zyx",
            [info["row"]["z"], info["row"]["y"], info["row"]["x"]],
        )
        ax.plot(head[2], head[1], "o", color="red", markersize=5)
        plot_uncaging_marker(ax, unc[2], unc[1], markersize=6)
        ax.text(
            head[2] + 1.5,
            head[1] - 1.5,
            str(spine_id),
            color="yellow",
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.0, "edgecolor": "none"},
        )
    ax.set_title(overview_title or f"RESPAN spines (n={len(mushrooms)})")
    ax.axis("off")
    fig.savefig(roi_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  saved overview MIP: {mip_path}")
    print(f"  saved overview ROI: {roi_path}")


_MUSHROOM_FEATURE_CSV_EMPTY_COLUMNS = (
    "respan_spine_id",
    "ini_path",
    "auto_rating",
    "auto_rating_label",
)


def mark_highmag_batch_processed_empty(
    flim_path: str | Path,
    *,
    reason: str = "no_spines",
) -> str:
    """
    Write header-only feature CSV so batch watch skips this highmag field.

    Used when RESPAN or mushroom filtering yields zero exportable spines.
    """
    savefolder = savefolder_from_flim_path(str(flim_path))
    base_name = base_name_from_flim_path(str(flim_path))
    os.makedirs(savefolder, exist_ok=True)
    out_path = Path(savefolder) / f"{base_name}_respan_mushroom_features.csv"
    if out_path.is_file():
        return str(out_path)
    import pandas as pd

    pd.DataFrame(columns=list(_MUSHROOM_FEATURE_CSV_EMPTY_COLUMNS)).to_csv(
        out_path, index=False, encoding="utf-8"
    )
    print(
        f"  marked batch processed (0 spines, reason={reason}): {out_path}"
    )
    return str(out_path)


def save_mushroom_features_csv(
    savefolder: str | Path,
    base_name: str,
    feature_rows: list[dict[str, Any]],
) -> str:
    if not feature_rows:
        return ""
    import pandas as pd

    out_path = Path(savefolder) / f"{base_name}_respan_mushroom_features.csv"
    pd.DataFrame(feature_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"  saved RESPAN mushroom features CSV ({len(feature_rows)} rows): {out_path}")
    return str(out_path)


def save_mushroom_assign_summary_csv(
    parent_folder: str | Path,
    feature_rows: list[dict[str, Any]],
) -> str:
    """Write folder-level summary CSV for manual 1-4 spine rating (DeepD3-style)."""
    if not feature_rows:
        return ""
    import pandas as pd

    out_path = Path(parent_folder) / MUSHROOM_ASSIGN_SUMMARY_FILENAME
    pd.DataFrame(feature_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(
        f"Summary CSV ({len(feature_rows)} rows): {out_path} "
        f"(rate with rate_mushroom_spines_gui.py)"
    )
    return str(out_path)


def collect_respan_feature_rows_from_folder(parent_folder: str | Path) -> list[dict[str, Any]]:
    """Load all per-FLIM RESPAN feature CSVs under a high-mag folder."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for csv_path in sorted(Path(parent_folder).rglob("*_respan_mushroom_features.csv")):
        rows.extend(pd.read_csv(csv_path).to_dict(orient="records"))
    return rows


def build_feature_row(
    *,
    flim_path: str,
    base_name: str,
    spine_index: int,
    respan_spine_id: int,
    mushroom_info: dict[str, Any],
    geometry: dict[str, Any],
    xy_pixel_um: float,
    z_pixel_um: float,
    min_shaft_to_head_um: float,
    dedupe_sep_um: float,
    n_candidates: int,
    n_mushroom_before_dedupe: int,
    n_mushroom_after_dedupe: int,
    png_path: str = "",
    ini_path: str = "",
    spine_mask_path: str = "",
    shaft_mask_path: str = "",
    bg_mask_path: str = "",
    seg_metrics: dict[str, Any] | None = None,
    shaft_metrics: dict[str, Any] | None = None,
    outline_metrics: dict[str, Any] | None = None,
    shaft_rect_metrics: dict[str, Any] | None = None,
    bg_metrics: dict[str, Any] | None = None,
    auto_rating: int | None = None,
    auto_rating_label: str = "",
    auto_rating_reason: str = "",
) -> dict[str, Any]:
    row = mushroom_info["row"]
    head = geometry["head_zyx"]
    unc = geometry["uncaging_zyx"]
    shaft = geometry["shaft_anchor_yx"]
    feature = {
        "flim_path": flim_path,
        "base_name": base_name,
        "spine_index": spine_index,
        "respan_spine_id": respan_spine_id,
        "png_path": png_path,
        "ini_path": ini_path,
        "spine_mask_tif": spine_mask_path,
        "shaft_mask_tif": shaft_mask_path,
        "bg_mask_tif": bg_mask_path,
        "xy_pixel_um": xy_pixel_um,
        "z_pixel_um": z_pixel_um,
        "min_shaft_to_head_um": min_shaft_to_head_um,
        "dedupe_mushroom_xy_sep_um": dedupe_sep_um,
        "n_candidates": n_candidates,
        "n_mushroom_before_dedupe": n_mushroom_before_dedupe,
        "n_mushroom_after_dedupe": n_mushroom_after_dedupe,
        "head_z_pix": float(head[0]),
        "head_y_pix": float(head[1]),
        "head_x_pix": float(head[2]),
        "uncaging_z_pix": float(unc[0]),
        "uncaging_y_pix": float(unc[1]),
        "uncaging_x_pix": float(unc[2]),
        "shaft_anchor_y_pix": float(shaft[0]),
        "shaft_anchor_x_pix": float(shaft[1]),
        "shaft_to_head_um": float(geometry["shaft_to_head_um"]),
        "head_to_uncaging_um": float(geometry["head_to_uncaging_um"]),
        "head_vol_um3": float(mushroom_info["head_vol_um3"]),
        "head_area_um2": float(row["head_area"]),
        "spine_type": row.get("spine_type", ""),
        "dend_slope": float(geometry["dend_slope"]),
        "dend_intercept": float(geometry["dend_intercept"]),
        "nearest_neighbor_xy_um": float(row.get("nearest_neighbor_xy_um", np.nan))
        if row.get("nearest_neighbor_xy_um")
        else np.nan,
        "nearest_neighbor_respan_id": int(row["nearest_neighbor_spine_id"])
        if row.get("nearest_neighbor_spine_id") not in (None, "")
        else np.nan,
        "nearest_neighbor_label": int(row["nearest_neighbor_spine_id"])
        if row.get("nearest_neighbor_spine_id") not in (None, "")
        else np.nan,
        "respan_head_euclidean_dist_to_dend": float(row["head_euclidean_dist_to_dend"]),
        "bg_center_y_pix": float(geometry["bg_center_yx"][0])
        if geometry.get("bg_center_yx") is not None
        else np.nan,
        "bg_center_x_pix": float(geometry["bg_center_yx"][1])
        if geometry.get("bg_center_yx") is not None
        else np.nan,
        "bg_radius_px": float(geometry.get("bg_radius_px", np.nan)),
        "bg_radius_um": float(geometry.get("bg_radius_px", np.nan)) * xy_pixel_um
        if geometry.get("bg_radius_px") is not None
        else np.nan,
        "param_dedupe_mushroom_xy_sep_um": dedupe_sep_um,
        "detection_pipeline": "respan",
        "auto_rating": auto_rating if auto_rating is not None else np.nan,
        "auto_rating_label": auto_rating_label,
        "auto_rating_reason": auto_rating_reason,
    }
    if seg_metrics:
        feature.update(seg_metrics)
    if shaft_metrics:
        feature.update(shaft_metrics)
    if outline_metrics:
        feature.update(outline_metrics)
    if shaft_rect_metrics:
        feature.update(shaft_rect_metrics)
    if bg_metrics:
        feature.update(bg_metrics)
    return feature


def verify_outputs_for_spine_manager(
    flim_path: str,
    inipaths: list[str],
    *,
    require_png: bool = True,
) -> list[str]:
    """Check ini/png pairs for run_multi_spine_manager compatibility."""
    savefolder = savefolder_from_flim_path(flim_path)
    base_name = base_name_from_flim_path(flim_path)
    expected_prefix = os.path.join(savefolder, base_name + "_")
    issues: list[str] = []

    for inipath in inipaths:
        basename = os.path.basename(inipath)
        if not basename.startswith(base_name + "_"):
            issues.append(f"ini basename mismatch: {basename}")
        if not re.search(r"_\d{3}\.ini$", basename):
            issues.append(f"ini index pattern mismatch: {basename}")
        if os.path.dirname(inipath) != savefolder:
            issues.append(f"ini not in savefolder: {inipath}")
        if not inipath.startswith(expected_prefix):
            issues.append(f"ini path prefix mismatch: {inipath}")

        pngpath = inipath[:-4] + ".png"
        if require_png and not os.path.exists(pngpath):
            issues.append(f"missing png: {pngpath}")

    return issues


def nearest_shaft_anchor_xy(
    labels_zyx: np.ndarray,
    z_pix: float,
    head_y: float,
    head_x: float,
    spine_mask_2d: np.ndarray,
    xy_pixel_um: float,
) -> tuple[float, float, float]:
    """Nearest dendrite pixel outside the spine mask (DeepD3-style shaft anchor)."""
    z_idx = int(round(z_pix))
    z0 = max(0, z_idx - CONTOUR_Z_HALF_WINDOW)
    z1 = min(labels_zyx.shape[0], z_idx + CONTOUR_Z_HALF_WINDOW + 1)
    dend_mip = np.any(labels_zyx[z0:z1] == DENDRITE_LABEL_VALUE, axis=0)
    dend_mip = dend_mip & ~np.asarray(spine_mask_2d, dtype=bool)
    ys, xs = np.where(dend_mip)
    if len(xs) == 0:
        return 0.0, float(head_y), float(head_x)

    nearest = int(np.argmin(np.hypot(xs - head_x, ys - head_y)))
    shaft_x = float(xs[nearest])
    shaft_y = float(ys[nearest])
    dist_um = float(
        np.hypot((head_x - shaft_x) * xy_pixel_um, (head_y - shaft_y) * xy_pixel_um)
    )
    return dist_um, shaft_y, shaft_x


def _compute_spine_geometry(
    row: dict[str, Any],
    zyx: np.ndarray,
    class_labels_zyx: np.ndarray,
    xy_pixel_um: float,
    respan_volumes: dict[str, np.ndarray] | None,
    spine_id: int,
    *,
    low_int_mask_2d: np.ndarray | None = None,
) -> dict[str, Any]:
    head_zyx = [float(row["z"]), float(row["y"]), float(row["x"])]

    if respan_volumes is not None:
        spine_mask_3d, spine_mask_2d = spine_instance_masks(
            respan_volumes, spine_id, head_zyx[0]
        )
    else:
        spine_mask_3d = _spine_mask_3d(
            class_labels_zyx, head_zyx[0], head_zyx[1], head_zyx[2]
        )
        spine_mask_2d = _spine_mask_2d(
            class_labels_zyx, spine_id, head_zyx[0], head_zyx[1], head_zyx[2]
        )

    shaft_to_head_um, shaft_y, shaft_x = nearest_shaft_anchor_xy(
        class_labels_zyx,
        head_zyx[0],
        head_zyx[1],
        head_zyx[2],
        spine_mask_2d,
        xy_pixel_um,
    )
    shaft_mip_binary = shaft_mask_mip_near_z(class_labels_zyx, head_zyx[0])
    dend_slope, dend_intercept = estimate_dendrite_line(
        shaft_mip_binary,
        shaft_y,
        shaft_x,
        xy_pixel_um=xy_pixel_um,
    )
    uncaging_zyx = list(
        estimate_uncaging_point(
            zyx,
            spine_mask_3d,
            head_zyx[0],
            head_zyx[1],
            head_zyx[2],
            dend_slope=dend_slope,
            shaft_y=shaft_y,
            shaft_x=shaft_x,
            xy_pixel_um=xy_pixel_um,
            spine_mask_2d=spine_mask_2d,
            shaft_to_head_um=shaft_to_head_um,
        )
    )
    head_to_uncaging_um = float(
        np.hypot(
            (uncaging_zyx[2] - head_zyx[2]) * xy_pixel_um,
            (uncaging_zyx[1] - head_zyx[1]) * xy_pixel_um,
        )
    )
    z_idx = int(round(head_zyx[0]))
    shaft_rect_mask, _, _ = _shaft_rectangle_mask_2d(
        shaft_x, shaft_y, dend_slope, dend_intercept, zyx.shape[1:]
    )
    shaft_bin_mask = _shaft_binary_mask_from_labels(
        class_labels_zyx,
        shaft_rect_mask,
        z_idx,
        exclude_mask=spine_mask_2d if spine_mask_2d.any() else None,
    )
    z_center = int(np.clip(round(head_zyx[0]), 0, spine_mask_3d.shape[0] - 1))
    z0, z1 = _z_window_indices(z_center, spine_mask_3d.shape[0], SHAFT_Z_HALF_WINDOW)
    target_spine_mip = np.any(spine_mask_3d[z0:z1], axis=0)
    spine_outline_mask_2d = build_spine_outline_mask_2d(target_spine_mip, shaft_mip_binary)
    shaft_fit_radius_mask_2d = shaft_mask_within_radius_2d(
        shaft_mip_binary,
        shaft_x,
        shaft_y,
        xy_pixel_um,
    )
    bg_circle_mask = np.zeros(zyx.shape[1:], dtype=bool)
    bg_center_yx: tuple[float, float] | None = None
    bg_radius_px = 0.0
    if low_int_mask_2d is not None:
        bg_circle_mask, bg_center_yx, bg_radius_px = compute_background_circle_roi(
            low_int_mask_2d,
            spine_outline_mask_2d,
            shaft_fit_radius_mask_2d,
            shaft_rect_mask,
        )
    return {
        "head_zyx": head_zyx,
        "uncaging_zyx": uncaging_zyx,
        "shaft_anchor_yx": (shaft_y, shaft_x),
        "dend_slope": dend_slope,
        "dend_intercept": dend_intercept,
        "shaft_to_head_um": shaft_to_head_um,
        "head_to_uncaging_um": head_to_uncaging_um,
        "spine_mask_3d": spine_mask_3d,
        "spine_mask_2d": spine_mask_2d,
        "shaft_rect_mask": shaft_rect_mask,
        "shaft_bin_mask": shaft_bin_mask,
        "shaft_mip_binary": shaft_mip_binary,
        "target_spine_mip": target_spine_mip,
        "spine_outline_mask_2d": spine_outline_mask_2d,
        "shaft_fit_radius_mask_2d": shaft_fit_radius_mask_2d,
        "bg_circle_mask": bg_circle_mask,
        "bg_center_yx": bg_center_yx,
        "bg_radius_px": bg_radius_px,
        "uses_instance_labels": respan_volumes is not None,
    }


def _add_nearest_neighbor_xy(rows: list[dict[str, Any]], xy_pixel_um: float) -> None:
    coords = np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64)
    spine_ids = [int(float(r["spine_id"])) for r in rows]
    for idx, row in enumerate(rows):
        dists = np.hypot(coords[:, 0] - coords[idx, 0], coords[:, 1] - coords[idx, 1])
        dists[idx] = np.inf
        nearest_idx = int(np.argmin(dists))
        nearest_px = float(dists[nearest_idx])
        row["nearest_neighbor_xy_um"] = nearest_px * xy_pixel_um
        row["nearest_neighbor_spine_id"] = spine_ids[nearest_idx]


def detect_mushroom_from_flim_respan(
    flim_path: str | Path,
    *,
    channel: int = 2,
    min_shaft_to_head_um: float = MIN_SHAFT_TO_HEAD_UM,
    max_head_to_dend_um: float = MAX_HEAD_TO_DEND_UM,
    dedupe_mushroom_xy_sep_um: float = DEDUPE_MUSHROOM_XY_SEP_UM,
    min_head_vol_um3: float = MIN_HEAD_VOL_UM3,
    max_head_vol_um3: float = MAX_HEAD_VOL_UM3,
    use_bandpass: bool = True,
    edge_exclude_percent: float = EDGE_EXCLUDE_PERCENT,
    rerun_respan: bool = False,
    save_per_spine_png: bool = True,
    save_per_spine_ini: bool = True,
    save_per_spine_seg_tiff: bool = True,
    save_overview_pngs: bool = True,
    save_z_triplets: bool = True,
    save_per_mushroom_z_triplets: bool = False,
    auto_rate: bool = True,
) -> list[dict[str, Any]]:
    """Run RESPAN on a FLIM file, keep mushroom spines only, and save markers."""
    flim_path = Path(flim_path)
    if not flim_path.is_file():
        raise FileNotFoundError(f"FLIM file not found: {flim_path}")

    from mushroom_bandpass import SEG_AREA_UM2_BAND, format_band

    print("FLIM:", flim_path)
    if use_bandpass:
        print("Mushroom band-pass:")
        print(f"  seg area (post-geometry): {format_band(SEG_AREA_UM2_BAND, 'um^2')}")
    else:
        print(f"Mushroom threshold: head-to-dendrite XY >= {min_shaft_to_head_um} um")
        if max_head_to_dend_um > 0:
            print(f"Max head-to-dendrite XY: {max_head_to_dend_um} um")
        print(f"Head volume: {min_head_vol_um3}-{max_head_vol_um3} um^3")
    print(
        f"Mushroom dedupe: min XY separation = {dedupe_mushroom_xy_sep_um} um "
        f"(vs all RESPAN candidates)"
    )
    print(f"Edge exclusion: outer {edge_exclude_percent:g}% margin")
    print(
        f"Shaft-fit filter: >= {MIN_SHAFT_FIT_PIXELS_IN_RADIUS} dendrite pixels "
        f"within {SHAFT_FIT_RADIUS_UM:g} um of anchor"
    )

    tiff_path, json_path = ensure_respan_analysis(
        flim_path, channel=channel, rerun=rerun_respan
    )
    run_dir = respan_run_dir(flim_path)
    csv_path = run_dir / "Tables" / f"{tiff_path.stem}_detected_spines.csv"
    label_path = run_dir / "Validation_Data" / "Segmentation_Labels" / tiff_path.name

    xy_pixel_um, _, z_pixel_um = load_pixel_um(json_path)
    if label_path.is_file():
        save_label_z_mip_volume_preview(
            tf.imread(tiff_path),
            tf.imread(label_path),
            xy_pixel_um=xy_pixel_um,
            z_pixel_um=z_pixel_um,
            out_path=label_path.parent / f"{tiff_path.stem}_label_z_mip_volumes.png",
            min_dendrite_vol_um=1.0,
        )
    if not csv_path.is_file():
        print(
            "  RESPAN produced no detected_spines.csv "
            "(no dendrites passed volume threshold or no spines found)."
        )
        mark_highmag_batch_processed_empty(
            flim_path, reason="no_detected_spines_csv"
        )
        return []
    rows = load_spine_rows(csv_path)
    _add_nearest_neighbor_xy(rows, xy_pixel_um)
    if not rows:
        print("  detected_spines.csv is empty; no spines to assign.")
        mark_highmag_batch_processed_empty(
            flim_path, reason="empty_detected_spines_csv"
        )
        return []

    zyx = tf.imread(tiff_path)
    image_shape_yx = (int(zyx.shape[1]), int(zyx.shape[2]))

    mushrooms = filter_mushroom_spines_respan(
        rows,
        use_bandpass=use_bandpass,
        min_shaft_to_head_um=min_shaft_to_head_um,
        max_head_to_dend_um=max_head_to_dend_um,
        min_head_vol_um3=min_head_vol_um3,
        max_head_vol_um3=max_head_vol_um3,
        image_shape_yx=image_shape_yx,
        edge_exclude_percent=edge_exclude_percent,
    )
    n_before = len(mushrooms)
    mushrooms = dedupe_mushrooms_by_xy_respan(
        mushrooms,
        xy_pixel_um,
        dedupe_mushroom_xy_sep_um,
        all_rows=rows,
    )
    if n_before != len(mushrooms):
        print(
            f"  mushroom dedupe: {n_before} -> {len(mushrooms)} "
            f"(removed {n_before - len(mushrooms)} near-duplicate spines)"
        )

    savefolder = savefolder_from_flim_path(str(flim_path))
    base_name = base_name_from_flim_path(str(flim_path))
    os.makedirs(savefolder, exist_ok=True)

    mip_img = zyx.max(axis=0).astype(np.float32)
    class_labels_zyx = tf.imread(label_path)
    respan_volumes = load_respan_volumes(run_dir, tiff_path.stem)
    if respan_volumes is None:
        print(
            "  WARNING: Validation_Vols not found; falling back to coarse class labels "
            "for per-spine masks. Re-run RESPAN with save_intermediate_data=True."
        )
    else:
        print(f"  using RESPAN instance labels from {validation_vols_path(run_dir, tiff_path.stem)}")

    low_int_mask_2d = build_low_intensity_mask_2d(zyx)

    for spine_id, mushroom_info in mushrooms.items():
        mushroom_info["geometry"] = _compute_spine_geometry(
            mushroom_info["row"],
            zyx,
            class_labels_zyx,
            xy_pixel_um,
            respan_volumes,
            spine_id,
            low_int_mask_2d=low_int_mask_2d,
        )

    n_before_shaft_fit = len(mushrooms)
    mushrooms = filter_mushrooms_by_shaft_fit_radius(mushrooms, xy_pixel_um)
    if n_before_shaft_fit != len(mushrooms):
        print(
            f"  shaft-fit radius filter: {n_before_shaft_fit} -> {len(mushrooms)} "
            f"(removed {n_before_shaft_fit - len(mushrooms)} spines with insufficient "
            f"shaft pixels within {SHAFT_FIT_RADIUS_UM:g} um)"
        )

    if use_bandpass:
        n_before_seg = len(mushrooms)
        mushrooms = filter_mushrooms_by_seg_area_bandpass(mushrooms, xy_pixel_um)
        if n_before_seg != len(mushrooms):
            print(
                f"  seg-area bandpass: {n_before_seg} -> {len(mushrooms)} "
                f"(removed {n_before_seg - len(mushrooms)} spines outside seg area band)"
            )

    if save_overview_pngs and mushrooms:
        save_respan_overview_pngs(mip_img, mushrooms, savefolder, base_name)

    z_triplet_dir = run_dir / "Validation_Data" / "Z_slice_triplets" / tiff_path.stem
    if save_z_triplets and mushrooms:
        save_z_triplet_overlays(
            zyx,
            class_labels_zyx,
            mushrooms,
            z_triplet_dir,
            tiff_path.stem,
            respan_volumes=respan_volumes,
        )

    if not mushrooms:
        print("  no mushroom spines detected")
        mark_highmag_batch_processed_empty(
            flim_path, reason="no_mushroom_spines_after_filter"
        )
        return []

    n_removed = clear_existing_mushroom_spine_outputs(savefolder, base_name)
    if n_removed:
        print(f"  cleared {n_removed} prior mushroom spine output file(s) in {savefolder}")

    feature_rows: list[dict[str, Any]] = []
    saved_inipaths: list[str] = []
    for spine_idx, (spine_id, mushroom_info) in enumerate(mushrooms.items()):
        geom = mushroom_info["geometry"]
        head_zyx = geom["head_zyx"]
        uncaging_zyx = geom["uncaging_zyx"]

        stem = ini_stem_for_respan_spine_id(base_name, spine_id)
        inipath = os.path.join(savefolder, f"{stem}.ini")
        pngpath = os.path.join(savefolder, f"{stem}.png")
        seg_dir = seg_mask_save_dir(savefolder)
        spine_mask_path = seg_dir / f"{stem}_spine_mask.tif"
        shaft_mask_path = seg_dir / f"{stem}_shaft_mask.tif"
        spine_outline_mask_path = seg_dir / f"{stem}_spine_outline_mask.tif"
        shaft_fit_radius_mask_path = seg_dir / f"{stem}_shaft_fit_radius_mask.tif"
        bg_mask_path = seg_dir / f"{stem}_bg_mask.tif"
        saved_parts: list[str] = []

        seg_metrics = _regionprops_metrics(geom["spine_mask_2d"], xy_pixel_um, prefix="seg")
        auto_rating: int | None = None
        auto_rating_label = ""
        auto_rating_reason = ""
        if auto_rate:
            from mushroom_auto_rating import predict_auto_rating

            auto_rating, auto_rating_label, auto_rating_reason = predict_auto_rating(
                head_vol_um3=float(mushroom_info["head_vol_um3"]),
                head_area_um2=float(mushroom_info["row"]["head_area"]),
                respan_head_euclidean_dist_to_dend=float(
                    mushroom_info["row"]["head_euclidean_dist_to_dend"]
                ),
                shaft_to_head_um=float(geom["shaft_to_head_um"]),
                seg_area_um2=float(seg_metrics.get("seg_area_um2", np.nan))
                if seg_metrics.get("seg_area_um2") is not None
                else None,
            )

        if save_per_spine_seg_tiff:
            save_spine_segmentation_tiff(geom["spine_mask_3d"], spine_mask_path)
            tf.imwrite(
                shaft_mask_path,
                geom["shaft_bin_mask"].astype(np.uint8),
                photometric="minisblack",
            )
            save_mask_2d_tiff(geom["spine_outline_mask_2d"], spine_outline_mask_path)
            save_mask_2d_tiff(geom["shaft_fit_radius_mask_2d"], shaft_fit_radius_mask_path)
            save_mask_2d_tiff(geom["bg_circle_mask"], bg_mask_path)
            saved_parts.append(f"{SEG_MASK_SUBFOLDER}/masks")

        if save_per_spine_png:
            save_mushroom_overlay_png_respan(
                zyx,
                class_labels_zyx,
                head_zyx,
                uncaging_zyx,
                geom["dend_slope"],
                geom["dend_intercept"],
                pngpath,
                spine_mask_3d=geom["spine_mask_3d"],
                shaft_rect_mask=geom["shaft_rect_mask"],
                bg_circle_mask=geom["bg_circle_mask"],
                spine_id=spine_id,
                auto_rating=auto_rating,
                auto_rating_label=auto_rating_label,
            )
            saved_parts.append("png")

        if save_per_spine_ini:
            ini_excluded = ini_excluded_for_auto_rating(auto_rating)
            save_spine_dend_info(
                [round(head_zyx[0]), round(head_zyx[1]), round(head_zyx[2])],
                geom["dend_slope"],
                geom["dend_intercept"],
                inipath,
                excluded=ini_excluded,
            )
            saved_inipaths.append(inipath)
            saved_parts.append("ini")

        shaft_metrics = _regionprops_metrics(
            geom["shaft_bin_mask"], xy_pixel_um, prefix="shaft_bin"
        )
        outline_metrics = _regionprops_metrics(
            geom["spine_outline_mask_2d"], xy_pixel_um, prefix="outline"
        )
        shaft_rect_metrics = _regionprops_metrics(
            geom["shaft_rect_mask"], xy_pixel_um, prefix="shaft_roi"
        )
        bg_metrics = _regionprops_metrics(geom["bg_circle_mask"], xy_pixel_um, prefix="bg")

        if save_per_mushroom_z_triplets:
            save_mushroom_z_triplet_overlays(
                spine_id,
                mushroom_info,
                zyx,
                class_labels_zyx,
                Path(savefolder),
                stem,
                respan_volumes=respan_volumes,
            )

        if saved_parts:
            rating_note = ""
            if auto_rating is not None:
                rating_note = f", auto-rating {auto_rating} ({auto_rating_label})"
            print(
                f"  saved mushroom {', '.join(saved_parts)} "
                f"{stem} (RESPAN spine {spine_id}, "
                f"shaft-to-head {geom['shaft_to_head_um']:.3f} um, "
                f"uncaging offset {geom['head_to_uncaging_um']:.3f} um{rating_note})"
            )

        feature_rows.append(
            build_feature_row(
                flim_path=str(flim_path),
                base_name=base_name,
                spine_index=int(spine_id),
                respan_spine_id=spine_id,
                mushroom_info=mushroom_info,
                geometry=geom,
                xy_pixel_um=xy_pixel_um,
                z_pixel_um=z_pixel_um,
                min_shaft_to_head_um=min_shaft_to_head_um,
                dedupe_sep_um=dedupe_mushroom_xy_sep_um,
                n_candidates=len(rows),
                n_mushroom_before_dedupe=n_before,
                n_mushroom_after_dedupe=len(mushrooms),
                png_path=pngpath if save_per_spine_png else "",
                ini_path=inipath if save_per_spine_ini else "",
                spine_mask_path=str(spine_mask_path) if save_per_spine_seg_tiff else "",
                shaft_mask_path=str(shaft_mask_path) if save_per_spine_seg_tiff else "",
                bg_mask_path=str(bg_mask_path) if save_per_spine_seg_tiff else "",
                seg_metrics=seg_metrics,
                shaft_metrics=shaft_metrics,
                outline_metrics=outline_metrics,
                shaft_rect_metrics=shaft_rect_metrics,
                bg_metrics=bg_metrics,
                auto_rating=auto_rating,
                auto_rating_label=auto_rating_label,
                auto_rating_reason=auto_rating_reason,
            )
        )

    save_mushroom_features_csv(savefolder, base_name, feature_rows)

    if save_per_spine_ini:
        issues = verify_outputs_for_spine_manager(
            str(flim_path), saved_inipaths, require_png=save_per_spine_png
        )
        if issues:
            print("  VERIFICATION ISSUES:")
            for issue in issues:
                print("   -", issue)
        else:
            print(
                f"  verified {len(saved_inipaths)} ini"
                f"{'/png pair(s)' if save_per_spine_png else '(s)'} for run_multi_spine_manager"
            )

    return feature_rows


def _spine_info_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row": row,
        "shaft_to_head_um": float(row["head_euclidean_dist_to_dend"]),
        "head_vol_um3": float(row["head_vol"]),
    }


def filter_detected_spines_respan(
    rows: list[dict[str, Any]],
    *,
    min_head_vol_um3: float = 0.0,
    image_shape_yx: tuple[int, int] | None = None,
    edge_exclude_percent: float = EDGE_EXCLUDE_PERCENT,
) -> dict[int, dict[str, Any]]:
    """Keep RESPAN CSV rows with optional edge and minimum head-volume filters."""
    spines: dict[int, dict[str, Any]] = {}
    for row in rows:
        spine_id = int(float(row["spine_id"]))
        head_x = float(row["x"])
        head_y = float(row["y"])
        head_vol_um3 = float(row["head_vol"])
        if image_shape_yx is not None and centroid_in_edge_band(
            head_x, head_y, image_shape_yx, edge_exclude_percent
        ):
            print(
                f"  skip spine {spine_id}: centroid ({head_x:.1f}, {head_y:.1f}) "
                f"in outer {edge_exclude_percent:g}% image margin"
            )
            continue
        if head_vol_um3 < min_head_vol_um3:
            print(
                f"  skip spine {spine_id}: head volume {head_vol_um3:.3f} um^3 "
                f"(< {min_head_vol_um3} um^3)"
            )
            continue
        spines[spine_id] = _spine_info_from_row(row)
    return spines


def save_respan_spine_features_csv(
    savefolder: str | Path,
    base_name: str,
    feature_rows: list[dict[str, Any]],
) -> str:
    if not feature_rows:
        return ""
    import pandas as pd

    out_path = Path(savefolder) / f"{base_name}_respan_spine_features.csv"
    pd.DataFrame(feature_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"  saved RESPAN spine features CSV ({len(feature_rows)} rows): {out_path}")
    return str(out_path)


def detect_spines_from_flim_respan(
    flim_path: str | Path,
    *,
    channel: int = 2,
    min_head_vol_um3: float = 0.0,
    edge_exclude_percent: float = EDGE_EXCLUDE_PERCENT,
    dedupe_xy_sep_um: float = 0.0,
    rerun_respan: bool = False,
    nnunet_fold: str | int = "all",
    low_mag_mode: bool = True,
    save_per_spine_png: bool = False,
    save_per_spine_ini: bool = False,
    save_per_spine_seg_tiff: bool = False,
    save_overview_pngs: bool = True,
    save_z_triplets: bool = False,
    save_class_overlay_pngs: bool = True,
) -> list[dict[str, Any]]:
    """Run RESPAN on a FLIM file and save spine detection outputs.

    low_mag_mode=True (default): class-overlay Z slices + Z projection PNGs only.
    low_mag_mode=False: per-spine ini/png/seg masks with full geometry (slower).
    """
    flim_path = Path(flim_path)
    if not flim_path.is_file():
        raise FileNotFoundError(f"FLIM file not found: {flim_path}")

    print("FLIM:", flim_path)
    print(f"Mode: {'low-mag (class overlays)' if low_mag_mode else 'full per-spine ROI'}")
    print(f"Spine filter: min head volume = {min_head_vol_um3} um^3")
    print(f"Edge exclusion: outer {edge_exclude_percent:g}% margin")
    if dedupe_xy_sep_um > 0:
        print(f"Dedupe: min XY separation = {dedupe_xy_sep_um} um")

    if low_mag_mode:
        save_per_spine_png = False
        save_per_spine_ini = False
        save_per_spine_seg_tiff = False

    tiff_path, json_path = ensure_respan_analysis(
        flim_path,
        channel=channel,
        rerun=rerun_respan,
        nnunet_fold=nnunet_fold,
    )
    run_dir = respan_run_dir(flim_path)
    csv_path = run_dir / "Tables" / f"{tiff_path.stem}_detected_spines.csv"
    label_path = run_dir / "Validation_Data" / "Segmentation_Labels" / tiff_path.name

    xy_pixel_um, _, z_pixel_um = load_pixel_um(json_path)
    if not csv_path.is_file():
        print(
            "  RESPAN produced no detected_spines.csv "
            "(no dendrites passed volume threshold or no spines found)."
        )
        return []
    rows = load_spine_rows(csv_path)
    _add_nearest_neighbor_xy(rows, xy_pixel_um)
    print(f"  RESPAN detected {len(rows)} spine(s) in CSV")
    if not rows:
        print("  detected_spines.csv is empty; nothing to export.")
        return []

    zyx = tf.imread(tiff_path)
    image_shape_yx = (int(zyx.shape[1]), int(zyx.shape[2]))

    spines = filter_detected_spines_respan(
        rows,
        min_head_vol_um3=min_head_vol_um3,
        image_shape_yx=image_shape_yx,
        edge_exclude_percent=edge_exclude_percent,
    )
    n_before = len(spines)
    if dedupe_xy_sep_um > 0:
        spines = dedupe_mushrooms_by_xy_respan(
            spines,
            xy_pixel_um,
            dedupe_xy_sep_um,
            all_rows=rows,
        )
        if n_before != len(spines):
            print(
                f"  dedupe: {n_before} -> {len(spines)} "
                f"(removed {n_before - len(spines)} near-duplicate spines)"
            )

    savefolder = savefolder_from_flim_path(str(flim_path))
    base_name = base_name_from_flim_path(str(flim_path))
    os.makedirs(savefolder, exist_ok=True)

    class_labels_zyx = tf.imread(label_path)

    if not spines:
        print("  no spines kept after filtering")
        if save_class_overlay_pngs:
            save_lowmag_class_overlay_pngs(zyx, class_labels_zyx, savefolder, base_name)
        return []

    if low_mag_mode:
        n_removed = clear_existing_mushroom_spine_outputs(savefolder, base_name)
        if n_removed:
            print(f"  cleared {n_removed} prior per-spine output file(s) in {savefolder}")

        overlay_mip_rgb = build_class_overlay_mip(zyx, class_labels_zyx)
        if save_class_overlay_pngs:
            save_lowmag_class_overlay_pngs(zyx, class_labels_zyx, savefolder, base_name)
        if save_overview_pngs:
            save_spine_centroids_overview_png(
                overlay_mip_rgb,
                spines,
                savefolder,
                base_name,
                overview_title=f"RESPAN spines (n={len(spines)})",
            )

        feature_rows = [
            build_minimal_spine_feature_row(
                flim_path=str(flim_path),
                base_name=base_name,
                spine_index=int(spine_id),
                respan_spine_id=spine_id,
                spine_info=spine_info,
                xy_pixel_um=xy_pixel_um,
                z_pixel_um=z_pixel_um,
                dedupe_sep_um=dedupe_xy_sep_um,
                n_candidates=len(rows),
                n_before_dedupe=n_before,
                n_after_dedupe=len(spines),
            )
            for spine_idx, (spine_id, spine_info) in enumerate(spines.items())
        ]
        save_respan_spine_features_csv(savefolder, base_name, feature_rows)
        print(f"  saved {len(feature_rows)} spine(s) to feature CSV (no per-spine ini/png)")
        return feature_rows

    mip_img = zyx.max(axis=0).astype(np.float32)
    respan_volumes = load_respan_volumes(run_dir, tiff_path.stem)
    if respan_volumes is None:
        print(
            "  WARNING: Validation_Vols not found; falling back to coarse class labels "
            "for per-spine masks."
        )
    else:
        print(f"  using RESPAN instance labels from {validation_vols_path(run_dir, tiff_path.stem)}")

    low_int_mask_2d = build_low_intensity_mask_2d(zyx)

    n_spines = len(spines)
    print(f"  computing geometry for {n_spines} spine(s)...")
    for idx, (spine_id, spine_info) in enumerate(spines.items(), start=1):
        spine_info["geometry"] = _compute_spine_geometry(
            spine_info["row"],
            zyx,
            class_labels_zyx,
            xy_pixel_um,
            respan_volumes,
            spine_id,
            low_int_mask_2d=low_int_mask_2d,
        )
        if n_spines >= 20 and (idx == 1 or idx % 25 == 0 or idx == n_spines):
            print(f"    geometry {idx}/{n_spines}")

    if save_overview_pngs and spines:
        save_respan_overview_pngs(
            mip_img,
            spines,
            savefolder,
            base_name,
            overview_title=f"RESPAN spines (n={len(spines)})",
        )

    z_triplet_dir = run_dir / "Validation_Data" / "Z_slice_triplets" / tiff_path.stem
    if save_z_triplets and spines:
        save_z_triplet_overlays(
            zyx,
            class_labels_zyx,
            spines,
            z_triplet_dir,
            tiff_path.stem,
            respan_volumes=respan_volumes,
        )

    n_removed = clear_existing_mushroom_spine_outputs(savefolder, base_name)
    if n_removed:
        print(f"  cleared {n_removed} prior spine output file(s) in {savefolder}")

    feature_rows: list[dict[str, Any]] = []
    saved_inipaths: list[str] = []
    for spine_idx, (spine_id, spine_info) in enumerate(spines.items()):
        geom = spine_info["geometry"]
        head_zyx = geom["head_zyx"]
        uncaging_zyx = geom["uncaging_zyx"]

        stem = ini_stem_for_respan_spine_id(base_name, spine_id)
        inipath = os.path.join(savefolder, f"{stem}.ini")
        pngpath = os.path.join(savefolder, f"{stem}.png")
        seg_dir = seg_mask_save_dir(savefolder)
        spine_mask_path = seg_dir / f"{stem}_spine_mask.tif"
        shaft_mask_path = seg_dir / f"{stem}_shaft_mask.tif"
        spine_outline_mask_path = seg_dir / f"{stem}_spine_outline_mask.tif"
        shaft_fit_radius_mask_path = seg_dir / f"{stem}_shaft_fit_radius_mask.tif"
        bg_mask_path = seg_dir / f"{stem}_bg_mask.tif"
        saved_parts: list[str] = []

        seg_metrics = _regionprops_metrics(geom["spine_mask_2d"], xy_pixel_um, prefix="seg")

        if save_per_spine_seg_tiff:
            save_spine_segmentation_tiff(geom["spine_mask_3d"], spine_mask_path)
            tf.imwrite(
                shaft_mask_path,
                geom["shaft_bin_mask"].astype(np.uint8),
                photometric="minisblack",
            )
            save_mask_2d_tiff(geom["spine_outline_mask_2d"], spine_outline_mask_path)
            save_mask_2d_tiff(geom["shaft_fit_radius_mask_2d"], shaft_fit_radius_mask_path)
            save_mask_2d_tiff(geom["bg_circle_mask"], bg_mask_path)
            saved_parts.append(f"{SEG_MASK_SUBFOLDER}/masks")

        if save_per_spine_png:
            save_mushroom_overlay_png_respan(
                zyx,
                class_labels_zyx,
                head_zyx,
                uncaging_zyx,
                geom["dend_slope"],
                geom["dend_intercept"],
                pngpath,
                spine_mask_3d=geom["spine_mask_3d"],
                shaft_rect_mask=geom["shaft_rect_mask"],
                bg_circle_mask=geom["bg_circle_mask"],
                spine_id=spine_id,
            )
            saved_parts.append("png")

        if save_per_spine_ini:
            save_spine_dend_info(
                [round(head_zyx[0]), round(head_zyx[1]), round(head_zyx[2])],
                geom["dend_slope"],
                geom["dend_intercept"],
                inipath,
            )
            saved_inipaths.append(inipath)
            saved_parts.append("ini")

        shaft_metrics = _regionprops_metrics(
            geom["shaft_bin_mask"], xy_pixel_um, prefix="shaft_bin"
        )
        outline_metrics = _regionprops_metrics(
            geom["spine_outline_mask_2d"], xy_pixel_um, prefix="outline"
        )
        shaft_rect_metrics = _regionprops_metrics(
            geom["shaft_rect_mask"], xy_pixel_um, prefix="shaft_roi"
        )
        bg_metrics = _regionprops_metrics(geom["bg_circle_mask"], xy_pixel_um, prefix="bg")

        if saved_parts:
            print(
                f"  saved spine {', '.join(saved_parts)} "
                f"{stem} (RESPAN spine {spine_id}, "
                f"shaft-to-head {geom['shaft_to_head_um']:.3f} um, "
                f"uncaging offset {geom['head_to_uncaging_um']:.3f} um)"
            )

        feature_rows.append(
            build_feature_row(
                flim_path=str(flim_path),
                base_name=base_name,
                spine_index=int(spine_id),
                respan_spine_id=spine_id,
                mushroom_info=spine_info,
                geometry=geom,
                xy_pixel_um=xy_pixel_um,
                z_pixel_um=z_pixel_um,
                min_shaft_to_head_um=0.0,
                dedupe_sep_um=dedupe_xy_sep_um,
                n_candidates=len(rows),
                n_mushroom_before_dedupe=n_before,
                n_mushroom_after_dedupe=len(spines),
                png_path=pngpath if save_per_spine_png else "",
                ini_path=inipath if save_per_spine_ini else "",
                spine_mask_path=str(spine_mask_path) if save_per_spine_seg_tiff else "",
                shaft_mask_path=str(shaft_mask_path) if save_per_spine_seg_tiff else "",
                bg_mask_path=str(bg_mask_path) if save_per_spine_seg_tiff else "",
                seg_metrics=seg_metrics,
                shaft_metrics=shaft_metrics,
                outline_metrics=outline_metrics,
                shaft_rect_metrics=shaft_rect_metrics,
                bg_metrics=bg_metrics,
            )
        )

    save_respan_spine_features_csv(savefolder, base_name, feature_rows)

    if save_per_spine_ini:
        issues = verify_outputs_for_spine_manager(
            str(flim_path), saved_inipaths, require_png=save_per_spine_png
        )
        if issues:
            print("  VERIFICATION ISSUES:")
            for issue in issues:
                print("   -", issue)
        else:
            print(
                f"  verified {len(saved_inipaths)} ini"
                f"{'/png pair(s)' if save_per_spine_png else '(s)'} for run_multi_spine_manager"
            )

    return feature_rows


def _allocate_ini_stem(savefolder: str | Path, base_name: str) -> str:
    """Deprecated: use ini_stem_for_respan_spine_id with RESPAN instance id."""
    savefolder = Path(savefolder)
    indices: list[int] = []
    for ini_path in savefolder.glob(f"{base_name}_*.ini"):
        match = re.search(r"_id(\d+)\.ini$", ini_path.name)
        if match:
            indices.append(int(match.group(1)))
        else:
            match = re.search(r"_(\d{3})\.ini$", ini_path.name)
            if match:
                indices.append(int(match.group(1)))
    next_idx = (max(indices) + 1) if indices else 0
    return f"{base_name}_{str(next_idx).zfill(3)}"


def load_flim_respan_review_bundle(
    flim_path: str | Path,
    *,
    channel: int = 2,
) -> dict[str, Any]:
    """Load arrays and tables needed for post-batch RESPAN spine review."""
    flim_path = Path(flim_path)
    if not flim_path.is_file():
        raise FileNotFoundError(f"FLIM file not found: {flim_path}")

    if not respan_outputs_ready(flim_path, channel):
        raise FileNotFoundError(
            f"RESPAN outputs missing for {flim_path}. "
            "Run run_respan_spine_manager.py first."
        )

    run_dir = respan_run_dir(flim_path)
    tiff_path, json_path = respan_export_paths(flim_path, channel=channel)
    csv_path = run_dir / "Tables" / f"{tiff_path.stem}_detected_spines.csv"
    label_path = run_dir / "Validation_Data" / "Segmentation_Labels" / tiff_path.name
    z_triplet_dir = run_dir / "Validation_Data" / "Z_slice_triplets" / tiff_path.stem

    xy_pixel_um, _, z_pixel_um = load_pixel_um(json_path)
    rows = {
        int(float(r["spine_id"])): r
        for r in load_spine_rows(csv_path, missing_ok=True)
    }
    zyx = tf.imread(tiff_path)
    class_labels_zyx = tf.imread(label_path)
    respan_volumes = load_respan_volumes(run_dir, tiff_path.stem)
    savefolder = savefolder_from_flim_path(str(flim_path))
    base_name = base_name_from_flim_path(str(flim_path))

    feature_csv = Path(savefolder) / f"{base_name}_respan_mushroom_features.csv"
    feature_by_spine_id: dict[int, dict[str, Any]] = {}
    if feature_csv.is_file():
        import pandas as pd

        for rec in pd.read_csv(feature_csv).to_dict(orient="records"):
            feature_by_spine_id[int(rec["respan_spine_id"])] = rec

    return {
        "flim_path": str(flim_path),
        "savefolder": savefolder,
        "base_name": base_name,
        "run_dir": str(run_dir),
        "tiff_stem": tiff_path.stem,
        "z_triplet_dir": str(z_triplet_dir),
        "raw_zyx": zyx,
        "class_labels_zyx": class_labels_zyx,
        "respan_volumes": respan_volumes,
        "detected_rows": rows,
        "xy_pixel_um": xy_pixel_um,
        "z_pixel_um": z_pixel_um,
        "feature_by_spine_id": feature_by_spine_id,
        "feature_csv": str(feature_csv),
        "low_int_mask_2d": build_low_intensity_mask_2d(zyx),
    }


def upsert_feature_csv_row(
    savefolder: str | Path,
    base_name: str,
    feature_row: dict[str, Any],
) -> str:
    """Insert or replace one spine row in the per-FLIM feature CSV."""
    import pandas as pd

    out_path = Path(savefolder) / f"{base_name}_respan_mushroom_features.csv"
    rows: list[dict[str, Any]] = []
    if out_path.is_file():
        rows = pd.read_csv(out_path).to_dict(orient="records")
    spine_id = int(feature_row["respan_spine_id"])
    rows = [r for r in rows if int(r.get("respan_spine_id", -1)) != spine_id]
    rows.append(feature_row)
    rows.sort(key=lambda r: int(r.get("spine_index", 0)))
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    return str(out_path)


def ini_excluded_for_auto_rating(
    auto_rating: int | None,
    *,
    manual_excluded: int | None = None,
    force_excluded: bool = False,
    auto_accept_min_rating: int = 4,
) -> int:
    """
    Decide ini excluded flag (0=accepted, 1=rejected/pending).

    Manual accept/reject (manual_excluded) always wins. Batch default: auto_rating>=4
    is accepted; rating<3 is rejected; rating 3 or unrated stays pending (excluded=1).
    """
    if force_excluded:
        return 1
    if manual_excluded is not None:
        return int(manual_excluded)
    if auto_rating is not None and int(auto_rating) >= int(auto_accept_min_rating):
        return 0
    if auto_rating is not None and int(auto_rating) < 3:
        return 1
    return 1


def export_single_mushroom_spine(
    flim_path: str | Path,
    spine_id: int,
    *,
    channel: int = 2,
    excluded: int | None = None,
    stem: str | None = None,
    auto_rate: bool = True,
    force_excluded: bool = False,
) -> dict[str, Any]:
    """
    Export ini/png/seg_masks for one RESPAN spine (manual accept or re-enable).

    Does not re-run RESPAN inference. Creates outputs compatible with the legacy
    spine manager when the spine was previously filtered out.
    """
    bundle = load_flim_respan_review_bundle(flim_path, channel=channel)
    row = bundle["detected_rows"].get(int(spine_id))
    if row is None:
        raise ValueError(f"RESPAN spine_id {spine_id} not found in detected_spines.csv")

    savefolder = bundle["savefolder"]
    base_name = bundle["base_name"]
    zyx = bundle["raw_zyx"]
    class_labels_zyx = bundle["class_labels_zyx"]
    respan_volumes = bundle["respan_volumes"]
    xy_pixel_um = float(bundle["xy_pixel_um"])
    z_pixel_um = float(bundle["z_pixel_um"])

    mushroom_info = {
        "row": row,
        "shaft_to_head_um": float(row.get("head_euclidean_dist_to_dend", 0.0)),
        "head_vol_um3": float(row["head_vol"]),
    }
    geom = _compute_spine_geometry(
        row,
        zyx,
        class_labels_zyx,
        xy_pixel_um,
        respan_volumes,
        int(spine_id),
        low_int_mask_2d=bundle["low_int_mask_2d"],
    )
    mushroom_info["geometry"] = geom

    existing = bundle["feature_by_spine_id"].get(int(spine_id))
    stem = ini_stem_for_respan_spine_id(base_name, int(spine_id))
    if existing and existing.get("ini_path"):
        old_stem = Path(str(existing["ini_path"])).stem
        if old_stem != stem:
            remove_spine_stem_outputs(savefolder, old_stem)

    inipath = os.path.join(savefolder, f"{stem}.ini")
    pngpath = os.path.join(savefolder, f"{stem}.png")
    seg_dir = seg_mask_save_dir(savefolder)
    spine_mask_path = seg_dir / f"{stem}_spine_mask.tif"
    shaft_mask_path = seg_dir / f"{stem}_shaft_mask.tif"
    spine_outline_mask_path = seg_dir / f"{stem}_spine_outline_mask.tif"
    shaft_fit_radius_mask_path = seg_dir / f"{stem}_shaft_fit_radius_mask.tif"
    bg_mask_path = seg_dir / f"{stem}_bg_mask.tif"

    head_zyx = geom["head_zyx"]
    uncaging_zyx = geom["uncaging_zyx"]
    seg_metrics = _regionprops_metrics(geom["spine_mask_2d"], xy_pixel_um, prefix="seg")

    auto_rating: int | None = None
    auto_rating_label = ""
    auto_rating_reason = ""
    if auto_rate:
        from mushroom_auto_rating import predict_auto_rating

        auto_rating, auto_rating_label, auto_rating_reason = predict_auto_rating(
            head_vol_um3=float(mushroom_info["head_vol_um3"]),
            head_area_um2=float(row["head_area"]),
            respan_head_euclidean_dist_to_dend=float(row["head_euclidean_dist_to_dend"]),
            shaft_to_head_um=float(geom["shaft_to_head_um"]),
            seg_area_um2=float(seg_metrics.get("seg_area_um2", np.nan))
            if seg_metrics.get("seg_area_um2") is not None
            else None,
        )

    ini_excluded = ini_excluded_for_auto_rating(
        auto_rating,
        manual_excluded=excluded,
        force_excluded=force_excluded,
    )

    save_spine_segmentation_tiff(geom["spine_mask_3d"], spine_mask_path)
    tf.imwrite(
        shaft_mask_path,
        geom["shaft_bin_mask"].astype(np.uint8),
        photometric="minisblack",
    )
    save_mask_2d_tiff(geom["spine_outline_mask_2d"], spine_outline_mask_path)
    save_mask_2d_tiff(geom["shaft_fit_radius_mask_2d"], shaft_fit_radius_mask_path)
    save_mask_2d_tiff(geom["bg_circle_mask"], bg_mask_path)

    save_mushroom_overlay_png_respan(
        zyx,
        class_labels_zyx,
        head_zyx,
        uncaging_zyx,
        geom["dend_slope"],
        geom["dend_intercept"],
        pngpath,
        spine_mask_3d=geom["spine_mask_3d"],
        shaft_rect_mask=geom["shaft_rect_mask"],
        bg_circle_mask=geom["bg_circle_mask"],
        spine_id=int(spine_id),
        auto_rating=auto_rating,
        auto_rating_label=auto_rating_label,
    )

    save_spine_dend_info(
        [round(head_zyx[0]), round(head_zyx[1]), round(head_zyx[2])],
        geom["dend_slope"],
        geom["dend_intercept"],
        inipath,
        excluded=ini_excluded,
    )

    shaft_metrics = _regionprops_metrics(geom["shaft_bin_mask"], xy_pixel_um, prefix="shaft_bin")
    outline_metrics = _regionprops_metrics(
        geom["spine_outline_mask_2d"], xy_pixel_um, prefix="outline"
    )
    shaft_rect_metrics = _regionprops_metrics(
        geom["shaft_rect_mask"], xy_pixel_um, prefix="shaft_roi"
    )
    bg_metrics = _regionprops_metrics(geom["bg_circle_mask"], xy_pixel_um, prefix="bg")
    spine_index = int(spine_id)

    feature_row = build_feature_row(
        flim_path=str(flim_path),
        base_name=base_name,
        spine_index=spine_index,
        respan_spine_id=int(spine_id),
        mushroom_info=mushroom_info,
        geometry=geom,
        xy_pixel_um=xy_pixel_um,
        z_pixel_um=z_pixel_um,
        min_shaft_to_head_um=MIN_SHAFT_TO_HEAD_UM,
        dedupe_sep_um=DEDUPE_MUSHROOM_XY_SEP_UM,
        n_candidates=len(bundle["detected_rows"]),
        n_mushroom_before_dedupe=len(bundle["detected_rows"]),
        n_mushroom_after_dedupe=len(bundle["detected_rows"]),
        png_path=pngpath,
        ini_path=inipath,
        spine_mask_path=str(spine_mask_path),
        shaft_mask_path=str(shaft_mask_path),
        bg_mask_path=str(bg_mask_path),
        seg_metrics=seg_metrics,
        shaft_metrics=shaft_metrics,
        outline_metrics=outline_metrics,
        shaft_rect_metrics=shaft_rect_metrics,
        bg_metrics=bg_metrics,
        auto_rating=auto_rating,
        auto_rating_label=auto_rating_label,
        auto_rating_reason=auto_rating_reason,
    )
    upsert_feature_csv_row(savefolder, base_name, feature_row)

    return {
        "ini_path": inipath,
        "png_path": pngpath,
        "stem": stem,
        "excluded": ini_excluded,
        "auto_rating": auto_rating,
        "feature_row": feature_row,
    }


def set_spine_ini_excluded(ini_path: str | Path, excluded: int) -> None:
    """Update excluded flag in an existing spine ini."""
    ini_path = Path(ini_path)
    config = configparser.ConfigParser()
    config.read(ini_path, encoding="utf-8")
    if "uncaging_settings" not in config:
        raise ValueError(f"Invalid ini file: {ini_path}")
    config["uncaging_settings"]["excluded"] = str(int(excluded))
    with ini_path.open("w", encoding="utf-8") as handle:
        config.write(handle)
