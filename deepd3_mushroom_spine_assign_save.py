# -*- coding: utf-8 -*-
"""
Batch-assign mushroom spines from .flim files using DeepD3.

Mushroom criterion: shaft-to-head XY distance >= MIN_SHAFT_TO_HEAD_UM (Z not counted).
Outputs optional per-spine .ini + .png (run_multi_spine_manager layout) and DeepD3 overviews.
Does not launch define_uncagingPoint_dend_click_multiple after detection.
"""
import glob
import os
import re
import datetime
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import label as ndimage_label
from scipy.stats import percentileofscore
from skimage.draw import polygon
from skimage.measure import find_contours, regionprops
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.segmentation import watershed

from deepd3_spine_head_detector import SpinePosDeepD3, resolve_savefolder
from flimage_graph_func import calc_point_on_line_close_to_xy
from FLIMageFileReader2 import FileReader
from multidim_tiff_viewer import save_spine_dend_info, read_xyz_single

# --- user settings (edit here) ---
# Mushroom: shaft-to-head XY >= this (um). 0 = almost all result_dict entries pass.
MIN_SHAFT_TO_HEAD_UM = 0.5
# Suppress duplicate detections: keep one mushroom per cluster within this XY sep (um).
# Floodfill often splits one spine into several 3D ROI labels at nearly the same head.
DEDUPE_MUSHROOM_XY_SEP_UM = 2.5

# DEEPD3_MODEL_PATH = r"C:\Users\WatabeT\Documents\Git\DeepD3\deepd3\DeepD3_32F_94nm.h5"
DEEPD3_MODEL_PATH = r"C:\Users\WatabeT\Documents\Git\DeepD3\deepd3\DeepD3_8F.h5"

DEFAULT_HIGHMAG_FOLDER = (
    # r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260515\copied_auto1_for_find_mushroom"
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260515\lowmags"

)
# DEFAULT_HIGHMAG_FILENAME = "*_highmag_*002.flim"
DEFAULT_HIGHMAG_FILENAME = "*pos6_256_001.flim"
DEFAULT_EXCLUDE_INI_SAVED = False
DEFAULT_FROM_LATEST_FILE = False
DEFAULT_MIN_SECONDS_SINCE_MODIFICATION = 20

IGNORE_PLANES_FIRST_LAST = 1
IGNORE_EDGE_PERCENTILE = 2

# --- Which knob does what (read before tuning) ---
# DEEPD3_ROI_MODE=floodfill: roi_peakThreshold is the MAIN lever for ROI count (see sweep30).
#   roi_areaThreshold still passed to floodfill but often weak in 0.10–0.25 range.
# DEEPD3_ROI_MODE=thresholded: roi_areaThreshold is the main lever; peak is ignored.
# dendrite_skeleton_threshold : spine-position tracing along shaft AFTER ROIs; not shaft pred map
# Shaft map tuning: shaft_detection_study/ (256_4x: tophat_clahe + no fusion)

DEEPD3_ROI_MODE = "floodfill"

# --- DeepD3 input stack (before plane-by-plane CNN) ---
# Local Z-MIP: each Z fed to DeepD3 is max of +/- LOCAL_Z_MIP_RADIUS planes
USE_LOCAL_Z_MIP = False
LOCAL_Z_MIP_RADIUS = 0

# --- thin branch (changes shaft pred MIP — not roi_areaThreshold) ---
# AP5_pos6_256_4x_001: preprocess-only (nofusion) balanced best vs fusion grid
ENHANCE_THIN_BRANCHES = False
STACK_PREPROCESS = "tophat_clahe"
# Used only if ENHANCE_THIN_BRANCHES=True (mild fusion: try p96-98, w0.2-0.3)
IMAGE_FUSION_PERCENTILE = 94.0
IMAGE_FUSION_WEIGHT = 0.4
DENDRITE_CLOSING_ITERATIONS = 1

DEEPD3_ROI_PARAMS = {
    "roi_peakThreshold": 0.4,
    "roi_areaThreshold": 0.2,
    "roi_seedDelta": 0.1,
    "roi_distanceToSeed": 2,
    "min_roi_size": 2,
    "max_roi_size": 100,
    "min_planes": 1,
    "dendrite_skeleton_threshold": 0.5,
}
MAX_DIST_SPINE_DEND_UM =5.0

RAW_MIP_VMAX_PERCENTILE = 95
SAVE_SHAFT_RAW_MIP_COMPARE = True
SHAFT_PRED_CONTOUR_LEVEL = 0.2

UPPER_SPINE_PIXEL_PERCENTILE = 99
LOWER_SPINE_PIXEL_PERCENTILE = 1
UPPER_SPINE_INTENSITY_PERCENTILE = 99
LOWER_SPINE_INTENSITY_PERCENTILE = 1

# DeepD3 overview outputs per .flim (in savefolder): {base_name}_mip.png, _roi.png
SAVE_DEEPD3_OVERVIEW_PNGS = True
# Prediction stacks: {base_name}_S_shaft.tif, _S_spine.tif
SAVE_DEEPD3_PREDICTION_TIFFS = True
SKELETON_3D_FOR_ROI_PANEL = True

# Per-mushroom spine files for run_multi_spine_manager: {base_name}_NNN.ini / .png
SAVE_PER_SPINE_INI = False
SAVE_PER_SPINE_PNG = SAVE_PER_SPINE_INI
# Per-flim feature table: {base_name}_mushroom_features.csv
SAVE_MUSHROOM_FEATURES_CSV = True


def print_prediction_diagnostics(S):
    """Console summary to guide threshold tuning."""
    dend = S.prediction[..., 0]
    spine = S.prediction[..., 1]
    for name, arr in ("dendrite(shaft)", dend), ("spine", spine):
        print(
            f"  pred {name}: min={arr.min():.3f} max={arr.max():.3f} "
            f"mean={arr.mean():.3f} frac>0.2={np.mean(arr > 0.2):.3f} "
            f"frac>0.1={np.mean(arr > 0.1):.3f}"
        )


def save_shaft_raw_mip_compare_png(
    raw_mip,
    dendrite_prediction_zyx,
    savepath,
    dendrite_fused_zyx=None,
    raw_vmax_percentile=RAW_MIP_VMAX_PERCENTILE,
    shaft_contour_level=SHAFT_PRED_CONTOUR_LEVEL,
):
    """Raw Z-MIP | DeepD3 shaft pred | fused shaft pred (thin-branch enhance)."""
    raw_mip = np.asarray(raw_mip, dtype=np.float32)
    shaft_raw_mip = np.max(dendrite_prediction_zyx, axis=0)
    show_fused = dendrite_fused_zyx is not None
    if show_fused:
        shaft_fused_mip = np.max(dendrite_fused_zyx, axis=0)
    if np.any(raw_mip > 0):
        vmax_raw = float(np.percentile(raw_mip[raw_mip > 0], raw_vmax_percentile))
    else:
        vmax_raw = float(raw_mip.max()) if raw_mip.size else 1.0
    vmax_raw = max(vmax_raw, 1e-6)

    ncols = 3 if show_fused else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 2:
        axes = [axes[0], axes[1]]

    axes[0].imshow(raw_mip, cmap="gray", vmin=0, vmax=vmax_raw)
    axes[0].set_title(f"Raw Z-MIP (p{raw_vmax_percentile})")
    axes[0].axis("off")

    axes[1].imshow(shaft_raw_mip, cmap="gray", vmin=0, vmax=1.0)
    if shaft_contour_level > 0:
        axes[1].contour(
            shaft_raw_mip, levels=[shaft_contour_level], colors="cyan", linewidths=0.5
        )
    axes[1].set_title("DeepD3 shaft pred (raw)")
    axes[1].axis("off")

    if show_fused:
        axes[2].imshow(shaft_fused_mip, cmap="gray", vmin=0, vmax=1.0)
        if shaft_contour_level > 0:
            axes[2].contour(
                shaft_fused_mip,
                levels=[shaft_contour_level],
                colors="lime",
                linewidths=0.5,
            )
        axes[2].set_title("Shaft pred + image fusion")
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  save as", savepath)


def apply_permissive_deepd3_params(spine_assign, min_shaft_to_head_um):
    """Apply broad ROI / skeleton thresholds (see DEEPD3_ROI_PARAMS at top of file)."""
    for key, value in DEEPD3_ROI_PARAMS.items():
        spine_assign.params[key] = value
    spine_assign.params["max_dist_spine_dend_um"] = max(
        min_shaft_to_head_um + 2.0,
        MAX_DIST_SPINE_DEND_UM,
    )
    spine_assign.params["use_local_z_mip"] = USE_LOCAL_Z_MIP
    spine_assign.params["local_z_mip_radius"] = LOCAL_Z_MIP_RADIUS
    spine_assign.params["enhance_thin_branches"] = ENHANCE_THIN_BRANCHES
    spine_assign.params["stack_preprocess"] = STACK_PREPROCESS
    spine_assign.params["image_fusion_percentile"] = IMAGE_FUSION_PERCENTILE
    spine_assign.params["image_fusion_weight"] = IMAGE_FUSION_WEIGHT
    spine_assign.params["dendrite_closing_iterations"] = DENDRITE_CLOSING_ITERATIONS
    spine_assign.params["roi_mode"] = DEEPD3_ROI_MODE
    z_mip_note = (
        f"local_z_mip r={LOCAL_Z_MIP_RADIUS}"
        if USE_LOCAL_Z_MIP
        else "local_z_mip off"
    )
    print(
        "  DeepD3 params:",
        f"roi_mode={spine_assign.params['roi_mode']},",
        f"peak={spine_assign.params['roi_peakThreshold']},",
        f"area={spine_assign.params['roi_areaThreshold']},",
        f"seed_delta={spine_assign.params['roi_seedDelta']},",
        f"dist_seed={spine_assign.params['roi_distanceToSeed']},",
        f"skeleton_thresh={spine_assign.params['dendrite_skeleton_threshold']},",
        f"max_dist={spine_assign.params['max_dist_spine_dend_um']:.1f} um,",
        f"{z_mip_note}, preprocess={STACK_PREPROCESS},",
        f"thin_branch={ENHANCE_THIN_BRANCHES}",
    )


def savefolder_from_flim_path(flim_path):
    """Same convention as multi_spine savefolder_func (with SMB fallback)."""
    return resolve_savefolder(flim_path)


def base_name_from_flim_path(flim_path):
    """Same convention as ongoing/ASIcontroller/multi_spine_for_eachfile.base_name_func."""
    return os.path.basename(flim_path[:-9])


def shaft_to_head_distance_um(entry, xy_pixel_um):
    """
    Shaft-to-head distance in the XY plane only (um). Z is not included.
    Uses the skeleton neighborhood point closest to the head in XY.
    """
    head_x = float(entry["centroid_x_pix"])
    head_y = float(entry["centroid_y_pix"])
    neighborhood = np.asarray(entry["neighborhood_points"])
    if neighborhood.ndim == 2 and neighborhood.shape[1] == 3:
        shaft_ys = neighborhood[:, 1]
        shaft_xs = neighborhood[:, 2]
    else:
        shaft_ys = neighborhood[:, 0]
        shaft_xs = neighborhood[:, 1]
    dx_um = (head_x - shaft_xs) * xy_pixel_um
    dy_um = (head_y - shaft_ys) * xy_pixel_um
    return float(np.min(np.hypot(dx_um, dy_um)))


def dend_slope_intercept_from_neighborhood(coordinates):
    coordinates = np.asarray(coordinates)
    if coordinates.shape[1] == 2:
        return np.polyfit(coordinates[:, 1], coordinates[:, 0], 1)
    if coordinates.shape[1] == 3:
        return np.polyfit(coordinates[:, 2], coordinates[:, 1], 1)
    raise ValueError(f"Unexpected neighborhood_points shape: {coordinates.shape}")


def filter_mushroom_spines(result_dict, xy_pixel_um, min_shaft_to_head_um):
    """Return {label: {entry, shaft_to_head_um}} for mushroom spines only."""
    mushrooms = {}
    for label, entry in result_dict.items():
        dist_um = shaft_to_head_distance_um(entry, xy_pixel_um)
        if dist_um >= min_shaft_to_head_um:
            mushrooms[label] = {"entry": entry, "shaft_to_head_um": dist_um}
        else:
            print(f"  skip label {label}: shaft-to-head XY {dist_um:.3f} um "
                  f"(< {min_shaft_to_head_um} um)")
    return mushrooms


def dedupe_mushrooms_by_xy(
    mushrooms: dict,
    xy_pixel_um: float,
    min_sep_um: float,
) -> dict:
    """
    Keep one mushroom per XY cluster (greedy NMS on head centroid).

    Floodfill ROI mode often yields several labels on one physical spine; roi_mip_sweep
    only colors ROIs and does not save one marker per label, so duplicates are less obvious.
    """
    if min_sep_um <= 0 or len(mushrooms) <= 1:
        return mushrooms

    ranked = sorted(
        mushrooms.items(),
        key=lambda kv: (
            kv[1]["shaft_to_head_um"],
            kv[1]["entry"].get("equivalent_diameter_area", 0),
        ),
        reverse=True,
    )
    kept: dict = {}
    kept_xy_um: list[tuple[float, float]] = []

    for label, info in ranked:
        entry = info["entry"]
        x_um = float(entry["centroid_x_pix"]) * xy_pixel_um
        y_um = float(entry["centroid_y_pix"]) * xy_pixel_um
        if any(
            np.hypot(x_um - kx, y_um - ky) < min_sep_um for kx, ky in kept_xy_um
        ):
            print(
                f"  dedupe skip label {label}: head within {min_sep_um} um "
                f"of a kept mushroom"
            )
            continue
        kept[label] = info
        kept_xy_um.append((x_um, y_um))
    return kept


def _dendrite_line_xy(mip_img, dend_slope, dend_intercept):
    """Return x/y arrays for the fitted dendrite shaft line within image bounds."""
    dend_x, dend_y = [], []
    for x in range(1, mip_img.shape[1] - 1):
        y = dend_slope * x + dend_intercept
        if (1 < y) and (y < mip_img.shape[0] - 1):
            dend_x.append(x)
            dend_y.append(y)
    return dend_x, dend_y


SPINE_CONTOUR_SPINE_THRESHOLD = 0.15
SPINE_CONTOUR_SHAFT_THRESHOLD = 0.10
SPINE_CONTOUR_Z_HALF_WINDOW = 2
SHAFT_ROI_RECT_LENGTH_PX = 10
SHAFT_ROI_RECT_HEIGHT_PX = 2
SHAFT_BIN_DENDRITE_THRESHOLD = SPINE_CONTOUR_SHAFT_THRESHOLD


def _nearest_foreground_point(mask, y_pix, x_pix, max_radius=8):
    """Return the closest foreground pixel to (y, x) within max_radius."""
    ny, nx = mask.shape
    best_dist = max_radius + 1
    best_yx = None
    y0 = max(0, y_pix - max_radius)
    y1 = min(ny, y_pix + max_radius + 1)
    x0 = max(0, x_pix - max_radius)
    x1 = min(nx, x_pix + max_radius + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if not mask[yy, xx]:
                continue
            dist = np.hypot(yy - y_pix, xx - x_pix)
            if dist < best_dist:
                best_dist = dist
                best_yx = (yy, xx)
    return best_yx


def _spine_full_mask_2d_from_prediction(
    spine_prediction,
    dendrite_prediction,
    spine_zyx,
    *,
    spine_threshold=SPINE_CONTOUR_SPINE_THRESHOLD,
    shaft_threshold=SPINE_CONTOUR_SHAFT_THRESHOLD,
    z_half_window=SPINE_CONTOUR_Z_HALF_WINDOW,
):
    """
    Segment one full spine (head + neck) from DeepD3 prediction maps.

    Uses the same spine-minus-shaft + watershed strategy as
    gui_roi_analysis/make_spine_roi_based_on_deepd3.py, but keeps only the
    watershed basin seeded at the detected head position.
    """
    z_pix = int(round(spine_zyx[0]))
    y_pix = int(round(spine_zyx[1]))
    x_pix = int(round(spine_zyx[2]))
    spine_prediction = np.asarray(spine_prediction)
    dendrite_prediction = np.asarray(dendrite_prediction)
    ny, nx = spine_prediction.shape[1], spine_prediction.shape[2]
    empty = np.zeros((ny, nx), dtype=bool)

    if not (0 <= y_pix < ny and 0 <= x_pix < nx):
        return empty

    z0 = max(0, z_pix - z_half_window)
    z1 = min(spine_prediction.shape[0], z_pix + z_half_window + 1)
    spine_mip = np.max(spine_prediction[z0:z1], axis=0)
    shaft_mip = np.max(dendrite_prediction[z0:z1], axis=0)

    spine_mask = spine_mip > spine_threshold
    shaft_mask = shaft_mip > shaft_threshold
    spine_mask = binary_closing(binary_opening(spine_mask, disk(1)), disk(2))
    shaft_mask = binary_closing(binary_opening(shaft_mask, disk(1)), disk(2))
    spine_minus_shaft = spine_mask & ~shaft_mask
    if not spine_minus_shaft.any():
        return empty

    seed_y, seed_x = y_pix, x_pix
    if not spine_minus_shaft[seed_y, seed_x]:
        nearest = _nearest_foreground_point(spine_minus_shaft, y_pix, x_pix)
        if nearest is None:
            return empty
        seed_y, seed_x = nearest

    distance = ndi.distance_transform_edt(spine_minus_shaft)
    markers = np.zeros(spine_minus_shaft.shape, dtype=np.int32)
    markers[seed_y, seed_x] = 1
    spine_seg = watershed(
        -distance,
        markers,
        mask=spine_minus_shaft,
        connectivity=2,
        compactness=0,
    )
    return spine_seg == 1


def save_spine_overlay_png(
    mip_img,
    spine_zyx,
    dend_slope,
    dend_intercept,
    savepath,
    pix_size=512,
    spine_prediction=None,
    dendrite_prediction=None,
):
    """
    Save a side-by-side PNG: raw MIP | MIP with shaft line, head, and spine contour.

    Left panel: raw MIP only.
    Right panel: MIP + green dendrite line + red head + cyan full-spine outline.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    for ax in axs:
        ax.imshow(mip_img, cmap="gray")
        ax.axis("off")

    dend_x, dend_y = _dendrite_line_xy(mip_img, dend_slope, dend_intercept)
    axs[1].scatter(dend_x, dend_y, c="g", s=2)
    axs[1].scatter(spine_zyx[2], spine_zyx[1], c="r", s=4)

    if spine_prediction is not None and dendrite_prediction is not None:
        mask_2d = _spine_full_mask_2d_from_prediction(
            spine_prediction,
            dendrite_prediction,
            spine_zyx,
        )
        if mask_2d.any():
            for contour in find_contours(mask_2d.astype(np.float32), 0.5):
                axs[1].plot(contour[:, 1], contour[:, 0], c="c", lw=0.8)

    fig.savefig(
        savepath,
        bbox_inches="tight",
        pad_inches=0,
        dpi=int(pix_size / 4 * 1.3264),
    )
    plt.close(fig)


def _prefix_dict(data: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in data.items()}


def _shaft_rectangle_mask_2d(
    head_x: float,
    head_y: float,
    dend_slope: float,
    dend_intercept: float,
    image_shape: tuple[int, int],
    rect_length: float = SHAFT_ROI_RECT_LENGTH_PX,
    rect_height: float = SHAFT_ROI_RECT_HEIGHT_PX,
) -> tuple[np.ndarray, float, float]:
    """DendriticShaft-style rectangle on the fitted dendrite line (calc_spine_dend_GCaMP)."""
    y_c, x_c = calc_point_on_line_close_to_xy(head_x, head_y, dend_slope, dend_intercept)
    theta = np.arctan(dend_slope)
    dx = (rect_length / 2) * np.cos(theta)
    dy = (rect_length / 2) * np.sin(theta)
    px = (rect_height / 2) * -np.sin(theta)
    py = (rect_height / 2) * np.cos(theta)
    corners_x = [x_c - dx - px, x_c - dx + px, x_c + dx + px, x_c + dx - px]
    corners_y = [y_c - dy - py, y_c - dy + py, y_c + dy + py, y_c + dy - py]
    rr_rect, cc_rect = polygon(corners_y, corners_x, shape=image_shape)
    mask = np.zeros(image_shape, dtype=bool)
    mask[rr_rect, cc_rect] = True
    return mask, float(y_c), float(x_c)


def _shaft_binary_mask_from_prediction(
    dendrite_prediction: np.ndarray,
    shaft_rect_mask: np.ndarray,
    z_pix: int,
    *,
    dendrite_threshold: float = SHAFT_BIN_DENDRITE_THRESHOLD,
    exclude_mask: np.ndarray | None = None,
    z_half_window: int = SPINE_CONTOUR_Z_HALF_WINDOW,
) -> np.ndarray:
    """Binarized dendrite prediction within the shaft rectangle, excluding spine if given."""
    z0 = max(0, z_pix - z_half_window)
    z1 = min(dendrite_prediction.shape[0], z_pix + z_half_window + 1)
    dend_mip = np.max(np.asarray(dendrite_prediction)[z0:z1], axis=0)
    mask = (dend_mip >= dendrite_threshold) & shaft_rect_mask
    if exclude_mask is not None:
        mask &= ~np.asarray(exclude_mask, dtype=bool)
    if not mask.any():
        return mask

    labeled, n_labels = ndimage_label(mask)
    if n_labels <= 1:
        return mask

    # Keep the connected component with the largest overlap inside the shaft rectangle.
    best_label = 1 + int(np.argmax([np.sum(labeled == i) for i in range(1, n_labels + 1)]))
    return labeled == best_label


def _mask_raw_intensity_stats(mip_img: np.ndarray, mask_2d: np.ndarray) -> dict:
    """Raw-image intensity stats and whole-image percentile rank of the mask mean."""
    if not mask_2d.any():
        return {}

    vals = np.asarray(mip_img)[mask_2d].astype(np.float64)
    mean_val = float(vals.mean())
    image_vals = np.asarray(mip_img, dtype=np.float64).ravel()
    return {
        "raw_intensity_mean": mean_val,
        "raw_intensity_median": float(np.median(vals)),
        "raw_intensity_std": float(vals.std()),
        "raw_intensity_max": float(vals.max()),
        "raw_intensity_percentile_image": float(
            percentileofscore(image_vals, mean_val, kind="weak")
        ),
    }


def _prediction_stats_in_mask_simple(
    spine_mip: np.ndarray,
    dend_mip: np.ndarray,
    mask_2d: np.ndarray,
) -> dict:
    if not mask_2d.any():
        return {}
    spine_vals = spine_mip[mask_2d]
    dend_vals = dend_mip[mask_2d]
    return {
        "spine_pred_mean": float(spine_vals.mean()),
        "spine_pred_max": float(spine_vals.max()),
        "dendrite_pred_mean": float(dend_vals.mean()),
        "dendrite_pred_max": float(dend_vals.max()),
    }


def _skeleton_branch_length_um(neighborhood_points, xy_pixel_um: float, z_pixel_um: float) -> float:
    """Polyline length of the local dendrite skeleton branch paired to the spine."""
    pts = np.asarray(neighborhood_points)
    if pts.shape[0] < 2:
        return 0.0
    if pts.shape[1] == 3:
        scaled = pts.astype(np.float64) * np.array([z_pixel_um, xy_pixel_um, xy_pixel_um])
    else:
        scaled = pts.astype(np.float64) * np.array([xy_pixel_um, xy_pixel_um])
    diffs = np.diff(scaled, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _regionprops_metrics(mask_2d: np.ndarray, xy_pixel_um: float, prefix: str = "seg") -> dict:
    """Shape metrics from a 2D binary mask."""
    labeled, n_labels = ndimage_label(mask_2d)
    if n_labels < 1:
        return {}

    prop = regionprops(labeled)[0]
    xy2 = xy_pixel_um ** 2
    return {
        f"{prefix}_area_px": int(prop.area),
        f"{prefix}_area_um2": float(prop.area * xy2),
        f"{prefix}_perimeter_px": float(prop.perimeter),
        f"{prefix}_perimeter_um": float(prop.perimeter * xy_pixel_um),
        f"{prefix}_major_axis_px": float(prop.major_axis_length),
        f"{prefix}_major_axis_um": float(prop.major_axis_length * xy_pixel_um),
        f"{prefix}_minor_axis_px": float(prop.minor_axis_length),
        f"{prefix}_minor_axis_um": float(prop.minor_axis_length * xy_pixel_um),
        f"{prefix}_equivalent_diameter_px": float(prop.equivalent_diameter_area),
        f"{prefix}_equivalent_diameter_um": float(prop.equivalent_diameter_area * xy_pixel_um),
        f"{prefix}_eccentricity": float(prop.eccentricity),
        f"{prefix}_solidity": float(prop.solidity),
        f"{prefix}_extent": float(prop.extent),
        f"{prefix}_orientation_rad": float(prop.orientation),
        f"{prefix}_bbox_min_row": int(prop.bbox[0]),
        f"{prefix}_bbox_min_col": int(prop.bbox[1]),
        f"{prefix}_bbox_max_row": int(prop.bbox[2]),
        f"{prefix}_bbox_max_col": int(prop.bbox[3]),
    }


def _prediction_stats_in_mask(
    spine_prediction: np.ndarray,
    dendrite_prediction: np.ndarray,
    mask_2d: np.ndarray,
    z_pix: int,
    z_half_window: int = SPINE_CONTOUR_Z_HALF_WINDOW,
) -> dict:
    """Spine/dendrite prediction statistics inside the segmented mask."""
    if not mask_2d.any():
        return {}

    z0 = max(0, z_pix - z_half_window)
    z1 = min(spine_prediction.shape[0], z_pix + z_half_window + 1)
    spine_mip = np.max(spine_prediction[z0:z1], axis=0)
    dend_mip = np.max(dendrite_prediction[z0:z1], axis=0)
    spine_vals = spine_mip[mask_2d]
    dend_vals = dend_mip[mask_2d]
    return {
        "seg_spine_pred_mean": float(spine_vals.mean()),
        "seg_spine_pred_max": float(spine_vals.max()),
        "seg_spine_pred_min": float(spine_vals.min()),
        "seg_dendrite_pred_mean": float(dend_vals.mean()),
        "seg_dendrite_pred_max": float(dend_vals.max()),
    }


def build_mushroom_spine_feature_row(
    *,
    flim_path: str,
    base_name: str,
    spine_index: int,
    deepd3_label,
    mushroom_info: dict,
    outputs_context: dict,
    xy_pixel_um: float,
    z_pixel_um: float,
    min_shaft_to_head_um: float,
    n_roi_all: int,
    n_uncaging_candidates: int,
    n_mushroom_before_dedupe: int,
    n_mushroom_after_dedupe: int,
    detection_params: dict | None = None,
    mip_img: np.ndarray | None = None,
    png_path: str = "",
    ini_path: str = "",
) -> dict:
    """Collect shape, distance, ROI, and prediction features for one mushroom spine."""
    entry = mushroom_info["entry"]
    label_key = str(deepd3_label)
    prop_dict = outputs_context["prop_dict"]
    cand_spines = outputs_context["cand_spines"]
    roi_map = outputs_context["r"].roi_map
    spine_prediction = outputs_context["S"].prediction[..., 1]
    dendrite_prediction = outputs_context["S"].prediction[..., 0]

    head_z = float(entry["z_pix"])
    head_y = float(entry["centroid_y_pix"])
    head_x = float(entry["centroid_x_pix"])
    uncaging_y = float(entry.get("y_pix", head_y))
    uncaging_x = float(entry.get("x_pix", head_x))
    dend_slope, dend_intercept = dend_slope_intercept_from_neighborhood(
        entry["neighborhood_points"]
    )

    row = {
        "flim_path": flim_path,
        "base_name": base_name,
        "spine_index": int(spine_index),
        "deepd3_label": label_key,
        "png_path": png_path,
        "ini_path": ini_path,
        "xy_pixel_um": float(xy_pixel_um),
        "z_pixel_um": float(z_pixel_um),
        "min_shaft_to_head_um": float(min_shaft_to_head_um),
        "n_roi_all": int(n_roi_all),
        "n_uncaging_candidates": int(n_uncaging_candidates),
        "n_mushroom_before_dedupe": int(n_mushroom_before_dedupe),
        "n_mushroom_after_dedupe": int(n_mushroom_after_dedupe),
        "head_z_pix": head_z,
        "head_y_pix": head_y,
        "head_x_pix": head_x,
        "uncaging_y_pix": uncaging_y,
        "uncaging_x_pix": uncaging_x,
        "shaft_to_head_um": float(mushroom_info["shaft_to_head_um"]),
        "head_to_uncaging_um": float(
            np.hypot((head_x - uncaging_x) * xy_pixel_um, (head_y - uncaging_y) * xy_pixel_um)
        ),
        "dend_slope": float(dend_slope),
        "dend_intercept": float(dend_intercept),
        "shaft_orientation_rad": float(entry.get("orientation", np.nan)),
        "trace_direction": float(entry.get("direction", np.nan)),
        "floodfill_equivalent_diameter_px": float(
            entry.get("equivalent_diameter_area", np.nan)
        ),
        "floodfill_equivalent_diameter_um": float(
            entry.get("equivalent_diameter_area", np.nan) * xy_pixel_um
        ),
    }

    if label_key in prop_dict:
        prop = prop_dict[label_key]
        row.update({
            "floodfill_roi_num_pixels": int(prop["num_pixels"]),
            "floodfill_roi_intensity_sum": float(prop["intensity"]),
            "floodfill_roi_z_pix": float(prop["z"]),
            "floodfill_roi_y_pix": float(prop["y"]),
            "floodfill_roi_x_pix": float(prop["x"]),
        })

    label_id = int(label_key)
    floodfill_mask_3d = np.asarray(roi_map) == label_id
    row["floodfill_roi_voxels_3d"] = int(floodfill_mask_3d.sum())
    row["floodfill_roi_proj_px_2d"] = int(np.any(floodfill_mask_3d, axis=0).sum())

    if label_key in cand_spines.index:
        cand = cand_spines.loc[label_key]
        row["nearest_neighbor_distance_um"] = float(cand.get("distance_to_nearest", np.nan))
        row["nearest_neighbor_label"] = str(cand.get("nearest_point_label", ""))

    z_pix = int(round(head_z))
    y_pix = int(round(head_y))
    x_pix = int(round(head_x))
    if (
        0 <= z_pix < spine_prediction.shape[0]
        and 0 <= y_pix < spine_prediction.shape[1]
        and 0 <= x_pix < spine_prediction.shape[2]
    ):
        row["head_spine_pred"] = float(spine_prediction[z_pix, y_pix, x_pix])
        row["head_dendrite_pred"] = float(dendrite_prediction[z_pix, y_pix, x_pix])

    spine_zyx = [head_z, head_y, head_x]
    seg_mask = _spine_full_mask_2d_from_prediction(
        spine_prediction,
        dendrite_prediction,
        spine_zyx,
    )
    row["seg_mask_found"] = bool(seg_mask.any())
    shape_metrics = _regionprops_metrics(seg_mask, xy_pixel_um)
    row.update(shape_metrics)
    if shape_metrics.get("seg_minor_axis_um", 0) > 0:
        row["seg_aspect_ratio"] = float(
            shape_metrics["seg_major_axis_um"] / shape_metrics["seg_minor_axis_um"]
        )
    row.update(
        _prediction_stats_in_mask(
            spine_prediction,
            dendrite_prediction,
            seg_mask,
            z_pix,
        )
    )

    image_shape = (spine_prediction.shape[1], spine_prediction.shape[2])
    shaft_rect_mask, shaft_anchor_y, shaft_anchor_x = _shaft_rectangle_mask_2d(
        head_x,
        head_y,
        dend_slope,
        dend_intercept,
        image_shape,
    )
    row["shaft_anchor_y_pix"] = shaft_anchor_y
    row["shaft_anchor_x_pix"] = shaft_anchor_x
    row["shaft_skeleton_branch_length_um"] = _skeleton_branch_length_um(
        entry["neighborhood_points"],
        xy_pixel_um,
        z_pixel_um,
    )
    row.update(_regionprops_metrics(shaft_rect_mask, xy_pixel_um, prefix="shaft_roi"))
    if row.get("shaft_roi_minor_axis_um", 0) > 0:
        row["shaft_roi_aspect_ratio"] = float(
            row["shaft_roi_major_axis_um"] / row["shaft_roi_minor_axis_um"]
        )

    z0 = max(0, z_pix - SPINE_CONTOUR_Z_HALF_WINDOW)
    z1 = min(spine_prediction.shape[0], z_pix + SPINE_CONTOUR_Z_HALF_WINDOW + 1)
    spine_mip = np.max(spine_prediction[z0:z1], axis=0)
    dend_mip = np.max(dendrite_prediction[z0:z1], axis=0)

    row.update(_prefix_dict(_prediction_stats_in_mask_simple(spine_mip, dend_mip, shaft_rect_mask), "shaft_roi"))
    shaft_bin_mask = _shaft_binary_mask_from_prediction(
        dendrite_prediction,
        shaft_rect_mask,
        z_pix,
        exclude_mask=seg_mask if seg_mask.any() else None,
    )
    row["shaft_bin_mask_found"] = bool(shaft_bin_mask.any())
    row.update(_regionprops_metrics(shaft_bin_mask, xy_pixel_um, prefix="shaft_bin"))
    if row.get("shaft_bin_minor_axis_um", 0) > 0:
        row["shaft_bin_aspect_ratio"] = float(
            row["shaft_bin_major_axis_um"] / row["shaft_bin_minor_axis_um"]
        )
    row.update(_prefix_dict(_prediction_stats_in_mask_simple(spine_mip, dend_mip, shaft_bin_mask), "shaft_bin"))

    if mip_img is not None:
        row.update(_prefix_dict(_mask_raw_intensity_stats(mip_img, shaft_rect_mask), "shaft_roi"))
        row.update(_prefix_dict(_mask_raw_intensity_stats(mip_img, shaft_bin_mask), "shaft_bin"))

    if detection_params:
        for key, value in detection_params.items():
            row[f"param_{key}"] = value
    return row


def save_mushroom_features_csv(savefolder: str, base_name: str, feature_rows: list[dict]) -> str:
    """Write per-flim mushroom feature table to CSV."""
    if not feature_rows:
        return ""
    df = pd.DataFrame(feature_rows)
    out_path = os.path.join(savefolder, f"{base_name}_mushroom_features.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  saved mushroom features CSV ({len(df)} rows): {out_path}")
    return out_path


def get_next_ini_indices(savefolder, base_name, count):
    """Allocate consecutive 3-digit indices compatible with multi_spine manager."""
    existing_inilist = glob.glob(os.path.join(savefolder, base_name + "*.ini"))
    existing_indices = []
    for inipath in existing_inilist:
        match = re.search(r"_(\d{3})\.ini$", os.path.basename(inipath))
        if match:
            existing_indices.append(int(match.group(1)))
    start = max(existing_indices) + 1 if existing_indices else 0
    return [start + i for i in range(count)]


def verify_outputs_for_spine_manager(
    flim_path,
    inipaths,
    *,
    require_png: bool = True,
):
    """
    Check that saved files match what define_multiple_files_spine_manager expects.
    """
    savefolder = savefolder_from_flim_path(flim_path)
    base_name = base_name_from_flim_path(flim_path)
    expected_prefix = os.path.join(savefolder, base_name + "_")
    issues = []

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

        if os.path.exists(inipath):
            try:
                spine_zyx, dend_slope, dend_intercept, excluded = read_xyz_single(
                    inipath, return_excluded=True
                )
            except Exception as exc:
                issues.append(f"read_xyz_single failed for {basename}: {exc}")
                continue

            if excluded:
                issues.append(f"ini marked excluded: {basename}")
            if spine_zyx[0] < 0:
                issues.append(f"invalid spine z: {basename}")

    manager_pattern = os.path.join(savefolder, base_name + "*.ini")
    found = glob.glob(manager_pattern)
    if len(found) < len(inipaths):
        issues.append(
            f"glob {manager_pattern!r} found {len(found)} ini, expected >= {len(inipaths)}"
        )

    return issues


def verify_deepd3_overview_outputs(savefolder, base_name):
    """Check per-file DeepD3 overview PNGs and optional prediction TIFFs."""
    issues = []
    expected = [
        os.path.join(savefolder, f"{base_name}_mip.png"),
        os.path.join(savefolder, f"{base_name}_roi.png"),
    ]
    if SAVE_DEEPD3_OVERVIEW_PNGS:
        for path in expected:
            if not os.path.exists(path):
                issues.append(f"missing overview: {path}")
    if SAVE_DEEPD3_PREDICTION_TIFFS:
        for suffix in ("_S_shaft.tif", "_S_spine.tif"):
            path = os.path.join(savefolder, f"{base_name}{suffix}")
            if not os.path.exists(path):
                issues.append(f"missing prediction stack: {path}")
    return issues


def extract_prefix_and_number(file_path):
    match = re.match(r"(.*)_(\d{3})\.flim$", file_path)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def get_first_flim_list(highmag_folder, highmag_filename):
    filepath_list = glob.glob(os.path.join(highmag_folder, highmag_filename))
    filtered_files = {}
    for file_path in filepath_list:
        prefix, number = extract_prefix_and_number(file_path)
        if prefix:
            if prefix not in filtered_files or number < filtered_files[prefix][1]:
                filtered_files[prefix] = (file_path, number)
    return [file_info[0] for file_info in filtered_files.values()]


def loop_mushroom_spine_assign_save(
    highmag_folder,
    highmag_filename,
    exclude_ini_saved,
    from_latest_file,
    min_shaft_to_head_um=MIN_SHAFT_TO_HEAD_UM,
    min_seconds_since_modification=DEFAULT_MIN_SECONDS_SINCE_MODIFICATION,
    deepd3_model_path=DEEPD3_MODEL_PATH,
    *,
    dedupe_mushroom_xy_sep_um: float | None = None,
    save_per_spine_ini: bool | None = None,
    save_per_spine_png: bool | None = None,
):
    first_flim_list = get_first_flim_list(highmag_folder, highmag_filename)
    first_flim_list = [
        f for f in first_flim_list
        if not os.path.basename(f).lower().startswith(("for_align", "for_aling"))
    ]
    first_flim_list = sorted(
        first_flim_list,
        key=lambda file: os.path.getmtime(file),
        reverse=from_latest_file,
    )
    if save_per_spine_ini is None:
        save_per_spine_ini = SAVE_PER_SPINE_INI
    if save_per_spine_png is None:
        save_per_spine_png = SAVE_PER_SPINE_PNG
    if dedupe_mushroom_xy_sep_um is None:
        dedupe_mushroom_xy_sep_um = DEDUPE_MUSHROOM_XY_SEP_UM

    print("The number of highmag flim files:", len(first_flim_list))
    print(f"Mushroom threshold: shaft-to-head XY distance >= {min_shaft_to_head_um} um")
    print(f"Mushroom dedupe: min XY separation = {dedupe_mushroom_xy_sep_um} um")
    print(
        "Per-spine outputs:",
        f"ini={save_per_spine_ini}, png={save_per_spine_png}",
    )

    summary_rows = []

    for flim_path in first_flim_list:
        savefolder = savefolder_from_flim_path(flim_path)
        base_name = base_name_from_flim_path(flim_path)
        existing_inilist = glob.glob(os.path.join(savefolder, base_name + "*.ini"))

        if len(existing_inilist) and exclude_ini_saved and save_per_spine_ini:
            print("defined:", flim_path)
            continue

        modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(flim_path))
        delta = (datetime.datetime.now() - modified_date).total_seconds()
        if delta <= min_seconds_since_modification:
            continue

        print("\nProcessing:", flim_path)
        spine_assign = SpinePosDeepD3()
        spine_assign.trainingdata_path = deepd3_model_path
        apply_permissive_deepd3_params(spine_assign, min_shaft_to_head_um)

        iminfo = FileReader()
        iminfo.read_imageFile(flim_path, True)
        imagearray = np.array(iminfo.image)
        mip_img = np.sum(
            np.sum(np.sum(np.sum(imagearray, axis=-1), axis=1), axis=1), axis=0
        )

        result_dict, outputs_context = spine_assign.return_uncaging_pos_based_on_roi_sum(
            flim_path,
            plot_them=False,
            upper_lim_spine_pixel_percentile=UPPER_SPINE_PIXEL_PERCENTILE,
            lower_lim_spine_pixel_percentile=LOWER_SPINE_PIXEL_PERCENTILE,
            upper_lim_spine_intensity_percentile=UPPER_SPINE_INTENSITY_PERCENTILE,
            lower_lim_spine_intensity_percentile=LOWER_SPINE_INTENSITY_PERCENTILE,
            save_prediction_stacks=SAVE_DEEPD3_PREDICTION_TIFFS,
            ignore_edge_percentile=IGNORE_EDGE_PERCENTILE,
            ignore_first_n_plane=IGNORE_PLANES_FIRST_LAST,
            ignore_last_n_plane=IGNORE_PLANES_FIRST_LAST,
            skeleton_3d=SKELETON_3D_FOR_ROI_PANEL,
            define_save_folder=True,
            save_folder=savefolder,
            save_filename_stem=base_name,
            return_outputs_context=True,
        )

        xy_pixel_um = spine_assign.params["xy_pixel_um"]
        print("xy_pixel_um:", xy_pixel_um)
        print_prediction_diagnostics(outputs_context["S"])
        print(
            f"  ROI count (all labels): {len(outputs_context['prop_dict'])} | "
            f"uncaging candidates: {len(result_dict)}"
        )

        if SAVE_SHAFT_RAW_MIP_COMPARE:
            compare_path = os.path.join(
                savefolder, f"{base_name}_shaft_raw_mip_compare.png"
            )
            save_shaft_raw_mip_compare_png(
                mip_img,
                outputs_context.get(
                    "dendrite_pred_raw",
                    outputs_context["S"].prediction[..., 0],
                ),
                compare_path,
                dendrite_fused_zyx=outputs_context.get("dendrite_pred_fused"),
            )

        mushrooms = filter_mushroom_spines(result_dict, xy_pixel_um, min_shaft_to_head_um)
        n_mushroom_before_dedupe = len(mushrooms)
        mushrooms = dedupe_mushrooms_by_xy(
            mushrooms, xy_pixel_um, dedupe_mushroom_xy_sep_um
        )
        if n_mushroom_before_dedupe != len(mushrooms):
            print(
                f"  mushroom dedupe: {n_mushroom_before_dedupe} -> {len(mushrooms)} "
                f"(removed {n_mushroom_before_dedupe - len(mushrooms)} near-duplicate ROI labels)"
            )
        mushroom_result_dict = {
            label: info["entry"] for label, info in mushrooms.items()
        }

        if SAVE_DEEPD3_OVERVIEW_PNGS or SAVE_DEEPD3_PREDICTION_TIFFS:
            spine_assign.save_detection_outputs(
                outputs_context["S"],
                outputs_context["r"],
                outputs_context["prop_dict"],
                outputs_context["cand_spines"],
                outputs_context["skeleton"],
                mushroom_result_dict,
                savefolder,
                base_name,
                save_overview_pngs=SAVE_DEEPD3_OVERVIEW_PNGS,
                save_prediction_stacks=SAVE_DEEPD3_PREDICTION_TIFFS,
                skeleton_3d=SKELETON_3D_FOR_ROI_PANEL,
                show_interactive=False,
            )
            overview_issues = verify_deepd3_overview_outputs(savefolder, base_name)
            if overview_issues:
                print("  DeepD3 overview issues:")
                for issue in overview_issues:
                    print("   -", issue)
            else:
                print(
                    f"  saved DeepD3 overview for {base_name} "
                    f"({len(mushroom_result_dict)} mushroom marker(s) on MIP)"
                )

        z_pixel_um = spine_assign.params["z_pixel_um"]
        n_mushroom_after_dedupe = len(mushrooms)

        if not mushrooms:
            print("  no mushroom spines detected")
            summary_rows.append({
                "flim_path": flim_path,
                "n_roi_all": len(outputs_context["prop_dict"]),
                "n_uncaging_candidates": len(result_dict),
                "n_mushroom_detected": 0,
                "n_mushroom_saved": 0,
            })
            continue

        ini_indices = (
            get_next_ini_indices(savefolder, base_name, len(mushrooms))
            if save_per_spine_ini
            else list(range(len(mushrooms)))
        )
        saved_inipaths = []
        n_saved = 0
        feature_rows = []
        detection_params = {
            key: spine_assign.params.get(key)
            for key in (
                "roi_mode",
                "roi_peakThreshold",
                "roi_areaThreshold",
                "roi_seedDelta",
                "roi_distanceToSeed",
                "min_roi_size",
                "max_roi_size",
                "min_planes",
                "dendrite_skeleton_threshold",
                "max_dist_spine_dend_um",
                "use_local_z_mip",
                "local_z_mip_radius",
                "stack_preprocess",
                "enhance_thin_branches",
            )
        }
        detection_params["seg_spine_threshold"] = SPINE_CONTOUR_SPINE_THRESHOLD
        detection_params["seg_shaft_threshold"] = SPINE_CONTOUR_SHAFT_THRESHOLD
        detection_params["seg_z_half_window"] = SPINE_CONTOUR_Z_HALF_WINDOW
        detection_params["shaft_roi_rect_length_px"] = SHAFT_ROI_RECT_LENGTH_PX
        detection_params["shaft_roi_rect_height_px"] = SHAFT_ROI_RECT_HEIGHT_PX
        detection_params["shaft_bin_dendrite_threshold"] = SHAFT_BIN_DENDRITE_THRESHOLD
        detection_params["dedupe_mushroom_xy_sep_um"] = dedupe_mushroom_xy_sep_um

        for spine_idx, ((label, mushroom_info), ini_idx) in enumerate(
            zip(mushrooms.items(), ini_indices)
        ):
            entry = mushroom_info["entry"]
            dist_um = mushroom_info["shaft_to_head_um"]

            spine_zyx = [
                entry["z_pix"],
                entry["centroid_y_pix"],
                entry["centroid_x_pix"],
            ]
            dend_slope, dend_intercept = dend_slope_intercept_from_neighborhood(
                entry["neighborhood_points"]
            )

            inipath = os.path.join(
                savefolder, f"{base_name}_{str(ini_idx).zfill(3)}.ini"
            )
            pngpath = inipath[:-4] + ".png"
            saved_parts = []

            if save_per_spine_png:
                save_spine_overlay_png(
                    mip_img,
                    spine_zyx,
                    dend_slope,
                    dend_intercept,
                    savepath=pngpath,
                    spine_prediction=outputs_context["S"].prediction[..., 1],
                    dendrite_prediction=outputs_context["S"].prediction[..., 0],
                )
                saved_parts.append("png")
            if save_per_spine_ini:
                save_spine_dend_info(spine_zyx, dend_slope, dend_intercept, inipath)
                saved_inipaths.append(inipath)
                saved_parts.append("ini")
            if saved_parts:
                n_saved += 1
                print(
                    f"  saved mushroom {','.join(saved_parts)} "
                    f"{os.path.basename(inipath if save_per_spine_ini else pngpath)} "
                    f"(DeepD3 label {label}, shaft-to-head XY {dist_um:.3f} um)"
                )

            feature_row = build_mushroom_spine_feature_row(
                flim_path=flim_path,
                base_name=base_name,
                spine_index=spine_idx,
                deepd3_label=label,
                mushroom_info=mushroom_info,
                outputs_context=outputs_context,
                xy_pixel_um=xy_pixel_um,
                z_pixel_um=z_pixel_um,
                min_shaft_to_head_um=min_shaft_to_head_um,
                n_roi_all=len(outputs_context["prop_dict"]),
                n_uncaging_candidates=len(result_dict),
                n_mushroom_before_dedupe=n_mushroom_before_dedupe,
                n_mushroom_after_dedupe=n_mushroom_after_dedupe,
                detection_params=detection_params,
                mip_img=mip_img,
                png_path=pngpath if save_per_spine_png else "",
                ini_path=inipath if save_per_spine_ini else "",
            )
            feature_rows.append(feature_row)
            summary_rows.append(feature_row)

        if SAVE_MUSHROOM_FEATURES_CSV:
            save_mushroom_features_csv(savefolder, base_name, feature_rows)

        if not save_per_spine_ini and not save_per_spine_png:
            print(
                f"  {len(mushrooms)} mushroom(s) detected; "
                "per-spine ini/png skipped (SAVE_PER_SPINE_* = False)"
            )

        if save_per_spine_ini:
            issues = verify_outputs_for_spine_manager(
                flim_path,
                saved_inipaths,
                require_png=save_per_spine_png,
            )
            if issues:
                print("  VERIFICATION ISSUES:")
                for issue in issues:
                    print("   -", issue)
            else:
                print(
                    f"  verified {len(saved_inipaths)} ini"
                    f"{'/png pair(s)' if save_per_spine_png else '(s)'} for "
                    "run_multi_spine_manager"
                )
        elif n_saved:
            print(f"  saved {n_saved} png(s); ini skipped (no verification)")
        else:
            print(f"  saved feature table for {len(feature_rows)} mushroom(s)")

    summary_csv = os.path.join(
        highmag_folder, "mushroom_spine_assign_summary.csv"
    )
    if summary_rows:
        fieldnames = sorted({key for row in summary_rows for key in row})
        with open(summary_csv, "w", newline="", encoding="utf-8") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print("\nSummary CSV:", summary_csv)

    return summary_rows


def main():
    loop_mushroom_spine_assign_save(
        highmag_folder=DEFAULT_HIGHMAG_FOLDER,
        highmag_filename=DEFAULT_HIGHMAG_FILENAME,
        exclude_ini_saved=DEFAULT_EXCLUDE_INI_SAVED,
        from_latest_file=DEFAULT_FROM_LATEST_FILE,
        min_shaft_to_head_um=MIN_SHAFT_TO_HEAD_UM,
    )
    print("\n\ndone — review spines with ongoing/ASIcontroller/run_multi_spine_manager.py")


if __name__ == "__main__":
    main()
