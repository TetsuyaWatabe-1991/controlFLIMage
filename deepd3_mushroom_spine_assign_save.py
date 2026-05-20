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

from deepd3_spine_head_detector import SpinePosDeepD3, resolve_savefolder
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


def save_spine_overlay_png(mip_img, spine_zyx, dend_slope, dend_intercept,
                           savepath, pix_size=512):
    """Save MIP overlay PNG without opening an interactive window."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mip_img, cmap="gray")
    ax.axis("off")
    ax.scatter(spine_zyx[2], spine_zyx[1], c="r", s=4)
    dend_x, dend_y = [], []
    for x in range(1, mip_img.shape[1] - 1):
        y = dend_slope * x + dend_intercept
        if (1 < y) and (y < mip_img.shape[0] - 1):
            dend_x.append(x)
            dend_y.append(y)
    ax.scatter(dend_x, dend_y, c="g", s=2)
    fig.savefig(
        savepath,
        bbox_inches="tight",
        pad_inches=0,
        dpi=int(pix_size / 4 * 1.3264),
    )
    plt.close(fig)


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
        f for f in first_flim_list if "for_align" not in f
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

        if not mushrooms:
            print("  no mushroom spines detected")
            summary_rows.append({
                "flim_path": flim_path,
                "n_detected": len(result_dict),
                "n_mushroom_detected": 0,
                "n_mushroom_saved": 0,
            })
            continue

        if not save_per_spine_ini and not save_per_spine_png:
            print(
                f"  {len(mushrooms)} mushroom(s) detected; "
                "per-spine ini/png skipped (SAVE_PER_SPINE_* = False)"
            )
            summary_rows.append({
                "flim_path": flim_path,
                "n_detected": len(result_dict),
                "n_mushroom_detected": len(mushrooms),
                "n_mushroom_saved": 0,
            })
            continue

        ini_indices = get_next_ini_indices(savefolder, base_name, len(mushrooms))
        saved_inipaths = []
        n_saved = 0

        for idx, (label, mushroom_info) in zip(ini_indices, mushrooms.items()):
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
                savefolder, f"{base_name}_{str(idx).zfill(3)}.ini"
            )
            pngpath = inipath[:-4] + ".png"
            saved_parts = []

            if save_per_spine_png:
                save_spine_overlay_png(
                    mip_img, spine_zyx, dend_slope, dend_intercept, savepath=pngpath
                )
                saved_parts.append("png")
            if save_per_spine_ini:
                save_spine_dend_info(spine_zyx, dend_slope, dend_intercept, inipath)
                saved_inipaths.append(inipath)
                saved_parts.append("ini")
            n_saved += 1

            print(
                f"  saved mushroom {','.join(saved_parts)} "
                f"{os.path.basename(inipath if save_per_spine_ini else pngpath)} "
                f"(DeepD3 label {label}, shaft-to-head XY {dist_um:.3f} um)"
            )
            summary_rows.append({
                "flim_path": flim_path,
                "ini_path": inipath if save_per_spine_ini else "",
                "png_path": pngpath if save_per_spine_png else "",
                "deepd3_label": label,
                "shaft_to_head_um": dist_um,
                "spine_z": spine_zyx[0],
                "spine_y": spine_zyx[1],
                "spine_x": spine_zyx[2],
            })

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
        else:
            print(f"  saved {n_saved} png(s); ini skipped (no verification)")

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
