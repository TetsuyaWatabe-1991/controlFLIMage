import json
import os
import re
import sys
import glob
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import median_filter, shift as ndimage_shift
from skimage.segmentation import find_boundaries

sys.path.append("..\\")
from FLIMageFileReader2 import FileReader
from simple_dialog import ask_yes_no_gui, ask_open_path_gui

SCRIPT_VERSION = "2026-06-09-unc-mw-gcamp-ff0"
ROI_MASK_RAW_SUFFIX = "_roi_mask_raw"
POWERMETER_FOLDER = r"//RY-LAB-WS04/Users/yasudalab/Documents/Tetsuya_Imaging/powermeter"
FROM_THORLAB_TO_COHERENT_FACTOR = 1 / 3
_powermeter_calib_cache: Dict[str, Tuple[float, float]] = {}


def _extract_flim_sequence_number(path: str) -> int:
    """Extract trailing numeric file index from a .flim basename (e.g. *_089 -> 89)."""
    base = os.path.splitext(os.path.basename(str(path).replace("\\", "/")))[0]
    match = re.search(r"(\d{3})$", base)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)$", base)
    return int(match.group(1)) if match else 0


def _choose_lowmag_candidate(candidates: pd.DataFrame) -> pd.Series:
    """Pick the lowmag file with the smallest trailing file number."""
    if len(candidates) == 0:
        raise ValueError("No lowmag candidates to choose from.")
    ranked = candidates.copy()
    ranked["_flim_num"] = ranked["lowmag_file_path"].astype(str).map(_extract_flim_sequence_number)
    return ranked.sort_values(["_flim_num", "acq_dt"], ascending=[True, True]).iloc[0]


def _is_set_rejected(rows: pd.DataFrame) -> bool:
    """Return True when this set is marked rejected in combined_df."""
    if "reject" not in rows.columns or len(rows) == 0:
        return False
    reject_val = rows["reject"].iloc[0]
    return reject_val is True or reject_val == 1 or str(reject_val).lower() in ("true", "1")


def _reject_comment(rows: pd.DataFrame) -> str:
    if "comment" not in rows.columns or len(rows) == 0:
        return ""
    comment = rows["comment"].iloc[0]
    if comment is None or pd.isna(comment):
        return ""
    return str(comment).strip()


def _uncaging_marker_um_from_row(
    row: pd.Series,
    image_shape: Tuple[int, int],
    highmag_side_length_um: float,
    highmag_pixel: int = 128,
) -> Tuple[Optional[float], Optional[float], float, float]:
    """Compute uncaging marker position in um on a highmag Z projection."""
    height, width = image_shape
    unc_x = row.get("corrected_uncaging_x", np.nan)
    unc_y = row.get("corrected_uncaging_y", np.nan)
    if pd.isna(unc_x) or pd.isna(unc_y):
        unc_x = row.get("uncaging_display_x", np.nan)
        unc_y = row.get("uncaging_display_y", np.nan)
    if pd.isna(unc_x) or pd.isna(unc_y):
        return None, None, 0.0, 0.0

    small_x_from = float(row.get("small_x_from", 0) or 0)
    small_y_from = float(row.get("small_y_from", 0) or 0)
    unc_px_x = float(unc_x) - small_x_from
    unc_px_y = float(unc_y) - small_y_from
    trimmed_x_um = highmag_side_length_um / highmag_pixel * width
    trimmed_y_um = highmag_side_length_um / highmag_pixel * height
    unc_x_um = unc_px_x * highmag_side_length_um / highmag_pixel
    unc_y_um = unc_px_y * highmag_side_length_um / highmag_pixel
    return unc_x_um, unc_y_um, trimmed_x_um, trimmed_y_um


def _get_powermeter_slope_intercept(acq_dt: Optional[datetime]) -> Optional[Tuple[float, float]]:
    """Return (slope, intercept) for Laser2 percent -> mW from latest powermeter JSON."""
    if acq_dt is None or not os.path.exists(POWERMETER_FOLDER):
        return None
    cache_key = acq_dt.strftime("%Y%m%d%H%M")
    if cache_key in _powermeter_calib_cache:
        return _powermeter_calib_cache[cache_key]
    powermeter_files = glob.glob(os.path.join(POWERMETER_FOLDER, "*.json"))
    powermeter_files += glob.glob(os.path.join(POWERMETER_FOLDER, "old", "*.json"))
    if not powermeter_files:
        return None
    acq_int = int(acq_dt.strftime("%Y%m%d%H%M"))
    datetime_only = []
    for each_file in powermeter_files:
        base = os.path.splitext(os.path.basename(each_file))[0]
        if base.isdigit():
            datetime_only.append(int(base))
    if not datetime_only:
        return None
    datetime_arr = np.array(datetime_only)
    valid = datetime_arr[datetime_arr < acq_int]
    if len(valid) == 0:
        return None
    latest_json = f"{valid.max()}.json"
    latest_path = next((p for p in powermeter_files if os.path.basename(p) == latest_json), None)
    if latest_path is None:
        return None
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            powermeter_calib = json.load(f)
        x_percent = np.array(list(powermeter_calib["Laser2"].keys()), dtype=float)
        y_mw = np.array(list(powermeter_calib["Laser2"].values()), dtype=float)
        slope, intercept = np.polyfit(x_percent, y_mw, 1)
        _powermeter_calib_cache[cache_key] = (float(slope), float(intercept))
        return _powermeter_calib_cache[cache_key]
    except Exception:
        return None


def _uncaging_power_mw_from_row(row: pd.Series, acq_dt: Optional[datetime] = None) -> Optional[float]:
    """Resolve uncaging laser power in mW (coherent-corrected) for title display."""
    for col in ("uncaging_power_coherent_mW", "uncaging_power_mW"):
        if col in row.index:
            val = row.get(col)
            if val is not None and not pd.isna(val) and float(val) > 0:
                return round(float(val), 1)
    statedict = row.get("statedict")
    if not isinstance(statedict, dict):
        return None
    power_pct = statedict.get("State.Uncaging.Power")
    if power_pct is None:
        return None
    calib = _get_powermeter_slope_intercept(acq_dt)
    if calib is None:
        return None
    slope, intercept = calib
    pow_mw = float(slope) * float(power_pct) + float(intercept)
    return round(pow_mw * FROM_THORLAB_TO_COHERENT_FACTOR, 1)


def _extract_gcamp_pre_unc_sum(imagearray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract GCaMP (Ch1) pre/unc Z-sum images from uncaging FLIM stack."""
    n_frames = imagearray.shape[0]
    if n_frames in (4, 33, 34):
        gc_pre = imagearray[0, 0, 0, :, :, :].sum(axis=-1)
        gc_unc = imagearray[3, 0, 0, :, :, :].sum(axis=-1)
    elif n_frames == 32:
        gc_pre = imagearray[8 * 0 + 1 : 8 * 1, 0, 0, :, :, :].sum(axis=-1).sum(axis=0)
        gc_unc = imagearray[8 * 3 + 1 : 8 * 4, 0, 0, :, :, :].sum(axis=-1).sum(axis=0)
    elif n_frames == 55:
        gc_pre = imagearray[4, 0, 0, :, :, :].sum(axis=-1)
        gc_unc = imagearray[5, 0, 0, :, :, :].sum(axis=-1)
    else:
        return None, None
    return gc_pre, gc_unc


def _uncaging_center_xy(unc_df: pd.DataFrame, imagearray: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Get uncaging center in image pixels from combined_df or statedict."""
    center_x = unc_df["center_x"].iloc[0] if "center_x" in unc_df.columns else np.nan
    center_y = unc_df["center_y"].iloc[0] if "center_y" in unc_df.columns else np.nan
    if not pd.isna(center_x) and not pd.isna(center_y):
        return float(center_x), float(center_y)
    statedict = unc_df["statedict"].iloc[0] if "statedict" in unc_df.columns else None
    if isinstance(statedict, dict):
        pos = statedict.get("State.Uncaging.Position")
        if pos is not None and len(pos) >= 2:
            return float(imagearray.shape[-3] * pos[0]), float(imagearray.shape[-2] * pos[1])
    return None, None


def _plot_gcamp_ff0_subplot(
    unc_df: pd.DataFrame,
    subplot_spec: Tuple[int, int, int],
    ff0_vmin: float = 1.0,
    ff0_vmax: float = 10.0,
    cmap: str = "inferno",
) -> None:
    """Plot GCaMP F/F0 on uncaging FLIM (same method as lowmag_img_save_with_GCdata.py)."""
    plt.subplot(*subplot_spec)
    if unc_df is None or len(unc_df) == 0:
        plt.text(0.5, 0.5, "GCaMP F/F0\n(no uncaging row)", ha="center", va="center", transform=plt.gca().transAxes)
        plt.axis("off")
        return

    flim_filepath = unc_df["file_path"].iloc[0]
    if not isinstance(flim_filepath, str) or not os.path.exists(flim_filepath):
        plt.text(0.5, 0.5, "GCaMP F/F0\n(uncaging file missing)", ha="center", va="center", transform=plt.gca().transAxes)
        plt.axis("off")
        return

    try:
        iminfo = FileReader()
        iminfo.read_imageFile(flim_filepath, True)
        imagearray = np.array(iminfo.image)
        gc_pre, gc_unc = _extract_gcamp_pre_unc_sum(imagearray)
        if gc_pre is None or gc_unc is None:
            plt.text(
                0.5, 0.5, f"GCaMP F/F0\n(unsupported stack: {imagearray.shape[0]} frames)",
                ha="center", va="center", transform=plt.gca().transAxes,
            )
            plt.axis("off")
            return

        gc_pre_med = median_filter(gc_pre, size=3)
        gc_unc_med = median_filter(gc_unc, size=3)
        gc_ff0 = gc_unc_med / gc_pre_med
        gc_ff0[gc_pre_med == 0] = 0

        center_x, center_y = _uncaging_center_xy(unc_df, imagearray)
        im = plt.imshow(gc_ff0, cmap=cmap, interpolation="none", vmin=ff0_vmin, vmax=ff0_vmax)
        if center_x is not None and center_y is not None:
            plt.plot(center_x, center_y, "c+", markersize=10)
        plt.axis("off")
        plt.title("GCaMP F/F0")

        cax = inset_axes(
            plt.gca(),
            width="40%",
            height="100%",
            loc="upper center",
            bbox_to_anchor=(0.0, -0.03, 1.0, 0.05),
            bbox_transform=plt.gca().transAxes,
            borderpad=1,
        )
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.set_xticks([])
        cbar.ax.set_yticks([])
        cbar.ax.text(ff0_vmin, 0.5, f"{ff0_vmin}", ha="right", va="center", fontsize=10, color="black")
        cbar.ax.text(ff0_vmax, 0.5, f"{ff0_vmax}", ha="left", va="center", fontsize=10, color="black")
    except Exception as exc:
        plt.text(0.5, 0.5, f"GCaMP F/F0\n(error: {exc})", ha="center", va="center", transform=plt.gca().transAxes)
        plt.axis("off")


def _get_motor_xy_um(statedict: dict) -> Optional[Tuple[float, float]]:
    """Extract motor X, Y center position in um from statedict.

    Returns (x_um, y_um) or None if not available.
    The motor position [x, y, z] stored as 'State.Motor.motorPosition' represents
    the center of the FOV for X and Y axes.
    """
    pos = statedict.get("State.Motor.motorPosition") if statedict else None
    if pos is None:
        return None
    try:
        return float(pos[0]), float(pos[1])
    except Exception:
        return None


def _stack_frame_idx_from_row(row: pd.Series, set_df: pd.DataFrame) -> Optional[int]:
    """Map a combined_df row to ROI stack frame index (pre -> unc -> post order).

    Matches stack indexing used in gui_roi_fast_simple quantification.
    """
    phase = str(row.get("phase", ""))
    if {"n_pre_frames", "n_unc_frames"}.issubset(set_df.columns):
        n_pre = int(set_df["n_pre_frames"].iloc[0])
        n_unc = int(set_df["n_unc_frames"].iloc[0])
        if phase == "pre":
            pre_df = set_df[set_df["phase"] == "pre"].sort_values("nth_omit_induction")
            order = list(pre_df.index)
            if row.name in order:
                return order.index(row.name)
        elif phase == "post":
            post_df = set_df[set_df["phase"] == "post"].sort_values("nth_omit_induction")
            order = list(post_df.index)
            if row.name in order:
                return n_pre + n_unc + order.index(row.name)
        elif phase in ("unc", "uncaging"):
            unc_df = set_df[set_df["phase"].isin(["unc", "uncaging"])]
            if len(unc_df) > 0:
                unc_path = str(unc_df.iloc[0].get("file_path", ""))
                if row.name in set_df.index and str(set_df.loc[row.name].get("file_path", "")) == unc_path:
                    t = row.get("slice", row.get("nth_omit_induction", 0))
                    if t is not None and not pd.isna(t):
                        return n_pre + int(t)
    rel = row.get("relative_nth_omit_induction", None)
    if rel is not None and not pd.isna(rel) and phase in ("pre", ""):
        return int(rel)
    nth = row.get("nth_omit_induction", None)
    if nth is not None and not pd.isna(nth) and phase in ("pre", ""):
        return int(nth)
    return None


def _read_roi_stack(tiff_dir: str, base_name: str, roi_type: str) -> Tuple[Optional[np.ndarray], bool]:
    """Load ROI stack, preferring drift-corrected raw-FLIM masks when available."""
    raw_path = os.path.join(tiff_dir, f"{base_name}_{roi_type}{ROI_MASK_RAW_SUFFIX}.tif")
    aligned_path = os.path.join(tiff_dir, f"{base_name}_{roi_type}_roi_mask.tif")
    if os.path.exists(raw_path):
        return tifffile.imread(raw_path), True
    if os.path.exists(aligned_path):
        return tifffile.imread(aligned_path), False
    return None, False


def _apply_inverse_drift_to_mask(roi_frame: np.ndarray, row: pd.Series, set_df: pd.DataFrame) -> np.ndarray:
    """Convert aligned-stack ROI coordinates to raw FLIM coordinates."""
    phase = str(row.get("phase", ""))
    if phase in ("unc", "uncaging"):
        unc_rows = set_df[set_df["phase"].isin(["unc", "uncaging"])]
        shift_y = float(unc_rows.iloc[0].get("unc_drift_y", 0) or 0) if len(unc_rows) else 0.0
        shift_x = float(unc_rows.iloc[0].get("unc_drift_x", 0) or 0) if len(unc_rows) else 0.0
    else:
        shift_y = float(row.get("shift_y", 0) or 0)
        shift_x = float(row.get("shift_x", 0) or 0)
    shifted = ndimage_shift(roi_frame.astype(np.float32), (-shift_y, -shift_x), order=0, mode="constant", cval=0)
    return shifted > 0.5


def _load_roi_mask_from_tiff(
    row: pd.Series,
    roi_type: str,
    cache: Dict[Tuple[str, str, str], Optional[np.ndarray]],
    set_df: Optional[pd.DataFrame] = None,
) -> Optional[np.ndarray]:
    """Load one ROI mask frame for overlay on raw FLIM (drift-corrected when needed)."""
    after_align_path = row.get("after_align_save_path", None)
    if not isinstance(after_align_path, str) or not os.path.exists(after_align_path):
        return None
    if set_df is None:
        set_df = pd.DataFrame([row])

    stack_frame_idx = _stack_frame_idx_from_row(row, set_df)
    if stack_frame_idx is None:
        return None

    tiff_dir = os.path.dirname(after_align_path)
    base_name = os.path.splitext(os.path.basename(after_align_path))[0]
    key = (after_align_path, roi_type, base_name)
    if key not in cache:
        try:
            roi_stack, is_raw = _read_roi_stack(tiff_dir, base_name, roi_type)
            cache[key] = None if roi_stack is None else (roi_stack, is_raw)
        except Exception:
            cache[key] = None
            return None

    cached = cache.get(key)
    if cached is None:
        return None
    roi_stack, is_raw = cached

    if roi_stack.ndim == 2:
        frame = roi_stack
    else:
        if stack_frame_idx < 0 or stack_frame_idx >= roi_stack.shape[0]:
            return None
        frame = roi_stack[stack_frame_idx]

    if is_raw:
        return frame.astype(bool)
    return _apply_inverse_drift_to_mask(frame, row, set_df)


def plt_zpro_with_roi_tiff(
    mask_row: pd.Series,
    roi_types,
    color_dict,
    vmax: float,
    vmin: float,
    highmag_side_length_um: float,
    roi_cache: Dict[Tuple[str, str, str], Optional[np.ndarray]],
    set_df: Optional[pd.DataFrame] = None,
    ch_1or2: int = 1,
) -> None:
    flim_filepath = mask_row["file_path"]
    iminfo = FileReader()
    iminfo.read_imageFile(flim_filepath, True)
    six_dim = np.array(iminfo.image)

    # Backward compatibility:
    # Some combined_df variants do not have z_from/z_to columns.
    if "z_from" in mask_row.index and "z_to" in mask_row.index:
        z_from = int(mask_row["z_from"])
        z_to = int(mask_row["z_to"])
    else:
        # Fallback: use small z-window around corrected_uncaging_z if available,
        # otherwise use full z range.
        z_len = six_dim.shape[0]
        corrected_uncaging_z = mask_row.get("corrected_uncaging_z", None)
        if corrected_uncaging_z is not None and not pd.isna(corrected_uncaging_z):
            z_center = int(round(float(corrected_uncaging_z)))
            z_from = max(0, z_center - 1)
            z_to = min(z_len, z_center + 2)
            if z_to <= z_from:
                z_from, z_to = 0, z_len
        else:
            z_from, z_to = 0, z_len

    # Final safety clamp
    z_from = max(0, min(z_from, six_dim.shape[0] - 1))
    z_to = max(z_from + 1, min(z_to, six_dim.shape[0]))
    z_projection = six_dim[z_from:z_to, 0, ch_1or2 - 1, :, :, :].sum(axis=-1).max(axis=0)

    plt.imshow(
        z_projection,
        cmap="gray",
        interpolation="none",
        vmax=vmax,
        vmin=vmin,
        extent=(0, highmag_side_length_um, highmag_side_length_um, 0),
    )

    linewidth = 0.5
    for roi_type in roi_types:
        mask_col = f"{roi_type}_shifted_mask"
        temp_mask = None
        if mask_col in mask_row.index and isinstance(mask_row[mask_col], np.ndarray):
            temp_mask = mask_row[mask_col]
        else:
            temp_mask = _load_roi_mask_from_tiff(mask_row, roi_type, roi_cache, set_df=set_df)

        if temp_mask is None or temp_mask.size == 0 or not np.any(temp_mask):
            continue

        boundaries = find_boundaries(temp_mask, mode="thick")
        height, width = temp_mask.shape
        inc_x = highmag_side_length_um / width
        inc_y = highmag_side_length_um / height
        color = color_dict[roi_type]
        for y in range(height):
            for x in range(width):
                if not boundaries[y, x]:
                    continue
                px = x * inc_x
                py = y * inc_y
                if y == 0 or temp_mask[y - 1, x] != temp_mask[y, x]:
                    plt.plot([px, px + inc_x], [py, py], color=color, linewidth=linewidth)
                if y == height - 1 or temp_mask[y + 1, x] != temp_mask[y, x]:
                    plt.plot([px, px + inc_x], [py + inc_y, py + inc_y], color=color, linewidth=linewidth)
                if x == 0 or temp_mask[y, x - 1] != temp_mask[y, x]:
                    plt.plot([px, px], [py, py + inc_y], color=color, linewidth=linewidth)
                if x == width - 1 or temp_mask[y, x + 1] != temp_mask[y, x]:
                    plt.plot([px + inc_x, px + inc_x], [py, py + inc_y], color=color, linewidth=linewidth)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Y (um)")
    plt.xlabel("X (um)")


def build_lowmag_df(one_of_filepath_dict, save_images: bool, ch_1or2: int) -> pd.DataFrame:
    dt_formatter = "%Y-%m-%dT%H:%M:%S.%f"
    lowmag_df = pd.DataFrame()
    for file_path in one_of_filepath_dict.keys():
        lowmag_file_list = [
            f
            for f in glob.glob(os.path.join(os.path.dirname(file_path), "*[0-9][0-9][0-9].flim"))
            if "highmag" not in os.path.basename(f).lower()
        ]

        tiff_array_savefolder = os.path.join(os.path.dirname(file_path), "lowmag_tiff_array")
        z_projection_savefolder = os.path.join(os.path.dirname(file_path), "lowmag_tiff_z_projection")
        y_projection_savefolder = os.path.join(os.path.dirname(file_path), "lowmag_tiff_y_projection")
        for folder in [tiff_array_savefolder, z_projection_savefolder, y_projection_savefolder]:
            os.makedirs(folder, exist_ok=True)

        for flim_path in lowmag_file_list:
            iminfo = FileReader()
            try:
                iminfo.read_imageFile(flim_path, save_images)
            except Exception as exc:
                print(f"Could not read lowmag file: {flim_path}\n{exc}")
                continue

            acq_dt = datetime.strptime(iminfo.acqTime[0], dt_formatter)
            statedict = iminfo.statedict.copy()
            tiff_array_savepath = os.path.join(tiff_array_savefolder, os.path.basename(flim_path).replace(".flim", ".tif"))
            z_projection_savepath = os.path.join(z_projection_savefolder, os.path.basename(flim_path).replace(".flim", ".tif"))
            y_projection_savepath = os.path.join(y_projection_savefolder, os.path.basename(flim_path).replace(".flim", ".tif"))

            if save_images:
                six_dim = np.array(iminfo.image)
                each_tiff_array = six_dim[:, 0, ch_1or2 - 1, :, :, :].sum(axis=-1)
                tifffile.imwrite(tiff_array_savepath, each_tiff_array)
                tifffile.imwrite(z_projection_savepath, each_tiff_array.max(axis=0))
                tifffile.imwrite(y_projection_savepath, each_tiff_array.max(axis=1))

            lowmag_df = pd.concat(
                [
                    lowmag_df,
                    pd.DataFrame(
                        {
                            "highmag_one_of_file_path": [file_path],
                            "lowmag_file_path": [flim_path],
                            "acq_dt": [acq_dt],
                            "lowmag_tiff_array_savepath": [tiff_array_savepath],
                            "lowmag_tiff_z_projection_savepath": [z_projection_savepath],
                            "lowmag_tiff_y_projection_savepath": [y_projection_savepath],
                            "statedict": [statedict],
                        }
                    ),
                ],
                ignore_index=True,
            )
    return lowmag_df


def _compute_norm_series_from_intensity_csv(
    intensity_df: pd.DataFrame,
    group_value,
    set_label_value,
    ch_1or2: int,
) -> Optional[pd.DataFrame]:
    """Compute relative-time and normalized spine intensity from all-frames CSV."""
    if intensity_df is None or len(intensity_df) == 0:
        return None
    if "group" not in intensity_df.columns or "set_label" not in intensity_df.columns:
        return None

    each = intensity_df[
        (intensity_df["group"].astype(str) == str(group_value))
        & (intensity_df["set_label"].astype(float) == float(set_label_value))
    ].copy()
    if len(each) == 0:
        return None

    spine_col = f"Spine_Ch{ch_1or2}_intensity"
    bg_col = f"Background_Ch{ch_1or2}_intensity"
    if spine_col not in each.columns or bg_col not in each.columns:
        return None
    if "elapsed_time_sec" not in each.columns or "phase" not in each.columns:
        return None
    if "nAveFrame" not in each.columns:
        each["nAveFrame"] = 1.0

    each["spine_subbg_div_nAve"] = (each[spine_col] - each[bg_col]) / each["nAveFrame"].replace(0, np.nan)
    pre_vals = each.loc[each["phase"] == "pre", "spine_subbg_div_nAve"].dropna()
    if len(pre_vals) == 0:
        return None
    baseline = pre_vals.mean()
    if baseline == 0 or pd.isna(baseline):
        return None

    unc_rows = each[each["phase"] == "unc"]
    if len(unc_rows) > 0:
        unc_t = unc_rows["elapsed_time_sec"].min()
    else:
        unc_t = each["elapsed_time_sec"].min()
    each["relative_time_min"] = (each["elapsed_time_sec"] - unc_t) / 60.0
    each["norm_intensity"] = each["spine_subbg_div_nAve"] / baseline - 1.0
    return each[["relative_time_min", "norm_intensity"]].dropna().sort_values("relative_time_min")


def _norm_path_for_match(path_str: str) -> str:
    """Normalize slashes/case for robust UNC path matching."""
    return str(path_str).replace("\\", "/").lower()


def _filepath_dict_from_combined_df(combined_df: pd.DataFrame) -> Dict[str, str]:
    """Build one representative highmag path per directory from combined_df."""
    one_of_filepath_dict: Dict[str, str] = {}
    if "file_path" not in combined_df.columns:
        return one_of_filepath_dict
    seen_dirs = set()
    for file_path in combined_df["file_path"].dropna().astype(str).unique():
        dir_path = os.path.dirname(file_path)
        if dir_path not in seen_dirs:
            seen_dirs.add(dir_path)
            one_of_filepath_dict[file_path] = " "
    return one_of_filepath_dict


def _extract_lowmag_prefix_from_highmag_path(highmag_path: str) -> str:
    """Extract basename prefix like 'B2_1_pos1_' from '*__highmag_*' path."""
    base = os.path.basename(str(highmag_path).replace("\\", "/"))
    mark = base.find("__highmag_")
    if mark >= 0:
        return base[:mark] + "_"
    # fallback: keep old behavior if naming is unusual
    mark2 = base.find("_highmag_")
    if mark2 >= 0:
        return base[:mark2] + "_"
    return os.path.splitext(base)[0]


def main() -> None:
    print(f"[DEBUG] script_version={SCRIPT_VERSION}")
    ch_1or2 = 2 if ask_yes_no_gui("Use Ch2 (tdTomato/volume)? [Yes=Ch2, No=Ch1]") else 1
    combined_df_reject_bad_data_pkl_path = ask_open_path_gui(
        filetypes=[("Pickle files", "*.pkl")],
    )
    intensity_csv_default = combined_df_reject_bad_data_pkl_path.replace(".pkl", "_intensity_lifetime_all_frames.csv")
    combined_df_dir = os.path.dirname(combined_df_reject_bad_data_pkl_path)
    savefolder = os.path.join(combined_df_dir, "lowmag_combined_summary")
    lowmag_df_path = os.path.join(combined_df_dir, "lowmag_df.pkl")

    combined_df_reject_bad_data_df = pd.read_pickle(combined_df_reject_bad_data_pkl_path)
    intensity_allframes_df = None
    if os.path.exists(intensity_csv_default):
        try:
            intensity_allframes_df = pd.read_csv(intensity_csv_default)
            print(f"[DEBUG] loaded intensity csv: {intensity_csv_default}")
        except Exception as exc:
            print(f"[DEBUG] could not read intensity csv: {exc}")
    # Backward/variant compatibility: some combined_df files do not have `label`.
    if "label" not in combined_df_reject_bad_data_df.columns:
        if {"filepath_without_number", "nth_set_label"}.issubset(combined_df_reject_bad_data_df.columns):
            combined_df_reject_bad_data_df["label"] = (
                combined_df_reject_bad_data_df["filepath_without_number"].astype(str)
                + "__set"
                + combined_df_reject_bad_data_df["nth_set_label"].astype(str)
            )
        elif {"group", "nth_set_label"}.issubset(combined_df_reject_bad_data_df.columns):
            combined_df_reject_bad_data_df["label"] = (
                combined_df_reject_bad_data_df["group"].astype(str)
                + "__set"
                + combined_df_reject_bad_data_df["nth_set_label"].astype(str)
            )
        elif "group" in combined_df_reject_bad_data_df.columns:
            combined_df_reject_bad_data_df["label"] = combined_df_reject_bad_data_df["group"].astype(str)
        else:
            combined_df_reject_bad_data_df["label"] = combined_df_reject_bad_data_df["file_path"].astype(str)
    roi_types = ["Spine", "DendriticShaft", "Background"]
    color_dict = {"Spine": "red", "DendriticShaft": "blue", "Background": "green"}

    rebuild_lowmag_df = (
        not os.path.exists(lowmag_df_path)
        or ask_yes_no_gui("Rebuild lowmag_df?")
    )
    if rebuild_lowmag_df:
        one_of_filepath_dict = _filepath_dict_from_combined_df(combined_df_reject_bad_data_df)
        if not one_of_filepath_dict:
            raise ValueError("combined_df has no valid file_path entries to build lowmag_df.")
        lowmag_df = build_lowmag_df(one_of_filepath_dict, save_images=True, ch_1or2=ch_1or2)
        lowmag_df.to_pickle(lowmag_df_path)
        print(f"[DEBUG] built and saved lowmag_df: {lowmag_df_path}")
    else:
        lowmag_df = pd.read_pickle(lowmag_df_path)
        print(f"[DEBUG] loaded lowmag_df: {lowmag_df_path}")

    mapped_count = 0
    no_candidate_count = 0
    lowmag_df = lowmag_df.copy()
    lowmag_df["lowmag_norm_path"] = lowmag_df["lowmag_file_path"].astype(str).map(_norm_path_for_match)
    lowmag_df["lowmag_basename"] = lowmag_df["lowmag_file_path"].astype(str).map(lambda p: os.path.basename(str(p).replace("\\", "/")))
    for idx, _ in combined_df_reject_bad_data_df.iterrows():
        highmag_path = combined_df_reject_bad_data_df.at[idx, "file_path"]
        highmag_dt = combined_df_reject_bad_data_df.at[idx, "dt"] if "dt" in combined_df_reject_bad_data_df.columns else None
        if not isinstance(highmag_path, str):
            no_candidate_count += 1
            continue
        highmag_norm = _norm_path_for_match(highmag_path)
        highmag_dir_norm = _norm_path_for_match(os.path.dirname(highmag_path))
        lowmag_prefix = _extract_lowmag_prefix_from_highmag_path(highmag_path)
        if "__highmag_" not in os.path.basename(str(highmag_path).replace("\\", "/")) and "_highmag_" not in os.path.basename(str(highmag_path).replace("\\", "/")):
            no_candidate_count += 1
            continue
        # Primary: same directory + basename prefix match
        candidates = lowmag_df[
            lowmag_df["lowmag_norm_path"].str.contains(highmag_dir_norm, regex=False)
            & lowmag_df["lowmag_basename"].str.startswith(lowmag_prefix)
        ]
        if highmag_dt is not None and not pd.isna(highmag_dt):
            candidates = candidates[candidates["acq_dt"] < highmag_dt]
        # Fallback 1: same directory only
        if len(candidates) == 0:
            candidates = lowmag_df[lowmag_df["lowmag_norm_path"].str.contains(highmag_dir_norm, regex=False)]
            if highmag_dt is not None and not pd.isna(highmag_dt):
                candidates = candidates[candidates["acq_dt"] < highmag_dt]
        # Fallback 2: prefix only (ignore directory)
        if len(candidates) == 0:
            candidates = lowmag_df[lowmag_df["lowmag_basename"].str.startswith(lowmag_prefix)]
        # Fallback 3: if still none and dt filter may be too strict, ignore dt + directory
        if len(candidates) == 0:
            candidates = lowmag_df[lowmag_df["lowmag_norm_path"].str.contains(highmag_dir_norm, regex=False)]
        if len(candidates) == 0:
            no_candidate_count += 1
            continue
        # Use the lowmag .flim with the smallest trailing file number (e.g. *_001).
        chosen = _choose_lowmag_candidate(candidates)
        combined_df_reject_bad_data_df.at[idx, "lowmag_file_path"] = chosen["lowmag_file_path"]
        combined_df_reject_bad_data_df.at[idx, "lowmag_tiff_z_projection_savepath"] = chosen["lowmag_tiff_z_projection_savepath"]
        combined_df_reject_bad_data_df.at[idx, "lowmag_tiff_y_projection_savepath"] = chosen["lowmag_tiff_y_projection_savepath"]
        mapped_count += 1

    print(f"[DEBUG] lowmag mapping done: mapped={mapped_count}, no_candidate={no_candidate_count}, total_rows={len(combined_df_reject_bad_data_df)}")

    roi_cache: Dict[Tuple[str, str, str], Optional[np.ndarray]] = {}
    unique_id = 0
    total_labels = 0
    plotted_labels = 0
    skip_no_phase_count = 0
    skip_no_lowmag_count = 0
    skip_no_tiff_count = 0
    for each_label in combined_df_reject_bad_data_df["label"].dropna().unique():
        total_labels += 1
        each_label_df = combined_df_reject_bad_data_df[
            (combined_df_reject_bad_data_df["label"] == each_label) & (combined_df_reject_bad_data_df["phase"] != "unc")
        ]
        if len(each_label_df) == 0:
            skip_no_phase_count += 1
            continue
        if "lowmag_tiff_z_projection_savepath" not in each_label_df.columns:
            skip_no_lowmag_count += 1
            continue
        if pd.isna(each_label_df["lowmag_tiff_z_projection_savepath"].iloc[0]):
            skip_no_lowmag_count += 1
            continue

        # Keep only rows that look like actual analysis frames
        valid_rows = each_label_df.copy()
        if "phase" in valid_rows.columns:
            valid_rows = valid_rows[valid_rows["phase"].isin(["pre", "post"])]
        if "nth_set_label" in valid_rows.columns:
            valid_rows = valid_rows[valid_rows["nth_set_label"] != -1]
        if len(valid_rows) == 0:
            skip_no_phase_count += 1
            continue

        # Robustly pick an existing before-align TIFF path (avoid NaN/float at iloc[0])
        before_align_path = None
        if "before_align_save_path" in valid_rows.columns:
            for p in valid_rows["before_align_save_path"].dropna().astype(str).tolist():
                if os.path.exists(p):
                    before_align_path = p
                    break
        if before_align_path is None and "after_align_save_path" in valid_rows.columns:
            for p in valid_rows["after_align_save_path"].dropna().astype(str).tolist():
                if os.path.exists(p):
                    before_align_path = p
                    break
        if before_align_path is None:
            print(f"[DEBUG] skip label(no valid tif path): {each_label}")
            skip_no_tiff_count += 1
            continue

        highmag_zoom = 15
        highmag_side_length_um = 128.0 / highmag_zoom
        if "statedict" in valid_rows.columns and isinstance(valid_rows["statedict"].iloc[0], dict):
            try:
                highmag_side_length_um = valid_rows["statedict"].iloc[0]["State.Acq.FOV_default"][0] / highmag_zoom
            except Exception:
                pass
        before_align_txy = tifffile.imread(before_align_path)
        pre_df = valid_rows[valid_rows["phase"] == "pre"]
        post_df = valid_rows[valid_rows["phase"] == "post"]
        if len(pre_df) == 0 or len(post_df) == 0:
            print(f"[DEBUG] skip label(pre/post missing): {each_label}")
            skip_no_phase_count += 1
            continue
        before_unc_nth = pre_df.iloc[-1]["nth_omit_induction"] - pre_df.iloc[0]["nth_omit_induction"]
        try:
            before_unc_nth = int(before_unc_nth)
        except Exception:
            before_unc_nth = 0
        before_unc_nth = max(0, min(before_unc_nth, before_align_txy.shape[0] - 1))
        before_uncaging_zproj = before_align_txy[before_unc_nth, :, :]
        vmax = before_uncaging_zproj.max() / 3
        vmin = before_uncaging_zproj.min()

        lowmag_row = lowmag_df[lowmag_df["lowmag_file_path"] == valid_rows["lowmag_file_path"].iloc[0]].iloc[0]
        lowmag_zoom = lowmag_row["statedict"]["State.Acq.zoom"]
        lowmag_side_length_um = lowmag_row["statedict"]["State.Acq.FOV_default"][0] / lowmag_zoom
        lowmag_z_um = lowmag_row["statedict"]["State.Acq.sliceStep"]
        lowmag_vmax = vmax * (lowmag_row["statedict"]["State.Acq.nAveFrame"] / 3)
        lowmag_vmin = vmin * (lowmag_row["statedict"]["State.Acq.nAveFrame"] / 3)

        # Compute highmag FOV position on lowmag image using motor positions.
        # State.Motor.motorPosition[x, y, z]: x,y = center of FOV; z = bottom of Z stack.
        # Motor offset directly maps to image coordinate offset (directionMotorX/Y = 1).
        highmag_sd = (
            valid_rows["statedict"].iloc[0]
            if "statedict" in valid_rows.columns and isinstance(valid_rows["statedict"].iloc[0], dict)
            else None
        )
        lowmag_motor_xy = _get_motor_xy_um(lowmag_row["statedict"])
        highmag_motor_xy = _get_motor_xy_um(highmag_sd)
        _hmag_dx: Optional[float] = None
        _hmag_dy: Optional[float] = None
        if highmag_motor_xy is not None and lowmag_motor_xy is not None:
            _hmag_dx = highmag_motor_xy[0] - lowmag_motor_xy[0]
            _hmag_dy = highmag_motor_xy[1] - lowmag_motor_xy[1]

        pre_row = pre_df.iloc[-1]
        unc_x_um, unc_y_um, trimmed_high_x_um, trimmed_high_y_um = _uncaging_marker_um_from_row(
            pre_row, before_uncaging_zproj.shape, highmag_side_length_um,
        )
        each_label_unc_df = combined_df_reject_bad_data_df[
            (combined_df_reject_bad_data_df["label"] == each_label)
            & (combined_df_reject_bad_data_df["phase"] == "unc")
        ]
        dt_val = valid_rows["dt"].iloc[0] if "dt" in valid_rows.columns else None
        unc_row_for_power = each_label_unc_df.iloc[0] if len(each_label_unc_df) > 0 else pre_row
        uncaging_power_mw = _uncaging_power_mw_from_row(unc_row_for_power, acq_dt=dt_val)

        fig_dim = [2, 4]
        fig = plt.figure(figsize=(4 * fig_dim[1], 4 * fig_dim[0]))
        fig.suptitle(each_label)
        if _is_set_rejected(valid_rows):
            reject_line = "REJECTED"
            reject_comment = _reject_comment(valid_rows)
            if reject_comment:
                reject_line = f"{reject_line}  ({reject_comment})"
            fig.text(
                0.5, 0.97, reject_line,
                ha="center", va="top", color="red", fontsize=14, fontweight="bold",
                transform=fig.transFigure,
            )

        lowmag_file_basename = os.path.basename(str(lowmag_row["lowmag_file_path"]))
        plt.subplot(fig_dim[0], fig_dim[1], 1)
        zpro = tifffile.imread(valid_rows["lowmag_tiff_z_projection_savepath"].iloc[0])
        plt.imshow(zpro, cmap="gray", vmin=lowmag_vmin, vmax=lowmag_vmax, extent=(0, lowmag_side_length_um, lowmag_side_length_um, 0))
        cx: Optional[float] = None
        cy: Optional[float] = None
        if _hmag_dx is not None and _hmag_dy is not None:
            cx = lowmag_side_length_um / 2 + _hmag_dx
            cy = lowmag_side_length_um / 2 + _hmag_dy
            rect = Rectangle(
                (cx - highmag_side_length_um / 2, cy - highmag_side_length_um / 2),
                highmag_side_length_um, highmag_side_length_um,
                linewidth=1.5, edgecolor="yellow", facecolor="none",
            )
            plt.gca().add_patch(rect)
        if cx is not None and cy is not None and unc_x_um is not None and unc_y_um is not None:
            lowmag_unc_x = cx + unc_x_um - trimmed_high_x_um / 2
            lowmag_unc_y = cy + unc_y_um - trimmed_high_y_um / 2
            plt.plot(lowmag_unc_x, lowmag_unc_y, "c+", markersize=10)
        plt.ylabel("Y (um)")
        plt.xlabel("X (um)")
        plt.title(f"Low mag Z projection\n{lowmag_file_basename}")

        plt.subplot(fig_dim[0], fig_dim[1], 2)
        ypro = tifffile.imread(valid_rows["lowmag_tiff_y_projection_savepath"].iloc[0])
        lowmag_z_total = ypro.shape[0] * lowmag_z_um
        plt.imshow(ypro, cmap="gray", vmin=lowmag_vmin, vmax=lowmag_vmax, origin="lower", extent=(0, lowmag_side_length_um, lowmag_z_total, 0))
        if _hmag_dx is not None and highmag_sd is not None:
            try:
                lowmag_motor_z = float(lowmag_row["statedict"].get("State.Motor.motorPosition", [0, 0, 0])[2])
                highmag_motor_z = float(highmag_sd.get("State.Motor.motorPosition", [0, 0, 0])[2])
                delta_z = highmag_motor_z - lowmag_motor_z
                highmag_nslices = int(highmag_sd.get("State.Acq.nSlices", 1))
                highmag_slice_step = float(highmag_sd.get("State.Acq.sliceStep", lowmag_z_um))
                highmag_z_len = max(highmag_nslices - 1, 1) * highmag_slice_step
                y_z_top = lowmag_z_total - delta_z - highmag_z_len
                x_left = lowmag_side_length_um / 2 + _hmag_dx - highmag_side_length_um / 2
                rect_z = Rectangle(
                    (x_left, y_z_top), highmag_side_length_um, highmag_z_len,
                    linewidth=1.5, edgecolor="yellow", facecolor="none",
                )
                plt.gca().add_patch(rect_z)
            except Exception:
                pass
        plt.ylabel("Z (um)")
        plt.xlabel("X (um)")
        plt.title(f"Low mag Y projection\n{lowmag_file_basename}")

        plt.subplot(fig_dim[0], fig_dim[1], 3)
        plt_zpro_with_roi_tiff(
            pre_row, roi_types, color_dict, vmax, vmin, highmag_side_length_um, roi_cache,
            set_df=valid_rows, ch_1or2=ch_1or2,
        )
        plt.title("Pre with ROI")

        plt.subplot(fig_dim[0], fig_dim[1], 4)
        post_row = post_df.iloc[0]
        plt_zpro_with_roi_tiff(
            post_row, roi_types, color_dict, vmax, vmin, highmag_side_length_um, roi_cache,
            set_df=valid_rows, ch_1or2=ch_1or2,
        )
        plt.title("Post with ROI")

        plt.subplot(fig_dim[0], fig_dim[1], 5)
        plt.imshow(
            before_uncaging_zproj, cmap="gray", vmin=vmin * 2, vmax=vmax * 2,
            extent=(0, trimmed_high_x_um, trimmed_high_y_um, 0),
        )
        if unc_x_um is not None and unc_y_um is not None:
            plt.plot(unc_x_um, unc_y_um, "c+", markersize=10)
        plt.ylabel("Y (um)")
        plt.xlabel("X (um)")
        if uncaging_power_mw is not None:
            plt.title(f"Uncaging {uncaging_power_mw} mW")
        else:
            plt.title("Uncaging")

        _plot_gcamp_ff0_subplot(each_label_unc_df, (fig_dim[0], fig_dim[1], 6))

        plt.subplot(fig_dim[0], fig_dim[1], 7)
        has_norm_intensity = ("norm_intensity" in valid_rows.columns) and valid_rows["norm_intensity"].notna().any()
        norm_plot_df = None
        if has_norm_intensity:
            norm_plot_df = valid_rows[["relative_time_min", "norm_intensity"]].dropna().sort_values("relative_time_min")
        if not has_norm_intensity:
            csv_series = None
            if "group" in valid_rows.columns and "nth_set_label" in valid_rows.columns:
                csv_series = _compute_norm_series_from_intensity_csv(
                    intensity_allframes_df,
                    valid_rows["group"].iloc[0],
                    valid_rows["nth_set_label"].iloc[0],
                    ch_1or2=ch_1or2,
                )
            if csv_series is not None and len(csv_series) > 0:
                has_norm_intensity = True
                norm_plot_df = csv_series
        if has_norm_intensity:
            plt.plot(norm_plot_df["relative_time_min"], norm_plot_df["norm_intensity"], color="k", marker="o")
        else:
            plt.text(
                0.05,
                0.7,
                "norm_intensity not available\n(use combined_df_with_roi_mask...pkl)",
                transform=plt.gca().transAxes,
            )
        plt.plot([-20, 40], [0, 0], color="gray", linestyle="--")
        plt.xlim([-20, 40])
        plt.ylim([-0.4, 2.1])
        plt.xlabel("Time (min)")
        plt.ylabel("normalized intensity")
        plt.title("Spine volume")

        plt.subplot(fig_dim[0], fig_dim[1], 8)
        late_candidate = valid_rows[valid_rows["relative_time_min"] > 25]
        if len(late_candidate) == 0:
            late_row = valid_rows.iloc[-1]
        else:
            late_row = late_candidate.iloc[0]
        plt_zpro_with_roi_tiff(
            late_row, roi_types, color_dict, vmax, vmin, highmag_side_length_um, roi_cache,
            set_df=valid_rows, ch_1or2=ch_1or2,
        )
        plt.title("Late post with ROI")

        plt.tight_layout()
        unique_id += 1
        wo_date_savefolder = os.path.join(savefolder, "wo_date")
        with_date_savefolder = os.path.join(savefolder, "with_date")
        os.makedirs(wo_date_savefolder, exist_ok=True)
        os.makedirs(with_date_savefolder, exist_ok=True)
        savepath_wo_date = os.path.join(wo_date_savefolder, f"{os.path.basename(each_label)}_{unique_id}.png")
        dt_str = dt_val.strftime("%Y%m%d_%H%M%S") if hasattr(dt_val, "strftime") else "unknown_dt"
        savepath_with_date = os.path.join(with_date_savefolder, f"{dt_str}_{os.path.basename(each_label)}_{unique_id}.png")
        plt.savefig(savepath_wo_date, dpi=150, bbox_inches="tight")
        plt.savefig(savepath_with_date, dpi=150, bbox_inches="tight")
        plt.close()
        plotted_labels += 1

    print(
        "[DEBUG] plotting summary: "
        f"total_labels={total_labels}, plotted={plotted_labels}, "
        f"skip_no_phase={skip_no_phase_count}, skip_no_lowmag={skip_no_lowmag_count}, "
        f"skip_no_tiff={skip_no_tiff_count}"
    )
    if plotted_labels == 0:
        print("[DEBUG] No output image created. Check lowmag mapping and phase names in combined_df.")
    print("done (TIFF ROI version)")


if __name__ == "__main__":
    main()
