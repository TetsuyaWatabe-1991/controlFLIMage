# -*- coding: utf-8 -*-
"""
TIFF ROI definition with Pre + Uncaging (3D, unprojected) + Post.

Based on example_gui_usage_reduce_asking_tiff_only.py and transient_roi_analysis.py.
Does NOT modify any existing files.

Behavior:
- Runs the same data preparation as example (first_processing_for_flim_files):
  Pre and Post are aligned (on small crop) and Z-projected, then saved to small TIFF.
- Builds a FULL-SIZE stack for ROI definition: Pre Z-proj (full) + Uncaging (full, drift
  applied) + Post Z-proj (full). User defines ROIs on this full-size image.
- Two ROI TIFFs are saved per ROI type:
  (1) Aligned ROI: mask as drawn on the full-size stack (same as GUI).
  (2) Raw ROI: drift-corrected so it fits pre-drift FLIM frames; use this for
      quantification from FLIM (per file, pre/post with z_plus_minus Z-proj).
- Quantification: from FLIM only; pre/post use z_plus_minus; intensity per frame, Ch, ROI.

TEST MODE:
- Uses a fixed FLIM path; no predefined df. first_processing is always run.
"""

import os
import sys
sys.path.append('..\\')
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import tifffile
from datetime import datetime
from PyQt5.QtWidgets import QApplication
from scipy.ndimage import fourier_shift as _fourier_shift
from skimage.registration import phase_cross_correlation

from FLIMageFileReader2 import FileReader
from fitting.flim_lifetime_fitting import FLIMLifetimeFitter
from gui_integration import (
    first_processing_for_flim_files,
    load_and_align_data,
    process_uncaging_positions,
)
from file_selection_gui_tiff_only import launch_file_selection_gui_tiff_only
from simple_dialog import ask_yes_no_gui, ask_open_path_gui, ask_save_path_gui

# -----------------------------------------------------------------------------
# TEST CONFIG: fixed path, no predefined df, no dialogs
# -----------------------------------------------------------------------------
TEST_MODE = False
TEST_FLIM_PATH = r"C:\Users\WatabeT\Desktop\temp2\BrUSGFP_7_pos1__highmag_1_002.flim"
# first_processing uses glob "*_highmag_*002.flim" in the folder; if 002 is missing,
# we pass the folder's first FLIM (e.g. 004) so get_uncaging_pos_multiple still gets
# the full group via get_flimfile_list(004) -> 001,002,...,004.


SAVE_PLOT_TF = True
SAVE_TIF_TF = True

TIFF_WITH_UNCAGING_SUFFIX = "_with_uncaging"
ROI_MASK_RAW_SUFFIX = "_roi_mask_raw"
ROI_TYPES = ["Spine", "DendriticShaft", "Background"]

# Default ROI geometry (same as calc_spine_dend_GCaMP in flimage_graph_func.py)
DEFAULT_CIRCLE_RADIUS = 3
DEFAULT_RECT_LENGTH = 10
DEFAULT_RECT_HEIGHT = 2
DEFAULT_BG_PERCENTILE = 10

# Lifetime fitting (transient_roi_analysis style)
SYNC_RATE = 80e6  # Hz


def _low_intensity_largest_mask_2d(image: np.ndarray, percentile: float = 10) -> np.ndarray:
    """
    Build a 2D boolean mask for the largest connected region of low-intensity pixels.
    Same logic as spine_roi_from_S.low_intensity_largest_mask but returns mask and no matplotlib.
    Used for auto Background ROI in create_initial_roi_masks_from_ini.
    """
    from scipy.ndimage import label, median_filter, binary_opening, binary_closing
    threshold = np.percentile(image, percentile)
    image_smooth = median_filter(image, size=5)
    low_mask = image_smooth <= threshold
    low_mask = binary_opening(low_mask, structure=np.ones((3, 3)))
    low_mask = binary_closing(low_mask, structure=np.ones((3, 3)))
    h, w = image.shape
    margin_y, margin_x = max(1, h // 20), max(1, w // 20)
    low_mask[:margin_y, :] = False
    low_mask[:, :margin_x] = False
    low_mask[:, -margin_x:] = False
    low_mask[-margin_y:, :] = False
    labeled, num_features = label(low_mask)
    if num_features == 0:
        out = np.zeros_like(image, dtype=bool)
        out[10:20, 10:20] = True
        return out
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = np.argmax(sizes)
    if sizes[largest_label] == 0:
        out = np.zeros_like(image, dtype=bool)
        out[10:20, 10:20] = True
        return out
    out = np.zeros_like(image, dtype=bool)
    out[labeled == largest_label] = True
    return out


def _parse_acq_time(acq_time_str: str) -> datetime:
    """Parse FLIM acqTime string to datetime."""
    return datetime.fromisoformat(acq_time_str.strip())


def _get_acq_time_str(file_path: str, frame_idx: int = 0) -> str:
    """Get raw acqTime string for one frame from FLIM file (tags only)."""
    iminfo = FileReader()
    iminfo.read_imageFile(file_path, False)
    if not iminfo.acqTime or frame_idx >= len(iminfo.acqTime):
        return ""
    return str(iminfo.acqTime[frame_idx]).strip()


def _get_frame_times_from_flim(file_path: str) -> list:
    """
    Get acquisition times for each frame from FLIM file metadata.
    Returns list of times in seconds relative to first frame (first frame = 0).
    """
    iminfo = FileReader()
    iminfo.read_imageFile(file_path, False)
    if not iminfo.acqTime or len(iminfo.acqTime) == 0:
        return []
    try:
        first_time = _parse_acq_time(iminfo.acqTime[0])
        frame_times = []
        for acq_time_str in iminfo.acqTime:
            frame_time = _parse_acq_time(acq_time_str)
            delta = (frame_time - first_time).total_seconds()
            frame_times.append(delta)
        return frame_times
    except Exception as e:
        print(f"  Warning: Could not parse frame times: {e}")
        return []


def _load_uncaging_full(unc_path: str, ch: int):
    """
    Load uncaging FLIM file at full size (no crop).
    Returns list of 2D frames (Y, X): either Z-slices (if Z>1) or time frames (if Z==1).
    """
    iminfo = FileReader()
    iminfo.read_imageFile(unc_path, True)
    imagearray = np.array(iminfo.image)
    intensity = (12 * np.sum(imagearray, axis=-1)) / getattr(iminfo.State.Acq, "nAveFrame", 1)
    tyx = intensity[:, :, ch - 1, :, :].astype(np.float32)
    T, Z, H, W = tyx.shape
    if Z > 1:
        frames = [tyx[0, z].copy() for z in range(Z)]
    else:
        frames = [tyx[t, 0].copy() for t in range(T)]
    return frames


def _load_flim_zproj_full(file_path: str, ch: int, z_from: int, z_to: int):
    """
    Load one FLIM file and return full-size Z-projection image for channel `ch`.
    Uses T=0 and max projection over z_from:z_to.
    """
    iminfo = FileReader()
    iminfo.read_imageFile(file_path, True)
    imagearray = np.array(iminfo.image)
    intensity = (12 * np.sum(imagearray, axis=-1)) / getattr(iminfo.State.Acq, "nAveFrame", 1)
    # Keep axis handling consistent with quantification (_row_from_flim_data):
    # axis-0 is treated as the projection axis for pre/post frames.
    intensity_raw = intensity.astype(np.float32)  # (axis0, axis1, C, Y, X)
    axis0_len = intensity_raw.shape[0]
    z0 = max(0, min(int(z_from), axis0_len - 1))
    z1 = min(axis0_len, int(z_to))
    z1 = max(z1, z0 + 1)
    return intensity_raw[z0:z1, 0, ch - 1, :, :].max(axis=0)


def _load_uncaging_crop(unc_path: str, ch: int,
                        small_x_from: int, small_x_to: int,
                        small_y_from: int, small_y_to: int,
                        small_z_from: int, small_z_to: int):
    """
    Load uncaging FLIM file and crop with the same region as pre (small_*).
    flim_files_to_nparray skips files with different shape, so we load uncaging separately.
    Returns (n_frames, Y_crop, X_crop) as list of 2D frames: either Z-slices (if Z>1) or
    time frames (if Z==1), so "3D unprojected" is preserved as multiple 2D frames.
    """
    iminfo = FileReader()
    iminfo.read_imageFile(unc_path, True)
    imagearray = np.array(iminfo.image)
    # (T, Z, C, Y, X, bins) -> sum bins, take channel
    intensity = (12 * np.sum(imagearray, axis=-1)) / getattr(iminfo.State.Acq, "nAveFrame", 1)
    # (T, Z, C, Y, X)
    tyx = intensity[:, :, ch - 1, :, :].astype(np.float32)
    T, Z, H, W = tyx.shape
    # Clamp crop to array bounds
    x0 = max(0, min(small_x_from, W - 1))
    x1 = min(W, small_x_to)
    y0 = max(0, min(small_y_from, H - 1))
    y1 = min(H, small_y_to)
    z0 = max(0, min(small_z_from, Z - 1))
    z1 = min(Z, small_z_to)
    if x0 >= x1 or y0 >= y1:
        return []
    crop = tyx[:, z0:z1, y0:y1, x0:x1]
    # crop (T, Z_crop, Y_crop, X_crop)
    if crop.shape[1] > 1:
        # Multiple Z: use first time point, one 2D frame per Z
        frames = [crop[0, z].copy() for z in range(crop.shape[1])]
    else:
        # Single Z (e.g. 55-frame time series): one 2D frame per time
        frames = [crop[t, 0].copy() for t in range(crop.shape[0])]
    return frames


def rebuild_tiff_with_uncaging_3d(combined_df: pd.DataFrame, ch: int):
    """
    For each set, rebuild the after_align TIFF as:
      [Pre Z-proj..., Uncaging (unprojected 3D as multiple 2D frames), Post Z-proj...]

    - Trimming: same small_* crop (from pre/post aligned stack) is used for uncaging.
    - Drift: uncaging is not in the pre/post alignment. We load uncaging with the same
      crop box (small_*), then align the uncaging crop to "uncaging直前のpre" (last pre
      frame) via phase_cross_correlation and apply that shift to every uncaging frame
      so the stack is in the same coordinate system as pre/post.
    - Uncaging is loaded separately because flim_files_to_nparray skips different n_images.
    """
    required_cols = ["small_x_from", "small_x_to", "small_y_from", "small_y_to", "small_z_from", "small_z_to"]
    if not all(c in combined_df.columns for c in required_cols):
        print("Warning: combined_df missing small_* columns (run first_processing first). Skipping rebuild.")
        return combined_df

    for each_filepath_without_number in combined_df['filepath_without_number'].unique():
        each_filegroup_df = combined_df[combined_df['filepath_without_number'] == each_filepath_without_number]
        folder = os.path.dirname(each_filepath_without_number)
        tif_savefolder = os.path.join(folder, "tif")
        os.makedirs(tif_savefolder, exist_ok=True)

        for each_group in each_filegroup_df['group'].unique():
            each_group_df = each_filegroup_df[each_filegroup_df['group'] == each_group]

            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]

                uncaging_rows = each_set_df[each_set_df["phase"] == "unc"]
                if len(uncaging_rows) == 0:
                    print(f"  Set {each_set_label}: no uncaging row, skip rebuild")
                    continue
                uncaging_row = uncaging_rows.iloc[0]
                unc_path = uncaging_row["file_path"]
                if not os.path.exists(unc_path):
                    print(f"  Set {each_set_label}: uncaging file not found: {unc_path}")
                    continue

                first_row = each_set_df.iloc[0]
                small_x_from = int(first_row["small_x_from"])
                small_x_to = int(first_row["small_x_to"])
                small_y_from = int(first_row["small_y_from"])
                small_y_to = int(first_row["small_y_to"])
                small_z_from = int(first_row["small_z_from"])
                small_z_to = int(first_row["small_z_to"])

                old_tiff_path = first_row["after_align_save_path"]
                if pd.isna(old_tiff_path) or not os.path.exists(old_tiff_path):
                    print(f"  Set {each_set_label}: missing after_align TIFF, skip")
                    continue

                current_tiff = tifffile.imread(old_tiff_path)
                n_pre = len(each_set_df[each_set_df["phase"] == "pre"])
                n_post = len(each_set_df[each_set_df["phase"] == "post"])
                n_total = current_tiff.shape[0]
                # Use available frames: first n_pre as pre, rest as post (may be less than n_post)
                n_pre_use = min(n_pre, n_total)
                n_post_use = n_total - n_pre_use
                if n_total == 0:
                    print(f"  Set {each_set_label}: after_align TIFF has 0 frames, skip")
                    continue
                if n_pre_use < n_pre or n_post_use < n_post:
                    print(f"  Set {each_set_label}: TIFF has {n_total} frames (pre {n_pre_use}, post {n_post_use}); expected pre {n_pre}, post {n_post}")

                pre_zproj = current_tiff[:n_pre_use]
                post_zproj = current_tiff[n_pre_use:]

                try:
                    uncaging_frames = _load_uncaging_crop(
                        unc_path, ch,
                        small_x_from, small_x_to,
                        small_y_from, small_y_to,
                        small_z_from, small_z_to,
                    )
                except Exception as e:
                    print(f"  Set {each_set_label}: failed to load uncaging crop: {e}")
                    continue
                if not uncaging_frames:
                    print(f"  Set {each_set_label}: no uncaging frames after crop, skip")
                    continue

                # Resize uncaging frames to match pre_zproj XY if needed
                hw_pre = (pre_zproj.shape[1], pre_zproj.shape[2])
                unc_stack = []
                for f in uncaging_frames:
                    if (f.shape[0], f.shape[1]) != hw_pre:
                        from skimage.transform import resize
                        f = resize(f, hw_pre, order=1, preserve_range=True).astype(np.float32)
                    unc_stack.append(f)
                unc_stack = np.stack(unc_stack, axis=0)

                # Apply pre/post alignment to uncaging: align uncaging crop to "uncaging直前のpre"
                # (last pre frame). Shift is computed in crop space; apply to each uncaging frame.
                if n_pre_use > 0 and unc_stack.shape[0] > 0:
                    ref_2d = pre_zproj[-1].astype(np.float64)
                    query_2d = unc_stack[0].astype(np.float64)
                    try:
                        shift_2d, _, _ = phase_cross_correlation(ref_2d, query_2d, upsample_factor=4)
                        # shift_2d is (dy, dx). Apply to each uncaging frame so they match pre coords.
                        unc_aligned = []
                        for k in range(unc_stack.shape[0]):
                            f_fft = np.fft.fftn(unc_stack[k].astype(np.float64))
                            f_shifted = _fourier_shift(f_fft, shift_2d)
                            f_aligned = np.fft.ifftn(f_shifted).real.astype(np.float32)
                            unc_aligned.append(f_aligned)
                        unc_stack = np.stack(unc_aligned, axis=0)
                    except Exception as e:
                        print(f"  Set {each_set_label}: uncaging align to pre failed ({e}), using unaligned crop")

                new_stack = np.concatenate([pre_zproj, unc_stack, post_zproj], axis=0)
                total_frames = new_stack.shape[0]
                if TEST_MODE and total_frames < 56:
                    print(f"  [TEST] Set {each_set_label}: total frames = {total_frames} (expected >= 56 for 55-frame uncaging)")

                base_name = os.path.splitext(os.path.basename(old_tiff_path))[0]
                new_name = base_name + TIFF_WITH_UNCAGING_SUFFIX + ".tif"
                new_tiff_path = os.path.join(tif_savefolder, new_name)
                tifffile.imwrite(new_tiff_path, new_stack.astype(np.float32))
                print(f"  Set {each_set_label}: saved {new_name} (Pre {n_pre_use} + Uncaging {len(uncaging_frames)} + Post {n_post_use}) -> total {total_frames}")

                combined_df.loc[each_set_df.index, "after_align_save_path"] = new_tiff_path
                combined_df.loc[each_set_df.index, "n_pre_frames"] = n_pre_use
                combined_df.loc[each_set_df.index, "n_unc_frames"] = len(uncaging_frames)
                combined_df.loc[each_set_df.index, "n_post_frames"] = n_post_use

    return combined_df


def rebuild_tiff_full_size_for_roi(
    combined_df: pd.DataFrame,
    ch: int,
    z_plus_minus: int,
    skip_tiff_if_exists: bool = False,
):
    """
    Build full-size stack per set for ROI definition: Pre Z-proj (full) + Uncaging
    (full, drift applied to last pre) + Post Z-proj (full). Saves to after_align_full_save_path.
    Stores unc_drift_y, unc_drift_x on the uncaging row for later raw-ROI generation.

    Args:
        skip_tiff_if_exists: If True, skip writing TIFF files that already exist on disk.
            Alignment (load_and_align_data) is always run to regenerate frame_info.csv with
            correct runtime shifts. This is faster when TIFFs are already built but
            frame_info.csv needs to be refreshed.
    """
    required = ["corrected_uncaging_z", "small_x_from", "small_x_to"]
    if not all(c in combined_df.columns for c in required):
        print("Warning: combined_df missing required columns. Skipping full-size build.")
        return combined_df

    for each_filepath_without_number in combined_df["filepath_without_number"].unique():
        each_filegroup_df = combined_df[combined_df["filepath_without_number"] == each_filepath_without_number]
        folder = os.path.dirname(each_filepath_without_number)
        tif_savefolder = os.path.join(folder, "tif")
        os.makedirs(tif_savefolder, exist_ok=True)

        for each_group in each_filegroup_df["group"].unique():
            each_group_df = each_filegroup_df[each_filegroup_df["group"] == each_group]
            pre_post_df = each_group_df[each_group_df["phase"].isin(["pre", "post"])].sort_values("nth_omit_induction")
            filelist = pre_post_df["file_path"].tolist()
            if len(filelist) == 0:
                continue
            try:
                Aligned_4d_array, shifts, _ = load_and_align_data(filelist, ch=ch - 1)
            except Exception as e:
                print(f"  Group {each_group}: load_and_align_data failed: {e}")
                continue
            _, Z_full, Y_full, X_full = Aligned_4d_array.shape
            # Map file_path -> index in Aligned_4d_array (filelist order); needed when group has multiple sets
            file_path_to_array_idx = {path: i for i, path in enumerate(filelist)}
            # Runtime alignment shifts from load_and_align_data (same order as filelist):
            # shifts[:, 1] = shift_y, shifts[:, 2] = shift_x
            runtime_shift_map = {}
            try:
                for i, path in enumerate(filelist):
                    if i < len(shifts):
                        runtime_shift_map[path] = (
                            float(shifts[i][1]) if len(shifts[i]) > 1 else 0.0,
                            float(shifts[i][2]) if len(shifts[i]) > 2 else 0.0,
                        )
            except Exception:
                runtime_shift_map = {}

            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
                uncaging_rows = each_set_df[each_set_df["phase"] == "unc"]
                if len(uncaging_rows) == 0:
                    continue
                unc_path = uncaging_rows.iloc[0]["file_path"]
                if not os.path.exists(unc_path):
                    continue
                corrected_uncaging_z = int(each_set_df["corrected_uncaging_z"].values[0])
                z_from = max(0, min(corrected_uncaging_z - z_plus_minus, Z_full - 1))
                z_to = min(Z_full, corrected_uncaging_z + z_plus_minus + 1)
                z_to = max(z_to, z_from + 1)

                # Use each row's file_path -> array index so pre/post order matches group filelist (time series)
                pre_list = []
                pre_raw_list = []
                pre_filenames = []
                pre_file_paths = []
                pre_shifts = []
                for _, row in each_set_df[each_set_df["phase"] == "pre"].sort_values("nth_omit_induction").iterrows():
                    fp = row["file_path"]
                    if fp not in file_path_to_array_idx:
                        continue
                    array_idx = file_path_to_array_idx[fp]
                    zproj = Aligned_4d_array[array_idx, z_from:z_to, :, :].max(axis=0)
                    pre_list.append(zproj)
                    pre_filenames.append(os.path.basename(fp))
                    pre_file_paths.append(fp)
                    if fp in runtime_shift_map:
                        pre_shifts.append(runtime_shift_map[fp])
                    else:
                        pre_shifts.append((float(row.get("shift_y", 0) or 0), float(row.get("shift_x", 0) or 0)))
                    try:
                        pre_raw_list.append(_load_flim_zproj_full(fp, ch, z_from, z_to))
                    except Exception:
                        pre_raw_list.append(zproj.copy())
                pre_stack = np.stack(pre_list, axis=0) if pre_list else np.empty((0, Y_full, X_full), dtype=np.float32)
                pre_raw_stack = np.stack(pre_raw_list, axis=0) if pre_raw_list else np.empty((0, Y_full, X_full), dtype=np.float32)

                post_list = []
                post_raw_list = []
                post_filenames = []
                post_file_paths = []
                post_shifts = []
                for _, row in each_set_df[each_set_df["phase"] == "post"].sort_values("nth_omit_induction").iterrows():
                    fp = row["file_path"]
                    if fp not in file_path_to_array_idx:
                        continue
                    array_idx = file_path_to_array_idx[fp]
                    zproj = Aligned_4d_array[array_idx, z_from:z_to, :, :].max(axis=0)
                    post_list.append(zproj)
                    post_filenames.append(os.path.basename(fp))
                    post_file_paths.append(fp)
                    if fp in runtime_shift_map:
                        post_shifts.append(runtime_shift_map[fp])
                    else:
                        post_shifts.append((float(row.get("shift_y", 0) or 0), float(row.get("shift_x", 0) or 0)))
                    try:
                        post_raw_list.append(_load_flim_zproj_full(fp, ch, z_from, z_to))
                    except Exception:
                        post_raw_list.append(zproj.copy())
                post_stack = np.stack(post_list, axis=0) if post_list else np.empty((0, Y_full, X_full), dtype=np.float32)
                post_raw_stack = np.stack(post_raw_list, axis=0) if post_raw_list else np.empty((0, Y_full, X_full), dtype=np.float32)

                try:
                    unc_frames_full = _load_uncaging_full(unc_path, ch)
                except Exception as e:
                    print(f"  Set {each_set_label}: _load_uncaging_full failed: {e}")
                    continue
                if not unc_frames_full:
                    continue
                unc_stack = np.stack(unc_frames_full, axis=0)
                if unc_stack.shape[1] != Y_full or unc_stack.shape[2] != X_full:
                    from skimage.transform import resize
                    unc_resized = []
                    for k in range(unc_stack.shape[0]):
                        f = resize(unc_stack[k], (Y_full, X_full), order=1, preserve_range=True).astype(np.float32)
                        unc_resized.append(f)
                    unc_stack = np.stack(unc_resized, axis=0)

                unc_drift_y, unc_drift_x = 0.0, 0.0
                if pre_stack.shape[0] > 0 and unc_stack.shape[0] > 0:
                    ref_2d = pre_stack[-1].astype(np.float64)
                    query_2d = unc_stack[0].astype(np.float64)
                    try:
                        shift_2d, _, _ = phase_cross_correlation(ref_2d, query_2d, upsample_factor=4)
                        unc_drift_y, unc_drift_x = float(shift_2d[0]), float(shift_2d[1])
                        unc_aligned = []
                        for k in range(unc_stack.shape[0]):
                            f_fft = np.fft.fftn(unc_stack[k].astype(np.float64))
                            f_shifted = _fourier_shift(f_fft, shift_2d)
                            f_aligned = np.fft.ifftn(f_shifted).real.astype(np.float32)
                            unc_aligned.append(f_aligned)
                        unc_stack = np.stack(unc_aligned, axis=0)
                    except Exception as e:
                        print(f"  Set {each_set_label}: uncaging align failed ({e}), using unaligned")

                unc_row_idx = each_set_df[each_set_df["phase"] == "unc"].index[0]
                combined_df.loc[unc_row_idx, "unc_drift_y"] = unc_drift_y
                combined_df.loc[unc_row_idx, "unc_drift_x"] = unc_drift_x

                new_stack = np.concatenate([pre_stack, unc_stack, post_stack], axis=0)
                base_name = f"{each_group}_{each_set_label}_after_align_full"
                new_tiff_path = os.path.join(tif_savefolder, base_name + ".tif")
                if not skip_tiff_if_exists or not os.path.exists(new_tiff_path):
                    tifffile.imwrite(new_tiff_path, new_stack.astype(np.float32))
                    print(f"  Set {each_set_label}: saved full-size {base_name}.tif (Pre {pre_stack.shape[0]} + Unc {unc_stack.shape[0]} + Post {post_stack.shape[0]})")
                else:
                    print(f"  Set {each_set_label}: {base_name}.tif already exists, skipping write (skip_tiff_if_exists=True)")
                combined_df.loc[each_set_df.index, "after_align_full_save_path"] = new_tiff_path
                before_stack = np.concatenate([pre_raw_stack, np.stack(unc_frames_full, axis=0), post_raw_stack], axis=0)
                before_base_name = f"{each_group}_{each_set_label}_before_align_full"
                before_tiff_path = os.path.join(tif_savefolder, before_base_name + ".tif")
                if not skip_tiff_if_exists or not os.path.exists(before_tiff_path):
                    tifffile.imwrite(before_tiff_path, before_stack.astype(np.float32))
                    print(f"  Set {each_set_label}: saved full-size {before_base_name}.tif (non-aligned)")
                else:
                    print(f"  Set {each_set_label}: {before_base_name}.tif already exists, skipping write")
                combined_df.loc[each_set_df.index, "before_align_full_save_path"] = before_tiff_path
                combined_df.loc[each_set_df.index, "n_pre_frames"] = pre_stack.shape[0]
                combined_df.loc[each_set_df.index, "n_unc_frames"] = unc_stack.shape[0]
                combined_df.loc[each_set_df.index, "n_post_frames"] = post_stack.shape[0]
                # Frame order for verification: 0..n_pre-1=pre, n_pre..n_pre+n_unc-1=uncaging, then post
                print(f"    Frame order: 0-{pre_stack.shape[0]-1}=pre, {pre_stack.shape[0]}-{pre_stack.shape[0]+unc_stack.shape[0]-1}=uncaging, {pre_stack.shape[0]+unc_stack.shape[0]}-{new_stack.shape[0]-1}=post")
                # Per-frame source CSV for easy verification (test use); include acq_time_str and elapsed_time_sec
                unc_basename = os.path.basename(unc_path)
                iminfo_unc = FileReader()
                iminfo_unc.read_imageFile(unc_path, False)
                acq_time_unc = list(iminfo_unc.acqTime) if iminfo_unc.acqTime else []
                frame_info_rows = []
                for i in range(pre_stack.shape[0]):
                    acq_str = _get_acq_time_str(pre_file_paths[i], 0) if i < len(pre_file_paths) else ""
                    sy, sx = pre_shifts[i] if i < len(pre_shifts) else (np.nan, np.nan)
                    frame_info_rows.append({"frame": i, "Z_projection": True, "nth_slice": "", "filename": pre_filenames[i] if i < len(pre_filenames) else "", "phase": "pre", "acq_time_str": acq_str, "z_from": z_from, "z_to": z_to, "shift_y": sy, "shift_x": sx})
                for k in range(unc_stack.shape[0]):
                    acq_str = acq_time_unc[k] if k < len(acq_time_unc) else ""
                    if acq_str:
                        acq_str = str(acq_str).strip()
                    frame_info_rows.append({"frame": pre_stack.shape[0] + k, "Z_projection": False, "nth_slice": k, "filename": unc_basename, "phase": "uncaging", "acq_time_str": acq_str, "z_from": np.nan, "z_to": np.nan, "shift_y": unc_drift_y, "shift_x": unc_drift_x})
                for i in range(post_stack.shape[0]):
                    acq_str = _get_acq_time_str(post_file_paths[i], 0) if i < len(post_file_paths) else ""
                    sy, sx = post_shifts[i] if i < len(post_shifts) else (np.nan, np.nan)
                    frame_info_rows.append({"frame": pre_stack.shape[0] + unc_stack.shape[0] + i, "Z_projection": True, "nth_slice": "", "filename": post_filenames[i] if i < len(post_filenames) else "", "phase": "post", "acq_time_str": acq_str, "z_from": z_from, "z_to": z_to, "shift_y": sy, "shift_x": sx})
                # elapsed_time_sec: per set, seconds from first moment (min parsed acq time = 0)
                datetimes_list = []
                for r in frame_info_rows:
                    s = r.get("acq_time_str", "") or ""
                    if s:
                        try:
                            datetimes_list.append(_parse_acq_time(s))
                        except Exception:
                            datetimes_list.append(None)
                    else:
                        datetimes_list.append(None)
                t0 = min((dt for dt in datetimes_list if dt is not None), default=None)
                for idx, r in enumerate(frame_info_rows):
                    if t0 is not None and datetimes_list[idx] is not None:
                        r["elapsed_time_sec"] = (datetimes_list[idx] - t0).total_seconds()
                    else:
                        r["elapsed_time_sec"] = np.nan
                frame_info_path = os.path.join(tif_savefolder, base_name + "_frame_info.csv")
                pd.DataFrame(frame_info_rows).to_csv(frame_info_path, index=False)
                print(f"    Saved {base_name}_frame_info.csv")

    return combined_df


def _rebuild_tiff_uncaging_only_for_roi(combined_df: pd.DataFrame, ch: int, z_plus_minus: int):
    """
    Build full-size stack per set containing only uncaging frames (no pre/post, no alignment).
    Saves to after_align_full_save_path and after_align_save_path.
    Sets n_pre_frames=0, n_post_frames=0, n_unc_frames=N. Unc drift is set to 0.
    Used by run_tiff_uncaging_roi_no_zstack.
    """
    required = ["corrected_uncaging_z", "corrected_uncaging_x", "corrected_uncaging_y"]
    if not all(c in combined_df.columns for c in required):
        print("Warning: combined_df missing required columns. Skipping uncaging-only build.")
        return combined_df

    for each_filepath_without_number in combined_df["filepath_without_number"].unique():
        each_filegroup_df = combined_df[combined_df["filepath_without_number"] == each_filepath_without_number]
        folder = os.path.dirname(each_filepath_without_number)
        tif_savefolder = os.path.join(folder, "tif")
        os.makedirs(tif_savefolder, exist_ok=True)

        for each_group in each_filegroup_df["group"].unique():
            each_group_df = each_filegroup_df[each_filegroup_df["group"] == each_group]

            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
                uncaging_rows = each_set_df[each_set_df["phase"] == "unc"]
                if len(uncaging_rows) == 0:
                    continue
                unc_path = uncaging_rows.iloc[0]["file_path"]
                if not os.path.exists(unc_path):
                    continue

                try:
                    unc_frames_full = _load_uncaging_full(unc_path, ch)
                except Exception as e:
                    print(f"  Set {each_set_label}: _load_uncaging_full failed: {e}")
                    continue
                if not unc_frames_full:
                    continue
                unc_stack = np.stack(unc_frames_full, axis=0)

                unc_row_idx = each_set_df[each_set_df["phase"] == "unc"].index[0]
                combined_df.loc[unc_row_idx, "unc_drift_y"] = 0.0
                combined_df.loc[unc_row_idx, "unc_drift_x"] = 0.0

                base_name = f"{each_group}_{each_set_label}_unc_only"
                new_tiff_path = os.path.join(tif_savefolder, base_name + ".tif")
                tifffile.imwrite(new_tiff_path, unc_stack.astype(np.float32))
                combined_df.loc[each_set_df.index, "after_align_full_save_path"] = new_tiff_path
                combined_df.loc[each_set_df.index, "after_align_save_path"] = new_tiff_path
                combined_df.loc[each_set_df.index, "n_pre_frames"] = 0
                combined_df.loc[each_set_df.index, "n_unc_frames"] = unc_stack.shape[0]
                combined_df.loc[each_set_df.index, "n_post_frames"] = 0
                print(f"  Set {each_set_label}: saved {base_name}.tif (Unc only: {unc_stack.shape[0]} frames)")

    return combined_df


def _build_initial_roi_masks_for_set(
    inipath: str,
    unc_stack: np.ndarray,
    tiff_dir: str,
    base_name: str,
    circle_radius: int = DEFAULT_CIRCLE_RADIUS,
    rect_length: int = DEFAULT_RECT_LENGTH,
    rect_height: int = DEFAULT_RECT_HEIGHT,
    bg_percentile: float = DEFAULT_BG_PERCENTILE,
) -> None:
    """
    Create Spine, DendriticShaft, and Background ROI mask TIFFs for one set.
    Same geometry as calc_spine_dend_GCaMP (circle for spine, rectangle for shaft, low-intensity for bg).
    Writes {base_name}_Spine_roi_mask.tif, _DendriticShaft_roi_mask.tif, _Background_roi_mask.tif.
    If inipath is None or empty, only Background is created.
    """
    import sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    _foruse = os.path.join(_root, "ForUse", "temporal_use_1")
    if _foruse not in sys.path:
        sys.path.insert(0, _foruse)
    from multidim_tiff_viewer import read_xyz_single
    from skimage.draw import disk, polygon

    n_frames, h, w = unc_stack.shape
    max_proj = unc_stack.max(axis=0)
    spine_mask_2d = None
    shaft_mask_2d = None
    if inipath and os.path.exists(inipath):
        try:
            from flimage_graph_func import calc_point_on_line_close_to_xy
            spine_zyx, dend_slope, dend_intercept, _ = read_xyz_single(inipath, return_excluded=True)
            spine_x, spine_y = float(spine_zyx[2]), float(spine_zyx[1])
            y_c, x_c = calc_point_on_line_close_to_xy(spine_x, spine_y, dend_slope, dend_intercept)
            rr_circ, cc_circ = disk((spine_y, spine_x), circle_radius, shape=(h, w))
            spine_mask_2d = np.zeros((h, w), dtype=bool)
            spine_mask_2d[rr_circ, cc_circ] = True
            theta = np.arctan(dend_slope)
            dx, dy = (rect_length / 2) * np.cos(theta), (rect_length / 2) * np.sin(theta)
            px, py = (rect_height / 2) * -np.sin(theta), (rect_height / 2) * np.cos(theta)
            corners_x = [x_c - dx - px, x_c - dx + px, x_c + dx + px, x_c + dx - px]
            corners_y = [y_c - dy - py, y_c - dy + py, y_c + dy + py, y_c + dy - py]
            rr_rect, cc_rect = polygon(corners_y, corners_x, shape=(h, w))
            shaft_mask_2d = np.zeros((h, w), dtype=bool)
            shaft_mask_2d[rr_rect, cc_rect] = True
        except Exception as e:
            print(f"    INI ROI build failed: {e}")
            spine_mask_2d = shaft_mask_2d = None
    bg_mask_2d = _low_intensity_largest_mask_2d(max_proj, bg_percentile)
    for roi_type, mask_2d in [("Spine", spine_mask_2d), ("DendriticShaft", shaft_mask_2d), ("Background", bg_mask_2d)]:
        if mask_2d is None:
            continue
        path = os.path.join(tiff_dir, f"{base_name}_{roi_type}_roi_mask.tif")
        tifffile.imwrite(path, np.stack([mask_2d.astype(np.uint8)] * n_frames, axis=0), photometric="minisblack")
        print(f"    {base_name}: saved {roi_type} ROI -> {os.path.basename(path)}")


def create_initial_roi_masks_from_ini(
    combined_df: pd.DataFrame,
    max_distance_pix: float = 20.0,
    circle_radius: int = DEFAULT_CIRCLE_RADIUS,
    rect_length: int = DEFAULT_RECT_LENGTH,
    rect_height: int = DEFAULT_RECT_HEIGHT,
    bg_percentile: float = DEFAULT_BG_PERCENTILE,
) -> None:
    """
    Standalone: build Spine / DendriticShaft / Background ROI mask TIFFs from INI for all sets in combined_df.
    Uses combined_df to get file paths (after_align_save_path, file_path for unc row), finds matching INI
    per set via flim_ini_match_by_uncaging_pos.get_matching_ini_for_flim, then writes *_roi_mask.tif
    next to each set's TIFF. Fully independent of run_tiff_uncaging_roi_no_zstack.

    Example:
        combined_df = pd.read_pickle(r"path/to/combined_df_1.pkl")
        create_initial_roi_masks_from_ini(combined_df, max_distance_pix=20)
    """
    required = ["filepath_without_number", "group", "nth_set_label", "phase", "file_path", "after_align_save_path"]
    missing = [c for c in required if c not in combined_df.columns]
    if missing:
        print(f"create_initial_roi_masks_from_ini: combined_df missing columns: {missing}. Skipping.")
        return
    import sys
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    _foruse = os.path.join(_root, "ForUse", "temporal_use_1")
    if _foruse not in sys.path:
        sys.path.insert(0, _foruse)
    try:
        from flim_ini_match_by_uncaging_pos import get_matching_ini_for_flim
    except ImportError as e:
        print(f"create_initial_roi_masks_from_ini: import failed: {e}. Skipping.")
        return
    print("Creating initial ROI masks from INI (Spine, DendriticShaft, Background)...")
    for each_filepath_without_number in combined_df["filepath_without_number"].unique():
        each_filegroup_df = combined_df[combined_df["filepath_without_number"] == each_filepath_without_number]
        for each_group in each_filegroup_df["group"].unique():
            each_group_df = each_filegroup_df[each_filegroup_df["group"] == each_group]
            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
                tiff_path = each_set_df["after_align_save_path"].iloc[0]
                if pd.isna(tiff_path) or not os.path.exists(tiff_path):
                    print(f"  Set {each_group}_{each_set_label}: TIFF not found, skip")
                    continue
                unc_rows = each_set_df[each_set_df["phase"] == "unc"]
                if len(unc_rows) == 0:
                    print(f"  Set {each_group}_{each_set_label}: no unc row, skip")
                    continue
                unc_path = unc_rows.iloc[0]["file_path"]
                if not os.path.exists(unc_path):
                    print(f"  Set {each_group}_{each_set_label}: unc FLIM not found, skip")
                    continue
                inipath = get_matching_ini_for_flim(unc_path, max_distance_pix=max_distance_pix)
                if inipath is None:
                    print(f"  Set {each_group}_{each_set_label}: no matching INI (max_dist={max_distance_pix} px), Background only")
                try:
                    unc_stack = tifffile.imread(tiff_path)
                    if unc_stack.ndim == 2:
                        unc_stack = unc_stack[np.newaxis, ...]
                    _build_initial_roi_masks_for_set(
                        inipath=inipath or "",
                        unc_stack=unc_stack,
                        tiff_dir=os.path.dirname(tiff_path),
                        base_name=os.path.splitext(os.path.basename(tiff_path))[0],
                        circle_radius=circle_radius,
                        rect_length=rect_length,
                        rect_height=rect_height,
                        bg_percentile=bg_percentile,
                    )
                except Exception as e:
                    print(f"  Set {each_group}_{each_set_label}: failed: {e}")
    print("create_initial_roi_masks_from_ini: done.")


def _get_shift_per_stack_frame(each_set_df: pd.DataFrame, n_pre: int, n_unc: int, n_post: int):
    """
    Return list of (shift_y, shift_x) for each stack frame index 0..n_pre+n_unc+n_post-1.
    Pre/post use row shift_y, shift_x; uncaging uses unc_drift_y, unc_drift_x.
    """
    shifts = []
    pre_df = each_set_df[each_set_df["phase"] == "pre"].sort_values("nth_omit_induction")
    for _, row in pre_df.iterrows():
        sy = float(row.get("shift_y", 0))
        sx = float(row.get("shift_x", 0))
        shifts.append((sy, sx))
    unc_row = each_set_df[each_set_df["phase"] == "unc"].iloc[0]
    unc_drift_y = float(unc_row.get("unc_drift_y", 0))
    unc_drift_x = float(unc_row.get("unc_drift_x", 0))
    for _ in range(n_unc):
        shifts.append((unc_drift_y, unc_drift_x))
    post_df = each_set_df[each_set_df["phase"] == "post"].sort_values("nth_omit_induction")
    for _, row in post_df.iterrows():
        sy = float(row.get("shift_y", 0))
        sx = float(row.get("shift_x", 0))
        shifts.append((sy, sx))
    return shifts


def save_drift_corrected_roi_masks(combined_df: pd.DataFrame):
    """
    For each set with after_align_full_save_path, load aligned ROI masks (Type A),
    apply inverse drift per frame so ROI fits pre-drift FLIM, save as Type B
    (*_roi_mask_raw.tif). Use Type B for quantification from FLIM.
    """
    from scipy.ndimage import shift as ndimage_shift

    def _mask_stats(mask_2d: np.ndarray):
        """Return (nonzero_count, centroid_y, centroid_x) for binary-like mask."""
        ys, xs = np.where(mask_2d > 0.5)
        if ys.size == 0:
            return 0, np.nan, np.nan
        return int(ys.size), float(np.mean(ys)), float(np.mean(xs))

    def _build_shift_list_from_frame_info(each_set_df: pd.DataFrame, frame_info_df: pd.DataFrame, n_total: int):
        """
        Build per-frame (shift_y, shift_x) using actual frame order in frame_info.
        Priority:
          1) frame_info has shift_y/shift_x columns -> use directly.
          2) frame_info phase order -> map pre/post by sorted nth_omit_induction, uncaging by unc_drift.
        """
        if frame_info_df is None or len(frame_info_df) < n_total:
            return None
        # 1) direct use when available
        if "shift_y" in frame_info_df.columns and "shift_x" in frame_info_df.columns:
            try:
                out = []
                for i in range(n_total):
                    sy = float(frame_info_df.iloc[i].get("shift_y", 0) or 0)
                    sx = float(frame_info_df.iloc[i].get("shift_x", 0) or 0)
                    out.append((sy, sx))
                return out
            except Exception:
                pass
        # 2) infer by frame_info phase sequence
        if "phase" not in frame_info_df.columns:
            return None
        pre_df = each_set_df[each_set_df["phase"] == "pre"].sort_values("nth_omit_induction").reset_index(drop=True)
        post_df = each_set_df[each_set_df["phase"] == "post"].sort_values("nth_omit_induction").reset_index(drop=True)
        unc_rows = each_set_df[each_set_df["phase"] == "unc"]
        unc_drift_y, unc_drift_x = 0.0, 0.0
        if len(unc_rows) > 0:
            unc_drift_y = float(unc_rows.iloc[0].get("unc_drift_y", 0) or 0)
            unc_drift_x = float(unc_rows.iloc[0].get("unc_drift_x", 0) or 0)
        out = []
        pre_i = 0
        post_i = 0
        for i in range(n_total):
            phase = str(frame_info_df.iloc[i].get("phase", "")).lower()
            if phase == "pre":
                if pre_i < len(pre_df):
                    sy = float(pre_df.iloc[pre_i].get("shift_y", 0) or 0)
                    sx = float(pre_df.iloc[pre_i].get("shift_x", 0) or 0)
                    pre_i += 1
                else:
                    sy, sx = 0.0, 0.0
            elif phase in ("uncaging", "unc"):
                sy, sx = unc_drift_y, unc_drift_x
            elif phase == "post":
                if post_i < len(post_df):
                    sy = float(post_df.iloc[post_i].get("shift_y", 0) or 0)
                    sx = float(post_df.iloc[post_i].get("shift_x", 0) or 0)
                    post_i += 1
                else:
                    sy, sx = 0.0, 0.0
            else:
                sy, sx = 0.0, 0.0
            out.append((sy, sx))
        return out

    for each_filepath_without_number in combined_df["filepath_without_number"].unique():
        each_filegroup_df = combined_df[combined_df["filepath_without_number"] == each_filepath_without_number]
        for each_group in each_filegroup_df["group"].unique():
            each_group_df = each_filegroup_df[each_filegroup_df["group"] == each_group]
            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
                tiff_path = each_set_df["after_align_save_path"].iloc[0]
                if pd.isna(tiff_path) or not os.path.exists(tiff_path):
                    continue
                n_pre = int(each_set_df["n_pre_frames"].iloc[0]) if "n_pre_frames" in each_set_df.columns else 0
                n_unc = int(each_set_df["n_unc_frames"].iloc[0]) if "n_unc_frames" in each_set_df.columns else 0
                n_post = int(each_set_df["n_post_frames"].iloc[0]) if "n_post_frames" in each_set_df.columns else 0
                n_total = n_pre + n_unc + n_post
                if n_total == 0:
                    continue
                shift_list = _get_shift_per_stack_frame(each_set_df, n_pre, n_unc, n_post)
                if len(shift_list) != n_total:
                    continue
                tiff_dir = os.path.dirname(tiff_path)
                tiff_basename = os.path.splitext(os.path.basename(tiff_path))[0]
                frame_info_path = os.path.join(tiff_dir, f"{tiff_basename}_frame_info.csv")
                frame_info_df = None
                shift_list_frameinfo = None
                if os.path.exists(frame_info_path):
                    try:
                        frame_info_df = pd.read_csv(frame_info_path)
                        shift_list_frameinfo = _build_shift_list_from_frame_info(each_set_df, frame_info_df, n_total)
                    except Exception:
                        frame_info_df = None
                        shift_list_frameinfo = None
                if shift_list_frameinfo is not None:
                    shift_list = shift_list_frameinfo
                debug_rows = []
                for roi_type in ROI_TYPES:
                    roi_path = os.path.join(tiff_dir, f"{tiff_basename}_{roi_type}_roi_mask.tif")
                    if not os.path.exists(roi_path):
                        continue
                    try:
                        roi_aligned = tifffile.imread(roi_path)
                        if roi_aligned.ndim == 2:
                            roi_aligned = roi_aligned[np.newaxis, ...]
                        raw_stack = []
                        for i in range(min(roi_aligned.shape[0], n_total)):
                            sy, sx = shift_list[i]
                            roi_frame = roi_aligned[i].astype(np.float32)
                            # Candidate A (current implementation): inverse drift
                            roi_raw = ndimage_shift(roi_frame, (-sy, -sx), order=0, mode="constant", cval=0)
                            # Candidate B (debug only): opposite sign
                            roi_raw_plus = ndimage_shift(roi_frame, (sy, sx), order=0, mode="constant", cval=0)

                            aligned_n, aligned_cy, aligned_cx = _mask_stats(roi_frame)
                            raw_minus_n, raw_minus_cy, raw_minus_cx = _mask_stats(roi_raw)
                            raw_plus_n, raw_plus_cy, raw_plus_cx = _mask_stats(roi_raw_plus)

                            phase_name = ""
                            source_name = ""
                            if frame_info_df is not None and i < len(frame_info_df):
                                phase_name = str(frame_info_df.iloc[i].get("phase", ""))
                                source_name = str(frame_info_df.iloc[i].get("filename", ""))
                            debug_rows.append({
                                "set_label": each_set_label,
                                "group": each_group,
                                "roi_type": roi_type,
                                "frame": i,
                                "phase": phase_name,
                                "source_filename": source_name,
                                "shift_y": float(sy),
                                "shift_x": float(sx),
                                "aligned_nonzero": aligned_n,
                                "aligned_centroid_y": aligned_cy,
                                "aligned_centroid_x": aligned_cx,
                                "raw_minus_nonzero": raw_minus_n,
                                "raw_minus_centroid_y": raw_minus_cy,
                                "raw_minus_centroid_x": raw_minus_cx,
                                "raw_plus_nonzero": raw_plus_n,
                                "raw_plus_centroid_y": raw_plus_cy,
                                "raw_plus_centroid_x": raw_plus_cx,
                            })
                            raw_stack.append((roi_raw > 0.5).astype(np.uint8))
                        raw_stack = np.stack(raw_stack, axis=0)
                        raw_path = os.path.join(tiff_dir, f"{tiff_basename}_{roi_type}{ROI_MASK_RAW_SUFFIX}.tif")
                        tifffile.imwrite(raw_path, raw_stack, photometric="minisblack")
                        print(f"  Set {each_set_label}: saved {roi_type} raw ROI -> {os.path.basename(raw_path)}")
                    except Exception as e:
                        print(f"  Set {each_set_label}: {roi_type} raw ROI failed: {e}")
                if debug_rows:
                    debug_csv_path = os.path.join(tiff_dir, f"{tiff_basename}_roi_shift_debug.csv")
                    try:
                        pd.DataFrame(debug_rows).to_csv(debug_csv_path, index=False)
                        print(f"  Set {each_set_label}: saved ROI shift debug CSV -> {os.path.basename(debug_csv_path)}")
                    except Exception as e:
                        print(f"  Set {each_set_label}: failed to save ROI shift debug CSV: {e}")

    return combined_df


def _row_from_flim_data(
    imagearray: np.ndarray,
    iminfo,
    file_path: str,
    frame_idx: int,
    stack_frame_idx: int,
    phase: str,
    time_sec: float,
    acq_time_str: str,
    roi_raw: dict,
    corrected_uncaging_z: int,
    z_plus_minus: int,
    each_set_label: int,
    each_group: int,
    fitter: FLIMLifetimeFitter,
    sync_rate: float,
    photon_threshold: int,
    total_photon_threshold: int,
) -> dict:
    """
    Build one quantification row from FLIM imagearray (with bins).
    Pre/post: Z-proj over z_from:z_to; lifetime = fit on histogram summed over Z and ROI.
    Uncaging: single frame at frame_idx; lifetime = fit on histogram over ROI.
    """
    n_ave_frame = int(getattr(iminfo.State.Acq, "nAveFrame", 1))
    # intensity_raw = (12 * np.sum(imagearray, axis=-1)).astype(np.float64)
    # 20260323 probably, this is not required anymore or rather incorrect.
    # originally this was used to avoid dividing small value by 12 in the intensity calculation.
    intensity_raw = (np.sum(imagearray, axis=-1)).astype(np.float64)

    axis0_len, _, C, H, W = intensity_raw.shape
    n_bins = imagearray.shape[5]
    ps_per_unit = (10 ** 12) / sync_rate / n_bins

    if phase in ("pre", "post"):
        z_from = max(0, min(corrected_uncaging_z - z_plus_minus, axis0_len - 1))
        z_to = min(axis0_len, corrected_uncaging_z + z_plus_minus + 1)
        z_to = max(z_to, z_from + 1)
        n_z_slices_used = z_to - z_from
    else:
        n_z_slices_used = 1
        z_from = np.nan
        z_to = np.nan

    row = {
        "set_label": each_set_label,
        "group": each_group,
        "file_path": file_path,
        "phase": phase,
        "slice": frame_idx,
        "time_sec": time_sec,
        "acq_time_str": acq_time_str,
        "n_z_slices_used": n_z_slices_used,
        "nAveFrame": n_ave_frame,
        "z_from": z_from,
        "z_to": z_to,
    }

    for ch_idx, ch_name in enumerate(["Ch1", "Ch2"]):
        if ch_idx >= C:
            continue
        if phase in ("pre", "post"):
            img = intensity_raw[z_from:z_to, 0, ch_idx, :, :].max(axis=0).astype(np.float32)
            # For lifetime: sum bins over Z then (H,W,bins)
            frame_bins = np.sum(imagearray[z_from:z_to, 0, ch_idx, :, :, :], axis=0).astype(np.float64)
        else:
            img = intensity_raw[frame_idx, 0, ch_idx, :, :].astype(np.float32)
            frame_bins = np.array(imagearray[frame_idx, 0, ch_idx, :, :, :], dtype=np.float64)

        if stack_frame_idx >= roi_raw[list(roi_raw.keys())[0]].shape[0]:
            continue
        for roi_type in roi_raw:
            roi = roi_raw[roi_type][stack_frame_idx]
            if roi.shape[0] != img.shape[0] or roi.shape[1] != img.shape[1]:
                continue
            mask = roi > 0
            if not np.any(mask):
                row[f"{roi_type}_{ch_name}_intensity"] = np.nan
                if roi_type != "Background":
                    row[f"{roi_type}_{ch_name}_total_photon"] = np.nan
                    row[f"{roi_type}_{ch_name}_lifetime"] = np.nan
                continue
            row[f"{roi_type}_{ch_name}_intensity"] = float(np.nanmean(img[mask]))

            if roi_type == "Background":
                continue
            # Lifetime (Spine, DendriticShaft): transient_roi_analysis style
            intensity_image = frame_bins.sum(axis=-1)
            photon_exclude_mask = intensity_image <= photon_threshold
            thresholded_data = frame_bins.copy()
            thresholded_data[photon_exclude_mask, :] = 0
            voxels_of_interest = thresholded_data[mask, :]
            y_data = np.sum(voxels_of_interest, axis=0)
            total_photon = float(np.sum(y_data))
            row[f"{roi_type}_{ch_name}_total_photon"] = total_photon
            if total_photon < total_photon_threshold:
                row[f"{roi_type}_{ch_name}_lifetime"] = np.nan
            else:
                try:
                    x = np.arange(len(y_data))
                    result = fitter.fit_double_exponential(x, y_data, ps_per_unit, sync_rate)
                    row[f"{roi_type}_{ch_name}_lifetime"] = result.get("lifetime", np.nan)
                except Exception:
                    row[f"{roi_type}_{ch_name}_lifetime"] = np.nan
    return row


def quantify_intensity_from_flim(
    combined_df: pd.DataFrame,
    ch_1or2: int,
    z_plus_minus: int,
    output_csv_path: str,
    sync_rate: float = SYNC_RATE,
    photon_threshold: int = 15,
    total_photon_threshold: int = 1000,
):
    """
    Quantify intensity and lifetime from FLIM. Per FLIM file, per frame, per Ch, per ROI.
    Pre/post: Z-proj with z_plus_minus; time_sec=0. Uncaging: per-frame time_sec from FLIM metadata.
    Output CSV includes intensity, time_sec, total_photon, lifetime (Spine/DendriticShaft).
    """
    def _is_rejected(set_df):
        """Return True if this group/set is rejected (skip intensity/lifetime quantification)."""
        if 'reject' not in set_df.columns or len(set_df) == 0:
            return False
        r = set_df['reject'].iloc[0]
        return r == 1 or r is True

    total_items = 0
    for each_filepath_without_number in combined_df["filepath_without_number"].unique():
        each_filegroup_df = combined_df[combined_df["filepath_without_number"] == each_filepath_without_number]
        for each_group in each_filegroup_df["group"].unique():
            each_group_df = each_filegroup_df[each_filegroup_df["group"] == each_group]
            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
                if _is_rejected(each_set_df):
                    continue
                tiff_path = each_set_df["after_align_save_path"].iloc[0]
                if pd.isna(tiff_path) or not os.path.exists(tiff_path):
                    continue
                n_pre = int(each_set_df["n_pre_frames"].iloc[0]) if "n_pre_frames" in each_set_df.columns else 0
                n_unc = int(each_set_df["n_unc_frames"].iloc[0]) if "n_unc_frames" in each_set_df.columns else 0
                n_post = int(each_set_df["n_post_frames"].iloc[0]) if "n_post_frames" in each_set_df.columns else 0
                total_items += n_pre + n_unc + n_post
    if total_items == 0:
        print("No frames to quantify. Skipping.")
        return pd.DataFrame()
    print(f"Intensity & lifetime quantification: {total_items} frames (printing every 10%)")
    next_pct = 10
    processed = 0
    fitter = FLIMLifetimeFitter()

    all_rows = []
    for each_filepath_without_number in combined_df["filepath_without_number"].unique():
        each_filegroup_df = combined_df[combined_df["filepath_without_number"] == each_filepath_without_number]
        for each_group in each_filegroup_df["group"].unique():
            each_group_df = each_filegroup_df[each_filegroup_df["group"] == each_group]
            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
                if _is_rejected(each_set_df):
                    continue
                tiff_path = each_set_df["after_align_save_path"].iloc[0]
                if pd.isna(tiff_path) or not os.path.exists(tiff_path):
                    continue
                n_pre = int(each_set_df["n_pre_frames"].iloc[0]) if "n_pre_frames" in each_set_df.columns else 0
                n_unc = int(each_set_df["n_unc_frames"].iloc[0]) if "n_unc_frames" in each_set_df.columns else 0
                n_post = int(each_set_df["n_post_frames"].iloc[0]) if "n_post_frames" in each_set_df.columns else 0
                corrected_uncaging_z = int(each_set_df["corrected_uncaging_z"].iloc[0])
                tiff_dir = os.path.dirname(tiff_path)
                tiff_basename = os.path.splitext(os.path.basename(tiff_path))[0]
                roi_raw = {}
                unc_only = n_pre == 0 and n_post == 0
                for roi_type in ROI_TYPES:
                    raw_path = os.path.join(tiff_dir, f"{tiff_basename}_{roi_type}{ROI_MASK_RAW_SUFFIX}.tif")
                    if not os.path.exists(raw_path) and unc_only:
                        raw_path = os.path.join(tiff_dir, f"{tiff_basename}_{roi_type}_roi_mask.tif")
                    if not os.path.exists(raw_path):
                        continue

                    # 20260323 for debugging
                    # print(f"roi_type: {roi_type}, raw_path: {raw_path}")
                    roi_raw[roi_type] = tifffile.imread(raw_path)
                    if roi_raw[roi_type].ndim == 2:
                        roi_raw[roi_type] = roi_raw[roi_type][np.newaxis, ...]
                if not roi_raw:
                    continue

                def _row_from_file(file_path: str, frame_idx: int, stack_frame_idx: int, phase: str, time_sec: float, acq_time_str: str = ""):
                    iminfo = FileReader()
                    iminfo.read_imageFile(file_path, True)
                    imagearray = np.array(iminfo.image)
                    if acq_time_str == "" and iminfo.acqTime and frame_idx < len(iminfo.acqTime):
                        acq_time_str = str(iminfo.acqTime[frame_idx]).strip()
                    return _row_from_flim_data(
                        imagearray, iminfo, file_path, frame_idx, stack_frame_idx, phase, time_sec, acq_time_str,
                        roi_raw, corrected_uncaging_z, z_plus_minus, each_set_label, each_group,
                        fitter, sync_rate, photon_threshold, total_photon_threshold,
                    )

                pre_df = each_set_df[each_set_df["phase"] == "pre"].sort_values("nth_omit_induction")
                for stack_i, (_, row_pre) in enumerate(pre_df.iterrows()):
                    fp = str(row_pre["file_path"])
                    if not os.path.exists(fp):
                        continue
                    all_rows.append(_row_from_file(fp, 0, stack_i, "pre", 0.0, ""))
                    processed += 1
                    if total_items > 0 and 100 * processed >= total_items * next_pct:
                        print(f"  Progress: {next_pct}% ({processed}/{total_items} frames)")
                        next_pct += 10

                unc_df = each_set_df[each_set_df["phase"] == "unc"]
                if len(unc_df) > 0:
                    unc_path = unc_df.iloc[0]["file_path"]
                    if os.path.exists(unc_path):
                        iminfo_unc = FileReader()
                        iminfo_unc.read_imageFile(unc_path, True)
                        imagearray_unc = np.array(iminfo_unc.image)
                        T = imagearray_unc.shape[0]
                        frame_times = _get_frame_times_from_flim(unc_path)
                        if len(frame_times) != T:
                            frame_times = [np.nan] * T
                        acq_time_list = iminfo_unc.acqTime if iminfo_unc.acqTime else []
                        for t in range(T):
                            time_sec = frame_times[t] if frame_times else np.nan
                            acq_str = acq_time_list[t] if t < len(acq_time_list) else ""
                            all_rows.append(_row_from_flim_data(
                                imagearray_unc, iminfo_unc, unc_path, t, n_pre + t, "unc", time_sec, acq_str,
                                roi_raw, corrected_uncaging_z, z_plus_minus, each_set_label, each_group,
                                fitter, sync_rate, photon_threshold, total_photon_threshold,
                            ))
                            processed += 1
                            if total_items > 0 and 100 * processed >= total_items * next_pct:
                                print(f"  Progress: {next_pct}% ({processed}/{total_items} frames)")
                                next_pct += 10

                post_df = each_set_df[each_set_df["phase"] == "post"].sort_values("nth_omit_induction")
                for stack_i, (_, row_post) in enumerate(post_df.iterrows()):
                    fp = str(row_post["file_path"])
                    if not os.path.exists(fp):
                        continue
                    all_rows.append(_row_from_file(fp, 0, n_pre + n_unc + stack_i, "post", 0.0, ""))
                    processed += 1
                    if total_items > 0 and 100 * processed >= total_items * next_pct:
                        print(f"  Progress: {next_pct}% ({processed}/{total_items} frames)")
                        next_pct += 10

    if total_items > 0 and processed > 0:
        print(f"  Progress: 100% ({processed}/{total_items} frames)")
    out_df = pd.DataFrame(all_rows)
    if len(out_df) == 0 and total_items > 0:
        print("WARNING: No rows were produced. CSV not saved. Check that *_roi_mask_raw.tif files exist")
        print("  (e.g. Spine_roi_mask_raw.tif, DendriticShaft_roi_mask_raw.tif, Background_roi_mask_raw.tif)")
        print("  in the same folder as each set's uncaging TIFF. They are created by save_drift_corrected_roi_masks")
        print("  after ROI definition. If you skipped ROI definition or closed without saving, run the ROI GUI again.")
    if len(out_df) > 0:
        # frame: per-set sequential index (0, 1, 2, ...) in order pre -> unc -> post (same as frame_info)
        out_df["frame"] = out_df.groupby(["set_label", "group"]).cumcount()
        # elapsed_time_sec: per set, seconds from first moment (same as frame_info: use acq_time_str)
        def _elapsed_from_acq_time(g: pd.Series) -> pd.Series:
            datetimes = []
            for s in g:
                s = (s or "").strip() if isinstance(s, str) else ""
                if s:
                    try:
                        datetimes.append(_parse_acq_time(s))
                    except Exception:
                        datetimes.append(None)
                else:
                    datetimes.append(None)
            t0 = min((dt for dt in datetimes if dt is not None), default=None)
            if t0 is None:
                return pd.Series(np.nan, index=g.index)
            values = [(dt - t0).total_seconds() if dt is not None else np.nan for dt in datetimes]
            return pd.Series(values, index=g.index)
        out_df["elapsed_time_sec"] = out_df.groupby(["set_label", "group"], group_keys=False)["acq_time_str"].transform(_elapsed_from_acq_time)
        # Reorder: Frame after slice, elapsed_time_sec after time_sec, z_from/z_to after n_z_slices_used
        lead = ["set_label", "group", "file_path", "phase", "slice", "frame", "time_sec", "acq_time_str", "elapsed_time_sec", "n_z_slices_used", "z_from", "z_to", "nAveFrame"]
        rest = [c for c in out_df.columns if c not in lead]
        out_df = out_df[[c for c in lead if c in out_df.columns] + rest]
        out_df.to_csv(output_csv_path, index=False)
        print(f"Saved intensity & lifetime quantification to {output_csv_path}")
    return out_df


def run_tiff_uncaging_roi(
    ch_1or2 = 1,
    z_plus_minus = 2,
    pre_length = 2,
    photon_threshold: int = 15,
    total_photon_threshold: int = 1000,
    ask_stop_here_TF = False,
    predefined_df_path: str | None = None,
    skip_roi_gui: bool = False,
):
    """
    Run the full TIFF uncaging ROI workflow: first_processing, full-size stack build,
    GUI for ROI definition, drift-corrected ROI masks, intensity/lifetime quantification.
    Call from other modules with e.g. run_tiff_uncaging_roi(photon_threshold=20).
    """
    summary_str = ""
    summary_str += datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
    summary_str += f"ch_1or2: {ch_1or2}\n"
    summary_str += f"z_plus_minus: {z_plus_minus}\n"
    summary_str += f"pre_length: {pre_length}\n"
    summary_str += f"SAVE_PLOT_TF: {SAVE_PLOT_TF}\n"
    summary_str += f"SAVE_TIF_TF: {SAVE_TIF_TF}\n"
    summary_str += f"SYNC_RATE: {SYNC_RATE}\n"
    summary_str += f"photon_threshold: {photon_threshold}\n"
    summary_str += f"total_photon_threshold: {total_photon_threshold}\n"
    summary_str += f"predefined_df_path: {predefined_df_path}\n"
    summary_str += f"skip_roi_gui: {skip_roi_gui}\n"
    summary_str += "="*60+"\n"

    # If predefined_df_path exists, skip FLIM/file dialogs and load directly.
    use_predefined_df = bool(predefined_df_path and os.path.exists(predefined_df_path))
    if use_predefined_df:
        flim_file_select_dialog_TF = False
        one_of_filepath_list = []
        print("=" * 60)
        print("Using predefined combined_df (dialog-free mode)")
        print("  combined_df:", predefined_df_path)
        print("=" * 60)
    elif TEST_MODE:
        Do_without_asking = False
        skip_gui_TF = False
        flim_file_select_dialog_TF = False
        # first_processing globs "*_highmag_*002.flim"; use 002 if present so the group is found
        base = TEST_FLIM_PATH[:-8]
        path_002 = base + "002.flim"
        if os.path.isfile(path_002):
            one_of_filepath_list = [path_002]
            print("=" * 60)
            print("TEST MODE: using 002 in same group (no predefined df)")
            print("  FLIM:", path_002)
            print("=" * 60)
        else:
            one_of_filepath_list = [TEST_FLIM_PATH]
            print("=" * 60)
            print("TEST MODE: using fixed path (no predefined df)")
            print("  FLIM:", TEST_FLIM_PATH)
            print("  If first_processing finds no data, add 002.flim to the same folder.")
            print("=" * 60)
    else:
        flim_file_select_dialog_TF = True
        one_of_filepath_list = []

    if flim_file_select_dialog_TF:
        one_of_filepath = ask_open_path_gui(filetypes=[("FLIM files", "*.flim")])
        if not one_of_filepath:
            print("No file selected. Exiting.")
            return
        one_of_filepath_list = [one_of_filepath]

    summary_str += f"one_of_filepath: {one_of_filepath_list}\n"

    if (not one_of_filepath_list) and (not use_predefined_df):
        print("No FLIM path. Exiting.")
        return

    # Deduplicate by group (filepath_without_number): keep one path per group so
    # first_processing is not run twice for the same folder (avoids duplicate rows
    # and "Multiple uncaging rows" / double file list).
    seen_groups = {}
    deduped = []
    for p in one_of_filepath_list:
        if len(p) > 8 and p.endswith(".flim"):
            base = os.path.normpath(p[:-8])
        else:
            base = p
        if base not in seen_groups:
            seen_groups[base] = p
            deduped.append(p)
    if len(deduped) < len(one_of_filepath_list):
        print(f"Deduplicated FLIM list: {len(one_of_filepath_list)} -> {len(deduped)} paths (one per group).")
    one_of_filepath_list = deduped

    # No predefined df in test: always run first_processing
    if use_predefined_df:
        df_save_path_1 = predefined_df_path
    else:
        df_save_path_1 = os.path.join(os.path.dirname(one_of_filepath_list[0]), "combined_df_1.pkl")
    combined_df = None

    loaded_existing_combined_df = False
    if use_predefined_df:
        try:
            combined_df = pd.read_pickle(df_save_path_1)
            loaded_existing_combined_df = True
            print(f"Loaded predefined df: {df_save_path_1}")
        except Exception as e:
            print(f"Failed to load predefined_df_path: {e}")
            raise
    elif not TEST_MODE:
        yn_already_have_combined_df = ask_yes_no_gui("Do you already have combined_df_1.pkl?")
        if yn_already_have_combined_df:
            df_save_path_1 = ask_open_path_gui()
            if df_save_path_1 and os.path.exists(df_save_path_1):
                combined_df = pd.read_pickle(df_save_path_1)
                loaded_existing_combined_df = True
                print(f"Loaded: {df_save_path_1}")

    if combined_df is None:
        # Run data preparation (no predefined df)
        combined_df = pd.DataFrame()
        for one_of_filepath in one_of_filepath_list:
            print(f"\nProcessing ... {one_of_filepath}\n")
            try:
                temp_df = first_processing_for_flim_files(
                    one_of_filepath,
                    z_plus_minus,
                    ch_1or2,
                    pre_length=pre_length,
                    save_plot_TF=SAVE_PLOT_TF,
                    save_tif_TF=SAVE_TIF_TF,
                    return_error_dict=False,
                )
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
            except Exception as e:
                print(f"first_processing failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        
            df_save_path_1 = ask_save_path_gui()
            if df_save_path_1 and (df_save_path_1[-4:] != ".pkl"):
                df_save_path_1 = df_save_path_1 + ".pkl"
        else:
            df_save_path_1 = os.path.join(os.path.dirname(one_of_filepath_list[0]), "combined_df_1.pkl")
        combined_df.to_pickle(df_save_path_1)
        combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))
        print(f"Saved: {df_save_path_1}")
    
    summary_str += f"df_save_path_1: {df_save_path_1}\n"
    summary_str += f"df_save_path_1_csv: {df_save_path_1.replace(".pkl", ".csv")}\n"

    if combined_df is None or len(combined_df) == 0:
        print("No data. Exiting.")
        return

    # Determine whether to skip building full-size stacks.
    # NOTE: When using predefined_df, we must NOT skip rebuild entirely because
    # frame_info.csv (which stores runtime alignment shifts) must be regenerated
    # with correct shift values from load_and_align_data. Skipping this causes
    # save_drift_corrected_roi_masks to use stale/wrong shift values from combined_df
    # (which differ from the actual alignment shifts), resulting in heavily shifted
    # roi_mask_raw.tif files for pre/post frames.
    # Instead, use skip_tiff_if_exists=True so existing TIFFs are not rewritten
    # (fast) but frame_info.csv is always regenerated (correct).
    skip_full_size_build = False
    skip_tiff_if_exists_flag = False
    if use_predefined_df and loaded_existing_combined_df:
        # Predefined df mode: always run rebuild to refresh frame_info.csv,
        # but skip TIFF writes if files already exist (skip_tiff_if_exists=True).
        skip_tiff_if_exists_flag = True
        print("Predefined df mode: running rebuild with skip_tiff_if_exists=True (refresh frame_info.csv, skip TIFF write if exists)")
    elif loaded_existing_combined_df:
        skip_full_size_build = ask_yes_no_gui(
            "Skip 'Building full-size stacks for ROI definition' and use existing stack paths?"
        )
    if skip_full_size_build:
        print("Skipped building full-size stacks for ROI definition.")
    else:
        print("\n" + "=" * 60)
        print("Building full-size stacks for ROI definition (Pre + Uncaging + Post)")
        print("=" * 60)
        combined_df = rebuild_tiff_full_size_for_roi(combined_df, ch_1or2, z_plus_minus, skip_tiff_if_exists=skip_tiff_if_exists_flag)
    if df_save_path_1:
        combined_df.to_pickle(df_save_path_1)
        combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))

    # Use full-size TIFF for ROI definition so user draws on full image
    if "after_align_full_save_path" in combined_df.columns:
        combined_df["after_align_save_path"] = combined_df["after_align_full_save_path"].fillna(combined_df["after_align_save_path"])
        # For full-size TIFF, uncaging frames are aligned with unc_drift (query->ref shift).
        # So display coordinate should follow uncaging raw center + unc_drift.
        full_mask = combined_df["after_align_full_save_path"].notna()
        combined_df.loc[full_mask, "uncaging_display_x"] = combined_df.loc[full_mask, "corrected_uncaging_x"]
        combined_df.loc[full_mask, "uncaging_display_y"] = combined_df.loc[full_mask, "corrected_uncaging_y"]
        if all(c in combined_df.columns for c in ["center_x", "center_y", "unc_drift_x", "unc_drift_y"]):
            key_cols = ["filepath_without_number", "group", "nth_set_label"]
            for _, each_set_df in combined_df[full_mask].groupby(key_cols, sort=False):
                unc_rows = each_set_df[each_set_df["phase"] == "unc"]
                if len(unc_rows) == 0:
                    continue
                unc_row = unc_rows.iloc[0]
                display_x = float(unc_row.get("center_x", 0) or 0) + float(unc_row.get("unc_drift_x", 0) or 0)
                display_y = float(unc_row.get("center_y", 0) or 0) + float(unc_row.get("unc_drift_y", 0) or 0)
                combined_df.loc[each_set_df.index, "uncaging_display_x"] = display_x
                combined_df.loc[each_set_df.index, "uncaging_display_y"] = display_y
    if df_save_path_1:
        combined_df.to_pickle(df_save_path_1)
        combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))

    if df_save_path_1:
        if skip_roi_gui:
            print("skip_roi_gui=True: skip ROI definition GUI and continue.")
        else:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            file_selection_gui = launch_file_selection_gui_tiff_only(
                combined_df,
                df_save_path_1,
                additional_columns=['dt'],
                save_auto=False,
            )
            app.exec_()
            print("ROI definition (full-size) finished.")
            if os.path.exists(df_save_path_1):
                combined_df = pd.read_pickle(df_save_path_1)
        print("\nSaving drift-corrected ROI masks (Type B: raw FLIM coordinates)...")
        save_drift_corrected_roi_masks(combined_df)
        if df_save_path_1:
            combined_df.to_pickle(df_save_path_1)
            combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))

    #stop here?
    if ask_stop_here_TF:
        if ask_yes_no_gui("Stop here?"):
            return

    # Ensure reject column is 0/1 (0=accept, 1=reject) for consistent filtering
    if 'reject' in combined_df.columns:
        combined_df['reject'] = ((combined_df['reject'] == True) | (combined_df['reject'] == 1)).astype(int)
    else:
        combined_df['reject'] = 0

    print("Start intensity and lifetime quantification from .flim files")
    out_csv_path = df_save_path_1.replace(".pkl", "_intensity_lifetime_all_frames.csv")
    quantify_intensity_from_flim(
        combined_df, ch_1or2, z_plus_minus, out_csv_path,
        photon_threshold=photon_threshold,
        total_photon_threshold=total_photon_threshold,
    )
    summary_str += f"out_csv_path: {out_csv_path}\n"
    summary_str += "="*60+"\n"
    summary_str += "finished at "+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n"
    print("Done.")

    #save summary_str to a file
    save_summary_str_path = os.path.join(os.path.dirname(df_save_path_1), "summary_str.txt")
    with open(save_summary_str_path, "w") as f:
        f.write(summary_str)

    try:
        display(datetime.now())
    except:
        pass
    print(summary_str)
    return df_save_path_1, out_csv_path


def run_tiff_uncaging_roi_no_zstack(
    ch_1or2=1,
    z_plus_minus=2,
    pre_length=2,
    photon_threshold: int = 15,
    total_photon_threshold: int = 1000,
    ask_stop_here_TF=False,
    uncaging_frame_num=None,
    titration_frame_num=None,
):
    """
    Same workflow as run_tiff_uncaging_roi, but builds only uncaging-frame stacks for ROI
    (no pre/post, no alignment). Phase detection (pre/unc/post) is unchanged; only the
    TIFF used for ROI definition and the frame counts (n_pre=0, n_post=0) differ.
    Quantification runs on uncaging frames only.
    """
    summary_str = ""
    summary_str += datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
    summary_str += "run_tiff_uncaging_roi_no_zstack (uncaging-only stacks)\n"
    summary_str += f"ch_1or2: {ch_1or2}\n"
    summary_str += f"z_plus_minus: {z_plus_minus}\n"
    summary_str += f"pre_length: {pre_length}\n"
    summary_str += f"photon_threshold: {photon_threshold}\n"
    summary_str += f"total_photon_threshold: {total_photon_threshold}\n"
    summary_str += "=" * 60 + "\n"

    if TEST_MODE:
        one_of_filepath_list = [TEST_FLIM_PATH] if not os.path.isfile(TEST_FLIM_PATH[:-8] + "002.flim") else [TEST_FLIM_PATH[:-8] + "002.flim"]
        print("TEST MODE: using", one_of_filepath_list[0])
    else:
        one_of_filepath = ask_open_path_gui(filetypes=[("FLIM files", "*.flim")])
        if not one_of_filepath:
            print("No file selected. Exiting.")
            return
        one_of_filepath_list = [one_of_filepath]

    if not one_of_filepath_list:
        print("No FLIM path. Exiting.")
        return

    seen_groups = {}
    deduped = []
    for p in one_of_filepath_list:
        base = os.path.normpath(p[:-8]) if len(p) > 8 and p.endswith(".flim") else p
        if base not in seen_groups:
            seen_groups[base] = p
            deduped.append(p)
    one_of_filepath_list = deduped

    df_save_path_1 = os.path.join(os.path.dirname(one_of_filepath_list[0]), "combined_df_1.pkl")
    combined_df = None

    if not TEST_MODE:
        yn_already_have_combined_df = ask_yes_no_gui("Do you already have combined_df_1.pkl?")
        if yn_already_have_combined_df:
            df_save_path_1 = ask_open_path_gui()
            if df_save_path_1 and os.path.exists(df_save_path_1):
                combined_df = pd.read_pickle(df_save_path_1)
                print(f"Loaded: {df_save_path_1}")

    if combined_df is None:
        combined_df = pd.DataFrame()
        for one_of_filepath in one_of_filepath_list:
            print(f"\nProcessing ... {one_of_filepath}\n")
            try:
                fp_kwargs = dict(
                    one_of_filepath=one_of_filepath,
                    z_plus_minus=z_plus_minus,
                    ch_1or2=ch_1or2,
                    pre_length=pre_length,
                    save_plot_TF=SAVE_PLOT_TF,
                    save_tif_TF=SAVE_TIF_TF,
                    return_error_dict=False,
                )
                if uncaging_frame_num is not None:
                    fp_kwargs["uncaging_frame_num"] = uncaging_frame_num
                if titration_frame_num is not None:
                    fp_kwargs["titration_frame_num"] = titration_frame_num
                temp_df = first_processing_for_flim_files(**fp_kwargs)
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
            except Exception as e:
                print(f"first_processing failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        df_save_path_1 = ask_save_path_gui()
        if df_save_path_1 and (df_save_path_1[-4:] != ".pkl"):
            df_save_path_1 = df_save_path_1 + ".pkl"
        else:
            df_save_path_1 = os.path.join(os.path.dirname(one_of_filepath_list[0]), "combined_df_1.pkl")
        combined_df.to_pickle(df_save_path_1)
        combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))
        print(f"Saved: {df_save_path_1}")

    if combined_df is None or len(combined_df) == 0:
        print("No data. Exiting.")
        return

    print("\n" + "=" * 60)
    print("Building uncaging-only stacks for ROI definition (no pre/post alignment)")
    print("=" * 60)
    combined_df = _rebuild_tiff_uncaging_only_for_roi(combined_df, ch_1or2, z_plus_minus)
    if df_save_path_1:
        combined_df.to_pickle(df_save_path_1)
        combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))

    if "after_align_full_save_path" in combined_df.columns:
        combined_df["after_align_save_path"] = combined_df["after_align_full_save_path"].fillna(combined_df["after_align_save_path"])
        full_mask = combined_df["after_align_full_save_path"].notna()
        combined_df.loc[full_mask, "uncaging_display_x"] = combined_df.loc[full_mask, "corrected_uncaging_x"]
        combined_df.loc[full_mask, "uncaging_display_y"] = combined_df.loc[full_mask, "corrected_uncaging_y"]
        if all(c in combined_df.columns for c in ["center_x", "center_y", "unc_drift_x", "unc_drift_y"]):
            key_cols = ["filepath_without_number", "group", "nth_set_label"]
            for _, each_set_df in combined_df[full_mask].groupby(key_cols, sort=False):
                unc_rows = each_set_df[each_set_df["phase"] == "unc"]
                if len(unc_rows) == 0:
                    continue
                unc_row = unc_rows.iloc[0]
                display_x = float(unc_row.get("center_x", 0) or 0) + float(unc_row.get("unc_drift_x", 0) or 0)
                display_y = float(unc_row.get("center_y", 0) or 0) + float(unc_row.get("unc_drift_y", 0) or 0)
                combined_df.loc[each_set_df.index, "uncaging_display_x"] = display_x
                combined_df.loc[each_set_df.index, "uncaging_display_y"] = display_y
    if df_save_path_1:
        combined_df.to_pickle(df_save_path_1)
        combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))

    if df_save_path_1:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        file_selection_gui = launch_file_selection_gui_tiff_only(
            combined_df,
            df_save_path_1,
            additional_columns=["dt"],
            save_auto=False,
        )
        app.exec_()
        print("ROI definition (uncaging-only) finished.")
        if os.path.exists(df_save_path_1):
            combined_df = pd.read_pickle(df_save_path_1)
        print("\nSaving drift-corrected ROI masks (Type B: raw FLIM coordinates)...")
        save_drift_corrected_roi_masks(combined_df)
        if df_save_path_1:
            combined_df.to_pickle(df_save_path_1)
            combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))

    if ask_stop_here_TF:
        if ask_yes_no_gui("Stop here?"):
            return

    if "reject" in combined_df.columns:
        combined_df["reject"] = ((combined_df["reject"] == True) | (combined_df["reject"] == 1)).astype(int)
    else:
        combined_df["reject"] = 0

    print("Start intensity and lifetime quantification from .flim files (uncaging only)")
    out_csv_path = df_save_path_1.replace(".pkl", "_intensity_lifetime_all_frames.csv")
    quantify_intensity_from_flim(
        combined_df, ch_1or2, z_plus_minus, out_csv_path,
        photon_threshold=photon_threshold,
        total_photon_threshold=total_photon_threshold,
    )
    summary_str += f"out_csv_path: {out_csv_path}\n"
    summary_str += "=" * 60 + "\n"
    summary_str += "finished at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
    print("Done.")

    save_summary_str_path = os.path.join(os.path.dirname(df_save_path_1), "summary_str_no_zstack.txt")
    with open(save_summary_str_path, "w") as f:
        f.write(summary_str)
    try:
        display(datetime.now())
    except Exception:
        pass
    print(summary_str)
    return df_save_path_1, out_csv_path


if __name__ == "__main__":
    run_tiff_uncaging_roi(ch_1or2 = 1,
            z_plus_minus = 2,
            pre_length = 1)
