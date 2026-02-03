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
from simple_dialog import ask_yes_no_gui, ask_open_path_gui

# -----------------------------------------------------------------------------
# TEST CONFIG: fixed path, no predefined df, no dialogs
# -----------------------------------------------------------------------------
TEST_MODE = True
TEST_FLIM_PATH = r"C:\Users\WatabeT\Desktop\temp2\mChGFP_1_pos1__highmag_1_002.flim"
# first_processing uses glob "*_highmag_*002.flim" in the folder; if 002 is missing,
# we pass the folder's first FLIM (e.g. 004) so get_uncaging_pos_multiple still gets
# the full group via get_flimfile_list(004) -> 001,002,...,004.

Do_without_asking = True if TEST_MODE else False
# Set skip_gui_TF = True to run without opening GUI (e.g. quick test from terminal)
skip_gui_TF = False
pre_defined_df_TF = False
df_defined_path = None

ch_1or2 = 1
z_plus_minus = 2
pre_length = 1
save_plot_TF = True
save_tif_TF = True

TIFF_WITH_UNCAGING_SUFFIX = "_with_uncaging"
ROI_MASK_RAW_SUFFIX = "_roi_mask_raw"
ROI_TYPES = ["Spine", "DendriticShaft", "Background"]

# Lifetime fitting (transient_roi_analysis style)
SYNC_RATE = 80e6  # Hz
PHOTON_THRESHOLD = 15
TOTAL_PHOTON_THRESHOLD = 1000


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


def rebuild_tiff_full_size_for_roi(combined_df: pd.DataFrame, ch: int, z_plus_minus: int):
    """
    Build full-size stack per set for ROI definition: Pre Z-proj (full) + Uncaging
    (full, drift applied to last pre) + Post Z-proj (full). Saves to after_align_full_save_path.
    Stores unc_drift_y, unc_drift_x on the uncaging row for later raw-ROI generation.
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
                pre_filenames = []
                pre_file_paths = []
                for _, row in each_set_df[each_set_df["phase"] == "pre"].sort_values("nth_omit_induction").iterrows():
                    fp = row["file_path"]
                    if fp not in file_path_to_array_idx:
                        continue
                    array_idx = file_path_to_array_idx[fp]
                    zproj = Aligned_4d_array[array_idx, z_from:z_to, :, :].max(axis=0)
                    pre_list.append(zproj)
                    pre_filenames.append(os.path.basename(fp))
                    pre_file_paths.append(fp)
                pre_stack = np.stack(pre_list, axis=0) if pre_list else np.empty((0, Y_full, X_full), dtype=np.float32)

                post_list = []
                post_filenames = []
                post_file_paths = []
                for _, row in each_set_df[each_set_df["phase"] == "post"].sort_values("nth_omit_induction").iterrows():
                    fp = row["file_path"]
                    if fp not in file_path_to_array_idx:
                        continue
                    array_idx = file_path_to_array_idx[fp]
                    zproj = Aligned_4d_array[array_idx, z_from:z_to, :, :].max(axis=0)
                    post_list.append(zproj)
                    post_filenames.append(os.path.basename(fp))
                    post_file_paths.append(fp)
                post_stack = np.stack(post_list, axis=0) if post_list else np.empty((0, Y_full, X_full), dtype=np.float32)

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
                tifffile.imwrite(new_tiff_path, new_stack.astype(np.float32))
                combined_df.loc[each_set_df.index, "after_align_full_save_path"] = new_tiff_path
                combined_df.loc[each_set_df.index, "n_pre_frames"] = pre_stack.shape[0]
                combined_df.loc[each_set_df.index, "n_unc_frames"] = unc_stack.shape[0]
                combined_df.loc[each_set_df.index, "n_post_frames"] = post_stack.shape[0]
                print(f"  Set {each_set_label}: saved full-size {base_name}.tif (Pre {pre_stack.shape[0]} + Unc {unc_stack.shape[0]} + Post {post_stack.shape[0]})")
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
                    frame_info_rows.append({"frame": i, "Z_projection": True, "nth_slice": "", "filename": pre_filenames[i] if i < len(pre_filenames) else "", "phase": "pre", "acq_time_str": acq_str, "z_from": z_from, "z_to": z_to})
                for k in range(unc_stack.shape[0]):
                    acq_str = acq_time_unc[k] if k < len(acq_time_unc) else ""
                    if acq_str:
                        acq_str = str(acq_str).strip()
                    frame_info_rows.append({"frame": pre_stack.shape[0] + k, "Z_projection": False, "nth_slice": k, "filename": unc_basename, "phase": "uncaging", "acq_time_str": acq_str, "z_from": np.nan, "z_to": np.nan})
                for i in range(post_stack.shape[0]):
                    acq_str = _get_acq_time_str(post_file_paths[i], 0) if i < len(post_file_paths) else ""
                    frame_info_rows.append({"frame": pre_stack.shape[0] + unc_stack.shape[0] + i, "Z_projection": True, "nth_slice": "", "filename": post_filenames[i] if i < len(post_filenames) else "", "phase": "post", "acq_time_str": acq_str, "z_from": z_from, "z_to": z_to})
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
                            roi_raw = ndimage_shift(roi_frame, (-sy, -sx), order=0, mode="constant", cval=0)
                            raw_stack.append((roi_raw > 0.5).astype(np.uint8))
                        raw_stack = np.stack(raw_stack, axis=0)
                        raw_path = os.path.join(tiff_dir, f"{tiff_basename}_{roi_type}{ROI_MASK_RAW_SUFFIX}.tif")
                        tifffile.imwrite(raw_path, raw_stack, photometric="minisblack")
                        print(f"  Set {each_set_label}: saved {roi_type} raw ROI -> {os.path.basename(raw_path)}")
                    except Exception as e:
                        print(f"  Set {each_set_label}: {roi_type} raw ROI failed: {e}")

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
    intensity_raw = (12 * np.sum(imagearray, axis=-1)).astype(np.float64)
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
    photon_threshold: int = PHOTON_THRESHOLD,
    total_photon_threshold: int = TOTAL_PHOTON_THRESHOLD,
):
    """
    Quantify intensity and lifetime from FLIM. Per FLIM file, per frame, per Ch, per ROI.
    Pre/post: Z-proj with z_plus_minus; time_sec=0. Uncaging: per-frame time_sec from FLIM metadata.
    Output CSV includes intensity, time_sec, total_photon, lifetime (Spine/DendriticShaft).
    """
    total_items = 0
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
                for roi_type in ROI_TYPES:
                    raw_path = os.path.join(tiff_dir, f"{tiff_basename}_{roi_type}{ROI_MASK_RAW_SUFFIX}.tif")
                    if not os.path.exists(raw_path):
                        continue
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


def main():
    if TEST_MODE:
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

    if not one_of_filepath_list:
        print("No FLIM path. Exiting.")
        return

    # No predefined df in test: always run first_processing
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
                    save_plot_TF=save_plot_TF,
                    save_tif_TF=save_tif_TF,
                    return_error_dict=False,
                )
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
            except Exception as e:
                print(f"first_processing failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        if not Do_without_asking:
            from simple_dialog import ask_save_path_gui
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
    print("Building full-size stacks for ROI definition (Pre + Uncaging + Post)")
    print("=" * 60)
    combined_df = rebuild_tiff_full_size_for_roi(combined_df, ch_1or2, z_plus_minus)
    if df_save_path_1:
        combined_df.to_pickle(df_save_path_1)
        combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))

    # Use full-size TIFF for ROI definition so user draws on full image
    if "after_align_full_save_path" in combined_df.columns:
        combined_df["after_align_save_path"] = combined_df["after_align_full_save_path"].fillna(combined_df["after_align_save_path"])
        # Red dot: use full image coords (corrected_uncaging_x/y) when displaying full-size TIFF
        full_mask = combined_df["after_align_full_save_path"].notna()
        combined_df.loc[full_mask, "uncaging_display_x"] = combined_df.loc[full_mask, "corrected_uncaging_x"]
        combined_df.loc[full_mask, "uncaging_display_y"] = combined_df.loc[full_mask, "corrected_uncaging_y"]
    if df_save_path_1:
        combined_df.to_pickle(df_save_path_1)
        combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))

    if not (Do_without_asking and skip_gui_TF) and df_save_path_1:
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

    if not Do_without_asking and ask_yes_no_gui("Stop here?"):
        return
    if (Do_without_asking or ask_yes_no_gui("Quantify intensity from FLIM (per frame, Ch, ROI)?")) and df_save_path_1:
        out_csv = df_save_path_1.replace(".pkl", "_intensity_from_flim.csv")
        quantify_intensity_from_flim(combined_df, ch_1or2, z_plus_minus, out_csv)
    print("Done.")


if __name__ == "__main__":
    main()
