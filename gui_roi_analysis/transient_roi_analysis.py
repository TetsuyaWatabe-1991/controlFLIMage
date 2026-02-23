# -*- coding: utf-8 -*-
"""
Transient and Z-stack ROI Analysis Script

Supports two modes (auto-detected by frame count in FLIM files):

- **Transient**: Uses ONLY files with n_images in TRANSIENT_FRAME_NUM (no Z-stack pre/post).
  Pre/Uncaging/Post are frame ranges within each transient file (e.g. frames 0-1, 2-33, 34-end).
  Creates one TIFF per transient file for ROI definition. Analysis includes F/F0 for uncaging and post.

- **Z-stack**: Files with n_images in ZSTACK_FRAME_NUM. Uses first_processing_for_flim_files
  (same as example_gui_usage_reduce_asking_tiff_only). ROI GUI and TIFF-only saving
  are shared; full Z-stack analysis (GCaMP, etc.) is done via the example script.

Uses the existing PyQt5 GUI ROI system with TIFF-only saving.

Usage:
    1. Run this script
    2. Select a FLIM file (folder is scanned; mode = transient or zstack)
    3. Define ROIs using the GUI
    4. ROI masks are saved as TIFF files
"""

# %%
import os
import sys
sys.path.append('..\\')
sys.path.append(os.path.dirname(__file__))

import glob
import numpy as np
import pandas as pd
import tifffile
from datetime import datetime
from PyQt5.QtWidgets import QApplication

# Import from existing modules
from simple_dialog import ask_yes_no_gui, ask_save_path_gui, ask_open_path_gui
from FLIMageFileReader2 import FileReader
from FLIMageAlignment import Align_4d_array, flim_files_to_nparray

# Import transient-specific function
from AnalysisForFLIMage.get_transient_pos import get_transient_pos, get_transient_files_from_folder

# Import lifetime fitting
from fitting.flim_lifetime_fitting import FLIMLifetimeFitter

# Z-stack path (optional)
try:
    from gui_integration import first_processing_for_flim_files
    HAS_GUI_INTEGRATION = True
except ImportError:
    HAS_GUI_INTEGRATION = False


# %%
# =============================================================================
# Configuration
# =============================================================================

# Frame numbers that identify experiment type (by n_images in FLIM file).
# If any file in folder has n_images in ZSTACK_FRAME_NUM -> Z-stack path (first_processing).
# Else if any file has n_images in TRANSIENT_FRAME_NUM -> transient path only (no Z-stack pre/post).
# Transient mode uses ONLY TRANSIENT_FRAME_NUM files; Z-stack pre/post are never considered.
# These lists should be disjoint.
TRANSIENT_FRAME_NUM = [33, 34, 35, 55]
ZSTACK_FRAME_NUM = [7,11]  # Typical Z-stack frames per file (e.g. 5 z-positions)

# Transient: frame indices for Pre / Uncaging / Post. Non-overlapping ranges.
# Pre = [0, 1], Uncaging = [2 .. POST_START-1], Post = [POST_START .. end].
TRANSIENT_PRE_FRAME_INDICES = [0, 1]
TRANSIENT_POST_FRAME_START = 34  # Uncaging = 2..33, Post = 34..n_frames-1
TRANSIENT_UNCAGING_FRAME_INDICES = [3]  # Used only for expand row count (1 unc row per file)
TRANSIENT_POST_FRAME_INDICES = None  # In code: range(TRANSIENT_POST_FRAME_START, n_frames)
TRANSIENT_POST_FRAME_INDICES = None  # Derived: range(TRANSIENT_POST_FRAME_START, n_frames)

# Z-stack processing parameters (used when mode is Z-stack)
Z_PLUS_MINUS = 2
PRE_LENGTH = 1

# ROI types to define
ROI_TYPES = ["Spine", "DendriticShaft", "Background"]
COLOR_DICT = {"Spine": "red", "DendriticShaft": "blue", "Background": "green"}

# Image processing parameters
CH_1OR2 = 1  # Channel to use (1=GCaMP, 2=tdTomato typically)


# %%
# =============================================================================
# Helper Functions
# =============================================================================

def _parse_acq_time(acq_time_str: str) -> datetime:
    """
    Parse acquisition time string from FLIM file metadata.
    
    Args:
        acq_time_str: Time string in ISO 8601 format like "2026-01-22T21:52:40.335"
    
    Returns:
        datetime object
    """
    # Use fromisoformat which handles ISO 8601 format (with 'T' separator)
    # This is the format used by FLIMage: "2026-01-22T21:52:40.335"
    return datetime.fromisoformat(acq_time_str.strip())


def _get_frame_times_from_flim(file_path: str) -> list:
    """
    Get acquisition times for each frame from FLIM file metadata.
    
    Args:
        file_path: Path to .flim file
    
    Returns:
        List of times in seconds relative to first frame (first frame = 0)
    """
    iminfo = FileReader()
    iminfo.read_imageFile(file_path, False)  # Read tags only, not image data
    
    if not iminfo.acqTime or len(iminfo.acqTime) == 0:
        return []
    
    try:
        # Parse first frame time as reference
        first_time = _parse_acq_time(iminfo.acqTime[0])
        
        # Calculate relative times for each frame
        frame_times = []
        for acq_time_str in iminfo.acqTime:
            frame_time = _parse_acq_time(acq_time_str)
            delta = (frame_time - first_time).total_seconds()
            frame_times.append(delta)
        
        return frame_times
    except Exception as e:
        print(f"  Warning: Could not parse frame times: {e}")
        return []


def detect_mode(folder_path: str,
                transient_frame_num: list = None,
                zstack_frame_num: list = None) -> str:
    """
    Detect whether the folder contains transient or Z-stack data by checking
    n_images of FLIM files. Z-stack takes priority if any file matches.
    
    Args:
        folder_path: Path to folder containing .flim files
        transient_frame_num: List of frame counts for transient (default: TRANSIENT_FRAME_NUM)
        zstack_frame_num: List of frame counts for Z-stack (default: ZSTACK_FRAME_NUM)
    
    Returns:
        "zstack" or "transient"
    """
    transient_frame_num = transient_frame_num or TRANSIENT_FRAME_NUM
    zstack_frame_num = zstack_frame_num or ZSTACK_FRAME_NUM
    file_list = glob.glob(os.path.join(folder_path, "*.flim"))
    has_zstack = False
    has_transient = False
    for file_path in file_list:
        try:
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, False)
            n = iminfo.n_images
            if n in zstack_frame_num:
                has_zstack = True
            if n in transient_frame_num:
                has_transient = True
        except Exception:
            continue
    if has_zstack:
        return "zstack"
    if has_transient:
        return "transient"
    return "transient"  # Default to transient if no match (backward compat)


def expand_transient_to_pre_unc_post(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand transient DataFrame from 1 row per file to (1 + n_unc + 1) rows per file:
    Pre (1), Uncaging (n_unc = len(TRANSIENT_UNCAGING_FRAME_INDICES)), Post (1).
    Same order as example: Pre, then all Uncaging frames, then Post.
    """
    n_unc = len(TRANSIENT_UNCAGING_FRAME_INDICES)
    rows = []
    for _, row in combined_df.iterrows():
        rel_nth = 0
        new_row = row.copy()
        new_row["phase"] = "pre"
        new_row["relative_nth_omit_induction"] = rel_nth
        new_row["nth_omit_induction"] = rel_nth
        rows.append(new_row)
        rel_nth += 1
        for _ in range(n_unc):
            new_row = row.copy()
            new_row["phase"] = "unc"
            new_row["relative_nth_omit_induction"] = rel_nth
            new_row["nth_omit_induction"] = rel_nth
            rows.append(new_row)
            rel_nth += 1
        new_row = row.copy()
        new_row["phase"] = "post"
        new_row["relative_nth_omit_induction"] = rel_nth
        new_row["nth_omit_induction"] = rel_nth
        rows.append(new_row)
    out = pd.DataFrame(rows)
    return out.reset_index(drop=True)


def process_transient_files_simple(combined_df: pd.DataFrame,
                                   ch_1or2: int = 1,
                                   save_tiff: bool = True) -> pd.DataFrame:
    """
    Process transient files: create TIFFs for ROI definition (Pre, Uncaging, Post order).

    Pre = frames in TRANSIENT_PRE_FRAME_INDICES (each frame Z-projected -> one image).
    Uncaging = frames 2 .. TRANSIENT_POST_FRAME_START-1 (each frame Z-projected -> one image).
    Post = frames TRANSIENT_POST_FRAME_START .. end (each frame Z-projected -> one image).
    TIFF = [Pre images..., Uncaging images..., Post images...] (XYT stack).

    Args:
        combined_df: DataFrame from expand_transient_to_pre_unc_post (1 + n_unc + 1 rows per file)
        ch_1or2: Channel to use (1 or 2)
        save_tiff: Whether to save TIFF files

    Returns:
        Updated DataFrame with TIFF paths and n_pre_frames, n_unc_frames, n_post_frames
    """
    print("\n" + "=" * 60)
    print("Processing transient files for ROI analysis (Pre / Uncaging / Post)...")
    print("=" * 60)

    # Create output folders
    for filepath_without_number in combined_df['filepath_without_number'].unique():
        base_folder = os.path.dirname(filepath_without_number)
        tif_folder = os.path.join(base_folder, "tif_transient")
        os.makedirs(tif_folder, exist_ok=True)

    # One TIFF per unique file_path (combined_df has 1 + n_unc + 1 rows per file)
    unique_files = combined_df.drop_duplicates(subset=['file_path'])[['file_path', 'nth_set_label', 'group']]
    for _, row in unique_files.iterrows():
        file_path = row['file_path']
        set_label = row['nth_set_label']
        group = row['group']
        center_x = combined_df.loc[combined_df['file_path'] == file_path].iloc[0]['center_x']
        center_y = combined_df.loc[combined_df['file_path'] == file_path].iloc[0]['center_y']

        print(f"\n[Set {set_label}] {os.path.basename(file_path)}")
        print(f"    Uncaging Position: ({center_x:.1f}, {center_y:.1f}) pixels")

        try:
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, True)
            image_array = np.array(iminfo.image)
            # [T, Z, C, Y, X, photon_bins]
            n_frames = image_array.shape[0]
            n_z = image_array.shape[1]
            ch_idx = ch_1or2 - 1
            print(f"    Image shape: {image_array.shape} (T, Z, C, Y, X, bins)")

            pre_indices = [i for i in TRANSIENT_PRE_FRAME_INDICES if i < n_frames]
            uncaging_indices = list(range(2, min(TRANSIENT_POST_FRAME_START, n_frames)))
            post_indices = list(range(TRANSIENT_POST_FRAME_START, n_frames))

            def z_proj_frame(t):
                # One XY image per time point: sum over bins, then Z projection (max over Z).
                frame_data = image_array[t, :, ch_idx, :, :, :].sum(axis=-1)  # [Z, Y, X]
                if n_z > 1:
                    return frame_data.max(axis=0).astype(np.float32)
                return frame_data[0].astype(np.float32)

            pre_imgs = [z_proj_frame(t) for t in pre_indices]
            unc_imgs = [z_proj_frame(t) for t in uncaging_indices]
            post_imgs = [z_proj_frame(t) for t in post_indices]
            stack_list = pre_imgs + unc_imgs + post_imgs
            stack = np.stack(stack_list, axis=0)

            n_pre = len(pre_imgs)
            n_unc = len(unc_imgs)
            n_post = len(post_imgs)

            if save_tiff:
                base_folder = os.path.dirname(file_path)
                tif_folder = os.path.join(base_folder, "tif_transient")
                tiff_filename = f"{group}_set{set_label}.tif"
                tiff_path = os.path.join(tif_folder, tiff_filename)
                tifffile.imwrite(tiff_path, stack)
                print(f"    Saved: {tiff_filename} (shape: {stack.shape} [Pre x{n_pre}, Unc x{n_unc}, Post x{n_post}])")
                combined_df.loc[combined_df['file_path'] == file_path, 'after_align_save_path'] = tiff_path
                combined_df.loc[combined_df['file_path'] == file_path, 'n_pre_frames'] = n_pre
                combined_df.loc[combined_df['file_path'] == file_path, 'n_unc_frames'] = n_unc
                combined_df.loc[combined_df['file_path'] == file_path, 'n_post_frames'] = n_post

        except Exception as e:
            print(f"    Error processing: {e}")
            import traceback
            traceback.print_exc()
            continue

    return combined_df


def launch_transient_roi_gui(combined_df: pd.DataFrame, df_save_path: str):
    """
    Launch the ROI definition GUI for transient data.
    
    Args:
        combined_df: DataFrame with transient data
        df_save_path: Path to save the DataFrame
    """
    from file_selection_gui_tiff_only import launch_file_selection_gui_tiff_only
    
    print("\n" + "=" * 60)
    print("Launching ROI Definition GUI (TIFF-only mode)")
    print("=" * 60)
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        print("Created new QApplication instance")
    else:
        print("Using existing QApplication instance")
    
    # Launch file selection GUI
    file_selection_gui = launch_file_selection_gui_tiff_only(
        combined_df,
        df_save_path,
        additional_columns=['dt', 'n_frames'],
        save_auto=False
    )
    
    app.exec_()
    print("ROI GUI closed.")
    
    # Reload DataFrame after GUI
    if os.path.exists(df_save_path):
        combined_df = pd.read_pickle(df_save_path)
    
    return combined_df


def _load_roi_mask_from_tiff(tiff_path: str, roi_type: str, frame_idx: int = None) -> np.ndarray:
    """
    Load ROI mask from TIFF file.
    
    Args:
        tiff_path: Path to the data TIFF file (after_align_save_path)
        roi_type: Type of ROI ("Spine", "DendriticShaft", "Background")
        frame_idx: For 3D mask (e.g. Pre/Uncaging/Post), which frame to use (0=Pre, 1=Uncaging, 2=Post).
                   If None, uses first frame (backward compatible).
    
    Returns:
        2D boolean mask array, or None if not found
    """
    if not tiff_path:
        return None
    
    tiff_dir = os.path.dirname(tiff_path)
    tiff_basename = os.path.splitext(os.path.basename(tiff_path))[0]
    roi_mask_path = os.path.join(tiff_dir, f"{tiff_basename}_{roi_type}_roi_mask.tif")
    
    if os.path.exists(roi_mask_path):
        try:
            roi_mask_3d = tifffile.imread(roi_mask_path)
            if len(roi_mask_3d.shape) == 3:
                idx = 0 if frame_idx is None else min(frame_idx, roi_mask_3d.shape[0] - 1)
                return roi_mask_3d[idx] > 0
            else:
                return roi_mask_3d > 0
        except Exception as e:
            print(f"    Error loading ROI mask from {roi_mask_path}: {e}")
            return None
    return None


def _check_reject_status(tiff_path: str) -> bool:
    """
    Check if this data is rejected (file-based check).
    
    Args:
        tiff_path: Path to the data TIFF file
    
    Returns:
        True if rejected, False otherwise
    """
    if not tiff_path:
        return False
    
    tiff_dir = os.path.dirname(tiff_path)
    tiff_basename = os.path.splitext(os.path.basename(tiff_path))[0]
    flag_path = os.path.join(tiff_dir, f"{tiff_basename}_rejected.flag")
    
    return os.path.exists(flag_path)


def analyze_transient_gcamp(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze GCaMP F/F0 for transient data using Pre/Uncaging/Post ROIs.

    Pre/Uncaging/Post frame indices use TRANSIENT_PRE_FRAME_INDICES, TRANSIENT_UNCAGING_FRAME_INDICES,
    TRANSIENT_POST_FRAME_INDICES (same convention as existing code). Uses phase-specific ROI masks
    (frame 0=Pre, 1=Uncaging, 2=Post). F0 from Pre phase, response from Uncaging and Post.
    ROI masks are loaded from TIFF files (TIFF-only mode). Rejected files are skipped.

    Args:
        combined_df: DataFrame with transient data (3 rows per file: pre/unc/post)

    Returns:
        DataFrame with F/F0 values added (GCaMP_Spine_F_F0, _uncaging, _post; same for Shaft)
    """
    print("\n" + "=" * 60)
    print("Analyzing GCaMP transient responses (Pre / Uncaging / Post)...")
    print("=" * 60)

    analyzed_count = 0
    skipped_count = 0
    rejected_count = 0

    # Iterate per unique file (combined_df has 3 rows per file)
    unique_files = combined_df.drop_duplicates(subset=['file_path'])[['file_path', 'nth_set_label', 'group']]
    for _, row in unique_files.iterrows():
        file_path = row['file_path']
        set_label = row['nth_set_label']
        group = row['group']
        file_rows = combined_df[combined_df['file_path'] == file_path]
        tiff_path = file_rows['after_align_save_path'].iloc[0]
        n_pre = int(file_rows['n_pre_frames'].iloc[0]) if 'n_pre_frames' in file_rows.columns and pd.notna(file_rows['n_pre_frames'].iloc[0]) else 1
        n_unc = int(file_rows['n_unc_frames'].iloc[0]) if 'n_unc_frames' in file_rows.columns and pd.notna(file_rows['n_unc_frames'].iloc[0]) else 1
        n_post = int(file_rows['n_post_frames'].iloc[0]) if 'n_post_frames' in file_rows.columns and pd.notna(file_rows['n_post_frames'].iloc[0]) else 1
        pre_frame_idx = 0
        unc_frame_idx = n_pre
        post_frame_idx = n_pre + n_unc

        if _check_reject_status(tiff_path):
            rejected_count += 1
            continue

        spine_pre = _load_roi_mask_from_tiff(tiff_path, "Spine", frame_idx=pre_frame_idx)
        spine_unc = _load_roi_mask_from_tiff(tiff_path, "Spine", frame_idx=unc_frame_idx)
        spine_post = _load_roi_mask_from_tiff(tiff_path, "Spine", frame_idx=post_frame_idx)
        shaft_pre = _load_roi_mask_from_tiff(tiff_path, "DendriticShaft", frame_idx=pre_frame_idx)
        shaft_unc = _load_roi_mask_from_tiff(tiff_path, "DendriticShaft", frame_idx=unc_frame_idx)
        shaft_post = _load_roi_mask_from_tiff(tiff_path, "DendriticShaft", frame_idx=post_frame_idx)
        bg_pre = _load_roi_mask_from_tiff(tiff_path, "Background", frame_idx=pre_frame_idx)
        bg_unc = _load_roi_mask_from_tiff(tiff_path, "Background", frame_idx=unc_frame_idx)
        bg_post = _load_roi_mask_from_tiff(tiff_path, "Background", frame_idx=post_frame_idx)

        if spine_pre is None and spine_unc is None and spine_post is None:
            skipped_count += 1
            continue
        spine_mask = spine_pre if spine_pre is not None else (spine_unc if spine_unc is not None else spine_post)
        if spine_mask is None or not np.any(spine_mask):
            skipped_count += 1
            continue

        try:
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, True)
            image_array = np.array(iminfo.image)
            gcamp_tyx = image_array[:, 0, 0, :, :, :].sum(axis=-1)
            tdtom_tyx = image_array[:, 0, 1, :, :, :].sum(axis=-1)
            n_t = gcamp_tyx.shape[0]
            pre_indices = [i for i in TRANSIENT_PRE_FRAME_INDICES if i < n_t]
            uncaging_indices = list(range(2, min(TRANSIENT_POST_FRAME_START, n_t)))
            post_indices = list(range(TRANSIENT_POST_FRAME_START, n_t))

            def mean_in_phase(mask, indices):
                if mask is None or not np.any(mask):
                    return np.nan
                valid = [i for i in indices if i < n_t]
                if not valid:
                    return np.nan
                return np.mean([gcamp_tyx[t][mask].mean() for t in valid])

            def mean_bg(mask, indices):
                if mask is None or not np.any(mask):
                    return 0.0
                valid = [i for i in indices if i < n_t]
                if not valid:
                    return 0.0
                return np.mean([gcamp_tyx[t][mask].mean() for t in valid])

            print(f"  [Set {set_label}] {os.path.basename(file_path)}")

            # Spine: F0 from Pre, F_unc from Uncaging, F_post from Post (with phase-specific ROI)
            f0_spine = mean_in_phase(spine_pre, pre_indices) - mean_bg(bg_pre, pre_indices)
            f_unc_spine = mean_in_phase(spine_unc, uncaging_indices) - mean_bg(bg_unc, uncaging_indices)
            f_post_spine = mean_in_phase(spine_post, post_indices) - mean_bg(bg_post, post_indices)
            if f0_spine > 0:
                f_f0_unc = f_unc_spine / f0_spine
                f_f0_post = f_post_spine / f0_spine
                f_f0_main = max(f_f0_unc, f_f0_post)
                file_mask = combined_df['file_path'] == file_path
                combined_df.loc[file_mask, 'GCaMP_Spine_F_F0'] = f_f0_main
                combined_df.loc[file_mask & (combined_df['phase'] == 'unc'), 'GCaMP_Spine_F_F0_uncaging'] = f_f0_unc
                combined_df.loc[file_mask & (combined_df['phase'] == 'post'), 'GCaMP_Spine_F_F0_post'] = f_f0_post
                print(f"    Spine F/F0 (uncaging) = {f_f0_unc:.3f}, (post) = {f_f0_post:.3f}")

            # Shaft
            f0_shaft = mean_in_phase(shaft_pre, pre_indices) - mean_bg(bg_pre, pre_indices)
            f_unc_shaft = mean_in_phase(shaft_unc, uncaging_indices) - mean_bg(bg_unc, uncaging_indices)
            f_post_shaft = mean_in_phase(shaft_post, post_indices) - mean_bg(bg_post, post_indices)
            if f0_shaft > 0:
                f_f0_unc_s = f_unc_shaft / f0_shaft
                f_f0_post_s = f_post_shaft / f0_shaft
                f_f0_main_s = max(f_f0_unc_s, f_f0_post_s)
                file_mask = combined_df['file_path'] == file_path
                combined_df.loc[file_mask, 'GCaMP_Shaft_F_F0'] = f_f0_main_s
                combined_df.loc[file_mask & (combined_df['phase'] == 'unc'), 'GCaMP_Shaft_F_F0_uncaging'] = f_f0_unc_s
                combined_df.loc[file_mask & (combined_df['phase'] == 'post'), 'GCaMP_Shaft_F_F0_post'] = f_f0_post_s
                print(f"    Shaft F/F0 (uncaging) = {f_f0_unc_s:.3f}, (post) = {f_f0_post_s:.3f}")

            analyzed_count += 1
        except Exception as e:
            print(f"  Error analyzing set {set_label}: {e}")
            skipped_count += 1
            continue

    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Analyzed: {analyzed_count}")
    print(f"Skipped (no ROI): {skipped_count}")
    print(f"Rejected: {rejected_count}")
    print("========================\n")

    return combined_df


def analyze_transient_full_timeseries(combined_df: pd.DataFrame,
                                      output_csv_path: str,
                                      sync_rate: float = 80e6,
                                      photon_threshold: int = 15,
                                      total_photon_threshold: int = 1000) -> pd.DataFrame:
    """
    Analyze intensity and lifetime for all time frames for Ch1 and Ch2.
    
    Outputs a CSV file with intensity and lifetime for each ROI type,
    channel, and time frame.
    
    Args:
        combined_df: DataFrame with transient data
        output_csv_path: Path to save the CSV output
        sync_rate: Laser sync rate (Hz), default 80 MHz
        photon_threshold: Minimum photon count per pixel for lifetime calculation
        total_photon_threshold: Minimum total photon count for lifetime fitting
    
    Returns:
        DataFrame with full time-series data
    """
    print("\n" + "=" * 60)
    print("Analyzing full time-series (intensity & lifetime)...")
    print("=" * 60)
    
    # Initialize lifetime fitter
    fitter = FLIMLifetimeFitter()
    
    # Collect all results
    all_results = []
    
    analyzed_count = 0
    skipped_count = 0
    rejected_count = 0
    
    # One time-series per unique file (combined_df has 3 rows per file for Pre/Uncaging/Post)
    unique_files = combined_df.drop_duplicates(subset=['file_path'])[['file_path', 'nth_set_label', 'group', 'after_align_save_path', 'center_x', 'center_y']]
    for idx, row in unique_files.iterrows():
        set_label = row['nth_set_label']
        group = row['group']
        file_path = row['file_path']
        tiff_path = row.get('after_align_save_path', None)
        center_x = row.get('center_x', 0)
        center_y = row.get('center_y', 0)
        
        if _check_reject_status(tiff_path):
            rejected_count += 1
            continue
        
        # Use Pre-phase ROI (frame 0) for full time-series
        spine_mask = _load_roi_mask_from_tiff(tiff_path, "Spine", frame_idx=0)
        shaft_mask = _load_roi_mask_from_tiff(tiff_path, "DendriticShaft", frame_idx=0)
        bg_mask = _load_roi_mask_from_tiff(tiff_path, "Background", frame_idx=0)
        if spine_mask is None:
            skipped_count += 1
            continue
        
        try:
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, True)
            image_array = np.array(iminfo.image)
            n_frames = image_array.shape[0]
            n_bins = image_array.shape[5]
            frame_times = _get_frame_times_from_flim(file_path)
            if len(frame_times) != n_frames:
                frame_times = [np.nan] * n_frames
            
            print(f"  [Set {set_label}] {os.path.basename(file_path)} ({n_frames} frames)")
            
            # Calculate ps_per_unit
            ps_per_unit = (10**12) / sync_rate / n_bins
            
            # ROI types and masks
            roi_dict = {
                "Spine": spine_mask,
                "DendriticShaft": shaft_mask,
                "Background": bg_mask
            }
            
            # Process each time frame
            for t in range(n_frames):
                result_row = {
                    'set_label': set_label,
                    'group': group,
                    'file_path': file_path,
                    'frame': t,
                    'time_sec': frame_times[t],  # Time in seconds from first frame
                    'center_x': center_x,
                    'center_y': center_y,
                }
                
                # Process each channel (Ch1=GCaMP, Ch2=tdTomato)
                for ch_idx, ch_name in enumerate(['Ch1', 'Ch2']):
                    # Get single frame data: [Y, X, photon_bins]
                    frame_data = image_array[t, 0, ch_idx, :, :, :]
                    
                    # Process each ROI
                    for roi_type, roi_mask in roi_dict.items():
                        if roi_mask is None or not np.any(roi_mask):
                            result_row[f'{roi_type}_{ch_name}_intensity'] = np.nan
                            # Skip lifetime columns for Background
                            if roi_type != "Background":
                                result_row[f'{roi_type}_{ch_name}_total_photon'] = np.nan
                                result_row[f'{roi_type}_{ch_name}_lifetime'] = np.nan
                            continue
                        
                        # Intensity: sum of all photon bins (mean per pixel)
                        intensity_image = frame_data.sum(axis=-1)  # [Y, X]
                        intensity = intensity_image[roi_mask].mean()
                        result_row[f'{roi_type}_{ch_name}_intensity'] = intensity
                        
                        # Skip lifetime calculation for Background
                        if roi_type == "Background":
                            continue
                        
                        # Lifetime fitting (for Spine and DendriticShaft only)
                        # Apply photon threshold
                        photon_exclude_mask = intensity_image <= photon_threshold
                        thresholded_data = frame_data.copy()
                        thresholded_data[photon_exclude_mask, :] = 0
                        
                        # Get histogram for ROI
                        voxels_of_interest = thresholded_data[roi_mask, :]
                        y_data = np.sum(voxels_of_interest, axis=0)
                        
                        # Output total photon count for later threshold filtering
                        total_photon = np.sum(y_data)
                        result_row[f'{roi_type}_{ch_name}_total_photon'] = total_photon
                        
                        if total_photon < total_photon_threshold:
                            result_row[f'{roi_type}_{ch_name}_lifetime'] = np.nan
                        else:
                            try:
                                x = np.arange(len(y_data))
                                result = fitter.fit_double_exponential(x, y_data, ps_per_unit, sync_rate)
                                result_row[f'{roi_type}_{ch_name}_lifetime'] = result.get("lifetime", np.nan)
                            except Exception as e:
                                result_row[f'{roi_type}_{ch_name}_lifetime'] = np.nan
                
                all_results.append(result_row)
            
            analyzed_count += 1
                    
        except Exception as e:
            print(f"  Error analyzing set {set_label}: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue
    
    # Create output DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved full time-series data to: {output_csv_path}")
    
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Analyzed: {analyzed_count} files")
    print(f"Total rows: {len(results_df)}")
    print(f"Skipped (no ROI): {skipped_count}")
    print(f"Rejected: {rejected_count}")
    print("========================\n")
    
    return results_df


# %%
# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Transient / Z-stack ROI Analysis")
    print("=" * 60)
    print(f"Transient frame numbers: {TRANSIENT_FRAME_NUM}")
    print(f"Z-stack frame numbers: {ZSTACK_FRAME_NUM}")
    print(f"Channel: {CH_1OR2}")
    print()
    
    resume_mode = False
    combined_df = None
    df_save_path = None
    folder_path = None
    data_mode = "transient"  # "transient" or "zstack"
    
    # Step 0: Resume from existing DataFrame?
    print("Step 0: Resume from existing analysis or start new?")
    if ask_yes_no_gui("Resume from existing DataFrame? (No = start new analysis)"):
        print("Select the existing combined_df .pkl file to resume:")
        df_save_path = ask_open_path_gui(filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        
        if df_save_path and os.path.exists(df_save_path):
            try:
                combined_df = pd.read_pickle(df_save_path)
                folder_path = os.path.dirname(df_save_path)
                resume_mode = True
                data_mode = str(combined_df["data_mode"].iat[0]) if "data_mode" in combined_df.columns else "transient"
                print(f"Loaded existing DataFrame from: {df_save_path}")
                print(f"  - {len(combined_df)} rows, mode: {data_mode}")
                print(f"  - Columns: {list(combined_df.columns)}")
            except Exception as e:
                print(f"Error loading DataFrame: {e}")
                resume_mode = False
        else:
            print("No file selected or file not found. Starting new analysis.")
            resume_mode = False
    
    if not resume_mode:
        print("\nStep 1: Select a FLIM file (transient or Z-stack folder)")
        one_of_filepath = ask_open_path_gui(filetypes=[("FLIM files", "*.flim")])
        
        if not one_of_filepath:
            print("No file selected. Exiting.")
            sys.exit(0)
        
        print(f"Selected: {one_of_filepath}")
        folder_path = os.path.dirname(one_of_filepath)
        file_list = glob.glob(os.path.join(folder_path, "*.flim"))
        print(f"Found {len(file_list)} FLIM files in folder")
        
        # Detect mode: Z-stack vs Transient (Z-stack takes priority)
        data_mode = detect_mode(folder_path)
        print(f"\nDetected mode: {data_mode}")
        
        if data_mode == "zstack" and HAS_GUI_INTEGRATION:
            # Z-stack path: use first_processing_for_flim_files (same as example_gui_usage_reduce_asking_tiff_only)
            print("\nStep 2 (Z-stack): Running first_processing_for_flim_files...")
            combined_df = first_processing_for_flim_files(
                one_of_filepath,
                Z_PLUS_MINUS,
                CH_1OR2,
                pre_length=PRE_LENGTH,
                save_plot_TF=True,
                save_tif_TF=True,
                return_error_dict=False
            )
            combined_df["data_mode"] = "zstack"
            df_save_path = os.path.join(folder_path, "combined_df.pkl")
            combined_df.to_pickle(df_save_path)
            combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))
            print(f"\nSaved DataFrame to: {df_save_path}")
        elif data_mode == "zstack" and not HAS_GUI_INTEGRATION:
            print("Z-stack mode detected but gui_integration not available. Fallback to transient.")
            data_mode = "transient"
        
        if data_mode == "transient":
            # Transient path: only TRANSIENT_FRAME_NUM files (Z-stack pre/post are not used at all).
            # Pre/Uncaging/Post = frame indices within each transient file, not separate Z-stack files.
            print("\nStep 2 (Transient): Processing transient files only (no Z-stack pre/post)...")
            combined_df = get_transient_pos(
                file_list,
                transient_frame_num=TRANSIENT_FRAME_NUM,
                group_by_pattern=True
            )
            if len(combined_df) == 0:
                print("No valid transient files found. Exiting.")
                sys.exit(0)
            combined_df = expand_transient_to_pre_unc_post(combined_df)
            print("\nStep 3: Creating Pre/Uncaging/Post TIFFs...")
            combined_df = process_transient_files_simple(
                combined_df,
                ch_1or2=CH_1OR2,
                save_tiff=True
            )
            combined_df["data_mode"] = "transient"
            df_save_path = os.path.join(folder_path, "transient_combined_df.pkl")
            combined_df.to_pickle(df_save_path)
            combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))
            print(f"\nSaved DataFrame to: {df_save_path}")
    
    # Step 4: Launch ROI GUI
    if combined_df is None or df_save_path is None:
        print("No DataFrame or save path. Skipping GUI and analysis.")
    else:
        print("\nStep 4: Launch ROI definition GUI?")
        if ask_yes_no_gui("Do you want to define ROIs now?"):
            combined_df = launch_transient_roi_gui(combined_df, df_save_path)
            if df_save_path and os.path.exists(df_save_path):
                combined_df = pd.read_pickle(df_save_path)
        
        # Step 5: Analyze GCaMP F/F0 (transient mode only)
        if data_mode == "transient" and combined_df is not None and df_save_path:
            print("\nStep 5: Analyze GCaMP F/F0?")
            if ask_yes_no_gui("Do you want to analyze GCaMP F/F0 (Pre/Uncaging/Post)?"):
                combined_df = analyze_transient_gcamp(combined_df)
                combined_df.to_pickle(df_save_path)
                combined_df.to_csv(df_save_path.replace(".pkl", "_analyzed.csv"))
                print(f"\nSaved analysis results to: {df_save_path}")
        
        # Step 6: Full time-series analysis (transient mode only)
        if data_mode == "transient" and combined_df is not None and df_save_path:
            print("\nStep 6: Full time-series analysis (intensity & lifetime)?")
            if ask_yes_no_gui("Analyze intensity and lifetime for all time frames? (Takes time)"):
                timeseries_csv_path = df_save_path.replace(".pkl", "_full_timeseries.csv")
                analyze_transient_full_timeseries(
                    combined_df,
                    output_csv_path=timeseries_csv_path,
                    sync_rate=80e6,
                    photon_threshold=5,
                    total_photon_threshold=100
                )
                print(f"\nSaved full time-series data to: {timeseries_csv_path}")
    
    print("\n" + "=" * 60)
    print("Transient ROI Analysis Complete!")
    print("=" * 60)

