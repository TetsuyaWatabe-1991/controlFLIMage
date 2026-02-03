# -*- coding: utf-8 -*-
"""
Transient experiment data processing functions.

This module provides functions to process transient-only experiments 
(induction frames only, without pre/post normal frames).

Each file is treated as a separate set for individual analysis.
"""

import sys
sys.path.append('..\\')
import os
import glob
import pandas as pd
from FLIMageAlignment import get_flimfile_list, get_xyz_pixel_um
from FLIMageFileReader2 import FileReader
from datetime import datetime
from typing import List, Optional


def get_transient_pos(file_list: List[str], 
                      transient_frame_num: List[int] = [33, 34, 35, 4],
                      group_by_pattern: bool = True) -> pd.DataFrame:
    """
    Create DataFrame for transient-only experiments.
    
    Each file is treated as a SEPARATE SET (nth_set_label) for individual analysis.
    
    Args:
        file_list: List of FLIM file paths to process
        transient_frame_num: List of acceptable frame numbers for transient files.
                            Files with frame numbers not in this list will be skipped.
        group_by_pattern: If True, use base filename (without number) as group name.
                         If False, use full filename as group name.
    
    Returns:
        pd.DataFrame with columns compatible with GUI ROI analysis system
    """
    combined_df = pd.DataFrame()
    
    # Filter and validate files
    valid_files = []
    skipped_files = []
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"File not found, skipping: {file_path}")
            skipped_files.append((file_path, "not found"))
            continue
            
        try:
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, False)
            n_images = iminfo.n_images
            
            if n_images in transient_frame_num:
                valid_files.append(file_path)
                print(f"Valid transient file ({n_images} frames): {file_path}")
            else:
                skipped_files.append((file_path, f"frame count {n_images} not in {transient_frame_num}"))
                print(f"Skipping file (frame count {n_images} not in {transient_frame_num}): {file_path}")
        except Exception as e:
            skipped_files.append((file_path, str(e)))
            print(f"Error reading file, skipping: {file_path} - {e}")
    
    if len(valid_files) == 0:
        print("No valid transient files found!")
        print(f"Skipped {len(skipped_files)} files:")
        for f, reason in skipped_files:
            print(f"  {f}: {reason}")
        return combined_df
    
    print(f"\nProcessing {len(valid_files)} valid transient files...")
    print(f"Skipped {len(skipped_files)} files\n")
    
    # Sort all files by filename
    valid_files_sorted = sorted(valid_files)
    
    # First pass: find the earliest acquisition time (first_dt)
    print("Finding earliest acquisition time...")
    dt_formatter = "%Y-%m-%dT%H:%M:%S.%f"
    first_dt = None
    
    for file_path in valid_files_sorted:
        try:
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, False)
            dt_str = iminfo.acqTime[0]
            dt = datetime.strptime(dt_str, dt_formatter)
            
            if first_dt is None or dt < first_dt:
                first_dt = dt
        except Exception as e:
            continue
    
    if first_dt is not None:
        print(f"Earliest acquisition time: {first_dt}")
    
    # Second pass: process all files with the correct first_dt reference
    # Each file is a separate set
    for set_label, file_path in enumerate(valid_files_sorted):
        # Extract group name from filepath
        filepath_without_number = file_path[:-8]  # Remove _XXX.flim
        if group_by_pattern:
            group_name = os.path.basename(filepath_without_number)
        else:
            group_name = os.path.basename(file_path)[:-5]  # Remove .flim
        
        try:
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, False)
            
            # Get image metadata
            y_pix = iminfo.statedict["State.Acq.linesPerFrame"]
            x_pix = iminfo.statedict["State.Acq.pixelsPerLine"]
            x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
            
            # Get uncaging position
            uncaging_x_y_0to1 = [0, 0]
            try:
                uncaging_x_y_0to1 = iminfo.statedict["State.Uncaging.Position"]
            except:
                pass
            
            center_x = x_pix * uncaging_x_y_0to1[0]
            center_y = y_pix * uncaging_x_y_0to1[1]
            
            # Print file info with uncaging position
            print(f"\n[Set {set_label}] {os.path.basename(file_path)}")
            print(f"    Uncaging Position: ({center_x:.1f}, {center_y:.1f}) pixels")
            print(f"    Uncaging Position (0-1): ({uncaging_x_y_0to1[0]:.3f}, {uncaging_x_y_0to1[1]:.3f})")
            
            # Get motor position
            motor_position = iminfo.statedict["State.Motor.motorPosition"]
            z_position = motor_position[2]
            
            # Get acquisition time
            dt_str = iminfo.acqTime[0]
            dt = datetime.strptime(dt_str, dt_formatter)
            
            relative_time_sec = (dt - first_dt).total_seconds()
            
            # Create row for this file - each file is its own set
            each_df = pd.DataFrame({
                "nth": [set_label],
                "nth_omit_induction": [0],  # Each set has one "frame" at index 0
                "relative_nth_omit_induction": [0],
                "nth_set_label": [set_label],  # Each file is a separate set
                "group": [group_name],
                "filepath_without_number": [filepath_without_number],
                "file_path": [file_path],
                "transient_frame": [True],
                "uncaging_frame": [True],  # For compatibility
                "titration_frame": [False],
                "unknown_frame": [False],
                "phase": ["transient"],
                "center_x": [center_x],
                "center_y": [center_y],
                "uncaging_x_0to1": [uncaging_x_y_0to1[0]],
                "uncaging_y_0to1": [uncaging_x_y_0to1[1]],
                "stepZ": [iminfo.statedict["State.Acq.sliceStep"]],
                "z_position": [z_position],
                "z_relative_step_nth": [0],
                "dt": [dt],
                "dt_str": [dt_str],
                "relative_time_sec": [relative_time_sec],
                "relative_time_min": [round(relative_time_sec / 60, 1)],
                "x_um": [x_um],
                "y_um": [y_um],
                "z_um": [z_um],
                "n_frames": [iminfo.n_images],
                "statedict": [iminfo.statedict],
                "reject": [False],
                "comment": [""],
            })
            
            combined_df = pd.concat([combined_df, each_df], ignore_index=True)
            
        except Exception as e:
            print(f"  Error processing file: {file_path} - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n=== TRANSIENT DATA SUMMARY ===")
    print(f"Total files processed: {len(combined_df)}")
    print(f"Total sets: {len(combined_df)}")
    if len(combined_df) > 0:
        print(f"Frame counts: {combined_df['n_frames'].unique()}")
    print("=== END SUMMARY ===\n")
    
    return combined_df


def get_transient_files_from_folder(folder_path: str,
                                    pattern: str = "*.flim",
                                    transient_frame_num: List[int] = [33, 34, 35, 4]) -> List[str]:
    """
    Get list of transient files from a folder.
    
    Args:
        folder_path: Path to folder containing FLIM files
        pattern: Glob pattern for file matching
        transient_frame_num: List of acceptable frame numbers
    
    Returns:
        List of valid transient file paths
    """
    file_list = glob.glob(os.path.join(folder_path, pattern))
    
    valid_files = []
    for file_path in file_list:
        try:
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, False)
            if iminfo.n_images in transient_frame_num:
                valid_files.append(file_path)
        except:
            pass
    
    return sorted(valid_files)


if __name__ == "__main__":
    # Example usage
    print("Transient Position Data Extraction")
    print("=" * 50)
    
    print("\nUsage:")
    print("  from get_transient_pos import get_transient_pos, get_transient_files_from_folder")
    print("  file_list = get_transient_files_from_folder(folder_path, transient_frame_num=[33, 34, 4])")
    print("  combined_df = get_transient_pos(file_list)")
