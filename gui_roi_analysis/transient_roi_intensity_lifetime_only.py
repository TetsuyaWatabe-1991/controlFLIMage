# -*- coding: utf-8 -*-
"""
Transient (Uncaging) ROI: ROI definition + full-frame intensity and lifetime only.

This script does ONLY:
  1. Load uncaging FLIM files and create TIFFs for ROI definition (Pre/Uncaging/Post).
  2. Launch ROI GUI to define ROIs.
  3. Compute intensity and lifetime for ALL frames (full time-series).

No F/F0, no Pre/Uncaging/Post phase analysis. Uses uncaging FLIM files only.

Usage:
  1. Run this script
  2. Select a FLIM file (folder must contain transient/uncaging files only)
  3. Define ROIs in the GUI
  4. Full time-series intensity and lifetime are computed and saved to CSV
"""

import os
import sys
sys.path.append('..\\')
sys.path.append(os.path.dirname(__file__))

import glob
import pandas as pd

from simple_dialog import ask_yes_no_gui, ask_open_path_gui
from AnalysisForFLIMage.get_annotation_unc_multiple import get_uncaging_pos_multiple

# Import from transient_roi_analysis (shared logic)
from transient_roi_analysis import (
    TRANSIENT_FRAME_NUM,
    CH_1OR2,
    PRE_LENGTH,
    process_transient_files_simple,
    launch_transient_roi_gui,
    analyze_transient_full_timeseries,
)


def main():
    print("=" * 60)
    print("Transient ROI: Intensity & Lifetime (full frames only)")
    print("=" * 60)
    print(f"Transient frame numbers: {TRANSIENT_FRAME_NUM}")
    print(f"Channel: {CH_1OR2}")
    print()

    combined_df = None
    df_save_path = None
    folder_path = None

    # Step 0: Resume from existing DataFrame?
    print("Step 0: Resume from existing analysis or start new?")
    if ask_yes_no_gui("Resume from existing DataFrame? (No = start new)"):
        print("Select the existing combined_df .pkl file to resume:")
        df_save_path = ask_open_path_gui(filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if df_save_path and os.path.exists(df_save_path):
            try:
                combined_df = pd.read_pickle(df_save_path)
                folder_path = os.path.dirname(df_save_path)
                print(f"Loaded existing DataFrame from: {df_save_path}")
                print(f"  - {len(combined_df)} rows")
            except Exception as e:
                print(f"Error loading DataFrame: {e}")
                combined_df = None
        else:
            print("No file selected or file not found. Starting new analysis.")

    if combined_df is None:
        # Step 1: Select a FLIM file
        print("\nStep 1: Select a FLIM file (uncaging/transient folder)")
        one_of_filepath = ask_open_path_gui(filetypes=[("FLIM files", "*.flim")])
        if not one_of_filepath:
            print("No file selected. Exiting.")
            sys.exit(0)

        print(f"Selected: {one_of_filepath}")
        folder_path = os.path.dirname(one_of_filepath)
        one_of_file_list = glob.glob(os.path.join(folder_path, "*_highmag_*002.flim"))
        ignore_words = ["for_align"]
        one_of_file_list = [f for f in one_of_file_list if not any(w in f for w in ignore_words)]
        if one_of_filepath and one_of_filepath.endswith(".flim") and len(one_of_filepath) > 8:
            base = one_of_filepath[:-8]
            first_in_group = base + "002.flim"
            if os.path.isfile(first_in_group) and first_in_group not in one_of_file_list:
                one_of_file_list.append(first_in_group)
        if len(one_of_file_list) == 0:
            print("No *_highmag_*002.flim files found. Exiting.")
            sys.exit(1)

        # Step 2: Get phase per file (same as first_processing), then build TIFFs by phase
        print("\nStep 2: Getting phase per file (get_uncaging_pos_multiple)...")
        combined_df = get_uncaging_pos_multiple(one_of_file_list, pre_length=PRE_LENGTH)
        combined_df = combined_df[combined_df["phase"].isin(["pre", "unc", "post"])].copy()
        if len(combined_df) == 0:
            print("No pre/unc/post files found. Exiting.")
            sys.exit(1)
        print("\nStep 3: Creating Pre/Uncaging/Post TIFFs (by phase, Uncaging no Z-proj)...")
        combined_df = process_transient_files_simple(
            combined_df,
            ch_1or2=CH_1OR2,
            save_tiff=True
        )

        df_save_path = os.path.join(folder_path, "transient_combined_df.pkl")
        combined_df.to_pickle(df_save_path)
        combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))
        print(f"\nSaved DataFrame to: {df_save_path}")

    # Step 4: Launch ROI GUI
    if combined_df is None or df_save_path is None:
        print("No DataFrame or save path. Exiting.")
        sys.exit(1)

    print("\nStep 4: Define ROIs?")
    if ask_yes_no_gui("Do you want to define ROIs now?"):
        combined_df = launch_transient_roi_gui(combined_df, df_save_path)
        if df_save_path and os.path.exists(df_save_path):
            combined_df = pd.read_pickle(df_save_path)

    # Step 5: Full time-series intensity and lifetime only
    print("\nStep 5: Compute intensity and lifetime for all frames?")
    if ask_yes_no_gui("Analyze intensity and lifetime for all time frames?"):
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
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
