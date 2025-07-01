import sys
import os
sys.path.append("..\\")
sys.path.append(os.path.dirname(__file__))
import glob
import numpy as np
import pandas as pd
import tifffile
from PyQt5.QtWidgets import QApplication
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict, Optional
from AnalysisForFLIMage.get_annotation_unc_multiple import get_uncaging_pos_multiple
from FLIMageAlignment import Align_4d_array, flim_files_to_nparray                            
from roi_analysis_gui import ROIAnalysisGUI



def first_processing_for_flim_files(
    one_of_filepath,
    z_plus_minus,
    ch_1or2,
    pre_length,
    save_plot_TF = True,
    save_tif_TF = True,    
    ignore_words = ["for_align"]
    ) -> pd.DataFrame:

    # Load initial data
    
    one_of_file_list = glob.glob(
        os.path.join(
            os.path.dirname(one_of_filepath), 
            "*_highmag_*002.flim"
            )
        )
    one_of_file_list = [each_file for each_file in one_of_file_list if not any(ignore_word in each_file for ignore_word in ignore_words)]

    plot_savefolder = os.path.join(os.path.dirname(one_of_filepath), "plot")
    tif_savefolder = os.path.join(os.path.dirname(one_of_filepath), "tif")
    roi_savefolder = os.path.join(os.path.dirname(one_of_filepath), "roi")
    for each_folder in [plot_savefolder, tif_savefolder, roi_savefolder]:
        os.makedirs(each_folder, exist_ok=True)
    
    combined_df = get_uncaging_pos_multiple(one_of_file_list, pre_length=pre_length)
    
    # Add data validation check
    print(f"\n=== DATA VALIDATION FOR {os.path.basename(one_of_filepath)} ===")
    print(f"Total files found: {len(one_of_file_list)}")
    print(f"Combined dataframe shape: {combined_df.shape}")
    
    # Check for potential issues
    if 'nth' in combined_df.columns:
        nth_values = combined_df['nth'].unique()
        print(f"Unique nth values: {sorted(nth_values)}")
        
        # Check for negative or very large nth values
        negative_nth = combined_df[combined_df['nth'] < 0]
        if len(negative_nth) > 0:
            print(f"WARNING: Found {len(negative_nth)} rows with negative nth values")
            print(f"Negative nth values: {negative_nth['nth'].unique()}")
    
    if 'phase' in combined_df.columns:
        phase_counts = combined_df['phase'].value_counts()
        print(f"Phase distribution: {phase_counts.to_dict()}")
        
        # Check if uncaging data exists
        unc_data = combined_df[combined_df['phase'] == 'unc']
        if len(unc_data) == 0:
            print("WARNING: No uncaging data found in this dataset")
        else:
            print(f"Found {len(unc_data)} uncaging rows")
    
    print("=== END DATA VALIDATION ===\n")
    
    # Process each group
    for each_filepath_without_number in combined_df['filepath_without_number'].unique():
        each_filegroup_df = combined_df[combined_df['filepath_without_number'] == each_filepath_without_number]
        for each_group in each_filegroup_df['group'].unique():
            each_group_df = each_filegroup_df[each_filegroup_df['group'] == each_group]

            filelist = each_group_df["file_path"].tolist()
            # Load and align data
            Aligned_4d_array, shifts, _ = load_and_align_data(filelist, ch=ch_1or2 - 1)
            
            # Add alignment data validation
            print(f"Group {each_group}: Aligned array shape: {Aligned_4d_array.shape}, Shifts shape: {shifts.shape}")
            
            # Process uncaging positions
            each_group_df = process_uncaging_positions(each_group_df, shifts, Aligned_4d_array)
            
            # Update combined_df with corrected uncaging positions
            for col in ['corrected_uncaging_x', 'corrected_uncaging_y', 'corrected_uncaging_z']:
                if col in each_group_df.columns:
                    combined_df.loc[each_group_df.index, col] = each_group_df[col].values
            
            # Store individual shift values for each frame
            valid_df = each_group_df[each_group_df["nth_omit_induction"] != -1].copy()
            valid_df_sorted = valid_df.sort_values("nth_omit_induction")
            
            for i, (idx, row) in enumerate(valid_df_sorted.iterrows()):
                if i < len(shifts):
                    shift_z, shift_y, shift_x = shifts[i][0], shifts[i][1], shifts[i][2]
                    combined_df.loc[idx, 'shift_z'] = shift_z
                    combined_df.loc[idx, 'shift_y'] = shift_y
                    combined_df.loc[idx, 'shift_x'] = shift_x
            
            # Save full region plots
            list_of_save_path = save_full_region_plots(each_group_df, Aligned_4d_array, plot_savefolder, 
                                                       z_plus_minus,
                                                       return_list_of_save_path=True)
            
            getting_length_df = each_group_df[(each_group_df["nth_omit_induction"] != -1) & 
                                        (each_group_df["nth_set_label"] != -1)]
            # print(f"length of getting_length_df: {len(getting_length_df)}")
            # print(f"length of list_of_save_path: {len(list_of_save_path)}")
            combined_df.loc[getting_length_df.index, "save_full_region_plot_path"] = list_of_save_path

            # Process small regions
            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                    
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
                
                small_Tiff_MultiArray, small_Aligned_4d_array, corrected_positions, each_set_df = process_small_region(
                    each_set_df, Aligned_4d_array
                )

                # Update combined_df with small region boundaries and small shifts
                for col in ['small_z_from', 'small_z_to', 'small_x_from', 'small_x_to', 'small_y_from', 'small_y_to']:
                    if col in each_set_df.columns:
                        combined_df.loc[each_set_df.index, col] = each_set_df[col].values

                combined_df.loc[each_set_df.index, "small_shift_z"] = each_set_df["small_shift_z"].values
                combined_df.loc[each_set_df.index, "small_shift_y"] = each_set_df["small_shift_y"].values
                combined_df.loc[each_set_df.index, "small_shift_x"] = each_set_df["small_shift_x"].values

                
                list_of_save_path = save_small_region_plots(
                    small_Aligned_4d_array,
                    corrected_positions,
                    each_set_label,
                    plot_savefolder,
                    z_plus_minus,
                    each_set_df,
                    return_list_of_save_path=True,
                    save_plot_TF=save_plot_TF
                )
                no_uncaging_df = each_set_df[each_set_df["nth_omit_induction"] != -1]
                count_up = -1
                for ind, each_row in no_uncaging_df.iterrows():
                    count_up += 1
                    combined_df.loc[ind, "relative_nth_omit_induction"] = count_up

                combined_df.loc[no_uncaging_df.index, "save_small_region_plot_path"] = list_of_save_path
            

                savepath_dict = save_small_region_tiffs(
                    small_Tiff_MultiArray,
                    small_Aligned_4d_array,
                    corrected_positions,
                    each_group,
                    each_set_label,
                    tif_savefolder,
                    z_plus_minus,
                    return_save_path=True,
                    save_tif_TF=save_tif_TF
                )
                combined_df.loc[each_set_df.index, "before_align_save_path"] = savepath_dict["save_path_before_align"]
                combined_df.loc[each_set_df.index, "after_align_save_path"] = savepath_dict["save_path_after_align"]

    if save_tif_TF*save_plot_TF:    
        for each_group in combined_df['group'].unique():
            each_group_df = combined_df[combined_df['group'] == each_group]
            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                max_nth_omit_ind_before_unc = each_group_df[each_group_df["phase"] == "pre"]["nth_omit_induction"].max()
                plot_path = each_group_df[each_group_df["nth_omit_induction"] == max_nth_omit_ind_before_unc]["save_full_region_plot_path"].values[0]
                img = plt.imread(plot_path)
                plt.imshow(img)
                plt.axis("off")
                plt.show()
                after_align_tiff_path = each_group_df[each_group_df["nth_omit_induction"] == max_nth_omit_ind_before_unc]["after_align_save_path"].values[0]
                after_align_tiff = tifffile.imread(after_align_tiff_path)
                max_proj = after_align_tiff.max(axis=0)
                plt.imshow(max_proj, cmap="gray")
                plt.show() 
    return combined_df


def shift_coords_small_to_full_for_each_rois(combined_df, z_plus_minus, 
                roi_types = ["Spine", "DendriticShaft", "Background"],
                image_shape = (128, 128)):

    for each_roi_type in roi_types:
        # Ensure shifted_mask column exists and is object type for storing numpy arrays
        if f'{each_roi_type}_shifted_mask' not in combined_df.columns:
            combined_df[f'{each_roi_type}_shifted_mask'] = None
        combined_df[f'{each_roi_type}_shifted_mask'] = combined_df[f'{each_roi_type}_shifted_mask'].astype(object)

        for filepath_without_number in combined_df["filepath_without_number"].unique():
            each_group_df = combined_df[combined_df["filepath_without_number"] == filepath_without_number]
            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
                for nth_omit_induction in each_set_df["nth_omit_induction"].unique():
                    if nth_omit_induction == -1:
                        continue
                    current_index = each_set_df[each_set_df["nth_omit_induction"] == nth_omit_induction].index[0]

                    each_roi_mask = each_set_df.loc[current_index, f"{each_roi_type}_roi_mask"]
                    if each_roi_mask is None:
                        continue
                    
                    shift_x = each_set_df.loc[current_index, "shift_x"]
                    shift_y = each_set_df.loc[current_index, "shift_y"]
                    small_shift_x = each_set_df.loc[current_index, "small_shift_x"]
                    small_shift_y = each_set_df.loc[current_index, "small_shift_y"]
                    small_x_from = each_set_df.loc[current_index, "small_x_from"]
                    small_y_from = each_set_df.loc[current_index, "small_y_from"]
                    
                    total_shift_x = shift_x + small_shift_x
                    total_shift_y = shift_y + small_shift_y

                    coords = np.argwhere(each_roi_mask)
                    shifted_coords = coords.copy()
                    try:
                        shifted_coords[:, 0] = coords[:, 0] + small_y_from - total_shift_y
                        shifted_coords[:, 1] = coords[:, 1] + small_x_from - total_shift_x
                        shifted_mask = np.zeros(image_shape, dtype=bool)
                        shifted_mask[shifted_coords[:, 0], shifted_coords[:, 1]] = True
                    except:
                        print(f"error at {filepath_without_number} {each_set_label} {nth_omit_induction}")
                        print(f"small_y_from: {small_y_from}")
                        print(f"small_x_from: {small_x_from}")
                        print(f"total_shift_y: {total_shift_y}")
                        print(f"total_shift_x: {total_shift_x}")
                        print(f"shifted_coords[:, 0].max(): {shifted_coords[:, 0].max()}")
                        print(f"shifted_coords[:, 0].min(): {shifted_coords[:, 0].min()}")
                        print(f"shifted_coords[:, 1].max(): {shifted_coords[:, 1].max()}")
                        print(f"shifted_coords[:, 1].min(): {shifted_coords[:, 1].min()}")
                        print("No shift was applied")
                        shifted_coords = coords.copy()
                        shifted_mask = np.zeros(image_shape, dtype=bool)
                        shifted_mask[shifted_coords[:, 0], shifted_coords[:, 1]] = True
                    combined_df.at[current_index, f"{each_roi_type}_shifted_mask"] = shifted_mask.copy()

    return combined_df


def shift_coords_small_to_full(combined_df, z_plus_minus, image_shape = (128, 128)):
    # Ensure shifted_mask column exists and is object type for storing numpy arrays
    if 'shifted_mask' not in combined_df.columns:
        combined_df['shifted_mask'] = None
    combined_df['shifted_mask'] = combined_df['shifted_mask'].astype(object)
    
    for nth_omit_induction in combined_df["nth_omit_induction"].unique():
        if nth_omit_induction == -1:
            continue
        if combined_df[combined_df["nth_omit_induction"] == nth_omit_induction]["nth_set_label"].values[0] == -1:
            continue
        # each_tiff_array = Tiff_MultiArray[nth_image,:,:,:]
        current_index = combined_df[combined_df["nth_omit_induction"] == nth_omit_induction].index[0]
        shift_x = combined_df.loc[current_index, "shift_x"]
        shift_y = combined_df.loc[current_index, "shift_y"]
        small_shift_x = combined_df.loc[current_index, "small_shift_x"]
        small_shift_y = combined_df.loc[current_index, "small_shift_y"]
        small_x_from = combined_df.loc[current_index, "small_x_from"]
        small_y_from = combined_df.loc[current_index, "small_y_from"]
        
        total_shift_x = shift_x + small_shift_x
        total_shift_y = shift_y + small_shift_y

        each_roi_mask = combined_df.loc[current_index, "roi_mask"]
        coords = np.argwhere(each_roi_mask)

        shifted_coords = coords.copy()
        shifted_coords[:, 0] = coords[:, 0] + small_y_from - total_shift_y
        shifted_coords[:, 1] = coords[:, 1] + small_x_from - total_shift_x
        shifted_mask = np.zeros(image_shape, dtype=bool)
        shifted_mask[shifted_coords[:, 0], shifted_coords[:, 1]] = True
        combined_df.at[current_index, "shifted_mask"] = shifted_mask

    return combined_df


def load_and_align_data(
        filelist: List[str], ch: int
        ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Load FLIM files and perform alignment."""
    Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray(
        filelist, ch=ch, normalize_by_averageNum=True
      )   
    shifts, Aligned_4d_array = Align_4d_array(Tiff_MultiArray)
    return Aligned_4d_array, shifts, relative_sec_list

def process_uncaging_positions(
    each_group_df: pd.DataFrame,
    shifts: np.ndarray,
    Aligned_4d_array: np.ndarray
    ) -> pd.DataFrame:
    """Process and correct uncaging positions based on alignment shifts."""
    for each_set_label in each_group_df["nth_set_label"].unique():
        if each_set_label == -1:
            continue
                
        each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
        each_set_unc_row = each_set_df[each_set_df["phase"] == "unc"]
        if len(each_set_unc_row) == 0:
            print(f"No uncaging data found for group {each_group_df['group'].iloc[0]}, set {each_set_label}")
            continue
        assert len(each_set_unc_row) == 1
        
        # Get the nth value for uncaging row
        unc_nth = each_set_unc_row.loc[:, "nth"].values[0]
        
        # Validate nth value against shifts array bounds
        if unc_nth < 0 or unc_nth >= len(shifts):
            print(f"ERROR: Invalid nth value {unc_nth} for shifts array of length {len(shifts)}")
            print(f"Group: {each_group_df['group'].iloc[0]}, Set: {each_set_label}")
            print(f"each_set_unc_row: {each_set_unc_row}")
            print(f"shifts shape: {shifts.shape}")
            print(f"Available nth values in each_set_df: {each_set_df['nth'].unique()}")
            continue
        
        # Calculate z shifts with bounds checking
        if unc_nth - 1 >= 0 and unc_nth - 1 < len(shifts):
            z_shift_last_pre = shifts[unc_nth - 1, 0]
        else:
            print(f"WARNING: Cannot access shifts[{unc_nth - 1}] for z shift calculation")
            print(f"Using 0 as default z shift")
            z_shift_last_pre = 0
            
        z_shift_last_pre_rounded = round(z_shift_last_pre, 0)
        z_relative_to_last_pre = each_set_unc_row.loc[:, "z_relative_step_nth"].values[0]
        z_nth_relative_to_first = round(z_relative_to_last_pre - z_shift_last_pre_rounded)
        
        # Update positions with bounds checking
        # print("each_set_df", each_set_df)
        # print("each_set_unc_row", each_set_unc_row)
        # print("len(each_set_unc_row)", len(each_set_unc_row))
        # print("shifts", shifts)
        # print("len(shifts)", len(shifts))
        # print("unc_nth", unc_nth)
        # print("z_nth_relative_to_first", z_nth_relative_to_first)

        # Get shift values with bounds checking
        if unc_nth < len(shifts):
            shift_x = shifts[unc_nth, 2]
            shift_y = shifts[unc_nth, 1]
        else:
            print(f"WARNING: nth value {unc_nth} exceeds shifts array bounds")
            shift_x = 0
            shift_y = 0

        each_group_df.loc[each_set_df.index, "corrected_uncaging_x"] = (
            each_set_unc_row.loc[:, "center_x"] + shift_x
        ).values[0]
        each_group_df.loc[each_set_df.index, "corrected_uncaging_y"] = (
            each_set_unc_row.loc[:, "center_y"] + shift_y
        ).values[0]
        each_group_df.loc[each_set_df.index, "corrected_uncaging_z"] = z_nth_relative_to_first
    
    return each_group_df

def save_full_region_plots(
    each_group_df: pd.DataFrame,
    Aligned_4d_array: np.ndarray,
    plot_savefolder: str,
    z_plus_minus: int = 1,
    return_list_of_save_path: bool = False,
    ) -> Optional[List[str]]:
    """Save plots of the full region for each set."""
    os.makedirs(plot_savefolder, exist_ok=True)
    list_of_save_path = [] 
    group_name = each_group_df["group"].values[0]
    for each_set_label in each_group_df["nth_set_label"].unique():
        if each_set_label == -1:
            continue

        each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
        corrected_uncaging_z = each_set_df["corrected_uncaging_z"].values[0]
        corrected_uncaging_x = each_set_df["corrected_uncaging_x"].values[0]
        corrected_uncaging_y = each_set_df["corrected_uncaging_y"].values[0]
        
        z_from = min(Aligned_4d_array.shape[1]-1,
                     int(max(0, corrected_uncaging_z - z_plus_minus)))
        z_to = max(z_from+1,
                   int(min(Aligned_4d_array.shape[1], corrected_uncaging_z + z_plus_minus + 1)))

        for _, each_row in each_set_df.iterrows():
            nth_omit_induction = each_row["nth_omit_induction"]
            if nth_omit_induction == -1:
                continue
            try:
                pre_post_phase = each_set_df[each_set_df["nth_omit_induction"] == nth_omit_induction]["phase"].values[0]
            except:
                pre_post_phase = " "
                print("could not read pre_post_phase")

            Zproj = Aligned_4d_array[nth_omit_induction, z_from:z_to, :, :].max(axis=0)
            plt.imshow(Zproj, cmap="gray")
            plt.title(f"group: {group_name}\nset_label: {each_set_label}   {pre_post_phase}\nnth_omit_induction: {nth_omit_induction}")
            plt.scatter(corrected_uncaging_x, corrected_uncaging_y, color="red", s=100)
            save_path = os.path.join(plot_savefolder, f"{group_name}_{each_set_label}_{nth_omit_induction}.png")
            plt.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight"
            )
            plt.close()
            plt.clf()
            list_of_save_path.append(save_path)
    if return_list_of_save_path:
        return list_of_save_path
    else:
        return None

def process_small_region(
    each_set_df: pd.DataFrame,
    Aligned_4d_array: np.ndarray,
    small_region_size: int = 60,
    small_z_plus_minus: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], pd.DataFrame]:
    """Process a small region around the uncaging position."""
    small_region_size_half = int(small_region_size / 2)
    
    min_nth = each_set_df["nth_omit_induction"].values[each_set_df["nth_omit_induction"].values > 0].min()
    max_nth = each_set_df["nth_omit_induction"].values[each_set_df["nth_omit_induction"].values > 0].max()
    
    corrected_uncaging_z = each_set_df["corrected_uncaging_z"].values[0]
    corrected_uncaging_x = each_set_df["corrected_uncaging_x"].values[0]
    corrected_uncaging_y = each_set_df["corrected_uncaging_y"].values[0]
    
    # Calculate region boundaries
    small_z_from = min(Aligned_4d_array.shape[1]-1,
                       int(max(0, corrected_uncaging_z - small_z_plus_minus)))
    small_z_to = max(small_z_from+1,
                     int(min(Aligned_4d_array.shape[1], corrected_uncaging_z + small_z_plus_minus + 1)))
    small_x_from = int(max(0, corrected_uncaging_x - small_region_size_half))
    small_x_to = int(min(Aligned_4d_array.shape[2], corrected_uncaging_x + small_region_size_half + 1))
    small_y_from = int(max(0, corrected_uncaging_y - small_region_size_half))
    small_y_to = int(min(Aligned_4d_array.shape[3], corrected_uncaging_y + small_region_size_half + 1))
    
    # Store region boundaries (avoid SettingWithCopyWarning)
    each_set_df = each_set_df.copy()  # Make explicit copy
    each_set_df.loc[:, "small_z_from"] = small_z_from
    each_set_df.loc[:, "small_z_to"] = small_z_to
    each_set_df.loc[:, "small_x_from"] = small_x_from
    each_set_df.loc[:, "small_x_to"] = small_x_to
    each_set_df.loc[:, "small_y_from"] = small_y_from
    each_set_df.loc[:, "small_y_to"] = small_y_to

    # Extract and align small region
    small_Tiff_MultiArray = Aligned_4d_array[
        min_nth:max_nth + 1,
        small_z_from:small_z_to,
        small_y_from:small_y_to,
        small_x_from:small_x_to
    ]
    small_shifts, small_Aligned_4d_array = Align_4d_array(small_Tiff_MultiArray)
    
    # Store individual shift values for each frame
    valid_df = each_set_df[each_set_df["nth_omit_induction"] != -1].copy()
    valid_df_sorted = valid_df.sort_values("nth_omit_induction")

    for i, (idx, row) in enumerate(valid_df_sorted.iterrows()):
        if i < len(small_shifts):
            shift_z, shift_y, shift_x = small_shifts[i][0], small_shifts[i][1], small_shifts[i][2]
            each_set_df.loc[idx, 'small_shift_z'] = shift_z
            each_set_df.loc[idx, 'small_shift_y'] = shift_y
            each_set_df.loc[idx, 'small_shift_x'] = shift_x

    corrected_positions = {
        "z": corrected_uncaging_z - small_z_from,
        "x": corrected_uncaging_x - small_x_from,
        "y": corrected_uncaging_y - small_y_from,
        "min_nth": min_nth
    }
    
    return small_Tiff_MultiArray, small_Aligned_4d_array, corrected_positions, each_set_df

def save_small_region_plots(
    small_Aligned_4d_array: np.ndarray,
    corrected_positions: Dict[str, int],
    each_set_label: int,
    plot_savefolder: str,
    z_plus_minus: int = 1,
    each_set_df: pd.DataFrame = None,
    return_list_of_save_path: bool = False,
    save_plot_TF: bool = True,
    ) -> Optional[List[str]]:
    """Save plots for the small region.
    
    Args:
        small_Aligned_4d_array: The aligned 4D array containing the small region data
        corrected_positions: Dictionary containing corrected x, y, z positions and min_nth
        each_set_label: Label for the current set
        plot_savefolder: Directory to save the plots
        z_plus_minus: Number of z-slices to include above and below the center
    """
    os.makedirs(plot_savefolder, exist_ok=True)
    
    z_from = min(small_Aligned_4d_array.shape[1]-1,
                 int(max(0, corrected_positions["z"] - z_plus_minus)))
    z_to = max(z_from+1,
               int(min(small_Aligned_4d_array.shape[1], corrected_positions["z"] + z_plus_minus + 1)))
    
    list_of_save_path = []
    group_name = each_set_df["group"].values[0]
    for nth in range(small_Aligned_4d_array.shape[0]):
        savepath_name = f"small_region_{group_name}_{each_set_label}_{nth + corrected_positions['min_nth']}.png"
        save_path = os.path.join(plot_savefolder, savepath_name)
        list_of_save_path.append(save_path)

        if save_plot_TF:
            Zproj = small_Aligned_4d_array[nth, z_from:z_to, :, :].max(axis=0)
            
            try:
                relative_nth = nth + each_set_df[each_set_df["nth_omit_induction"]>0]["nth_omit_induction"].values .min()
                pre_post_phase = each_set_df[each_set_df["nth_omit_induction"] == relative_nth]["phase"].values[0]
            except:
                pre_post_phase = " "
                print("could not read pre_post_phase")

            plt.imshow(Zproj, cmap="gray")
            plt.title(f"each_set_label: {each_set_label}   {pre_post_phase}\nnth_omit_induction: {nth + corrected_positions['min_nth']}")
            plt.scatter(corrected_positions["x"], corrected_positions["y"], color="red", s=100)

            plt.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight"
            )
            plt.close()
            plt.clf()
            
    if return_list_of_save_path:
        return list_of_save_path
    else:
        return None

def save_small_region_tiffs(
    small_Tiff_MultiArray: np.ndarray,
    small_Aligned_4d_array: np.ndarray,
    corrected_positions: Dict[str, int],
    each_group: str,
    each_set_label: int,
    tif_savefolder: str,
    z_plus_minus: int = 1,
    return_save_path: bool = False,
    save_tif_TF: bool = True,
    ) -> Optional[Dict[str, str]]:
    """Save TIFF files for the small region before and after alignment.
    
    Args:
        small_Tiff_MultiArray: The original 4D array containing the small region data
        small_Aligned_4d_array: The aligned 4D array containing the small region data
        corrected_positions: Dictionary containing corrected x, y, z positions and min_nth
        each_group: Group identifier
        each_set_label: Label for the current set
        tif_savefolder: Directory to save the TIFF files
        z_plus_minus: Number of z-slices to include above and below the center
    """
    os.makedirs(tif_savefolder, exist_ok=True)

    save_path_before_align = os.path.join(tif_savefolder, f"{each_group}_{each_set_label}_before_align.tif")
    save_path_after_align = os.path.join(tif_savefolder, f"{each_group}_{each_set_label}_after_align.tif")

    if save_tif_TF:
        
        z_from = min(small_Aligned_4d_array.shape[1]-1,
                    int(max(0, corrected_positions["z"] - z_plus_minus)))
        z_to = max(z_from+1,
                int(min(small_Aligned_4d_array.shape[1], corrected_positions["z"] + z_plus_minus + 1)))
        
        zproj_before_align = small_Tiff_MultiArray[:, z_from:z_to, :, :].max(axis=1)
        zproj_after_align = small_Aligned_4d_array[:, z_from:z_to, :, :].max(axis=1)
        
        tifffile.imwrite(
            save_path_before_align,
            zproj_before_align
        )
        tifffile.imwrite(
            save_path_after_align,
            zproj_after_align
        )
    if return_save_path:
        return {"save_path_before_align": save_path_before_align, 
                "save_path_after_align": save_path_after_align}
    else:
        return None


def temp_add_align_info(
    one_of_filepath,
    plot_savefolder,
    tif_savefolder,
    z_plus_minus,
    save_plot_TF = True,
    save_tif_TF = True,    
    ):

    os.makedirs(plot_savefolder, exist_ok=True)
    os.makedirs(tif_savefolder, exist_ok=True)
    # Load initial data
    
    one_of_file_list = glob.glob(
        os.path.join(
            os.path.dirname(one_of_filepath), 
            "*_highmag_*002.flim"
            )
        )

    ch_1or2 = 2
    combined_df = get_uncaging_pos_multiple(one_of_file_list, pre_length=2)
    
    # Process each group
    for each_group in combined_df['group'].unique():
        each_group_df = combined_df[combined_df['group'] == each_group]
        filelist = each_group_df["file_path"].tolist()
        
        # Load and align data
        Aligned_4d_array, shifts, _ = load_and_align_data(filelist, ch=ch_1or2 - 1)
        
        # Process uncaging positions
        each_group_df = process_uncaging_positions(each_group_df, shifts, Aligned_4d_array)
        
        # Update combined_df with corrected uncaging positions
        for col in ['corrected_uncaging_x', 'corrected_uncaging_y', 'corrected_uncaging_z']:
            if col in each_group_df.columns:
                combined_df.loc[each_group_df.index, col] = each_group_df[col].values
        
        # Store individual shift values for each frame
        valid_df = each_group_df[each_group_df["nth_omit_induction"] != -1].copy()
        valid_df_sorted = valid_df.sort_values("nth_omit_induction")
        
        for i, (idx, row) in enumerate(valid_df_sorted.iterrows()):
            if i < len(shifts):
                shift_z, shift_y, shift_x = shifts[i][0], shifts[i][1], shifts[i][2]
                combined_df.loc[idx, 'shift_z'] = shift_z
                combined_df.loc[idx, 'shift_y'] = shift_y
                combined_df.loc[idx, 'shift_x'] = shift_x
        
        # Save full region plots
        list_of_save_path = save_full_region_plots(each_group_df, Aligned_4d_array, plot_savefolder, 
                                                   return_list_of_save_path=True)
        
        getting_length_df = each_group_df[(each_group_df["nth_omit_induction"] != -1) & 
                                     (each_group_df["nth_set_label"] != -1)]
        # print(f"length of getting_length_df: {len(getting_length_df)}")
        # print(f"length of list_of_save_path: {len(list_of_save_path)}")
        combined_df.loc[getting_length_df.index, "save_full_region_plot_path"] = list_of_save_path

        # Process small regions
        for each_set_label in each_group_df["nth_set_label"].unique():
            if each_set_label == -1:
                continue
                
            each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
            
            small_Tiff_MultiArray, small_Aligned_4d_array, corrected_positions = process_small_region(
                each_set_df, Aligned_4d_array
            )

            # Update combined_df with small region boundaries and small shifts
            for col in ['small_z_from', 'small_z_to', 'small_x_from', 'small_x_to', 'small_y_from', 'small_y_to']:
                if col in each_set_df.columns:
                    combined_df.loc[each_set_df.index, col] = each_set_df[col].values

            combined_df.loc[each_set_df.index, "small_shift_z"] = each_set_df["small_shift_z"].values
            combined_df.loc[each_set_df.index, "small_shift_y"] = each_set_df["small_shift_y"].values
            combined_df.loc[each_set_df.index, "small_shift_x"] = each_set_df["small_shift_x"].values

            if save_plot_TF:
                list_of_save_path = save_small_region_plots(
                    small_Aligned_4d_array,
                    corrected_positions,
                    each_set_label,
                    plot_savefolder,
                    z_plus_minus,
                    each_set_df,
                    return_list_of_save_path=True
                )
                no_uncaging_df = each_set_df[each_set_df["nth_omit_induction"] != -1]
                count_up = -1
                for ind, each_row in no_uncaging_df.iterrows():
                    count_up += 1
                    combined_df.loc[ind, "relative_nth_omit_induction"] = count_up

                combined_df.loc[no_uncaging_df.index, "save_small_region_plot_path"] = list_of_save_path
            
            if save_tif_TF:
                savepath_dict = save_small_region_tiffs(
                    small_Tiff_MultiArray,
                    small_Aligned_4d_array,
                    corrected_positions,
                    each_group,
                    each_set_label,
                    tif_savefolder,
                    z_plus_minus,
                    return_save_path=True
                )
                combined_df.loc[each_set_df.index, "before_align_save_path"] = savepath_dict["save_path_before_align"]
                combined_df.loc[each_set_df.index, "after_align_save_path"] = savepath_dict["save_path_after_align"]
            
    for each_group in combined_df['group'].unique():
        each_group_df = combined_df[combined_df['group'] == each_group]
        for each_set_label in each_group_df["nth_set_label"].unique():
            if each_set_label == -1:
                continue
            max_nth_omit_ind_before_unc = each_group_df[each_group_df["phase"] == "pre"]["nth_omit_induction"].max()
            plot_path = each_group_df[each_group_df["nth_omit_induction"] == max_nth_omit_ind_before_unc]["save_full_region_plot_path"].values[0]
            img = plt.imread(plot_path)
            plt.imshow(img)
            plt.axis("off")
            plt.show()
            after_align_tiff_path = each_group_df[each_group_df["nth_omit_induction"] == max_nth_omit_ind_before_unc]["after_align_save_path"].values[0]
            after_align_tiff = tifffile.imread(after_align_tiff_path)
            max_proj = after_align_tiff.max(axis=0)
            plt.imshow(max_proj, cmap="gray")
            plt.show() 
    return combined_df


def launch_roi_analysis_gui(combined_df, tiff_data_path, each_group, each_set_label, header="ROI"):
    """Launch the ROI analysis GUI with the specified data.
    
    Args:
        combined_df: The combined dataframe containing analysis data
        tiff_data_path: Path to the after_align TIFF file
        each_group: Group identifier (could be 'group' or 'filepath_without_number' value)
        each_set_label: Set label identifier
        header: Header string to use for column names and GUI display
    """
    
    # Load the TIFF data
    if os.path.exists(tiff_data_path):
        after_align_tiff_data = tifffile.imread(tiff_data_path)
    else:
        print(f"Error: TIFF file not found at {tiff_data_path}")
        return None
    
    # Create max projection properly handling different data dimensions
    print(f"TIFF data shape: {after_align_tiff_data.shape}")
    
    if len(after_align_tiff_data.shape) == 4:
        # 4D data: (time, z, y, x) - take max over both time and z axes
        max_proj_image = after_align_tiff_data.max(axis=(0, 1))
    elif len(after_align_tiff_data.shape) == 3:
        # 3D data: could be (time, y, x) or (z, y, x) - take max over first axis
        max_proj_image = after_align_tiff_data.max(axis=0)
    else:
        # 2D data: use as is
        max_proj_image = after_align_tiff_data
    
    print(f"Max projection shape: {max_proj_image.shape}")
    
    # Ensure we have a 2D image for display
    if len(max_proj_image.shape) != 2:
        print(f"Warning: Max projection has unexpected shape {max_proj_image.shape}")
        # Take max over all but last two dimensions
        while len(max_proj_image.shape) > 2:
            max_proj_image = max_proj_image.max(axis=0)
        print(f"Adjusted max projection shape: {max_proj_image.shape}")
    
    # Filter dataframe for the specific group and set
    # Try filtering by 'group' first, then fall back to 'filepath_without_number'
    filtered_df = combined_df[
        (combined_df['group'] == each_group) & 
        (combined_df['nth_set_label'] == each_set_label)
    ].copy()
    
    # If no data found with 'group', try with 'filepath_without_number'
    if len(filtered_df) == 0:
        print(f"No data found using 'group'={each_group}. Trying 'filepath_without_number'...")
        if 'filepath_without_number' in combined_df.columns:
            # First find the filepath_without_number that corresponds to this group
            filepath_without_number_candidates = combined_df[
                combined_df['group'] == each_group
            ]['filepath_without_number'].unique()
            
            if len(filepath_without_number_candidates) > 0:
                for filepath_without_number in filepath_without_number_candidates:
                    temp_filtered_df = combined_df[
                        (combined_df['filepath_without_number'] == filepath_without_number) & 
                        (combined_df['nth_set_label'] == each_set_label)
                    ].copy()
                    if len(temp_filtered_df) > 0:
                        filtered_df = temp_filtered_df
                        print(f"Found data using 'filepath_without_number'={filepath_without_number}")
                        break
            
            # If still no data, try direct match with filepath_without_number
            if len(filtered_df) == 0:
                filtered_df = combined_df[
                    (combined_df['filepath_without_number'] == each_group) & 
                    (combined_df['nth_set_label'] == each_set_label)
                ].copy()
                if len(filtered_df) > 0:
                    print(f"Found data using direct 'filepath_without_number' match")
    
    if len(filtered_df) == 0:
        print(f"Error: No data found for group={each_group}, set_label={each_set_label}")
        print(f"Available groups in 'group' column: {combined_df['group'].unique()}")
        if 'filepath_without_number' in combined_df.columns:
            print(f"Available groups in 'filepath_without_number' column: {combined_df['filepath_without_number'].unique()}")
        print(f"Available set_labels: {combined_df['nth_set_label'].unique()}")
        return None
    
    print(f"Successfully filtered data: {len(filtered_df)} rows found")
    
    # Get uncaging position information from the filtered dataframe
    uncaging_info = {}
    if 'corrected_uncaging_x' in filtered_df.columns and 'corrected_uncaging_y' in filtered_df.columns:
        # Get uncaging position from the first row (same for all rows in a set)
        first_row = filtered_df.iloc[0]
        if pd.notna(first_row.get('corrected_uncaging_x')) and pd.notna(first_row.get('corrected_uncaging_y')):
            # These are already corrected to small region coordinates
            small_x_from = first_row.get('small_x_from', 0)
            small_y_from = first_row.get('small_y_from', 0)
            corrected_uncaging_x = first_row.get('corrected_uncaging_x', 0)
            corrected_uncaging_y = first_row.get('corrected_uncaging_y', 0)
            
            # Convert to small region coordinates
            uncaging_info = {
                'x': corrected_uncaging_x - small_x_from,
                'y': corrected_uncaging_y - small_y_from,
                'has_uncaging': True
            }
        else:
            uncaging_info = {'has_uncaging': False}
    else:
        uncaging_info = {'has_uncaging': False}
    
    # Get file information for display
    file_info = {}
    if 'file_path' in filtered_df.columns:
        first_file_path = filtered_df.iloc[0].get('file_path', '')
        if first_file_path:
            file_info['filename'] = os.path.basename(first_file_path)
            file_info['directory'] = os.path.dirname(first_file_path)
        else:
            file_info['filename'] = 'Unknown'
            file_info['directory'] = 'Unknown'
    else:
        file_info['filename'] = 'Unknown'
        file_info['directory'] = 'Unknown'
    
    # Add group and set information
    file_info['group'] = each_group
    file_info['set_label'] = each_set_label
    
    # Check if QApplication already exists
    app = QApplication.instance()
    if app is None:
        # Create new application if none exists
        app = QApplication(sys.argv)
        app_created = True
    else:
        # Use existing application
        app_created = False
        print("Using existing QApplication instance")
    
    # Create GUI window with additional information and header
    window = ROIAnalysisGUI(filtered_df, after_align_tiff_data, max_proj_image, uncaging_info, file_info, header=header)
    
    # Store the filtered dataframe in the GUI instance for verification during saving
    window.filtered_df = filtered_df
    
    # Set window flags to ensure proper cleanup
    from PyQt5.QtCore import Qt
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    
    window.show()
    
    # Run application only if we created it
    if app_created:
        result = app.exec_()
    else:
        # For existing application, use a different approach
        # Set up event loop for the window
        from PyQt5.QtCore import QEventLoop
        loop = QEventLoop()
        
        # Connect window close event to loop quit
        def on_window_close():
            try:
                loop.quit()
            except RuntimeError:
                # Handle case where loop is already finished
                pass
        
        # Override closeEvent to ensure proper cleanup
        original_close_event = window.closeEvent
        def enhanced_close_event(event):
            try:
                original_close_event(event)
                on_window_close()
            except Exception as e:
                print(f"Warning: Error during window close: {e}")
                event.accept()
                on_window_close()
        
        window.closeEvent = enhanced_close_event
        
        # Connect analysis complete to window close
        original_complete = window.complete_analysis
        def complete_and_close():
            try:
                original_complete()
                window.close()
            except Exception as e:
                print(f"Warning: Error during analysis completion: {e}")
                window.close()
        window.complete_analysis = complete_and_close
        
        # Run the event loop with error handling
        try:
            loop.exec_()
            result = 0
        except Exception as e:
            print(f"Warning: Event loop error: {e}")
            result = 1
    
    # Save ROI data if analysis was completed
    if hasattr(window, 'analysis_completed') and window.analysis_completed:
        print("=== ROI DATA SAVING ===")
        print(f"Saving {header} ROI analysis results to combined_df...")
        
        # Print GUI instance data for debugging
        print(f"GUI ROI shape: {window.roi_shape}")
        print(f"GUI ROI parameters: {window.roi_parameters}")
        print(f"Intensity data length: {len(window.intensity_data['mean'])}")
        
        try:
            save_roi_data_to_combined_df(window, combined_df, each_group, each_set_label, header)
            print("ROI data saved successfully!")
            
            # Verify the data was saved correctly
            verify_roi_data_in_combined_df(combined_df, each_group, each_set_label, header)
            
        except Exception as save_error:
            print(f"Error saving ROI data: {save_error}")
            import traceback
            traceback.print_exc()
    else:
        print("Analysis not completed or analysis_completed flag not set")
        if hasattr(window, 'analysis_completed'):
            print(f"analysis_completed = {window.analysis_completed}")
        else:
            print("analysis_completed attribute not found")
    
    # Clean up
    try:
        window.deleteLater()
        
        # Process events to ensure cleanup
        if app:
            app.processEvents()
            
        # Force garbage collection
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")
    
    return result

def integrate_gui_with_existing_analysis():
    """Integration function to add GUI functionality to existing analysis workflow."""
    
    # This function can be called from the existing analysis code
    # to launch the GUI for ROI analysis
    
    print("GUI Integration Module Loaded Successfully!")
    print("Available functions:")
    print("- launch_roi_analysis_gui(combined_df, tiff_data_path, each_group, each_set_label)")
    print("- launch_multi_roi_analysis_gui(combined_df, tiff_data_path, each_group, each_set_label)")
    print("\nExample usage:")
    print("from gui_integration import launch_roi_analysis_gui, launch_multi_roi_analysis_gui")
    print("# Single ROI analysis:")
    print("launch_roi_analysis_gui(combined_df, 'path/to/tiff/file.tif', 'group1', 1)")
    print("# Multi-ROI analysis (Spine, Dendritic Shaft, Background):")
    print("launch_multi_roi_analysis_gui(combined_df, 'path/to/tiff/file.tif', 'group1', 1)")

def save_roi_data_to_combined_df(gui_instance, combined_df, each_group, each_set_label, header="ROI"):
    """Save ROI analysis results back to the combined dataframe.
    
    Args:
        gui_instance: Instance of ROIAnalysisGUI
        combined_df: The combined dataframe to update
        each_group: Group identifier (could be 'group' or 'filepath_without_number' value)
        each_set_label: Set label identifier
        header: Header string to use for column names and GUI display
    """
    
    print(f"Saving {header} ROI data for group {each_group}, set {each_set_label}")
    
    # Get the exact same filtered dataframe that was used in the GUI
    # This ensures we only update the rows that were actually analyzed
    
    # Try filtering by 'group' first, then fall back to 'filepath_without_number'
    base_mask = (combined_df['group'] == each_group) & (combined_df['nth_set_label'] == each_set_label)
    filtered_df = combined_df[base_mask]
    
    # If no data found with 'group', try with 'filepath_without_number'
    if len(filtered_df) == 0:
        print(f"No data found using 'group'={each_group}. Trying 'filepath_without_number'...")
        if 'filepath_without_number' in combined_df.columns:
            # First find the filepath_without_number that corresponds to this group
            filepath_without_number_candidates = combined_df[
                combined_df['group'] == each_group
            ]['filepath_without_number'].unique()
            
            if len(filepath_without_number_candidates) > 0:
                for filepath_without_number in filepath_without_number_candidates:
                    temp_base_mask = (combined_df['filepath_without_number'] == filepath_without_number) & (combined_df['nth_set_label'] == each_set_label)
                    temp_filtered_df = combined_df[temp_base_mask]
                    if len(temp_filtered_df) > 0:
                        base_mask = temp_base_mask
                        filtered_df = temp_filtered_df
                        print(f"Found data using 'filepath_without_number'={filepath_without_number}")
                        break
            
            # If still no data, try direct match with filepath_without_number
            if len(filtered_df) == 0:
                base_mask = (combined_df['filepath_without_number'] == each_group) & (combined_df['nth_set_label'] == each_set_label)
                filtered_df = combined_df[base_mask]
                if len(filtered_df) > 0:
                    print(f"Found data using direct 'filepath_without_number' match")
    
    # Filter to only include rows that were actually analyzed (nth_omit_induction >= 0)
    analysis_mask = base_mask & (combined_df['nth_omit_induction'] >= 0)
    analysis_filtered_df = combined_df[analysis_mask]
    
    if len(analysis_filtered_df) == 0:
        print("No analysis data found for this group/set combination")
        print(f"Available groups in 'group' column: {combined_df['group'].unique()}")
        if 'filepath_without_number' in combined_df.columns:
            print(f"Available groups in 'filepath_without_number' column: {combined_df['filepath_without_number'].unique()}")
        print(f"Available set_labels: {combined_df['nth_set_label'].unique()}")
        return
    
    print(f"Found {len(filtered_df)} total rows, {len(analysis_filtered_df)} analysis rows for this group/set")
    
    # IMPORTANT: Filter to only include rows that were actually used in the GUI
    # The GUI only analyzes rows with nth_omit_induction >= 0
    analysis_mask = base_mask & (combined_df['nth_omit_induction'] >= 0)
    analysis_filtered_df = combined_df[analysis_mask]
    
    print(f"Base filter found {len(filtered_df)} total rows")
    print(f"Analysis filter found {len(analysis_filtered_df)} rows with nth_omit_induction >= 0")
    
    # Debug: Print the indices that will be updated
    print(f"Indices that will be updated: {list(analysis_filtered_df.index)}")
    if 'nth_omit_induction' in analysis_filtered_df.columns:
        nth_values = analysis_filtered_df['nth_omit_induction'].tolist()
        print(f"nth_omit_induction values: {nth_values}")
    
    # Verify that we found the exact same data that was used in the GUI
    if len(analysis_filtered_df) == 0:
        print(f"Warning: No matching analysis data found for group {each_group}, set {each_set_label}")
        print(f"Available groups in 'group' column: {combined_df['group'].unique()}")
        if 'filepath_without_number' in combined_df.columns:
            print(f"Available groups in 'filepath_without_number' column: {combined_df['filepath_without_number'].unique()}")
        print(f"Available set_labels: {combined_df['nth_set_label'].unique()}")
        return combined_df
    
    # Verify the filtered data matches what was used in GUI
    gui_df_length = len(gui_instance.filtered_df) if hasattr(gui_instance, 'filtered_df') else 0
    if gui_df_length > 0:
        gui_analysis_length = len(gui_instance.filtered_df[gui_instance.filtered_df['nth_omit_induction'] >= 0])
        if len(analysis_filtered_df) != gui_analysis_length:
            print(f"Warning: Analysis data length mismatch. GUI used {gui_analysis_length} rows, but found {len(analysis_filtered_df)} rows for saving.")
    
    print(f"Found {len(analysis_filtered_df)} analysis rows to update")
    
    # Get only the indices from the analysis-filtered data
    indices = analysis_filtered_df.index
    
    # Column names with header prefix
    roi_shape_col = f"{header}_roi_shape"
    roi_parameters_col = f"{header}_roi_parameters"
    intensity_mean_col = f"{header}_intensity_mean"
    intensity_max_col = f"{header}_intensity_max"
    intensity_sum_col = f"{header}_intensity_sum"
    roi_mask_col = f"{header}_roi_mask"
    frame_roi_parameters_col = f"{header}_frame_roi_parameters"
    roi_area_pixels_col = f"{header}_roi_area_pixels"
    roi_analysis_timestamp_col = f"{header}_roi_analysis_timestamp"
    quantified_datetime_col = f"{header}_quantified_datetime"
    
    # Save basic ROI parameters for all rows in this group/set
    combined_df.loc[indices, roi_shape_col] = gui_instance.roi_shape
    combined_df.loc[indices, roi_parameters_col] = str(gui_instance.roi_parameters)
    
    # Get the image shape from the GUI data
    frame_data = gui_instance.after_align_tiff_data[0]
    if len(frame_data.shape) == 3:
        image_shape = frame_data.max(axis=0).shape  # (y, x)
    else:
        image_shape = frame_data.shape[-2:]  # Get last 2 dimensions
    
    print(f"Image shape for ROI masks: {image_shape}")
    
    # Ensure roi_mask column exists and is object type for storing numpy arrays
    if roi_mask_col not in combined_df.columns:
        combined_df[roi_mask_col] = None
    # Ensure the column is object type and can store numpy arrays
    combined_df[roi_mask_col] = combined_df[roi_mask_col].astype(object)
    
    # Ensure other new columns exist
    new_columns = [frame_roi_parameters_col, roi_area_pixels_col]
    for col in new_columns:
        if col not in combined_df.columns:
            combined_df[col] = None
    
    # Save frame-specific ROI data - use the same mask we used for filtering
    set_df_sorted = analysis_filtered_df.sort_values('nth_omit_induction')
    
    # Check if GUI instance has frame-specific ROI parameters
    has_frame_specific_params = hasattr(gui_instance, 'frame_roi_parameters') and gui_instance.frame_roi_parameters
    
    print(f"Frame-specific ROI mode: {has_frame_specific_params}")
    if has_frame_specific_params:
        print(f"Available frame ROI parameters: {list(gui_instance.frame_roi_parameters.keys())}")
    
    # Verify we have the expected number of frames
    expected_frames = len(gui_instance.intensity_data['mean'])
    actual_frames = len(set_df_sorted)
    if expected_frames != actual_frames:
        print(f"Warning: Frame count mismatch. GUI has {expected_frames} frames, but filtered data has {actual_frames} frames.")
    
    for frame_idx, (df_idx, row) in enumerate(set_df_sorted.iterrows()):
        if frame_idx < len(gui_instance.intensity_data['mean']):
            # Save intensity data
            combined_df.loc[df_idx, intensity_mean_col] = gui_instance.intensity_data['mean'][frame_idx]
            combined_df.loc[df_idx, intensity_max_col] = gui_instance.intensity_data['max'][frame_idx]
            combined_df.loc[df_idx, intensity_sum_col] = gui_instance.intensity_data['sum'][frame_idx]
            
            # Get ROI parameters for this frame
            if has_frame_specific_params and frame_idx in gui_instance.frame_roi_parameters:
                # Use frame-specific parameters if available
                frame_roi_params = gui_instance.frame_roi_parameters[frame_idx]
                print(f"Using frame-specific ROI parameters for frame {frame_idx}: {frame_roi_params}")
            else:
                # Use global parameters for all frames (fallback)
                frame_roi_params = gui_instance.roi_parameters
                print(f"Using global ROI parameters for frame {frame_idx} (fallback): {frame_roi_params}")
            
            # Create ROI mask for this frame
            roi_mask = create_roi_mask_from_params(
                frame_roi_params, gui_instance.roi_shape, image_shape
            )
            
            # Ensure roi_mask is a single numpy array (not a list or nested structure)
            if not isinstance(roi_mask, np.ndarray):
                print(f"Warning: roi_mask is not a numpy array: {type(roi_mask)}")
                roi_mask = np.array(roi_mask, dtype=bool)
            
            # Save ROI mask using integer-based indexing to avoid pandas array interpretation issues
            try:
                # Get the integer position of this row and column
                row_pos = combined_df.index.get_loc(df_idx)
                col_pos = combined_df.columns.get_loc(roi_mask_col)
                # Use iat for direct integer-based assignment
                combined_df.iat[row_pos, col_pos] = roi_mask
                print(f"Successfully saved ROI mask for frame {frame_idx} using iat method")
            except (ValueError, KeyError) as e:
                print(f"Warning: Error with iat method: {e}. Trying alternative approach...")
                try:
                    # Alternative: Create a temporary dataframe and use update
                    temp_series = pd.Series([roi_mask], index=[df_idx], dtype=object)
                    combined_df[roi_mask_col].update(temp_series)
                    print(f"Successfully saved ROI mask for frame {frame_idx} using update method")
                except Exception as e2:
                    print(f"Warning: Error with update method: {e2}. Using direct dictionary assignment...")
                    # Last resort: direct dictionary-style assignment
                    combined_df[roi_mask_col] = combined_df[roi_mask_col].astype(object)
                    values = combined_df[roi_mask_col].values
                    idx_pos = combined_df.index.get_loc(df_idx)
                    values[idx_pos] = roi_mask
            
            combined_df.loc[df_idx, frame_roi_parameters_col] = str(frame_roi_params)
            
            # Calculate and save ROI area
            roi_area = np.sum(roi_mask)
            combined_df.loc[df_idx, roi_area_pixels_col] = roi_area
            
            # Get the relative_nth_omit_induction for better debugging
            rel_nth = combined_df.loc[df_idx, 'relative_nth_omit_induction'] if 'relative_nth_omit_induction' in combined_df.columns else frame_idx
            print(f"Frame {frame_idx} (rel_nth: {rel_nth}): ROI area = {roi_area} pixels, mean intensity = {gui_instance.intensity_data['mean'][frame_idx]:.2f}")
    
    # Add timestamp of ROI analysis
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    combined_df.loc[indices, roi_analysis_timestamp_col] = timestamp
    
    # Add quantified datetime
    quantified_datetime = datetime.datetime.now()
    combined_df.loc[indices, quantified_datetime_col] = quantified_datetime
    
    print(f"ROI analysis timestamp: {timestamp}")
    if has_frame_specific_params:
        print(f"{header} ROI data saved successfully for {len(indices)} rows with frame-specific masks")
    else:
        print(f"{header} ROI data saved successfully for {len(indices)} rows with identical masks (global parameters used)")
    
    return combined_df

def create_roi_mask_from_params(roi_params, roi_shape, image_shape):
    """Create a boolean ROI mask from ROI parameters.
    
    Args:
        roi_params: Dictionary containing ROI parameters
        roi_shape: Shape of ROI ('rectangle', 'ellipse', 'polygon')
        image_shape: Shape of the image (height, width)
    
    Returns:
        numpy.ndarray: Boolean mask where True indicates pixels inside ROI
    """
    import numpy as np
    from matplotlib.path import Path
    
    mask = np.zeros(image_shape, dtype=bool)
    
    if roi_shape == 'rectangle':
        x, y, width, height = (
            int(round(roi_params['x'])),
            int(round(roi_params['y'])),
            int(round(roi_params['width'])),
            int(round(roi_params['height']))
        )
        # Ensure bounds are within image
        x = max(0, min(x, image_shape[1]-1))
        y = max(0, min(y, image_shape[0]-1))
        x_end = max(0, min(x + width, image_shape[1]))
        y_end = max(0, min(y + height, image_shape[0]))
        mask[y:y_end, x:x_end] = True
        
    elif roi_shape == 'ellipse':
        center_x = roi_params['center_x']
        center_y = roi_params['center_y']
        width = roi_params['width']
        height = roi_params['height']
        
        y_coords, x_coords = np.mgrid[0:image_shape[0], 0:image_shape[1]]
        mask = (((x_coords - center_x) / (width/2))**2 + 
               ((y_coords - center_y) / (height/2))**2) <= 1
               
    elif roi_shape == 'polygon':
        points = roi_params['points']
        if len(points) >= 3:
            # Ensure points are in the correct format
            polygon_points = [(float(p[0]), float(p[1])) for p in points]
            
            # Create a path from polygon points
            path = Path(polygon_points)
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:image_shape[0], 0:image_shape[1]]
            
            # Create coordinate pairs for all pixels
            coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
            
            # Test which points are inside the polygon
            mask_flat = path.contains_points(coords)
            
            # Reshape back to image shape
            mask = mask_flat.reshape(image_shape)
               
    return mask

def verify_roi_data_in_combined_df(combined_df, each_group, each_set_label, header="ROI"):
    """Verify that ROI data was saved correctly to the combined dataframe.
    
    Args:
        combined_df: The combined dataframe to check
        each_group: Group identifier (could be 'group' or 'filepath_without_number' value)
        each_set_label: Set label identifier
        header: Header string to use for column names and GUI display
    """
    
    print(f"\n=== VERIFYING {header} ROI DATA FOR GROUP {each_group}, SET {each_set_label} ===")
    
    # Get the filtered dataframe
    # Try filtering by 'group' first, then fall back to 'filepath_without_number'
    base_mask = (combined_df['group'] == each_group) & (combined_df['nth_set_label'] == each_set_label)
    filtered_df = combined_df[base_mask]
    
    # If no data found with 'group', try with 'filepath_without_number'
    if len(filtered_df) == 0:
        print(f"No data found using 'group'={each_group}. Trying 'filepath_without_number'...")
        if 'filepath_without_number' in combined_df.columns:
            # First find the filepath_without_number that corresponds to this group
            filepath_without_number_candidates = combined_df[
                combined_df['group'] == each_group
            ]['filepath_without_number'].unique()
            
            if len(filepath_without_number_candidates) > 0:
                for filepath_without_number in filepath_without_number_candidates:
                    temp_mask = (combined_df['filepath_without_number'] == filepath_without_number) & (combined_df['nth_set_label'] == each_set_label)
                    temp_filtered_df = combined_df[temp_mask]
                    if len(temp_filtered_df) > 0:
                        filtered_df = temp_filtered_df
                        print(f"Found data using 'filepath_without_number'={filepath_without_number}")
                        break
            
            # If still no data, try direct match with filepath_without_number
            if len(filtered_df) == 0:
                mask = (combined_df['filepath_without_number'] == each_group) & (combined_df['nth_set_label'] == each_set_label)
                filtered_df = combined_df[mask]
                if len(filtered_df) > 0:
                    print(f"Found data using direct 'filepath_without_number' match")
    
    if len(filtered_df) == 0:
        print("No data found for this group/set combination")
        print(f"Available groups in 'group' column: {combined_df['group'].unique()}")
        if 'filepath_without_number' in combined_df.columns:
            print(f"Available groups in 'filepath_without_number' column: {combined_df['filepath_without_number'].unique()}")
        print(f"Available set_labels: {combined_df['nth_set_label'].unique()}")
        return
    
    print(f"Found {len(filtered_df)} rows for this group/set")
    
    # Apply analysis filter (only rows with nth_omit_induction >= 0)
    analysis_filtered_df = filtered_df[filtered_df['nth_omit_induction'] >= 0]
    print(f"Found {len(analysis_filtered_df)} analysis rows (nth_omit_induction >= 0)")
    
    # Column names with header prefix
    roi_shape_col = f"{header}_roi_shape"
    roi_parameters_col = f"{header}_roi_parameters"
    intensity_mean_col = f"{header}_intensity_mean"
    intensity_max_col = f"{header}_intensity_max"
    intensity_sum_col = f"{header}_intensity_sum"
    roi_mask_col = f"{header}_roi_mask"
    frame_roi_parameters_col = f"{header}_frame_roi_parameters"
    roi_area_pixels_col = f"{header}_roi_area_pixels"
    roi_analysis_timestamp_col = f"{header}_roi_analysis_timestamp"
    quantified_datetime_col = f"{header}_quantified_datetime"
    
    # Check which columns exist
    roi_columns = [roi_shape_col, roi_parameters_col, intensity_mean_col, intensity_max_col, 
                  intensity_sum_col, roi_mask_col, frame_roi_parameters_col, roi_area_pixels_col,
                  roi_analysis_timestamp_col, quantified_datetime_col]
    
    existing_cols = [col for col in roi_columns if col in combined_df.columns]
    missing_cols = [col for col in roi_columns if col not in combined_df.columns]
    
    print(f"Existing ROI columns: {existing_cols}")
    if missing_cols:
        print(f"Missing ROI columns: {missing_cols}")
    
    # Check mask data specifically - only check analysis rows
    if roi_mask_col in combined_df.columns:
        valid_masks = 0
        mask_identical = True
        first_mask = None
        
        for idx, row in analysis_filtered_df.iterrows():
            mask_data = row[roi_mask_col]
            if mask_data is not None and isinstance(mask_data, np.ndarray):
                valid_masks += 1
                if first_mask is None:
                    first_mask = mask_data
                elif not np.array_equal(first_mask, mask_data):
                    mask_identical = False
                
                # Get frame info
                rel_nth = row.get('relative_nth_omit_induction', 'Unknown')
                roi_area = row.get(roi_area_pixels_col, 'Unknown')
                mean_intensity = row.get(intensity_mean_col, 'Unknown')
                timestamp = row.get(roi_analysis_timestamp_col, 'Unknown')
                
                # Format mean intensity safely
                if isinstance(mean_intensity, (int, float)):
                    mean_str = f"{mean_intensity:.2f}"
                else:
                    mean_str = str(mean_intensity)
                
                print(f"  Frame {rel_nth}: ROI area={roi_area} pixels, mean={mean_str}, timestamp={timestamp}")
        
        print(f"Valid ROI masks: {valid_masks}/{len(analysis_filtered_df)}")
        print(f"Masks identical across frames: {mask_identical}")
        
        if valid_masks > 0 and first_mask is not None:
            print(f"ROI mask shape: {first_mask.shape}")
            print(f"ROI mask area (first frame): {np.sum(first_mask)} pixels")
    
    # Check timestamp data - only check analysis rows
    if roi_analysis_timestamp_col in combined_df.columns:
        timestamps = analysis_filtered_df[roi_analysis_timestamp_col].dropna().unique()
        print(f"Analysis timestamps: {timestamps}")
    
    # Check if frame-specific parameters were used - only check analysis rows
    if frame_roi_parameters_col in combined_df.columns:
        unique_params = analysis_filtered_df[frame_roi_parameters_col].dropna().unique()
        print(f"Unique frame ROI parameters: {len(unique_params)}")
        if len(unique_params) > 1:
            print("Frame-specific ROI parameters detected!")
        else:
            print("Identical ROI parameters across all frames")
    
    print("=== VERIFICATION COMPLETE ===\n")

def create_roi_analysis_workflow(plot_savefolder, tif_savefolder, one_of_filepath, ch_1or2=2):
    """Create a complete ROI analysis workflow with GUI integration.
    
    Args:
        plot_savefolder: Directory for saving plots
        tif_savefolder: Directory for saving TIFF files
        one_of_filepath: Path to one of the FLIM files
        ch_1or2: Channel to analyze (1 or 2)
    
    Returns:
        Function to launch GUI for specific group and set
    """
    
    # Import necessary functions from existing code
    import glob
    from get_annotation_unc_multiple import get_uncaging_pos_multiple
    
    # Setup directories
    os.makedirs(plot_savefolder, exist_ok=True)
    os.makedirs(tif_savefolder, exist_ok=True)
    
    # Load initial data
    one_of_file_list = glob.glob(os.path.join(
        os.path.dirname(one_of_filepath), "*_highmag_*002.flim"
    ))
    combined_df = get_uncaging_pos_multiple(one_of_file_list, pre_length=2)
    
    def launch_gui_for_group_set(each_group, each_set_label):
        """Launch GUI for specific group and set."""
        
        # Get the group dataframe
        each_group_df = combined_df[combined_df['group'] == each_group]
        filelist = each_group_df["file_path"].tolist()
        
        # Load and align data
        Aligned_4d_array, shifts, _ = load_and_align_data(filelist, ch=ch_1or2 - 1)
        
        # Process uncaging positions
        each_group_df = process_uncaging_positions(each_group_df, shifts, Aligned_4d_array)
        
        # Process small region for the specific set
        each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
        if len(each_set_df) == 0:
            print(f"No data found for group {each_group}, set {each_set_label}")
            return
            
        small_Tiff_MultiArray, small_Aligned_4d_array, corrected_positions = process_small_region(
            each_set_df, Aligned_4d_array
        )
        
        print(f"Small aligned array shape: {small_Aligned_4d_array.shape}")
        
        # Save the small aligned TIFF for GUI use
        temp_tiff_path = os.path.join(tif_savefolder, f"temp_gui_{each_group}_{each_set_label}.tif")
        
        try:
            # Ensure data is in the right format for saving
            if len(small_Aligned_4d_array.shape) == 4:
                # 4D data: (time, z, y, x)
                tifffile.imwrite(temp_tiff_path, small_Aligned_4d_array)
            elif len(small_Aligned_4d_array.shape) == 3:
                # 3D data: might need to add a dimension
                tifffile.imwrite(temp_tiff_path, small_Aligned_4d_array)
            else:
                print(f"Warning: Unexpected data shape {small_Aligned_4d_array.shape}")
                tifffile.imwrite(temp_tiff_path, small_Aligned_4d_array)
            
            # Launch GUI with improved error handling
            try:
                result = launch_roi_analysis_gui(combined_df, temp_tiff_path, each_group, each_set_label)
                print(f"GUI session completed for group {each_group}, set {each_set_label}")
            except Exception as gui_error:
                print(f"GUI error for group {each_group}, set {each_set_label}: {gui_error}")
                result = None
            
        except Exception as e:
            print(f"Error processing data for GUI: {e}")
            result = None
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_tiff_path):
                try:
                    os.remove(temp_tiff_path)
                except:
                    print(f"Warning: Could not remove temporary file {temp_tiff_path}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
        return result
    
    return launch_gui_for_group_set, combined_df

def safe_multiple_launch(combined_df, groups_and_sets):
    """Safely launch multiple GUI sessions.
    
    Args:
        combined_df: The combined dataframe containing analysis data
        groups_and_sets: List of tuples (group, set_label) to process
        
    Returns:
        List of results from each GUI session
    """
    
    results = []
    
    # Check if QApplication exists globally
    app = QApplication.instance()
    if app is None:
        print("Creating global QApplication for multiple GUI sessions")
        app = QApplication(sys.argv)
        global_app_created = True
    else:
        global_app_created = False
        print("Using existing global QApplication")
    
    try:
        for i, (each_group, each_set_label) in enumerate(groups_and_sets):
            print(f"\n=== Processing session {i+1}/{len(groups_and_sets)}: Group {each_group}, Set {each_set_label} ===")
            
            # Create the launch function for this specific case
            from LTPanalysis_auto_test_20250527 import (
                load_and_align_data, process_uncaging_positions, process_small_region
            )
            
            try:
                # Get the group dataframe
                each_group_df = combined_df[combined_df['group'] == each_group]
                filelist = each_group_df["file_path"].tolist()
                
                # Load and align data
                Aligned_4d_array, shifts, _ = load_and_align_data(filelist, ch=1)  # ch=1 for ch_1or2=2
                
                # Process uncaging positions
                each_group_df = process_uncaging_positions(each_group_df, shifts, Aligned_4d_array)
                
                # Process small region for the specific set
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
                if len(each_set_df) == 0:
                    print(f"No data found for group {each_group}, set {each_set_label}")
                    results.append(None)
                    continue
                    
                small_Tiff_MultiArray, small_Aligned_4d_array, corrected_positions = process_small_region(
                    each_set_df, Aligned_4d_array
                )
                
                # Save temporary TIFF
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                    temp_tiff_path = tmp_file.name
                
                tifffile.imwrite(temp_tiff_path, small_Aligned_4d_array)
                
                # Launch GUI (this will use the existing QApplication)
                result = launch_roi_analysis_gui(combined_df, temp_tiff_path, each_group, each_set_label)
                results.append(result)
                
                # Clean up
                os.remove(temp_tiff_path)
                
                # Force cleanup between sessions
                import gc
                gc.collect()
                
                # Process any pending Qt events
                if app:
                    app.processEvents()
                
            except Exception as e:
                print(f"Error in session {i+1}: {e}")
                results.append(None)
                
                # Still try to clean up temp file
                try:
                    if 'temp_tiff_path' in locals() and os.path.exists(temp_tiff_path):
                        os.remove(temp_tiff_path)
                except:
                    pass
                
    finally:
        # Only quit the app if we created it
        if global_app_created:
            app.quit()
            
    return results

def load_roi_mask_from_combined_df(combined_df, df_index):
    """Load ROI mask from the combined dataframe.
    
    Args:
        combined_df: The combined dataframe
        df_index: Index of the row to get ROI from
    
    Returns:
        numpy.ndarray: Boolean mask where True indicates pixels inside ROI
    """
    roi_mask = combined_df.at[df_index, 'roi_mask']
    if roi_mask is None or pd.isna(roi_mask):
        print("Error: No ROI mask found for this frame")
        return None
    
    return roi_mask

def get_roi_pixels_from_combined_df(combined_df, df_index, image_data):
    """Extract ROI pixel values from image data using saved ROI mask.
    
    Args:
        combined_df: The combined dataframe
        df_index: Index of the row to get ROI from
        image_data: 2D numpy array of image data
    
    Returns:
        numpy.ndarray: 1D array of pixel values inside the ROI
    """
    
    roi_mask = load_roi_mask_from_combined_df(combined_df, df_index)
    if roi_mask is None:
        return None
    
    # Extract ROI pixels using boolean indexing
    roi_pixels = image_data[roi_mask]
    return roi_pixels

def create_roi_overlay_image(combined_df, df_index, image_data, overlay_color='red'):
    """Create an overlay image showing the ROI on top of the original image.
    
    Args:
        combined_df: The combined dataframe
        df_index: Index of the row to get ROI from
        image_data: 2D numpy array of image data
        overlay_color: Color for ROI overlay ('red', 'green', 'blue', 'cyan', 'magenta', 'yellow')
    
    Returns:
        numpy.ndarray: 3D RGB image with ROI overlay
    """
    import numpy as np
    
    roi_mask = load_roi_mask_from_combined_df(combined_df, df_index)
    if roi_mask is None:
        return None
    
    # Normalize image to 0-1 range
    img_normalized = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    
    # Create RGB image
    rgb_image = np.stack([img_normalized] * 3, axis=-1)
    
    # Define colors
    colors = {
        'red': [1, 0, 0],
        'green': [0, 1, 0],
        'blue': [0, 0, 1],
        'cyan': [0, 1, 1],
        'magenta': [1, 0, 1],
        'yellow': [1, 1, 0]
    }
    
    color_rgb = colors.get(overlay_color, [1, 0, 0])  # Default to red
    
    # Apply overlay where ROI is True
    for i in range(3):
        rgb_image[roi_mask, i] = img_normalized[roi_mask] * 0.7 + color_rgb[i] * 0.3
    
    return rgb_image

def example_roi_usage():
    """Example of how to use the saved ROI data."""
    
    print("""
=== SUPER SIMPLE ROI Usage ===

1. DIRECT ACCESS (Recommended!):
    roi_mask = combined_df.loc[df_index, 'roi_mask']  # Just get it directly!
    roi_pixels = image_data[roi_mask]  # The key feature!

2. Helper function (same result):
    roi_mask = load_roi_mask_from_combined_df(combined_df, df_index)
    roi_pixels = image_data[roi_mask]

3. Batch processing (super simple):
    for idx, row in combined_df.iterrows():
        if pd.notna(row['roi_mask']):
            roi_mask = row['roi_mask']
            roi_pixels = image_data[roi_mask]
            # Your analysis here...

4. Get other saved data:
    roi_area = combined_df.loc[df_index, 'roi_area_pixels']
    roi_shape = combined_df.loc[df_index, 'roi_shape']
    intensity_mean = combined_df.loc[df_index, 'intensity_mean']

Key Point: ROI masks are now stored as numpy arrays directly in the DataFrame!
No more complex encoding/decoding - just direct access!
    """)

#%%
if __name__ == "__main__":
    # integrate_gui_with_existing_analysis()
    # example_roi_usage() 

    one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250626\lowmag5__highmag_1_063.flim"
    z_plus_minus = 1
    ch_1or2 = 2
    pre_length = 2
    save_plot_TF = True
    save_tif_TF = True
    ignore_words = ["for_align"]

    one_of_file_list = glob.glob(
    os.path.join(
        os.path.dirname(one_of_filepath), 
        "lowmag5*_highmag_*002.flim"
        )
    )
    one_of_file_list = [each_file for each_file in one_of_file_list if not any(ignore_word in each_file for ignore_word in ignore_words)]

    plot_savefolder = os.path.join(os.path.dirname(one_of_filepath), "plot")
    tif_savefolder = os.path.join(os.path.dirname(one_of_filepath), "tif")
    roi_savefolder = os.path.join(os.path.dirname(one_of_filepath), "roi")
    for each_folder in [plot_savefolder, tif_savefolder, roi_savefolder]:
        os.makedirs(each_folder, exist_ok=True)
    
    combined_df = get_uncaging_pos_multiple(one_of_file_list, pre_length=pre_length)
    
    assert False


    # Process each group
    for each_filepath_without_number in combined_df['filepath_without_number'].unique():
        each_filegroup_df = combined_df[combined_df['filepath_without_number'] == each_filepath_without_number]
        for each_group in each_filegroup_df['group'].unique():
            each_group_df = each_filegroup_df[each_filegroup_df['group'] == each_group]
            if not each_group_df["phase"].isin(["unc"]).any():
                print(f"No uncaging data found for group {each_group}")
                continue

            filelist = each_group_df["file_path"].tolist()
            # Load and align data
            Aligned_4d_array, shifts, _ = load_and_align_data(filelist, ch=ch_1or2 - 1)
            
            # Process uncaging positions
            print(each_group_df["phase"])
            each_group_df = process_uncaging_positions(each_group_df, shifts, Aligned_4d_array)
            
            # Update combined_df with corrected uncaging positions
            for col in ['corrected_uncaging_x', 'corrected_uncaging_y', 'corrected_uncaging_z']:
                if col in each_group_df.columns:
                    combined_df.loc[each_group_df.index, col] = each_group_df[col].values
            
            # Store individual shift values for each frame
            valid_df = each_group_df[each_group_df["nth_omit_induction"] != -1].copy()
            valid_df_sorted = valid_df.sort_values("nth_omit_induction")
            
            for i, (idx, row) in enumerate(valid_df_sorted.iterrows()):
                if i < len(shifts):
                    shift_z, shift_y, shift_x = shifts[i][0], shifts[i][1], shifts[i][2]
                    combined_df.loc[idx, 'shift_z'] = shift_z
                    combined_df.loc[idx, 'shift_y'] = shift_y
                    combined_df.loc[idx, 'shift_x'] = shift_x
            
            # Save full region plots
            list_of_save_path = save_full_region_plots(each_group_df, Aligned_4d_array, plot_savefolder, 
                                                       z_plus_minus,
                                                       return_list_of_save_path=True)
            
            getting_length_df = each_group_df[(each_group_df["nth_omit_induction"] != -1) & 
                                        (each_group_df["nth_set_label"] != -1)]
            # print(f"length of getting_length_df: {len(getting_length_df)}")
            # print(f"length of list_of_save_path: {len(list_of_save_path)}")
            combined_df.loc[getting_length_df.index, "save_full_region_plot_path"] = list_of_save_path

            # Process small regions
            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                    
                each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
                
                small_Tiff_MultiArray, small_Aligned_4d_array, corrected_positions, each_set_df = process_small_region(
                    each_set_df, Aligned_4d_array
                )

                # Update combined_df with small region boundaries and small shifts
                for col in ['small_z_from', 'small_z_to', 'small_x_from', 'small_x_to', 'small_y_from', 'small_y_to']:
                    if col in each_set_df.columns:
                        combined_df.loc[each_set_df.index, col] = each_set_df[col].values

                combined_df.loc[each_set_df.index, "small_shift_z"] = each_set_df["small_shift_z"].values
                combined_df.loc[each_set_df.index, "small_shift_y"] = each_set_df["small_shift_y"].values
                combined_df.loc[each_set_df.index, "small_shift_x"] = each_set_df["small_shift_x"].values

                
                list_of_save_path = save_small_region_plots(
                    small_Aligned_4d_array,
                    corrected_positions,
                    each_set_label,
                    plot_savefolder,
                    z_plus_minus,
                    each_set_df,
                    return_list_of_save_path=True,
                    save_plot_TF=save_plot_TF
                )
                no_uncaging_df = each_set_df[each_set_df["nth_omit_induction"] != -1]
                count_up = -1
                for ind, each_row in no_uncaging_df.iterrows():
                    count_up += 1
                    combined_df.loc[ind, "relative_nth_omit_induction"] = count_up

                combined_df.loc[no_uncaging_df.index, "save_small_region_plot_path"] = list_of_save_path
            

                savepath_dict = save_small_region_tiffs(
                    small_Tiff_MultiArray,
                    small_Aligned_4d_array,
                    corrected_positions,
                    each_group,
                    each_set_label,
                    tif_savefolder,
                    z_plus_minus,
                    return_save_path=True,
                    save_tif_TF=save_tif_TF
                )
                combined_df.loc[each_set_df.index, "before_align_save_path"] = savepath_dict["save_path_before_align"]
                combined_df.loc[each_set_df.index, "after_align_save_path"] = savepath_dict["save_path_after_align"]

    if save_tif_TF*save_plot_TF:    
        for each_group in combined_df['group'].unique():
            each_group_df = combined_df[combined_df['group'] == each_group]
            for each_set_label in each_group_df["nth_set_label"].unique():
                if each_set_label == -1:
                    continue
                max_nth_omit_ind_before_unc = each_group_df[each_group_df["phase"] == "pre"]["nth_omit_induction"].max()
                plot_path = each_group_df[each_group_df["nth_omit_induction"] == max_nth_omit_ind_before_unc]["save_full_region_plot_path"].values[0]
                img = plt.imread(plot_path)
                plt.imshow(img)
                plt.axis("off")
                plt.show()
                after_align_tiff_path = each_group_df[each_group_df["nth_omit_induction"] == max_nth_omit_ind_before_unc]["after_align_save_path"].values[0]
                after_align_tiff = tifffile.imread(after_align_tiff_path)
                max_proj = after_align_tiff.max(axis=0)
                plt.imshow(max_proj, cmap="gray")
                plt.show() 
# %%

def validate_flim_data_integrity(one_of_filepath, pre_length=2):
    """Validate FLIM data integrity before processing.
    
    Args:
        one_of_filepath: Path to one of the FLIM files
        pre_length: Pre-length parameter for data loading
    
    Returns:
        dict: Validation results and potential issues
    """
    import glob
    from AnalysisForFLIMage.get_annotation_unc_multiple import get_uncaging_pos_multiple
    
    # Find all related files
    one_of_file_list = glob.glob(
        os.path.join(
            os.path.dirname(one_of_filepath), 
            "*_highmag_*002.flim"
            )
        )
    
    print(f"=== FLIM DATA INTEGRITY VALIDATION ===")
    print(f"Base file: {os.path.basename(one_of_filepath)}")
    print(f"Directory: {os.path.dirname(one_of_filepath)}")
    print(f"Total files found: {len(one_of_file_list)}")
    
    if len(one_of_file_list) == 0:
        return {
            'status': 'ERROR',
            'message': 'No FLIM files found',
            'files': []
        }
    
    # Load combined dataframe
    try:
        combined_df = get_uncaging_pos_multiple(one_of_file_list, pre_length=pre_length)
        print(f"Combined dataframe shape: {combined_df.shape}")
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to load combined dataframe: {str(e)}',
            'files': one_of_file_list
        }
    
    # Check for required columns
    required_columns = ['nth', 'phase', 'group', 'nth_set_label', 'nth_omit_induction']
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    
    if missing_columns:
        return {
            'status': 'ERROR',
            'message': f'Missing required columns: {missing_columns}',
            'files': one_of_file_list,
            'available_columns': list(combined_df.columns)
        }
    
    # Check nth values
    nth_issues = []
    if 'nth' in combined_df.columns:
        nth_values = combined_df['nth'].unique()
        negative_nth = combined_df[combined_df['nth'] < 0]
        if len(negative_nth) > 0:
            nth_issues.append(f"Found {len(negative_nth)} rows with negative nth values: {negative_nth['nth'].unique()}")
    
    # Check phase distribution
    phase_issues = []
    if 'phase' in combined_df.columns:
        phase_counts = combined_df['phase'].value_counts()
        unc_data = combined_df[combined_df['phase'] == 'unc']
        if len(unc_data) == 0:
            phase_issues.append("No uncaging data found")
        else:
            print(f"Found {len(unc_data)} uncaging rows")
    
    # Check group and set structure
    structure_issues = []
    if 'group' in combined_df.columns and 'nth_set_label' in combined_df.columns:
        groups = combined_df['group'].unique()
        for group in groups:
            group_df = combined_df[combined_df['group'] == group]
            sets = group_df['nth_set_label'].unique()
            for set_label in sets:
                if set_label == -1:
                    continue
                set_df = group_df[group_df['nth_set_label'] == set_label]
                unc_in_set = set_df[set_df['phase'] == 'unc']
                if len(unc_in_set) == 0:
                    structure_issues.append(f"Group {group}, Set {set_label}: No uncaging data")
                elif len(unc_in_set) > 1:
                    structure_issues.append(f"Group {group}, Set {set_label}: Multiple uncaging rows ({len(unc_in_set)})")
    
    # Compile results
    issues = nth_issues + phase_issues + structure_issues
    
    if issues:
        result = {
            'status': 'WARNING',
            'message': 'Data integrity issues found',
            'files': one_of_file_list,
            'issues': issues,
            'dataframe_shape': combined_df.shape,
            'nth_range': f"{combined_df['nth'].min()} to {combined_df['nth'].max()}" if 'nth' in combined_df.columns else 'N/A'
        }
    else:
        result = {
            'status': 'OK',
            'message': 'Data integrity validation passed',
            'files': one_of_file_list,
            'dataframe_shape': combined_df.shape,
            'nth_range': f"{combined_df['nth'].min()} to {combined_df['nth'].max()}" if 'nth' in combined_df.columns else 'N/A'
        }
    
    print(f"Validation status: {result['status']}")
    print(f"Message: {result['message']}")
    if 'issues' in result:
        print("Issues found:")
        for issue in result['issues']:
            print(f"  - {issue}")
    
    print("=== END VALIDATION ===\n")
    return result

def debug_index_error(filepath, group_name, set_label):
    """Debug specific IndexError for a given file, group, and set.
    
    Args:
        filepath: Path to the FLIM file
        group_name: Group name to debug
        set_label: Set label to debug
    """
    print(f"=== DEBUGGING INDEX ERROR ===")
    print(f"File: {filepath}")
    print(f"Group: {group_name}")
    print(f"Set: {set_label}")
    
    # Load data
    validation_result = validate_flim_data_integrity(filepath)
    if validation_result['status'] == 'ERROR':
        print(f"Validation failed: {validation_result['message']}")
        return
    
    # Load combined dataframe
    import glob
    from AnalysisForFLIMage.get_annotation_unc_multiple import get_uncaging_pos_multiple
    
    one_of_file_list = glob.glob(
        os.path.join(
            os.path.dirname(filepath), 
            "*_highmag_*002.flim"
            )
        )
    combined_df = get_uncaging_pos_multiple(one_of_file_list, pre_length=2)
    
    # Find the specific group and set
    group_df = combined_df[combined_df['group'] == group_name]
    if len(group_df) == 0:
        print(f"Group {group_name} not found")
        print(f"Available groups: {combined_df['group'].unique()}")
        return
    
    set_df = group_df[group_df['nth_set_label'] == set_label]
    if len(set_df) == 0:
        print(f"Set {set_label} not found in group {group_name}")
        print(f"Available sets in group {group_name}: {group_df['nth_set_label'].unique()}")
        return
    
    # Check uncaging data
    unc_data = set_df[set_df['phase'] == 'unc']
    print(f"Uncaging data in set: {len(unc_data)} rows")
    
    if len(unc_data) > 0:
        print("Uncaging row details:")
        for idx, row in unc_data.iterrows():
            print(f"  Row {idx}: nth={row.get('nth', 'N/A')}, center_x={row.get('center_x', 'N/A')}, center_y={row.get('center_y', 'N/A')}")
    
    # Load alignment data
    filelist = group_df["file_path"].tolist()
    try:
        Aligned_4d_array, shifts, _ = load_and_align_data(filelist, ch=1)  # ch=1 for ch_1or2=2
        print(f"Alignment data loaded successfully")
        print(f"  Aligned array shape: {Aligned_4d_array.shape}")
        print(f"  Shifts shape: {shifts.shape}")
        
        # Check if nth values are within bounds
        if len(unc_data) > 0:
            for idx, row in unc_data.iterrows():
                nth_value = row.get('nth', None)
                if nth_value is not None:
                    if nth_value < 0 or nth_value >= len(shifts):
                        print(f"  ERROR: nth value {nth_value} is out of bounds for shifts array (length: {len(shifts)})")
                    else:
                        print(f"  OK: nth value {nth_value} is within bounds")
                        
    except Exception as e:
        print(f"Failed to load alignment data: {e}")
    
    print("=== END DEBUG ===\n")
