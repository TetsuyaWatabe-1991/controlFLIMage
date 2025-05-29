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




def shift_coords_small_to_full_for_each_rois(combined_df, z_plus_minus, 
                roi_types = ["Spine", "DendriticShaft", "Background"],
                image_shape = (128, 128)):

    for each_roi_type in roi_types:
        # Ensure shifted_mask column exists and is object type for storing numpy arrays
        if f'{each_roi_type}_shifted_mask' not in combined_df.columns:
            combined_df[f'{each_roi_type}_shifted_mask'] = None
        combined_df[f'{each_roi_type}_shifted_mask'] = combined_df[f'{each_roi_type}_shifted_mask'].astype(object)

        for each_group in combined_df["group"].unique():
            each_group_df = combined_df[combined_df["group"] == each_group]
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
                    shifted_coords[:, 0] = coords[:, 0] + small_y_from - total_shift_y
                    shifted_coords[:, 1] = coords[:, 1] + small_x_from - total_shift_x
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
        assert len(each_set_unc_row) == 1
        
        # Calculate z shifts
        z_shift_last_pre = shifts[each_set_unc_row.loc[:, "nth"] - 1, 0][0]
        z_shift_last_pre_rounded = round(z_shift_last_pre, 0)
        z_relative_to_last_pre = each_set_unc_row.loc[:, "z_relative_step_nth"].values[0]
        z_nth_relative_to_first = round(z_relative_to_last_pre - z_shift_last_pre_rounded)
        
        # Update positions
        each_group_df.loc[each_set_df.index, "corrected_uncaging_x"] = (
            each_set_unc_row.loc[:, "center_x"] + shifts[each_set_unc_row.loc[:, "nth"], 2]
        ).values[0]
        each_group_df.loc[each_set_df.index, "corrected_uncaging_y"] = (
            each_set_unc_row.loc[:, "center_y"] + shifts[each_set_unc_row.loc[:, "nth"], 1]
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
        
        z_from = int(max(0, corrected_uncaging_z - z_plus_minus))
        z_to = int(min(Aligned_4d_array.shape[1], corrected_uncaging_z + z_plus_minus + 1))

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
    small_z_from = int(max(0, corrected_uncaging_z - small_z_plus_minus))
    small_z_to = int(min(Aligned_4d_array.shape[1], corrected_uncaging_z + small_z_plus_minus + 1))
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
    
    z_from = int(max(0, corrected_positions["z"] - z_plus_minus))
    z_to = int(min(small_Aligned_4d_array.shape[1], corrected_positions["z"] + z_plus_minus + 1))
    
    list_of_save_path = []
    group_name = each_set_df["group"].values[0]
    for nth in range(small_Aligned_4d_array.shape[0]):
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
        savepath_name = f"small_region_{group_name}_{each_set_label}_{nth + corrected_positions['min_nth']}.png"
        save_path = os.path.join(plot_savefolder, savepath_name)
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

def save_small_region_tiffs(
    small_Tiff_MultiArray: np.ndarray,
    small_Aligned_4d_array: np.ndarray,
    corrected_positions: Dict[str, int],
    each_group: str,
    each_set_label: int,
    tif_savefolder: str,
    z_plus_minus: int = 1,
    return_save_path: bool = False,
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
    
    z_from = int(max(0, corrected_positions["z"] - z_plus_minus))
    z_to = int(min(small_Aligned_4d_array.shape[1], corrected_positions["z"] + z_plus_minus + 1))
    
    zproj_before_align = small_Tiff_MultiArray[:, z_from:z_to, :, :].max(axis=1)
    zproj_after_align = small_Aligned_4d_array[:, z_from:z_to, :, :].max(axis=1)
    
    save_path_before_align = os.path.join(tif_savefolder, f"{each_group}_{each_set_label}_before_align.tif")
    save_path_after_align = os.path.join(tif_savefolder, f"{each_group}_{each_set_label}_after_align.tif")
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
        each_group: Group identifier
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
    filtered_df = combined_df[
        (combined_df['group'] == each_group) & 
        (combined_df['nth_set_label'] == each_set_label)
    ].copy()
    
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
        each_group: Group identifier
        each_set_label: Set label identifier
        header: Header string to use for column names and GUI display
    """
    
    print(f"Saving {header} ROI data for group {each_group}, set {each_set_label}")
    
    # Get the filtered dataframe indices
    mask = (combined_df['group'] == each_group) & (combined_df['nth_set_label'] == each_set_label)
    indices = combined_df[mask].index
    
    if len(indices) == 0:
        print(f"Warning: No matching indices found for group {each_group}, set {each_set_label}")
        return combined_df
    
    print(f"Found {len(indices)} rows to update")
    
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
    combined_df[roi_mask_col] = combined_df[roi_mask_col].astype(object)
    
    # Ensure other new columns exist
    new_columns = [frame_roi_parameters_col, roi_area_pixels_col]
    for col in new_columns:
        if col not in combined_df.columns:
            combined_df[col] = None
    
    # Save frame-specific ROI data
    set_df = combined_df[mask].copy()
    set_df_sorted = set_df[set_df['nth_omit_induction'] >= 0].sort_values('nth_omit_induction')
    
    # Check if GUI instance has frame-specific ROI parameters
    has_frame_specific_params = hasattr(gui_instance, 'frame_roi_parameters') and gui_instance.frame_roi_parameters
    
    print(f"Frame-specific ROI mode: {has_frame_specific_params}")
    if has_frame_specific_params:
        print(f"Available frame ROI parameters: {list(gui_instance.frame_roi_parameters.keys())}")
    
    for frame_idx, df_idx in enumerate(set_df_sorted.index):
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
            
            # Save ROI mask directly as numpy array using .at for safety
            combined_df.at[df_idx, roi_mask_col] = roi_mask
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
        each_group: Group identifier
        each_set_label: Set label identifier
        header: Header string to use for column names and GUI display
    """
    
    print(f"\n=== VERIFYING {header} ROI DATA FOR GROUP {each_group}, SET {each_set_label} ===")
    
    # Get the filtered dataframe
    mask = (combined_df['group'] == each_group) & (combined_df['nth_set_label'] == each_set_label)
    filtered_df = combined_df[mask]
    
    if len(filtered_df) == 0:
        print("No data found for this group/set combination")
        return
    
    print(f"Found {len(filtered_df)} rows for this group/set")
    
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
    
    # Check mask data specifically
    if roi_mask_col in combined_df.columns:
        valid_masks = 0
        mask_identical = True
        first_mask = None
        
        for idx, row in filtered_df.iterrows():
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
        
        print(f"Valid ROI masks: {valid_masks}/{len(filtered_df)}")
        print(f"Masks identical across frames: {mask_identical}")
        
        if valid_masks > 0 and first_mask is not None:
            print(f"ROI mask shape: {first_mask.shape}")
            print(f"ROI mask area (first frame): {np.sum(first_mask)} pixels")
    
    # Check timestamp data
    if roi_analysis_timestamp_col in combined_df.columns:
        timestamps = filtered_df[roi_analysis_timestamp_col].dropna().unique()
        print(f"Analysis timestamps: {timestamps}")
    
    # Check if frame-specific parameters were used
    if frame_roi_parameters_col in combined_df.columns:
        unique_params = filtered_df[frame_roi_parameters_col].dropna().unique()
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

if __name__ == "__main__":
    integrate_gui_with_existing_analysis()
    example_roi_usage() 