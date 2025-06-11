# %%

import os
import sys
sys.path.append('..\\')
import glob
import numpy as np
import pandas as pd
import tifffile
from PyQt5.QtWidgets import QApplication
from matplotlib import pyplot as plt
from gui_integration import shift_coords_small_to_full_for_each_rois, first_processing_for_flim_files
from gui_roi_analysis.file_selection_gui import launch_file_selection_gui
from fitting.flim_lifetime_fitting import FLIMLifetimeFitter
from FLIMageFileReader2 import FileReader
from simple_dialog import ask_yes_no_gui, ask_save_path_gui, ask_open_path_gui, ask_save_folder_gui
from plot_functions import draw_boundaries
from send_notification import send_slack_notification

# %%

# Setup parameters    
ch_1or2 = 1

z_plus_minus = 1
pre_length = 2
lifetime_measure_ch_1or2_list = [1,2]
photon_threshold = 15

total_photon_threshold = 1000
sync_rate = 80e6
fixed_tau_bool = False
fixed_tau1 = 2.6
fixed_tau2 = 1.1

one_of_filepath_list = [
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\4lines_2_auto\lowmag1__highmag_1_002.flim",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\2lines_1\lowmag1__highmag_1_002.flim",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250601\2lines3_auto\lowmag1__highmag_1_002.flim",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250601\4lines_3_auto\lowmag1__highmag_1_002.flim",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250602\lowmag1__highmag_1_002.flim",
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250602\4lines_neuron4\lowmag1__highmag_1_002.flim",        
]
# one_of_filepath_list = [
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250527\2ndlowmag\lowmag1__highmag_1_002.flim",
# ]

# データ準備ステップの確認
if ask_yes_no_gui("Do you already have combined_df_1.pkl?"):
    df_save_path_1 = ask_open_path_gui()
    print(f"Loading data from: {df_save_path_1}")
    combined_df = pd.read_pickle(df_save_path_1)
else:
    if ask_yes_no_gui("Do you want to run the data preparation step?"):
        combined_df = pd.DataFrame()
        for one_of_filepath in one_of_filepath_list:
            print(f"\n\n\n Processing ... \n\n{one_of_filepath}\n\n\n")
            temp_df = first_processing_for_flim_files(
                one_of_filepath,
                z_plus_minus,
                ch_1or2,
                pre_length = pre_length,
                save_plot_TF = False,
                save_tif_TF = False,
                )
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

        print("define save path for combined_df")
        df_save_path_1 = ask_save_path_gui()
        combined_df.to_pickle(df_save_path_1)
        send_slack_notification(message = "finished reading files. dataframe was saved.")
    else:
        print("canceled")
        assert False

df_save_path_2 = os.path.join(os.path.dirname(df_save_path_1), "combined_df_2.pkl")

roi_types = ["Spine", "DendriticShaft", "Background"]
color_dict = {"Spine": "red", "DendriticShaft": "blue", "Background": "green"}




# %%
# Launch the new file selection GUI instead of the old loop
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
    print("Created new QApplication instance")
else:
    print("Using existing QApplication instance")

# Define additional columns to show in the GUI (optional)
# These will be displayed between the standard columns and ROI columns
additional_columns_to_show = [
    'dt', 
     "donotexist" # Show timing information
]

file_selection_gui = launch_file_selection_gui(
    combined_df, 
    df_save_path_2, 
    additional_columns=additional_columns_to_show
)
app.exec_()
print("ROI visualization plots completed.")

# %%
# DEBUG: Verify ROI masks are updated after GUI operations
if False:
    print("\n" + "="*60)
    print("DEBUG: Checking if ROI masks were updated after GUI operations")
    print("="*60)

    # Check for ROI mask columns
    roi_mask_columns = [col for col in combined_df.columns if 'roi_mask' in col.lower()]
    print(f"Available ROI mask columns: {roi_mask_columns}")

    if len(roi_mask_columns) > 0:
        # Sample a few rows to check if ROI masks exist
        sample_mask = combined_df[combined_df['nth_omit_induction'] >= 0]
        print(f"Found {len(sample_mask)} analysis rows to check")
        
        for roi_col in roi_mask_columns:
            non_null_masks = sample_mask[roi_col].notna().sum()
            print(f"{roi_col}: {non_null_masks}/{len(sample_mask)} rows have masks")
            
            # Check the first non-null mask
            first_mask_row = sample_mask[sample_mask[roi_col].notna()]
            if len(first_mask_row) > 0:
                first_mask = first_mask_row.iloc[0][roi_col]
                if isinstance(first_mask, np.ndarray):
                    print(f"  First mask shape: {first_mask.shape}, non-zero pixels: {np.sum(first_mask)}")
                else:
                    print(f"  First mask type: {type(first_mask)}")

        # Check for ROI analysis timestamps to see which ones were recently updated
        timestamp_columns = [col for col in combined_df.columns if 'roi_analysis_timestamp' in col.lower()]
        print(f"ROI analysis timestamp columns: {timestamp_columns}")
        
        for ts_col in timestamp_columns:
            recent_analyses = sample_mask[sample_mask[ts_col].notna()]
            if len(recent_analyses) > 0:
                latest_timestamp = recent_analyses[ts_col].max()
                count_recent = len(recent_analyses)
                print(f"{ts_col}: {count_recent} analyses found, latest: {latest_timestamp}")
    else:
        print("No ROI mask columns found. This suggests no ROI analysis has been performed yet.")

    print("="*60)

    combined_df.to_pickle(df_save_path_2)



    df = combined_df.sort_values(by='Spine_quantified_datetime')
    each_analysis_timestamp = df.Spine_roi_analysis_timestamp.unique()[-2]
    each_group_df = df[df["Spine_roi_analysis_timestamp"] == each_analysis_timestamp]

    for ind, row in each_group_df.iterrows():
        print(row.Spine_shifted_mask)
        if row["Spine_roi_mask"] is not None:
            plt.contour(row["Spine_roi_mask"])
            plt.title(row.Spine_quantified_datetime)
            plt.show()





if ask_yes_no_gui("stop here?"):
    assert False


# %%
# save plots showing image and roi masks
print("\n" + "="*40)
print("Creating ROI visualization plots...")

# 小さい画像とROIマスクのプロット保存確認
if ask_yes_no_gui("Do you want to save small image and ROI mask plots?"):
    for each_filepath in combined_df['filepath_without_number'].unique():
        each_group_df = combined_df[combined_df['filepath_without_number'] == each_filepath]
        # Get the display group name for this filepath
        display_group = each_group_df['group'].iloc[0]
        for each_set_label in each_group_df["nth_set_label"].unique():
            if each_set_label == -1:
                continue
            temp_tiff_path = each_group_df[(each_group_df['nth_set_label'] == each_set_label)]["after_align_save_path"].values[0]
            if not os.path.exists(temp_tiff_path):
                print(f"Warning: TIFF file not found: {temp_tiff_path}")
                continue
            print(f"Processing Group: {display_group}, Set: {each_set_label}")
            tiff_array = tifffile.imread(temp_tiff_path)
            each_label_df = each_group_df[(each_group_df['filepath_without_number'] == each_filepath) & 
                                        (each_group_df['nth_set_label'] == each_set_label) &
                                        (each_group_df['nth_omit_induction'] != -1)]
            for nth_omit_induction in each_label_df["relative_nth_omit_induction"].unique():
                each_tiff_array = tiff_array[int(nth_omit_induction),:,:]
                
                # Check if ROI masks exist before trying to access them
                current_df = each_label_df[each_label_df["relative_nth_omit_induction"] == nth_omit_induction]
                if len(current_df) == 0:
                    continue
                    
                # Get ROI masks safely
                spine_masks = current_df["Spine_roi_mask"] if "Spine_roi_mask" in current_df.columns else pd.Series([None])
                dendritic_masks = current_df["DendriticShaft_roi_mask"] if "DendriticShaft_roi_mask" in current_df.columns else pd.Series([None])
                background_masks = current_df["Background_roi_mask"] if "Background_roi_mask" in current_df.columns else pd.Series([None])
                
                each_spine_roi_mask = spine_masks.values[0] if len(spine_masks) > 0 and spine_masks.values[0] is not None else None
                each_dendritic_shaft_roi_mask = dendritic_masks.values[0] if len(dendritic_masks) > 0 and dendritic_masks.values[0] is not None else None
                each_background_roi_mask = background_masks.values[0] if len(background_masks) > 0 and background_masks.values[0] is not None else None
                
                # Check if at least one ROI mask is defined
                any_roi_defined = False
                if each_spine_roi_mask is not None and np.any(each_spine_roi_mask):
                    any_roi_defined = True
                if each_dendritic_shaft_roi_mask is not None and np.any(each_dendritic_shaft_roi_mask):
                    any_roi_defined = True
                if each_background_roi_mask is not None and np.any(each_background_roi_mask):
                    any_roi_defined = True
                
                # Skip plotting if no ROI masks are defined
                if not any_roi_defined:
                    print(f"  Skipping Group: {display_group}, Set: {each_set_label}, Frame: {nth_omit_induction} - No ROI masks defined")
                    continue
                
                print(f"  Creating plot for Group: {display_group}, Set: {each_set_label}, Frame: {nth_omit_induction}")
                
                plt.figure(figsize=(10, 8))
                plt.imshow(each_tiff_array, cmap="gray", 
                           interpolation="none", 
                           extent=(0, each_tiff_array.shape[1], each_tiff_array.shape[0], 0))

                legend_handles = []
                if each_spine_roi_mask is not None and np.any(each_spine_roi_mask):
                    # contour_spine = plt.contour(each_spine_roi_mask, colors="red", levels=[0.99])
                    draw_boundaries(each_spine_roi_mask, "red")
                    legend_handles.append(plt.Line2D([0], [0], color="red", label="Spine ROI"))
                    
                if each_dendritic_shaft_roi_mask is not None and np.any(each_dendritic_shaft_roi_mask):
                    # contour_dendritic_shaft = plt.contour(each_dendritic_shaft_roi_mask, colors="blue", levels=[0.99])
                    draw_boundaries(each_dendritic_shaft_roi_mask, "blue")
                    legend_handles.append(plt.Line2D([0], [0], color="blue", label="Dendritic Shaft ROI"))
                    
                if each_background_roi_mask is not None and np.any(each_background_roi_mask):
                    # contour_background = plt.contour(each_background_roi_mask, colors="green", levels=[0.99])
                    draw_boundaries(each_background_roi_mask, "green")
                    legend_handles.append(plt.Line2D([0], [0], color="green", label="Background ROI"))
                
                if legend_handles:
                    plt.legend(handles=legend_handles)
                plt.title(f"Group: {display_group}, Set: {each_set_label}, Frame: {nth_omit_induction}")

                roi_savefolder = os.path.join(os.path.dirname(current_df["filepath_without_number"].values[0]), "roi")

                os.makedirs(roi_savefolder, exist_ok=True)
                save_path = os.path.join(roi_savefolder, f"{display_group}_{each_set_label}_{nth_omit_induction}.png")
                plt.savefig(save_path, dpi=150,bbox_inches="tight")
                plt.close();plt.clf()
    send_slack_notification(message = "finished saving small image and ROI mask plots.")


# %%
# Shift the ROI masks to the full size image

combined_df = shift_coords_small_to_full_for_each_rois(combined_df, z_plus_minus,
                roi_types = roi_types,  
                image_shape = [128,128])


# %%
# Full size image and ROI mask plots

print("\n" + "="*40)
print("Creating ROI visualization plots...")
if ask_yes_no_gui("Do you want to save fullsize image and ROI mask plots?"):
    if ask_yes_no_gui("Do you want to analyze lifetime on the ROI?"):
        fitter = FLIMLifetimeFitter()
        measure_lifetime = True
    else:
        measure_lifetime = False
    # Check if any ROI masks are defined before processing fullsize images
    roi_columns = [col for col in combined_df.columns if col.endswith('_roi_mask')]
    filelist = combined_df[(combined_df["nth_omit_induction"] != -1) & 
                            (combined_df["nth_set_label"] != -1) &
                            (combined_df["Spine_roi_mask"].notna())
                            ]["file_path"].tolist()

    # Tiff_MultiArray, iminfo, _ = flim_files_to_nparray(
    #     filelist, ch=ch_1or2 - 1, normalize_by_averageNum=False
    # )   
    # combined_df = shift_coords_small_to_full_for_each_rois(combined_df, z_plus_minus,
    #                 roi_types = roi_types,  
    #                 image_shape = Tiff_MultiArray.shape[2:])

    for each_filepath in combined_df["filepath_without_number"].unique():
        each_group_df = combined_df[combined_df["filepath_without_number"] == each_filepath]
        # Get the display group name for this filepath
        display_group = each_group_df['group'].iloc[0]
        for each_set_label in each_group_df["nth_set_label"].unique():
            if each_set_label == -1:
                continue
            each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
            first = True
            for nth_omit_induction in each_set_df["nth_omit_induction"].unique():
                if nth_omit_induction == -1:
                    continue
                current_index = each_set_df[each_set_df["nth_omit_induction"] == nth_omit_induction].index[0]

                file_path = each_set_df.at[current_index, "file_path"]                
                if file_path not in filelist:
                    print(f"File not found: {file_path}")
                    continue
                else:
                    fullsize_save_folder = os.path.join(os.path.dirname(file_path), "fullsize_plots")
                    os.makedirs(fullsize_save_folder, exist_ok=True)

                iminfo = FileReader()
                iminfo.read_imageFile(file_path, True)     
                six_dim = np.array(iminfo.image)
                
                # nth_image = filelist.index(file_path)
                # each_tiff_array = Tiff_MultiArray[nth_image,:,:,:]
                # each_tiff_array = Tiff_MultiArray[0,:,:,:]

                corrected_uncaging_z = each_set_df.at[current_index, "corrected_uncaging_z"]
                # z_from = int(max(0, corrected_uncaging_z - z_plus_minus))
                # z_to = int(min(six_dim.shape[0], corrected_uncaging_z + z_plus_minus + 1))

                shift_z = each_set_df.at[current_index, "shift_z"]

                z_from = min(six_dim.shape[0]-1,
                            int(max(0, corrected_uncaging_z - z_plus_minus - shift_z)))
                z_to = max(z_from+1,
                        int(min(six_dim.shape[0], corrected_uncaging_z + z_plus_minus + 1 - shift_z)))
                
                combined_df.loc[current_index, "z_from"] = z_from
                combined_df.loc[current_index, "z_to"] = z_to
                combined_df.loc[current_index, "len_z"] = z_to - z_from


                roi_defined_list = []
                for each_roi_type in roi_types:
                    shifted_mask = each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"]
                    if shifted_mask is not None:
                        roi_defined_list.append(each_roi_type)

                    for each_ch_1or2 in lifetime_measure_ch_1or2_list:
                        combined_df.loc[current_index, f"{each_roi_type}_ch{each_ch_1or2}_photon_fitting"] = None
                        if measure_lifetime:
                            two_dim_image = six_dim[z_from:z_to, 0, each_ch_1or2 - 1, :, :].sum(axis=0).sum(axis=-1)
                            photon_exclude_mask = two_dim_image <= photon_threshold
                            photon_thresholded_image = six_dim[z_from:z_to, 0, each_ch_1or2 - 1, :,:, :].copy()
                            photon_thresholded_image[:,photon_exclude_mask,:] = 0
                            voxels_of_interest = photon_thresholded_image[:, shifted_mask, :]
                            y_data = np.sum(voxels_of_interest, axis=(0,1))
                            ps_per_unit = (10**12)/sync_rate/len(y_data)
                            x = np.arange(len(y_data))
                            if np.sum(y_data) < total_photon_threshold:
                                print(f"Not enough photons for {each_roi_type}_ch{each_ch_1or2} in {file_path}")
                                continue
                            if fixed_tau_bool == False:
                                result_double_normal = fitter.fit_double_exponential(x, y_data, ps_per_unit, sync_rate)
                                # free_2component_fit_tau = result_double_normal["lifetime"]
                                # print(f"free_2component_fit_tau: {free_2component_fit_tau}")
                                combined_df.loc[current_index, f"{each_roi_type}_ch{each_ch_1or2}_photon_fitting"] = result_double_normal["lifetime"]
                            else:
                                result_fix_both = fitter.fit_double_exponential(x, y_data, ps_per_unit, sync_rate,
                                                fix_tau1=fixed_tau1, fix_tau2=fixed_tau2)
                                fixed_2component_fit_tau = result_fix_both["lifetime"]
                                combined_df.loc[current_index, f"{each_roi_type}_ch{each_ch_1or2}_photon_fitting"] = fixed_2component_fit_tau

                if len(roi_defined_list) == 0:
                    print(f"No ROI masks defined for nth_omit_induction {nth_omit_induction}, skipping...")
                    assert False
                    continue


                each_tiff_array = six_dim[z_from:z_to,0,ch_1or2 - 1,:,:,:].sum(axis=-1)
                z_projection = each_tiff_array.max(axis=0)
                if first:
                    first = False
                    vmax = z_projection.max()
                    vmin = 0

                plt.imshow(z_projection, cmap="gray",interpolation="nearest",
                           vmax=vmax, vmin=vmin)
                plt.title(f"{display_group}_{each_set_label}_{nth_omit_induction}\nZ range: {z_from + 1} - {z_to}")
                
                for each_roi_type in roi_defined_list:
                    # plt.contour(each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"],
                    #                     colors=color_dict[each_roi_type], levels=[0.99])
                    draw_boundaries(each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"],
                                    color=color_dict[each_roi_type], linewidth=1)
                    intensity_per_pixel = z_projection[each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"]].sum()/each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"].sum()
                    combined_df.loc[current_index, f"{each_roi_type}_intensity_per_pixel_from_full_image"] = intensity_per_pixel

                combined_df.loc[current_index, f"Spine_subBG_intensity_per_pixel_from_full_image"] = combined_df.loc[current_index, f"Spine_intensity_per_pixel_from_full_image"] - combined_df.loc[current_index, f"Background_intensity_per_pixel_from_full_image"]
                combined_df.loc[current_index, f"Dendritic_shaft_subBG_intensity_per_pixel_from_full_image"] = combined_df.loc[current_index, f"DendriticShaft_intensity_per_pixel_from_full_image"] - combined_df.loc[current_index, f"Background_intensity_per_pixel_from_full_image"]
                                    
                save_path = os.path.join(fullsize_save_folder, f"{display_group}_{each_set_label}_{nth_omit_induction}.png")
                plt.savefig(save_path, dpi=150,bbox_inches="tight")
                plt.close();plt.clf()
                combined_df.loc[current_index, "full_plot_with_roi_path"] = save_path

    print("All processing completed successfully!")
    send_slack_notification(message = "finished saving fullsize image and ROI mask plots.")

combined_df.to_pickle(df_save_path_2)


# %%
# GCaMP analysis
# roi_types = ["Spine", "DendriticShaft", "Background"]
if ask_yes_no_gui("Do you want to analyze GCaMP?"):
    # GCaMPch_1or2 = 1
    # RFPch_1or2 = 2
    Fluor_ch_1or2_dict = {"GCaMP":1, "tdTomato":2}
    base_frame_index = [0, 1]
    afterbase_frame_index = [3]
    for each_fluor_ch_1or2 in Fluor_ch_1or2_dict.keys():
        for each_roi_type in roi_types:
            combined_df[f"{each_fluor_ch_1or2}_{each_roi_type}_array"] = pd.Series(dtype=object)
            if each_roi_type != "Background":
                combined_df[f"{each_fluor_ch_1or2}_{each_roi_type}_subBG"] = pd.Series(dtype=object)
                combined_df[f"{each_fluor_ch_1or2}_{each_roi_type}_F_F0"] = pd.Series(dtype=object)

    for each_filepath in combined_df["filepath_without_number"].unique():
        each_group_df = combined_df[combined_df["filepath_without_number"] == each_filepath]
        # Get the display group name for this filepath
        display_group = each_group_df['group'].iloc[0]
        for each_set_label in each_group_df["nth_set_label"].unique():
            if each_set_label == -1:
                continue
                
            # Get uncaging phase data
            uncaging_data = each_group_df[(each_group_df["phase"] == "unc") & (each_group_df["nth_set_label"] == each_set_label)]
            if len(uncaging_data) == 0:
                print(f"No uncaging data found for group {display_group}, set {each_set_label}")
                continue
                
            uncaging_path = uncaging_data["file_path"].values[0]
            uncaging_df_index = uncaging_data.index[0]  # Get the actual index
            print(uncaging_df_index,"/",len(combined_df))
            # Read the image file
            iminfo = FileReader()
            iminfo.read_imageFile(uncaging_path, True)
            six_dim = np.array(iminfo.image)

            for each_fluor_ch_1or2 in Fluor_ch_1or2_dict.keys():
                for each_roi_type in roi_types:
                    uncaging_array = six_dim[:, 0, Fluor_ch_1or2_dict[each_fluor_ch_1or2] - 1, :, :, :].sum(axis=-1)
                    shifted_mask = combined_df.loc[uncaging_df_index - 1, f"{each_roi_type}_shifted_mask"]
                    each_roi_array = uncaging_array[:,shifted_mask].sum(axis=-1)
                    combined_df.at[uncaging_df_index, f"{each_fluor_ch_1or2}_{each_roi_type}_array"] = each_roi_array

            for each_fluor_ch_1or2 in Fluor_ch_1or2_dict.keys():
                spine_subBG = combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_Spine_array"]/combined_df.loc[uncaging_df_index-1, "Spine_roi_area_pixels"] - combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_Background_array"]/combined_df.loc[uncaging_df_index-1, "Background_roi_area_pixels"]
                dendritic_shaft_subBG = combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_DendriticShaft_array"]/combined_df.loc[uncaging_df_index-1, "DendriticShaft_roi_area_pixels"] - combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_Background_array"]/combined_df.loc[uncaging_df_index-1, "Background_roi_area_pixels"]
                spine_F_F0 = spine_subBG[afterbase_frame_index].mean()/spine_subBG[base_frame_index].mean()
                dendritic_shaft_F_F0 = dendritic_shaft_subBG[afterbase_frame_index].mean()/dendritic_shaft_subBG[base_frame_index].mean()
                combined_df.at[uncaging_df_index, f"{each_fluor_ch_1or2}_Spine_subBG"] = spine_subBG
                combined_df.at[uncaging_df_index, f"{each_fluor_ch_1or2}_DendriticShaft_subBG"] = dendritic_shaft_subBG
                combined_df.at[uncaging_df_index, f"{each_fluor_ch_1or2}_Spine_F_F0"] = spine_F_F0
                combined_df.at[uncaging_df_index, f"{each_fluor_ch_1or2}_DendriticShaft_F_F0"] = dendritic_shaft_F_F0
    send_slack_notification(message = "finished analyzing GCaMP.")

        
# %%
# calculate norm_intensity, normalized time
combined_df["norm_intensity"] = np.nan
for each_filepath in combined_df['filepath_without_number'].unique():
    each_group_df = combined_df[combined_df['filepath_without_number'] == each_filepath]
    # Get the display group name for this filepath
    
    for each_set_label in each_group_df["nth_set_label"].unique():
        if each_set_label == -1:
            continue
        uncaging_df = each_group_df[(each_group_df["phase"] == "unc") & (each_group_df["nth_set_label"] == each_set_label)]
        combined_df.loc[uncaging_df.index, "label"] = f"{each_filepath}_{each_set_label}"

        each_set_df = each_group_df[(each_group_df["nth_set_label"] == each_set_label) &
                                    (each_group_df["nth_omit_induction"] != -1)]
        # combined_df.loc[each_set_df.index, "subBG_intensity_per_pixel"] = each_set_df["Spine_intensity_sum"]/each_set_df["Spine_roi_area_pixels"] - each_set_df["Background_intensity_sum"]/each_set_df["Background_roi_area_pixels"]
        # combined_df.loc[current_index, f"Spine_subBG_intensity_per_pixel_from_full_image"] = combined_df.loc[current_index, f"Spine_intensity_per_pixel_from_full_image"] - combined_df.loc[current_index, f"Background_intensity_per_pixel_from_full_image"]
        # combined_df.loc[current_index, f"Dendritic_shaft_subBG_intensity_per_pixel_from_full_image"] = combined_df.loc[current_index, f"DendriticShaft_intensity_per_pixel_from_full_image"] - combined_df.loc[current_index, f"Background_intensity_per_pixel_from_full_image"]
                            


        numerator = combined_df.loc[each_set_df.index,"Spine_subBG_intensity_per_pixel_from_full_image"]
        # denominator = combined_df.loc[each_set_df[each_set_df["phase"] == "pre"].index, "subBG_intensity_per_pixel"].mean(skipna=False)
        denominator = float(combined_df.loc[each_set_df[each_set_df["phase"] == "pre"].index, "Spine_subBG_intensity_per_pixel_from_full_image"].mean(skipna=False))

        combined_df.loc[each_set_df.index, "norm_intensity"] = (numerator/denominator).astype(float) - 1.0
        
        #Below will be impremented in other function??
        for NthFrame in each_set_df["nth_omit_induction"].unique()[:-1]:
            Time_N = each_set_df[(each_set_df["nth_omit_induction"]==NthFrame)]["relative_time_sec"]
            Time_Nplus1 = each_set_df[(each_set_df["nth_omit_induction"]==NthFrame+1)]["relative_time_sec"]
            rownum = Time_N.keys()
            delta_sec = float(Time_Nplus1.values[0] - Time_N.values[0])
            combined_df.loc[rownum,"delta_sec"] = delta_sec

datetime_started_experiment = combined_df["dt"].min()
combined_df["time_after_started_experiment"] = combined_df["dt"] - datetime_started_experiment
combined_df["time_after_started_experiment_sec"] = combined_df["time_after_started_experiment"].dt.total_seconds()
combined_df["time_after_started_experiment_min"] = combined_df["time_after_started_experiment_sec"]/60
combined_df["min_from_start"] = combined_df["time_after_started_experiment_sec"]//60
combined_df.to_pickle(df_save_path_2)


combined_df_reject_bad_data = combined_df[(combined_df["reject"] == False)]
time_threshold = 5
bin_percent_median = 0.99
bin_sec = combined_df_reject_bad_data[combined_df_reject_bad_data['delta_sec']>time_threshold]['delta_sec'].median()*bin_percent_median
bin_sec = 60*bin_sec//60
combined_df_reject_bad_data["binned_sec"] = bin_sec*(combined_df_reject_bad_data["relative_time_sec"]//bin_sec)
combined_df_reject_bad_data["binned_min"] = combined_df_reject_bad_data["binned_sec"]/60
combined_df_reject_bad_data = combined_df_reject_bad_data[combined_df_reject_bad_data["binned_min"]!=0]
# combined_df_reject_bad_data.dropna(subset=['label'])
combined_df_reject_bad_data = combined_df_reject_bad_data[~combined_df_reject_bad_data["label"].isna()]

savepath_for_combined_df_reject_bad_data = os.path.join(os.path.dirname(df_save_path_2), "combined_df_reject_bad_data.pkl")
combined_df_reject_bad_data.to_pickle(savepath_for_combined_df_reject_bad_data)


# %%
#plot intensity time
default_save_plot_folder = os.path.dirname(df_save_path_1)
save_plot_folder = ask_save_folder_gui(default_folder=default_save_plot_folder)
os.makedirs(save_plot_folder, exist_ok=True)
# combined_df["intensity"] = combined_df["Spine_intensity"] + combined_df["DendriticShaft_intensity"] + combined_df["Background_intensity"]
# clean_df = df.dropna()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
sns.lineplot(x='relative_time_min', y="norm_intensity", 
            data=combined_df_reject_bad_data, hue="label",legend=False)
plt.xlim([-10,30])
plt.ylim([-0.5,2.5])
plt.ylabel("Normalized $\Delta$volume (a.u.)")
plt.xlabel("Time (min)")
plt.savefig(os.path.join(save_plot_folder, "intensity_time.png"), dpi=150,bbox_inches="tight")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()

plt.figure(figsize=(4, 3))
sns.lineplot(x='binned_min', y="norm_intensity", 
            data=combined_df_reject_bad_data, hue="label",legend=False)
sns.lineplot(x='binned_min', y="norm_intensity", 
            data=combined_df_reject_bad_data, 
            errorbar="se", legend=True,
            linewidth=5, color="red")
plt.xlim([-10,30])
plt.ylim([-0.5,2.5])
plt.ylabel("Normalized $\Delta$volume (a.u.)")
plt.xlabel("Time (min)")
plt.savefig(os.path.join(save_plot_folder, "intensity_time_binned.png"), dpi=150,bbox_inches="tight")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()



# %%
# Extract norm_intensity values when relative_time_min first exceeds 25 minutes
bool_analyze_GCaMP = ask_yes_no_gui("do you analyze GCaMP?")


threshold_min = 25
results = []
for label in combined_df_reject_bad_data['label'].dropna().unique():
    label_df = combined_df_reject_bad_data[combined_df_reject_bad_data['label'] == label]
    above_threshold = label_df[label_df['relative_time_min'] > threshold_min]
    uncaging_df = combined_df[(combined_df['phase'] == "unc") & (combined_df['label'] == label)]
    if len(above_threshold) > 0:
        min_row = above_threshold.loc[above_threshold['relative_time_min'].idxmin()]
        if bool_analyze_GCaMP:
            results.append({
                'label': label,
                'time_min': min_row['relative_time_min'],
                'norm_intensity': min_row['norm_intensity'],
                'time_after_started_experiment_min': min_row['time_after_started_experiment_min'],
                'GCaMP_Spine_F_F0': uncaging_df['GCaMP_Spine_F_F0'],
                'GCaMP_DendriticShaft_F_F0': uncaging_df['GCaMP_DendriticShaft_F_F0']
            })
        else:
            results.append({
                'label': label,
                'time_min': min_row['relative_time_min'],
                'norm_intensity': min_row['norm_intensity'],
                'time_after_started_experiment_min': min_row['time_after_started_experiment_min'],
            })

LTP_point_df = pd.DataFrame(results)
LTP_point_df.to_pickle(os.path.join(os.path.dirname(df_save_path_1), "LTP_point_df.pkl"))

reject_threshold_too_large = 3
LTP_point_df_cut_too_large = LTP_point_df[LTP_point_df["norm_intensity"] < reject_threshold_too_large]



if bool_analyze_GCaMP:
    LTP_point_df["GCaMP_Spine_F_F0"] = LTP_point_df["GCaMP_Spine_F_F0"].astype(float)
    LTP_point_df["GCaMP_DendriticShaft_F_F0"] = LTP_point_df["GCaMP_DendriticShaft_F_F0"].astype(float) 

    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='GCaMP_Spine_F_F0', y='norm_intensity', data=LTP_point_df)
    plt.xlim([0,15])
    plt.ylim([-0.5,4.5])
    plt.ylabel("Normalized $\Delta$volume (a.u.)")
    plt.xlabel("GCaMP Spine F/F0")
    plt.savefig(os.path.join(save_plot_folder, "deltavolume_against_GCaMP_Spine_F_F0.png"), dpi=150,bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(4, 3))
    sns.scatterplot(x='GCaMP_DendriticShaft_F_F0', y='norm_intensity', data=LTP_point_df)
    plt.xlim([0,15])
    plt.ylim([-0.5,4.5])
    plt.ylabel("Normalized $\Delta$volume (a.u.)")
    plt.xlabel("GCaMP Dendritic Shaft F/F0")
    plt.savefig(os.path.join(save_plot_folder, "deltavolume_against_GCaMP_DendriticShaft_F_F0.png"), dpi=150,bbox_inches="tight")
    plt.show()


# %%
# plot norm_intensity distribution
print("before cut")
print("mean: ",LTP_point_df["norm_intensity"].mean())
print("std: ",LTP_point_df["norm_intensity"].std())
print("number of data: ",len(LTP_point_df))

print("after cut")
print("mean: ",LTP_point_df_cut_too_large["norm_intensity"].mean())
print("std: ",LTP_point_df_cut_too_large["norm_intensity"].std())
print("number of data: ",len(LTP_point_df_cut_too_large), " / ", len(LTP_point_df))

plt.figure(figsize=(2, 3))
sns.swarmplot(y="norm_intensity", 
            data=LTP_point_df, legend=False)
#plot mean line and std line
plt.axhline(LTP_point_df["norm_intensity"].mean(), color="red", linestyle="-",
            xmin=0.3, xmax=0.7)
plt.axhline(LTP_point_df["norm_intensity"].mean() + LTP_point_df["norm_intensity"].std(), color="red", linestyle=":",
            xmin=0.4, xmax=0.6)
plt.axhline(LTP_point_df["norm_intensity"].mean() - LTP_point_df["norm_intensity"].std(), color="red", linestyle=":",
            xmin=0.4, xmax=0.6)
#plot text showing mean and std in mean plusminus std way
plt.text(0.5, LTP_point_df["norm_intensity"].mean(), f"{LTP_point_df['norm_intensity'].mean():.2f} ± {LTP_point_df['norm_intensity'].std():.2f}",
        ha='center', va='bottom', fontsize=8, color="black")
y_label = plt.ylabel("Normalized $\Delta$volume (a.u.)")
y_lim = plt.ylim()
#delete the right and top axis
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(save_plot_folder, "intensity_swarm_binned.png"), dpi=150,bbox_inches="tight")
plt.show()

plt.figure(figsize=(2, 3))
sns.swarmplot(y="norm_intensity", 
            data=LTP_point_df_cut_too_large, legend=False)
#plot mean line
mean_intensity = LTP_point_df_cut_too_large["norm_intensity"].mean()
sd_intensity = LTP_point_df_cut_too_large["norm_intensity"].std()
plt.axhline(mean_intensity, color="red", linestyle="-",
            xmin=0.3, xmax=0.7)
plt.axhline(mean_intensity + sd_intensity, color="red", linestyle=":",
            xmin=0.4, xmax=0.6)
plt.axhline(mean_intensity - sd_intensity, color="red", linestyle=":",
            xmin=0.4, xmax=0.6)

#plot text showing mean and std in mean plusminus std way
plt.text(0.5, mean_intensity, f"{mean_intensity:.2f} ± {sd_intensity:.2f}",
        ha='center', va='bottom', fontsize=8, color="black")
y_label = plt.ylabel("Normalized $\Delta$volume (a.u.)")
plt.ylim(y_lim)
#delete the right and top axis
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(save_plot_folder, "intensity_swarm_binned_cut_too_large.png"), dpi=150,bbox_inches="tight")
plt.show()

# %%

plt.figure(figsize=(4, 3))
sns.scatterplot(x='time_after_started_experiment_min', y="norm_intensity", 
            data=LTP_point_df, hue="label",legend=False, palette=['k'])
y_label = plt.ylabel("Normalized $\Delta$volume (a.u.)")
x_label = plt.xlabel("Time (min) after started experiment")
plt.xlim([0,LTP_point_df["time_after_started_experiment_min"].max()*1.05])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(save_plot_folder, "deltavolume_against_time_after_start.png"), dpi=150,bbox_inches="tight")
plt.show()


# %%
# Enhanced ROI mask verification with detailed information
print("\n" + "="*80)
print("ENHANCED ROI MASK VERIFICATION")
print("="*80)

# Get all unique groups for comprehensive check
if 'filepath_without_number' in combined_df.columns:
    unique_groups = combined_df['filepath_without_number'].unique()
    group_col = 'filepath_without_number'
else:
    unique_groups = combined_df['group'].unique()
    group_col = 'group'

print(f"Found {len(unique_groups)} unique groups to check")

roi_types = ["Spine", "DendriticShaft", "Background"]
colors = ['red', 'blue', 'green']

for group in unique_groups[:3]:  # Check first 3 groups to avoid too much output
    print(f"\n--- GROUP: {group} ---")
    group_df = combined_df[combined_df[group_col] == group]
    
    # Check each set within this group
    for set_label in group_df['nth_set_label'].unique():
        if set_label == -1:
            continue
            
        set_df = group_df[(group_df['nth_set_label'] == set_label) & 
                         (group_df['nth_omit_induction'] >= 0)]
        
        if len(set_df) == 0:
            continue
            
        print(f"  Set {set_label}: {len(set_df)} analysis frames")
        
        # Check each ROI type
        for roi_type in roi_types:
            mask_col = f"{roi_type}_roi_mask"
            timestamp_col = f"{roi_type}_roi_analysis_timestamp"
            
            if mask_col in combined_df.columns:
                masks_exist = set_df[mask_col].notna().sum()
                total_frames = len(set_df)
                
                if masks_exist > 0:
                    # Get timestamp of the analysis
                    latest_timestamp = set_df[timestamp_col].max()
                    
                    # Get a sample mask to check properties
                    sample_mask = set_df[set_df[mask_col].notna()].iloc[0][mask_col]
                    mask_area = np.sum(sample_mask) if isinstance(sample_mask, np.ndarray) else 0
                    
                    print(f"    {roi_type}: {masks_exist}/{total_frames} frames, "
                          f"area={mask_area} pixels, updated={latest_timestamp}")
                else:
                    print(f"    {roi_type}: No masks found")

# %%

