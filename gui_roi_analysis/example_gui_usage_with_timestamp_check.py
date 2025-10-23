# %%

import os
import sys
import datetime
import configparser
sys.path.append('..\\')
import glob
import numpy as np
import pandas as pd
import tifffile
from PyQt5.QtWidgets import QApplication
from matplotlib import pyplot as plt  
import seaborn as sns
from gui_integration import shift_coords_small_to_full_for_each_rois, first_processing_for_flim_files
from gui_roi_analysis.file_selection_gui import launch_file_selection_gui
from fitting.flim_lifetime_fitting import FLIMLifetimeFitter
from FLIMageFileReader2 import FileReader
from simple_dialog import ask_yes_no_gui, ask_save_path_gui, ask_open_path_gui, ask_save_folder_gui
from plot_functions import draw_boundaries
from utility.send_notification import send_slack_url_default
from save_S_from_tzyx_2 import save_deepd3_S_tif
from spine_roi_from_S import define_roi_from_S


# %%
# Simple INI-based timestamp checking functions

def get_timestamp_ini_path(filepath_without_number):
    """Get path for timestamp INI file for a filepath group"""
    directory = os.path.dirname(filepath_without_number)
    filename = os.path.basename(filepath_without_number)
    ini_filename = f"{filename}_plot_timestamps.ini"
    print(f"ini_save_path: {os.path.join(directory, ini_filename)}")
    return os.path.join(directory, ini_filename)

def save_plot_timestamp_to_ini(filepath_without_number, plot_type):
    """Save current timestamp to INI file for this filepath group"""
    ini_path = get_timestamp_ini_path(filepath_without_number)

    config = configparser.ConfigParser()
    if os.path.exists(ini_path):
        config.read(ini_path)

    if 'plot_timestamps' not in config:
        config['plot_timestamps'] = {}

    current_time = datetime.datetime.now().isoformat()
    config['plot_timestamps'][f'{plot_type}_last_generated'] = current_time

    with open(ini_path, 'w') as configfile:
        config.write(configfile)

    return ini_path

def should_regenerate_group_plots(each_group_df, plot_type):
    """Check if plots should be regenerated for this filepath group"""
    filepath_without_number = each_group_df['filepath_without_number'].iloc[0]
    ini_path = get_timestamp_ini_path(filepath_without_number)
    print("start analyzing timestamp")
    # If INI file doesn't exist, regenerate
    if not os.path.exists(ini_path):
        return True, "No timestamp INI file found"

    try:
        config = configparser.ConfigParser()
        config.read(ini_path)

        # Check if this plot type has been generated before
        timestamp_key = f'{plot_type}_last_generated'
        if 'plot_timestamps' not in config or timestamp_key not in config['plot_timestamps']:
            return True, f"No {plot_type} timestamp found in INI"

        # Get the last generation time
        last_generated_str = config['plot_timestamps'][timestamp_key]
        last_generated = datetime.datetime.fromisoformat(last_generated_str)

        # Get the latest ROI analysis timestamp from this group
        roi_timestamps = []
        for roi_type in roi_types:
            timestamp_col = f"{roi_type}_roi_analysis_timestamp"
            if timestamp_col in each_group_df.columns:
                valid_timestamps = each_group_df[timestamp_col].dropna()
                if len(valid_timestamps) > 0:
                    # Ensure all timestamps are datetime objects
                    for ts in valid_timestamps:
                        if isinstance(ts, str):
                            roi_timestamps.append(datetime.datetime.fromisoformat(ts))
                        elif isinstance(ts, datetime.datetime):
                            roi_timestamps.append(ts)

        if not roi_timestamps:
            return False, "No ROI timestamps found - skipping"

        # Find the latest ROI timestamp
        latest_roi_timestamp = max(roi_timestamps)

        print(f"  Latest ROI timestamp: {latest_roi_timestamp} (type: {type(latest_roi_timestamp)})")
        print(f"  Last plot generated: {last_generated} (type: {type(last_generated)})")

        # Compare timestamps (both are now datetime objects)
        if latest_roi_timestamp > last_generated:
            return True, f"ROI updated at {latest_roi_timestamp}, plots generated at {last_generated}"
        else:
            return False, f"ROI timestamps are up to date (latest: {latest_roi_timestamp}, plots: {last_generated})"

    except Exception as e:
        return True, f"Error reading INI file: {str(e)}"

def print_simple_summary(total_groups, regenerated_groups, skipped_groups, plot_type):
    """Print simple summary of what was processed"""
    print(f"\n{plot_type.upper()} PLOT SUMMARY:")
    print(f"Total groups: {total_groups}")
    print(f"Regenerated: {regenerated_groups}")
    print(f"Skipped: {skipped_groups}")
    print("-" * 40)
# %%


yn = ask_yes_no_gui("Do without asking?")
Do_without_asking = yn

skip_gui_TF = True
Do_lifetime_analysis_TF = False
Do_GCaMP_analysis_TF = True
roi_define_from_S_TF = True

pre_defined_df_TF = False
df_defined_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250904\auto1\B1cnt_pos2__highmag_1_002.flim"

skip_plot_if_not_updated = False
# Setup parameters
ch_1or2 = 2

z_plus_minus = 2
pre_length = 2
lifetime_measure_ch_1or2_list = [1,2]
photon_threshold = 15

total_photon_threshold = 1000
sync_rate = 80e6
fixed_tau_bool = False
fixed_tau1 = 2.6
fixed_tau2 = 1.1

# one_of_filepath_list = [
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\4lines_2_auto\lowmag1__highmag_1_002.flim",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\2lines_1\lowmag1__highmag_1_002.flim",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250601\2lines3_auto\lowmag1__highmag_1_002.flim",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250601\4lines_3_auto\lowmag1__highmag_1_002.flim",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250602\lowmag1__highmag_1_002.flim",
#     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250602\4lines_neuron4\lowmag1__highmag_1_002.flim",
# ]
# one_of_filepath_list = [
#     r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250930\auto1\5_pos1__highmag_6_012.flim",
#     r"\\RY-LAB-WS04\ImagingData\Tetsuya\20251001\auto1\Tln_2_pos1__highmag_1_022.flim",
#     r"\\RY-LAB-WS04\ImagingData\Tetsuya\20251002\auto1\6_pos1__highmag_4_008.flim"
#     # r"C:\Users\WatabeT\Desktop\20250701\auto1\lowmag2__highmag_2_002.flim"
# ]


one_of_filepath_list = [
r"G:\ImagingData\Tetsuya\20251016\auto1\1_pos1__highmag_1_002.flim"
]


save_plot_TF = True
save_tif_TF = True

roi_types = ["Spine", "DendriticShaft", "Background"]
color_dict = {"Spine": "red", "DendriticShaft": "blue", "Background": "green"}



# if False:
if Do_without_asking == False:
    yn = ask_yes_no_gui("Do you already have combined_df_1.pkl?")
    if yn:
        df_save_path_1 = ask_open_path_gui()
        print(f"Loading data from: {df_save_path_1}")
        combined_df = pd.read_pickle(df_save_path_1)
else:
    if pre_defined_df_TF:
        df_save_path_1 = df_defined_path
        combined_df = pd.read_pickle(df_save_path_1)
    else:
        df_save_path_1 = os.path.join(os.path.dirname(one_of_filepath_list[0]), "combined_df_1.pkl")

if Do_without_asking == False:
    yn1 = ask_yes_no_gui("Do you want to run the data preparation step?")
    if (yn1 == False) and (yn == False):
        raise Exception("dataframe is not loaded. You do not define it.")

if (Do_without_asking == True) and (pre_defined_df_TF == True):
    print("skipping data preparation step")

elif (Do_without_asking == True) or (yn1 == True):  
    combined_df = pd.DataFrame()
    for one_of_filepath in one_of_filepath_list:
        print(f"\n\n\n Processing ... \n\n{one_of_filepath}\n\n\n")
        temp_df = first_processing_for_flim_files(
            one_of_filepath,
            z_plus_minus,
            ch_1or2,
            pre_length = pre_length,
            save_plot_TF = save_plot_TF,
            save_tif_TF = save_tif_TF,
            )
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)


    if Do_without_asking == False:
        send_slack_url_default(message = "finished data preparation step.")
        print("define save path for combined_df")
        df_save_path_1 = ask_save_path_gui()
        if df_save_path_1[-4:] != ".pkl":
            df_save_path_1 = df_save_path_1 + ".pkl"
    else:
        df_save_path_1 = os.path.join(os.path.dirname(one_of_filepath_list[0]), "combined_df.pkl")
    combined_df.to_pickle(df_save_path_1)
    combined_df.to_csv(df_save_path_1.replace(".pkl", ".csv"))
    print(f"saved combined_df to {df_save_path_1}")


# %%
do_roi_define_from_S = False

if Do_without_asking == False:
    if ask_yes_no_gui("Do you want to define ROI from S?"):
        do_roi_define_from_S = True

if Do_without_asking * roi_define_from_S_TF:
    do_roi_define_from_S = True

if do_roi_define_from_S:
    save_deepd3_S_tif(df_save_path_1)
    define_roi_from_S(df_save_path_1, save_path_suffix = "")
    combined_df = pd.read_pickle(df_save_path_1)

    print("define ROI from S finished")

    # combined_df["reject"] = False


# %%
# Launch the new file selection GUI instead of the old loop
if (Do_without_asking and skip_gui_TF) == False:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        print("Created new QApplication instance")
    else:
        print("Using existing QApplication instance")

    # Define additional columns to show in the GUI (optional)
    additional_columns_to_show = [
        'dt',
        "donotexist"  # Show timing information
    ]

    file_selection_gui = launch_file_selection_gui(
        combined_df,
        df_save_path_1,
        additional_columns=additional_columns_to_show,
        save_auto = False
    )
    app.exec_()
    print("ROI visualization plots completed.")
    combined_df = pd.read_pickle(df_save_path_1)

# %%
if Do_without_asking == False:
    if ask_yes_no_gui("stop here?"):
        assert False
    

# %%
# Simple small image and ROI mask plotting with INI timestamp checking
print("\n" + "="*40)
print("Creating small ROI visualization plots...")

if Do_without_asking == False:
    yn2 = ask_yes_no_gui("Do you want to save small image and ROI mask plots?")
else:
    yn2 = True

if yn2:
    total_groups = 0
    regenerated_groups = 0
    skipped_groups = 0

    for each_filepath in combined_df['filepath_without_number'].unique():
        each_group_df = combined_df[combined_df['filepath_without_number'] == each_filepath]
        display_group = each_group_df['group'].iloc[0]
        total_groups += 1

        # Check if regeneration is needed BEFORE loading any images
        should_regen, reason = should_regenerate_group_plots(each_group_df, "small")
        if skip_plot_if_not_updated:
            if not should_regen:
                print(f"Skipping group: {display_group} - {reason}")
                skipped_groups += 1
                continue

        print(f"Regenerating small plots for Group: {display_group} - {reason}")
        regenerated_groups += 1

        for each_set_label in each_group_df["nth_set_label"].unique():
            if each_set_label == -1:
                continue

            temp_tiff_path = each_group_df[(each_group_df['nth_set_label'] == each_set_label)]["after_align_save_path"].values[0]
            if not os.path.exists(temp_tiff_path):
                print(f"Warning: TIFF file not found: {temp_tiff_path}")
                continue

            print(f"  Processing Set: {each_set_label}")
            tiff_array = tifffile.imread(temp_tiff_path)
            each_label_df = each_group_df[(each_group_df['filepath_without_number'] == each_filepath) &
                                        (each_group_df['nth_set_label'] == each_set_label) &
                                        (each_group_df['nth_omit_induction'] != -1)]

            for nth_omit_induction in each_label_df["relative_nth_omit_induction"].unique():
                each_tiff_array = tiff_array[int(nth_omit_induction),:,:]

                current_df = each_label_df[each_label_df["relative_nth_omit_induction"] == nth_omit_induction]
                if len(current_df) == 0:
                    continue

                # Generate save path
                roi_savefolder = os.path.join(os.path.dirname(each_filepath), "roi_small")
                os.makedirs(roi_savefolder, exist_ok=True)
                plot_save_path = os.path.join(roi_savefolder, f"{display_group}_{each_set_label}_{nth_omit_induction}.png")

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

                if not any_roi_defined:
                    print(f"    Skipping Set: {each_set_label}, Frame: {nth_omit_induction} - No ROI masks defined")
                    continue

                print(f"    Creating plot for Set: {each_set_label}, Frame: {nth_omit_induction}")

                fig = plt.figure(figsize=(10, 8))
                plt.imshow(each_tiff_array, cmap="gray",
                        interpolation="none",
                        extent=(0, each_tiff_array.shape[1], each_tiff_array.shape[0], 0))

                legend_handles = []
                if each_spine_roi_mask is not None and np.any(each_spine_roi_mask):
                    draw_boundaries(each_spine_roi_mask, "red")
                    legend_handles.append(plt.Line2D([0], [0], color="red", label="Spine ROI"))

                if each_dendritic_shaft_roi_mask is not None and np.any(each_dendritic_shaft_roi_mask):
                    draw_boundaries(each_dendritic_shaft_roi_mask, "blue")
                    legend_handles.append(plt.Line2D([0], [0], color="blue", label="Dendritic Shaft ROI"))

                if each_background_roi_mask is not None and np.any(each_background_roi_mask):
                    draw_boundaries(each_background_roi_mask, "green")
                    legend_handles.append(plt.Line2D([0], [0], color="green", label="Background ROI"))

                if legend_handles:
                    plt.legend(handles=legend_handles)
                plt.title(f"Group: {display_group}, Set: {each_set_label}, Frame: {nth_omit_induction}")

                # Save plot
                fig.savefig(plot_save_path, dpi=150, bbox_inches="tight")
                plt.close();plt.clf()

        # Save timestamp to INI after all plots for this group are completed
        save_plot_timestamp_to_ini(each_filepath, "small")

    print_simple_summary(total_groups, regenerated_groups, skipped_groups, "small")
    
    if Do_without_asking == False:
        send_slack_url_default(message = "finished saving small image and ROI mask plots with INI timestamp checking.")

# %%
# Shift the ROI masks to the full size image
combined_df = shift_coords_small_to_full_for_each_rois(combined_df, z_plus_minus,
                roi_types = roi_types,
                image_shape = [128,128])

# %%
# Simple full size image and ROI mask plots with INI timestamp checking
print("\n" + "="*40)
print("Creating full-size ROI visualization plots...")

if Do_without_asking == False:
    yn3 = ask_yes_no_gui("Do you want to save fullsize image and ROI mask plots?")
else:
    yn3 = True

if yn3:
    if Do_without_asking == False:
        yn4 = ask_yes_no_gui("Do you want to analyze lifetime on the ROI?")
        measure_lifetime = yn4
    else:        
        measure_lifetime = Do_lifetime_analysis_TF

    if measure_lifetime:
        fitter = FLIMLifetimeFitter()


    roi_columns = [col for col in combined_df.columns if col.endswith('_roi_mask')]
    filelist = combined_df[(combined_df["nth_omit_induction"] != -1) &
                            (combined_df["nth_set_label"] != -1) &
                            (combined_df["Spine_roi_mask"].notna())
                            ]["file_path"].tolist()

    total_groups = 0
    regenerated_groups = 0
    skipped_groups = 0

    for each_filepath in combined_df["filepath_without_number"].unique():
        each_group_df = combined_df[combined_df["filepath_without_number"] == each_filepath]
        display_group = each_group_df['group'].iloc[0]
        total_groups += 1

        # Check if regeneration is needed BEFORE loading any images
        should_regen, reason = should_regenerate_group_plots(each_group_df, "fullsize")

        if skip_plot_if_not_updated:
            if not should_regen:
                print(f"Skipping fullsize group: {display_group} - {reason}")
                skipped_groups += 1
                continue

        print(f"Regenerating fullsize plots for Group: {display_group} - {reason}")
        regenerated_groups += 1

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

                # Generate save path for fullsize plot
                fullsize_save_folder = os.path.join(os.path.dirname(file_path), "fullsize_plots")
                os.makedirs(fullsize_save_folder, exist_ok=True)
                plot_save_path = os.path.join(fullsize_save_folder, f"{display_group}_{each_set_label}_{nth_omit_induction}.png")

                print(f"  Creating fullsize plot for Set: {each_set_label}, Frame: {nth_omit_induction}")

                # Load and process the image
                iminfo = FileReader()
                iminfo.read_imageFile(file_path, True)
                six_dim = np.array(iminfo.image)

                corrected_uncaging_z = each_set_df.at[current_index, "corrected_uncaging_z"]
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

                    # Lifetime analysis
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
                                combined_df.loc[current_index, f"{each_roi_type}_ch{each_ch_1or2}_photon_fitting"] = result_double_normal["lifetime"]
                            else:
                                result_fix_both = fitter.fit_double_exponential(x, y_data, ps_per_unit, sync_rate,
                                                fix_tau1=fixed_tau1, fix_tau2=fixed_tau2)
                                fixed_2component_fit_tau = result_fix_both["lifetime"]
                                combined_df.loc[current_index, f"{each_roi_type}_ch{each_ch_1or2}_photon_fitting"] = fixed_2component_fit_tau

                if len(roi_defined_list) == 0:
                    print(f"No ROI masks defined for nth_omit_induction {nth_omit_induction}, skipping...")
                    continue

                # Create the fullsize plot
                each_tiff_array = six_dim[z_from:z_to,0,ch_1or2 - 1,:,:,:].sum(axis=-1)
                z_projection = each_tiff_array.max(axis=0)
                if first:
                    first = False
                    vmax = z_projection.max()
                    vmin = 0

                fig = plt.figure()
                plt.imshow(z_projection, cmap="gray",interpolation="nearest",
                        vmax=vmax, vmin=vmin)
                plt.title(f"{display_group}_{each_set_label}_{nth_omit_induction}\nZ range: {z_from + 1} - {z_to}")

                for each_roi_type in roi_defined_list:
                    draw_boundaries(each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"],
                                    color=color_dict[each_roi_type], linewidth=1)
                    intensity_per_pixel = z_projection[each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"]].sum()/each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"].sum()
                    combined_df.loc[current_index, f"{each_roi_type}_intensity_per_pixel_from_full_image"] = intensity_per_pixel

                combined_df.loc[current_index, f"Spine_subBG_intensity_per_pixel_from_full_image"] = combined_df.loc[current_index, f"Spine_intensity_per_pixel_from_full_image"] - combined_df.loc[current_index, f"Background_intensity_per_pixel_from_full_image"]
                combined_df.loc[current_index, f"Dendritic_shaft_subBG_intensity_per_pixel_from_full_image"] = combined_df.loc[current_index, f"DendriticShaft_intensity_per_pixel_from_full_image"] - combined_df.loc[current_index, f"Background_intensity_per_pixel_from_full_image"]

                # Save plot
                fig.savefig(plot_save_path, dpi=150, bbox_inches="tight")
                plt.close();plt.clf()
                combined_df.loc[current_index, "full_plot_with_roi_path"] = plot_save_path

        # Save timestamp to INI after all plots for this group are completed
        save_plot_timestamp_to_ini(each_filepath, "fullsize")

    print_simple_summary(total_groups, regenerated_groups, skipped_groups, "fullsize")
    print("All processing completed successfully!")
    # send_slack_url_default(message = "finished saving fullsize image and ROI mask plots with INI timestamp checking.")

combined_df.to_pickle(df_save_path_1)

# %%
# Rest of the analysis continues as before...
# GCaMP analysis
if Do_without_asking == False:
    yn5 = ask_yes_no_gui("Do you want to analyze GCaMP?")
else:
    yn5 = Do_GCaMP_analysis_TF

if yn5:
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
        display_group = each_group_df['group'].iloc[0]
        for each_set_label in each_group_df["nth_set_label"].unique():
            if each_set_label == -1:
                continue

            uncaging_data = each_group_df[(each_group_df["phase"] == "unc") & (each_group_df["nth_set_label"] == each_set_label)]
            if len(uncaging_data) == 0:
                print(f"No uncaging data found for group {display_group}, set {each_set_label}")
                raise Exception(f"No uncaging data found for group {display_group}, set {each_set_label}\n something wrong for the data processing before this step.  Check it")

            uncaging_path = uncaging_data["file_path"].values[0]
            uncaging_df_index = uncaging_data.index[0]
            print(uncaging_df_index,"/",len(combined_df))

            #get index of the nth_omit_induction before uncaging
            each_set_label_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
            uncaging_df_index = each_set_label_df[each_set_label_df["phase"] == "unc"].index[0]
            nth_omit_induction_before_unc = each_set_label_df.loc[:uncaging_df_index, "nth_omit_induction"].idxmax()

            iminfo = FileReader()
            iminfo.read_imageFile(uncaging_path, True)
            six_dim = np.array(iminfo.image)

            for each_fluor_ch_1or2 in Fluor_ch_1or2_dict.keys():
                for each_roi_type in roi_types:
                    uncaging_array = six_dim[:, 0, Fluor_ch_1or2_dict[each_fluor_ch_1or2] - 1, :, :, :].sum(axis=-1)
                    shifted_mask = combined_df.loc[nth_omit_induction_before_unc, f"{each_roi_type}_shifted_mask"]
                    each_roi_array = uncaging_array[:,shifted_mask].sum(axis=-1)
                    combined_df.at[uncaging_df_index, f"{each_fluor_ch_1or2}_{each_roi_type}_array"] = each_roi_array

            for each_fluor_ch_1or2 in Fluor_ch_1or2_dict.keys():
                # spine_subBG = combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_Spine_array"]/combined_df.loc[uncaging_df_index-1, "Spine_roi_area_pixels"] - combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_Background_array"]/combined_df.loc[uncaging_df_index-1, "Background_roi_area_pixels"]
                # dendritic_shaft_subBG = combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_DendriticShaft_array"]/combined_df.loc[uncaging_df_index-1, "DendriticShaft_roi_area_pixels"] - combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_Background_array"]/combined_df.loc[uncaging_df_index-1, "Background_roi_area_pixels"]
                spine_subBG = combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_Spine_array"]/combined_df.loc[nth_omit_induction_before_unc, "Spine_roi_area_pixels"] - combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_Background_array"]/combined_df.loc[nth_omit_induction_before_unc, "Background_roi_area_pixels"]
                dendritic_shaft_subBG = combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_DendriticShaft_array"]/combined_df.loc[nth_omit_induction_before_unc, "DendriticShaft_roi_area_pixels"] - combined_df.loc[uncaging_df_index, f"{each_fluor_ch_1or2}_Background_array"]/combined_df.loc[nth_omit_induction_before_unc, "Background_roi_area_pixels"]
                spine_F_F0 = spine_subBG[afterbase_frame_index].mean()/spine_subBG[base_frame_index].mean()
                dendritic_shaft_F_F0 = dendritic_shaft_subBG[afterbase_frame_index].mean()/dendritic_shaft_subBG[base_frame_index].mean()
                combined_df.at[uncaging_df_index, f"{each_fluor_ch_1or2}_Spine_subBG"] = spine_subBG
                combined_df.at[uncaging_df_index, f"{each_fluor_ch_1or2}_DendriticShaft_subBG"] = dendritic_shaft_subBG
                combined_df.at[uncaging_df_index, f"{each_fluor_ch_1or2}_Spine_F_F0"] = spine_F_F0
                combined_df.at[uncaging_df_index, f"{each_fluor_ch_1or2}_DendriticShaft_F_F0"] = dendritic_shaft_F_F0
    

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
        each_set_df = each_group_df[(each_group_df["nth_set_label"] == each_set_label)]
        combined_df.loc[each_set_df.index, "label"] = f"{each_filepath}_{each_set_label}"

        each_set_df_without_unc = each_group_df[(each_group_df["nth_set_label"] == each_set_label) &
                                    (each_group_df["nth_omit_induction"] != -1)]
        numerator = combined_df.loc[each_set_df_without_unc.index,"Spine_subBG_intensity_per_pixel_from_full_image"]
        denominator = float(combined_df.loc[each_set_df_without_unc[each_set_df_without_unc["phase"] == "pre"].index, "Spine_subBG_intensity_per_pixel_from_full_image"].mean(skipna=False))

        combined_df.loc[each_set_df.index, "norm_intensity"] = (numerator/denominator).astype(float) - 1.0
        #combined_df.loc[each_set_df.index, "label"] = f"{each_filepath}_{each_set_label}"
        
        #Below will be impremented in other function??
        for NthFrame in each_set_df_without_unc["nth_omit_induction"].unique()[:-1]:
            Time_N = each_set_df_without_unc[(each_set_df_without_unc["nth_omit_induction"]==NthFrame)]["relative_time_sec"]
            Time_Nplus1 = each_set_df_without_unc[(each_set_df_without_unc["nth_omit_induction"]==NthFrame+1)]["relative_time_sec"]
            rownum = Time_N.keys()
            delta_sec = float(Time_Nplus1.values[0] - Time_N.values[0])
            combined_df.loc[rownum,"delta_sec"] = delta_sec

datetime_started_experiment = combined_df["dt"].min()
combined_df["time_after_started_experiment"] = combined_df["dt"] - datetime_started_experiment
combined_df["time_after_started_experiment_sec"] = combined_df["time_after_started_experiment"].dt.total_seconds()
combined_df["time_after_started_experiment_min"] = combined_df["time_after_started_experiment_sec"]/60
combined_df["time_after_started_experiment_hour"] = combined_df["time_after_started_experiment_min"]/60
combined_df["min_from_start"] = combined_df["time_after_started_experiment_sec"]//60
combined_df.to_pickle(df_save_path_1)


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

savepath_for_combined_df_reject_bad_data = df_save_path_1.replace(".pkl", "_reject_bad_data.pkl")
combined_df_reject_bad_data.to_pickle(savepath_for_combined_df_reject_bad_data)



# %%
#plot intensity time
default_save_plot_folder = os.path.dirname(df_save_path_1)
if Do_without_asking == False:
    save_plot_folder = ask_save_folder_gui(default_folder=default_save_plot_folder)
else:
    save_plot_folder = os.path.join(default_save_plot_folder, "summary")
os.makedirs(save_plot_folder, exist_ok=True)

# plot LTP level against elapsed time (min) for each group
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

# plot LTP level against BINNED time (min) for each group
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

threshold_min = 25
results = []
for label in combined_df_reject_bad_data['label'].dropna().unique():
    label_df = combined_df_reject_bad_data[combined_df_reject_bad_data['label'] == label]
    above_threshold = label_df[label_df['relative_time_min'] > threshold_min]
    uncaging_df = combined_df[(combined_df['phase'] == "unc") & (combined_df['label'] == label)]
    print(label)
    if len(above_threshold) > 0:
        min_row = above_threshold.loc[above_threshold['relative_time_min'].idxmin()]
        if yn5:
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
reject_threshold_too_small = -0.7
LTP_point_df_cut_too_large = LTP_point_df[(LTP_point_df["norm_intensity"] < reject_threshold_too_large) 
                                        & (LTP_point_df["norm_intensity"] > reject_threshold_too_small)]




if yn5:
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

LTP_point_df["time_after_started_experiment_hour"] = LTP_point_df["time_after_started_experiment_min"]/60
plt.figure(figsize=(4, 3))
sns.scatterplot(x='time_after_started_experiment_hour', y="norm_intensity", 
            data=LTP_point_df, hue="label",legend=False, palette=['k'])
y_label = plt.ylabel("Normalized $\Delta$volume (a.u.)")
x_label = plt.xlabel("Time (hour) after started experiment")
plt.xlim([0,LTP_point_df["time_after_started_experiment_hour"].max()*1.05])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.savefig(os.path.join(save_plot_folder, "deltavolume_against_time_after_start_hour.png"), dpi=150,bbox_inches="tight")
plt.show()


# %%
#save df to csv
combined_df.to_csv(os.path.join(save_plot_folder, "combined_df_after_analysis.csv"))
LTP_point_df.to_csv(os.path.join(save_plot_folder, "LTP_point_df_after_analysis.csv"))
print("saved as ", os.path.join(save_plot_folder, "combined_df_after_analysis.csv"))
print("saved as ", os.path.join(save_plot_folder, "LTP_point_df_after_analysis.csv"))

# %%
#print summary
from itertools import groupby

def count_true_groups(arr):
    return sum(1 for k, g in groupby(arr) if k)

uncaging_tf = combined_df['uncaging_frame'] == True
titration_tf = combined_df['titration_frame'] == True
uncaging_or_titraion_tf = (combined_df['uncaging_frame'] == True) | (combined_df['titration_frame'] == True)

#Trueの隣接したの個数を数える
uncaging_tf_count = count_true_groups(uncaging_tf.values)
titration_tf_count = count_true_groups(titration_tf.values)
uncaging_or_titraion_tf_count = count_true_groups(uncaging_or_titraion_tf.values)


#show above in a table

from tabulate import tabulate

# Trial information table
trial_info = [
    ["number of titration trials", titration_tf_count],
    ["number of uncaging trials", uncaging_tf_count],
    ["number of excluded data", uncaging_tf_count - len(LTP_point_df)],
    ["number of included data", len(LTP_point_df)]
]

print(tabulate(trial_info, tablefmt="simple_grid"))


print("before cut")
# LTP statistics table
ltp_mean = round(LTP_point_df["norm_intensity"].mean(), 2)
ltp_std = round(LTP_point_df["norm_intensity"].std(), 2)

ltp_stats = [
    ["Number of data", len(LTP_point_df)],
    ["LTP Mean", ltp_mean],
    ["LTP Std", ltp_std],
    ["LTP Max", round(LTP_point_df["norm_intensity"].max(), 2)],
    ["LTP Min", round(LTP_point_df["norm_intensity"].min(), 2)],
    ["LTP SNR", round(ltp_mean / ltp_std, 2)]
]

print(tabulate(ltp_stats, tablefmt="simple_grid"))



# LTP statistics table after cut
print("after cut too large or too small")



ltp_mean_after_cut = round(LTP_point_df_cut_too_large["norm_intensity"].mean(), 2)
ltp_std_after_cut = round(LTP_point_df_cut_too_large["norm_intensity"].std(), 2)

ltp_stats = [
    ["Number of data", len(LTP_point_df_cut_too_large)],
    ["LTP Mean", ltp_mean_after_cut],
    ["LTP Std", ltp_std_after_cut],
    ["LTP Max", round(LTP_point_df_cut_too_large["norm_intensity"].max(), 2)],
    ["LTP Min", round(LTP_point_df_cut_too_large["norm_intensity"].min(), 2)],
    ["LTP SNR", round(ltp_mean_after_cut / ltp_std_after_cut, 2)]
]

print(tabulate(ltp_stats, tablefmt="simple_grid"))







# %%
high_GC = LTP_point_df[LTP_point_df["GCaMP_DendriticShaft_F_F0"]>6]
print("mean: ",high_GC["norm_intensity"].mean())
print("std: ",high_GC["norm_intensity"].std())
print("number of data: ",len(high_GC), " / ", len(LTP_point_df))




high_GC2 = LTP_point_df[
    (LTP_point_df["GCaMP_DendriticShaft_F_F0"]<10) &
    (LTP_point_df["GCaMP_DendriticShaft_F_F0"]>5)]
print("mean: ",high_GC2["norm_intensity"].mean())
print("std: ",high_GC2["norm_intensity"].std())
print("number of data: ",len(high_GC2), " / ", len(LTP_point_df))

