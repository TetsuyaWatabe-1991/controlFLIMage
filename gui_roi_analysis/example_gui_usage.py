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
# from FLIMageAlignment import flim_files_to_nparray
from gui_integration import launch_roi_analysis_gui, shift_coords_small_to_full_for_each_rois,temp_add_align_info
from file_selection_gui import launch_file_selection_gui
from fitting.flim_lifetime_fitting import FLIMLifetimeFitter
from FLIMageFileReader2 import FileReader
# %%

# Setup parameters    
one_of_filepath = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250527\2ndlowmag\lowmag1__highmag_1_002.flim"
ch_1or2 = 2
z_plus_minus = 1

plot_savefolder = os.path.join(os.path.dirname(one_of_filepath), "plot")
tif_savefolder = os.path.join(os.path.dirname(one_of_filepath), "tif")
roi_savefolder = os.path.join(os.path.dirname(one_of_filepath), "roi")
parent_dir = os.path.dirname(one_of_filepath)
df_save_path_1 = os.path.join(parent_dir, "combined_df_1.pkl")
df_save_path_2 = os.path.join(parent_dir, "combined_df_2.pkl")
roi_types = ["Spine", "DendriticShaft", "Background"]
color_dict = {"Spine": "red", "DendriticShaft": "blue", "Background": "green"}

# Check if the pickle file exists before trying to read it
if not os.path.exists(df_save_path_1):
    print(f"ERROR: Pickle file not found: {df_save_path_1}")
    print("Please run the data preparation step first.")
    sys.exit(1)

print(f"Loading data from: {df_save_path_2}")
try:
    combined_df = pd.read_pickle(df_save_path_2)
    print(f"Data loaded successfully. Shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")
    print(f"Groups: {combined_df['group'].unique() if 'group' in combined_df.columns else 'No group column'}")
    print(f"Set labels: {combined_df['nth_set_label'].unique() if 'nth_set_label' in combined_df.columns else 'No nth_set_label column'}")
except Exception as e:
    print(f"ERROR loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)



if False:
    # if you need to run the data preparation step
    for each_savefolder in [plot_savefolder, tif_savefolder, roi_savefolder]:
        os.makedirs(each_savefolder, exist_ok=True)

    combined_df = temp_add_align_info(
        one_of_filepath,
        plot_savefolder,
        tif_savefolder,
        z_plus_minus,
        )

    combined_df.to_pickle(df_save_path_1)

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
combined_df.to_pickle(df_save_path_2)

# %%
# save plots showing image and roi masks
print("\n" + "="*40)
print("Creating ROI visualization plots...")

while True:
    yn = input("Do you want to save small image and ROI mask plots? (y/n): ")
    if yn == "y":
        break
    elif yn == "n":
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
if yn == "y":
    for each_group in combined_df['group'].unique():
        each_group_df = combined_df[combined_df['group'] == each_group]
        for each_set_label in each_group_df["nth_set_label"].unique():
            if each_set_label == -1:
                continue
            temp_tiff_path = each_group_df[(each_group_df['nth_set_label'] == each_set_label)]["after_align_save_path"].values[0]
            if not os.path.exists(temp_tiff_path):
                print(f"Warning: TIFF file not found: {temp_tiff_path}")
                continue
            print(f"Processing Group: {each_group}, Set: {each_set_label}")
            tiff_array = tifffile.imread(temp_tiff_path)
            each_label_df = each_group_df[(each_group_df['group'] == each_group) & 
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
                    print(f"  Skipping Group: {each_group}, Set: {each_set_label}, Frame: {nth_omit_induction} - No ROI masks defined")
                    continue
                
                print(f"  Creating plot for Group: {each_group}, Set: {each_set_label}, Frame: {nth_omit_induction}")
                
                plt.figure(figsize=(10, 8))
                plt.imshow(each_tiff_array, cmap="gray")
                
                # Plot contours only if masks exist
                legend_handles = []
                if each_spine_roi_mask is not None and np.any(each_spine_roi_mask):
                    contour_spine = plt.contour(each_spine_roi_mask, colors="red", levels=[0.99])
                    legend_handles.append(plt.Line2D([0], [0], color="red", label="Spine ROI"))
                    
                if each_dendritic_shaft_roi_mask is not None and np.any(each_dendritic_shaft_roi_mask):
                    contour_dendritic_shaft = plt.contour(each_dendritic_shaft_roi_mask, colors="blue", levels=[0.99])
                    legend_handles.append(plt.Line2D([0], [0], color="blue", label="Dendritic Shaft ROI"))
                    
                if each_background_roi_mask is not None and np.any(each_background_roi_mask):
                    contour_background = plt.contour(each_background_roi_mask, colors="green", levels=[0.99])
                    legend_handles.append(plt.Line2D([0], [0], color="green", label="Background ROI"))
                
                if legend_handles:
                    plt.legend(handles=legend_handles)
                plt.title(f"Group: {each_group}, Set: {each_set_label}, Frame: {nth_omit_induction}")
                
                os.makedirs(roi_savefolder, exist_ok=True)
                save_path = os.path.join(roi_savefolder, f"{each_group}_{each_set_label}_{nth_omit_induction}.png")
                plt.savefig(save_path, dpi=150,bbox_inches="tight")
                plt.close();plt.clf()


combined_df = shift_coords_small_to_full_for_each_rois(combined_df, z_plus_minus,
                roi_types = roi_types,  
                image_shape = [128,128])

print("\n" + "="*40)
print("Creating ROI visualization plots...")

while True:
    yn = input("Do you want to save fullsize image and ROI mask plots? (y/n):")
    if yn == "y":
        break
    elif yn == "n":
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
photon_threshold = 3
total_photon_threshold = 100
lifetime_measure_ch_1or2_list = [1]
fixed_tau_bool = False
fixed_tau1 = 2.6
fixed_tau2 = 1.1
sync_rate = 80e6
if yn == "y":
    while True:
        yn2 = input("Do you want to analyze lifetime on the ROI? (y/n):")
        if yn2 == "y":
            fitter = FLIMLifetimeFitter()
            measure_lifetime = True
            break
        elif yn2 == "n":
            measure_lifetime = False
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


    fullsize_save_folder = os.path.join(parent_dir, "fullsize_plots")
    os.makedirs(fullsize_save_folder, exist_ok=True)
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

                file_path = each_set_df.at[current_index, "file_path"]                
                if file_path not in filelist:
                    print(f"File not found: {file_path}")
                    continue

                iminfo = FileReader()
                iminfo.read_imageFile(file_path, True)     
                six_dim = np.array(iminfo.image)
                
                # nth_image = filelist.index(file_path)
                # each_tiff_array = Tiff_MultiArray[nth_image,:,:,:]
                # each_tiff_array = Tiff_MultiArray[0,:,:,:]

                corrected_uncaging_z = each_set_df.at[current_index, "corrected_uncaging_z"]
                z_from = int(max(0, corrected_uncaging_z - z_plus_minus))
                z_to = int(min(six_dim.shape[0], corrected_uncaging_z + z_plus_minus + 1))
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
                    continue


                each_tiff_array = six_dim[z_from:z_to,0,ch_1or2 - 1,:,:,:].sum(axis=-1)
                z_projection = each_tiff_array.max(axis=0)
                plt.imshow(z_projection, cmap="gray",interpolation="nearest")

                for each_roi_type in roi_defined_list:
                    plt.contour(each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"],
                                        colors=color_dict[each_roi_type], levels=[0.99])
                save_path = os.path.join(fullsize_save_folder, f"{each_group}_{each_set_label}_{nth_omit_induction}.png")
                plt.savefig(save_path, dpi=150,bbox_inches="tight")
                plt.close();plt.clf()

    print("All processing completed successfully!")

combined_df.to_pickle(df_save_path_2)


# %%
# GCaMP analysis
# roi_types = ["Spine", "DendriticShaft", "Background"]
while True:
    yn = input("Do you want to analyze GCaMP? (y/n):")
    if yn == "y":
        break
    elif yn == "n":
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
        
if yn == "y":
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

    for each_group in combined_df["group"].unique():
        each_group_df = combined_df[combined_df["group"] == each_group]
        for each_set_label in each_group_df["nth_set_label"].unique():
            if each_set_label == -1:
                continue
                
            # Get uncaging phase data
            uncaging_data = each_group_df[(each_group_df["phase"] == "unc") & (each_group_df["nth_set_label"] == each_set_label)]
            if len(uncaging_data) == 0:
                print(f"No uncaging data found for group {each_group}, set {each_set_label}")
                continue
                
            uncaging_path = uncaging_data["file_path"].values[0]
            uncaging_df_index = uncaging_data.index[0]  # Get the actual index
            print(uncaging_df_index)
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

combined_df.to_pickle(df_save_path_2)

        
# %%
#plot intensity of each roi
combined_df["norm_intensity"] = -1.0
for each_group in combined_df["group"].unique():
    each_group_df = combined_df[combined_df["group"] == each_group]
    for each_set_label in each_group_df["nth_set_label"].unique():
        if each_set_label == -1:
            continue
        uncaging_df = each_group_df[(each_group_df["phase"] == "unc") & (each_group_df["nth_set_label"] == each_set_label)]
        combined_df.loc[uncaging_df.index, "label"] = f"{each_group}_{each_set_label}"

        each_set_df = each_group_df[(each_group_df["nth_set_label"] == each_set_label) &
                                     (each_group_df["nth_omit_induction"] != -1)]
        combined_df.loc[each_set_df.index, "subBG_intensity_per_pixel"] = each_set_df["Spine_intensity_sum"]/each_set_df["Spine_roi_area_pixels"] - each_set_df["Background_intensity_sum"]/each_set_df["Background_roi_area_pixels"]
        numerator = each_set_df["subBG_intensity_per_pixel"]
        denominator = each_set_df[each_set_df["phase"] == "pre"]["subBG_intensity_per_pixel"].mean(skipna=False)
        combined_df.loc[each_set_df.index, "norm_intensity"] = (numerator/denominator).astype(float) - 1.0
        combined_df.loc[each_set_df.index, "label"] = f"{each_group}_{each_set_label}"
        
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


# %%
#plot intensity time
save_plot_folder = os.path.join(parent_dir, "figure")
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
threshold_min = 20
results = []
for label in combined_df_reject_bad_data['label'].dropna().unique():
    label_df = combined_df_reject_bad_data[combined_df_reject_bad_data['label'] == label]
    above_threshold = label_df[label_df['relative_time_min'] > threshold_min]
    uncaging_df = combined_df[(combined_df['phase'] == "unc") & (combined_df['label'] == label)]
    if len(above_threshold) > 0:
        min_row = above_threshold.loc[above_threshold['relative_time_min'].idxmin()]
        
        results.append({
            'label': label,
            'time_min': min_row['relative_time_min'],
            'norm_intensity': min_row['norm_intensity'],
            'time_after_started_experiment_min': min_row['time_after_started_experiment_min'],
            'GCaMP_Spine_F_F0': uncaging_df['GCaMP_Spine_F_F0'],
            'GCaMP_DendriticShaft_F_F0': uncaging_df['GCaMP_DendriticShaft_F_F0']
        })

LTP_point_df = pd.DataFrame(results)
reject_threshold_too_large = 3
LTP_point_df_cut_too_large = LTP_point_df[LTP_point_df["norm_intensity"] < reject_threshold_too_large]



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
plt.axhline(LTP_point_df["norm_intensity"].mean() + LTP_point_df["norm_intensity"].std(), color="red", linestyle="-",
            xmin=0.4, xmax=0.6)
plt.axhline(LTP_point_df["norm_intensity"].mean() - LTP_point_df["norm_intensity"].std(), color="red", linestyle="-",
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
plt.axhline(mean_intensity + sd_intensity, color="red", linestyle="--",
            xmin=0.4, xmax=0.6)
plt.axhline(mean_intensity - sd_intensity, color="red", linestyle="--",
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
