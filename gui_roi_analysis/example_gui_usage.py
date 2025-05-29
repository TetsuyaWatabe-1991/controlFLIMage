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
from FLIMageAlignment import flim_files_to_nparray
from gui_integration import launch_roi_analysis_gui, shift_coords_small_to_full_for_each_rois,temp_add_align_info
from file_selection_gui import launch_file_selection_gui

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

print(f"Loading data from: {df_save_path_1}")
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
file_selection_gui = launch_file_selection_gui(combined_df)
app.exec_()
print("ROI visualization plots completed.")
combined_df.to_pickle(df_save_path_2)

# %%
# save plots showing image and roi masks
print("\n" + "="*40)
print("Creating ROI visualization plots...")

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




# %% fullsize, save plots showing image and roi masks on full size images
fullsize_save_folder = os.path.join(parent_dir, "fullsize_plots")
os.makedirs(fullsize_save_folder, exist_ok=True)
# Check if any ROI masks are defined before processing fullsize images
roi_columns = [col for col in combined_df.columns if col.endswith('_roi_mask')]
filelist = combined_df[(combined_df["nth_omit_induction"] != -1) & 
                        (combined_df["nth_set_label"] != -1) &
                        (combined_df["Spine_roi_mask"].notna())
                        ]["file_path"].tolist()

Tiff_MultiArray, iminfo, _ = flim_files_to_nparray(
    filelist, ch=ch_1or2 - 1, normalize_by_averageNum=False
)   
combined_df = shift_coords_small_to_full_for_each_rois(combined_df, z_plus_minus,
                roi_types = roi_types,  
                image_shape = Tiff_MultiArray.shape[2:])
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
            nth_image = filelist.index(file_path)
            each_tiff_array = Tiff_MultiArray[nth_image,:,:,:]

            roi_defined_list = []
            for each_roi_type in roi_types:
                shifted_mask = each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"]
                if shifted_mask is not None:
                    roi_defined_list.append(each_roi_type)

            if len(roi_defined_list) == 0:
                print(f"No ROI masks defined for nth_omit_induction {nth_omit_induction}, skipping...")
                continue

            corrected_uncaging_z = each_set_df.at[current_index, "corrected_uncaging_z"]

            z_from = int(max(0, corrected_uncaging_z - z_plus_minus))
            z_to = int(min(each_tiff_array.shape[0], corrected_uncaging_z + z_plus_minus + 1))

            each_tiff_array = each_tiff_array[z_from:z_to,:,:]
            z_projection = each_tiff_array.max(axis=0)
            plt.imshow(z_projection, cmap="gray",interpolation="nearest")

            for each_roi_type in roi_defined_list:
                plt.contour(each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"],
                                    colors=color_dict[each_roi_type], levels=[0.99])
            save_path = os.path.join(fullsize_save_folder, f"{each_group}_{each_set_label}_{nth_omit_induction}.png")
            plt.savefig(save_path, dpi=150,bbox_inches="tight")
            plt.close();plt.clf()

print("All processing completed successfully!")



# %%
