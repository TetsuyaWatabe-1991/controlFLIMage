# %%
# import
import os
import sys
sys.path.append('..\\')
import shutil
from time import sleep
# import glob
# import numpy as np
import pandas as pd
# import tifffile
# from PyQt5.QtWidgets import QApplication
from matplotlib import pyplot as plt
# from FLIMageAlignment import flim_files_to_nparray
# from gui_integration import launch_roi_analysis_gui, shift_coords_small_to_full_for_each_rois,temp_add_align_info
# from file_selection_gui import launch_file_selection_gui
# from fitting.flim_lifetime_fitting import FLIMLifetimeFitter
# from FLIMageFileReader2 import FileReader
from save_roi import save_multiple_rois_to_imagej
from controlflimage_threading import Control_flimage
# %%
# Setup parameters    
one_of_filepath = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250527\2ndlowmag\lowmag1__highmag_1_002.flim"
ch_1or2 = 2
z_plus_minus = 1

imagejroi_savefolder = os.path.join(os.path.dirname(one_of_filepath), "imagejroi")

parent_dir = os.path.dirname(one_of_filepath)
df_save_path_2 = os.path.join(parent_dir, "combined_df_2.pkl")

roi_types = ["Spine", "DendriticShaft", "Background"]


os.makedirs(imagejroi_savefolder, exist_ok=True)
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


# %%
#save 
while True:
    yn = input("Do you want to save the roi? (y/n)")
    if yn == "y":
        break
    elif yn == "n":
        break
    else:
        continue

if yn == "y":
    shifted_masks = {}
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
                
                for each_roi_type in roi_types:
                    shifted_masks[each_roi_type] = each_set_df.at[current_index, f"{each_roi_type}_shifted_mask"]

                file_path = each_set_df.at[current_index, "file_path"]                
                ij_roi_save_path = os.path.join(imagejroi_savefolder, f"{each_group}_{each_set_label}_{nth_omit_induction}.txt")
                save_multiple_rois_to_imagej(shifted_masks, roi_types, ij_roi_save_path)
                print(f"Saved {ij_roi_save_path}")
 
# %%

FLIMageCont = Control_flimage(ini_path = 'all_1')

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
            FLIMageCont.flim.sendCommand(f'OpenFile, {file_path}')
            
            # input("Press Enter to continue...")

            
            ij_roi_save_path = os.path.join(imagejroi_savefolder, f"{each_group}_{each_set_label}_{nth_omit_induction}.txt")
            roi_txt_path = os.path.join(os.path.dirname(file_path),
                                        "Analysis","ROI",
                                        os.path.basename(file_path).replace(".flim","_ROI.txt"))
            assert os.path.exists(roi_txt_path), f"roi_txt_path {roi_txt_path} does not exist"

            for i in range(20):
                try:
                    with open(roi_txt_path, 'r+'):
                        break
                except IOError:
                    sleep(0.1)
            else:
                print(f"Failed to copy {ij_roi_save_path} to {roi_txt_path}")
                continue

            shutil.copyfile(ij_roi_save_path, roi_txt_path)
            print(f"Copied {ij_roi_save_path} to {roi_txt_path}")

            FLIMageCont.flim.sendCommand(f'RecoverROIs')
            
            print("Recovered ROIs")


            # input("Press Enter to continue...")
            z_from = 1+ int(each_set_df.at[current_index, "z_from"])
            z_to = 1+ int(each_set_df.at[current_index, "z_to"])

            FLIMageCont.flim.sendCommand(f'CalcTimeCourse')
            # input("Press Enter to continue...")

            FLIMageCont.flim.sendCommand(f'ExtractPages, {z_from}-{z_to}')
            # input("Press Enter to continue...")
            FLIMageCont.flim.sendCommand(f'CalcTimeCourse')

            # input("Press Enter to continue...")


# %%
