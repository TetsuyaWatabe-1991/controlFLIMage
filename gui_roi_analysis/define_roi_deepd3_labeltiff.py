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
# from PyQt5.QtWidgets import QApplication
from matplotlib import pyplot as plt  
from multidim_tiff_viewer import read_xyz_single
from flimage_graph_func import calc_point_on_line_close_to_xy
from skimage.draw import disk
from scipy.ndimage import median_filter
from skimage.draw import polygon
from FLIMageFileReader2 import FileReader

# import seaborn as sns
# from gui_integration import shift_coords_small_to_full_for_each_rois, first_processing_for_flim_files
# from gui_roi_analysis.file_selection_gui import launch_file_selection_gui
# from fitting.flim_lifetime_fitting import FLIMLifetimeFitter
# from FLIMageFileReader2 import FileReader
# from simple_dialog import ask_yes_no_gui, ask_save_path_gui, ask_open_path_gui, ask_save_folder_gui
# from plot_functions import draw_boundaries
# from utility.send_notification import send_slack_url_default

#%%
SHIFT_DIRECTION = -1

simple_roi_savefolder = r"C:\Users\WatabeT\Desktop\20250701\auto1\simple_roi_no_shift"
plot_savefolder = os.path.join(simple_roi_savefolder, "plot")
for each_folder in [simple_roi_savefolder, plot_savefolder]:
    os.makedirs(each_folder, exist_ok=True)

df_path = r"C:\Users\WatabeT\Desktop\20250701\auto1\combined_df_2.pkl"
combined_df = pd.read_pickle(df_path)

# filepath_without_number = r"C:\Users\WatabeT\Desktop\20250701\auto1\lowmag1__highmag_3_"
filepath_without_number = r"C:\Users\WatabeT\Desktop\20250701\auto1\lowmag3__highmag_4_"
nth_set_label = 2


for filepath_without_number in combined_df["filepath_without_number"].unique():
    each_filepath_df = combined_df[(combined_df["nth_set_label"] >0)
                            & (combined_df["filepath_without_number"] == filepath_without_number)]
    if len(each_filepath_df) < 3:
        continue

    for each_nth_set_label in each_filepath_df["nth_set_label"].unique():
        # each_set_df = each_set_df[each_set_df["nth_set_label"] == each_nth_set_label]
                

        each_set_df = combined_df[(combined_df["nth_set_label"] == nth_set_label)
                                    & (combined_df["filepath_without_number"] == filepath_without_number)
                                    & (combined_df["nth_omit_induction"] > 0)
                                    ]
        if len(each_set_df) < 3:
            continue


        ini_path_list = glob.glob(os.path.join(
                                f"{filepath_without_number[:-1]}",
                                f"{os.path.basename(filepath_without_number)[:-1]}_*.ini"                           
                                ))

        non_rejected_inipath = []


        circle_radius = 4
        rect_length = 13
        rect_height = 4
        ch_1or2 = 2

        idx = each_set_df.index[0]

        for idx in each_set_df.index:

            file_path = each_set_df.loc[idx, "file_path"]
            uncaging_iminfo = FileReader()
            uncaging_iminfo.read_imageFile(file_path, True) 
            z_from = int(each_set_df.loc[idx, "z_from"])
            z_to = int(each_set_df.loc[idx, "z_to"])

            imagearray=np.array(uncaging_iminfo.image)
            zyx_array = imagearray[z_from:z_to,0,ch_1or2-1,:,:,:].sum(axis=-1)
            z_proj = zyx_array.max(axis=0)


            for each_ini in ini_path_list:
                _, _, _, excluded = read_xyz_single(each_ini, return_excluded=True)
                if excluded == 0:
                    non_rejected_inipath.append(each_ini)

            spine_inipath = non_rejected_inipath[nth_set_label]

            [spine_z, spine_y, spine_x], dend_slope, dend_intercept = read_xyz_single(spine_inipath,
                                                                return_excluded = False)

            shifted_spine_y = spine_y + (each_set_df.loc[idx, "shift_y"] + each_set_df.loc[idx, "small_shift_y"])*SHIFT_DIRECTION
            shifted_spine_x = spine_x + (each_set_df.loc[idx, "shift_x"] + each_set_df.loc[idx, "small_shift_x"])*SHIFT_DIRECTION
            shifted_dend_intercept = dend_intercept - dend_slope * (each_set_df.loc[idx, "shift_x"] + each_set_df.loc[idx, "small_shift_x"])*SHIFT_DIRECTION + (each_set_df.loc[idx, "shift_y"] + each_set_df.loc[idx, "small_shift_y"])*SHIFT_DIRECTION

            shifted_spine_y = spine_y 
            shifted_spine_x = spine_x 
            shifted_dend_intercept = dend_intercept 

            shifted_y_c, shifted_x_c = calc_point_on_line_close_to_xy(x = shifted_spine_x, y = shifted_spine_y, 
                                            slope = dend_slope, 
                                            intercept = shifted_dend_intercept)

            rr_circ, cc_circ = disk((shifted_spine_y, shifted_spine_x), circle_radius, shape=z_proj.shape)

            theta = np.arctan(dend_slope)
            dx = (rect_length / 2) * np.cos(theta)
            dy = (rect_length / 2) * np.sin(theta)
            px = (rect_height / 2) * -np.sin(theta)
            py = (rect_height / 2) * np.cos(theta)

            # Rectangle corners
            corners_x = [shifted_x_c - dx - px, shifted_x_c - dx + px, shifted_x_c + dx + px, shifted_x_c + dx - px]
            corners_y = [shifted_y_c - dy - py, shifted_y_c - dy + py, shifted_y_c + dy + py, shifted_y_c + dy - py]

            rr_rect, cc_rect = polygon(corners_y, corners_x, shape=z_proj.shape)

            plt.imshow(z_proj, cmap="gray")
            plt.plot(cc_circ, rr_circ, 'r.', markersize=1, alpha=0.5)
            plt.plot(cc_rect, rr_rect, 'b.', markersize=1, alpha=0.5)
            plt.savefig(os.path.join(simple_roi_savefolder, f"{os.path.basename(file_path)[:-4]}_simple_roi.png"), dpi=150, bbox_inches="tight")
            plt.close()

            spine_intensity = z_proj[rr_circ, cc_circ].sum()
            spine_pix = rr_circ.size
            spine_intensity_per_pix = spine_intensity / spine_pix

            each_set_df.loc[idx, "spine_intensity_per_pix"] = spine_intensity_per_pix
            combined_df.loc[idx, "spine_intensity_per_pix"] = spine_intensity_per_pix

        each_set_df["normalized_spine_intensity_per_pix"] = each_set_df["spine_intensity_per_pix"]/(each_set_df["spine_intensity_per_pix"].iloc[:2].mean()) - 1
        combined_df.loc[each_set_df.index, "normalized_spine_intensity_per_pix"] = each_set_df["normalized_spine_intensity_per_pix"]
        plt.plot(each_set_df["relative_time_min"], each_set_df["normalized_spine_intensity_per_pix"])

        ylim = [-0.4,2.1]
        xlim = [-11,36]
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.savefig(os.path.join(plot_savefolder, f"{os.path.basename(filepath_without_number)[:-1]}_{each_nth_set_label}_simple_roi_plot.png"), dpi=150, bbox_inches="tight")
        plt.close()



ltp_list = []
after_min = 25
for filepath_without_number in combined_df["filepath_without_number"].unique():
    each_filepath_df = combined_df[(combined_df["nth_set_label"] >0)
                            & (combined_df["filepath_without_number"] == filepath_without_number)]
    if len(each_filepath_df) < 3:
        continue

    for each_nth_set_label in each_filepath_df["nth_set_label"].unique():
        # each_set_df = each_set_df[each_set_df["nth_set_label"] == each_nth_set_label]
                

        each_set_df = combined_df[(combined_df["nth_set_label"] == nth_set_label)
                                    & (combined_df["filepath_without_number"] == filepath_without_number)
                                    & (combined_df["nth_omit_induction"] > 0)
                                    ]
        if len(each_set_df) < 3:
            continue

        each_set_df_after_min = each_set_df[each_set_df["relative_time_min"] > after_min]

        idx_after_min = each_set_df_after_min["relative_time_min"].idxmin()

        ltp_list.append(each_set_df_after_min.loc[idx_after_min, "normalized_spine_intensity_per_pix"])

import seaborn as sns
sns.swarmplot(ltp_list)
mean = np.mean(ltp_list)
plt.axhline(mean, color="red", linestyle="--")
# plt.savefig(os.path.join(plot_savefolder, f"{os.path.basename(filepath_without_number)[:-1]}_{each_nth_set_label}_simple_roi_plot.png"), dpi=150, bbox_inches="tight")
plt.show()

print("mean", mean)
print("std", np.std(ltp_list))
print("snr", mean/np.std(ltp_list))





# %%