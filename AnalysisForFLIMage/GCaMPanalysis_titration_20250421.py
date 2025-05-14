# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 13:46:04 2025

@author: yasudalab
"""

import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
from datetime import datetime
import os
import glob
import pandas as pd
import numpy as np
from FLIMageAlignment import get_flimfile_list
from FLIMageFileReader2 import FileReader
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
# one_of_filepath = r"G:\ImagingData\Tetsuya\20240827\24well_0808GFP\highmag_Trans5ms\tpem2\C1_00_1_2__highmag_1_002.flim"

# one_of_filepath = r"G:\ImagingData\Tetsuya\20250227\automation\highmag_highmag_list\tpem\14_1__highmag_1_002.flim"
one_of_filepath = r"G:\ImagingData\Tetsuya\20250227\0131Cas9_GC6stdTom_neuron1_dend1_001.flim"
# ch_1_or_2 = 2

# one_of_file_list = glob.glob(os.path.join(
#                                 os.path.dirname(one_of_filepath),"*_highmag_*002.flim"))

# one_of_file_list = glob.glob(os.path.join(
#                                 os.path.dirname(one_of_filepath),"*_dend*002.flim"))


each_firstfilepath = r"G:\ImagingData\Tetsuya\20250403\B6_cut0319_FlxGC6sTom_0322\highmag_Trans5ms\tpem\B3_00_3_1_dend1_004.flim"

each_firstfilepath =r"Z:\Users\WatabeT\20250225\neuron2_dend2_titration_006.flim"
each_firstfilepath =r"Z:\Users\WatabeT\20250225\neuron3_dend1_titration_001.flim"
each_firstfilepath =r"Z:\Users\WatabeT\20250225\neuron3_dend2_titration_001.flim"
each_firstfilepath =r"G:\ImagingData\Tetsuya\20250421\B6GC6sTom0331\tpem\pos2_manual_dend2_spine1_001.flim"

filelist = get_flimfile_list(each_firstfilepath)
uncaging_nth_list = []

# First=True
    
for each_file in filelist: 

    uncaging_iminfo = FileReader()
    uncaging_iminfo.read_imageFile(each_file, True) 
    
    unc_dt = datetime.fromisoformat(uncaging_iminfo.acqTime[2])
    imagearray=np.array(uncaging_iminfo.image)
    uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
    uncaging_pow = uncaging_iminfo.statedict["State.Uncaging.Power"]
    
    center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
    center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]
   
    # GCpre = imagearray[0:8,0,0,:,:,:].sum(axis = -1).sum(axis = 0)
    # GCunc = imagearray[17:25,0,0,:,:,:].sum(axis = -1).sum(axis = 0)
    GCpre = imagearray[0,0,0,:,:,:].sum(axis = -1)
    GCunc = imagearray[3,0,0,:,:,:].sum(axis = -1)
    Tdpre = imagearray[0,0,1,:,:,:].sum(axis = -1)
    Td1min = imagearray[-1,0,1,:,:,:].sum(axis = -1)
    # Plot each array
    
    GC_pre_med = median_filter(GCpre, size=3)
    GC_unc_med = median_filter(GCunc, size=3)
    
    
    # GCF_F0 = (GCunc/GCpre)
    GCF_F0 = (GC_unc_med/GC_pre_med)
    GCF_F0[GC_pre_med == 0] = 0
    plt.imshow(GCF_F0, cmap = "inferno", vmin = 1, vmax = 10)
    plt.plot(center_x, center_y, 'ro', markersize=2)   
    plt.title(f"{uncaging_pow} %")   
    
    folder = os.path.dirname(each_file)
    savefolder = os.path.join(folder,"plot")
    os.makedirs(savefolder, exist_ok=True)
    basename = os.path.basename(each_file)
                    
    savepath = os.path.join(savefolder, basename[:-5] + ".png")
    plt.savefig(savepath, dpi=150, bbox_inches = "tight")
    
    plt.show()
    continue
    

    plt.imshow(GC_pre_med, cmap = "gray", vmin = 0, vmax = 100)
    plt.show()
    plt.imshow(GC_unc_med, cmap = "gray", vmin = 0, vmax = 100)
    plt.show()

    # # Calculate the square boundaries
    # point = [center_y, center_x]
    # square_size = 30

    # row, col = point
    # half_size = square_size // 2
    # row_start = int(max(0, row - half_size))
    # row_end = int(min(imagearray.shape[-2], row + half_size + 1))
    # col_start = int(max(0, col - half_size))
    # col_end = int(min(imagearray.shape[-3], col + half_size + 1))
    
    # # Get the maximum value in the square region
    # GC_square_region = GCunc[row_start:row_end, col_start:col_end]
    # GC_max_val = np.max(GC_square_region)
    # GC_min_val = np.min(GC_square_region)
    
    # tdTom_square_region = Td1min[row_start:row_end, col_start:col_end]
    # tdTom_max_val = np.max(tdTom_square_region)
    # tdTom_min_val = np.min(tdTom_square_region)

    # fig, axes = plt.subplots(1, 7, figsize=(3*7, 3),
    #                          gridspec_kw={'wspace': 0.05, 'hspace': 0})  # Reduced wspace
    # Titles for each subplot
    # titles = ['GCaMP pre', 'GCaMP uncaging',  'GCaMP F/F0', 'tdTomato pre', 'tdTomato 1 min',
    #           'tdTomato 0 min', f'tdTomato {minutes_after_uncaging} min']
                                    
    # for ax, arr, title in zip(axes, [GCpre, GCunc, GCF_F0,Tdpre, Td1min, td_before, td_after], titles):
        
    #     if title in titles[0:2]:
    #         vmin = GC_min_val
    #         vmax = GC_max_val
    #     if title in titles[2:]:
    #         vmin = tdTom_min_val
    #         vmax = tdTom_max_val
        
    #     # plt.imshow(GCF_F0, vmin = 1, vmax = 15)
    #     if title == titles[2]:
    #         im = ax.imshow(arr, cmap='inferno', vmin = 1, vmax = 10)
            
    #     else:
    #         im = ax.imshow(arr, cmap='gray', vmin = vmin, vmax = vmax)
    #     ax.set_title(title)
    #     ax.set_xticks([])  # Remove x ticks
    #     ax.set_yticks([])  # Remove y ticks
    #     ax.set_xticklabels([])  # Remove x labels
    #     ax.set_yticklabels([])  # Remove y labels
    #     if title in titles[:4]:
    #         ax.plot(col, row, 'ro', markersize=2)   
    
    # fig.suptitle(each_set["unc"], y=1.05, fontsize=14)  # y controls vertical position
    # # plt.suptitle('Comparison of 2D Arrays', y=1.05, fontsize=14)  # y controls vertical position
    # plt.tight_layout()  # Adjust spacing between subplots
    
    # folder = os.path.dirname(each_set["unc"])
    # savefolder = os.path.join(folder,"plot")
    # os.makedirs(savefolder, exist_ok=True)
    # basename = os.path.basename(each_set["unc"])
                    
    # savepath = os.path.join(savefolder, basename[:-5] + ".png")
    # plt.savefig(savepath, dpi=150, bbox_inches = "tight")
    # plt.show()
            
    # break      
    # input()
                

    # uncaging_minus1 = uncaging_iminfo.filename[:-8]+str(num-1).zfill(3)+".flim"
    # uncaging_minus1_index = filelist.index(uncaging_minus1)
    # uncaging_nth_z_for_all = spine_z
    # Aligned_mip_array = Aligned_4d_array[:,
    #                                       max(uncaging_nth_z_for_all-1,0):
    #                                       min(uncaging_nth_z_for_all+2,Aligned_4d_array.shape[1]),
    #                                       :,:].max(axis = 1)
    # if len(center_x_list)>0:
    #     roi_text = save_roi.make_multi_circle_roi(center_x_list,
    #                                               center_y_list,
    #                                               radius_list
    #                                               )
    #     save_roi.save_roi_textfiles(flim_path = each_firstfilepath,
    #                                 roi_text = roi_text)
    # else:
    #     print("No uncaging. Skip this file set.")