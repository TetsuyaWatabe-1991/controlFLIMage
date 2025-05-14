# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 09:31:18 2025

@author: yasudalab
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:42:31 2024

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


ch_1_or_2 = 2




one_of_filepath = r"G:\ImagingData\Tetsuya\20250508\LTP2_13um_001.flim"

one_of_file_list = glob.glob(os.path.join(
                                os.path.dirname(one_of_filepath),"LTP*002.flim"))



# one_of_filepath = r"G:\ImagingData\Tetsuya\20250507\W_harp\lowmag1__highmag_3_087.flim"

# one_of_file_list = glob.glob(os.path.join(
#                                 os.path.dirname(one_of_filepath),"*_highmag_*002.flim"))

# one_of_file_list = glob.glob(os.path.join(
#                                 os.path.dirname(one_of_filepath),"*_d*002.flim"))

# one_of_file_list = glob.glob(os.path.join(
#                                 os.path.dirname(one_of_filepath),"A3_00_1_3__highmag_2_*002.flim"))

# one_of_file_list = "G:\ImagingData\Tetsuya\20241212\24well\highmag_GFP200ms55p\tpem_1\A3_00_2_1__highmag_1_004.flim"

# ch = ch_1_or_2 - 1
combined_df = pd.DataFrame()

## Not beautiful, but acceptable...
## 

pre_post_set_list = []

for each_firstfilepath in one_of_file_list:
    
    filelist = get_flimfile_list(each_firstfilepath)
    uncaging_nth_list = []
    
    First=True

    for nth, file_path in enumerate(filelist):
        iminfo = FileReader()
        # print(file_path)
        try:
            iminfo.read_imageFile(file_path, True) 
            imagearray=np.array(iminfo.image)
        except:
            print("\n\ncould not read\n")
            continue
           
        if First:
            First=False
            imageshape=imagearray.shape
    
        if imagearray.shape == imageshape:
            pass
        else:
            if (imagearray.shape[0] > 29):
                print(file_path,'<- uncaging')
                uncaging_nth_list.append(nth)
                
    print(uncaging_nth_list)                
    
    for number, nth in enumerate(uncaging_nth_list[:-1]):
        pre_post_set_list.append({
            "pre":filelist[nth-1],
            "unc":filelist[nth],
            "post":filelist[uncaging_nth_list[number + 1] - 2],
            })
    if len(uncaging_nth_list)>0:
        pre_post_set_list.append({
            "pre":filelist[uncaging_nth_list[-1] -1],
            "unc":filelist[uncaging_nth_list[-1]],
            "post":filelist[-1],
            })
    
    
for each_set in pre_post_set_list: 
    pre_iminfo = FileReader()
    pre_iminfo.read_imageFile(each_set["pre"], True) 
    pre_imagearray=np.array(pre_iminfo.image)
    pre_dt = datetime.fromisoformat(pre_iminfo.acqTime[-1])
    
    post_iminfo = FileReader()
    post_iminfo.read_imageFile(each_set["post"], True) 
    post_imagearray=np.array(post_iminfo.image)
    post_dt = datetime.fromisoformat(post_iminfo.acqTime[-1])
    
    td_before = pre_imagearray[:,0,1,:,:,:].sum(axis = -1).max(axis = 0)
    td_after = post_imagearray[:,0,1,:,:,:].sum(axis = -1).max(axis = 0)
    
    uncaging_iminfo = FileReader()
    uncaging_iminfo.read_imageFile(each_set["unc"], True) 
    
    unc_dt = datetime.fromisoformat(uncaging_iminfo.acqTime[2])
    imagearray=np.array(uncaging_iminfo.image)
    
    post_after_unc_dt = post_dt - unc_dt
    minutes_after_uncaging = (post_after_unc_dt.seconds)//60
    
    
    
    uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
    center_y = imageshape[-2] * uncaging_x_y_0to1[1]
    center_x = imageshape[-3] * uncaging_x_y_0to1[0]
   
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
    # plt.imshow(GCF_F0, cmap = "inferno", vmin = 1, vmax = 5)
    
    
    # Calculate the square boundaries
    point = [center_y, center_x]
    square_size = 30

    row, col = point
    half_size = square_size // 2
    row_start = int(max(0, row - half_size))
    row_end = int(min(imageshape[-2], row + half_size + 1))
    col_start = int(max(0, col - half_size))
    col_end = int(min(imageshape[-3], col + half_size + 1))
    
    # Get the maximum value in the square region
    GC_square_region = GCunc[row_start:row_end, col_start:col_end]
    GC_max_val = np.max(GC_square_region)
    GC_min_val = np.min(GC_square_region)
    
    tdTom_square_region = Td1min[row_start:row_end, col_start:col_end]
    tdTom_max_val = np.max(tdTom_square_region)
    tdTom_min_val = np.min(tdTom_square_region)

    fig, axes = plt.subplots(1, 7, figsize=(3*7, 3),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0})  # Reduced wspace
    # Titles for each subplot
    titles = ['GCaMP pre', 'GCaMP uncaging',  'GCaMP F/F0', 'tdTomato pre', 'tdTomato 1 min',
              'tdTomato 0 min', f'tdTomato {minutes_after_uncaging} min']
                                    
    for ax, arr, title in zip(axes, [GCpre, GCunc, GCF_F0,Tdpre, Td1min, td_before, td_after], titles):
        
        if title in titles[0:2]:
            vmin = GC_min_val
            vmax = GC_max_val
        if title in titles[2:]:
            vmin = tdTom_min_val
            vmax = tdTom_max_val
        
        # plt.imshow(GCF_F0, vmin = 1, vmax = 15)
        if title == titles[2]:
            im = ax.imshow(arr, cmap='inferno', vmin = 1, vmax = 10)
            
        else:
            im = ax.imshow(arr, cmap='gray', vmin = vmin, vmax = vmax)
        ax.set_title(title)
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        ax.set_xticklabels([])  # Remove x labels
        ax.set_yticklabels([])  # Remove y labels
        if title in titles[:4]:
            ax.plot(col, row, 'ro', markersize=2)   
    
    fig.suptitle(each_set["unc"], y=1.05, fontsize=14)  # y controls vertical position
    # plt.suptitle('Comparison of 2D Arrays', y=1.05, fontsize=14)  # y controls vertical position
    plt.tight_layout()  # Adjust spacing between subplots
    
    folder = os.path.dirname(each_set["unc"])
    savefolder = os.path.join(folder,"plot")
    os.makedirs(savefolder, exist_ok=True)
    basename = os.path.basename(each_set["unc"])
                    
    savepath = os.path.join(savefolder, basename[:-5] + ".png")
    plt.savefig(savepath, dpi=150, bbox_inches = "tight")
    plt.show()
            
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