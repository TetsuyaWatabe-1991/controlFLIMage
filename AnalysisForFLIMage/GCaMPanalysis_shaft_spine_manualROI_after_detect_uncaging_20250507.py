# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:46:20 2025

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
from skimage.draw import polygon
import re
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')

# one_of_filepath = r"G:\ImagingData\Tetsuya\20250506\onHarp\E4_roomair_lowmag_1__highmag_1_002.flim"

# one_of_file_list = glob.glob(os.path.join(
#                                 os.path.dirname(one_of_filepath),"*highmag_*002.flim"))

one_of_filepath = r"G:\ImagingData\Tetsuya\20250508\LTP2_13um_001.flim"


one_of_file_list = glob.glob(one_of_filepath[:one_of_filepath.rfind("\\")]+"\\LTP*002.flim")

pow_slope = 0.210
pow_intcpt = 0.1145

## Not beautiful, but acceptable...
## 
combined_df = pd.DataFrame()
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
   
# Example: print grouped results

result_txt = ""
result_txt += "{basename},{pow_mw_round},{pre_mean_intensity}, {post_mean_intensity}, {dend_F_F0}, {pre_spine_intensity}, {post_spine_intensity}, {spine_F_F0}".replace("{","").replace("}","")   + "\n"

    
for each_set in pre_post_set_list: 
    
    re_define_roi = True
    
    while re_define_roi:
        
        each_file = each_set["unc"]
    
        uncaging_iminfo = FileReader()
        uncaging_iminfo.read_imageFile(each_file, True) 
        
        unc_dt = datetime.fromisoformat(uncaging_iminfo.acqTime[2])
        imagearray=np.array(uncaging_iminfo.image)
        uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
        uncaging_pow = uncaging_iminfo.statedict["State.Uncaging.Power"]
        pulseWidth = uncaging_iminfo.statedict["State.Uncaging.pulseWidth"]
        center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
        center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]
       
        try:
            # GCpre = imagearray[0:8,0,0,:,:,:].sum(axis = -1).sum(axis = 0)
            # GCunc = imagearray[17:25,0,0,:,:,:].sum(axis = -1).sum(axis = 0)
            GCpre = imagearray[0,0,0,:,:,:].sum(axis = -1)
            GCunc = imagearray[3,0,0,:,:,:].sum(axis = -1)
            Tdpre = imagearray[0,0,1,:,:,:].sum(axis = -1)
            Td1min = imagearray[-1,0,1,:,:,:].sum(axis = -1)
        except:
            print("\n\n --------------------------------------")
            print("could not read image: ")
            print(each_file)
            print("\n\n --------------------------------------")
            continue
            
        # Plot each array
        
        GC_pre_med = median_filter(GCpre, size=3)
        GC_unc_med = median_filter(GCunc, size=3)
        
        # GCF_F0 = (GCunc/GCpre)
        GCF_F0 = (GC_unc_med/GC_pre_med)
        GCF_F0[GC_pre_med == 0] = 0
    
        # plt.title(f"Depth {depth} \u03BCm   {uncaging_pow} %, {pulseWidth} ms")  \

        
        pow_mw = pow_slope * uncaging_pow + pow_intcpt
        pow_mw_coherent = pow_mw/3
        pow_mw_round = round(pow_mw_coherent,1)
            
            
        fig, ax = plt.subplots()
        ax.imshow(Tdpre, cmap='gray')
        ax.plot(center_x, center_y, 'ro', markersize=2) 
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')
        except:
            pass
        ax.set_title("shaft")
    
        # Let user draw a polygon
        roi_points = plt.ginput(n=4, timeout=0)  # unlimited points, finish by closing window
        
        # Step 2: Convert ROI points to NumPy arrays
        roi_points = np.array(roi_points)
        r = roi_points[:, 0] 
        c = roi_points[:, 1] 
        ax.plot(list(r) + [list(r)[0]], 
                   list(c) + [list(c)[0]], "m", lw = 2)
        
        ax.set_title("SPINE")
    
        roi_points_spine = plt.ginput(n=4, timeout=0)  # unlimited points, finish by closing window
        roi_points_spine = np.array(roi_points_spine)
        
        r_spine = roi_points_spine[:, 0]  # y-coordinates
        c_spine = roi_points_spine[:, 1]  # x-coordinates
        ax.plot(list(r_spine) + [list(r_spine)[0]], 
                   list(c_spine) + [list(c_spine)[0]],"c", lw = 2)
        
        
        # Step 3: Create a mask from the polygon
        mask_rows, mask_cols = polygon(c, r, GC_pre_med.shape)
        mask = np.zeros(GC_pre_med.shape, dtype=bool)
        mask[mask_rows, mask_cols] = True
        
        # Step 3: Create a mask from the polygon
        mask_rows_spine, mask_cols_spine = polygon(c_spine, r_spine, GC_pre_med.shape)
        mask_spine = np.zeros(GC_pre_med.shape, dtype=bool)
        mask_spine[mask_rows_spine, mask_cols_spine] = True
                    
        fig.set_size_inches(4, 4)
        plt.text(.05, .95, 'Shaft', c="m", size=10,  ha='left', va='top', transform=ax.transAxes)
        plt.text(.05, .85, 'Spine', c="c", size=10,  ha='left', va='top', transform=ax.transAxes)

        ax.set_title("ROI")
        ax.axis("off")
        
        folder = os.path.dirname(each_file)
        savefolder = os.path.join(folder,"plot")
        os.makedirs(savefolder, exist_ok=True)
        basename = os.path.basename(each_file)
                        
        savepath = os.path.join(savefolder, basename[:-5] + "_roi.png")
        plt.savefig(savepath, dpi=150, bbox_inches = "tight")
        plt.close()
            
    
            
        # Step 4: Measure mean intensity inside the ROI
        pre_mean_intensity = round(GC_pre_med[mask].sum(),1)
        post_mean_intensity = round(GC_unc_med[mask].sum(),1)
        dend_F_F0 = post_mean_intensity/pre_mean_intensity
        
        print(pre_mean_intensity)
        print(post_mean_intensity)
        
        
        # Step 4: Measure mean intensity inside the ROI
        pre_spine_intensity = round(GC_pre_med[mask_spine].sum(),1)
        post_spine_intensity = round(GC_unc_med[mask_spine].sum(),1)
        spine_F_F0 = post_spine_intensity/pre_spine_intensity
        
        print(pre_spine_intensity)
        print(post_spine_intensity)
    
        
        while True:
            yn = input("next file? (y/n) ")
            if yn in ["Y","y"]:
                re_define_roi = False
                result_txt += f"{basename},{pow_mw_round},{pre_mean_intensity}, {post_mean_intensity}, {dend_F_F0}, {pre_spine_intensity}, {post_spine_intensity}, {spine_F_F0}"   + "\n"
                break

            elif yn in ["N","n"]:
                re_define_roi = True
                break
            else:
                continue
        
        
        
# break

# fig, ax = plt.subplots()
# ax.imshow(GC_pre_med, cmap='gray')
# ax.plot(center_x, center_y, 'ro', markersize=2) 
# mng = plt.get_current_fig_manager()
# try:
#     mng.window.state('zoomed')
# except:
#     pass
# ax.set_title("shaft")
# plt.imshow(GC_pre_med)
# roi_points = plt.ginput(n=4, timeout=0)
# plt.show()



res_save_path = os.path.join(savefolder, "result.csv")
with open(res_save_path, "w") as file:
    # Write the text to the file
    file.write(result_txt)



print(result_txt)

print("save as ")

print(res_save_path)