# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:19:25 2025

@author: yasudalab
"""


# if False:
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# csvpath = r"G:\ImagingData\Tetsuya\20250423\B6GC6sTom0331\tpem\plot\result_2.csv"

csvpath = r"G:\ImagingData\Tetsuya\20250508\plot\result_titration_Copy.csv"
df = pd.read_csv(csvpath)

spine_F_F0_max = 15
shaft_F_F0_max = 15

spine_ylim_max = spine_F_F0_max * 1.1
shaft_ylim_max = shaft_F_F0_max * 1.1

xlim = [0,40]

df.loc[df[df[' spine_F_F0']>spine_F_F0_max].index, ' spine_F_F0'] = spine_F_F0_max
df.loc[df[df[' dend_F_F0']>shaft_F_F0_max].index, ' dend_F_F0'] = shaft_F_F0_max

for each_pow in df['pow_mw_round'].unique():
    each_df = df[df['pow_mw_round'] == each_pow]
    plt.figure(figsize = (4,2))
    sns.regplot(x = "um",
                y = " dend_F_F0",
               data = each_df)
    plt.ylim([0,shaft_ylim_max])
    plt.xlim(xlim)
    plt.ylabel("Shaft F/F0")
    plt.xlabel("Target depth (\u03BCm)")
    plt.title(f"Uncaging power: {each_pow} mW")
    
    savepath = csvpath[:-5]+f"linreg_shaft_pow_{each_pow}.png"
    
    plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
    plt.show()



    plt.figure(figsize = (4,2))
    sns.regplot(x = "um",
                y = ' spine_F_F0',
                data = each_df)
    plt.ylim([0,spine_ylim_max])
    plt.xlim(xlim)
    plt.ylabel("Spine F/F0")
    plt.xlabel("Target depth (\u03BCm)")
    plt.title(f"Uncaging power: {each_pow} mW")
    
    savepath = csvpath[:-5]+f"linreg_spine_pow_{each_pow}.png"
    # savepath = csvpath[:-5]+f"linreg_shaft_pow_{each_pow}.png"
    
    plt.savefig(savepath, dpi = 150, bbox_inches = "tight")
    plt.show()

unique_powers = sorted(df['pow_mw_round'].unique())
ncols = len(unique_powers)

fig, axes = plt.subplots(
    nrows=2,
    ncols=ncols,
    figsize=(3 * ncols, 4),
    sharey='row',
    sharex='col'
)

for i, each_pow in enumerate(unique_powers):
    each_df = df[df['pow_mw_round'] == each_pow]

    # Shaft (bottom row)
    sns.regplot(
        x="um",
        y=" dend_F_F0",
        data=each_df,
        ax=axes[1, i]
    )
    axes[1, i].set_ylim([0, shaft_ylim_max])
    axes[1, i].set_xlim(xlim)
    axes[1, i].set_xlabel("Target depth (μm)")
    if i == 0:
        axes[1, i].set_ylabel("Shaft F/F0")
    else:
        axes[1, i].set_ylabel("")

    # Spine (top row)
    sns.regplot(
        x="um",
        y=" spine_F_F0",
        data=each_df,
        ax=axes[0, i]
    )
    axes[0, i].set_ylim([0, spine_ylim_max])
    axes[0, i].set_xlim(xlim)
    axes[0, i].set_title(f"{each_pow} mW")
    if i == 0:
        axes[0, i].set_ylabel("Spine F/F0")
    else:
        axes[0, i].set_ylabel("")
    axes[0, i].set_xlabel("")

plt.tight_layout()
savepath = csvpath[:-5] + "_tiled_plot.png"
plt.savefig(savepath, dpi=150, bbox_inches="tight")
plt.show()

# from controlflimage_threading import control_flimage
# from time import sleep

# import sys
# sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
# from datetime import datetime
# import os
# import glob
# import pandas as pd
# import numpy as np
# from FLIMageAlignment import get_flimfile_list
# from FLIMageFileReader2 import FileReader
# import matplotlib.pyplot as plt
# from scipy.ndimage import median_filter
# from skimage.draw import polygon
# import re
# from collections import defaultdict

# filelist = glob.glob(r"G:\ImagingData\Tetsuya\20250423\B6GC6sTom0331\tpem\pos*.flim")

# grouped_files = defaultdict(list)

# pattern = re.compile(r"(.*)_\d{3}\.flim$")

# for filepath in filelist:
#     filename = os.path.basename(filepath)
    
#     if filename.count("pos1_dend4_spine1") > 0:
#         print(filename, " <-- rejected")
#         continue
        
    
#     # print(filename)
#     match = pattern.search(filename)
    
    
#     if match:
#         group_key = match.group(1)
#         grouped_files[group_key].append(filepath)
# # Example: print grouped results

# result_txt = ""
# result_txt += "{group},{pow_mw_round},{pre_mean_intensity}, {post_mean_intensity}, {dend_F_F0}, {pre_spine_intensity}, {post_spine_intensity}, {spine_F_F0}".replace("{","").replace("}","")   + "\n"

# for group, files in grouped_files.items():
#     # print(f"\nGroup: {group}")
#     First = True
#     for each_file in files:
#         # print(f"  {each_file}")
                 
#         basename = os.path.basename(each_file)
#         # depth = basename[16 : basename.find("um")]
#         # continue
    
#         uncaging_iminfo = FileReader()
#         uncaging_iminfo.read_imageFile(each_file, True) 
        
#         unc_dt = datetime.fromisoformat(uncaging_iminfo.acqTime[2])
#         imagearray=np.array(uncaging_iminfo.image)
#         uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
#         uncaging_pow = uncaging_iminfo.statedict["State.Uncaging.Power"]
#         pulseWidth = uncaging_iminfo.statedict["State.Uncaging.pulseWidth"]
        
#         xyz = uncaging_iminfo.statedict["State.Motor.motorPosition"]
#         z_um = xyz[-1]
#         depth = round(1317 - z_um)
        
#         print(depth)
#         continue
       
#         center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
#         center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]
       
#         try:
#             # GCpre = imagearray[0:8,0,0,:,:,:].sum(axis = -1).sum(axis = 0)
#             # GCunc = imagearray[17:25,0,0,:,:,:].sum(axis = -1).sum(axis = 0)
#             GCpre = imagearray[0,0,0,:,:,:].sum(axis = -1)
#             GCunc = imagearray[3,0,0,:,:,:].sum(axis = -1)
#             Tdpre = imagearray[0,0,1,:,:,:].sum(axis = -1)
#             Td1min = imagearray[-1,0,1,:,:,:].sum(axis = -1)
#         except:
#             print("\n\n --------------------------------------")
#             print("could not read image: ")
#             print(each_file)
#             print("\n\n --------------------------------------")
            
#             continue
            
            
#         GC_pre_med = median_filter(GCpre, size=3)
#         GC_unc_med = median_filter(GCunc, size=3)
        
#         # GCF_F0 = (GCunc/GCpre)
#         GCF_F0 = (GC_unc_med/GC_pre_med)
#         GCF_F0[GC_pre_med == 0] = 0
#         # 
#         pow_mw = 0.202 * uncaging_pow + 0.277
#         pow_mw_coherent = pow_mw/3
#         pow_mw_round = round(pow_mw_coherent,1)
            
#         # GCF_F0 = (GCunc/GCpre)
#         GCF_F0 = (GC_unc_med/GC_pre_med)
#         GCF_F0[GC_pre_med == 0] = 0
#         plt.imshow(GCF_F0, cmap = "inferno", vmin = 1, vmax = 10)
#         plt.plot(center_x, center_y, 'ro', markersize=2)   
#         plt.title(f"Depth {depth} \u03BCm   {pow_mw_round} mW, {pulseWidth} ms")  
#         plt.axis("off")
#         folder = os.path.dirname(each_file)
#         savefolder = os.path.join(folder,"plot")
#         os.makedirs(savefolder, exist_ok=True)
#         basename = os.path.basename(each_file)
                        
#         savepath = os.path.join(savefolder, basename[:-5] + ".png")
#         plt.savefig(savepath, dpi=150, bbox_inches = "tight")
        
#         plt.show()
        
