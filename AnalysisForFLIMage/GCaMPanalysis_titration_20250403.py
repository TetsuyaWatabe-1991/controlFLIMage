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
each_firstfilepath =r"G:\ImagingData\Tetsuya\20250430\B60331_AAVGC6stdTom0405\dend6_8um_002.flim"

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
    pulseWidth = uncaging_iminfo.statedict["State.Uncaging.pulseWidth"]
    
    center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
    center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]
   
    # GCpre = imagearray[0:8,0,0,:,:,:].sum(axis = -1).sum(axis = 0)
    # GCunc = imagearray[17:25,0,0,:,:,:].sum(axis = -1).sum(axis = 0)
    GCpre = imagearray[0:2,0,0,:,:,:].sum(axis = -1).mean(axis = 0)
    GCunc = imagearray[3:4,0,0,:,:,:].sum(axis = -1).mean(axis = 0)
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
    plt.title(f"{uncaging_pow} %, {pulseWidth} ms")   
    
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
