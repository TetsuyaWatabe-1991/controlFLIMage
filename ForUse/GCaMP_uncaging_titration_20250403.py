# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:40:55 2025

@author: yasudalab
"""


import sys
sys.path.append("../")
from controlflimage_threading import Control_flimage
from time import sleep

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

# Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\test.txt"
Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128.txt"
# Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_256_7slices.txt"

direction_ini = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"

if "FLIMageCont" not in globals():
    FLIMageCont = Control_flimage(ini_path=direction_ini)

# FLIMageCont.directionMotorY = FLIMageCont.directionMotorY*1
# FLIMageCont.directionGalvoY= FLIMageCont.directionGalvoY*1

# interval_sec = 60

uncaging_power = [37, 43, 54, 70, 85]


FLIMageCont.flim.sendCommand('State.Files.baseName = "pos1_manual_dend1_spine2_"')
FLIMageCont.flim.sendCommand("State.Files.fileCounter = 1")

for each_pow in uncaging_power:
    
    FLIMageCont.set_uncaging_power(each_pow)
    sleep(0.5)
    

    
    FLIMageCont.flim.sendCommand('StartGrab')
    
    sleep(18)

    
a = FLIMageCont.flim.sendCommand('GetFullFileName')
each_file = a[a.find(",")+2:]
# each_firstfilepath = r"G:\ImagingData\Tetsuya\20250403\B6_cut0319_FlxGC6sTom_0322\highmag_Trans5ms\tpem\B3_00_2_1_dend1_004.flim"

filelist = get_flimfile_list(each_file)
# uncaging_nth_list = []

# # First=True
    
# for each_file in filelist: 

#     uncaging_iminfo = FileReader()
#     uncaging_iminfo.read_imageFile(each_file, True) 
    
#     unc_dt = datetime.fromisoformat(uncaging_iminfo.acqTime[2])
#     imagearray=np.array(uncaging_iminfo.image)
#     uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
#     uncaging_pow = uncaging_iminfo.statedict["State.Uncaging.Power"]
    
#     center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
#     center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]
   
#     GCpre = imagearray[0,0,0,:,:,:].sum(axis = -1)
#     GCunc = imagearray[3,0,0,:,:,:].sum(axis = -1)
#     # Tdpre = imagearray[0,0,1,:,:,:].sum(axis = -1)
#     # Td1min = imagearray[-1,0,1,:,:,:].sum(axis = -1)
#     # Plot each array
    
#     GC_pre_med = median_filter(GCpre, size=3)
#     GC_unc_med = median_filter(GCunc, size=3)
    
    
#     # GCF_F0 = (GCunc/GCpre)
#     GCF_F0 = (GC_unc_med/GC_pre_med)
#     GCF_F0[GC_pre_med == 0] = 0
#     plt.imshow(GCF_F0, cmap = "inferno", vmin = 1, vmax = 10)
#     plt.plot(center_x, center_y, 'ro', markersize=2)   
#     plt.title(f"{uncaging_pow} %")   
    
#     folder = os.path.dirname(each_file)
#     savefolder = os.path.join(folder,"plot")
#     os.makedirs(savefolder, exist_ok=True)
#     basename = os.path.basename(each_file)
                    
#     savepath = os.path.join(savefolder, basename[:-5] + ".png")
#     plt.savefig(savepath, dpi=150, bbox_inches = "tight")
    
#     plt.show()
