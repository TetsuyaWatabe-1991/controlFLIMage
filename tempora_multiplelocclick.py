# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:36:49 2023

@author: yasudalab
"""

#assign single or multiple uncaging pos and save them in text file
import numpy as np
from tifffile import imread
from FLIMageAlignment import flim_files_to_nparray
from multidim_tiff_viewer import multiple_uncaging_click
from controlflimage_threading import control_flimage

flimpath = r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\test2_multipos\pos8_highmag_032.flim"
Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray([flimpath],
                                                                   ch=0)
FirstStack=Tiff_MultiArray[0]

z, ylist, xlist = multiple_uncaging_click(FirstStack,SampleImg=None,
                                          ShowPoint=False,ShowPoint_YX=[110,134])
print(z, ylist, xlist)

num_pos = len(ylist)

txtpath = flimpath[:-5]+".txt"

with open(txtpath, 'w') as f:
    f.write(str(num_pos)+'\n')
    for nth_pos in range(num_pos):
        f.write(f'{z},{ylist[nth_pos]},{xlist[nth_pos]}\n')



if False:
    FLIMageCont = control_flimage()    
    FLIMageCont.flim.sendCommand("ClearUncagingLocation")
    
    for nth_pos in range(num_pos):

        FLIMageCont.flim.sendCommand(f"CreateUncagingLocation,{int(xlist[nth_pos])},{int(ylist[nth_pos])}")

        print(f"CreateUncagingLocation,{xlist[nth_pos]},{ylist[nth_pos]}")