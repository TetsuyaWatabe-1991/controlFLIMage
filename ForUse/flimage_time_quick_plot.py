# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:27:19 2024

@author: yasudalab
"""
import os
import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
from FLIMageFileReader2 import FileReader
import numpy as np
import matplotlib.pyplot as plt
import tifffile


for header in ["a","b","c","d","e"]:

    motorposlist = []
    first = True
    for nthacq in range(71):
        #file_path = rf"G:\ImagingData\Tetsuya\20240626\multipos_2\a__highmag_1_{str(nthacq).zfill(3)}.flim"
        file_path = rf"G:\ImagingData\Tetsuya\20240626\multipos_3\{header}_{str(nthacq).zfill(3)}.flim"    
        
        if os.path.exists(file_path)!=True:
            continue
        iminfo = FileReader()
        iminfo.read_imageFile(file_path, True) 
        ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5]))
        maxproj = np.max(ZYXarray,axis=0)
        motorposlist.append(iminfo.statedict['State.Motor.motorPosition'])
        plt.imshow(maxproj,cmap="gray")
        plt.show()
        if first:
            first = False
            tifffile.imwrite(file_path[:-5]+".tif", ZYXarray)
        
    mtrpos_arr = np.array(motorposlist)
    
    relative_pos_arr = mtrpos_arr - mtrpos_arr[0]
    
    plt.plot(relative_pos_arr[:,0],"r*", label = "x")
    plt.plot(relative_pos_arr[:,1],"g.", label = "y")
    plt.plot(relative_pos_arr[:,2],"b+", label = "z")
    plt.ylabel("Displacement (um)")
    plt.xlabel("Time point")
    plt.legend()
    plt.show()
