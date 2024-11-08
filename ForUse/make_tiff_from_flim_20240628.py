# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 08:34:34 2024

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


        
    for onetwothree in range(1,4):
        first = True
        for nthacq in range(71):
            #file_path = rf"G:\ImagingData\Tetsuya\20240626\multipos_2\a__highmag_1_{str(nthacq).zfill(3)}.flim"
            file_path = rf"G:\ImagingData\Tetsuya\20240626\multipos_3\{header}__highmag_{onetwothree}_{str(nthacq).zfill(3)}.flim"    
            
            if os.path.exists(file_path)!=True:
                continue
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, True) 
            ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5]))
            maxproj = np.max(ZYXarray,axis=0)
            
            plt.imshow(maxproj,cmap="gray")
            plt.show()
            if first:
                first = False
                tifffile.imwrite(file_path[:-5]+".tif", ZYXarray)
                break