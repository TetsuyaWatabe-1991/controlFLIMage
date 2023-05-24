# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:26:05 2023

@author: yasudalab
"""
from FLIMageFileReader2 import FileReader
import os
import numpy as np
import cv2  
import glob    
import matplotlib.pyplot as plt
import pandas as pd
 
# If you will not use tkinter, declare path for .flim file.
file_path=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230508\HEK_mCherry_004.flim"


df = pd.DataFrame()
filelist = glob.glob(r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230508\*.flim")

for file_path in filelist:
    iminfo = FileReader()
    iminfo.read_imageFile(file_path, True) 
    
    # Get intensity only data
    imagearray=np.array(iminfo.image)
    intensityarray=np.sum(imagearray,axis=-1)
    Ch1_array = intensityarray[0,0,0,:,:]
    Ch2_array = intensityarray[0,0,1,:,:]
    
    ForBinary = (Ch1_array+Ch2_array)
    ForBinary8bit = (255*(ForBinary/ForBinary.max())).astype(np.uint8)
    
    ret, thresh1 = cv2.threshold(ForBinary8bit, 0, 255, cv2.THRESH_BINARY + 
                                                cv2.THRESH_OTSU)    
    
    # Creating kernel
    kernel = np.ones((11, 11), np.uint8)
    erode_image = cv2.dilate(thresh1, kernel) 
    
    BG_ch1 = np.mean(Ch1_array[erode_image==0])
    BG_ch2 = np.mean(Ch2_array[erode_image==0])
    
    Ch1intensity = Ch1_array[thresh1>0].mean() - BG_ch1
    Ch2intensity = Ch2_array[thresh1>0].mean() - BG_ch2
    Ch1over2 = Ch1intensity/Ch2intensity
    
    eachdf = pd.DataFrame(data={
                        "file_path":[file_path],
                        "BG_ch1":[BG_ch1],
                        "BG_ch2":[BG_ch2],
                        "Ch1intensity":[Ch1intensity],
                        "Ch2intensity":[Ch2intensity],
                        "Ch1over2":[Ch1over2]
                        })
    df = pd.concat([df, eachdf])
    f, axarr = plt.subplots(1,2) 
    axarr[0].imshow(ForBinary8bit,cmap="gray")
    axarr[1].imshow(thresh1,cmap="gray")
    f.savefig(file_path[:-5]+".png")
    f.show()
df.to_csv(r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230508\summary.csv")