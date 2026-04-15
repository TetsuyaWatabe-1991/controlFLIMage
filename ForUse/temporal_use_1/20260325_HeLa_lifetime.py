# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:44:15 2022

@author: yasudalab
"""
import sys,pathlib,os,glob
sys.path.append(r"..\..")
from FLIMageFileReader2 import FileReader
import matplotlib.pyplot as plt
import numpy as np

def get_flimfile_list(one_file_path):
    filelist=glob.glob(one_file_path[:-8]+'*.flim')
    return filelist

# If you will not use tkinter, declare path for .flim file.
one_file_path=r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260325\p38_egfp\auto1 - Copy\p38et_5_pos1_002.flim"
filelist=get_flimfile_list(one_file_path)

FourDimList=[]
for file_path in filelist:
    iminfo = FileReader()
    iminfo.read_imageFile(file_path, True) 
    # Get intensity only data
    imagearray=np.array(iminfo.image)
    intensityarray=np.sum(imagearray,axis=-1)
    FourDimList.append(intensityarray)
    
ch=0

intensity_range=[0,20]
for ZYX in FourDimList:
    for i in range(ZYX.shape[0]):
        plt.imshow(intensityarray[i,0,ch,:,:],cmap="gray",
                   vmin=intensity_range[0],vmax=intensity_range[1])
        # plt.text(0,-20,str(i));plt.axis('off')
        plt.text(0,-10,str(i));plt.axis('off')
        plt.show()