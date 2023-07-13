# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:00:38 2023

@author: WatabeT
"""
from FLIMageFileReader2 import FileReader
import os,codecs,re,glob
import numpy as np
from datetime import datetime
import tifffile
import matplotlib.pyplot as plt

folder = r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230712\set1_"

flimfilelist = glob.glob(os.path.join(folder,"pos*.flim"))
flimfilelist.sort(key=os.path.getmtime)

search_list = flimfilelist[:-2]

for file_path in search_list:

    iminfo = FileReader()
    iminfo.read_imageFile(file_path, True) 
    
    # Get intensity only data
    imagearray=np.array(iminfo.image)
    
    imshape = imagearray.shape
    
    if imshape[0]==1 and imshape[1]==1:
        intensityarray=np.sum(imagearray,axis=-1)
        
        # Showing intensity image and lifetime image
        
        ch=1
        Xpos_list = iminfo.State.Uncaging.UncagingPositionsX
        Ypos_list = iminfo.State.Uncaging.UncagingPositionsY
        
        intensity_range=[0,202]
        
        vmax = np.percentile(intensityarray, 99.5)
        
        img = intensityarray[0,0,ch,:,:]
        plt.imshow(img,cmap="gray",
                   vmin=0,vmax=vmax)
        for x_relative, y_relative in zip(Xpos_list,Ypos_list):
            plt.scatter(x = x_relative*img.shape[1],
                        y = y_relative*img.shape[0],
                        s = 50,
                        c = 'y',marker = "+")
            
        plt.title(f"Uncaging - {file_path[-8:-5]}")
        plt.axis('off')
        
        
        savefolder_glob = glob.glob(os.path.join(folder,os.path.basename(file_path)[:4]+"*high*/"))
        if len(savefolder_glob)==1:
            savepath = os.path.join(savefolder_glob[0],f"{os.path.basename(file_path)[:4]}_t_{file_path[-8:-5]}.png")
            plt.savefig(savepath, dpi = 72, bbox_inches = 'tight')
        
        plt.show()
                            
    # print(f"Dendrite information was saved as {txtpath}")