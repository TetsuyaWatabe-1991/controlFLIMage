# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 19:39:44 2023

@author: yasudalab
"""


import os, sys, glob
sys.path.append(r"C:\Users\WatabeT\Documents\Git\controlFLIMage")

import numpy as np
import matplotlib.pyplot as plt
from FLIMageFileReader2 import  FileReader

ch=0
intensity_range=[0,202]

for nthspine in range(3,7):

    one_of_path=r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230717\GFP_neuron1_spine"+str(nthspine)+"_001.flim"
    
    savefolder = one_of_path[:-8]
    
    filelist = sorted(glob.glob(f"{one_of_path[:-8]}*.flim"), key=os.path.getmtime)
    
    for file_path in filelist[:-1]:
        # file_path = f"{one_of_path[:-8]}{str(i).zfill(3)}.flim"
        
        savefolder = file_path[:-8]
        os.makedirs(savefolder, exist_ok=True)
        
        iminfo = FileReader()
        iminfo.read_imageFile(file_path, True) 
        
        # Get intensity only data
        imagearray=np.array(iminfo.image)
        intensityarray=np.sum(imagearray,axis=-1)
        maxproj = np.max(intensityarray,axis=0)
        
        maxproj_singlech = np.max(intensityarray,axis=0)[0,ch,:,:]
        vmax = np.percentile(maxproj_singlech,99.5)
        plt.imshow(maxproj_singlech, vmin=0, vmax=vmax, cmap='gray')
        
        numstr = file_path[-8:-5]
        plt.title(numstr)
        plt.axis('off')
        savepath = os.path.join(savefolder,f"{numstr}.png")
        plt.savefig(savepath, dpi = 72, bbox_inches = 'tight')
        plt.close;plt.clf();
    

# Showing intensity image and lifetime image
# for i in range(intensityarray.shape[0]):
#     plt.imshow(intensityarray[i,0,ch,:,:],cmap="gray",
#                vmin=intensity_range[0],vmax=intensity_range[1])
#     plt.text(0,-10,str(i));plt.axis('off')
#     plt.show()




# for i in range(intensityarray.shape[0]):
#     iminfo.calculatePage(page = i, fastZpage = 0, channel = 0, 
#                   lifetimeRange = [5, 62], intensityLimit = [2, 100], 
#                   lifetimeLimit = [1.6, 2.8], lifetimeOffset = 1.6)    
#     plt.imshow(iminfo.rgbLifetime)
#     plt.text(0,-10,str(i));plt.axis('off')
#     plt.show()



