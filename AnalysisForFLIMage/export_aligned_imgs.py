# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:11:52 2023

@author: WatabeT
"""

import os, sys, glob
sys.path.append(r"C:\Users\WatabeT\Documents\Git\controlFLIMage")

import numpy as np
import matplotlib.pyplot as plt
from FLIMageFileReader2 import  FileReader

ch=0
# intensity_range=[0,202]

file_path = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230718\set1\pos4_high_aligned.flim"
# "\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230717\GFP_neuron1_spine6_aligned.flim"
# "\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230717\GFP_neuron1_spine5_aligned.flim"
# "\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230717\GFP_neuron1_spine4_aligned.flim"
# "\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230717\GFP_neuron1_spine3_aligned.flim"
# "\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230717\GFP_neuron1_spine2_aligned.flim"

savefolder = file_path[:-8]
os.makedirs(savefolder, exist_ok=True)

iminfo = FileReader()
iminfo.read_imageFile(file_path, True) 

# Get intensity only data
imagearray=np.array(iminfo.image)
intensityarray=np.sum(imagearray,axis=-1)

vmax = np.percentile(intensityarray,99)
kwargs = {"vmin":0, 
          "vmax":vmax, 
          "cmap":'gray'}

for nthframe in range(intensityarray.shape[0]):
    eachtime_singlech = intensityarray[nthframe,0,ch,:,:]
    plt.imshow(eachtime_singlech,**kwargs)
    
    numstr = str(nthframe).zfill(3)
    plt.title(numstr)
    plt.axis('off')
    savepath = os.path.join(savefolder,f"{numstr}.png")
    plt.savefig(savepath, dpi = 72, bbox_inches = 'tight')
    plt.close;plt.clf();


savefolder = file_path[:-8]+"_leftFirst"
os.makedirs(savefolder, exist_ok=True)

for nthframe in range(intensityarray.shape[0]):
    
    
    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2) 
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    firsttime_singlech = intensityarray[0,0,ch,:,:]
    axarr[0].imshow(firsttime_singlech,**kwargs)
    
    eachtime_singlech = intensityarray[nthframe,0,ch,:,:]
    axarr[1].imshow(eachtime_singlech,**kwargs)
    
    # eachtime_singlech = intensityarray[nthframe,0,ch,:,:]
    # plt.imshow(eachtime_singlech,kwargs)
    
    xpos = intensityarray.shape[-1]/2
    ypos = -intensityarray.shape[-2]/20
    numstr = str(nthframe).zfill(3)
    axarr[0].text(xpos,ypos,"001",ha='center',va='bottom')
    axarr[0].axis('off')
    axarr[1].text(xpos,ypos,numstr,ha='center',va='bottom')
    axarr[1].axis('off')
    savepath = os.path.join(savefolder,f"{numstr}.png")
    plt.savefig(savepath, dpi = 72, bbox_inches = 'tight')
    plt.close;plt.clf();



