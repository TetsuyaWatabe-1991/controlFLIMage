# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:06:27 2024

@author: WatabeT
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:00:38 2023

@author: WatabeT
"""
import sys
sys.path.append(r"C:\Users\WatabeT\Documents\Git\controlFLIMage")
from FLIMageFileReader2 import FileReader
import os,codecs,re,glob
import numpy as np
from datetime import datetime
import tifffile
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

savefolder = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240827\24well_0808GFP\highmag_Trans5ms\tpem\uncaging_img"
folder = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240827\24well_0808GFP\highmag_Trans5ms\tpem"
#"\\RY-LAB-WS04\ImagingData\Tetsuya\20240827\24well_0808GFP\highmag_Trans5ms\tpem\C1_00_5_1__highmag_1_003.flim"
flimfilelist = glob.glob(os.path.join(folder,"*highmag*003.flim"))
flimfilelist.sort(key=os.path.getmtime)

search_list = flimfilelist[:]

# showcurrent = True
num = 0
for file_path in search_list:
    num+=1

    iminfo = FileReader()
    iminfo.read_imageFile(file_path, True) 
    
    # Get intensity only data
    imagearray=np.array(iminfo.image)
    imshape = imagearray.shape
    
    print(imshape)
    if imshape[0]>30 :
        
        for nthind in [1,1,-1]:
            imagearray=np.sum(imagearray,axis=nthind)
        
        imagearray=imagearray[0,:,:]
            
        vmax = np.percentile(imagearray, 99.5)    
        imagearray=np.clip((imagearray / vmax) * 255, 0, 255).astype(np.uint8)
        
        Xpos_list = iminfo.State.Uncaging.UncagingPositionsX
        Ypos_list = iminfo.State.Uncaging.UncagingPositionsY
        
        currentCYuncaging = iminfo.State.Uncaging.Position
        
        savepath = os.path.join(savefolder,f"{os.path.basename(file_path)[:10]}_t_{file_path[-8:-5]}_{num}.png")
        # plt.savefig(savepath, dpi = 72, bbox_inches = 'tight')
        # tifffile.imwrite(savepath, imagearray)
        
        image = Image.fromarray(imagearray).convert("RGB")
        
        draw = ImageDraw.Draw(image)
         
        center_x = int(imshape[-3] * Xpos_list[0])
        center_y = int(imshape[-2] * Ypos_list[0])  
        
        radius = 7
        
        draw.ellipse(
            [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
            outline='yellow', width=2
        )
        
                
        image.save(savepath)
                            
        # print(f"Dendrite information was saved as {txtpath}")