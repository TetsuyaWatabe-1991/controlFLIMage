# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 22:22:40 2022

@author: yasudalab
"""

import os
import sys
sys.path.append('../')
import glob
from FLIMageAlignment import flim_files_to_nparray,make_save_folders
import matplotlib.pyplot as plt
import numpy as np


one_file_path=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221228\GFPslice3_dendrite8_001.flim"
saveFolder, EachImgsaveFolder = make_save_folders(one_file_path)

filelist=glob.glob(one_file_path[:-8]+'*.flim')
uncaging_filelist=[]
uncagingpos=[]

for path in filelist:
    Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray([path])    
    if Tiff_MultiArray.shape[1]==1:
        uncaging_filelist.append(path)
        uncagingpos.append(iminfo.State.Uncaging.Position)
# filelist = get_flimfile_list(one_file_path)

Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray(uncaging_filelist)

vmax=Tiff_MultiArray.max()*0.3

kwargs_ex = {'cmap':'gray', 
             'vmin':0,
             'vmax':vmax,
             'interpolation':'none'}
kwargs_uncag = dict(c="yellow",marker="+",s=500,lw=4)

uncagingpos=np.array(uncagingpos)
uncagingpos=uncagingpos*Tiff_MultiArray.shape[3]
meanX=uncagingpos[:,0].mean()
meanY=uncagingpos[:,1].mean()

halfwidth=25
ylim=[meanY - halfwidth, meanY + halfwidth]
xlim=[meanX - halfwidth, meanX + halfwidth]

for i in range(Tiff_MultiArray.shape[0]):
    image = Tiff_MultiArray[i,0,:,:]
    
    plt.figure()
    plt.imshow(image,**kwargs_ex);
    plt.scatter(uncagingpos[i][0],
                uncagingpos[i][1],
                **kwargs_uncag)
    
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.axis('off')
    
    savepath = os.path.join(EachImgsaveFolder,uncaging_filelist[i][-8:-5]+".png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
      