# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 22:22:40 2022

@author: yasudalab
"""

import os
import sys
sys.path.append('../')
import glob
from FLIMageAlignment import flim_files_to_nparray,make_save_folders,get_xyz_pixel_um
import matplotlib.pyplot as plt
import numpy as np


onefilelist=[
            r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230125\PTEN_Slice1_neuron1_dendrite1_stack_054.flim",
            # r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221228\GFPslice3_dendrite6_001.flim",
            # r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221228\GFPslice3_dendrite7_001.flim",
            # r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221228\GFPslice3_dendrite8_001.flim",
            # r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221228\GFPslice3_dendrite9_001.flim",
            ]
bar_um = 1
channel = 0 # 0 or 1
for one_file_path in onefilelist:
    
    saveFolder, EachImgsaveFolder = make_save_folders(one_file_path)
    
    # filelist=glob.glob(one_file_path[:-8]+'*.flim')
    filelist=glob.glob(one_file_path[:-8]+'*.flim')[:-1]
    
    uncaging_filelist=[]
    uncagingpos=[]
    
    Zstack_filelist=[]
    
    for path in filelist:
        Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray([path])    
        if Tiff_MultiArray.shape[1]==1:
            uncaging_filelist.append(path)
            uncagingpos.append(iminfo.State.Uncaging.Position)
        else:
            Zstack_filelist.append(path)
    # filelist = get_flimfile_list(one_file_path)
    
    Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray(uncaging_filelist,ch=channel)
    Tiff_MultiArray_2, iminfo_2, relative_sec_list_2 = flim_files_to_nparray(Zstack_filelist,ch=channel)
    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
    
    vmax=Tiff_MultiArray.max()*0.3
    
    kwargs_ex = {'cmap':'gray', 
                 'vmin':0,
                 'vmax':vmax,
                 'interpolation':'none'}
    kwargs_uncag = dict(c="yellow",marker="+",s=500,lw=4)
    kwargs_bar = dict(c="yellow",lw=5)
    
    uncagingpos=np.array(uncagingpos)
    uncagingpos=uncagingpos*Tiff_MultiArray.shape[3]
    meanX=uncagingpos[:,0].mean()
    meanY=uncagingpos[:,1].mean()
    
    halfwidth=25
    # ylim=[meanY - halfwidth, meanY + halfwidth]
    # xlim=[meanX - halfwidth, meanX + halfwidth]
    ylim=[meanY + halfwidth, meanY - halfwidth]
    xlim=[meanX + halfwidth, meanX - halfwidth]
    
    xbar_pos = [meanX + halfwidth*0.9, meanX + halfwidth*0.9 - bar_um/x_um]
    ybar_pos = [meanY + halfwidth*0.9, meanY + halfwidth*0.9]
    
    for i in range(Tiff_MultiArray.shape[0]):
        image = Tiff_MultiArray[i,0,:,:]
        
        plt.figure()
        plt.imshow(image,**kwargs_ex);
        plt.scatter(uncagingpos[i][0],
                    uncagingpos[i][1],
                    **kwargs_uncag)
        
        
        plt.text(xlim[0],ylim[1],s=f"{str(relative_sec_list[i]+relative_sec_list_2[10]+30).zfill(4)} sec")
        plt.plot(xbar_pos,ybar_pos,**kwargs_bar)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.axis('off')
        
        savepath = os.path.join(EachImgsaveFolder,uncaging_filelist[i][-8:-5]+".png")
        plt.savefig(savepath,dpi=100,transparent=True,bbox_inches='tight')
        plt.show()
          
    for i in range(Tiff_MultiArray_2.shape[0]):
        image = Tiff_MultiArray_2[i,:,:,:].max(axis=0)
        
        plt.figure()
        plt.imshow(image,**kwargs_ex);
        # plt.scatter(uncagingpos[i][0],
        #             uncagingpos[i][1],
        #             **kwargs_uncag)
        plt.text(xlim[0],ylim[1],s=f"{str(relative_sec_list_2[i]).zfill(4)} sec")
        plt.plot(xbar_pos,ybar_pos,**kwargs_bar)    
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.axis('off')
        
        savepath = os.path.join(EachImgsaveFolder,Zstack_filelist[i][-8:-5]+".png")
        plt.savefig(savepath,dpi=100,transparent=True,bbox_inches='tight')
        plt.show()
    
    
        