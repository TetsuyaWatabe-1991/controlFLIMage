# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:04:14 2023

@author: yasudalab
"""
import sys
sys.path.append(r'../')
from FLIMageAlignment import flim_files_to_nparray, get_flimfile_list, mean_square_error,Align_4d_array,plot_alignment_shifts
import numpy as np
import matplotlib.pyplot as plt
import cv2

one_file_path=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221228\GFPslice3_dendrite8_001.flim"
flimfile_list=get_flimfile_list(one_file_path)

ImageArray, iminfo, relative_sec_list  = flim_files_to_nparray(flimfile_list,ch=1)

shifts, Aligned_4d_array=Align_4d_array(ImageArray)
plot_alignment_shifts(shifts,iminfo, saveFolder="",savefigure=False,
                      relative_sec_list=relative_sec_list)


cut_region=np.abs(shifts).astype(int)

threshold,th2 = cv2.threshold(ImageArray[0,:,:,:].max(axis=0),0,65535,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

MSE_threshold = threshold*0.2
print("MSE_threshold, ",MSE_threshold)

plt.imshow(ImageArray[0,:,:,:].max(axis=0) > MSE_threshold)
plt.show()



MSElist=[]
x=[]
for Nth in range(Aligned_4d_array.shape[0]):

    NthCut=cut_region[Nth].copy()
    
    for j in range(3):
        if NthCut[j]==0:
            NthCut[j]=-Aligned_4d_array.shape[j+1]
    
    
    ref_array=Aligned_4d_array[0,
                              NthCut[0]:-NthCut[0],
                              NthCut[1]:-NthCut[1],
                              NthCut[2]:-NthCut[2]]
    
    query_array=Aligned_4d_array[Nth,
                              NthCut[0]:-NthCut[0],
                              NthCut[1]:-NthCut[1],
                              NthCut[2]:-NthCut[2]]
    
    
    
    MSE = mean_square_error(ref_array,query_array,
                            intensity_threshold=MSE_threshold,required_pixel=100)
    MSElist.append(MSE)
    x.append(Nth+1)    
    
    
MaxMSEindex = MSElist.index(max(MSElist))

NthCut=cut_region[MaxMSEindex].copy()
for j in range(3):
    if NthCut[j]==0:
        NthCut[j]=-Aligned_4d_array.shape[j+1]
        
query_array=Aligned_4d_array[MaxMSEindex,
                          NthCut[0]:-NthCut[0],
                          NthCut[1]:-NthCut[1],
                          NthCut[2]:-NthCut[2]]
ref_array=Aligned_4d_array[0,
                          NthCut[0]:-NthCut[0],
                          NthCut[1]:-NthCut[1],
                          NthCut[2]:-NthCut[2]]

vmax=query_array.max()*0.7
f, axarr = plt.subplots(1,2) 
axarr[0].imshow(ref_array.max(axis=0),cmap='gray', vmin=0, vmax=vmax)
axarr[1].imshow(query_array.max(axis=0),cmap='gray', vmin=0, vmax=vmax)
# plt.imshow(Tiff_MultiArray[i,ShowZ,:,:],cmap=cmap, vmin=vmin, vmax=vmax)
savepath = one_file_path[:-4]+f"_MAX_mean_square_error_img_right_{MaxMSEindex}th_.png"
plt.savefig(savepath,dpi=150,transparent=True,bbox_inches='tight')
plt.show()


plt.plot(x,MSElist)
plt.xlabel("NthAcquisition");plt.ylabel("Mean square error");
savepath = one_file_path[:-4]+"_mean_square_error.png"
plt.savefig(savepath,dpi=300,bbox_inches="tight")
plt.show()




