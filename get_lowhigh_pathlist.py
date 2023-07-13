# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:03:41 2023

@author: yasudalab
"""


import os, glob
from multidim_tiff_viewer import multiple_uncaging_click_savetext, dend_props_forEach
from FLIMageAlignment import plot_maxproj


basefolder = r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230712\set2"

pathlist = []

for posnum in range(8):
    lowhigh = []

    for filenum in range(20):
        lowpathlist = glob.glob(os.path.join(basefolder,f"*pos{posnum}*low*{str(filenum).zfill(3)}.flim"))
        if len(lowpathlist)==1:
            print(lowpathlist[0])
            lowhigh.append(lowpathlist[0])
            break

    for filenum in range(20):
        lowpathlist = glob.glob(os.path.join(basefolder,f"*pos{posnum}*high*{str(filenum).zfill(3)}.flim"))
        if len(lowpathlist)==1:
            print(lowpathlist[0])
            lowhigh.append(lowpathlist[0])
            break
    if len(lowhigh)==2:
        pathlist.append(lowhigh)
    elif len(lowhigh)>0:
        print('something wrong.....', lowhigh)

print(pathlist) 
        
    
    
    
if True:
    ch1or2=2
    nth=-1
    for eachlowhigh in pathlist:
        nth+=1

        # if nth not in [3]:
        #     continue
        
        multiple_uncaging_click_savetext(eachlowhigh[1], ch1or2=ch1or2)
        dend_props_forEach(eachlowhigh[1], ch1or2=ch1or2, square_side_half_len = 30, plot_img=True)
        
        plot_maxproj(eachlowhigh[0], ch1or2=ch1or2)
        
    print(pathlist) 
