# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:50:06 2024

@author: yasudalab
"""

import os

def make_circle_roi(center_x, center_y, radius):
    """
    center_x: pixel
    center_y: pixel
    radius: pixel
    
    returns text
    """
    
    text1 = """SelectROI
ROI-type,Elipsoid,Roi-ID,0,polyLineRadius,8
"""
    text2 = """bgROI
ROI-type,Rectangle,Roi-ID,0,polyLineRadius,8
Rect,0,0,0,0
MultiROI
ROI-type,Elipsoid,Roi-ID,0,polyLineRadius,8
"""

    ROI = f"Rect,{center_x - radius},{center_y - radius},{radius*2},{radius*2}\n"
    
    roi_text = text1 + ROI + text2 + ROI
    return roi_text

def save_roi_textfiles(flim_path : str, 
                       roi_text : str) -> None:
    
     parent_folder = os.path.dirname(flim_path)
     ROI_folder = os.path.join(parent_folder,r"Analysis\ROI")
     os.makedirs(ROI_folder, exist_ok=True)
     ROI_filename = os.path.basename(flim_path)[:-5]+"_ROI.txt"
     ROI_savepath = os.path.join(ROI_folder, ROI_filename)
     
     with open(ROI_savepath, 'w') as f:
         f.write(roi_text)

if __name__ =="__main__":
    flimpath = r"G:\ImagingData\Tetsuya\20241213\24well\highmagGFP200ms55p\tpem_1\C1_00_2_1__highmag_2_002.flim"
    roi_text = make_circle_roi(10,10,5)
    save_roi_textfiles(flimpath, roi_text)
