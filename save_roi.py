# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:50:06 2024

@author: yasudalab
"""

import os

def get_roi_path(flim_path):
    parent_folder = os.path.dirname(flim_path)
    ROI_folder = os.path.join(parent_folder,r"Analysis\ROI")
    os.makedirs(ROI_folder, exist_ok=True)
    ROI_filename = os.path.basename(flim_path)[:-5]+"_ROI.txt"
    ROI_path = os.path.join(ROI_folder, ROI_filename)
    return ROI_path

def read_roi(flim_path):
    ROI_path = get_roi_path(flim_path)
    
    with open(ROI_path, 'r') as f:
        roi_text_list = f.read().splitlines()
    
    selected_roi_text = roi_text_list[2]
    
    selected_roi_nth = -1

    roi_dict = {}
    nth_roi = 0
    for nth_line in range(8,len(roi_text_list),2):
        nth_roi += 1
        each_roi_dict = {}
        each_roi_dict["rad1"] = float(roi_text_list[nth_line].split(",")[-1])/2
        each_roi_dict["rad2"] = float(roi_text_list[nth_line].split(",")[-2])/2
        each_roi_dict["radius"] = (each_roi_dict["rad1"] + each_roi_dict["rad2"] )/2
        each_roi_dict["center_x"] = float(roi_text_list[nth_line].split(",")[1]) + each_roi_dict["rad1"]
        each_roi_dict["center_y"] = float(roi_text_list[nth_line].split(",")[2]) + each_roi_dict["rad2"]
        roi_dict[nth_roi] = each_roi_dict    
        if roi_text_list[nth_line] == selected_roi_text:
            selected_roi_nth = nth_roi
    return roi_dict, selected_roi_nth
        

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


def make_multi_circle_roi(center_x_list, 
                          center_y_list, 
                          radius_list, 
                          select_ROI_1toN = 1):
    assert len(center_x_list) == len(center_y_list)
    assert len(center_x_list) == len(radius_list)
    
    text_1 = "SelectROI\n"
    text_2 = """bgROI
ROI-type,Rectangle,Roi-ID,0,polyLineRadius,8
Rect,0,0,0,0
MultiROI
"""
    text_3 = "ROI-type,Elipsoid,Roi-ID,0,polyLineRadius,8\n"
    
    
    print("len(center_x_list)", len(center_x_list))
    print("select_ROI_1toN0, ",select_ROI_1toN)
    selectedROI = f"Rect,{center_x_list[select_ROI_1toN -1] - radius_list[select_ROI_1toN -1]},{center_y_list[select_ROI_1toN -1] - radius_list[select_ROI_1toN -1]},{radius_list[select_ROI_1toN -1]*2},{radius_list[select_ROI_1toN -1]*2}\n"
    roi_text = text_1 + selectedROI + text_2
    
    for center_x, center_y, radius in zip(center_x_list, 
                                          center_y_list, 
                                          radius_list):   
        
        eachROI = f"Rect,{center_x - radius},{center_y - radius},{radius*2},{radius*2}\n"
        roi_text += text_3 + eachROI
        
    return roi_text


def save_roi_textfiles(flim_path : str, 
                       roi_text : str) -> None:
     ROI_savepath = get_roi_path(flim_path)
     with open(ROI_savepath, 'w') as f:
         f.write(roi_text)

if __name__ =="__main__":
    flim_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250104\24well\highmag_Trans5ms\tpem\A2_00_1_1__highmag_1_002.flim"
    # roi_text = make_circle_roi(10,10,5)
    # save_roi_textfiles(flimpath, roi_text)
    roi_dict, selected_roi_nth = read_roi(flim_path)
    
