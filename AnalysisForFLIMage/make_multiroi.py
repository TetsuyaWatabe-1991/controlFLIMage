# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:42:31 2024

@author: yasudalab
"""

import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
import os
import glob
import pandas as pd
import numpy as np
import save_roi
from FLIMageAlignment import get_flimfile_list
from FLIMageFileReader2 import FileReader
# one_of_filepath = r"G:\ImagingData\Tetsuya\20240827\24well_0808GFP\highmag_Trans5ms\tpem2\C1_00_1_2__highmag_1_002.flim"

one_of_filepath = r"G:\ImagingData\Tetsuya\20250415\B6_cut0326_FlxGC6s_tdTomato\highmag_Trans5ms\tpem\C1_00_1_1__highmag_1_042.flim"
one_of_filepath = r"G:\ImagingData\Tetsuya\20250418\B6_cut0326_FlxGC6s_tdTomato0330\highmag_RFP200ms20p\tpem\C5_00_1_1__highmag_1_003.flim"


# ch_1_or_2 = 2

one_of_file_list = glob.glob(os.path.join(
                                os.path.dirname(one_of_filepath),"*_highmag_*002.flim"))


one_of_filepath = r"G:\ImagingData\Tetsuya\20250507\W_harp\lowmag1__highmag_3_087.flim"
# ch_1_or_2 = 2

one_of_file_list = glob.glob(os.path.join(
                                os.path.dirname(one_of_filepath),"*_highmag_*002.flim"))

# one_of_file_list = glob.glob(os.path.join(
#                                 os.path.dirname(one_of_filepath),"A3_00_1_3__highmag_2_*002.flim"))
# one_of_file_list = "G:\ImagingData\Tetsuya\20241212\24well\highmag_GFP200ms55p\tpem_1\A3_00_2_1__highmag_1_004.flim"

# ch = ch_1_or_2 - 1



combined_df = pd.DataFrame()
for each_firstfilepath in one_of_file_list:
# for each_firstfilepath in [one_of_file_list[0]]:

    filelist= get_flimfile_list(each_firstfilepath)
    First=True
    uncaging_dict = dict()
    center_x_list = []
    center_y_list = []
    radius_list = []
    for file_path in filelist:
        iminfo = FileReader()
        print(file_path)
        try:
            iminfo.read_imageFile(file_path, True) 
            imagearray=np.array(iminfo.image)
        except:
            print("\n\ncould not read\n")
            continue
           
        if First:
            First=False
            imageshape=imagearray.shape
    
        if imagearray.shape == imageshape:
            pass
        else:
            if (imagearray.shape[0] > 29):
                print(file_path,'<- uncaging')
                uncaging_iminfo = iminfo
                
                uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
                center_y_list.append(imageshape[-2] * uncaging_x_y_0to1[1])
                center_x_list.append(imageshape[-3] * uncaging_x_y_0to1[0])
                radius_list.append(5)
                
                
    if len(center_x_list)>0:
        roi_text = save_roi.make_multi_circle_roi(center_x_list,
                                                  center_y_list,
                                                  radius_list
                                                  )
        save_roi.save_roi_textfiles(flim_path = each_firstfilepath,
                                    roi_text = roi_text)
    else:
        print("No uncaging. Skip this file set.")