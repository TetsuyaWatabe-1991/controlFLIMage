# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:34:13 2024

@author: yasudalab
"""

import os
import numpy as np
import tifffile 


basefolder = r"G:\ImagingData\Tetsuya\20240618\GFPslice_2\tpem"
destfolder = r"G:\ImagingData\Tetsuya\20240618\GFPslice_2\tpem_each"

for eachwell in ["B1_00"]:
    for nth_pos in range(1,9):
        eachmag = ""
        for eachmag in ["", "highzoom_"]:
            for eachch in [1,2]:            
                
                dest_tif_name = f"{eachwell}_{nth_pos}_1_{eachmag}_Ch{eachch}_concat.tif" 
                dest_tif_path = os.path.join(destfolder, dest_tif_name)
                
                miplist = []
                for nthtime in range(1,101):

                    filename = f"{eachwell}_{nth_pos}_1_{eachmag}_Ch{eachch}_{str(nthtime).zfill(3)}.tif"
                    filepath = os.path.join(basefolder, filename)
                    
                    if os.path.exists(filepath):
                        
                        tiffarr = tifffile.imread(filepath)
                        
                        mip = tiffarr.max(axis = 0)
                        
                        miplist.append(mip)
                    # else:
                    #     raise Exception(filepath)
                           
                mip_time_arr = np.array(miplist).astype(np.uint16)
                tifffile.imwrite(dest_tif_path, mip_time_arr, photometric = "minisblack")


if False:
    import numpy as np

    datelist = []
    for eachwell in ["B1_00"]:
        for nth_pos in [1]:
            eachmag = ""
            for eachch in [1]:            
                for nthtime in range(1,101):
    
                    filename = f"{eachwell}_{nth_pos}_1_{eachmag}_Ch{eachch}_{str(nthtime).zfill(3)}.tif"
                    filepath = os.path.join(basefolder, filename)
                    
                    if os.path.exists(filepath):
                        datelist.append(os.path.getmtime(filepath))
    
    
    sec_array = np.array(datelist) -     np.array(datelist)[0]
    min_array = sec_array//60
    
                    
                        
                            
                            
                    
        