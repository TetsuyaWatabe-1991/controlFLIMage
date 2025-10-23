import os
import sys
import glob
sys.path.append("..\\..\\")
from flimage_graph_func import plot_max_proj_uncaging
from FLIMageFileReader2 import FileReader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

flim_path = r"G:\ImagingData\Tetsuya\20251007\auto1\C1_DMSO_pos1_001.flim"
ch_1or2 = 2

flim_path_list = glob.glob(os.path.join(r"\\RY-LAB-WS04\ImagingData\Tetsuya\20251007\auto1", 
                            "C3_CytD_pos1__highmag_2*[0-9].flim"))
savefolder = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20251007\auto1\plot_maxproj_with_time"
os.makedirs(savefolder, exist_ok=True)

drug = "Cyt D"

drug_added_datetime = datetime(2025, 10, 7, 22, 56, 0, 0)

first = True
for each_flim_path in flim_path_list:

    iminfo = FileReader()
    iminfo.read_imageFile(each_flim_path, True) 
    ch = ch_1or2 - 1
    
    acq_datetime = datetime.strptime(iminfo.acqTime[0], '%Y-%m-%dT%H:%M:%S.%f')
    elapsed_time = acq_datetime - drug_added_datetime
    elapsed_time_min = int(elapsed_time.total_seconds() / 60)
    print(elapsed_time_min)
    hour = elapsed_time_min // 60
    minute = elapsed_time_min % 60
    if elapsed_time_min < 0:
        hour = abs(hour + 1)
        minute = abs(elapsed_time_min)

    imagearray=np.array(iminfo.image)
    
    uncaging_x_y_0to1 = iminfo.statedict["State.Uncaging.Position"]
    center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
    center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]
    
    maxproj = imagearray[:, 0, ch, :,:,:].sum(axis=-1).sum(axis=0)
    
    if first:
        first = False
        first_vmax = maxproj.max() * 0.7

    plt.imshow(maxproj, cmap = 'gray', vmin = 0, vmax = first_vmax)
    plt.axis('off')
    if elapsed_time_min < 0:
        # 00:00
        title_txt = f"-{str(hour).zfill(2)} h {str(minute).zfill(2)} m, pre"
    else:
        title_txt = f"{str(hour).zfill(2)} h {str(minute).zfill(2)} m, {drug}"
    plt.title(title_txt)
   

    basename = os.path.basename(each_flim_path)                
    savepath = os.path.join(savefolder, basename[:-5] + "_maxproj.png")
    plt.savefig(savepath, dpi=150, bbox_inches = "tight")
    print("maxproj_savepath ", savepath)
    
    plt.show()
    plt.close(); plt.clf();plt.close("all");
