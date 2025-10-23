import os
import sys
import glob
sys.path.append("..\\..\\")
from controlflimage_threading import Control_flimage
from flimage_graph_func import plot_max_proj_uncaging
from FLIMageFileReader2 import FileReader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import re
import cv2
from PIL import Image
from time import sleep

ch_1or2 = 2
drug = "Lat B 5 μM"

drug_added_datetime = datetime(2025, 10, 10, 14, 25, 0, 0)

flim_folder = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20251014\LatB"

savefolder = os.path.join(flim_folder, "plot_maxproj_with_time")
os.makedirs(savefolder, exist_ok=True)

filelist = glob.glob(os.path.join(flim_folder, "*highmag_*_[0-9][0-9][0-9].flim"))

grouped_files = defaultdict(list)

pattern = re.compile(r"(.*)_\d{3}\.flim$")

for filepath in filelist:
    filename = os.path.basename(filepath)
    # print(filename)
    match = pattern.search(filename)
    if match:
        group_key = match.group(1)
        if "for_align" in group_key:
            continue
        else:
            grouped_files[group_key].append(filepath)

if True:
    for group, files in grouped_files.items():
        print(f"\nGroup: {group}")


first_file = grouped_files[list(grouped_files.keys())[1]][0]


FLIMageCont = Control_flimage(ini_path=r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini")

for group, each_flim_list in grouped_files.items():
    first = True
    first_file = each_flim_list[0]

    FLIMageCont.flim.sendCommand(f'OpenFile, {first_file}')

    input("click project and sum or max and press enter")

    concat_save_path = first_file[:-8] + "concat.flim"
    FLIMageCont.flim.sendCommand(f'ConcatenateImages, {concat_save_path}')


    sleep(1.5*len(each_flim_list))
    FLIMageCont.flim.sendCommand(f'AlignFrames')
    sleep(0.4*len(each_flim_list))
    FLIMageCont.flim.sendCommand(f'SaveCurrentImage, {concat_save_path}')
    sleep(1)




# ##
# #%%
# brokenfile = r"G:\ImagingData\Tetsuya\20251014\LatB\CS_1_pos1__highmag_1_002_.flim"
# with open(brokenfile, "rb+") as f:
#     data = f.read()
#     new_data = data.replace(b"State.Acq.ZStack = False", 
#                             b"State.Acq.ZStack = True")  # must be same length!
#     f.seek(0)

# filename = r"G:\ImagingData\Tetsuya\20251014\LatB\CS_1_pos1__highmag_1_002.flim"
# with open(filename, "wb+") as f:
#     f.write(new_data)
# # %%
