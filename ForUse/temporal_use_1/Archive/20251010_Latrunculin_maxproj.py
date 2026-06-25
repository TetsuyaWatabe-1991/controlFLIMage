import os
import sys
import glob
sys.path.append("..\\..\\")
from flimage_graph_func import plot_max_proj_uncaging
from FLIMageFileReader2 import FileReader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import re
import cv2
from PIL import Image

#make movie from each_savefolder
def create_movie_from_pngs(folder_path, output_path, fps=2):
    """
    Create movie from PNG files
    
    Args:
        folder_path: Path to folder containing PNG files
        output_path: Output movie file path
        fps: Frame rate (default: 2)
    """
    # Get PNG files and sort them
    png_files = glob.glob(os.path.join(folder_path, "*.png"))
    png_files.sort()  # Sort by filename
    
    if not png_files:
        print(f"No PNG files found in {folder_path}")
        return
    
    # Get dimensions from first image
    first_image = cv2.imread(png_files[0])
    height, width, layers = first_image.shape
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating movie from {len(png_files)} PNG files...")
    
    for png_file in png_files:
        frame = cv2.imread(png_file)
        if frame is not None:
            video_writer.write(frame)
            print(f"Added frame: {os.path.basename(png_file)}")
        else:
            print(f"Failed to read: {png_file}")
    
    video_writer.release()
    print(f"Movie saved to: {output_path}")


ch_1or2 = 2
drug = "Lat B 5 μM"

drug_added_datetime = datetime(2025, 10, 10, 14, 25, 0, 0)

flim_folder = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20251010\LatB"


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


#all file save folder
all_savefolder = os.path.join(savefolder, "all")
os.makedirs(all_savefolder, exist_ok=True)

for group, each_flim_list in grouped_files.items():
    first = True
    each_savefolder = os.path.join(savefolder, group)
    os.makedirs(each_savefolder, exist_ok=True)
    for each_flim_path in each_flim_list:
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
        savepath = os.path.join(all_savefolder, basename[:-5] + "_maxproj.png")
        plt.savefig(savepath, dpi=150, bbox_inches = "tight")
        print("maxproj_savepath ", savepath)

        savepath = os.path.join(each_savefolder, basename[:-5] + "_maxproj.png")
        plt.savefig(savepath, dpi=150, bbox_inches = "tight")
        print("maxproj_savepath ", savepath)       
        
        plt.show()
        plt.close(); plt.clf();plt.close("all");



# Create movies from each group folder
for group, each_flim_list in grouped_files.items():
    each_savefolder = os.path.join(savefolder, group)
    movie_output_path = os.path.join(savefolder, f"{group}_movie.mp4")
    
    if os.path.exists(each_savefolder):
        create_movie_from_pngs(each_savefolder, movie_output_path, fps=2)
    else:
        print(f"Folder not found: {each_savefolder}")
