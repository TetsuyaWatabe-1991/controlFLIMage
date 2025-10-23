import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
from flimage_graph_func import plot_GCaMP_F_F0
import glob
from FLIMageFileReader2 import FileReader
import os
one_of_filepath = r"G:\ImagingData\Tetsuya\20250904\auto1\B1cnt_pos5_001.flim"
flim_path_list = glob.glob(os.path.join(os.path.dirname(one_of_filepath),"*.flim"))


for nth_file, each_file in enumerate(flim_path_list):
    
    iminfo = FileReader()
    iminfo.read_imageFile(file_path = each_file, readImage = False)

    frames = int(iminfo.statedict['State.Acq.nFrames'])

    if frames == 264:
        plot_GCaMP_F_F0(each_file, slope = 0, intercept = 0, 
                    from_Thorlab_to_coherent_factor = 1/3,
                    vmin = 1, vmax = 10, cmap='inferno', 
                    acceptable_image_shape_0th_list = [4,32, 33,34])
