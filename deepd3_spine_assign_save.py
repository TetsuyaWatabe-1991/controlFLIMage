# run under jupyter environment
import glob
import os
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
from deepd3_spine_head_detector import SpinePosDeepD3
from FLIMageFileReader2 import FileReader
from multidim_tiff_viewer import save_spine_dend_info,\
    define_uncagingPoint_dend_click_multiple, read_xyz_single

def main():    
    highmag_folder = r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20250520\E4bufferMNI4Ca"
    highmag_filename = "*_highmag_*002.flim"
    exclude_ini_saved = True
    from_latest_file = False
    loop_spine_assign_save(highmag_folder, highmag_filename, exclude_ini_saved, from_latest_file)
    print("\n\n done")

def imshow_with_dend_spine_uncaging(mip_img, spine_zyx, 
                                    dend_slope, dend_intercept,
                                    savefig = False,
                                    savepath = "sample.png",
                                    pix_size = 512
                                    ):
    plt.figure(figsize = (4,4))
    plt.imshow(mip_img, cmap = 'gray')
    plt.axis('off')
    plt.scatter(spine_zyx[2],
                spine_zyx[1], c = "r", s = 4 )
    dend_x = []
    dend_y = []
    for x in range(1,mip_img.shape[1]-1):
        y = dend_slope*x + dend_intercept
        if (1<y) * (y < mip_img.shape[0]-1):
            dend_x.append(x)
            dend_y.append(y)
    plt.scatter(dend_x, dend_y,c = "g", s = 2)
    if savefig:
        plt.savefig(savepath, bbox_inches='tight', pad_inches = 0, dpi = int(pix_size/4*1.3264))
    plt.show()
    
    
def extract_prefix_and_number(file_path):
    match = re.match(r"(.*)_(\d{3})\.flim$", file_path)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return prefix, number
    return None, None

def get_first_flim_list(sample_path_for_glob):
    filtered_files = {}
    for file_path in glob.glob(sample_path_for_glob):
        prefix, number = extract_prefix_and_number(file_path)
        if prefix:
            if prefix not in filtered_files or number < filtered_files[prefix][1]:
                filtered_files[prefix] = (file_path, number)
    result_files = [file_info[0] for file_info in filtered_files.values()]
    return result_files


def loop_spine_assign_save(highmag_folder, highmag_filename, exclude_ini_saved, from_latest_file):
    sample_path_for_glob = os.path.join(highmag_folder, highmag_filename)
    first_flim_list = get_first_flim_list(sample_path_for_glob)
    first_flim_list = sorted(first_flim_list, key=lambda file: os.path.getmtime(file), reverse=from_latest_file)
    print("The number of highmag flim files: ", len(first_flim_list))
    for flim_path in first_flim_list:
        inipath = flim_path[:-9] + ".ini"
        inilist = glob.glob(os.path.join(flim_path[:-9],
                            os.path.basename(flim_path[:-9])+"*.ini"))
        if len(inilist)*exclude_ini_saved:
            print("defined: ",flim_path)
            continue
            
        now = datetime.datetime.now()
        modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(flim_path))
        delta = (now - modified_date).total_seconds()
        if delta > 20:

            SpineAssign = SpinePosDeepD3()
            SpineAssign.trainingdata_path = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\ongoing\deepd3\DeepD3_32F.h5"
            
            iminfo = FileReader()        

            iminfo.read_imageFile(flim_path, True)
            imagearray=np.array(iminfo.image)
            mip_img = np.sum(np.sum(np.sum(np.sum(imagearray,axis=-1),axis=1),axis=1),axis=0)
            
            result_dict = SpineAssign.return_uncaging_pos_based_on_roi_sum(flim_path, 
                                                                    plot_them = True,
                                                                    upper_lim_spine_pixel_percentile = 99,
                                                                    lower_lim_spine_pixel_percentile = 1,
                                                                    upper_lim_spine_intensity_percentile = 99,
                                                                    lower_lim_spine_intensity_percentile = 1,
                                                                    ignore_edge_percentile = 10)

            if len(result_dict)<1:
                print("\n\n no uncaging pos detected")

            ext = flim_path.split(".")[-1]
            savefolder = flim_path[:-len(ext)-5]
            os.makedirs(savefolder, exist_ok=True)

            for each_key in result_dict:
                x = result_dict[each_key]["centroid_x_pix"]
                y = result_dict[each_key]["centroid_y_pix"]
                z = result_dict[each_key]["z_pix"]
                spine_zyx = [z,y,x]
                dend_slope = result_dict[each_key]["orientation"]

                coordinates = result_dict[each_key]["neighborhood_points"]
                dend_slope, dend_intercept = np.polyfit(coordinates[:,1], 
                                                        coordinates[:,0], 
                                                        1)
                inipath = os.path.join(savefolder, 
                                    os.path.basename(flim_path)[:-9] + f"_{each_key}.ini")
                pngpath = inipath[:-4]+".png"
                imshow_with_dend_spine_uncaging(mip_img, spine_zyx, dend_slope, dend_intercept,
                                                savefig = True,
                                                savepath = pngpath
                                                )
                save_spine_dend_info(spine_zyx, dend_slope, dend_intercept, inipath)


            for each_key in result_dict:    
                # inipath = flim_path[:-9] + f"_{each_key}.ini"
                inipath = os.path.join(savefolder, os.path.basename(flim_path)[:-9] + f"_{each_key}.ini")
                pngpath = inipath[:-4]+".png"
                spine_zyx, dend_slope, dend_intercept, excluded = read_xyz_single(inipath, return_excluded = True)
                res , _ , _ = define_uncagingPoint_dend_click_multiple(flim_path,
                                                                    read_ini = True,
                                                                    inipath = inipath,
                                                                    SampleImg = pngpath,
                                                                    only_for_exclusion = True)
                excluded = int(res[0] == -1)
                save_spine_dend_info(spine_zyx, dend_slope, dend_intercept, inipath,
                                    excluded = excluded)



if __name__ == "__main__":
    main()