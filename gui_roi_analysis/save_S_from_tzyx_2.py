#%%
import os
import sys
sys.path.append('..\\')
sys.path.append('..\\..\\ongoing')
import numpy as np
import pandas as pd
import tifffile
import datetime
from ASIcontroller.deepd3_spine_head_detector import SpinePosDeepD3


def save_deepd3_S_tif(combined_df_path,
                      trainingdata_path = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\ongoing\deepd3\DeepD3_32F_94nm.h5",
                      based_on_max_proj = True, based_on_mean_proj = False, based_on_min_proj = False):
    combined_df = pd.read_pickle(combined_df_path)
    start_time = datetime.datetime.now()
    file_processed_count = 0
    res_dict = {}
    for each_filepath_without_number in combined_df["filepath_without_number"].unique():
        each_filepath_without_number_df = combined_df[combined_df["filepath_without_number"] == each_filepath_without_number]

        # # ### this is for debug
        # if each_filepath_without_number != 'C:\\Users\\WatabeT\\Desktop\\20250701\\auto1\\lowmag4__highmag_2_':
        #     continue
        # # ################

        parent_dir = os.path.dirname(each_filepath_without_number)    
        for each_nth_set_label in each_filepath_without_number_df["nth_set_label"].unique():
            if each_nth_set_label < 0:
                continue
            each_start_time = datetime.datetime.now()
            # #################
            # ### this is for debug    
            # if each_nth_set_label != 0:
            #     continue
            # #################

            

            each_nth_set_label_df_include_unc = each_filepath_without_number_df[each_filepath_without_number_df["nth_set_label"] == each_nth_set_label]
            each_nth_set_label_df_without_unc = each_nth_set_label_df_include_unc[each_nth_set_label_df_include_unc["nth_omit_induction"] > -1]
            
            x_um, y_um, z_um = each_nth_set_label_df_without_unc.x_um.unique(), each_nth_set_label_df_without_unc.y_um.unique(), each_nth_set_label_df_without_unc.z_um.unique()
            assert len(x_um) == 1, f"x_um is not unique: {x_um}"
            assert len(y_um) == 1, f"y_um is not unique: {y_um}"
            assert len(z_um) == 1, f"z_um is not unique: {z_um}"
            x_um, y_um, z_um = float(x_um[0]), float(y_um[0]), float(z_um[0])

            save_path_tzyx_aligned_list = each_nth_set_label_df_include_unc.save_path_tzyx_aligned.unique()
            assert len(save_path_tzyx_aligned_list) == 1
            save_path_tzyx_aligned = save_path_tzyx_aligned_list[0]
            print(f"save_path_tzyx_aligned: {save_path_tzyx_aligned}")


            arr = tifffile.imread(save_path_tzyx_aligned)
            arr[arr < 0] = 0

            print("arr.shape", arr.shape)

            proj_dict = {}
            proj_dict_keys = []
            if based_on_max_proj:
                proj_dict_keys.append("max_proj")
            if based_on_mean_proj:
                proj_dict_keys.append("mean_proj")
            if based_on_min_proj:
                proj_dict_keys.append("min_proj")

            for each_proj_key in proj_dict_keys:
                if each_proj_key == "max_proj":
                    zyx_time_proj = arr.max(axis=0).astype(np.uint16)
                elif each_proj_key == "mean_proj":
                    zyx_time_proj = arr.mean(axis=0).astype(np.uint16)
                elif each_proj_key == "min_proj":
                    zyx_time_proj = arr.min(axis=0).astype(np.uint16)
                proj_dict[each_proj_key] = zyx_time_proj


            save_folder = os.path.join(parent_dir, "deepd3_roi")
            os.makedirs(save_folder, exist_ok=True)

            res_dict[save_path_tzyx_aligned] = {}

            for each_proj_key, each_proj in proj_dict.items():
                print("each_proj.shape", each_proj.shape)
                SpineAssign = SpinePosDeepD3()
                SpineAssign.params = {
                                "roi_areaThreshold" : 0.5,
                                "roi_peakThreshold" : 1.0,
                                "roi_seedDelta" : 0.1,
                                "roi_distanceToSeed" : 10,
                                "min_roi_size" : 5,
                                "max_roi_size" : 1000,
                                "min_planes" : 1,
                                "max_dist_spine_dend_um": 3}

                SpineAssign.trainingdata_path = trainingdata_path
                # #flim_path = r"G:\ImagingData\Tetsuya\20240728\24well\highmagRFP50ms10p\tpem2\B1_00_1_2__highmag_2_007.flim"

                save_filename_stem = os.path.basename(save_path_tzyx_aligned).replace(".tif", "") + "_" + each_proj_key

                print("save_filename_stem", save_filename_stem)
                print("save_folder", save_folder)
                # assert False

                result_dict = SpineAssign.return_uncaging_pos_based_on_roi_sum(
                                                        max_distance = False,
                                                        plot_them = True,
                                                        upper_lim_spine_pixel_percentile = 100,
                                                        lower_lim_spine_pixel_percentile = 0,
                                                        upper_lim_spine_intensity_percentile = 100,
                                                        lower_lim_spine_intensity_percentile = 0,
                                                        ignore_first_n_plane=0,
                                                        ignore_last_n_plane=0,
                                                        ignore_edge_percentile = 0,
                                                        skeleton_3d = True,
                                                        direct_ZYXarray_use = True,
                                                        direct_ZYXarray = each_proj,
                                                        save_folder = save_folder,
                                                        save_filename_stem = save_filename_stem,
                                                        save_S = True,
                                                        define_save_folder = True,
                                                        xy_pixel_um = x_um,
                                                        z_pixel_um = z_um,
                                                        )
                res_dict[save_path_tzyx_aligned][each_proj_key] = result_dict

            each_end_time = datetime.datetime.now()
            print(f"Time taken for {each_proj_key}: {round((each_end_time - each_start_time).total_seconds(), 1)} seconds")
            file_processed_count += 1
            print(f"File processed count: {file_processed_count}")

    end_time = datetime.datetime.now()
    print(f"Total time taken: {round((end_time - start_time).total_seconds(), 1)} seconds")
    print(f"Time per file: {round((end_time - start_time).total_seconds() / file_processed_count, 1)} seconds")
    return res_dict
# %%

if __name__ == "__main__":
    combined_df_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250703\24well\auto1\combined_df.pkl"
    res_dict = save_deepd3_S_tif(combined_df_path, 
                                based_on_max_proj = True, 
                                based_on_mean_proj = False, 
                                based_on_min_proj = False)

    import pickle
    save_path = os.path.join(os.path.dirname(combined_df_path), "deepd3_res_dict.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(res_dict, f)

    
# %%