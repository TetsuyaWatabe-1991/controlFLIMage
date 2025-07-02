# %%
import sys
sys.path.append('..\\')
import os
import glob
import pandas as pd
from FLIMageAlignment import get_flimfile_list
from FLIMageFileReader2 import FileReader
from datetime import datetime

def get_uncaging_pos_multiple(one_of_file_list, 
                              pre_length = 1,
                              uncaging_frame_num = [33,34],
                              titration_frame_num = [32]):
    combined_df = pd.DataFrame()
    for each_firstfilepath in one_of_file_list:
        filelist= get_flimfile_list(each_firstfilepath)
        First=True
        nth = -1
        nth_omit_induction = -1
        each_group_df = pd.DataFrame()
    
        for file_path in filelist:
            nth += 1
            iminfo = FileReader()
            print(file_path)
            
            iminfo.read_imageFile(file_path, False) 
            
            if First:
                First=False
                first_n_images = iminfo.n_images

            uncaging_TF = False
            titration_TF = False
            unknown_TF = False
            if iminfo.n_images == first_n_images:
                uncaging_TF = False
            elif iminfo.n_images in titration_frame_num:
                titration_TF = True
                print(file_path,'<- titration')
            elif iminfo.n_images in uncaging_frame_num:
                print(file_path,'<- uncaging')
                uncaging_TF = True        
            else:
                unknown_TF = True
                print(file_path,'<- unknown')
                
            y_pix = iminfo.statedict["State.Acq.linesPerFrame"]
            x_pix = iminfo.statedict["State.Acq.pixelsPerLine"]
            uncaging_x_y_0to1 = [0,0]
            motor_position = iminfo.statedict["State.Motor.motorPosition"]
            z_position = motor_position[2]
            z_relative_step_nth = -99
            dt_str = iminfo.acqTime[0]
            dt_formatter = "%Y-%m-%dT%H:%M:%S.%f"  # Fixed format string with capital T
            dt = datetime.strptime(dt_str, dt_formatter)

            df_nth_omit_induction = -1
            if titration_TF:
                df_nth_omit_induction = -1
            elif uncaging_TF:
                uncaging_x_y_0to1 = iminfo.statedict["State.Uncaging.Position"]
                previous_z_position = each_group_df[each_group_df["nth"] == nth -1]["z_position"].iloc[0]
                previous_stepZ = each_group_df[each_group_df["nth"] == nth -1]["stepZ"].iloc[0]
                z_relative_step_nth = int((z_position - previous_z_position) / previous_stepZ)
            elif unknown_TF:
                print(file_path,'<- unknown')
                df_nth_omit_induction = -1
            else:
                nth_omit_induction +=1
                df_nth_omit_induction = nth_omit_induction

            each_df = pd.DataFrame({
                "nth": [nth],
                "nth_omit_induction": [df_nth_omit_induction],
                "group": os.path.basename(file_path)[:-8],
                "filepath_without_number": file_path[:-8],
                "file_path": file_path,
                "uncaging_frame": uncaging_TF,
                "titration_frame": titration_TF,
                "unknown_frame": unknown_TF,
                "center_x": x_pix * uncaging_x_y_0to1[0],
                "center_y": y_pix * uncaging_x_y_0to1[1],
                "stepZ": [iminfo.statedict["State.Acq.sliceStep"]],
                "z_position": [z_position], 
                "z_relative_step_nth": [z_relative_step_nth],
                "dt": [dt],
                "dt_str": [dt_str],
                "relative_time_sec": [None],
                })
            
            each_group_df = pd.concat([each_group_df, each_df],ignore_index=True)

        uncaging_nth_list = each_group_df[each_group_df["uncaging_frame"]]["nth"].tolist()
        titration_nth_list = each_group_df[each_group_df["titration_frame"]]["nth"].tolist()
        unknown_nth_list = each_group_df[each_group_df["unknown_frame"]]["nth"].tolist()

        
        all_nth = sorted(uncaging_nth_list + titration_nth_list)
        boundary_nth_list = [all_nth[i] for i in range(len(all_nth)) if i == 0 or all_nth[i] != all_nth[i-1] + 1]

        boundaries = [x - pre_length for x in boundary_nth_list]
        boundaries.append(len(each_group_df) + 1)

        for ind, rows in each_group_df.iterrows():
            nth = each_group_df.loc[ind,"nth"]
            if nth < boundaries[0]:
                nth_set_label = -1
                phase = "None"
            else:
                nth_set_label = -1
                phase = "None"
                for label, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                    if start <= nth < end:
                        unc_idx = boundary_nth_list[label]
                        if nth < unc_idx:
                            phase = "pre"
                        # elif nth == unc_idx:
                        elif nth in unknown_nth_list:
                            phase = "None"
                        elif nth in titration_nth_list:
                            phase = "titration"
                        elif nth in uncaging_nth_list:
                            phase = "unc"
                        else:
                            phase = "post"
                        nth_set_label = label
                        break
            
            each_group_df.loc[ind, "nth_set_label"] = nth_set_label
            each_group_df.loc[ind, "phase"] = phase

        for each_set_label in each_group_df["nth_set_label"].unique():
            if each_set_label == -1:
                continue

            each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]

            if not each_set_df["phase"].isin(["unc"]).any():
                print(f"No uncaging data found for group {each_group_df['group'].iloc[0]}, set {each_set_label}")
                each_group_df.loc[each_set_df.index, "phase"] = "None"
            else:
                unc_trigger_time = each_set_df.loc[each_set_df["phase"] == "unc", "dt"].iloc[0]
                relative_time = (each_set_df.loc[:, "dt"] - unc_trigger_time).dt.total_seconds()
                each_group_df.loc[each_set_df.index, "relative_time_sec"] = relative_time
                each_group_df.loc[each_set_df.index, "relative_time_min"] = round(relative_time/60, 1)

        combined_df = pd.concat([combined_df, each_group_df],ignore_index=True)
    return combined_df


if __name__ == "__main__":

    # one_of_filepath_list = [
    #     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\4lines_2_auto\lowmag1__highmag_1_002.flim",
    #     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250531\2lines_1\lowmag1__highmag_1_002.flim",
    #     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250601\2lines3_auto\lowmag1__highmag_1_002.flim",
    #     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250601\2lines4_auto\lowmag1__highmag_1_002.flim",
    #     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250602\lowmag1__highmag_1_002.flim",
    #     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250602\4lines_neuron4\lowmag1__highmag_1_002.flim",        
    # ]


    # one_of_filepath_list = [
    #     r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250626\lowmag5__highmag_1_063.flim",
    # ]

    one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250626\lowmag5__highmag_1_063.flim"
    z_plus_minus = 1
    ch_1or2 = 2
    pre_length = 2
    save_plot_TF = True
    save_tif_TF = True
    ignore_words = ["for_align"]

    one_of_file_list = glob.glob(
    os.path.join(
        os.path.dirname(one_of_filepath), 
        "*_highmag_*002.flim"
        )
    )
    one_of_file_list = [each_file for each_file in one_of_file_list if not any(ignore_word in each_file for ignore_word in ignore_words)]

    combined_df = get_uncaging_pos_multiple(one_of_file_list, pre_length=pre_length)
    

# %%


# uncaging_frame_num = [33,34]
# titration_frame_num = [32]
# pre_length = 2


# # one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250626\lowmag5__highmag_1_063.flim"
# one_of_filepath = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250626\lowmag1__highmag_1_002.flim"

# one_of_file_list = glob.glob(os.path.join(
#                                 os.path.dirname(one_of_filepath),"*_highmag_*002.flim"))
# print(one_of_file_list)


# combined_df = pd.DataFrame()

# filelist= get_flimfile_list(one_of_filepath)
# print(filelist)

# First=True
# nth = -1
# nth_omit_induction = -1
# each_group_df = pd.DataFrame()

# for file_path in filelist:
#     nth += 1
#     iminfo = FileReader()
#     print(file_path)
    
#     iminfo.read_imageFile(file_path, False) 
    
#     if First:
#         First=False
#         first_n_images = iminfo.n_images

#     uncaging_TF = False
#     titration_TF = False
#     if iminfo.n_images == first_n_images:
#         uncaging_TF = False
#     elif iminfo.n_images in titration_frame_num:
#         titration_TF = True
#         print(file_path,'<- titration')
#     elif iminfo.n_images in uncaging_frame_num:
#         print(file_path,'<- uncaging')
#         uncaging_TF = True        
#     else:
#         print(file_path,'<- unknown')
        
#     y_pix = iminfo.statedict["State.Acq.linesPerFrame"]
#     x_pix = iminfo.statedict["State.Acq.pixelsPerLine"]
#     uncaging_x_y_0to1 = [0,0]
#     motor_position = iminfo.statedict["State.Motor.motorPosition"]
#     z_position = motor_position[2]
#     z_relative_step_nth = -99
#     dt_str = iminfo.acqTime[0]
#     dt_formatter = "%Y-%m-%dT%H:%M:%S.%f"  # Fixed format string with capital T
#     dt = datetime.strptime(dt_str, dt_formatter)

#     df_nth_omit_induction = -1
#     if uncaging_TF:
#         uncaging_x_y_0to1 = iminfo.statedict["State.Uncaging.Position"]
#         previous_z_position = each_group_df[each_group_df["nth"] == nth -1]["z_position"].iloc[0]
#         previous_stepZ = each_group_df[each_group_df["nth"] == nth -1]["stepZ"].iloc[0]
#         z_relative_step_nth = int((z_position - previous_z_position) / previous_stepZ)
#     else:
#         nth_omit_induction +=1
#         df_nth_omit_induction = nth_omit_induction

#     each_df = pd.DataFrame({
#         "nth": [nth],
#         "nth_omit_induction": [df_nth_omit_induction],
#         "group": os.path.basename(one_of_filepath)[:-8],
#         "filepath_without_number": one_of_filepath[:-8],
#         "file_path": file_path,
#         "uncaging_frame": uncaging_TF,
#         "titration_frame": titration_TF,
#         "center_x": x_pix * uncaging_x_y_0to1[0],
#         "center_y": y_pix * uncaging_x_y_0to1[1],
#         "stepZ": [iminfo.statedict["State.Acq.sliceStep"]],
#         "z_position": [z_position], 
#         "z_relative_step_nth": [z_relative_step_nth],
#         "dt": [dt],
#         "dt_str": [dt_str],
#         "relative_time_sec": [None],
#         })
    
#     each_group_df = pd.concat([each_group_df, each_df],ignore_index=True)

# uncaging_nth_list = each_group_df[each_group_df["uncaging_frame"]]["nth"].tolist()
# titration_nth_list = each_group_df[each_group_df["titration_frame"]]["nth"].tolist()

# # 結合して連続番号を最初の番号のみに統合
# all_nth = sorted(uncaging_nth_list + titration_nth_list)
# boundary_nth_list = [all_nth[i] for i in range(len(all_nth)) if i == 0 or all_nth[i] != all_nth[i-1] + 1]

# boundaries = [x - pre_length for x in boundary_nth_list]
# boundaries.append(len(each_group_df) + 1)

# for ind, rows in each_group_df.iterrows():
#     nth = each_group_df.loc[ind,"nth"]
#     if nth < boundaries[0]:
#         nth_set_label = -1
#         phase = "None"
#     else:
#         nth_set_label = -1
#         phase = "None"
#         for label, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
#             if start <= nth < end:
#                 unc_idx = boundary_nth_list[label]
#                 if nth < unc_idx:
#                     phase = "pre"
#                 # elif nth == unc_idx:
#                 elif nth in titration_nth_list:
#                     phase = "titration"
#                 elif nth in uncaging_nth_list:
#                     phase = "unc"
#                 else:
#                     phase = "post"
#                 nth_set_label = label
#                 break
    
#     each_group_df.loc[ind, "nth_set_label"] = nth_set_label
#     each_group_df.loc[ind, "phase"] = phase

# for each_set_label in each_group_df["nth_set_label"].unique():
#     if each_set_label == -1:
#         continue

#     each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
#     unc_trigger_time = each_set_df.loc[each_set_df["phase"] == "unc", "dt"].iloc[0]
#     relative_time = (each_set_df.loc[:, "dt"] - unc_trigger_time).dt.total_seconds()
#     each_group_df.loc[each_set_df.index, "relative_time_sec"] = relative_time
#     each_group_df.loc[each_set_df.index, "relative_time_min"] = round(relative_time/60, 1)

# combined_df = pd.concat([combined_df, each_group_df],ignore_index=True)

# %%
