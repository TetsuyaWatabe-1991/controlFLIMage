#%% 
#import
import sys
sys.path.append('..\\')
import os
import glob
import pandas as pd
from FLIMageAlignment import get_flimfile_list
from FLIMageFileReader2 import FileReader
from datetime import datetime

#%%
ignore_words = ["for_align"]
one_of_file_per_dir_list = [
    r"C:\Users\WatabeT\Desktop\20250626_Copy\lowmag1__highmag_1_002.flim"
]

pre_length = 2
z_stack_frame_num = [11]
uncaging_frame_num = [33,34]
titration_frame_num = [32]
combined_df = pd.DataFrame()

num_error_unknown = 0
error_list = []

for each_one_of_file_per_dir in one_of_file_per_dir_list:
    one_of_file_list = glob.glob(
        os.path.join(
            os.path.dirname(each_one_of_file_per_dir), 
            "*_highmag_*002.flim"
            )
        )
    one_of_file_list = [each_file for each_file in one_of_file_list if not any(ignore_word in each_file for ignore_word in ignore_words)]


    for each_firstfilepath in one_of_file_list:
        filelist= get_flimfile_list(each_firstfilepath)
        if len(filelist) == 1:
            continue
        # print(each_firstfilepath)

        First=True
        nth = -1
        nth_omit_induction = -1
        each_group_df = pd.DataFrame()

        for file_path in filelist:
            nth += 1
            # print(file_path)

            iminfo = FileReader()        
            iminfo.read_imageFile(file_path, False)

            z_stack_TF = False
            uncaging_TF = False
            titration_TF = False

            if iminfo.n_images in z_stack_frame_num:
                print(file_path,'<- z_stack')
                z_stack_TF = True
            elif iminfo.n_images in titration_frame_num:
                titration_TF = True
                print(file_path,'<- titration')
            elif iminfo.n_images in uncaging_frame_num:
                print(file_path,'<- uncaging')
                uncaging_TF = True        
            else:
                print(file_path,'<- unknown')
                num_error_unknown += 1
                error_list.append(file_path)
       
            # y_pix = iminfo.statedict["State.Acq.linesPerFrame"]
            # x_pix = iminfo.statedict["State.Acq.pixelsPerLine"]
            # uncaging_x_y_0to1 = [0,0]
            # motor_position = iminfo.statedict["State.Motor.motorPosition"]
            # z_position = motor_position[2]
            # z_relative_step_nth = -99
            dt_str = iminfo.acqTime[0]
            dt_formatter = "%Y-%m-%dT%H:%M:%S.%f"  # Fixed format string with capital T
            dt = datetime.strptime(dt_str, dt_formatter)

            df_nth_omit_induction = -1
            if titration_TF:
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
                "stem_name": os.path.basename(file_path),
                # "center_x": x_pix * uncaging_x_y_0to1[0],
                # "center_y": y_pix * uncaging_x_y_0to1[1],
                # "stepZ": [iminfo.statedict["State.Acq.sliceStep"]],
                # "z_position": [z_position], 
                # "z_relative_step_nth": [z_relative_step_nth],
                "dt": [dt],
                # "dt_str": [dt_str],
                # "relative_time_sec": [None],
                })
            
            each_group_df = pd.concat([each_group_df, each_df],ignore_index=True)

        uncaging_nth_list = each_group_df[each_group_df["uncaging_frame"]]["nth"].tolist()
        titration_nth_list = each_group_df[each_group_df["titration_frame"]]["nth"].tolist()
        
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
                # each_group_df.loc[each_set_df.index, "relative_time_sec"] = relative_time
                # each_group_df.loc[each_set_df.index, "relative_time_min"] = round(relative_time/60, 1)

        each_group_df["last_num"] = -99
        last_num = 1
        for each_set_label in each_group_df["nth_set_label"].unique():
            each_set_df = each_group_df[each_group_df["nth_set_label"] == each_set_label]
            for ind, rows in each_set_df.iterrows():
                if each_set_label == -1:
                    last_num+=1
                    each_group_df.at[ind, "last_num"] = last_num
                    each_group_df.at[ind, "stem_name_renamed"] =  each_group_df.at[ind, "stem_name"][:-8] + f"{last_num:03d}.flim"            
                else:
                    if each_set_df.at[ind, "phase"] in ["pre", "post", "unc"]:
                        last_num+=1
                        each_group_df.at[ind, "last_num"] = last_num        
                        each_group_df.at[ind, "stem_name_renamed"] =  each_group_df.at[ind, "stem_name"][:-8] + f"{last_num:03d}.flim"                
                    elif each_set_df.at[ind, "phase"] == "titration":
                        each_group_df.at[ind, "last_num"] = -100
                        each_group_df.at[ind, "stem_name_renamed"] =  "titration_" + each_group_df.at[ind, "stem_name"]
                    elif each_set_df.at[ind, "phase"] == "None":
                        each_group_df.at[ind, "last_num"] = -101
                        each_group_df.at[ind, "stem_name_renamed"] =  "None_" + each_group_df.at[ind, "stem_name"]

        combined_df = pd.concat([combined_df, each_group_df],ignore_index=True)

    df_savepath = os.path.join(os.path.dirname(each_one_of_file_per_dir), "rename_combined_df.csv")
    combined_df.to_csv(df_savepath, index=False)
    print(f"Saved to {df_savepath}")


            # y_pix = iminfo.statedict["State.Acq.linesPerFrame"]
            # x_pix = iminfo.statedict["State.Acq.pixelsPerLine"]
            # uncaging_x_y_0to1 = [0,0]
            # motor_position = iminfo.statedict["State.Motor.motorPosition"]
            # z_position = motor_position[2]
            # z_relative_step_nth = -99
            # dt_str = iminfo.acqTime[0]
            # dt_formatter = "%Y-%m-%dT%H:%M:%S.%f"  # Fixed format string with capital T
            # dt = datetime.strptime(dt_str, dt_formatter)


# %%
`2`