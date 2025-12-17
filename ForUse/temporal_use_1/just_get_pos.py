import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
import glob
import os
import datetime
import pandas as pd
import numpy as np
from FLIMageFileReader2 import FileReader
import matplotlib.pyplot as plt

one_of_filepath = r"G:\ImagingData\Tetsuya\20251016\auto1\1_pos1__highmag_1_018.flim"

file_list = glob.glob(os.path.join(os.path.dirname(one_of_filepath), "*.flim"))

each_group_df = pd.DataFrame()
for nth, each_filepath in enumerate(file_list):
    #if each_filepath start with for_align
    if each_filepath.startswith("for_align"):
        continue
    iminfo = FileReader()
    iminfo.read_imageFile(each_filepath, readImage = False)
    x,y,z = iminfo.statedict["State.Motor.motorPosition"]

    each_dict = {}
    each_dict["filepath"] = each_filepath
    each_dict["x"] = x
    each_dict["y"] = y
    each_dict["z"] = z
    each_dict["dt"] = iminfo.statedict["State.Acq.triggerTime"]
    each_dict["dt"] = datetime.datetime.strptime(each_dict["dt"], "%Y-%m-%dT%H:%M:%S.%f")
    each_group_df = pd.concat([each_group_df, pd.DataFrame([each_dict])], ignore_index=True)

    if nth % 100 == 0:
        print(f"{nth}/{len(file_list)}")

# filename before ***.flim as group
each_group_df["group"] = each_group_df["filepath"].apply(lambda x: os.path.basename(x)[:-8])

#convert datetime to 20251014120130 format
each_group_df["dt_str"] = each_group_df["dt"].apply(lambda x: x.strftime("%Y%m%d%H%M%S"))

result_save_folder = os.path.join(os.path.dirname(one_of_filepath), "pos_analysis")
os.makedirs(result_save_folder, exist_ok=True)


sudden_diff_threshold = 20
num_sudden_diff = 0
num_sudden_diff_group = 0


for group in each_group_df["group"].unique():
    each_group_df_each = each_group_df[each_group_df["group"] == group]
    earliest_dt = each_group_df_each["dt"].min()
    earliest_x = each_group_df_each[each_group_df_each["dt"] == earliest_dt]["x"].values[0]
    earliest_y = each_group_df_each[each_group_df_each["dt"] == earliest_dt]["y"].values[0]
    earliest_z = each_group_df_each[each_group_df_each["dt"] == earliest_dt]["z"].values[0]
    each_group_df_each["x_diff"] = each_group_df_each["x"] - earliest_x
    each_group_df_each["y_diff"] = each_group_df_each["y"] - earliest_y
    each_group_df_each["z_diff"] = each_group_df_each["z"] - earliest_z
    each_group_df_each["hour_diff"] = (each_group_df_each["dt"] - earliest_dt).dt.total_seconds() / 3600

    # apply to original each_group_df
    each_group_df.loc[each_group_df["group"] == group, "x_diff"] = each_group_df_each["x_diff"]
    each_group_df.loc[each_group_df["group"] == group, "y_diff"] = each_group_df_each["y_diff"]
    each_group_df.loc[each_group_df["group"] == group, "z_diff"] = each_group_df_each["z_diff"]
    each_group_df.loc[each_group_df["group"] == group, "hour_diff"] = each_group_df_each["hour_diff"]
    
    ylim = [min(-10,each_group_df_each["x_diff"].min(), each_group_df_each["y_diff"].min(), each_group_df_each["z_diff"].min()),
            max(10,each_group_df_each["x_diff"].max(), each_group_df_each["y_diff"].max(), each_group_df_each["z_diff"].max())]
    
    diff_x = np.abs(each_group_df_each["x_diff"].values[:-1] - each_group_df_each["x_diff"].values[1:])
    diff_y = np.abs(each_group_df_each["y_diff"].values[:-1] - each_group_df_each["y_diff"].values[1:])
    diff_z = np.abs(each_group_df_each["z_diff"].values[:-1] - each_group_df_each["z_diff"].values[1:])

    num_sudden_diff_each_group = np.sum((diff_x > sudden_diff_threshold) | (diff_y > sudden_diff_threshold) | (diff_z > sudden_diff_threshold))
    if num_sudden_diff_each_group > 0:
        num_sudden_diff_group += 1
        num_sudden_diff += num_sudden_diff_each_group

    plt.figure(figsize=(4, 3))
    plt.title(group)
    plt.scatter(each_group_df_each["hour_diff"], each_group_df_each["x_diff"], color="red", marker="o")
    plt.scatter(each_group_df_each["hour_diff"], each_group_df_each["y_diff"], color="blue", marker="s")
    plt.scatter(each_group_df_each["hour_diff"], each_group_df_each["z_diff"], color="green", marker="^")
    plt.xlabel("Time (hours)")
    plt.ylabel("Position drift (um)")
    plt.legend(["x", "y", "z"])
    plt.ylim(ylim)
    plt.savefig(os.path.join(result_save_folder, f"{group}_pos_analysis.png"), dpi=150,bbox_inches="tight")
    plt.show()

    np.array(each_group_df_each["x_diff"])

display(f"Number of groups with sudden diff: {num_sudden_diff_group}, out of {len(each_group_df['group'].unique())} groups")
display(f"Number of sudden diff: {num_sudden_diff}, out of {len(each_group_df)} data points ")


each_group_df["abs_x_diff"] = each_group_df["x_diff"].abs()
each_group_df["abs_y_diff"] = each_group_df["y_diff"].abs()
each_group_df["abs_z_diff"] = each_group_df["z_diff"].abs()

# average by grouping all groups every 1 hour for x, y, z
each_group_df["hour_diff_rounded"] = each_group_df["hour_diff"].round(0)

import seaborn as sns

plt.figure(figsize=(4, 3))
sns.lineplot(x="hour_diff_rounded", y="abs_x_diff", data=each_group_df, color="red", legend = True, errorbar = "se")
sns.lineplot(x="hour_diff_rounded", y="abs_y_diff", data=each_group_df, color="blue", legend = True, errorbar = "se")
sns.lineplot(x="hour_diff_rounded", y="abs_z_diff", data=each_group_df, color="green", legend = True, errorbar = "se")
plt.legend(["x","x_sem", "y", "y_sem", "z", "z_sem"])
plt.show()


#save as csv
each_group_df.to_csv(os.path.join(result_save_folder, "pos_analysis.csv"), index=False)
