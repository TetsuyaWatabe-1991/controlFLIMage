# %%
import sys
sys.path.append(r"C:\Users\WatabeT\Documents\Git\controlFLIMage")
import glob
from FLIMageFileReader2 import FileReader
from matplotlib import pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict
import datetime
import pandas as pd


folder_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250701\auto1"

file_list = glob.glob(os.path.join(folder_path, "for_align*highmag*.flim"))

file_list2 = glob.glob(os.path.join(folder_path, "lowmag[0-9]_[0-9][0-9][0-9].flim"))

file_list = file_list2 + file_list

grouped = defaultdict(list)

for filepath in file_list:
    filename = os.path.basename(filepath)
    # 拡張子を除く
    name_wo_ext = os.path.splitext(filename)[0]
    # 末尾3文字を除いた部分をキーにする
    key = name_wo_ext[:-4]  # [-8:-4]だと"_001"が抜ける
    grouped[key].append(filepath)

# グループを表示

result_df = pd.DataFrame()
for group_key, files in grouped.items():
    print(f"\nGroup: {group_key}")
    each_group_df = pd.DataFrame()
    for f in files:
        print(f"  {f}")
        now = datetime.datetime.now()
        modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        delta = (now - modified_date).total_seconds()
        print(f"delta: {delta}")
        if delta > 60:
            iminfo = FileReader()
            iminfo.read_imageFile(f, True) 
            # Get intensity only data
            imagearray=np.array(iminfo.image)
            motor_position = iminfo.statedict["State.Motor.motorPosition"]
            threeD_img = imagearray[:,:,0,:,:].sum(axis=1).sum(axis=-1)
            stdev = np.std(threeD_img)
            mean = np.mean(threeD_img)

            each_dict = {}
            each_dict["group"] = group_key
            each_dict["filepath"] = f
            each_dict["x"] = motor_position[0]
            each_dict["y"] = motor_position[1]
            each_dict["z"] = motor_position[2]
            each_dict["stdev"] = stdev
            each_dict["mean"] = mean
            dt = iminfo.statedict["State.Acq.triggerTime"]
            each_dict["dt"] = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
            each_group_df = pd.concat([each_group_df, pd.DataFrame([each_dict])], ignore_index=True)
            
    # calculate diff, first value should be 0
    each_group_df["x_diff"] = each_group_df["x"] - each_group_df["x"].iloc[0]
    each_group_df["y_diff"] = each_group_df["y"] - each_group_df["y"].iloc[0]
    each_group_df["z_diff"] = each_group_df["z"] - each_group_df["z"].iloc[0]
    each_group_df["stdev_diff"] = each_group_df["stdev"] - each_group_df["stdev"].iloc[0]
    each_group_df["mean_diff"] = each_group_df["mean"] - each_group_df["mean"].iloc[0]
    result_df = pd.concat([result_df, each_group_df], ignore_index=True)

min_dt = result_df["dt"].min()
relative_sec = (result_df["dt"] - min_dt).dt.total_seconds()
result_df["relative_sec"] = relative_sec

result_df.to_csv(folder_path+"result_df.csv", index = False)

print("--------------------------------")
print("saved as ", folder_path+"result_df.csv")



# #%%

# each_filepath = file_list[0]

# for each_filepath in file_list:
#     iminfo = FileReader()
#     iminfo.read_imageFile(each_filepath, True) 
#     # Get intensity only data
#     imagearray=np.array(iminfo.image)

#     motor_position = iminfo.statedict["State.Motor.motorPosition"]

#     z_proj = imagearray[:,:,ch1or2-1,:,:].sum(axis=0).sum(axis=0).sum(axis=-1)
#     y_proj = imagearray[:,:,ch1or2-1,:,:].max(axis=-2).sum(axis=1).sum(axis=-1)

#     title_txt = f"{os.path.basename(each_filepath)}\nx: {motor_position[0]:.0f},\ny: {motor_position[1]:.0f},\nz: {motor_position[2]:.0f}"


#     # サブプロットで上下に並べる
#     fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True, constrained_layout=True)
#     fig.suptitle(title_txt, x=0.01, y=0.98, ha='left', va='top')

#     axes[0].imshow(z_proj, cmap="gray", vmax=200, vmin=0)

#     axes[1].imshow(y_proj, cmap="gray", vmax=200, vmin=0, aspect='auto')


#     # 余白を自動調整
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.88)  # タイトルと被らないように

#     plt.savefig(os.path.join(r"C:\Users\WatabeT\Desktop\auto1 - Copy\check_pos",
#                         os.path.basename(each_filepath).replace(".flim", "_z_proj.png")),
#                         dpi=150, bbox_inches="tight")
#     plt.show()

# %%
