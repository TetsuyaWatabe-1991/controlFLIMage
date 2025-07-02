# %%
import sys
sys.path.append(r"C:\Users\WatabeT\Documents\Git\controlFLIMage")
import glob
from FLIMageFileReader2 import FileReader
from matplotlib import pyplot as plt
import numpy as np
import os
import datetime
from utility.send_notification import send_slack_url_default

ch1or2 = 2

#%%

folder_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250701\auto1"

file_list = glob.glob(os.path.join(folder_path, "for_align*highmag*.flim"))

file_list2 = glob.glob(os.path.join(folder_path, "lowmag[0-9]_[0-9][0-9][0-9].flim"))

file_list = file_list2 + file_list

each_filepath = file_list[0]

for each_filepath in file_list:
    
    now = datetime.datetime.now()
    modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(each_filepath))
    delta = (now - modified_date).total_seconds()
    print(each_filepath, delta)

    if delta < 60:
        continue

    iminfo = FileReader()
    iminfo.read_imageFile(each_filepath, True) 
    # Get intensity only data
    imagearray=np.array(iminfo.image)

    motor_position = iminfo.statedict["State.Motor.motorPosition"]

    z_proj = imagearray[:,:,ch1or2-1,:,:].sum(axis=0).sum(axis=0).sum(axis=-1)
    y_proj = imagearray[:,:,ch1or2-1,:,:].max(axis=-2).sum(axis=1).sum(axis=-1)

    title_txt = f"{os.path.basename(each_filepath)}\nx: {motor_position[0]:.0f},\ny: {motor_position[1]:.0f},\nz: {motor_position[2]:.0f}"


    # サブプロットで上下に並べる
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True, constrained_layout=True)
    fig.suptitle(title_txt, x=0.01, y=0.98, ha='left', va='top')

    axes[0].imshow(z_proj, cmap="gray", vmax=200, vmin=0)

    axes[1].imshow(y_proj, cmap="gray", vmax=200, vmin=0, aspect='auto')


    # 余白を自動調整
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # タイトルと被らないように

    savefolder = os.path.join(folder_path,"check_pos")
    os.makedirs(savefolder, exist_ok=True)
    plt.savefig(os.path.join(savefolder,
                        os.path.basename(each_filepath).replace(".flim", "_z_proj.png")),
                        dpi=150, bbox_inches="tight")
    plt.close();plt.clf();plt.cla();

print("================================================")
print("saved as \n", savefolder)
send_slack_url_default(f"saved as \n {savefolder}")



    # %%
