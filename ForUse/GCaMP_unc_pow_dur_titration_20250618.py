# %%
import glob, json, os
from time import sleep
import sys
sys.path.append("../")
from sklearn.linear_model import LinearRegression
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from controlflimage_threading import Control_flimage
from FLIMageAlignment import get_flimfile_list
from FLIMageFileReader2 import FileReader
plt.rcParams['image.interpolation'] = 'none'

def process_and_plot_flim_images(filelist, power_slope, power_intercept, 
                                 F_F0_vmin = 1, F_F0_vmax = 10,
                                 result_img_dict = None,
                                 from_Thorlab_to_coherent_factor=1/3):
    # Initialize result_img_dict if None
    if result_img_dict is None:
        result_img_dict = {}
    
    for each_file in filelist: 
        print(each_file)
        if each_file in result_img_dict:
            continue       
        uncaging_iminfo = FileReader()
        uncaging_iminfo.read_imageFile(each_file, True) 
        imagearray=np.array(uncaging_iminfo.image)
        
        allowed_shape_0th_list = [4,33,34]
        if (imagearray.shape)[0] not in allowed_shape_0th_list:
            print("skipped", each_file)
            print(imagearray.shape, "0th dim is not in", allowed_shape_0th_list)
            continue

        result_img_dict[each_file] = {}
        result_img_dict[each_file]["statedict"] = uncaging_iminfo.statedict
        result_img_dict[each_file]["imagearray"] = imagearray

    n_imgs = len(result_img_dict)
    each_fig_size = 2
    fig = plt.figure(figsize=(each_fig_size*(n_imgs+2), each_fig_size))
    gs = gridspec.GridSpec(1, n_imgs+2, width_ratios=[0.9] + [1]*n_imgs + [0.08], wspace=0.05)

    for nth_plot, each_file in enumerate(result_img_dict):
        each_imagearray = result_img_dict[each_file]["imagearray"]
        statedict = result_img_dict[each_file]["statedict"]
        uncaging_x_y_0to1 = statedict["State.Uncaging.Position"]
        uncaging_pow = statedict["State.Uncaging.Power"]
        pulseWidth = statedict["State.Uncaging.pulseWidth"]
        center_y = each_imagearray.shape[-2] * uncaging_x_y_0to1[1]
        center_x = each_imagearray.shape[-3] * uncaging_x_y_0to1[0]

        GCpre = each_imagearray[0,0,0,:,:,:].sum(axis = -1)
        GCunc = each_imagearray[3,0,0,:,:,:].sum(axis = -1)
        Tdpre = each_imagearray[0,0,1,:,:,:].sum(axis = -1)
        GC_pre_med = median_filter(GCpre, size=3)
        GC_unc_med = median_filter(GCunc, size=3)
        GCF_F0 = (GC_unc_med/GC_pre_med)
        GCF_F0[GC_pre_med == 0] = 0

        pow_mw = power_slope * uncaging_pow + power_intercept
        pow_mw_coherent = pow_mw*from_Thorlab_to_coherent_factor
        pow_mw_round = round(pow_mw_coherent,1)

        # Plot Tdpre at far left, only for the first image
        if nth_plot == 0:
            ax_td = fig.add_subplot(gs[0, 0])
            ax_td.imshow(Tdpre, cmap='gray')
            ax_td.plot(center_x, center_y, 'c+', markersize=5)
            ax_td.set_title('tdTomato')
            ax_td.axis('off')

        # Plot F/F0 images in columns 1,2,3,...
        ax = fig.add_subplot(gs[0, nth_plot+1])
        im = ax.imshow(GCF_F0, cmap = "inferno", vmin = F_F0_vmin, vmax = F_F0_vmax)
        ax.plot(center_x, center_y, 'c+', markersize=5)
        ax.set_title(f"{pow_mw_round} mW, {pulseWidth} ms")
        ax.axis('off')

    # Color bar at the far right
    ax_cbar = fig.add_subplot(gs[0, -1])
    norm = Normalize(vmin=F_F0_vmin, vmax=F_F0_vmax)
    sm = ScalarMappable(norm=norm, cmap='inferno')
    cbar = plt.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.ax.set_yticks([])
    cbar.ax.tick_params(size=0)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    fontsize = 12
    cbar.ax.text(0.5, 1.02, str(F_F0_vmax), ha='center', va='bottom', fontsize=fontsize, transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.02, str(F_F0_vmin), ha='center', va='top', fontsize=fontsize, transform=cbar.ax.transAxes)
    cbar.ax.text(1.3, 0.5, 'F/F0', ha='left', va='center', fontsize=fontsize, rotation=90, transform=cbar.ax.transAxes)

    folder = os.path.dirname(each_file)
    savefolder = os.path.join(folder,"plot")
    os.makedirs(savefolder, exist_ok=True)
    basename = os.path.basename(each_file)
    savepath = os.path.join(savefolder, basename[:-8] + ".png")
    plt.savefig(savepath, dpi=150, bbox_inches = "tight")
    plt.show()
    
    return result_img_dict

if False:
    power_slope = 0.1762
    power_intercept = 0.0646
    one_file = r"G:\ImagingData\Tetsuya\20250403\B6_cut0319_FlxGC6sTom_0322\highmag_Trans5ms\tpem\B3_00_2_1_dend1_004.flim"
    one_file_list = [
        r"G:\ImagingData\Tetsuya\20250618\titration_dend10_11um_001.flim",  
        r"G:\ImagingData\Tetsuya\20250618\titration_dend1_10um_001.flim",
        r"G:\ImagingData\Tetsuya\20250618\titration_dend2_23um_001.flim",
        r"G:\ImagingData\Tetsuya\20250618\titration_dend3_10um_001.flim",
        r"G:\ImagingData\Tetsuya\20250618\titration_dend4_18um_001.flim",
        r"G:\ImagingData\Tetsuya\20250618\titration_dend5_18um_001.flim",
        r"G:\ImagingData\Tetsuya\20250618\titration_dend6_8um_001.flim",
        r"G:\ImagingData\Tetsuya\20250618\titration_dend7_13um_001.flim",
        r"G:\ImagingData\Tetsuya\20250618\titration_dend9_15um_001.flim",
        r"G:\ImagingData\Tetsuya\20250618\titration_dend11_11um_001.flim",
    ]

    from_Thorlab_to_coherent_factor = 1/3
    for each_one_file in one_file_list:
        filelist = get_flimfile_list(each_one_file)
        _=process_and_plot_flim_images(filelist, power_slope = power_slope, power_intercept = power_intercept,
                                       F_F0_vmin = 1, F_F0_vmax = 5,
                                       result_img_dict=None,
                                       from_Thorlab_to_coherent_factor=from_Thorlab_to_coherent_factor)


# %%
#Run

direction_ini = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
SettingPath = r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\uncaging_2times.txt"

uncaging_every_sec = 30

send_counter = True
send_basename = True

nth_dend = 14
um = 10

vmin = 1
vmax = 6


basename = f"titration_dend{nth_dend}_{um}um_"

if "FLIMageCont" not in globals():
    FLIMageCont = Control_flimage(ini_path=direction_ini)

powermeter_dir = r"C:\Users\yasudalab\Documents\Tetsuya_Imaging\powermeter"
latest_json_path = glob.glob(os.path.join(powermeter_dir, "*.json"))[-1]

latest_json_basename =  os.path.basename(latest_json_path).replace(".json","")
latest_json_datetime = datetime.strptime(latest_json_basename, "%Y%m%d%H%M")

now = datetime.now()
#print delta time in hours and minutes
delta_sec = int((now - latest_json_datetime).total_seconds())
delta_hours = delta_sec // 3600
delta_minutes = (delta_sec % 3600) // 60
print(f"Power calibrated {delta_hours} hr {delta_minutes} min ago")
if delta_hours > 24:
    print("Power calibration is more than 24 hours ago")
    input("press enter to continue..")

with open(latest_json_path, "r") as f:
    data = json.load(f)
x_laser = np.array(list(data["Laser2"].keys())).reshape(-1, 1).astype(float)
y_laser = np.array(list(data["Laser2"].values())).astype(float)
model = LinearRegression()
model.fit(x_laser, y_laser)
power_slope = model.coef_[0]
power_intercept = model.intercept_

print(f"Laser2 slope = {round(power_slope,3)}, intercept = {round(power_intercept,3)}")

if False:
    #### Laser linear regression
    slope = 0.186
    intercept = 0.0466

FLIMageCont.flim.sendCommand(f'LoadSetting, {SettingPath}')
if send_basename:
    FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{basename}"')
if send_counter:
    FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = 1')

from_Thorlab_to_coherent_factor = 1/3



laser_mW_ms = [
    [2.0, 6],
    [2.4, 6],
    [2.8, 6],
    [3.3, 6],
    [4.0, 6],
    # [5.0, 6],
    #[6.5, 6],
    ]

unc_pow_dur = []
for each_mw_ms in laser_mW_ms:
    
    mW_in_Thorlabs = each_mw_ms[0]/from_Thorlab_to_coherent_factor
    pow_from_mw = int((mW_in_Thorlabs - power_intercept)/power_slope)
    if pow_from_mw <= 100:    
        unc_pow_dur.append([pow_from_mw,each_mw_ms[1]])
    else:
        print(f"pow_from_mw = {pow_from_mw} is over 100")
        
print(unc_pow_dur)
if len(laser_mW_ms) != len(unc_pow_dur):
    print("laser_mW_ms and unc_pow_dur are not the same length")
    input("press enter to continue..")

# FLIMageCont.set_param(RepeatNum=1, interval_sec=30, ch_1or2=2)

result_img_dict = {}
for each_pow_dur in unc_pow_dur:
    
    each_pow = each_pow_dur[0]
    each_dur = each_pow_dur[1]
    
    FLIMageCont.set_uncaging_power(each_pow)
    sleep(0.1)
    FLIMageCont.flim.sendCommand(f'State.Uncaging.pulseWidth = {each_dur}')
    # FLIMageCont.flim.sendCommand('State.Uncaging.pulseWidth = 6')
    sleep(0.1)
    
    tic = datetime.now()

    # FLIMageCont.acquisition_include_connect_wait()
    FLIMageCont.acquisition_include_connect_wait()

    a = FLIMageCont.flim.sendCommand('GetFullFileName')
    one_file = a[a.find(",")+2:]
    # each_firstfilepath = r"G:\ImagingData\Tetsuya\20250403\B6_cut0319_FlxGC6sTom_0322\highmag_Trans5ms\tpem\B3_00_2_1_dend1_004.flim"
    # one_file = os.path.join(folder, basename + "001.flim")
    if False:
        one_file = r"G:\ImagingData\Tetsuya\20250506\onHarp\E4_roomair_LTP2_8um_002.flim"
    filelist = get_flimfile_list(one_file)
    
    # Call the function instead of the inline code
    result_img_dict = process_and_plot_flim_images(filelist = filelist, 
                                                   power_slope = power_slope, power_intercept = power_intercept, 
                                                   F_F0_vmin = vmin, F_F0_vmax = vmax,
                                                   result_img_dict = result_img_dict, 
                                                   from_Thorlab_to_coherent_factor=from_Thorlab_to_coherent_factor)


    if each_pow_dur == unc_pow_dur[-1]:
        break
    else:
        toc = datetime.now()
        delta_sec = (toc - tic).total_seconds()
        for i in range(int(uncaging_every_sec - delta_sec)):
            print(int(uncaging_every_sec - delta_sec) - i , end = " ")
            sleep(1)



import winsound
for i in range(1):
    winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

print("end")

# %%
