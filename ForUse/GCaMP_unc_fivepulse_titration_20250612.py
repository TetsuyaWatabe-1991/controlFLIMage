# %%
import sys
sys.path.append("../")
from controlflimage_threading import Control_flimage
from time import sleep
from datetime import datetime
import os
import numpy as np
from FLIMageAlignment import get_flimfile_list
from FLIMageFileReader2 import FileReader
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

direction_ini = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"

if "FLIMageCont" not in globals():
    FLIMageCont = Control_flimage(ini_path=direction_ini)

#### Laser linear regression
slope = 0.158
intercept = 0.139


from_Thorlab_to_coherent_factor = 1/3

laser_mW_ms = [
    [2.8, 6],
    [3.3, 6],
    [4.0, 6],
    [5.0, 6],
    ]

unc_pow_dur = []
for each_mw_ms in laser_mW_ms:
    mW_in_Thorlabs = each_mw_ms[0]/from_Thorlab_to_coherent_factor
    pow_from_mw = int((mW_in_Thorlabs - intercept)/slope)
    if pow_from_mw <= 100:    
        unc_pow_dur.append([pow_from_mw,each_mw_ms[1]])
    else:
        print(f"pow_from_mw = {pow_from_mw} is over 100")
print(unc_pow_dur)


settingpath_list = [r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\uncaging_2times.txt",
                    r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\fivepulses.txt",
                    r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\LTP_by_fivepulses50Hz.txt"
                    ]

for each_settingpath in settingpath_list:
    FLIMageCont.flim.sendCommand(f'LoadSetting, {each_settingpath}')


    for each_pow_dur in unc_pow_dur:
        
        each_pow = each_pow_dur[0]
        each_dur = each_pow_dur[1]
        
        FLIMageCont.set_uncaging_power(each_pow)
        sleep(0.1)
        FLIMageCont.flim.sendCommand(f'State.Uncaging.pulseWidth = {each_dur}')
        # FLIMageCont.flim.sendCommand('State.Uncaging.pulseWidth = 6')
        sleep(0.1)
        
        FLIMageCont.flim.sendCommand('StartGrab')
        sleep(11)

        a = FLIMageCont.flim.sendCommand('GetFullFileName')
        one_file = a[a.find(",")+2:]

        if False:
            one_file = r"G:\ImagingData\Tetsuya\20250506\onHarp\E4_roomair_LTP2_8um_002.flim"
        filelist = get_flimfile_list(one_file)
        print(filelist)
        
        for each_file in filelist: 

            uncaging_iminfo = FileReader()
            uncaging_iminfo.read_imageFile(each_file, True) 
            
            unc_dt = datetime.fromisoformat(uncaging_iminfo.acqTime[2])
            imagearray=np.array(uncaging_iminfo.image)
            
            if (imagearray.shape)[0] not in [4,33,34]:
                continue
                
            uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
            uncaging_pow = uncaging_iminfo.statedict["State.Uncaging.Power"]
            pulseWidth = uncaging_iminfo.statedict["State.Uncaging.pulseWidth"]
            center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
            center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]

            GCpre = imagearray[0,0,0,:,:,:].sum(axis = -1)
            GCunc = imagearray[3,0,0,:,:,:].sum(axis = -1)
            Tdpre = imagearray[0,0,1,:,:,:].sum(axis = -1)
            Td1min = imagearray[-1,0,1,:,:,:].sum(axis = -1)

            GC_pre_med = median_filter(GCpre, size=3)
            GC_unc_med = median_filter(GCunc, size=3)

            GCF_F0 = (GC_unc_med/GC_pre_med)
            GCF_F0[GC_pre_med == 0] = 0
    
            pow_mw = slope * uncaging_pow + intercept
            pow_mw_coherent = pow_mw/3
            pow_mw_round = round(pow_mw_coherent,1)
            
            plt.imshow(GCF_F0, cmap = "inferno", vmin = 1, vmax = 10)
            plt.plot(center_x, center_y, 'ro', markersize=2)   
            plt.title(f"{pow_mw_round} mW, {pulseWidth} ms")  
            plt.axis('off')
            
            folder = os.path.dirname(each_file)
            savefolder = os.path.join(folder,"plot")
            os.makedirs(savefolder, exist_ok=True)
            basename = os.path.basename(each_file)
                            
            savepath = os.path.join(savefolder, basename[:-5] + ".png")
            plt.savefig(savepath, dpi=150, bbox_inches = "tight")
            
            plt.show()
            
        
        
        if (each_settingpath != settingpath_list[-1])*(each_pow_dur == unc_pow_dur[-1]):    
            break
        else:
            sleep_sec = 13
            for i in range(sleep_sec):
                print(sleep_sec - i , "sec ", end = " ")
                sleep(1)
            

import winsound
for i in range(1):
    winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

print("end")
for each_pow_dur, each_mW in zip(unc_pow_dur, laser_mW_ms):
    print(each_mW[0], "mW   ",each_pow_dur[0], " %")

# %%