# %%
import sys
import os
file_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(file_dir)
# sys.path.append("../")
from controlflimage_threading import Control_flimage
from time import sleep

inipath = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
if not os.path.exists(inipath):
    inipath = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
FLIMageCont = Control_flimage(ini_path = inipath)   
FLIMageCont.directionMotorY = FLIMageCont.directionMotorY 

# FLIMageCont.set_param(RepeatNum=80, interval_sec=60, ch_1or2=2,
#                       LoadSetting=False,drift_control=True,
#                       ShowUncagingDetection=True,drift_cont_galvo=False,expected_grab_duration_sec=40) 

FLIMageCont.set_param(RepeatNum=800, interval_sec=60, ch_1or2=2,
                      LoadSetting=False,                     
                      drift_control=True,
                      ShowUncagingDetection=True,
                      drift_cont_galvo=False,
                      expected_grab_duration_sec=15)       

FLIMageCont.start_repeat()

# for i in range(100):
#     FLIMageCont.acquisition_include_connect_wait()
#     sleep(5)

# %%
 