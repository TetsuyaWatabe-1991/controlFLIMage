# %%
# import sys
# sys.path.append("../")
from controlflimage_threading import Control_flimage
from time import sleep

FLIMageCont = Control_flimage()
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
 