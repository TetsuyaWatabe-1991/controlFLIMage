# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:28:44 2023

@author: yasudalab
"""
import sys
sys.path.append("../")
#%%
import time
from controlflimage_threading import Control_flimage
from send_line import line_notification
import winsound


## soundtest
# while True:
#     winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
#     yn = input("Could you hear sound?  (y/n) ")
#     if yn == "y":
#         break

ini_path = r'C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini'

Zstack_ini = r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\z7_10_kal8.txt"

FLIMageCont = Control_flimage(ini_path=ini_path)
FLIMageCont.directionMotorY = FLIMageCont.directionMotorY 


# sleeptime_sec = 60
# for i in range(sleeptime_sec):
#     print("  remaining sec: ",sleeptime_sec - i, sep='   ')
#     time.sleep(1)


repeat_times = 25
interval_sec = 60

# repeat_times = 3
# interval_sec = 60* 1


ch_1or2 = 1

expected_grab_duration_sec=20


FLIMageCont.set_param(RepeatNum=repeat_times,
                      interval_sec=interval_sec,
                      ch_1or2=ch_1or2,
                      expected_grab_duration_sec=expected_grab_duration_sec,
                      LoadSetting=True,
                      SettingPath=Zstack_ini,
                      drift_control=True,
                      drift_cont_XY = True,
                      drift_cont_galvo= True
                      )

FLIMageCont.start_repeat()


# if repeat_times>10:
line_notification(message = f"{repeat_times} loop  finished")

for i in range(1):
    winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

