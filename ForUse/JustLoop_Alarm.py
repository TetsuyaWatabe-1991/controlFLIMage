# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:28:44 2023

@author: yasudalab
"""
import sys
sys.path.append("../")
#%%
import time
from controlflimage_threading import control_flimage
from send_line import line_notification
import winsound


# while True:
#     winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
#     yn = input("Could you hear sound?  (y/n) ")
#     if yn == "y":
#         break


ini_path = r'C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini'

FLIMageCont = control_flimage(ini_path=ini_path)
FLIMageCont.directionMotorY = FLIMageCont.directionMotorY 


repeat_times = 16

interval_sec = 120
ch_1or2 = 2
expected_grab_duration_sec=20

FLIMageCont.set_param(RepeatNum=repeat_times, interval_sec=interval_sec, ch_1or2=ch_1or2,
                      LoadSetting=False,drift_control=True, 
                      drift_cont_XY =  False,
                      drift_cont_galvo=False,expected_grab_duration_sec=expected_grab_duration_sec)       
FLIMageCont.start_repeat()

line_notification(message = f"{repeat_times} loop  finished")

for i in range(1):
    winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

