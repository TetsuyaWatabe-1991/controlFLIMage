# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:28:44 2023

@author: yasudalab
"""
# import sys
# sys.path.append("../")
from controlflimage_threading import control_flimage

FLIMageCont = control_flimage()
FLIMageCont.directionMotorY = FLIMageCont.directionMotorY 

# FLIMageCont.set_param(RepeatNum=80, interval_sec=60, ch_1or2=2,
#                       LoadSetting=False,drift_control=True,
#                       ShowUncagingDetection=True,drift_cont_galvo=False,expected_grab_duration_sec=40) 

FLIMageCont.set_param(RepeatNum=800, interval_sec=120, ch_1or2=1,
                      LoadSetting=False,drift_control=True,
                      ShowUncagingDetection=True,drift_cont_galvo=False,expected_grab_duration_sec=20)       
FLIMageCont.start_repeat()


 