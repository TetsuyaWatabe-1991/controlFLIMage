# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:09:55 2023

@author: yasudalab
"""

import sys
sys.path.append("../")
from controlflimage_threading import control_flimage
from time import sleep

# Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\test.txt"
Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128.txt"
direction_ini = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"

FLIMageCont = control_flimage(ini_path=direction_ini)

# FLIMageCont.directionMotorY = FLIMageCont.directionMotorY*1
# FLIMageCont.directionGalvoY= FLIMageCont.directionGalvoY*1

interval_sec = 60
align_ch_1or2 = 1
expected_acq_duration_sec = 40
repeatnum = 100

FLIMageCont.set_param(RepeatNum = repeatnum, interval_sec=interval_sec, ch_1or2=align_ch_1or2,
                      LoadSetting=True,SettingPath=Zstack_ini,
                      track_uncaging=False,drift_control=True,
                      ShowUncagingDetection=False,drift_cont_galvo=False,
                      expected_grab_duration_sec=expected_acq_duration_sec)

FLIMageCont.start_repeat()
