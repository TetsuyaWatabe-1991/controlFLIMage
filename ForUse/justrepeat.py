# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:09:55 2023

@author: yasudalab
"""

import sys
sys.path.append("../")
from controlflimage_threading import Control_flimage

# Zstack_ini = r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\highmag_128_sum8_z1um_x7.txt"
direction_ini = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
FLIMageCont = Control_flimage(ini_path=direction_ini)

interval_sec = 15

align_ch_1or2 = 2
expected_acq_duration_sec = 10
repeatnum = 60
FLIMageCont.set_param(RepeatNum = repeatnum, interval_sec=interval_sec, 
                      ch_1or2=align_ch_1or2,
                      LoadSetting=False,
                    #   SettingPath=Zstack_ini,
                      track_uncaging=False,drift_control=True,
                      ShowUncagingDetection=False,drift_cont_galvo=False,
                      expected_grab_duration_sec=expected_acq_duration_sec)
FLIMageCont.start_repeat()




