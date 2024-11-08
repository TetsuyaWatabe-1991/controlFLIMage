# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:58:43 2023

@author: yasudalab
"""

import sys
from controlflimage_threading import control_flimage
from time import sleep
from multidim_tiff_viewer import define_uncagingPoint_dend_click_multiple

Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128.txt"
# Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128fast.txt"
singleplane_uncaging=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging.txt"
direction_ini = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"

FLIMageCont = control_flimage(ini_path=direction_ini)
# FLIMageCont.directionMotorY = FLIMageCont.directionMotorY*1
# FLIMageCont.directionGalvoY= FLIMageCont.directionGalvoY*-1


interval_sec = 60
align_ch_1or2 = 1
expected_acq_duration_sec = 40
pre_acquisition = 5
post_acquisition = 50

uncaging_power = 34


#######################################
# FIRST ACQUISITION
FLIMageCont.set_param(RepeatNum=1, interval_sec=interval_sec, ch_1or2=align_ch_1or2,
                      expected_grab_duration_sec=expected_acq_duration_sec)
FLIMageCont.start_repeat()

flim_file_path = FLIMageCont.flimlist[0]
spine_zyx, dend_slope, dend_intercept = define_uncagingPoint_dend_click_multiple(flim_file_path)

FLIMageCont.spine_ZYX = spine_zyx
FLIMageCont.dend_slope = dend_slope
FLIMageCont.intercept = dend_intercept
#######################################
#PRE ACQUISITION
FLIMageCont.set_param(RepeatNum = pre_acquisition - 1, interval_sec=interval_sec, ch_1or2=align_ch_1or2,
                LoadSetting=True,SettingPath=Zstack_ini,
                track_uncaging=True,drift_control=True,
                ShowUncagingDetection=True,drift_cont_galvo=True,expected_grab_duration_sec=expected_acq_duration_sec)

FLIMageCont.start_repeat()



#######################################
# UNCAGING
FLIMageCont.go_to_uncaging_plane()
FLIMageCont.set_param(RepeatNum=30, interval_sec=2, ch_1or2=align_ch_1or2,
                    LoadSetting=True,SettingPath=singleplane_uncaging,
                    track_uncaging=True,drift_control=False,drift_cont_galvo=True,
                    ShowUncagingDetection=True,DoUncaging=False,expected_grab_duration_sec=1.5)
sleep(5)
FLIMageCont.set_uncaging_power(uncaging_power)
sleep(0.5)


FLIMageCont.start_repeat_short(dend_slope_intercept = True)
FLIMageCont.back_to_stack_plane()

########################################
# POST ACQUISITION
FLIMageCont.set_param(RepeatNum=post_acquisition, interval_sec=interval_sec, ch_1or2=align_ch_1or2,
                    LoadSetting=True,SettingPath=Zstack_ini,
                    track_uncaging=True,drift_control=True,
                    ShowUncagingDetection=True,drift_cont_galvo=True,
                    expected_grab_duration_sec=expected_acq_duration_sec)

FLIMageCont.start_repeat()

# FLIMageCont.flim.sendCommand(f'LoadSetting, {Zstack_ini}')
