# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:58:43 2023

@author: yasudalab
"""

import sys
from controlflimage_threading import Control_flimage
from time import sleep
from multidim_tiff_viewer import define_uncagingPoint_dend_click_multiple

Zstack_ini=r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\z7_05_kal3.txt"
# Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128fast.txt"
singleplane_uncaging=r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\uncaging_singletime.txt"
direction_ini = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"

FLIMageCont = Control_flimage(ini_path=direction_ini)
# FLIMageCont.directionMotorY = FLIMageCont.directionMotorY*1
# FLIMageCont.directionGalvoY= FLIMageCont.directionGalvoY*-1

interval_sec = 20
align_ch_1or2 = 1
expected_acq_duration_sec = 12
pre_acquisition = 10
post_acquisition = 20

uncaging_power = 34


#######################################
# FIRST ACQUISITION

FLIMageCont.set_param(RepeatNum=1, interval_sec=interval_sec, ch_1or2=align_ch_1or2,
                      LoadSetting=True,SettingPath=Zstack_ini,
                      expected_grab_duration_sec=expected_acq_duration_sec)
FLIMageCont.start_repeat()

flim_file_path = FLIMageCont.flimlist[0]
spine_zyx, dend_slope, dend_intercept = define_uncagingPoint_dend_click_multiple(flim_file_path)


FLIMageCont.SpineHeadToUncaging_um=0.1
FLIMageCont.Spine_ZYX = spine_zyx
FLIMageCont.dend_slope = dend_slope
FLIMageCont.dend_intercept = dend_intercept



## FLIMageCont.Spine_ZYX = (2, 72, 87)
## FLIMageCont.dend_slope = -0.7499644512625536
## FLIMageCont.dend_intercept = dend_intercept

# #######################################
# #PRE ACQUISITION
FLIMageCont.set_param(RepeatNum = pre_acquisition - 1, interval_sec=interval_sec, ch_1or2=align_ch_1or2,
                LoadSetting=True,SettingPath=Zstack_ini,
                track_uncaging=False, drift_control=True,
                ShowUncagingDetection=False ,drift_cont_galvo=False,
                expected_grab_duration_sec=expected_acq_duration_sec)

FLIMageCont.start_repeat()

#######################################
# UNCAGING
FLIMageCont.go_to_uncaging_plane()
FLIMageCont.set_param(RepeatNum=32, interval_sec=2, ch_1or2=align_ch_1or2,
                    LoadSetting=True,SettingPath=singleplane_uncaging,
                    track_uncaging=True,drift_control=False,drift_cont_galvo=True,
                    ShowUncagingDetection=True,expected_grab_duration_sec=1.5,
                    num_no_uncaging_frames = 2)
FLIMageCont.set_uncaging_power(uncaging_power)

FLIMageCont.start_repeat_short(dend_slope_intercept = True)

FLIMageCont.back_to_stack_plane()



########################################
# POST ACQUISITION
FLIMageCont.set_param(RepeatNum=post_acquisition, interval_sec=interval_sec, ch_1or2=align_ch_1or2,
                    LoadSetting=True,SettingPath=Zstack_ini,
                    track_uncaging=False,drift_control=True,
                    ShowUncagingDetection=True,drift_cont_galvo=False,
                    expected_grab_duration_sec=expected_acq_duration_sec)
FLIMageCont.start_repeat()

FLIMageCont.flim.sendCommand(f'LoadSetting, {Zstack_ini}')

