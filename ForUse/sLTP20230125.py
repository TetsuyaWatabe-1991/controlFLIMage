# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:58:43 2023

@author: yasudalab
"""


import sys
sys.path.append("../")
from controlflimage_threading import control_flimage
from time import sleep


# Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128.txt"
Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128fast.txt"
singleplane_uncaging=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging.txt"
direction_ini = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"

FLIMageCont = control_flimage(ini_path=direction_ini)
FLIMageCont.directionMotorY = FLIMageCont.directionMotorY*-1
FLIMageCont.directionGalvoY= FLIMageCont.directionGalvoY*-1


interval_sec = 60 
align_ch_1or2 = 1
expected_acq_duration_sec = 4
pre_acquisition = 3
post_acquisition = 50

#######################################
# FIRST ACQUISITION
FLIMageCont.set_param(RepeatNum=1, interval_sec=interval_sec, ch_1or2=1,
                      LoadSetting=True,SettingPath=Zstack_ini)#,expected_grab_duration_sec=expected_acq_duration_sec)
FLIMageCont.start_repeat()
# FLIMageCont.define_uncagingPoint()

# FLIMageCont.Spine_ZYX =[5, 61, 61]
# FLIMageCont.Dendrite_ZYX = [0, 56, 65] 
# FLIMageCont.Spine_ZYX = [3, 50, 84]
# FLIMageCont.Dendrite_ZYX= [0, 57, 73] 
FLIMageCont.Spine_ZYX= [3, 81, 78]
FLIMageCont.Dendrite_ZYX= [0, 76, 77] 

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
FLIMageCont.start_repeat_short()

FLIMageCont.back_to_stack_plane()


########################################
# POST ACQUISITION
FLIMageCont.set_param(RepeatNum=post_acquisition, interval_sec=interval_sec, ch_1or2=align_ch_1or2,
                      LoadSetting=True,SettingPath=Zstack_ini,
                      track_uncaging=True,drift_control=True,
                      ShowUncagingDetection=True,drift_cont_galvo=True,expected_grab_duration_sec=expected_acq_duration_sec)    

FLIMageCont.start_repeat()

# FLIMageCont.flim.sendCommand(f'LoadSetting, {Zstack_ini}')
