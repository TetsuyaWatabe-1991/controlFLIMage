# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:54:43 2023

@author: yasudalab
"""
import sys
sys.path.append("../")
from controlflimage_threading import control_flimage
from time import sleep

singleplane_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep05_128_singleplane.txt"
# Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128.txt"
# Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep05_128.txt".
Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128fast.txt"
singleplane_uncaging=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging.txt"
# singleplane_uncaging=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging_test.txt"

FLIMageCont = control_flimage()
FLIMageCont.directionMotorY = FLIMageCont.directionMotorY * (-1)  #If you check Flip Y-axis

FLIMageCont.set_param(RepeatNum=5, interval_sec=30, ch_1or2=2,
                      LoadSetting=True,SettingPath=Zstack_ini)
FLIMageCont.start_repeat()

FLIMageCont.define_uncagingPoint()

# FLIMageCont.Spine_ZYX =[5, 61, 61]
# FLIMageCont.Dendrite_ZYX = [0, 56, 65] 

FLIMageCont.set_param(RepeatNum=3, interval_sec=30, ch_1or2=2,
                      LoadSetting=True,SettingPath=Zstack_ini,
                      track_uncaging=True,drift_control=True,
                      ShowUncagingDetection=True,drift_cont_galvo=True,expected_grab_duration_sec=22)        

FLIMageCont.start_repeat()

FLIMageCont.go_to_uncaging_plane()

FLIMageCont.set_param(RepeatNum=30, interval_sec=2, ch_1or2=2,
                      LoadSetting=True,SettingPath=singleplane_uncaging,
                      track_uncaging=True,drift_control=False,drift_cont_galvo=True,
                      ShowUncagingDetection=True,DoUncaging=False,expected_grab_duration_sec=1.5)
sleep(5)


FLIMageCont.start_repeat_short()

FLIMageCont.back_to_stack_plane()

FLIMageCont.set_param(RepeatNum=15, interval_sec=30, ch_1or2=2,
                      LoadSetting=True,SettingPath=Zstack_ini,
                      track_uncaging=True,drift_control=True,
                      ShowUncagingDetection=True,drift_cont_galvo=True,expected_grab_duration_sec=22)    
FLIMageCont.start_repeat()

FLIMageCont.flim.sendCommand(f'LoadSetting, {Zstack_ini}')