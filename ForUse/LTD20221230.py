# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:13:42 2022

@author: yasudalab
"""
import sys
sys.path.append("../")
from controlflimage_threading import control_flimage

singleplane_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep05_128_singleplane.txt"
Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128.txt"
# Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep05_128.txt"
singleplane_uncaging=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging.txt"
# singleplane_uncaging=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging_test.txt"
Zstack_noAve_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128_Ave2.txt"


FLIMageCont = control_flimage()

FLIMageCont.set_param(RepeatNum=1, interval_sec=60, ch_1or2=2,
                      LoadSetting=True,SettingPath=Zstack_ini,
                      expected_grab_duration_sec=20)

FLIMageCont.start_repeat()

FLIMageCont.define_uncagingPoint()

# FLIMageCont.Spine_ZYX= [4, 89, 58]
# FLIMageCont.Dendrite_ZYX= [0, 83, 57] 

FLIMageCont.set_param(RepeatNum=10, interval_sec=60, ch_1or2=2,
                      LoadSetting=True,SettingPath=Zstack_ini,
                      track_uncaging=True,drift_control=True,
                      ShowUncagingDetection=True,drift_cont_galvo=True,expected_grab_duration_sec=40)        
FLIMageCont.start_repeat()


for i in range(15):
    FLIMageCont.go_to_uncaging_plane()
    
    FLIMageCont.set_param(RepeatNum=6, interval_sec=10, ch_1or2=2,
                          LoadSetting=True,SettingPath=singleplane_uncaging,
                          track_uncaging=True,drift_control=False,drift_cont_galvo=True,
                          ShowUncagingDetection=True,DoUncaging=False,expected_grab_duration_sec=1.5)
    FLIMageCont.start_repeat_short()

    FLIMageCont.back_to_stack_plane()
    FLIMageCont.set_param(RepeatNum=1, interval_sec=30, ch_1or2=2,
                          LoadSetting=True,SettingPath=Zstack_noAve_ini,
                          track_uncaging=True,drift_control=True,
                          ShowUncagingDetection=True,drift_cont_galvo=True,expected_grab_duration_sec=9)        
    FLIMageCont.start_repeat()


FLIMageCont.set_param(RepeatNum=60, interval_sec=60, ch_1or2=2,
                      LoadSetting=True,SettingPath=Zstack_ini,
                      track_uncaging=True,drift_control=True,
                      ShowUncagingDetection=True,drift_cont_galvo=True,expected_grab_duration_sec=40)        
FLIMageCont.start_repeat()


# FLIMageCont.flim.sendCommand(f'LoadSetting, {Zstack_ini}')

