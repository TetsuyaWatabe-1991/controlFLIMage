# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:25:52 2025

@author: yasudalab
"""
from time import sleep
import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
from controlflimage_threading import Control_flimage

ini_path = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
FLIMageCont = Control_flimage(ini_path)




uncaging_pow_list = [5, 10, 15, 20, 25]

for each_power in uncaging_pow_list:
    
    FLIMageCont.set_uncaging_power(each_power)

    FLIMageCont.expected_grab_duration_sec = 5
    FLIMageCont.interval_sec = 10

    FLIMageCont.flim_connect_check()
    
    FLIMageCont.flim.sendCommand('StartGrab')  

    FLIMageCont.wait_while_grabbing()
    sleep(2)