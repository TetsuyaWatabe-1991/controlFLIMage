# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:23:07 2022

@author: yasudalab
"""
import pyautogui as gui
import time
import os
from read_thorlabs_pm100 import Thorlab_PM100
# for i in range(10000):
#     print(gui.position())
#     time.sleep(0.1)

class LaserSettingAuto():
    
    def __init__(self, power_png=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Power.png"):
        self.power_png=power_png        
        
        self.power_btn = gui.locateCenterOnScreen(power_png)
        self.Laser1_tab = [self.power_btn[0] - 25, self.power_btn[1] - 26]
        self.Laser2_tab = [self.power_btn[0] + 20, self.power_btn[1] - 26]
        self.Power_percent = [self.power_btn[0] + 167, self.power_btn[1] + 7]
        self.Focus = [self.power_btn[0] - 23, self.power_btn[1] + 184]            
        self.zero_all()
        
    def gui_auto(self,laser_1or2,percent_list,interval=10):
        
        pos = self.Laser1_tab
        if laser_1or2==2:
            pos = self.Laser2_tab

        gui.click(pos)
        gui.click(self.Focus)
    
        gui.click(self.Power_percent)
                
        for power in percent_list:
            for i in range(3):
                gui.press("backspace")
                
            for i in str(power):
                gui.press(i)
                
            gui.press('enter')
            
            time.sleep(interval)

        gui.click(self.Focus)
        
    def change_power(self,laser_1or2,percent):
        
        pos = self.Laser1_tab
        if laser_1or2==2:
            pos = self.Laser2_tab

        gui.click(pos)
    
        gui.click(self.Power_percent)

        #Clear input box                
        for i in range(3):
            gui.press("backspace")
            
        for i in str(percent):
            gui.press(i)
            
        gui.press('enter')
    

    def gui_focus_abort(self):
        gui.click(self.Focus)
        

    def zero_all(self):
        percent_list=[0]
        interval=1
        for laser_1or2 in [1,2]:
            self.gui_auto(laser_1or2,percent_list,interval=interval)
            

def main1():
    LaserAuto = LaserSettingAuto()
    
    laser_1or2 = 1
    
    
    percent_list = [0,10,20,30,40,50]
    LaserAuto.gui_auto(laser_1or2,percent_list,interval=5)


def main2():
    wavelength =920
    laser_1or2 = 1
    percent_list = [0,10,20,30,40,50]
    
    Thor = Thorlab_PM100()
    Thor.set_wavelength(wavelength)
    
    
    LaserAuto = LaserSettingAuto()
    
    # LaserAuto.gui_focus_abort()
    print("OPEN SHUTTER BY HAND")
    
    for percent in percent_list:
        LaserAuto.change_power(laser_1or2,percent)
        time.sleep(3)
        print(Thor.read())
    print("CLOSE SHUTTER BY HAND")
    # LaserAuto.gui_focus_abort()

if __name__ == '__main__':
    main1()





