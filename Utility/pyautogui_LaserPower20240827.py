# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:23:07 2022

@author: yasudalab
"""
import time
import datetime
import json
import os
import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
import pyvisa
import pyautogui as gui
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from controlflimage_threading import Control_flimage

class Thorlab_PM100():
    
    def __init__(self):
        rm = pyvisa.ResourceManager()
        if False:
            rm.list_resources()
        # self.my_instrument = rm.open_resource(rm.list_resources()[0])
        self.my_instrument = rm.open_resource('USB0::0x1313::0x8070::PM002347::INSTR')
        
    def set_wavelength(self, wavelength):
        self.my_instrument.write(f'CORR:WAV {wavelength}')
        time.sleep(0.2)

    def set_zero(self):
        self.my_instrument.write('CORR:COLL:ZERO')
        time.sleep(2)

    def read(self):
        power_mW =  1000*float(self.my_instrument.query('MEAS?'))
        time.sleep(1)
        return power_mW

class LaserSettingAuto():
    
    def __init__(self, power_png=r"Z:\Data Temp\Tetsuya\Power.png"):
        self.power_png=power_png        
        self.PM100D = Thorlab_PM100()
        self.pow_result = {}
        FLIMage_setting_ini = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
        self.FLIMageCont = Control_flimage(FLIMage_setting_ini)
        
        # window_remote = gw.getWindowsWithTitle("Remote control")[0]
        # window_remote.minimize()
        
        # window = gw.getWindowsWithTitle("FLIMage")[0]
        # window.activate()
        
        while True:
            try:
                self.power_btn = gui.locateOnScreen(power_png, confidence=0.80)
                break
            except:
                input("please activate FLIMage window and press any key to start")
        
            
        self.Laser1_tab = [self.power_btn[0] - 0, self.power_btn[1] - 20]
        self.Laser2_tab = [self.power_btn[0] + 45, self.power_btn[1] - 20]
        self.Focus = [self.power_btn[0] +2, self.power_btn[1] + 190]            
        self.Power_percent = [self.power_btn[0] + 192, self.power_btn[1] + 15]
        self.zero_all()
        
    def gui_auto(self,laser_1or2,percent_list,interval=10, no_record = False):
        self.FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 0")
        self.FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 1")
        
        time.sleep(1)
        if laser_1or2==1:
            pos = self.Laser1_tab
            self.FLIMageCont.flim.sendCommand("SetDIOPanel, 3, 0")
        if laser_1or2==2:
            pos = self.Laser2_tab
            self.FLIMageCont.flim.sendCommand("SetDIOPanel, 3, 1")
        
        gui.click(pos)
            
        gui.click(self.Power_percent)
                
        for power in percent_list:
                       
            for i in range(3):
                gui.press("backspace")
                
            for i in str(power):
                gui.press(i)
            gui.press('enter')
           
            time.sleep(0.2)
            self.FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 1")             
            
            if laser_1or2==2:
                pos = self.Laser2_tab
                time.sleep(0.2)
                self.FLIMageCont.flim.sendCommand("SetDIOPanel, 3, 1")

            time.sleep(interval)
            power_mW = self.PM100D.read()
            # print(power,power_mW)
            self.FLIMageCont.flim.sendCommand("SetDIOPanel, 3, 0")
            self.FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 0")
            if no_record == False:
                self.pow_result[power] = power_mW
            
        
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
        interval=0.5
        for laser_1or2 in [1,2]:
            self.gui_auto(laser_1or2, 
                          percent_list,
                          interval=interval)
            

def main1():
    LaserAuto = LaserSettingAuto()
    
    laser_1or2 = 1
    
    percent_list = [0,10,20,30,50,75,100]
    LaserAuto.gui_auto(laser_1or2,percent_list,interval=5)


def do_measurement(LaserAuto,
                   wavelength =920,
                   laser_1or2 = 1,
                   percent_list = [0,10,20,30,50,75,100]):                    
    
    LaserAuto.PM100D.set_wavelength(wavelength)
    LaserAuto.gui_auto(laser_1or2, percent_list, interval=2)
    print(f"Imaging laser, {wavelength} nm, Laser {laser_1or2}")
    print(LaserAuto.pow_result)
    copied_pow_result = LaserAuto.pow_result.copy()
    LaserAuto.pow_result = {}
    return copied_pow_result
    # for percent in percent_list:
    #     LaserAuto.change_power(laser_1or2,percent)
    #     time.sleep(3)
    #     print(Thor.read())
    # LaserAuto.gui_focus_abort()

if __name__ == '__main__':
    
    LaserAuto = LaserSettingAuto()
    pow_result_920 = do_measurement(LaserAuto,
                            wavelength =920,
                            laser_1or2 = 1,
                            percent_list = [0,10,20,30,50,75,100])
    
    LaserAuto.zero_all()
    pow_result_720 = do_measurement(LaserAuto,
                           wavelength =720,
                           laser_1or2 = 2,
                           percent_list = [0,10,20,30,50,70])
    

    
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (14,4))
    
    x_laser1 = np.array(list(pow_result_920.keys())).reshape(-1, 1)
    y_laser1 = np.array(list(pow_result_920.values()))
    
    model = LinearRegression()
    model.fit(x_laser1, y_laser1)  
    slope = model.coef_[0]
    intercept = model.intercept_
    
    y_laser1_pred = model.predict(x_laser1)
    r_squared = r2_score(y_laser1, y_laser1_pred)
    # Plot the results
    plusminus = intercept
    eq_text = f"y = {slope:.3f}x + {intercept:.4f}"
    
    ax0.scatter(x_laser1, y_laser1, color='blue', label="Data")
    ax0.plot(x_laser1, y_laser1_pred, color='red', label="Linear regression")
    ax0.set_xlabel('Power (%)')
    ax0.set_ylabel('Laser power (mW)')
    ax0.set_title('Laser1 920 nm')
    
    
    
    x_laser2 = np.array(list(pow_result_720.keys())).reshape(-1, 1)
    y_laser2 = np.array(list(pow_result_720.values()))
        
    model = LinearRegression()
    model.fit(x_laser2, y_laser2)  
    slope = model.coef_[0]
    intercept = model.intercept_
    
    y_laser2_pred = model.predict(x_laser2)
    r_squared = r2_score(y_laser2, y_laser2_pred)
    # Plot the results
    plusminus = intercept
    eq_text = f"y = {slope:.3f}x + {intercept:.4f}"
    
    ax1.scatter(x_laser2, y_laser2, color='blue', label="Data")
    ax1.plot(x_laser2, y_laser2_pred, color='red', label="Linear regression")
    ax1.text(0.1,0.95,eq_text, ha = 'left',va = "top", transform=ax1.transAxes)
    ax1.text(0.1,0.85,f"r^2 = {r_squared:.3f}", ha = 'left',va = "top", 
             transform=ax1.transAxes)
    ax1.set_xlabel('Power (%)')
    ax1.set_ylabel('Laser power (mW)')
    ax1.set_title('Laser2 720 nm')
    
    
    
    model.fit(np.array(list(pow_result_720.values())).reshape(-1, 1), 
              np.array(list(pow_result_720.keys())) )
    inv_slope = model.coef_[0]
    inv_intercept = model.intercept_
    
    # ax2.scatter(x, y, color='blue', label="Data")
    
    now = datetime.datetime.now()
    
    ax2.text(0.1,1,
             now.strftime("%Y-%m-%d 720 nm laser"), ha = 'left',va = "top", 
             transform=plt.gca().transAxes)
    offset = 0.2
    for mw in [0.7, 1.0, 1.3, 1.6, 2.0, 2.4, 2.8]:
        percent = inv_slope * mw + inv_intercept
        eachtext = f"{round(mw,1)} mW".rjust(7) + f"{round(percent,1)} %".rjust(8)
        ax2.text(0.1,1 - offset,
                 eachtext, ha = 'left',va = "top", 
                 transform=plt.gca().transAxes)
        offset += 0.1
    
    ax2.axis("off")
    
    savefolder = r"C:\Users\yasudalab\Documents\Tetsuya_Imaging\powermeter"
    savepath = os.path.join(savefolder,now.strftime("%Y%m%d%H%M.png"))
    plt.savefig(savepath, dpi = 300)
    plt.show()
    print("saved as  ",savepath)

    for each_res_dict in [pow_result_920,pow_result_720]:
        for each_key in each_res_dict:
            each_res_dict[each_key] = round(each_res_dict[each_key],2)

    result_dict = {
                "Laser1": pow_result_920,
                "Laser2": pow_result_720
                    }
    
    savejsonpath = os.path.join(savefolder,now.strftime("%Y%m%d%H%M.json"))
    with open(savejsonpath, "w") as outfile: 
        json.dump(result_dict, outfile)
