"""
Created on Fri Dec 23 17:23:07 2022

@author: yasudalab
"""
# %%
import time
import datetime
import json
import os
import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
import pyvisa
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from controlflimage_threading import Control_flimage

if False:
    import pyvisa
    rm = pyvisa.ResourceManager()
    rm.list_resources()
                                     
    my_instrument = rm.open_resource('USB0::0x1313::0x807B::250328403::INSTR')
    wavelength = 720
    print("Set wavelength to : ", wavelength, " nm")
    my_instrument.write(f'SENSE:CORR:WAV {wavelength}')

    print("Current wavelength: ", float(my_instrument.query('SENSE:CORR:WAV?')))
    
    power_mW =  1000*float(my_instrument.query('MEAS?'))

    print("Power (mW): ", round(power_mW,3))

class ThorlabPM100:
    """
    Handles communication with the Thorlab PM100 power meter.
    """
    def __init__(self, resource_addr='USB0::0x1313::0x807B::250328403::INSTR', sleep_sec=0.01):
        self.rm = pyvisa.ResourceManager()
        self.my_instrument = self.rm.open_resource(resource_addr)
        self.sleep_sec = sleep_sec

    def set_wavelength(self, wavelength):
        print(f"Set wavelength to : {wavelength} nm")
        self.my_instrument.write(f'SENSE:CORR:WAV {wavelength}')
        time.sleep(self.sleep_sec)
        print("Current wavelength: ", float(self.my_instrument.query('SENSE:CORR:WAV?')))
        time.sleep(self.sleep_sec)

    def set_zero(self):
        self.my_instrument.write('SENSE:CORR:COLL:ZERO')
        time.sleep(self.sleep_sec)

    def read(self):
        power_mW = 1000 * float(self.my_instrument.query('MEAS?'))
        time.sleep(self.sleep_sec)
        print("Current wavelength: ", float(self.my_instrument.query('SENSE:CORR:WAV?')))
        time.sleep(self.sleep_sec)
        print("Power (mW): ", round(power_mW, 3))
        return power_mW


class LaserSettingAuto:
    """
    Orchestrates laser power measurement using ThorlabPM100 and FLIMage remote control.
    """
    def __init__(self, flim_ini_path, pm_resource_addr=None):
        self.FLIMageCont = Control_flimage(flim_ini_path)
        self.FLIMageCont.flim.print_responses = False
        self.PM100D = ThorlabPM100(resource_addr=pm_resource_addr) if pm_resource_addr else ThorlabPM100()
        self.pow_result = {}
        self.zero_all()

    def set_power(self, laser_1or2, percent):
        """Set laser power via FLIMage remote control (SetPower command)."""
        self.FLIMageCont.flim.sendCommand(f"SetPower, {laser_1or2}, {percent}")

    def set_auto(self, laser_1or2, percent_list, interval=10, no_record=False):
        self.FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 0")
        self.FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 1")
        time.sleep(0.1)
        for power in percent_list:
            self.set_power(laser_1or2, power)
            self.FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 1")
            if laser_1or2 == 2:
                self.FLIMageCont.flim.sendCommand("SetDIOPanel, 3, 1")
            time.sleep(interval)
            power_mW = self.PM100D.read()
            self.FLIMageCont.flim.sendCommand("SetDIOPanel, 3, 0")
            self.FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 0")
            if not no_record:
                self.pow_result[power] = power_mW

    def change_power(self, laser_1or2, percent):
        self.set_power(laser_1or2, percent)

    def zero_all(self):
        percent_list = [0]
        interval = 0.5
        for laser_1or2 in [1, 2]:
            self.set_auto(laser_1or2, percent_list, interval=interval)

def main1():
    flim_ini_path = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
    LaserAuto = LaserSettingAuto(flim_ini_path)
    
    laser_1or2 = 1
    
    percent_list = [0,10,20,30,50,75,100]
    LaserAuto.set_auto(laser_1or2,percent_list,interval=5)


def do_measurement(LaserAuto,
                   wavelength =920,
                   laser_1or2 = 1,
                   percent_list = [0,10,20,30,50,75,100]):                    
    
    LaserAuto.PM100D.set_wavelength(wavelength)
    LaserAuto.set_auto(laser_1or2, percent_list, interval=1.5)
    print(f"Imaging laser, {wavelength} nm, Laser {laser_1or2}")
    print(LaserAuto.pow_result)
    copied_pow_result = LaserAuto.pow_result.copy()
    LaserAuto.pow_result = {}
    return copied_pow_result

def plot_and_save_results(pow_result_920, pow_result_720, savefolder):
    """
    Plot and save the measurement results for two lasers.
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4))

    # Laser 1
    x_laser1 = np.array(list(pow_result_920.keys())).reshape(-1, 1)
    y_laser1 = np.array(list(pow_result_920.values()))
    model = LinearRegression()
    model.fit(x_laser1, y_laser1)
    slope = model.coef_[0]
    intercept = model.intercept_
    y_laser1_pred = model.predict(x_laser1)
    r_squared_laser1 = r2_score(y_laser1, y_laser1_pred)
    eq_text_laser1 = f"y = {slope:.3f}x + {intercept:.4f}"
    ax0.scatter(x_laser1, y_laser1, color='blue', label="Data")
    ax0.plot(x_laser1, y_laser1_pred, color='red', label="Linear regression")
    ax0.text(0.1, 0.95, eq_text_laser1, ha='left', va="top", transform=ax0.transAxes)
    ax0.text(0.1, 0.85, f"r^2 = {r_squared_laser1:.3f}", ha='left', va="top", transform=ax0.transAxes)    
    ax0.set_xlabel('Power (%)')
    ax0.set_ylabel('Laser power (mW)')
    ax0.set_title('Laser1 920 nm')

    # Laser 2
    x_laser2 = np.array(list(pow_result_720.keys())).reshape(-1, 1)
    y_laser2 = np.array(list(pow_result_720.values()))
    
    model = LinearRegression()
    model.fit(x_laser2, y_laser2)
    slope = model.coef_[0]
    intercept = model.intercept_
    y_laser2_pred = model.predict(x_laser2)
    r_squared_laser2 = r2_score(y_laser2, y_laser2_pred)
    eq_text_laser2 = f"y = {slope:.3f}x + {intercept:.4f}"
    ax1.scatter(x_laser2, y_laser2, color='blue', label="Data")
    ax1.plot(x_laser2, y_laser2_pred, color='red', label="Linear regression")
    ax1.text(0.1, 0.95, eq_text_laser2, ha='left', va="top", transform=ax1.transAxes)
    ax1.text(0.1, 0.85, f"r^2 = {r_squared_laser2:.3f}", ha='left', va="top", transform=ax1.transAxes)
    ax1.set_xlabel('Power (%)')
    ax1.set_ylabel('Laser power (mW)')
    ax1.set_title('Laser2 720 nm')

    # Inverse fit for Laser 2
    model.fit(np.array(list(pow_result_720.values())).reshape(-1, 1), np.array(list(pow_result_720.keys())))
    inv_slope = model.coef_[0]
    inv_intercept = model.intercept_
    now = datetime.datetime.now()
    ax2.set_title("PM16-121 powermeter")
    ax2.text(0.1, 0.9, now.strftime("%Y-%m-%d 720 nm laser"), ha='left', va="top", transform=plt.gca().transAxes)
    offset = 0.2
    for mw in [5.0, 6.0, 7.0, 8.0, 9.0, 11, 13]:
        percent = inv_slope * mw + inv_intercept
        eachtext = f"{round(mw,1)} mW".rjust(7) + f"{round(percent,1)} %".rjust(8)
        ax2.text(0.1, 1 - offset, eachtext, ha='left', va="top", transform=plt.gca().transAxes)
        offset += 0.1
    ax2.axis("off")
    savepath = os.path.join(savefolder, now.strftime("%Y%m%d%H%M.png"))
    plt.savefig(savepath, dpi=300)
    plt.close();plt.clf();
    print("saved as  ", savepath)
    # Save JSON
    for each_res_dict in [pow_result_920, pow_result_720]:
        for each_key in each_res_dict:
            each_res_dict[each_key] = round(each_res_dict[each_key], 2)
    result_dict = {"Laser1": pow_result_920, "Laser2": pow_result_720}
    savejsonpath = os.path.join(savefolder, now.strftime("%Y%m%d%H%M.json"))
    with open(savejsonpath, "w") as outfile:
        json.dump(result_dict, outfile)
    os.system(f"start {savepath}")


def run_measurements(flim_ini_path, savefolder,
                    percent_list_1, percent_list_2,
                    pm_resource_addr=None):
    """
    Run the measurement workflow for two lasers and plot/save results.
    """
    laser_auto = LaserSettingAuto(flim_ini_path, pm_resource_addr=pm_resource_addr)
    pow_result_920 = do_measurement(laser_auto, wavelength=920, laser_1or2=1, percent_list=percent_list_1)
    laser_auto.zero_all()
    pow_result_720 = do_measurement(laser_auto, wavelength=720, laser_1or2=2, percent_list=percent_list_2)

    pow_offset_for_720 = do_measurement(laser_auto, wavelength=720, laser_1or2=2, percent_list=[0])
    offset_value = list(pow_offset_for_720.values())[0]
    pow_result_720 = {k: v - offset_value for k, v in pow_result_720.items()}
    plot_and_save_results(pow_result_920, pow_result_720, savefolder)
    return pow_result_920, pow_result_720


def main():
    """
    Main entry point for running the laser power measurement and plotting workflow.
    """
    flim_ini_path = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
    assert os.path.exists(flim_ini_path), f"FLIMage ini not found: {flim_ini_path}"
    savefolder = r"C:\Users\yasudalab\Documents\Tetsuya_Imaging\powermeter"
    percent_list_1 = [0, 10, 20, 30, 50, 70]
    percent_list_2 = [0, 10, 20, 30, 50, 70]
    
    run_measurements(flim_ini_path, savefolder, percent_list_1, percent_list_2)

# %%

if __name__ == '__main__':
    main()

    
# %%
