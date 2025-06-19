# %%

import sys
sys.path.append("../")
import glob
import json
import os
from time import sleep
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import winsound

from controlflimage_threading import Control_flimage
from flim_analysis_utils import (
    calculate_laser_power_settings,
    process_and_plot_flim_images
)

# File paths
DIRECTION_INI = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
POWERMETER_DIR = r"C:\Users\yasudalab\Documents\Tetsuya_Imaging\powermeter"

# Power conversion factor
FROM_THORLAB_TO_COHERENT_FACTOR = 1/3

# Acquisition parameters
UNCAGING_EVERY_SEC = 30
SEND_COUNTER = True
SEND_BASENAME = True

# Power calibration settings
MAX_POWER_PERCENTAGE = 100
POWER_CALIBRATION_WARNING_HOURS = 48

# Manual power calibration settings
USE_MANUAL_POWER_CALIBRATION = False  # Set to True to always use manual input

# Plotting modes
PLOT_TYPE_MULTIPLE = 'multiple'  # Plot all images in one figure
PLOT_TYPE_SINGLE = 'single'      # Plot each image individually

# Wait times
WAIT_BETWEEN_SETTINGS_SEC = 13
WAIT_BETWEEN_ACQUISITIONS_SEC = 30

def get_manual_power_calibration():
    """
    Get power calibration values from user input.
    
    Returns:
        tuple: (power_slope, power_intercept)
    """
    print("\n=== Manual Power Calibration Input ===")
    print("Please enter the power calibration values:")
    
    while True:
        try:
            slope_input = input("Enter power slope (e.g., 0.158): ").strip()
            power_slope = float(slope_input)
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    while True:
        try:
            intercept_input = input("Enter power intercept (e.g., 0.139): ").strip()
            power_intercept = float(intercept_input)
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    print(f"Using manual calibration: slope = {power_slope}, intercept = {power_intercept}")
    return power_slope, power_intercept

def load_power_calibration():
    """
    Load power calibration from the latest JSON file or get manual input.
    
    Returns:
        tuple: (power_slope, power_intercept)
    """
    # Check if manual calibration is requested
    if USE_MANUAL_POWER_CALIBRATION:
        return get_manual_power_calibration()
    
    try:
        latest_json_path = glob.glob(os.path.join(POWERMETER_DIR, "*.json"))[-1]
        
        latest_json_basename = os.path.basename(latest_json_path).replace(".json", "")
        latest_json_datetime = datetime.strptime(latest_json_basename, "%Y%m%d%H%M")
        
        now = datetime.now()
        delta_sec = int((now - latest_json_datetime).total_seconds())
        delta_hours = delta_sec // 3600
        delta_minutes = (delta_sec % 3600) // 60
        
        print(f"Power calibrated {delta_hours} hr {delta_minutes} min ago")
        if delta_hours > POWER_CALIBRATION_WARNING_HOURS:
            print(f"Power calibration is more than {POWER_CALIBRATION_WARNING_HOURS} hours ago")
            response = input("Continue with JSON calibration or use manual input? (json/manual): ").strip().lower()
            if response == 'manual':
                return get_manual_power_calibration()
        
        with open(latest_json_path, "r") as f:
            data = json.load(f)
        
        x_laser = np.array(list(data["Laser2"].keys())).reshape(-1, 1).astype(float)
        y_laser = np.array(list(data["Laser2"].values())).astype(float)
        model = LinearRegression()
        model.fit(x_laser, y_laser)
        power_slope = model.coef_[0]
        power_intercept = model.intercept_
        
        print(f"Laser2 slope = {round(power_slope, 3)}, intercept = {round(power_intercept, 3)}")
        return power_slope, power_intercept
        
    except (IndexError, FileNotFoundError, KeyError) as e:
        print(f"Could not load power calibration from JSON: {e}")
        print("Switching to manual power calibration input...")
        return get_manual_power_calibration()


# %%
# Alternative: Run with single setting path (like the original fivepulse script)
def run_single_setting_experiment(SETTINGPATH_LIST, experiment_basename_prefix):
    """
    Run experiment with a single setting path (original fivepulse behavior).
    """
    print("=== Single Setting Experiment ===")
    
    # Load power calibration
    power_slope, power_intercept = load_power_calibration()
    
    # Use only the first setting path
    setting_path = SETTINGPATH_LIST[0]
    
    # Run experiment
    results = run_titration_experiment(
        setting_path, power_slope, power_intercept,
        plot_type=PLOT_TYPE_SINGLE,  # Use single plot mode like original
        basename_prefix=f"{experiment_basename_prefix}single_"
    )
    
    # Play completion sound
    for i in range(1):
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
    
    print("=== Single setting experiment completed ===")


def run_titration_experiment(FLIMageCont, setting_path, power_slope, power_intercept, vmin, vmax,laser_mw_ms,
                           plot_type=PLOT_TYPE_MULTIPLE, basename_prefix=""):
    """
    Run titration experiment with a specific setting path.
    
    Args:
        setting_path: Path to FLIMage setting file
        power_slope: Slope of power calibration
        power_intercept: Intercept of power calibration
        plot_type: 'single' or 'multiple' plotting mode
        basename_prefix: Prefix for file basename
    """
    print(f"\n=== Running experiment with setting: {os.path.basename(setting_path)} ===")
    
    # Load setting
    FLIMageCont.flim.sendCommand(f'LoadSetting, {setting_path}')
    
    # Set basename if provided
    if basename_prefix and SEND_BASENAME:
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{basename_prefix}"')
    if SEND_COUNTER:
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = 1')
    
    # Calculate power settings
    unc_pow_dur = calculate_laser_power_settings(
        laser_mw_ms, power_slope, power_intercept, 
        FROM_THORLAB_TO_COHERENT_FACTOR, MAX_POWER_PERCENTAGE
    )
    
    print(f"Power settings: {unc_pow_dur}")
    if len(laser_mw_ms) != len(unc_pow_dur):
        print("laser_mW_ms and unc_pow_dur are not the same length")
        input("press enter to continue..")
    
    # Store results for this setting
    result_img_dict = {}
    
    # Run titration
    for each_pow_dur in unc_pow_dur:
        each_pow = each_pow_dur[0]
        each_dur = each_pow_dur[1]
        
        print(f"Setting power: {each_pow}%, duration: {each_dur}ms")
        
        FLIMageCont.set_uncaging_power(each_pow)
        sleep(0.1)
        FLIMageCont.flim.sendCommand(f'State.Uncaging.pulseWidth = {each_dur}')
        sleep(0.1)
        
        tic = datetime.now()
        
        # Start acquisition
        FLIMageCont.acquisition_include_connect_wait()
        
        # Get file path
        a = FLIMageCont.flim.sendCommand('GetFullFileName')
        one_file = a[a.find(",")+2:]
        
        # Process and plot images
        from FLIMageAlignment import get_flimfile_list
        filelist = get_flimfile_list(one_file)
        
        result_img_dict = process_and_plot_flim_images(
            filelist=filelist, 
            power_slope=power_slope, 
            power_intercept=power_intercept, 
            F_F0_vmin=vmin, 
            F_F0_vmax=vmax,
            result_img_dict=result_img_dict, 
            from_Thorlab_to_coherent_factor=FROM_THORLAB_TO_COHERENT_FACTOR,
            plot_type=plot_type
        )
        
        # Wait between acquisitions (except for the last one)
        if each_pow_dur != unc_pow_dur[-1]:
            toc = datetime.now()
            delta_sec = (toc - tic).total_seconds()
            sleep_time = max(0, UNCAGING_EVERY_SEC - delta_sec)
            print(f"Waiting {sleep_time:.1f} seconds...")
            for i in range(int(sleep_time)):
                print(f"{int(sleep_time) - i} ", end="")
                sleep(1)
            print()
    
    return result_img_dict

print("Experiment functions defined!")

if __name__ == "__main__":
    # Initialize FLIMage controller
    if "FLIMageCont" not in globals():
        FLIMageCont = Control_flimage(ini_path=DIRECTION_INI)

    # Load power calibration
    power_slope, power_intercept = load_power_calibration()
    print(f"\nFinal calibration values: slope = {power_slope}, intercept = {power_intercept}")

    # Laser power titration settings (mW, ms)
    LASER_MW_MS = [
        # [0.1, 6],
        # [0.2, 6],
        # [0.4, 6],
        # [2.4, 6],
        [2.8, 6],
        # [3.3, 6],
        # [4.0, 6],
        # [5.0, 6],
        # [6.5, 6],
    ]


    root = tk.Tk()
    root.geometry("600x200")
    root.withdraw()
    EXPERIMENT_BASENAME_PREFIX = simpledialog.askstring("Input", "Enter basename prefix:", 
                                                        initialvalue="titration_dend0_0um_")
    root.destroy()

    # Multiple setting paths for different experiments
    SETTINGPATH_LIST = [
        r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\uncaging_2times_no_ave.txt",
        r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\twopulses.txt",
        r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\fivepulses.txt",
        # r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\LTP_by_fivepulses50Hz.txt"
    ]
    for each_setting_path in SETTINGPATH_LIST:
        if not os.path.exists(each_setting_path):
            print(f"Setting path {each_setting_path} does not exist")
            input("press enter to continue..")
    

    # Plotting parameters
    VMIN = 1
    VMAX = 6


    print("=== GCaMP Uncaging Titration Experiment ===")

    # Run experiments for each setting path
    all_results = {}

    for i, setting_path in enumerate(SETTINGPATH_LIST):
        # Create basename prefix for this setting
        setting_name = os.path.splitext(os.path.basename(setting_path))[0]
        basename_prefix = f"{EXPERIMENT_BASENAME_PREFIX}{setting_name}_"
        
        # Run experiment with this setting
        results = run_titration_experiment(
            FLIMageCont = FLIMageCont,
            setting_path = setting_path, 
            power_slope = power_slope, 
            power_intercept = power_intercept, 
            vmin=VMIN, vmax=VMAX,
            laser_mw_ms=LASER_MW_MS,
            plot_type=PLOT_TYPE_MULTIPLE,  # Use multiple plot mode for each setting
            basename_prefix=basename_prefix
        )
        
        all_results[setting_path] = results
        
        # Wait between different settings (except for the last one)
        if setting_path != SETTINGPATH_LIST[-1]:
            print(f"\nWaiting {WAIT_BETWEEN_SETTINGS_SEC} seconds before next setting...")
            for i in range(WAIT_BETWEEN_SETTINGS_SEC):
                print(f"{WAIT_BETWEEN_SETTINGS_SEC - i} sec ", end="")
                sleep(1)
            print()

    # Play completion sound
    for i in range(1):
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

    print("\n=== Experiment completed ===")

    # Print summary
    print("\nPower settings used:")
    unc_pow_dur = calculate_laser_power_settings(
        LASER_MW_MS, power_slope, power_intercept, 
        FROM_THORLAB_TO_COHERENT_FACTOR, MAX_POWER_PERCENTAGE
    )
    for each_pow_dur, each_mW in zip(unc_pow_dur, LASER_MW_MS):
        print(f"{each_mW[0]} mW   {each_pow_dur[0]} %")

