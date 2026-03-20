import sys
sys.path.append("..")
from controlflimage_threading import Control_flimage
from time import sleep
from datetime import datetime


sleep_time = 5*60
ini_path = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
FLIMageCont = Control_flimage(ini_path)
FLIMageCont.check_pulse_rate = False

threshold_voltage_list = []
for i in range(11):
    threshold_voltage_list.append(-i*10)


for nth_acq in range(200):
    start_time = datetime.now()
    for voltage in threshold_voltage_list:
        FLIMageCont.flim.sendCommand(f"State.Spc.spcData.sync_threshold = {voltage}")
        FLIMageCont.acquisition_include_connect_wait()
        sleep(1)
    end_time = datetime.now()
    print(f"Time taken: {end_time - start_time}")
    print("let me sleep for ", sleep_time - (end_time - start_time).total_seconds())
    sleep(sleep_time - (end_time - start_time).total_seconds())


