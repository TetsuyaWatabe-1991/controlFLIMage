# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:18:17 2024

@author: yasudalab
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 08:12:32 2023

@author: yasudalab
"""

import time
import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")

sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\ongoing\ASIcontroller")

import pyvisa
from widefield_control import sendcommand_list, get_current_LED_MS2000_info
from ms2000_XYZ_fast import FastMS2000
from controlflimage_threading import Control_flimage

class Thorlab_PM100():
    
    def __init__(self):
        rm = pyvisa.ResourceManager()
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

wavelength = 920
Thor = Thorlab_PM100()
Thor.set_wavelength(wavelength)

setting = {
    "led_comport" : "COM8"
    }

FastMS2k = FastMS2000()
ini_path = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
FLIMageCont = Control_flimage(ini_path)


FLIMageCont.flim.sendCommand("State.Acq.XOffset = 0")
FLIMageCont.flim.sendCommand("State.Acq.YOffset = 0")

dict_command_and_sleeptime={"MOVE Z=1":0.5, "MOVE F=1":0.7}
sendcommand_list(comport = setting["led_comport"], 
                dict_command_and_sleeptime = dict_command_and_sleeptime)

FLIMageCont.flim_connect_check()
FLIMageCont.flim.sendCommand('StartGrab')
FLIMageCont.flim.sendCommand(f"State.Acq.power = [10, 0, 0, 0]")
FLIMageCont.flim.sendCommand(f"State.Acq.power = [100, 0, 0, 0]")

FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 1")
FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 0")
FLIMageCont.flim.sendCommand("State.Acq.zoom = 100")
FLIMageCont.flim_connect_check()
FLIMageCont.flim.sendCommand("FocusStart")
FLIMageCont.flim.sendCommand("StartFocus")
time.sleep(1)
FLIMageCont.flim.sendCommand("FocusStop")
FLIMageCont.flim.sendCommand("GrabStart")

for pow in [0,10,100]:   
    FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 0")
    FLIMageCont.flim.sendCommand(f"State.Acq.power = [{pow}, 0, 0, 0]\n")
    FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 1")
    time.sleep(1)
    print(Thor.read())

FLIMageCont.flim.sendCommand(f"State.Acq.power")
# if __name__ == "__main__":

#     Thor.set_wavelength(wavelength)
#     print(Thor.read())
