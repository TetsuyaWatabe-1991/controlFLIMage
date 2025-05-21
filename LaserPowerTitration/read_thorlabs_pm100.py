# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 08:12:32 2023

@author: yasudalab
"""
import pyvisa
import time

class Thorlab_PM100():
    
    def __init__(self):
        rm = pyvisa.ResourceManager()
        # self.my_instrument = rm.open_resource(rm.list_resources()[0])
        self.my_instrument = rm.open_resource('USB0::0x1313::0x8070::PM002347::INSTR')
        
        
    def set_wavelength(self, wavelength):
        self.my_instrument.write(f'CORR:WAV {wavelength}')
        self.my_instrument.write('CORR:COLL:ZERO')
        time.sleep(3)

    def read(self):
        power_mW =  1000*float(self.my_instrument.query('MEAS?'))
        time.sleep(1)
        return power_mW
    
if __name__ == "__main__":
    wavelength = 720
    Thor = Thorlab_PM100()
    Thor.set_wavelength(wavelength)
    print(Thor.read())