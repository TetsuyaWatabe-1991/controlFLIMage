# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:08:39 2023

@author: yasudalab
"""

import os,glob,pathlib,math
from FLIMageAlignment import flim_files_to_nparray,Align_4d_array, align_two_flimfile
from FLIMageFileReader2 import FileReader
from multidim_tiff_viewer import multiple_uncaging_click, read_multiple_uncagingpos, read_dendriteinfo
from controlflimage_threading import control_flimage
import numpy as np
import copy
from time import sleep
import random
import cv2
from datetime import datetime
import matplotlib.pyplot as plt

class Low_High_mag_assign():
    def __init__(self, lowmag_path, highmag_path, ch_1or2 = 1,
                 skip_uncaging_pos = False):
        self.lowmag_path = lowmag_path
        self.highmag_path = highmag_path
        self.lowmag_basename = pathlib.Path(lowmag_path).stem[:-3]
        self.highmag_basename = pathlib.Path(highmag_path).stem[:-3]  
        
        self.lowmag_iminfo = FileReader()
        self.lowmag_iminfo.read_imageFile(self.lowmag_path, True) 
        
        self.highmag_iminfo = FileReader()
        self.highmag_iminfo.read_imageFile(self.highmag_path, True)
        
        self.lowmag_magnification = self.lowmag_iminfo.statedict['State.Acq.zoom']
        self.highmag_magnification = self.highmag_iminfo.statedict['State.Acq.zoom']
        
        self.lowmag_pos = copy.copy(self.lowmag_iminfo.statedict['State.Motor.motorPosition'])
        self.highmag_pos = copy.copy(self.highmag_iminfo.statedict['State.Motor.motorPosition'])
        
        self.ch = ch_1or2 -1
        self.Spine_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Spine_example.png"
        self.Dendrite_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Dendrite_example.png"
        
        self.cuboid_ZYX=[2,20,20]
        
        if skip_uncaging_pos==False:
            self.define_uncaging_pos()
        
    
    def latest_path(self,highlow="high"):
        high_flimlist = glob.glob(os.path.join(self.highmag_iminfo.statedict["State.Files.pathName"],
                                               self.highmag_basename+"[0-9][0-9][0-9].flim"))
        low_flimlist = glob.glob(os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"],
                                              self.lowmag_basename+"[0-9][0-9][0-9].flim"))
        
        high_maxcount = get_max_flimfiles(high_flimlist)
        low_maxcount = get_max_flimfiles(low_flimlist)
        
        if highlow=="low":
            latestpath = os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"], 
                                      self.lowmag_basename + str(low_maxcount).zfill(3) + ".flim")
        else:
            latestpath = os.path.join(self.highmag_iminfo.statedict["State.Files.pathName"], 
                                      self.highmag_basename + str(high_maxcount).zfill(3) + ".flim")
        return latestpath
        
    def define_uncaging_pos(self):
        
        Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray([self.highmag_path],ch=self.ch)
        FirstStack=Tiff_MultiArray[0]
        
        text = "Click the center of the spine you will stimulate. (Not the uncaging position itself)"
        z, ylist, xlist = multiple_uncaging_click(FirstStack,text,
                                        SampleImg=self.Spine_example,ShowPoint=False)
        maxproj_aroundZ=FirstStack[max(0,z-self.cuboid_ZYX[0]):min(FirstStack.shape[1]-1,z+self.cuboid_ZYX[0]+1),:,:].max(axis=0)
        self.Spine_ZYX=[z,ylist[0],xlist[0]]
        self.uncaging_x=xlist[0]
        self.uncaging_y=ylist[0]
        

        

    def count_flimfiles(self):
        high_flimlist = glob.glob(os.path.join(self.highmag_iminfo.statedict["State.Files.pathName"],
                                               self.highmag_basename+"[0-9][0-9][0-9].flim"))
        low_flimlist = glob.glob(os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"],
                                              self.lowmag_basename+"[0-9][0-9][0-9].flim"))
        
        self.low_counter = get_max_plus_one_flimfiles(low_flimlist)
        self.high_counter = get_max_plus_one_flimfiles(high_flimlist)
        
        self.low_max_plus1_flim = os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"], 
                                             self.lowmag_basename + str(self.low_counter).zfill(3) + ".flim")
        self.high_max_plus1_flim = os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"], 
                                             self.highmag_basename + str(self.high_counter).zfill(3) + ".flim")


    def send_acq_info(self, FLIMageCont, low_or_high):
        if low_or_high == "low":
            FLIMageCont.flim.sendCommand(f'LoadSetting, {self.lowmag_iminfo.statedict["State.Files.initFileName"]}')
            FLIMageCont.flim.sendCommand(f'State.Acq.power = {self.lowmag_iminfo.statedict["State.Acq.power"]}')
            FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{self.lowmag_basename}')
            FLIMageCont.flim.sendCommand(f'State.Acq.zoom = {self.lowmag_iminfo.statedict["State.Acq.zoom"]}')
        elif low_or_high == "high":
            FLIMageCont.flim.sendCommand(f'LoadSetting, {self.highmag_iminfo.statedict["State.Files.initFileName"]}')
            FLIMageCont.flim.sendCommand(f'State.Acq.power = {self.highmag_iminfo.statedict["State.Acq.power"]}')
            FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{self.highmag_basename}')
            FLIMageCont.flim.sendCommand(f'State.Acq.zoom = {self.highmag_iminfo.statedict["State.Acq.zoom"]}')
        else:
            raise Exception("Assign low or high to low_or_high")
 
    def send_acq_info_highlow(self, FLIMageCont, low_or_high, posnum,highmag = 15, lowmag = 1):
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = "pos{posnum}_{low_or_high}_')
        if low_or_high == "low":
            FLIMageCont.flim.sendCommand(f'LoadSetting, {self.lowmag_iminfo.statedict["State.Files.initFileName"]}')
            FLIMageCont.flim.sendCommand(f'State.Acq.zoom = {lowmag}')
        elif low_or_high == "high":
            FLIMageCont.flim.sendCommand(f'LoadSetting, {self.highmag_iminfo.statedict["State.Files.initFileName"]}')
            FLIMageCont.flim.sendCommand(f'State.Acq.zoom = {highmag}')
        else:
            raise Exception("Assign low or high to low_or_high")
            
            
    def update_pos_fromcurrent(self, FLIMageCont,low_or_high='low'):
        if low_or_high == "low": 
            self.lowmag_pos = FLIMageCont.get_position()
        elif low_or_high == "high":
            self.highmag_pos = FLIMageCont.get_position()
        else:
            print("No update.  Other than low or high was assigned to low_or_high")
    
    
    
    def makingTYX_from3d_and_2d(self, z, ch, TwoDpath = False, TwoDflim_Nth=-1):
        flimlist = glob.glob(os.path.join(self.highmag_iminfo.statedict["State.Files.pathName"],
                                               self.highmag_basename+"[0-9][0-9][0-9].flim"))
        
        if TwoDpath == False:
            TwoDpath = os.path.join(self.highmag_iminfo.statedict["State.Files.pathName"],
                                    self.highmag_basename+f"{str(get_max_flimfiles(flimlist)+1+TwoDflim_Nth).zfill(3)}.flim")
        
        
        firstTiff_MultiArray, _, _ = flim_files_to_nparray([self.highmag_path], ch = ch)
        lastTiff_MultiArray, _, _ = flim_files_to_nparray([TwoDpath], ch = ch)
        
        firstYX = np.array(firstTiff_MultiArray[0, z, :, :], dtype = np.uint16)
        lastYX = np.array(lastTiff_MultiArray[0, 0, :, :], dtype = np.uint16)
        TYX = np.array([firstYX,lastYX], dtype = np.uint16)
        return TYX        
        
        
        


def get_max_flimfiles(flimlist):
    counter = 1
    for eachflim in flimlist:
        try:   
            num = int(eachflim[-8:-5])
            if num > counter:
                counter = num
        except:
            pass
        
    return counter    
        
def get_max_plus_one_flimfiles(flimlist):
    counter = get_max_flimfiles(flimlist)
    counter+=1
    return counter

list_of_fileset = [['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230608\\test2\\pos1_low_002.flim', 
                    'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230608\\test2\\pos1_high_001.flim']]

LowHighset_instances = []

ch_1or2 = 1
uncagingpower = 15
uncaging_times = 5
acquisition_laser1power = 20
laser1power_duringuncaging = 20
for eachfileset in list_of_fileset:
    LowHighset_instances.append(Low_High_mag_assign(lowmag_path = eachfileset[0],
                                                    highmag_path =eachfileset[1], 
                                                    ch_1or2 = ch_1or2,
                                                    skip_uncaging_pos=True))
    # print(LowHighset_instances[-1].uncaging_x,LowHighset_instances[-1].uncaging_y)

FLIMageCont = control_flimage()
FLIMageCont.interval_sec = 60
print("Now Grabbing")

posnum = 3


# nthacquisiton = 0
each_lowhigh_instance = LowHighset_instances[0]
each_lowhigh_instance.highmag_iminfo.statedict["State.Acq.power"][0] = acquisition_laser1power


###Low magnification
FLIMageCont.interval_sec = 60
FLIMageCont.expected_grab_duration_sec = 7        

each_lowhigh_instance.count_flimfiles()
each_lowhigh_instance.send_acq_info(FLIMageCont, 'high')
each_lowhigh_instance.send_acq_info_highlow(FLIMageCont, low_or_high = 'high', posnum = posnum)
FLIMageCont.flim.sendCommand(f'State.Acq.power = {each_lowhigh_instance.highmag_iminfo.statedict["State.Acq.power"]}')
FLIMageCont.acquisition_include_connect_wait()
                


each_lowhigh_instance.count_flimfiles()
each_lowhigh_instance.send_acq_info_highlow(FLIMageCont, low_or_high = 'low', posnum = posnum)

FLIMageCont.acquisition_include_connect_wait()

