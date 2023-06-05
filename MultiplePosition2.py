# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:24:13 2023

@author: WatabeT
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
        
        self.low_counter_flim = os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"], 
                                             self.lowmag_basename + str(self.low_counter).zfill(3) + ".flim")
        self.high_counter_flim = os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"], 
                                             self.highmag_basename + str(self.high_counter).zfill(3) + ".flim")


    def send_acq_info(self, FLIMageCont, low_or_high):
        if low_or_high == "low":
            FLIMageCont.flim.sendCommand(f'LoadSetting, {self.self.statedict["State.Files.initFileName"]}')
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
                                    self.highmag_basename+"{str(get_max_flimfiles(flimlist)+1+TwoDflim_Nth).zfill(3)}.flim")
        
        
        firstTiff_MultiArray, _, _ = flim_files_to_nparray([self.highmag_path], ch = ch)
        lastTiff_MultiArray, _, _ = flim_files_to_nparray([TwoDpath], ch = ch)
        
        firstYX = firstTiff_MultiArray[0, z, :, :]
        lastYX = lastTiff_MultiArray[0, 0, :, :]
        TYX = np.array([firstYX,lastYX])
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

    
singleplane_uncaging=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging.txt"

list_of_fileset = [[r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\test1\pos6_lowmag_002.flim",
                    r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\test1\pos6_highmag_001.flim"],
                    [r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\test1\pos7_lowmag_004.flim",
                    r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\test1\pos7_highmag_003.flim"],
                    [r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\test1\pos8_lowmag_006.flim",
                    r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\test1\pos8_highmag_005.flim"]]


LowHighset_instances = []
                   
for eachfileset in list_of_fileset:
    LowHighset_instances.append(Low_High_mag_assign(lowmag_path = eachfileset[0],
                                                    highmag_path =eachfileset[1], 
                                                    ch_1or2 = 1,
                                                    skip_uncaging_pos=True))
    # print(LowHighset_instances[-1].uncaging_x,LowHighset_instances[-1].uncaging_y)

FLIMageCont = control_flimage()
FLIMageCont.interval_sec = 60
print("Now Grabbing")

num_T = 50
FLIMageCont.expected_grab_duration_sec = 10

for nthacquisiton in range(num_T):
    print(f"ACQUISTION, {nthacquisiton+1}/{num_T}")
    
    for ind, each_lowhigh_instance in enumerate(LowHighset_instances):
        # nthacquisiton = 0
        # each_lowhigh_instance = LowHighset_instances[0]
        
        ###Low magnification
        each_lowhigh_instance.count_flimfiles()
        each_lowhigh_instance.send_acq_info(FLIMageCont, 'low')
        
        dest_x,dest_y,dest_z = each_lowhigh_instance.lowmag_pos
        FLIMageCont.go_to_absolute_pos_motor_checkstate(dest_x,dest_y,dest_z)
        
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {each_lowhigh_instance.low_counter}')
        FLIMageCont.acquisition_include_connect_wait()

        FLIMageCont.relative_zyx_um, FLIMageCont.Aligned_4d_array = align_two_flimfile(
                                                            each_lowhigh_instance.lowmag_path, 
                                                            each_lowhigh_instance.low_counter_flim, 
                                                            each_lowhigh_instance.lowmag_iminfo,
                                                            each_lowhigh_instance.ch)
        print(FLIMageCont.relative_zyx_um)
        FLIMageCont.go_to_relative_pos_motor_checkstate()
        each_lowhigh_instance.update_pos_fromcurrent(FLIMageCont)
        each_lowhigh_instance.send_acq_info(FLIMageCont, 'high')
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {each_lowhigh_instance.high_counter}')
        FLIMageCont.acquisition_include_connect_wait()
        
        do_uncaging = False
        
        if do_uncaging:
            FLIMageCont.relative_zyx_um, FLIMageCont.Aligned_4d_array = align_two_flimfile(each_lowhigh_instance.highmag_path,
                                                                 each_lowhigh_instance.high_counter_flim,
                                                                 each_lowhigh_instance.highmag_iminfo,
                                                                 each_lowhigh_instance.ch)
            print("align high mag frames ", FLIMageCont.relative_zyx_um)
            FLIMageCont.go_to_relative_pos_motor_checkstate()
            
            uncaging_Z, uncaging_Ylist, uncaging_Xlist = read_multiple_uncagingpos(each_lowhigh_instance.highmag_path)
            direction_list, orientation_list, _, _ = read_dendriteinfo(each_lowhigh_instance.highmag_path)
            FLIMageCont.go_to_uncaging_plane_z_assign(uncaging_Z)
            
            # take single plane image
            FLIMageCont.flim.sendCommand("State.Acq.nSlices = 1")
            FLIMageCont.acquisition_include_connect_wait()
            
            #This will assign TYX array in FLIMageCont.Aligned_TYX_array
            FLIMageCont.makingTYX_from3d_and_2d(first_flim=each_lowhigh_instance.highmag_path,
                                                TwoDflim_Nth=-1,
                                                z=uncaging_Z,
                                                ch=each_lowhigh_instance.ch
                                                )
            
            modified_uncaging_xlist = []
            modified_uncaging_ylist = []
            
            for uncaging_X, uncaging_Y, direction, dend_orientation in zip(uncaging_Xlist, 
                                                                           uncaging_Ylist,
                                                                           direction_list,
                                                                           orientation_list):
                FLIMageCont.Spine_ZYX = [uncaging_Z, uncaging_Y, uncaging_X]
                FLIMageCont.Aligned_TYX_array = each_lowhigh_instance.makingTYX_from3d_and_2d(uncaging_Z, each_lowhigh_instance.ch)
                FLIMageCont.AlignSmallRegion_2d()  #getting self.shifts_fromSmall = (np aray shift calculated from small region)
                
                FLIMageCont.analyze_uncaging_point_from_singleplane()
                FLIMageCont.find_best_point_dend_ori_given(direction, dend_orientation,
                                                           uncaging_Y, uncaging_X)
                modified_uncaging_xlist.append(FLIMageCont.uncaging_x)
                modified_uncaging_ylist.append(FLIMageCont.uncaging_y)
    
        FLIMageCont.flim.sendCommand("ClearUncagingLocation")
        for nth_pos in range(len(modified_uncaging_xlist)):
            FLIMageCont.flim.sendCommand(f"CreateUncagingLocation,{(modified_uncaging_xlist[nth_pos])},{(modified_uncaging_ylist[nth_pos])}")
            