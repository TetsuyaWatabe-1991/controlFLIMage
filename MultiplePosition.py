# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:24:13 2023

@author: WatabeT
"""
import os,glob,pathlib,math
from FLIMageAlignment import flim_files_to_nparray,Align_4d_array, align_two_flimfile
from FLIMageFileReader2 import FileReader
from multidim_tiff_viewer import threeD_array_click
from controlflimage_threading import control_flimage
import numpy as np
import copy
from time import sleep
import random



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
        z,y,x = threeD_array_click(FirstStack,text,
                                 SampleImg=self.Spine_example,ShowPoint=False)
        maxproj_aroundZ=FirstStack[max(0,z-self.cuboid_ZYX[0]):min(FirstStack.shape[1]-1,z+self.cuboid_ZYX[0]+1),:,:].max(axis=0)

        text2 = "Click the dendrite near the selected spine"
        while True:
            z_dend,y_dend,x_dend = threeD_array_click(maxproj_aroundZ,text2,
                                                    SampleImg=self.Dendrite_example,ShowPoint=True,ShowPoint_YX=[y,x])
            if abs(y_dend-y)<self.cuboid_ZYX[1] and abs(x_dend-x)<self.cuboid_ZYX[2]:
                break
            else:
                text2 = "Click the dendrite, which MUST be near the spine"
        self.Spine_ZYX=[z,y,x]
        self.Dendrite_ZYX=[z_dend,y_dend,x_dend]
        self.uncaging_x=x
        self.uncaging_y=y
        
        
# def align_two_flimfile(flim_1, flim_2, iminfo, ch):
#     filelist = [flim_1, flim_2]
#     Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray(filelist,ch=ch)
#     shifts_zyx_pixel, Aligned_4d_array=Align_4d_array(Tiff_MultiArray)
#     x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
#     z_relative, y_relative, x_relative = convert_shifts_pix_to_micro(x_um, y_um, z_um, shifts_zyx_pixel)
#     return z_relative, y_relative, x_relative
    


singleplane_uncaging=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging.txt"



list_of_fileset = [[r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\pos4_lowmag_002.flim",
                    r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\pos4_highmag_001.flim"],
                    [r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\pos5_lowmag_004.flim",
                    r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230531\pos5_highmag_003.flim"]]             

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
    
for nthacquisiton in range(num_T):
    print(f"ACQUISTION, {nthacquisiton+1}/{num_T}")
    
    for ind, each_lowhigh_instance in enumerate(LowHighset_instances):
        # nthacquisiton = 0
        # each_lowhigh_instance = LowHighset_instances[0]
        
        ###Low magnification
        FLIMageCont.flim.sendCommand(f'LoadSetting, {each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.initFileName"]}')
        FLIMageCont.flim.sendCommand(f'State.Acq.power = {each_lowhigh_instance.lowmag_iminfo.statedict["State.Acq.power"]}')
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{each_lowhigh_instance.lowmag_basename}')
        FLIMageCont.flim.sendCommand(f'State.Acq.zoom = {each_lowhigh_instance.lowmag_iminfo.statedict["State.Acq.zoom"]}')

        flimlist = glob.glob(os.path.join(each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.pathName"],
                                          each_lowhigh_instance.lowmag_basename+"[0-9][0-9][0-9].flim"))
        
        lowcounter = 1
        for eachflim in flimlist:
            try:   
                num = int(eachflim[-8:-5])
                if num > lowcounter:
                    lowcounter = num
            except:
                pass
        lowcounter+=1
        
        x_str = str(each_lowhigh_instance.lowmag_pos[0])
        y_str = str(each_lowhigh_instance.lowmag_pos[1])
        z_str = str(each_lowhigh_instance.lowmag_pos[2])
        FLIMageCont.flim.sendCommand(f"SetMotorPosition,{x_str},{y_str},{z_str}")
        sleep(4)

        #Sometimes (around one in ten), stage do not move.
        #To avoid that, check whether stage is in proper position
        #If not, stage move again
        for i in range(10):
            currentpos = FLIMageCont.get_val_sendCommand("State.Motor.motorPosition")
            currentpos_num = np.array(currentpos[1:-1].split(","), dtype=float)
            
            diff =  currentpos_num - np.array(each_lowhigh_instance.lowmag_pos)
            sum_sq_err = (diff*diff).sum()
            
            if sum_sq_err > 10:
                print("Not moved yet")
                print(f"{i+2} th trial .....")
                FLIMageCont.flim.sendCommand(f"SetMotorPosition,{x_str},{y_str},{z_str}")
                sleep(4)
                
        # if nthacquisiton>5 and nthacquisiton%4==0:
        #     zure = 3*random.randint(-2,2)
        #     z_str = str(each_lowhigh_instance.lowmag_pos[2]+zure)
        #     FLIMageCont.flim.sendCommand(f"SetMotorPosition,{x_str},{y_str},{z_str}")
        #     sleep(4)
        
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {lowcounter}')
        FLIMageCont.flim_connect_check()
        FLIMageCont.expected_grab_duration_sec = 10
        
        FLIMageCont.flim.sendCommand('StartGrab')  
        FLIMageCont.wait_while_grabbing()
        
        latestfilepath = os.path.join(each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.pathName"], 
                                      each_lowhigh_instance.lowmag_basename + str(lowcounter).zfill(3) + ".flim")
        
        FLIMageCont.relative_zyx_um, Aligned_4d_array = align_two_flimfile(each_lowhigh_instance.lowmag_path, 
                                                             latestfilepath, 
                                                             each_lowhigh_instance.lowmag_iminfo,
                                                             each_lowhigh_instance.ch)
        print(FLIMageCont.relative_zyx_um)
        
        FLIMageCont.go_to_relative_pos_motor()
        sleep(4)
        
        x=(each_lowhigh_instance.lowmag_pos[0] - FLIMageCont.directionMotorX * FLIMageCont.relative_zyx_um[2])
        y=(each_lowhigh_instance.lowmag_pos[1] - FLIMageCont.directionMotorY * FLIMageCont.relative_zyx_um[1])
        z=(each_lowhigh_instance.lowmag_pos[2] - FLIMageCont.directionMotorZ * FLIMageCont.relative_zyx_um[0])
        nextxyz = np.array([x,y,z])
        
        for i in range(10):
            currentpos2 = FLIMageCont.get_val_sendCommand("State.Motor.motorPosition")
            currentpos_num2 = np.array(currentpos2[1:-1].split(","), dtype=float)
            
            diff2 =  currentpos_num2 - nextxyz
            sum_sq_err2 = (diff2*diff2).sum()
            
            if sum_sq_err2 > 10:
                print("Not moved yet . . . . ")
                print(f"{i+2} th trial .....")
                FLIMageCont.flim.sendCommand(f"SetMotorPosition,{x},{y},{z}")
                sleep(2)
        
        each_lowhigh_instance.lowmag_pos = list(nextxyz)
        
        sleep(2)
        
        FLIMageCont.flim.sendCommand(f'LoadSetting, {each_lowhigh_instance.highmag_iminfo.statedict["State.Files.initFileName"]}')
        FLIMageCont.flim.sendCommand(f'State.Acq.power = {each_lowhigh_instance.highmag_iminfo.statedict["State.Acq.power"]}')
        # FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{each_lowhigh_instance.highmag_iminfo.statedict["State.Files.baseName"]}"')
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{each_lowhigh_instance.highmag_basename}')
        FLIMageCont.flim.sendCommand(f'State.Acq.zoom = {each_lowhigh_instance.highmag_iminfo.statedict["State.Acq.zoom"]}')
                
        flimlist = glob.glob(os.path.join(each_lowhigh_instance.highmag_iminfo.statedict["State.Files.pathName"],
                                each_lowhigh_instance.lowmag_basename+"[0-9][0-9][0-9].flim"))

        
        highcounter = 1
        for eachflim in flimlist:
            try:   
                num = int(eachflim[-8:-5])
                if num > highcounter:
                    highcounter = num
            except:
                pass
        highcounter+=1
        
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {highcounter}')
        
        FLIMageCont.flim_connect_check()
        FLIMageCont.flim.sendCommand('StartGrab')  
        FLIMageCont.wait_while_grabbing()
        
        sleep(3)


#
#
# Uncaging
#
#
if False:
    for ind, each_lowhigh_instance in enumerate(LowHighset_instances):
        # nthacquisiton = 0
        # each_lowhigh_instance = LowHighset_instances[1]
        
        ###Low magnification
        FLIMageCont.flim.sendCommand(f'LoadSetting, {each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.initFileName"]}')
        FLIMageCont.flim.sendCommand(f'State.Acq.power = {each_lowhigh_instance.lowmag_iminfo.statedict["State.Acq.power"]}')
        # FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.baseName"]}"')
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{each_lowhigh_instance.lowmag_basename}')
        FLIMageCont.flim.sendCommand(f'State.Acq.zoom = {each_lowhigh_instance.lowmag_iminfo.statedict["State.Acq.zoom"]}')
        flimlist = glob.glob(os.path.join(each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.pathName"],
                                          each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.baseName"]+"[0-9][0-9][0-9].flim"))
        
        lowcounter = 1
        for eachflim in flimlist:
            try:   
                num = int(eachflim[-8:-5])
                if num > lowcounter:
                    lowcounter = num
            except:
                pass
        lowcounter+=1
        
        
        # FLIMageCont.flim.sendCommand(f'State.Motor.motorPosition = {each_lowhigh_instance.lowmag_pos}')
        x_str = str(each_lowhigh_instance.lowmag_pos[0])
        y_str = str(each_lowhigh_instance.lowmag_pos[1])
        z_str = str(each_lowhigh_instance.lowmag_pos[2])
        FLIMageCont.flim.sendCommand(f"SetMotorPosition,{x_str},{y_str},{z_str}")
        sleep(3)
        
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {lowcounter}')
        FLIMageCont.flim_connect_check()
        FLIMageCont.expected_grab_duration_sec = 10
        
        FLIMageCont.flim.sendCommand('StartGrab')  
        FLIMageCont.wait_while_grabbing()
        
        latestfilepath = os.path.join(each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.pathName"], 
                                      each_lowhigh_instance.lowmag_basename + str(lowcounter).zfill(3) + ".flim")
        
        FLIMageCont.relative_zyx_um, Aligned_4d_array = align_two_flimfile(each_lowhigh_instance.lowmag_path, 
                                                                 latestfilepath, 
                                                                 each_lowhigh_instance.lowmag_iminfo,
                                                                 each_lowhigh_instance.ch)
        print(FLIMageCont.relative_zyx_um)
    
        FLIMageCont.go_to_relative_pos_motor()
        sleep(2)
        
        FLIMageCont.flim.sendCommand(f'LoadSetting, {each_lowhigh_instance.highmag_iminfo.statedict["State.Files.initFileName"]}')
        FLIMageCont.flim.sendCommand(f'State.Acq.power = {each_lowhigh_instance.highmag_iminfo.statedict["State.Acq.power"]}')
        # FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{each_lowhigh_instance.highmag_iminfo.statedict["State.Files.baseName"]}"')
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{each_lowhigh_instance.highmag_basename}')
        FLIMageCont.flim.sendCommand(f'State.Acq.zoom = {each_lowhigh_instance.highmag_iminfo.statedict["State.Acq.zoom"]}')
                
        flimlist = glob.glob(os.path.join(each_lowhigh_instance.highmag_iminfo.statedict["State.Files.pathName"],
                                          each_lowhigh_instance.highmag_basename+"[0-9][0-9][0-9].flim"))
        
        highcounter = 1
        for eachflim in flimlist:
            try:   
                num = int(eachflim[-8:-5])
                if num > highcounter:
                    highcounter = num
            except:
                pass
        highcounter+=1
        
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {highcounter}')
        
        FLIMageCont.flim_connect_check()
        FLIMageCont.flim.sendCommand('StartGrab')  
        FLIMageCont.wait_while_grabbing()
        
        latestfilepath = os.path.join(each_lowhigh_instance.highmag_iminfo.statedict["State.Files.pathName"], 
                                      each_lowhigh_instance.highmag_basename + str(highcounter).zfill(3) + ".flim")
        
        FLIMageCont.relative_zyx_um, Aligned_4d_array, shifts_zyx_pixel = align_two_flimfile(
                                                                            each_lowhigh_instance.highmag_path, 
                                                                            latestfilepath,
                                                                            each_lowhigh_instance.ch,
                                                                            return_pixel = True)
        
        # This is little bit unsual way.  Non intrinsic.  Fix later.
        FLIMageCont.shifts_zyx_pixel = shifts_zyx_pixel
        FLIMageCont.Aligned_4d_array = Aligned_4d_array
        
        print(FLIMageCont.relative_zyx_um)
        each_lowhigh_instance.lowmag_pos = list(np.array(each_lowhigh_instance.lowmag_pos) + np.array(FLIMageCont.relative_zyx_um))
        
        FLIMageCont.go_to_relative_pos_motor()
        
        sleep(2)
    
        FLIMageCont.Spine_ZYX = each_lowhigh_instance.Spine_ZYX
        FLIMageCont.set_xyz_um(each_lowhigh_instance.highmag_iminfo)
        FLIMageCont.go_to_uncaging_plane()
        
        FLIMageCont.set_param(RepeatNum = 5, interval_sec = 2, ch_1or2 = each_lowhigh_instance.ch + 2,
                              LoadSetting=True,SettingPath=singleplane_uncaging,
                              track_uncaging=True,drift_control=False,drift_cont_galvo=True,
                              ShowUncagingDetection=True,DoUncaging=False,expected_grab_duration_sec=1.5)
        FLIMageCont.start_repeat_short()
        FLIMageCont.back_to_stack_plane()
        
        sleep(5)
        
    
    
# def go_to_uncaging_plane(self):
#     # sleep(2)
#     print("go_to_uncaging_plane")
#     z=self.Spine_ZYX[0]
#     NumZ = self.Aligned_4d_array.shape[1]
#     z_move_um =  - self.z_um * (z -(NumZ - 1)/2)
#     z_relative = self.z_um*self.shifts_zyx_pixel[-1][0]
    
#     print("z_move_um ",z_move_um)
#     print("z_relative ",z_relative)
    
#     self.relative_zyx_um = [z_move_um + z_relative,0,0]
#     print(self.relative_zyx_um)
#     self.go_to_relative_pos_motor()
# ch=1

# filelist = [r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230524\Sample_pos2_zoom20_001.flim",
#             r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230524\Sample_pos2_zoom20_002.flim"]


# iminfo = FileReader()
# iminfo.read_imageFile(filelist[0], True) 
# print(iminfo.statedict["State.Motor.motorPosition"])

# iminfo = FileReader()
# iminfo.read_imageFile(filelist[1], True) 
# print(iminfo.statedict["State.Motor.motorPosition"])

# Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray(filelist,ch=ch,normalize_by_averageNum=True)
# shifts, Aligned_4d_array=Align_4d_array(Tiff_MultiArray)

# print(shifts)
