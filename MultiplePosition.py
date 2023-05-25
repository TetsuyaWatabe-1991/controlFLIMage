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
import copy
from time import sleep


class Low_High_mag_assign():
    def __init__(self, lowmag_path, highmag_path, ch_1or2 = 1):
        self.lowmag_path = lowmag_path
        self.highmag_path = highmag_path
        
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
    
        
list_of_fileset =[
                    [r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230524\Sample_pos1_zoom1_001.flim",
                     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230524\Sample_pos1_zoom20_001.flim"],
                    [r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230524\Sample_pos1_zoom1_001.flim",
                     r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230524\Sample_pos1_zoom20_001.flim"],
                ]

LowHighset_instances = []
                   
for eachfileset in list_of_fileset:
    LowHighset_instances.append(Low_High_mag_assign(lowmag_path = eachfileset[0],
                                                    highmag_path =eachfileset[1], 
                                                    ch_1or2 = 2))
    
    print(LowHighset_instances[-1].uncaging_x,LowHighset_instances[-1].uncaging_y)



FLIMageCont = control_flimage()
print("Now Grabbing")

num_T = 10
    
for nthacquisiton in range(num_T):
    print(f"ACQUISTION, {nthacquisiton+1}/{num_T}")
    
    for ind, each_lowhigh_instance in enumerate(LowHighset_instances):
        
        ###Low magnification
        FLIMageCont.flim.sendCommand(f'LoadSetting, {each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.initFileName"]}')
        FLIMageCont.flim.sendCommand(f'State.Acq.power = {each_lowhigh_instance.lowmag_iminfo.statedict["State.Acq.power"]}')
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = {each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.baseName"]}')
        FLIMageCont.flim.sendCommand(f'State.Files.zoom = {each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.zoom"]}')
                
        
        flimlist = os.path.join(each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.pathName"],
                                each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.baseName"]+"[0-9][0-9][0-9].flim")
        
        counter = 1
        for eachflim in flimlist:
            try:   
                num = int(eachflim[-8:-5])
                if num > counter:
                    counter = num
            except:
                pass
        
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {counter}')
        
        FLIMageCont.flim_connect_check()
        FLIMageCont.flim.sendCommand('StartGrab')  
        FLIMageCont.wait_while_grabbing()
        
        latestfilepath = os.path.join(each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.initFileName"], 
                                      each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.baseName"] + str(counter).zfill(3) + ".flim")
        
        FLIMageCont.relative_zyx_um = align_two_flimfile(flim1 = each_lowhigh_instance.lowmag_path, 
                                                                flim2 = latestfilepath, 
                                                                iminfo = each_lowhigh_instance.lowmag_iminfo,
                                                                ch = each_lowhigh_instance.ch_1or2)
        print(FLIMageCont.relative_zyx_um)
        
        FLIMageCont.go_to_relative_pos_motor()
        sleep(3)
        
        FLIMageCont.flim.sendCommand(f'LoadSetting, {each_lowhigh_instance.highwmag_iminfo.statedict["State.Files.initFileName"]}')
        FLIMageCont.flim.sendCommand(f'State.Acq.power = {each_lowhigh_instance.highmag_iminfo.statedict["State.Acq.power"]}')
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = {each_lowhigh_instance.highmag_iminfo.statedict["State.Files.baseName"]}')
        FLIMageCont.flim.sendCommand(f'State.Files.zoom = {each_lowhigh_instance.highmag_iminfo.statedict["State.Files.zoom"]}')
                
        flimlist = os.path.join(each_lowhigh_instance.highmag_iminfo.statedict["State.Files.pathName"],
                                each_lowhigh_instance.highmag_iminfo.statedict["State.Files.baseName"]+"[0-9][0-9][0-9].flim")
        
        counter = 1
        for eachflim in flimlist:
            try:   
                num = int(eachflim[-8:-5])
                if num > counter:
                    counter = num
            except:
                pass
        
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {counter}')
        
        FLIMageCont.flim_connect_check()
        FLIMageCont.flim.sendCommand('StartGrab')  
        FLIMageCont.wait_while_grabbing()
    
    print("LOOP END")
      
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
