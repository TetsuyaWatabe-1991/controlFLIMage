# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:19:07 2024

@author: yasudalab
"""


import os
import glob
import pathlib
import pandas as pd
import copy
from FLIMageAlignment import  align_two_flimfile
from FLIMageFileReader2 import FileReader
from controlflimage_threading import Control_flimage


class Multiarea_from_lowmag():
    def __init__(self, lowmag_path,
                 rel_pos_um_csv_path,
                 high_mag_setting_path,
                 high_mag_zoom = 16,
                 ch_1or2 = 1):
        self.lowmag_path = lowmag_path
        self.lowmag_basename = pathlib.Path(lowmag_path).stem[:-3]        
        self.lowmag_iminfo = FileReader()
        self.lowmag_iminfo.read_imageFile(self.lowmag_path, True)
        self.lowmag_magnification = self.lowmag_iminfo.statedict['State.Acq.zoom']
        latestpath = self.latest_path()
        self.lowmag_iminfo = FileReader()
        self.lowmag_iminfo.read_imageFile(latestpath, True)
        
        self.high_mag_setting_path = high_mag_setting_path
        self.high_mag_relpos_dict = {}
        self.high_mag_zoom = high_mag_zoom
        self.rel_pos_um_csv_path = rel_pos_um_csv_path
        self.read_rel_pos_um_csv()
        
        bottom_lowmag_xyz_um = list(copy.copy(self.lowmag_iminfo.statedict['State.Motor.motorPosition']))
        sliceStep = self.lowmag_iminfo.statedict['State.Acq.sliceStep']
        nSlices = self.lowmag_iminfo.statedict['State.Acq.nSlices']
        additionZ_um = sliceStep*(nSlices - 1)/2
        corrected_lowmag_xyz_um = copy.copy(bottom_lowmag_xyz_um)
        corrected_lowmag_xyz_um[2] += additionZ_um
        
        self.corrected_lowmag_xyz_um = copy.copy(corrected_lowmag_xyz_um)
        self.ch = ch_1or2 -1
        self.Spine_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Spine_example.png"
        self.Dendrite_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Dendrite_example.png"
        self.cuboid_ZYX=[2,20,20]
    
    def read_rel_pos_um_csv(self):
        self.rel_pos_df = pd.read_csv(self.rel_pos_um_csv_path)
        for ind in self.rel_pos_df.index:
            pos_id = self.rel_pos_df.loc[ind,"pos_id"]
            x_um = self.rel_pos_df.loc[ind,"x_um"]
            y_um = self.rel_pos_df.loc[ind,"y_um"]
            z_um = self.rel_pos_df.loc[ind,"z_um"]
            self.high_mag_relpos_dict[pos_id] = {}
            self.high_mag_relpos_dict[pos_id]["x_um"] = x_um
            self.high_mag_relpos_dict[pos_id]["y_um"] = y_um
            self.high_mag_relpos_dict[pos_id]["z_um"] = z_um
    
    def get_max_flimfiles(self, flimlist):
        counter = 1
        for eachflim in flimlist:
            try:   
                num = int(eachflim[-8:-5])
                if num > counter:
                    counter = num
            except:
                pass
        return counter    
            
    def get_max_plus_one_flimfiles(self, flimlist):
        counter = self.get_max_flimfiles(flimlist)
        counter+=1
        return counter
    
    def latest_path(self):
        low_flimlist = glob.glob(os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"],
                                              self.lowmag_basename+"[0-9][0-9][0-9].flim"))        
        low_maxcount = self.get_max_flimfiles(low_flimlist)
        latestpath = os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"], 
                                  self.lowmag_basename + str(low_maxcount).zfill(3) + ".flim")
        return latestpath

    def count_flimfiles(self) -> int:
        low_flimlist = glob.glob(os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"],
                                              self.lowmag_basename+"[0-9][0-9][0-9].flim"))
        self.low_counter = self.get_max_plus_one_flimfiles(low_flimlist)    
        self.low_max_plus1_flim = os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"], 
                                             self.lowmag_basename + str(self.low_counter).zfill(3) + ".flim")
        return self.low_counter

    def count_high_mag_flimfiles(self, pos_id) -> int:
        highmag_flimlist = glob.glob(os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"],
                                              f"{self.lowmag_basename}_highmag_{pos_id}_"+"[0-9][0-9][0-9].flim"))
        counter = self.get_max_plus_one_flimfiles(highmag_flimlist)    
        return counter
        # self.high_max_plus1_flim = os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"], 
                                              # f"{self.lowmag_basename}_highmag_{pos_id}_" + str(self.low_counter).zfill(3) + ".flim")
        
    def send_lowmag_acq_info(self, FLIMageCont):
        FLIMageCont.flim.sendCommand(f'LoadSetting, {self.lowmag_path}')
        FLIMageCont.flim.sendCommand(f'State.Acq.power = {self.lowmag_iminfo.statedict["State.Acq.power"]}')
        FLIMageCont.flim.sendCommand(f'State.Files.pathName = "{self.lowmag_iminfo.statedict["State.Files.pathName"]}"')
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{self.lowmag_basename}"')
        low_counter = self.count_flimfiles()
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {low_counter}')
        FLIMageCont.flim.sendCommand(f'State.Acq.zoom = {self.lowmag_iminfo.statedict["State.Acq.zoom"]}')
        FLIMageCont.flim.sendCommand('SetScanMirrorXY_um, 0, 0')
        FLIMageCont.flim.sendCommand('SetCenter')
    
    def send_highmag_acq_info(self, FLIMageCont, pos_id):
        FLIMageCont.flim.sendCommand(f'LoadSetting, {self.high_mag_setting_path}')
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{self.lowmag_basename}_highmag_{pos_id}_"')
        FLIMageCont.flim.sendCommand(f'State.Acq.zoom = {self.high_mag_zoom}')      
        counter = self.count_high_mag_flimfiles(pos_id = pos_id)
        FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {counter}')
        FLIMageCont.relative_zyx_um = [(-1)*self.high_mag_relpos_dict[pos_id]["z_um"],
                                       self.high_mag_relpos_dict[pos_id]["y_um"],
                                       self.high_mag_relpos_dict[pos_id]["x_um"]]
        FLIMageCont.go_to_absolute_pos_um_galvo(z_move = True)
        FLIMageCont.flim.sendCommand('SetCenter')
    
    def update_pos_fromcurrent(self, FLIMageCont):
        self.corrected_lowmag_xyz_um = FLIMageCont.get_position()
    
    def go_to_lowmag_center_pos(self, FLIMageCont, FastMS2k):
        dest_x,dest_y,dest_z = self.corrected_lowmag_xyz_um
        FLIMageCont.flim.sendCommand('MotorDisconnect')
        # FastMS2k.move_pos_mm(self, 
        #                      x_mm = dest_x*10**3,
        #                      y_mm = dest_y*10**3, 
        #                      z_mm = dest_z*10**3)
        FastMS2k.move_pos_mm(
                             x_mm = dest_x/10**3,
                             y_mm = dest_y/10**3, 
                             z_mm = dest_z/10**3)
        
        FLIMageCont.flim.sendCommand('MotorReopen')
        
    def go_to_relative_pos_after_(self, relative_zyx_um, FLIMageCont, FastMS2k):
        x,y,z=FLIMageCont.get_position()        
        dest_x = x - FLIMageCont.directionMotorX * relative_zyx_um[2]
        dest_y = y - FLIMageCont.directionMotorY * relative_zyx_um[1]
        dest_z = z - FLIMageCont.directionMotorZ * relative_zyx_um[0]
        FLIMageCont.flim.sendCommand('MotorDisconnect')
        FastMS2k.move_pos_mm(self, 
                             x_mm = dest_x*10**3,
                             y_mm = dest_y*10**3, 
                             z_mm = dest_z*10**3)
        FLIMageCont.flim.sendCommand('MotorReopen')
        


if __name__ == "__main__":
    high_mag_setting_path = r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\z7_10_kal8.txt"
    FLIMageCont = Control_flimage()
    FLIMageCont.interval_sec = 600
    FLIMageCont.expected_grab_duration_sec = 5
    ch_1or2 = 2
    lowmag_path_list = [r"G:\ImagingData\Tetsuya\20240626\multipos_3\a_001.flim",
                        r"G:\ImagingData\Tetsuya\20240626\multipos_3\b_001.flim",
                        r"G:\ImagingData\Tetsuya\20240626\multipos_3\c_001.flim",
                        r"G:\ImagingData\Tetsuya\20240626\multipos_3\d_001.flim",
                        r"G:\ImagingData\Tetsuya\20240626\multipos_3\e_001.flim"
                        ]
    
    lowmag_instance_list = []
    for each_lowmag in lowmag_path_list:
        # lowmag_path = r"G:\ImagingData\Tetsuya\20240626\b_001.flim"
        # rel_pos_um_csv_path = r"G:\ImagingData\Tetsuya\20240626\b_001\assigned_relative_um_pos.csv"
        
        rel_pos_um_csv_path = os.path.join(pathlib.Path(each_lowmag).parent, 
                                          pathlib.Path(each_lowmag).stem,
                                          "assigned_relative_um_pos.csv")
        
        lowmag_instance_list.append(Multiarea_from_lowmag(lowmag_path = each_lowmag,
                                                          rel_pos_um_csv_path = rel_pos_um_csv_path,
                                                          high_mag_setting_path = high_mag_setting_path))
    for nth_acq in range(100):
        
        for Each_lowmag_instance in lowmag_instance_list:    
            Each_lowmag_instance.go_to_lowmag_center_pos(FLIMageCont)
            Each_lowmag_instance.send_lowmag_acq_info(FLIMageCont)
            FLIMageCont.acquisition_include_connect_wait()
            FLIMageCont.set_param(RepeatNum = 1, interval_sec = 500, ch_1or2 = ch_1or2)
            FLIMageCont.relative_zyx_um, _ = align_two_flimfile(
                                                Each_lowmag_instance.lowmag_path, 
                                                Each_lowmag_instance.low_max_plus1_flim,
                                                Each_lowmag_instance.ch)
            FLIMageCont.go_to_relative_pos_motor_checkstate()
            Each_lowmag_instance.update_pos_fromcurrent(FLIMageCont)
            
            for each_high_mag_id in Each_lowmag_instance.high_mag_relpos_dict:   
                Each_lowmag_instance.go_to_lowmag_center_pos(FLIMageCont)
                Each_lowmag_instance.send_highmag_acq_info(FLIMageCont, each_high_mag_id)
                FLIMageCont.acquisition_include_connect_wait()
                
                
                
                