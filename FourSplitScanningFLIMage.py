# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:09:49 2023

@author: yasudalab
"""
import os
import glob
import math
from FLIMageAlignment import flim_files_to_nparray,Align_4d_array,Align_3d_array,get_xyz_pixel_um,single_plane_align_with3dstack_flimfile
from datetime import datetime
import matplotlib.pyplot as plt
from multidim_tiff_viewer import multiple_uncaging_click,threeD_array_click
import numpy as np
from tkinter_textinfowindow import TextWindow
from controlflimage_threading import control_flimage

class FourSplitScanningFLIMage(control_flimage):
    def __init__(self,ini_path=r'DirectionSetting.ini'):
        super().__init__(ini_path=r'DirectionSetting.ini')
        
        nsplit = self.get_val_sendCommand("State.Acq.nSplitScanning")
        if nsplit != 4:
            raise Exception("Please check whether the number of split scanning is 4.")
            
    def define_spine_multipoints(self):
        self.wait_while_grabbing()
        
        Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray([self.flimlist[0]],ch=self.ch)
        FirstStack = Tiff_MultiArray[0]
        
        text = "Click the center of the spine you will stimulate. (Not the uncaging position itself)"
        z, ylist, xlist = multiple_uncaging_click(FirstStack,text,
                                 SampleImg=self.Spine_example,ShowPoint=False)
        
        self.Spine_Z = z
        self.Spine_Ylist = ylist
        self.Spine_Xlist = xlist
        self.uncaging_posY = []
        self.uncaging_posX = []
        self.uncaging_vectorYX = []
        
        print("\n\n\n","self.Spine_ZYX=",self.Spine_ZYX)
        print("self.Dendrite_ZYX=",self.Dendrite_ZYX,"\n\n\n\n")

        maxproj_aroundZ = FirstStack[max(0,z-self.cuboid_ZYX[0]) : min(FirstStack.shape[1]-1,z+self.cuboid_ZYX[0]+1), :, :].max(axis=0)

        text2 = "Click the uncaging position"
        
        for eachy in Spine_Ylist:
            while True:                
                _, y_uncaging,x_uncaging = threeD_array_click(maxproj_aroundZ,text2,
                                                        SampleImg=self.Uncaging_example,
                                                        ShowPoint=True,ShowPoint_YX=[self.Spine_Ylist[NthSplit],self.Spine_Xlist[NthSplit]])
                if (
                    abs(self.Spine_Ylist[NthSplit] - y_uncaging) < self.cuboid_ZYX[1] 
                    and 
                    abs(self.Spine_Xlist[NthSplit] - x_uncaging) < self.cuboid_ZYX[2] 
                    ):
                    self.uncaging_posY.append(y_uncaging)
                    self.uncaging_posX.append(x_uncaging)
                    
                    EachVector = np.array(
                                        [y_uncaging - self.Spine_Ylist[NthSplit],
                                         x_uncaging - self.Spine_Xlist[NthSplit]]
                                         )
                    NormVector = EachVector / np.sqrt(np.sum(EachVector**2))
                    self.uncaging_vectorYX.append(NormVector)
                    
                    break
                else:
                    text2 = "Click the uncaging point which MUST be near the spine"


    def set_param(self, RepeatNum,interval_sec, ch_1or2,
                  LoadSetting = False, SettingPath = "",
                  expected_grab_duration_sec = 0.5,
                  track_uncaging = False, ShowUncagingDetection = False, 
                  DoUncaging = False):
        self.RepeatNum=RepeatNum
        self.interval_sec=interval_sec
        self.ch=ch_1or2-1
        self.track_uncaging=track_uncaging
        self.ShowUncagingDetection=ShowUncagingDetection
        self.DoUncaging=DoUncaging

        self.expected_grab_duration_sec=expected_grab_duration_sec
        if LoadSetting==True:
            self.flim.sendCommand(f'LoadSetting, {SettingPath}')


    def start_repeat_split(self):
        self.start=datetime.now()
        
        self.folder = self.get_val_sendCommand("State.Files.pathName")
        self.NameStem = self.get_val_sendCommand("State.Files.baseName")
        self.childname = self.NameStem + str(int(self.get_val_sendCommand("State.Files.fileCounter"))).zfill(3)+".flim"
        self.TxtWind = TextWindow()
        self.showWindow =True
        
        for NthAc in range(self.RepeatNum):
            each_acquisition_from=datetime.now()      
            self.TxtWind.udpate("Now Grabbing")
            self.flim_connect_check()
            self.flim.sendCommand('StartGrab')

            self.wait_while_grabbing()
            self.flimlist=glob.glob(os.path.join(self.folder,f"{self.NameStem}*.flim"))
            self.XOffset_Split = list(map(float,self.get_val_sendCommand('State.Acq.XOffset_Split')[1:-1].split(",")))
            self.YOffset_Split = list(map(float,self.get_val_sendCommand('State.Acq.YOffset_Split')[1:-1].split(",")))
            self.Rotation_Split = list(map(int,self.get_val_sendCommand('State.Acq.Rotation_Split')[1:-1].split(",")))

            if len(self.flimlist)>1:
                self.align_split_flimfile()
                
                if self.track_uncaging==True:
                    self.flim.sendCommand('ClearUncagingLocation')
                    
                    for NthSplit in range(4):
                        self.AlignSmallRegion_split()
                        uncaging_yx = self.analyze_uncaging_point_split()
                        self.flim.sendCommand("CreateUncagingLocation, {uncaging_yx[1]}, {uncaging_yx[0]}")
                
                    self.send_uncaging_pos()
            
            if NthAc < self.RepeatNum-1:
                self.wait_until_next(each_acquisition_from)


    
    def align_split_flimfile(self,last_flimNth=-1):
        filelist=[self.flimlist[0],self.flimlist[last_flimNth]]
        print(filelist)
        
        Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray(filelist,ch=self.ch)
        self.Tiff_MultiArray = Tiff_MultiArray
        
        z_drift_list=[]
        Next_XOffset_Split = []
        Next_YOffset_Split = []
        
        for NthSplit in range(4):            
            each_shift_zyx_pixel, _ = Align_4d_array(Tiff_MultiArray)
            z_drift_list.append(each_shift_zyx_pixel[0])
            
            angle = math.pi * self.Rotation_Split[NthSplit] / 180
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)

            xpixel_shift = +cos_val * each_shift_zyx_pixel[2]
            ypixel_shift = -sin_val * each_shift_zyx_pixel[1]
        
            x_shift_galvounit = round(self.XMaxVoltage*(xpixel_shift)/(self.zoom * self.pixelsPerLine),8)
            y_shift_galvounit = round(self.YMaxVoltage*(ypixel_shift)/(self.zoom * self.pixelsPerLine),8)

            Next_XOffset_Split.append(self.XOffset_Split[NthSplit] + x_shift_galvounit)
            Next_YOffset_Split.append(self.YOffset_Split[NthSplit] + y_shift_galvounit)
        
        self.flim_connect_check()
        Z_shift = np.median(np.array(z_drift_list))
        self.flim.sendCommand(f"State.Acq.XOffset_Split = {str(Next_XOffset_Split)}")
        self.flim.sendCommand(f"State.Acq.YOffset_Split = {str(Next_YOffset_Split)}")
        

    def AlignSmallRegion_split(self, NthSplit):
        
        Y_min = int(NthSplit * self.pixelsPerLine / 4)
        Y_max = Y_min + int(self.pixelsPerLine/4)
        
        Zfrom = max(0, self.Spine_Z - self.cuboid_ZYX[0])
        Zto = min(self.nSlices, self.Spine_Z + self.cuboid_ZYX[0]+1)
        Yfrom = max(Y_min, self.Spine_Ylist[NthSplit] - self.cuboid_ZYX[1])
        Yto= min(Y_max, self.Spine_Ylist[NthSplit] + self.cuboid_ZYX[1])
        Xfrom = max(0, self.Spine_Xlist[NthSplit] - self.cuboid_ZYX[2])
        Xto = min(self.pixelsPerLine, self.Spine_Xlist[NthSplit] - self.cuboid_ZYX[2])
        
        TrimmedAroundSpine = self.Tiff_MultiArray[
                            :,
                            Zfrom : Zto ,
                            Yfrom : Yto,
                            Xfrom : Xto,
                            ]
        
        self.shifts_fromSmall, self.Small_Aligned_4d_array = Align_4d_array(TrimmedAroundSpine)






    def find_best_point_split(self, TwoD = False):

        if TwoD==False:
            orientation = self.props.orientation
            y0,x0 = self.props.centroid
            self.orientation_based_on_3d = orientation
            self.dendrite_Centroid=[y0,x0]
            
        else:
                orientation = self.orientation_based_on_3d
                y0,x0 = self.dendrite_Centroid[0],self.dendrite_Centroid[1]
            except:
                orientation = self.props.orientation  
                y0,x0 = self.props.centroid
            
        candi_x, candi_y = self.cuboid_ZYX[2],self.cuboid_ZYX[1]
        
        x_moved = x0 - self.cuboid_ZYX[2]
        y_moved = y0 - self.cuboid_ZYX[1]

        x_rotated = x_moved*math.cos(orientation) - y_moved*math.sin(orientation)

        if x_rotated<=0:
            direction = 1
        else:
            direction = -1

        while True:
            try:
                if self.binarized[int(candi_y),int(candi_x)]>0:
                    candi_x = candi_x + math.cos(orientation)*direction
                    candi_y = candi_y - math.sin(orientation)*direction
                else:
                    # Assuming that x and y have same resolution
                    distance_pixel = self.SpineHeadToUncaging_um/self.x_um
                    candi_x = int(candi_x + math.cos(orientation)*direction*distance_pixel)
                    candi_y = int(candi_y - math.sin(orientation)*direction*distance_pixel)
                    break
            except:
                print("Error 103 - -  ")
                candi_x, candi_y = self.cuboid_ZYX[2],self.cuboid_ZYX[1]
                break
        
        self.candi_x = candi_x
        self.candi_y = candi_y

        if ignore_stage_drift==False:
            self.uncaging_x=self.Spine_ZYX[2]-self.cuboid_ZYX[2] +candi_x - self.shifts_zyx_pixel[-1][2] - self.shifts_fromSmall[-1][2]
            self.uncaging_y=self.Spine_ZYX[1]-self.cuboid_ZYX[1] +candi_y - self.shifts_zyx_pixel[-1][1] - self.shifts_fromSmall[-1][1]
        else:
            self.uncaging_x=self.Spine_ZYX[2]-self.cuboid_ZYX[2] +candi_x - self.shifts_fromSmall[-1][2]
            self.uncaging_y=self.Spine_ZYX[1]-self.cuboid_ZYX[1] +candi_y - self.shifts_fromSmall[-1][1]
        
