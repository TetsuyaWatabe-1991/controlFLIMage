# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:17:04 2022

@author: yasudalab
"""
import os
import glob
import math
import subprocess
from time import sleep
from datetime import datetime
from FLIMageAlignment import flim_files_to_nparray,Align_4d_array,Align_3d_array,get_xyz_pixel_um,single_plane_align_with3dstack_flimfile
from FLIM_pipeClient import FLIM_Com,FLIM_message_received
from find_close_remotecontrol import close_remote_control, window_exists
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from multidim_tiff_viewer import threeD_array_click, multiple_uncaging_click
import numpy as np
import cv2
from skimage.measure import label, regionprops
import configparser
import threading
from tkinter_textinfowindow import TextWindow

def long_axis_detection(props,HalfLen_c=0.35):
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * HalfLen_c * props.minor_axis_length
    y1 = y0 - math.sin(orientation) * HalfLen_c * props.minor_axis_length
    x2 = x0 - math.sin(orientation) * HalfLen_c * props.major_axis_length
    y2 = y0 - math.cos(orientation) * HalfLen_c * props.major_axis_length
    x1_1 = x0 - math.cos(orientation) * HalfLen_c * props.minor_axis_length
    y1_1 = y0 + math.sin(orientation) * HalfLen_c * props.minor_axis_length
    x2_1 = x0 + math.sin(orientation) * HalfLen_c * props.major_axis_length
    y2_1 = y0 + math.cos(orientation) * HalfLen_c * props.major_axis_length
    return x0,x1,x2,x1_1,x2_1,y0,y1,y2,y1_1,y2_1

def perpendicular_intersection(spine_x, spine_y, 
                               dend_slope, dend_intercept):        
    perpendicular_slope = -1 / dend_slope        
    perpendicular_intercept = spine_y - perpendicular_slope * x1        

    x_intersection = ((perpendicular_intercept - spine_y) / 
                      (perpendicular_slope - perpendicular_slope))
    y_intersection = dend_slope * x_intersection + dend_intercept
    
    return x_intersection, y_intersection

def plot_uncaging_point(props, binary, blur, image, candi_y,
                        candi_x, cuboid_ZYX,just_plot=True):
    vmax=blur.max()
    kwargs_ex = {'cmap':'gray',
                 'vmin':0,
                 'vmax':vmax,
                 'interpolation':'none'}
    kwargs_bin = {'cmap':'gray',
                 'vmin':0,
                 'vmax':1,
                 'interpolation':'none'}
    kwargs_uncag = dict(c="y",marker="+",s=200)

    plt.figure()
    f, axarr = plt.subplots(1,3)
    x0,x1,x2,x1_1,x2_1,y0,y1,y2,y1_1,y2_1 = long_axis_detection(props)
    axarr[2].imshow(binary,**kwargs_bin) #Make sure that vmin is less than 1.
    axarr[2].plot((x1_1, x1), (y1_1, y1), '-c', linewidth=2.5)
    axarr[2].plot((x2_1, x2), (y2_1, y2), '-c', linewidth=2.5)
    axarr[2].plot(x0, y0, '.g', markersize=15)
    axarr[2].plot([cuboid_ZYX[2],candi_x],[cuboid_ZYX[1],candi_y],
                  c="m",marker=".",lw=1)
    axarr[2].scatter(cuboid_ZYX[2],cuboid_ZYX[1],c="r",marker="*")
    axarr[2].scatter(candi_x,candi_y,**kwargs_uncag)
    axarr[1].imshow(blur,**kwargs_ex)
    axarr[1].scatter(candi_x,candi_y,**kwargs_uncag)

    axarr[0].imshow(image,**kwargs_ex);
    axarr[0].scatter(candi_x,candi_y,**kwargs_uncag)
    for j in range(3):
        axarr[j].axis('off')
    if just_plot==True:
        plt.show()
        return
    else:
        return f, axarr




class Control_flimage():

    def __init__(self,ini_path=r'DirectionSetting.ini'):
        print("START")
        self.flim = FLIM_Com()
        self.flim.start()
        self.error_dict = {}
        self.max_error_num = 20
        self.flimage_exe = r"C:\Users\yasudalab\Documents\GIT\flimage1_3\bin\Debug\FLIMage.exe"
        
        if self.flim.Connected:
            print("Good Connection")
            self.flim.messageReceived += FLIM_message_received #Add your function to  handle.
        else:
            self.reconnect()
        self.example_image()
        self.cuboid_ZYX=[2,20,20]
        self.uncaging_x=0
        self.uncaging_y=0
        self.SpineHeadToUncaging_um=0.2
        self.uncaging_relativeZ_moved = 0
        self.Spine_ZYX=False
        self.NthAc=0
        self.Num_zyx_drift = {}
        self.showWindow = False
        self.syncrate = 0
        
        self.x_um = 0 #For detecting not assigned value
        
        FOVres = self.get_val_sendCommand('State.Acq.FOV_default')
        self.FOV_default= [float(val) for val in FOVres.strip('][').split(', ')] 
        self.XMaxVoltage = self.get_val_sendCommand('State.Acq.XMaxVoltage')
        self.YMaxVoltage = self.get_val_sendCommand('State.Acq.YMaxVoltage')
        self.pixelsPerLine = self.get_val_sendCommand('State.Acq.pixelsPerLine')
        self.zoom = self.get_val_sendCommand('State.Acq.zoom')
        self.nSlices = self.get_val_sendCommand('State.Acq.nSlices')
        self.config_ini(ini_path)
    
    def config_ini(self,ini_path):
        config = configparser.ConfigParser()
        self.config=config
        config.read(ini_path)
        self.directionMotorX = int(config['Direction']['MotorX'])
        self.directionMotorY = int(config['Direction']['MotorY'])
        self.directionMotorZ = int(config['Direction']['MotorZ'])
        self.directionGalvoX = int(config['Direction']['GalvoX'])
        self.directionGalvoY = int(config['Direction']['GalvoY'])        
        print("\n\nDirection setting was modified at ",config['ModifiedDate']['Date'],"\n\n")
    
    def example_image(self):
        self.Spine_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Spine_example.png"
        self.Dendrite_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Dendrite_example.png"
        self.Uncaging_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Uncaging_example.png"

    def laser_sync(self):
        res = self.flim.sendCommand("State.Spc.datainfo.syncRate")
        # res = 'State.Spc.datainfo.syncRate = [0, 0]'
        self.syncrate = float(res.split(" = ")[1][1:-1].split(",")[0])
        
    def set_laser1_pow(self,laser_power: int):
        if laser_power>100 or laser_power < 0:
            print("input between 0 to 100")
        else:
            self.flim.sendCommand(f"State.Acq.power = [{laser_power}, 0, 0, 0]\n")

    def set_param(self,RepeatNum,interval_sec,ch_1or2,
                  uncaging_power = 20,
                  LoadSetting=False,SettingPath="",
                  expected_grab_duration_sec=0.5,
                  track_uncaging=False, drift_control=True,
                  ShowUncagingDetection=False, DoUncaging=False,
                  drift_cont_galvo=False, drift_cont_XY = True):
        self.RepeatNum=RepeatNum
        self.interval_sec=interval_sec
        self.uncaging_power = uncaging_power
        self.ch=ch_1or2-1
        self.track_uncaging=track_uncaging
        self.drift_control=drift_control
        self.ShowUncagingDetection=ShowUncagingDetection
        self.DoUncaging=DoUncaging
        self.drift_cont_galvo=drift_cont_galvo
        self.expected_grab_duration_sec=expected_grab_duration_sec
        self.drift_cont_XY = drift_cont_XY
        if LoadSetting==True:
            self.flim.sendCommand(f'LoadSetting, {SettingPath}')
            
    
    def get_position(self):
        for i in range(10):
            try:
                CurrentPos=self.flim.sendCommand('GetCurrentPosition') 
                a,x_str,y_str,z_str=CurrentPos.split(',')
                x,y,z=float(x_str),float(y_str),float(z_str)
                return x,y,z
            except:
                ("ERROR 105, Trouble in getting current position")
                pass
    
    def go_to_relative_pos_motor(self):
        x,y,z=self.get_position()
        z_str=str(z - self.directionMotorZ * self.relative_zyx_um[0])
        if self.drift_cont_XY == True:
            x_str=str(x - self.directionMotorX * self.relative_zyx_um[2])
            y_str=str(y - self.directionMotorY * self.relative_zyx_um[1])
        else: #leave 
            x_str=str(x)
            y_str=str(y)
        
        print(f"print SetMotorPosition,{x_str},{y_str},{z_str}")
        self.flim.sendCommand(f"SetMotorPosition,{x_str},{y_str},{z_str}")


    def go_to_relative_pos_motor_checkstate(self, sq_err_thre = 10, 
                                            first_wait_sec = 4, iter_wait_sec = 2):
        x,y,z=self.get_position()
        dest_x = x - self.directionMotorX * self.relative_zyx_um[2]
        dest_y = y - self.directionMotorY * self.relative_zyx_um[1]
        dest_z = z - self.directionMotorZ * self.relative_zyx_um[0]
        x_str=str(dest_x)
        y_str=str(dest_y)
        z_str=str(dest_z)
                
        print(f"print SetMotorPosition,{x_str},{y_str},{z_str}")
        self.flim.sendCommand(f"SetMotorPosition,{x_str},{y_str},{z_str}")
        sleep(first_wait_sec)
        
        dest_xyz = np.array([dest_x, dest_y, dest_z])
        for i in range(60):
            try:
                currentpos2 = self.get_val_sendCommand("State.Motor.motorPosition")
                currentpos_num2 = np.array(currentpos2[1:-1].split(","), dtype=float)
            except:
                sleep(1)
                continue
            
            diff2 =  currentpos_num2 - dest_xyz
            sum_sq_err2 = (diff2*diff2).sum()
            
            if sum_sq_err2 > 15:
                print("Not moved yet . . . . ")
                print(f"{i+2} th trial .....")
                sleep(iter_wait_sec)
                self.flim.sendCommand(f"SetMotorPosition,{x_str},{y_str},{z_str}")                
            else:
                break
        else:
            print(i," times attemps, but could not move stage.")

    def go_to_absolute_pos_motor_checkstate(self,dest_x,dest_y,dest_z, 
                                            sq_err_thre = 10, 
                                            first_wait_sec = 4, 
                                            iter_wait_sec = 2):
        x,y,z=self.get_position()
        x_str=str(dest_x)
        y_str=str(dest_y)
        z_str=str(dest_z)
                
        print(f"print SetMotorPosition,{x_str},{y_str},{z_str}")
        self.flim.sendCommand(f"SetMotorPosition,{x_str},{y_str},{z_str}")
        sleep(first_wait_sec)
        
        dest_xyz = np.array([dest_x, dest_y, dest_z])
        for i in range(10):
            # currentpos2 = self.get_val_sendCommand("State.Motor.motorPosition")
            # currentpos_num2 = np.array(currentpos2[1:-1].split(","), dtype=float)
            currentpos_num2=np.array(self.get_position())
            
            
            diff2 =  currentpos_num2 - dest_xyz
            sum_sq_err2 = (diff2*diff2).sum()
            
            if sum_sq_err2 > 10:
                print("Not moved yet . . . . ")
                print(f"{i+2} th trial .....")
                sleep(iter_wait_sec)
                self.flim.sendCommand(f"SetMotorPosition,{x_str},{y_str},{z_str}")

        #2023/6/6 added. MultiZ imaging after single plane requires this.
        self.flim.sendCommand('SetCenter')
        print("set center done")

    def get_galvo_xy(self):
        for i in range(10):
            try:
                res=self.flim.sendCommand('GetScanVoltageXY')
                x_galvo_now = float(res.split(',')[1])
                y_galvo_now = float(res.split(',')[2])
                return x_galvo_now, y_galvo_now
            except:
                sleep(0.2)
        return None


    def go_to_absolute_pos_um_galvo(self,z_move=False):
        self.flim.sendCommand(f"SetScanMirrorXY_um,{self.relative_zyx_um[2]},{self.relative_zyx_um[1]}")
        if z_move == True:
            self.relative_zyx_um[1]=0
            self.relative_zyx_um[2]=0
            self.go_to_relative_pos_motor()


    def go_to_relative_pos_galvo(self,z_move=False):
        x_galvo_now, y_galvo_now = self.get_galvo_xy()
        x_galvo_next = x_galvo_now - self.directionGalvoX*5*self.relative_zyx_um[2]/self.FOV_default[0]
        y_galvo_next = y_galvo_now - self.directionGalvoY*5*self.relative_zyx_um[1]/self.FOV_default[1]
        x_galvo_str = str(round(x_galvo_next,12))
        y_galvo_str = str(round(y_galvo_next,12))
        if self.drift_cont_XY == True:
            self.flim.sendCommand(f"SetScanVoltageXY,{x_galvo_str},{y_galvo_str}")
            print("y_galvo_now, ",y_galvo_now)
            print("y_galvo_next, ",y_galvo_next)
        if z_move == True:
            self.relative_zyx_um[1]=0
            self.relative_zyx_um[2]=0
            self.go_to_relative_pos_motor()
            
            
    def get_val_sendCommand(self,command):
        Reply = self.flim.sendCommand(command)
        value = Reply[len(command)+3:]
        try:
            return float(value)
        except:
            return value.replace('"','')
        
    
    def get_01_sendCommand(self,command):
        Reply = self.flim.sendCommand(command)
        value = Reply[len(command)+2:]
        return int(value)
    
    
    def reconnect(self):
        print("\n - - - Reconnect - - - \n")
        close_remote_control()
        self.flim = FLIM_Com()
        self.flim.start()
        self.flim.messageReceived += FLIM_message_received
        if self.flim.Connected:
            print("\n  Reconnected.  Good Connection now.\n")
        else:
            print("ERROR 101 - - - - - - ")
            print("Is FLIMage open?? ")
            self.nowGrabbing=False
            if window_exists('FLIMage! Version 4.0.4'):
                Exception("FLIMage is open, but cannot respond.")
            else:
                self.error_dict[datetime.now()] = "reconect" 
                print("Let me try to open FLIMage...")
                self.open_FLIMage()
                self.reconnect()
    
    def open_FLIMage(self):
        if len(self.error_dict) <= self.max_error_num:
            subprocess.Popen(self.flimage_exe)
            sleep(15)
        else:
            print(self.error_dict)
            Exception("Too many errors")
    
    
    def convert_shifts_pix_to_micro(self, shifts_zyx_pixel):
        x_relative = self.x_um*shifts_zyx_pixel[-1][2]
        y_relative = self.y_um*shifts_zyx_pixel[-1][1]
        z_relative = self.z_um*shifts_zyx_pixel[-1][0]
        
        self.relative_zyx_um = [z_relative,y_relative,x_relative]
        
    def append_drift_list(self):
        FileNumber = int(self.flimlist[-1][-8:-5])
        self.Num_zyx_drift[FileNumber]=[]
        for drift in self.relative_zyx_um:
            self.Num_zyx_drift[FileNumber].append(drift)
    
    def plot_drift(self,show=True):
        self.f = plt.figure()
        col_list=["g","k","m"]
        label_list=["z","y","x"]
        FileNum,zyx_drift_list=[int(self.flimlist[0][-8:-5])],[[0],[0],[0]]
        for Num in self.Num_zyx_drift:
            FileNum.append(Num)
            for i in range(3):
                zyx_drift_list[i].append(self.Num_zyx_drift[Num][i])
                
        for i in range(3):
            plt.plot(FileNum, zyx_drift_list[i], c=col_list[i], ls="-", label=label_list[i])
            plt.scatter(FileNum, zyx_drift_list[i],c=col_list[i])
            
        plt.xlabel("File#");plt.ylabel("\u03BCm")
        # ax = plt.figure().gca()
        self.f.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) #show only integer ticks
        plt.legend()
        savepath = self.NameStem+"drift.png"
        print(savepath)
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
        if show==True:
            plt.show()
    
    def align_two_flimfile(self,last_flimNth=-1):
        filelist=[self.flimlist[0],self.flimlist[last_flimNth]]
        print(filelist)
        # FileNum.append(int(flimlist[-1][-8:-5]))
        Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray(filelist,ch=self.ch)
        self.shifts_zyx_pixel, self.Aligned_4d_array=Align_4d_array(Tiff_MultiArray)
        self.x_um, self.y_um, self.z_um = get_xyz_pixel_um(iminfo)
        
        self.convert_shifts_pix_to_micro(self.shifts_zyx_pixel)
        self.append_drift_list()
      

    def align_2dframe_with_3d(self,ref_t=0,query_t=-1,ModifyZ=False):
        StackFilePath=self.flimlist[ref_t]
        SinglePlaneFilePath=self.flimlist[query_t]
        Z_plane,single_shift,Aligned_TYX_array = single_plane_align_with3dstack_flimfile(StackFilePath,SinglePlaneFilePath,ch=self.ch)
        self.Z_plane = Z_plane
        self.single_shift = single_shift
        self.Aligned_TYX_array = Aligned_TYX_array
        
        self.shifts_zyx_pixel = [[0,0,0],[self.Z_plane-self.Spine_ZYX[0],
                                     single_shift[0],single_shift[1]]]

        # self.convert_shifts_pix_to_micro(self.shifts_zyx_pixel)
      
    def acquisition_include_connect_wait(self):
        self.flim_connect_check()
        self.flim.sendCommand('StartGrab')
        self.wait_while_grabbing()
        
    def acquisition_include_connect_wait_short(self,sleep_every_sec=0.2):
        self.flim_connect_check()
        self.flim.sendCommand('StartGrab')
        self.wait_while_grabbing(sleep_every_sec=sleep_every_sec)
        
    def flim_connect_check(self):
        if self.flim.Connected==False:
            self.reconnect()
    
    def wait_while_grabbing(self,sleep_every_sec=2):
        sleep(self.expected_grab_duration_sec)
        error_count = 0
        for i in range(int(self.interval_sec/sleep_every_sec)):
            try:
                if self.get_01_sendCommand('IsGrabbing')==0:
                    print("BREAK wait_while_grabbing")
                    break
            except:
                print("ERROR on getting a reply for 'IsGrabbing'. Try again.")
                error_count+=1
                if error_count>5:
                    self.reconnect()
                    break
            sleep(sleep_every_sec)
            
    def send_uncaging_pos(self):
        self.flim.sendCommand(f"SetUncagingLocation, {self.uncaging_x}, {self.uncaging_y}")
    
    def wait_until_next(self,each_acquisition_from,sleep_every_sec=0.01):
        # print("wait from ",datetime.now()-each_acquisition_from)
        remainingSeconds = -1234
        previous_sec = -9876
        for i in range(int(self.interval_sec/sleep_every_sec)):
            each_acquisition_len=(datetime.now()-each_acquisition_from)
            if each_acquisition_len.seconds>=self.interval_sec:
                try:
                    self.uncaging_each.append(each_acquisition_len.total_seconds())
                except:
                    pass
                break
            
            if self.showWindow ==True:
                if remainingSeconds != int(each_acquisition_len.seconds - self.interval_sec):
                    remainingSeconds = int(self.interval_sec - each_acquisition_len.seconds)
                    if remainingSeconds != previous_sec:
                        previous_sec = remainingSeconds
                        try:
                            self.TxtWind.udpate(f"Time {str(remainingSeconds).rjust(3)} ".ljust(10))
                        except:
                            print("No tkinter window.", f" Reamining seconds: {remainingSeconds} ")
            sleep(sleep_every_sec)
            
            
    def define_uncagingPoint(self):
        # print("define uncaging point")
        # self.wait_while_grabbing()
        # print("END waiting")
        
        Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray([self.flimlist[0]],ch=self.ch)
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
        
        print("\n\n\n","self.Spine_ZYX=",self.Spine_ZYX)
        print("self.Dendrite_ZYX=",self.Dendrite_ZYX,"\n\n\n\n")

        
        
    def AlignSmallRegion(self):
        TrimmedAroundSpine=self.Aligned_4d_array[
                                                :,
                                                self.Spine_ZYX[0]-self.cuboid_ZYX[0]:self.Spine_ZYX[0]+self.cuboid_ZYX[0]+1,
                                                self.Spine_ZYX[1]-self.cuboid_ZYX[1]:self.Spine_ZYX[1]+self.cuboid_ZYX[1],
                                                self.Spine_ZYX[2]-self.cuboid_ZYX[2]:self.Spine_ZYX[2]+self.cuboid_ZYX[2],
                                                ]
        
        self.shifts_fromSmall, self.Small_Aligned_4d_array=Align_4d_array(TrimmedAroundSpine)

    

    def AlignSmallRegion_2d(self):
        print("self.Spine_ZYX",self.Spine_ZYX)
        print("self.cuboid_ZYX",self.cuboid_ZYX)
        
        TrimmedAroundSpine=self.Aligned_TYX_array[:,
                                                  int(self.Spine_ZYX[1]-self.cuboid_ZYX[1]):int(self.Spine_ZYX[1]+self.cuboid_ZYX[1]),
                                                  int(self.Spine_ZYX[2]-self.cuboid_ZYX[2]):int(self.Spine_ZYX[2]+self.cuboid_ZYX[2]),
                                                ]
        self.shifts_2d_fromSmall, self.Small_Aligned_3d_array=Align_3d_array(TrimmedAroundSpine)
        
        changeto3dlist=[[0,0,0],
                        [0,self.shifts_2d_fromSmall[-1][0],self.shifts_2d_fromSmall[-1][1]]]
        # for eachshift in self.shifts_2d_fromSmall:
        #     changeto3dlist.append(list(eachshift))
        print(np.array(changeto3dlist))
        self.shifts_fromSmall = np.array(changeto3dlist)



    def Align_2d_images_aroundspine(self):
        
        TrimmedAroundSpine=self.Aligned_TYX_array[
                                                :,
                                                self.Spine_ZYX[1]-self.cuboid_ZYX[1]:self.Spine_ZYX[1]+self.cuboid_ZYX[1],
                                                self.Spine_ZYX[2]-self.cuboid_ZYX[2]:self.Spine_ZYX[2]+self.cuboid_ZYX[2],
                                                ]
        self.shifts_2d_fromSmall, self.Small_Aligned_3d_array=Align_3d_array(TrimmedAroundSpine)
        
        changeto3dlist=[[0,0,0],
                        [0,self.shifts_2d_fromSmall[-1][0],self.shifts_2d_fromSmall[-1][1]]]
        
        print(np.array(changeto3dlist))
        self.shifts_fromSmall = np.array(changeto3dlist)


    def analyze_uncaging_point(self,threshold_coordinate=0.5,Gaussian_pixel=3):
        
        max_proj=self.Small_Aligned_4d_array[-1].max(axis=0)
        self.SpinePlaneImg=self.Small_Aligned_4d_array[-1,self.cuboid_ZYX[0],:,:]
        # single_plane_img = self.Small_Aligned_4d_array[self.cuboid_ZYX[0],:,:]
        blur = cv2.GaussianBlur(max_proj,(Gaussian_pixel,Gaussian_pixel),0)
        self.blur=blur; self.max_proj=max_proj
        
        dend_coord = [self.cuboid_ZYX[1] - (self.Spine_ZYX[1]-self.Dendrite_ZYX[1]),
                      self.cuboid_ZYX[2] - (self.Spine_ZYX[2]-self.Dendrite_ZYX[2])]
        Threshold =  min(blur[self.cuboid_ZYX[1],self.cuboid_ZYX[2]],blur[dend_coord[0],dend_coord[1]])*threshold_coordinate
        ret3,th3 = cv2.threshold(blur,Threshold,255,cv2.THRESH_BINARY)
        label_img = label(th3)
            
        self.binary_include_dendrite=np.zeros(label_img.shape)
        
        for each_label in range(1,label_img.max()+1):
            if label_img[dend_coord[0],dend_coord[1]] == each_label:
                self.binary_include_dendrite[label_img==each_label]=1
                
        if self.binary_include_dendrite.max() == 0:
            print("\n\n ERROR 102,  Cannot find dendrite \n No update in uncaging position. \n")
        
        else:
            regions = regionprops(label(self.binary_include_dendrite))
            self.props=regions[0]
            self.binarized=th3>0
            self.find_best_point()
            
    
    def analyze_uncaging_point_from_singleplane(self, single_plane_YXarray,
                                               threshold_coordinate=0.5,Gaussian_pixel=3):

        blur = cv2.GaussianBlur(single_plane_YXarray,(Gaussian_pixel,Gaussian_pixel),0)
        Threshold = blur[self.Spine_ZYX[1],self.Spine_ZYX[2]]*threshold_coordinate
        print(" x, y, Threshold",self.Spine_ZYX[1],self.Spine_ZYX[2],Threshold)
        ret3,th3 = cv2.threshold(blur,Threshold,255,cv2.THRESH_BINARY)
        label_img = label(th3)
        
        self.binary_include_dendrite=np.zeros(label_img.shape)
        for each_label in range(1,label_img.max()+1):
            if label_img[self.Spine_ZYX[1],self.Spine_ZYX[2]] == each_label:
                self.binary_include_dendrite[label_img==each_label]=1
                
        if self.binary_include_dendrite.max() == 0:
            print("\n\n ERROR 104,  Cannot find spine \n No update in uncaging position. \n")
    
        return 

    def find_best_point_dend_ori_given(self, direction, dend_orientation,
                                       uncaging_Y, uncaging_X,
                                       ignore_stage_drift=False):
        maxY,maxX = self.binary_include_dendrite.shape
        candi_y, candi_x = uncaging_Y, uncaging_X
        for i in range(1002):
            if self.binary_include_dendrite[int(candi_y),int(candi_x)]>0:
                candi_x = candi_x + math.cos(dend_orientation)*direction
                candi_y = candi_y - math.sin(dend_orientation)*direction
                if candi_y>=maxY or candi_x >=maxY or candi_x<=0 or candi_y<=0:
                    print("Could not find spine head")
                    candi_y, candi_x = uncaging_Y, uncaging_X
                    break
                
                # print("candi_x,candi_y",candi_x,candi_y)
            else:
                # Assuming that x and y have same resolution
                distance_pixel = self.SpineHeadToUncaging_um/self.x_um
                candi_x = candi_x + math.cos(dend_orientation)*direction*distance_pixel
                candi_y = candi_y - math.sin(dend_orientation)*direction*distance_pixel
                # print("distance_pixel,candi_x,candi_y",distance_pixel,candi_x,candi_y)
                break
        if i > 1000:
            print("Error 113 - -  ")
            candi_x, candi_y = self.cuboid_ZYX[2],self.cuboid_ZYX[1]
        
        self.candi_x = candi_x
        self.candi_y = candi_y

        if ignore_stage_drift==False:
            self.uncaging_x = candi_x - self.shifts_zyx_pixel[-1][2] - self.shifts_fromSmall[-1][2]
            self.uncaging_y = candi_y - self.shifts_zyx_pixel[-1][1] - self.shifts_fromSmall[-1][1]
        else:
            self.uncaging_x = candi_x - self.shifts_fromSmall[-1][2]
            self.uncaging_y = candi_y - self.shifts_fromSmall[-1][1]
    
        
    def analyze_uncaging_point_TYX(self,threshold_coordinate=0.5,Gaussian_pixel=3):
        single_plane = self.Small_Aligned_3d_array[-1]
        self.SpinePlaneImg=single_plane
        # single_plane_img = self.Small_Aligned_4d_array[self.cuboid_ZYX[0],:,:]
        blur = cv2.GaussianBlur(single_plane,(Gaussian_pixel,Gaussian_pixel),0)
        self.blur=blur
        
        Threshold = blur[self.cuboid_ZYX[1],self.cuboid_ZYX[2]]*threshold_coordinate
        print(Threshold)
        ret3,th3 = cv2.threshold(blur,Threshold,255,cv2.THRESH_BINARY)
            
        self.binary_include_dendrite=th3
        self.binarized=th3
        
        if self.binary_include_dendrite.max() == 0:
            print("\n\n ERROR 104,  Cannot find spine \n No update in uncaging position. \n")
        
        else:
            regions = regionprops(label(self.binary_include_dendrite))
            self.props=regions[0]
            
            if self.drift_control==False:
                ignore_stage_drift=False
            else:
                ignore_stage_drift=True
            self.find_best_point(TwoD=True, ignore_stage_drift=ignore_stage_drift)
        
        
    def find_best_point(self,TwoD=False, ignore_stage_drift=False):

        if TwoD==False:
            orientation = self.props.orientation
            y0,x0 = self.props.centroid
            self.orientation_based_on_3d = orientation
            self.dendrite_Centroid=[y0,x0]
            
        else:
            try:
                orientation = self.orientation_based_on_3d
                y0,x0 = self.dendrite_Centroid[0],self.dendrite_Centroid[1]
            except:
                orientation = self.props.orientation  
                y0,x0 = self.props.centroid
            
        candi_x, candi_y = self.cuboid_ZYX[2],self.cuboid_ZYX[1]

        ## Those x(y)_moved are used only for the calculation of the vector
        ## from dendrite center to spine. Uncaging spot will be calculated
        ## based on this vector.
        x_moved = x0 - self.cuboid_ZYX[2]
        y_moved = y0 - self.cuboid_ZYX[1]

        x_rotated = x_moved*math.cos(orientation) - y_moved*math.sin(orientation)

        if x_rotated<=0:
            direction = 1
        else:
            direction = -1

        # while True:
        for i in range(1000):
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
                pass
        
        if i > 990:
                print("Error 103 - -  ")
                candi_x, candi_y = self.cuboid_ZYX[2],self.cuboid_ZYX[1]
        
        self.candi_x = candi_x
        self.candi_y = candi_y

        if ignore_stage_drift==False:
            self.uncaging_x=self.Spine_ZYX[2]-self.cuboid_ZYX[2] +candi_x - self.shifts_zyx_pixel[-1][2] - self.shifts_fromSmall[-1][2]
            self.uncaging_y=self.Spine_ZYX[1]-self.cuboid_ZYX[1] +candi_y - self.shifts_zyx_pixel[-1][1] - self.shifts_fromSmall[-1][1]
        else:
            self.uncaging_x=self.Spine_ZYX[2]-self.cuboid_ZYX[2] +candi_x - self.shifts_fromSmall[-1][2]
            self.uncaging_y=self.Spine_ZYX[1]-self.cuboid_ZYX[1] +candi_y - self.shifts_fromSmall[-1][1]

        # print("\n self.shifts_zyx_pixel - - \n", self.shifts_zyx_pixel)
        # print("\n self.shifts_fromSmall - - \n", self.shifts_fromSmall)


        
    def find_best_point_by_slope(self,
                        spine_x, spine_y, 
                        dend_slope, dend_intercept,
                        TwoD=False, ignore_stage_drift=True):

        x_intersection, y_intersection = perpendicular_intersection(
                                            spine_x, spine_y, 
                                            dend_slope, dend_intercept)        
        vector_x = x_intersection - spine_x
        vector_y = y_intersection - spine_y
        len_vector = (vector_x**2 + vector_y**2)**0.5
        norm_vector_x = vector_x + len_vector
        norm_vector_y = vector_y + len_vector
        
        candi_x, candi_y = self.cuboid_ZYX[2],self.cuboid_ZYX[1]

        for i in range(1000):
            try:
                if self.binarized[int(candi_y),int(candi_x)]>0:
                    candi_x = candi_x + norm_vector_x
                    candi_y = candi_y + norm_vector_y
                else:
                    distance_pixel = self.SpineHeadToUncaging_um/self.x_um
                    candi_x = int(candi_x + norm_vector_x * distance_pixel)
                    candi_y = int(candi_y + norm_vector_y * distance_pixel)
                    break
            except:
                pass
        
        if i > 990:
                print("Error 104 - -  could not find spine boundary")
                candi_x, candi_y = self.cuboid_ZYX[2],self.cuboid_ZYX[1]
        
        self.candi_x = candi_x
        self.candi_y = candi_y

        if ignore_stage_drift==False:
            self.uncaging_x=self.Spine_ZYX[2]-self.cuboid_ZYX[2] +candi_x - self.shifts_zyx_pixel[-1][2] - self.shifts_fromSmall[-1][2]
            self.uncaging_y=self.Spine_ZYX[1]-self.cuboid_ZYX[1] +candi_y - self.shifts_zyx_pixel[-1][1] - self.shifts_fromSmall[-1][1]
        else:
            self.uncaging_x= self.Spine_ZYX[2] - self.cuboid_ZYX[2] + candi_x - self.shifts_fromSmall[-1][2]
            self.uncaging_y= self.Spine_ZYX[1] - self.cuboid_ZYX[1] + candi_y - self.shifts_fromSmall[-1][1]

        # print("\n self.shifts_zyx_pixel - - \n", self.shifts_zyx_pixel)
        # print("\n self.shifts_fromSmall - - \n", self.shifts_fromSmall)



    def go_to_uncaging_plane_z_assign(self,uncaging_Z):
        print("go_to_uncaging_plane")
        z = uncaging_Z
        NumZ = self.Aligned_4d_array.shape[1]
        z_move_um =  - self.z_um * (z -(NumZ - 1)/2)
        print("z_move_um ",z_move_um)
        self.relative_zyx_um = [z_move_um, 0, 0]
        self.go_to_relative_pos_motor_checkstate(first_wait_sec = 1,
                                                 iter_wait_sec = 1)

    def go_to_uncaging_plane(self):
        # sleep(2)
        print("go_to_uncaging_plane")
        z=self.Spine_ZYX[0]
        NumZ = self.Aligned_4d_array.shape[1]
        z_move_um =  - self.z_um * (z -(NumZ - 1)/2)
        z_relative = self.z_um*self.shifts_zyx_pixel[-1][0]
        print("z_move_um ",z_move_um)
        print("z_relative ",z_relative)
        self.relative_zyx_um = [z_move_um + z_relative,0,0]
        print(self.relative_zyx_um)
        self.go_to_relative_pos_motor()
        # self.uncaging_relativeZ_moved = z_move_um
        # sleep(2)


    def back_to_stack_plane(self):
        print("back_to_stack_plane")
        z=self.Spine_ZYX[0]
        NumZ = self.Aligned_4d_array.shape[1]
        z_move_um =  - self.z_um * (z -(NumZ + 1)/2)
        self.relative_zyx_um = [-z_move_um,0,0]
        print(self.relative_zyx_um)
        self.go_to_relative_pos_motor()
        # sleep(0.4)
        self.flim.sendCommand('SetCenter') #This is required. Otherwize, Z stage movement do not affect the Z stack center.
        # sleep(0.4)
        
        
    def drift_cont_single_plane(self,xy_stage_move=True,
                                z_stage_move=False):
        z=self.Spine_ZYX[0]
        shift_z = (self.Z_plane - z)

        if z_stage_move==True:
            temporary_shift_z = shift_z
        else:
            temporary_shift_z = 0
        
        if xy_stage_move==False:
            self.convert_shifts_pix_to_micro([[0,0,0],[temporary_shift_z,0,0]])
        else:
            self.convert_shifts_pix_to_micro([[0,0,0],[temporary_shift_z,
                                    self.single_shift[0],
                                    self.single_shift[1]]])

        self.append_drift_list()
        self.go_to_relative_pos_galvo(z_move=True)
        
        if xy_stage_move==False:
            self.convert_shifts_pix_to_micro([[0,0,0],[shift_z,
                                    self.single_shift[0],
                                    self.single_shift[1]]])
            
    def set_uncaging_power(self,power_percent):
        if power_percent not in range(101):
            raise Exception("power_percent should be integer and from 0 to 100.")
        else:
            self.flim.sendCommand(f"State.Uncaging.Power = {power_percent}")
            
          
    def acquire_independent(self):
        for NthAc in range(self.RepeatNum):
            self.NthAc=NthAc
            print("NthAc - - - - ,",NthAc)
            each_acquisition_from=datetime.now()
            
            self.wait_stop_acquisition_python()
            
            self.nowGrabbing=True
            
            self.flim.sendCommand('StartGrab')
            self.wait_while_grabbing(sleep_every_sec=0.2)
            self.nowGrabbing=False
            if NthAc < self.RepeatNum-1:
                self.wait_until_next(each_acquisition_from)
        self.loop=False
        
    def wait_stop_acquisition_python(self):
         print("wait_stop_acquisition_python")
         while self.stop_acquisition==True:
             sleep(0.01)
         print('break  wait_stop_acquisition_python')
         
    def wait_grab_status_python(self):
        print("wait_grab_status_python")
        while self.nowGrabbing==True:
            sleep(0.01)
        print('break  wait_grab_status_python')
    
    
    def set_xyz_um(self, iminfo):
        self.x_um, self.y_um, self.z_um = get_xyz_pixel_um(iminfo)


    def drift_uncaging_process(self):
        self.flimlist=glob.glob(os.path.join(self.folder,f"{self.NameStem}*.flim"))

        if self.x_um == 0:
            Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray([self.flimlist[0]],ch=self.ch)
            # self.x_um, self.y_um, self.z_um = get_xyz_pixel_um(iminfo)
            self.set_xyz_um(iminfo)

        checkedNth = -1
        for i in range(6000):
            if self.loop==False:
                print("BREAK  drift_uncaging_process")
                break
            # print("DRIFT CHECK, ",i)
            
            if self.NthAc>0 and checkedNth!=self.NthAc:
                checkedNth=self.NthAc
                query_t=-1
                print("Do ")
                
                self.wait_grab_status_python()
                self.flimlist=glob.glob(os.path.join(self.folder,f"{self.NameStem}*.flim"))
                self.align_2dframe_with_3d(query_t=query_t)
                self.stop_acquisition=True
                
                if self.drift_control==True:
                    self.drift_cont_single_plane(xy_stage_move=True)
                # self.flim_connect_check()
                
                if self.track_uncaging==True:
                    self.AlignSmallRegion_2d()
                    self.analyze_uncaging_point_TYX()
                    self.send_uncaging_pos()
            
                self.stop_acquisition=False
                plot_uncaging_point(self.props, self.binarized, self.blur, self.SpinePlaneImg, 
                                                self.candi_y, self.candi_x, self.cuboid_ZYX)

            else:
                sleep(0.3)
                
        print("thread2 while loop end")


    def start_repeat_short_for_rotate(self,each_lowhigh_instance,
                           uncaging_Z,
                           uncaging_Xlist, 
                           uncaging_Ylist,
                           direction_list,
                           orientation_list):
        
        self.start=datetime.now()
        
        self.folder = self.get_val_sendCommand("State.Files.pathName")
        self.NameStem = self.get_val_sendCommand("State.Files.baseName")
        self.childname = self.NameStem + str(int(self.get_val_sendCommand("State.Files.fileCounter"))).zfill(3)+".flim"
        self.uncaging_each=[]
        
        thread1 = threading.Thread(target=self.acquire_independent)
        self.stop_acquisition=False
        self.nowGrabbing=True
        self.NthAc = 0
        
        thread1.start()
        self.loop=True
        
        # self.drift_uncaging_process()
        
        modified_uncaging_xlist = []
        modified_uncaging_ylist = []
        
        for uncaging_X, uncaging_Y, direction, dend_orientation in zip(uncaging_Xlist, 
                                                                       uncaging_Ylist,
                                                                       direction_list,
                                                                       orientation_list):
            self.Spine_ZYX = [uncaging_Z, int(uncaging_Y), int(uncaging_X)]
            self.Aligned_TYX_array = each_lowhigh_instance.makingTYX_from3d_and_2d(uncaging_Z, each_lowhigh_instance.ch)
            self.AlignSmallRegion_2d()  #getting self.shifts_fromSmall = (np aray shift calculated from small region)
            
            flimfile_last = each_lowhigh_instance.latest_path()
            flimarray,_,_ = flim_files_to_nparray([flimfile_last],ch=0,normalize_by_averageNum=True)
            single_plane_YXarray = np.max(flimarray[0],axis=0)
            
            self.analyze_uncaging_point_from_singleplane(single_plane_YXarray)
            self.find_best_point_dend_ori_given(direction, dend_orientation,
                                                uncaging_Y, uncaging_X,ignore_stage_drift=True)
            modified_uncaging_xlist.append(self.uncaging_x)
            modified_uncaging_ylist.append(self.uncaging_y)
            
        self.flim.sendCommand("ClearUncagingLocation")
        for nth_pos in range(len(modified_uncaging_xlist)):
            self.flim.sendCommand(f"CreateUncagingLocation,{(modified_uncaging_xlist[nth_pos])},{(modified_uncaging_ylist[nth_pos])}")
        
        
        thread1.join()
        print("thread1 END")
        self.loop=False
        print("drift_uncaging_process END")
        
        plt.plot(self.uncaging_each)
        plt.ylabel("Each interval (sec)")
        plt.xlabel("Nth interval")
        plt.show()

        
        
        
    def start_repeat_short(self,single_plane_align=True):
        self.start=datetime.now()        
        self.folder = self.get_val_sendCommand("State.Files.pathName")
        self.NameStem = self.get_val_sendCommand("State.Files.baseName")
        self.childname = self.NameStem + str(int(self.get_val_sendCommand("State.Files.fileCounter"))).zfill(3)+".flim"
        self.uncaging_each=[]
        
        thread1 = threading.Thread(target=self.acquire_independent)
        self.stop_acquisition=False
        self.nowGrabbing=True
        self.NthAc = 0
        
        thread1.start()
        self.loop=True
        self.drift_uncaging_process()
        thread1.join()
        print("thread1 END")
        self.loop=False
        print("drift_uncaging_process END")
        
        plt.plot(self.uncaging_each)
        plt.ylabel("Each interval (sec)")
        plt.xlabel("Nth interval")
        plt.show()
        

        
        
    def start_repeat(self):
        self.start=datetime.now()
        
        self.folder = self.get_val_sendCommand("State.Files.pathName")
        self.NameStem = self.get_val_sendCommand("State.Files.baseName")
        self.childname = self.NameStem + str(int(self.get_val_sendCommand("State.Files.fileCounter"))).zfill(3)+".flim"
        self.TxtWind = TextWindow()
        self.showWindow =True
        
        for NthAc in range(self.RepeatNum):
            print("\n","= "*20,"\n"*2 ,NthAc+1 ," / ",self.RepeatNum,"\n"*2 , "= "*20 )
            each_acquisition_from=datetime.now()      
            self.TxtWind.udpate("Now Grabbing")
            # sleep(0.5) #### This small sleep will prevent from crash, sometimes....
            self.flim_connect_check()
            self.flim.sendCommand('StartGrab')  

            self.wait_while_grabbing()
            self.flimlist=glob.glob(os.path.join(self.folder,f"{self.NameStem}*.flim"))
            
            if len(self.flimlist)>1:
                self.align_two_flimfile()
                self.flim_connect_check()
                self.plot_drift(show=True)
                # self.plot_drift(show=False)

                if self.drift_control==True:
                    # if NthAc < self.RepeatNum-1:
                        if self.drift_cont_galvo==True:
                            self.go_to_relative_pos_galvo(z_move=True)
                        else:
                            self.go_to_relative_pos_motor()
                
                if self.track_uncaging==True:
                    if self.Spine_ZYX==False:
                        print("\n\n\n Spine position is not assigned. Continue.\n\n\n")
                        
                    self.AlignSmallRegion()
                    self.analyze_uncaging_point()
                    
                    #20230920
                    if self.drift_control==True:
                        # self.shifts_zyx_pixel =
                        self.uncaging_x += self.shifts_zyx_pixel[-1][2]
                        self.uncaging_y += self.shifts_zyx_pixel[-1][1]
                    
                    self.send_uncaging_pos()
                    
                    if self.ShowUncagingDetection==True:
                        try:
                            plot_uncaging_point(self.props, self.binarized, self.blur, self.SpinePlaneImg, 
                                            self.candi_y, self.candi_x, self.cuboid_ZYX)
                        except:
                            print("ERROR on plotting")
            
            if NthAc < self.RepeatNum-1:
                self.wait_until_next(each_acquisition_from)




if __name__ == "__main__":
    
    singleplane_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep05_128_singleplane.txt"
    # Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128.txt"
    # Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep05_128.txt"
    Zstack_ini=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128fast.txt"
    # singleplane_uncaging=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging.txt"
    # singleplane_uncaging=r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging_test.txt"
    
    FLIMageCont = Control_flimage()
    # FLIMageCont.directionMotorZ=-1 #sometimes, it changes. Why?
    
    # FLIMageCont.set_param(RepeatNum=5, interval_sec=30, ch_1or2=2,
    #                       LoadSetting=True,SettingPath=Zstack_ini)
    # FLIMageCont.start_repeat()
    
    # FLIMageCont.define_uncagingPoint()
    
    # # FLIMageCont.Spine_ZYX =[5, 61, 61]
    # # FLIMageCont.Dendrite_ZYX = [0, 56, 65] 

    # FLIMageCont.set_param(RepeatNum=3, interval_sec=30, ch_1or2=2,
    #                       LoadSetting=True,SettingPath=Zstack_ini,
    #                       track_uncaging=True,drift_control=True,
    #                       ShowUncagingDetection=True,drift_cont_galvo=True,expected_grab_duration_sec=22)        
    
    # FLIMageCont.start_repeat()
    
    # FLIMageCont.go_to_uncaging_plane()
    
    # FLIMageCont.set_param(RepeatNum=30, interval_sec=2, ch_1or2=2,
    #                       LoadSetting=True,SettingPath=singleplane_uncaging,
    #                       track_uncaging=True,drift_control=False,drift_cont_galvo=True,
    #                       ShowUncagingDetection=True,DoUncaging=False,expected_grab_duration_sec=1.5)
    # sleep(5)
    
    # FLIMageCont.start_repeat_short()

    # FLIMageCont.back_to_stack_plane()

    # FLIMageCont.set_param(RepeatNum=15, interval_sec=30, ch_1or2=2,
    #                       LoadSetting=True,SettingPath=Zstack_ini,
    #                       track_uncaging=True,drift_control=True,
    #                       ShowUncagingDetection=True,drift_cont_galvo=True,expected_grab_duration_sec=22)    
    
    # FLIMageCont.start_repeat()
    
    # FLIMageCont.flim.sendCommand(f'LoadSetting, {Zstack_ini}')
    # FLIMageCont.flim.sendCommand(f'SetDIOPanel, 1, 1')
    FLIMageCont.flim.sendCommand(f'SetUncagingLocation, 56.3, 8.2')
    
    # FLIMageCont.flim.sendCommand(f'State.Motor.resolutionX = 0.1')
    # FLIMageCont.flim.sendCommand(f'State.Motor.resolutionY = 0.1')
    # FLIMageCont.flim.sendCommand(f'State.Motor.resolutionZ = 0.2')
    
    # plt.imshow(FLIMageCont.Aligned_TYX_array[0])
    # plt.show()
    
    
    # plt.imshow(FLIMageCont.Aligned_TYX_array[1])
    # plt.show()

# each_acquisition_from=datetime.now() 
# for i in range(3):
#     FLIMageCont.flim.sendCommand('IsGrabbing')
#     print(datetime.now() - each_acquisition_from)
# print(i)    
