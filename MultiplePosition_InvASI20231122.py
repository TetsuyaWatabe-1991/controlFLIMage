# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:08:39 2023

@author: yasudalab
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:24:13 2023

@author: WatabeT
"""
import os,glob,pathlib,math, sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")

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
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift

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
        
        bottom_lowmag_pos = list(copy.copy(self.lowmag_iminfo.statedict['State.Motor.motorPosition']))
        
        sliceStep = self.lowmag_iminfo.statedict['State.Acq.sliceStep']
        nSlices = self.lowmag_iminfo.statedict['State.Acq.nSlices']
        additionZ = sliceStep*(nSlices - 1)/2
        
        corrected_lowmag_pos = copy.copy(bottom_lowmag_pos)
        corrected_lowmag_pos[2] += additionZ
        
        self.lowmag_pos = copy.copy(corrected_lowmag_pos)
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
        
        shift, error, diffphase = phase_cross_correlation(firstYX, lastYX)
        img_corr = fourier_shift(np.fft.fftn(lastYX[:,:]), shift)
        aligned_array = np.fft.ifftn(img_corr).real        
        TYX = np.array([firstYX, aligned_array], dtype = np.uint16)
        
        return TYX, shift   
        
  
# FLIMageCont.Aligned_TYX_array, shift = each_lowhigh_instance.makingTYX_from3d_and_2d(uncaging_Z, each_lowhigh_instance.ch)

# flimlist = glob.glob(os.path.join(each_lowhigh_instance.highmag_iminfo.statedict["State.Files.pathName"],
#                                        each_lowhigh_instance.highmag_basename+"[0-9][0-9][0-9].flim"))

# TwoDflim_Nth = -1
# TwoDpath = os.path.join(each_lowhigh_instance.highmag_iminfo.statedict["State.Files.pathName"],
#                         each_lowhigh_instance.highmag_basename+f"{str(get_max_flimfiles(flimlist)+1+TwoDflim_Nth).zfill(3)}.flim")        
# ch=1
# firstTiff_MultiArray, _, _ = flim_files_to_nparray([each_lowhigh_instance.highmag_path], ch = ch)
# lastTiff_MultiArray, _, _ = flim_files_to_nparray([TwoDpath], ch = ch)
# z = uncaging_Z
# firstYX = np.array(firstTiff_MultiArray[0, z, :, :], dtype = np.uint16)
# lastYX = np.array(lastTiff_MultiArray[0, 0, :, :], dtype = np.uint16)

# shift, error, diffphase = phase_cross_correlation(firstYX, lastYX)
# img_corr = fourier_shift(np.fft.fftn(lastYX[:,:]), shift)
# aligned_array = np.fft.ifftn(img_corr).real        
# TYX = aligned_array


        


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

    
# list_of_fileset =     [['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230616\\set2\\pos1_low_002.flim', 'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230616\\set2\\pos1_high_001.flim'], ['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230616\\set2\\pos2_low_004.flim', 'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230616\\set2\\pos2_high_003.flim'], ['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230616\\set2\\pos3_low_006.flim', 'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230616\\set2\\pos3_high_005.flim']]
list_of_fileset = [['G:\\ImagingData\\Tetsuya\\20231122\\set1\\pos1_low_002.flim', 'G:\\ImagingData\\Tetsuya\\20231122\\set1\\pos1_high_001.flim'], ['G:\\ImagingData\\Tetsuya\\20231122\\set1\\pos2_low_002.flim', 'G:\\ImagingData\\Tetsuya\\20231122\\set1\\pos2_high_001.flim'], ['G:\\ImagingData\\Tetsuya\\20231122\\set1\\pos3_low_002.flim', 'G:\\ImagingData\\Tetsuya\\20231122\\set1\\pos3_high_001.flim']]

uncaging_setting = r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zsingle_128_uncaging.txt"
highmag_setting = r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\high_z11_kal10.txt"
lowmag_setting = r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\low_z11_step3_kal5.txt"
# highmag_setting = r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\Zstep1_128.txt"


LowHighset_instances = []

ch_1or2 = 2
uncagingpower = 20
uncaging_times = 30
laser1power_duringuncaging = 100

interval_min = 10

uncaging_nthacquisition = -30000

for eachfileset in list_of_fileset:
    LowHighset_instances.append(Low_High_mag_assign(lowmag_path = eachfileset[0],
                                                    highmag_path =eachfileset[1], 
                                                    ch_1or2 = ch_1or2,
                                                    skip_uncaging_pos=True))
    # print(LowHighset_instances[-1].uncaging_x,LowHighset_instances[-1].uncaging_y)

FLIMageCont = control_flimage()
FLIMageCont.interval_sec = 600
print("Now Grabbing")

num_T = 1000
# each_lowhigh_instance = LowHighset_instances[0]
# FLIMageCont.set_xyz_um(each_lowhigh_instance.highmag_iminfo)

# First=True
for nthacquisiton in range(num_T):
    print(f"ACQUISTION, {nthacquisiton+1}/{num_T}")
    try:
            
        each_start_time = datetime.now()
        
        for ind, each_lowhigh_instance in enumerate(LowHighset_instances):
            # nthacquisiton = 0
            # each_lowhigh_instance = LowHighset_instances[0]
            
            # if nthacquisiton ==0 and ind in [0,1,2]:
            #     continue
            
            ###Low magnification
            FLIMageCont.interval_sec = 60
            FLIMageCont.expected_grab_duration_sec = 20        
            each_lowhigh_instance.count_flimfiles()
            each_lowhigh_instance.send_acq_info(FLIMageCont, 'low')
            FLIMageCont.flim.sendCommand(f'LoadSetting, {lowmag_setting}')
    
            dest_x,dest_y,dest_z = each_lowhigh_instance.lowmag_pos
            if nthacquisiton == 0:
                # First=False
                dest_z = dest_z - 30
            
            FLIMageCont.go_to_absolute_pos_motor_checkstate(dest_x,dest_y,dest_z)        
            FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {each_lowhigh_instance.low_counter}')        
            # FLIMageCont.flim.sendCommand(f'State.Acq.power = [20, 10, 10, 10]')
            FLIMageCont.acquisition_include_connect_wait()
    
            if True:
                latest_lowmag_flim = each_lowhigh_instance.low_max_plus1_flim
                iminfo = FileReader()
                iminfo.read_imageFile(latest_lowmag_flim, True)         
                imagearray=np.array(iminfo.image)
                intensityarray=np.sum(imagearray,axis=-1)
                maxproj = np.max(intensityarray,axis=0)
                maxproj_singlech = np.max(intensityarray,axis=0)[0, ch_1or2 - 1, :, :]
                vmax = np.percentile(maxproj_singlech,99.5)
                plt.imshow(maxproj_singlech, vmin=0, vmax=vmax, cmap='gray')
                plt.axis('off')
                plt.title(f"Pos {ind+1}, T = {str(each_lowhigh_instance.high_counter).zfill(3)}, vmax = {int(vmax)}")
                savefolder = os.path.join(each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.pathName"],
                                          each_lowhigh_instance.lowmag_basename)
                os.makedirs(savefolder, exist_ok=True)
                savepath = os.path.join(savefolder,f"pos{ind+1}_t_{str(each_lowhigh_instance.high_counter).zfill(3)}.png")
                plt.savefig(savepath, dpi = 72, bbox_inches = 'tight')
                plt.close;plt.clf();
            
            
            FLIMageCont.relative_zyx_um, FLIMageCont.Aligned_4d_array = align_two_flimfile(
                                                                each_lowhigh_instance.lowmag_path, 
                                                                each_lowhigh_instance.low_max_plus1_flim,
                                                                each_lowhigh_instance.ch)
    
    
            print(FLIMageCont.relative_zyx_um)
            FLIMageCont.go_to_relative_pos_motor_checkstate()
            each_lowhigh_instance.update_pos_fromcurrent(FLIMageCont)
            each_lowhigh_instance.send_acq_info(FLIMageCont, 'high')
            FLIMageCont.flim.sendCommand(f'LoadSetting, {highmag_setting}')
    
            # FLIMageCont.flim.sendCommand(f'State.Acq.power = [20, 10, 10, 10]')
            FLIMageCont.flim.sendCommand(f'State.Files.fileCounter = {each_lowhigh_instance.high_counter}')
            FLIMageCont.acquisition_include_connect_wait()
            
            latest_highmag_flim = each_lowhigh_instance.high_max_plus1_flim
            savefolder = os.path.join(each_lowhigh_instance.lowmag_iminfo.statedict["State.Files.pathName"],
                                      each_lowhigh_instance.highmag_basename)
            os.makedirs(savefolder, exist_ok=True)
           
            iminfo = FileReader()
            iminfo.read_imageFile(latest_highmag_flim, True) 
            
            # Get intensity only data
            imagearray=np.array(iminfo.image)
            intensityarray=np.sum(imagearray,axis=-1)
            maxproj = np.max(intensityarray,axis=0)
            maxproj_singlech = np.max(intensityarray,axis=0)[0, ch_1or2 - 1, :, :]
            vmax = np.percentile(maxproj_singlech,99.5)
            plt.imshow(maxproj_singlech, vmin=0, vmax=vmax, cmap='gray')
            plt.axis('off')
            
            plt.title(f"Pos {ind+1}, T = {str(each_lowhigh_instance.high_counter).zfill(3)}, vmax = {int(vmax)}")
            savepath = os.path.join(savefolder,f"pos{ind+1}_t_{str(each_lowhigh_instance.high_counter).zfill(3)}.png")
            plt.savefig(savepath, dpi = 72, bbox_inches = 'tight')
            plt.close;plt.clf();
            
            
            FLIMageCont.relative_zyx_um, FLIMageCont.Aligned_4d_array = align_two_flimfile(each_lowhigh_instance.highmag_path,
                                                                  each_lowhigh_instance.high_max_plus1_flim,
                                                                  each_lowhigh_instance.ch)
            print(FLIMageCont.relative_zyx_um)
            FLIMageCont.go_to_relative_pos_motor_checkstate()
            each_lowhigh_instance.update_pos_fromcurrent(FLIMageCont)
            
    except:
        FLIMageCont = control_flimage()
        continue
    
    now2 = datetime.now()
    
    while True:
        if (now2 - each_start_time).seconds/60 > interval_min:
            break
        else:
            print("remaining time (sec):", int((now2 - each_start_time).seconds - interval_min*60))
            sleep(5)
            now2 = datetime.now()
                        
            # FLIMageCont.start_repeat_short_for_rotate(each_lowhigh_instance,
            #                        uncaging_Z,
            #                        uncaging_Xlist, 
            #                        uncaging_Ylist,
            #                        direction_list,
            #                        orientation_list)
                
                
                
                
                
                
                
                
                