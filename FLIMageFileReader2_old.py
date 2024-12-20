# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:07:20 2022

This code is modified to avoid using libtiff
Instead, package tifffile is used.

@author: Tetsuya Watabe
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:39:42 2019
This class provides fucntion to read files created by FLIMage! software and calculate lifetime, intensity and lifetimeMap.
Detailed parameters are stored in FileReader.State

@author: Ryohei Yasuda
"""

# from libtiff import TIFF
#pip install libtiff will install this. 

import os,codecs,re
import numpy as np
from datetime import datetime
import tifffile
import ast

def convert_string(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            if s.lower() == 'true':
                return True
            elif s.lower() == 'false':
                return False
            else:
                if len(s)>2:
                    if s[0]=='"' and s[-1] == '"':
                        s = s[1:-1]
                return s
            
            
class FileReader:
    def __init__(self):        
        self.n_images = 1
        self.image = [] #List of images. image[slice][channel] is numpy array [y, x, t]
        self.acqTime = [] #List of acquired time.
        self.time = np.array(range(0, 64)) * 0.25 #time in nanoseconds.
        self.lifetime = np.array(64)  #Should be same length as slef.time.
        self.intensity = np.zeros((128,128))
        self.lifetimeMap = np.zeros((128,128))
        self.rgbLifetime = np.zeros((3, 128, 128))
        self.FLIM3D = np.array((128, 128, 64))
        self.filename = ''
        self.flim = False
        self.currentPage = 0
        self.currentChannel = 0
        self.currentZPage = 0
        
        #Parameters.
        self.n_time = [50,1] # number of time points
        self.nChannels = 0
        self.width = 128
        self.height = 128
        self.nFastZSlices = 1
        self.resolution = 250 #picoseconds
        self.FastZStack = False
        self.ZStack = False
            
        #Format
        self.ImageFormat = 'Linear'
        self.State = microscope_parameters() #All the detailed information is in State parameters.
        self.statedict = dict()
        
        #See 
        '''
        Parameters are in:
            State.Acq
            State.Spc.spcData
            State.spc.datainfo
            State.Uncaging
        '''
    
    def executeLine(self, info):
        #print(info)
        eq = info.split(' = ')
        try:
            if ('acqTime' not in eq[0]):
                exec('self.' + eq[0] + ' = ' + eq[1])
        except:
            pass #print('failed: self.' + eq[0] + ' = ' + eq[1])
    
    
    def decode_acquired_time(self,file_path):
        with codecs.open(file_path, 'r', 'utf-8', 'ignore') as file:
            lines = file.readlines()
            for line in lines:
                if "Acquired_Time =" in line:
                    # print(line[-26:-2])
                    self.acqTime.append(line[-26:-3])
            #     print(line)
    
    def decode_header(self, header, new = True):
        # infos = header.decode('ASCII').split('\r\n')
        infos = header.split('\r\n')
        statedict = dict()
        for info in infos:
            info = info.replace(";", "")
            #eq1 = info.split(' = ')
            #numPeriod = len(eq1[0].split('.')) - 1
            try:
                if 'Acq' or 'Spc.spcData' or 'Spc.datainfo' or 'Uncaging' in info:
                    #eq = info.split('.', numPeriod)
                    self.executeLine (info)
                if 'Format' in info:
                    format_str = info.split(' = ')
                    self.ImageFormat = format_str[1]
                if "State." in info:
                    keyitem = info.split(' = ')
                    if "[" in keyitem[1]:
                        keyitem[1] = ast.literal_eval(keyitem[1])
                    else:
                        keyitem[1] = convert_string(keyitem[1])
                    self.statedict[keyitem[0]] = keyitem[1]
            except:
                print("Could not read image info.")

            #Acquired time will be read in other function
            # if 'Acquired_Time' in info:
            #     format_str = info.split(' = ')
            #     self.acqTime.append(format_str[1])
        if self.currentPage == 0 and 'FLIMimage' not in infos[0]:
            print('This file may not be generated by FLIMage')
        
        self.n_time = []
        for i in range(0, self.State.Acq.nChannels):
            if self.State.Acq.acqFLIMA[i] and self.State.Acq.acquisition[i]:
                self.n_time.append(self.State.Spc.spcData.n_dataPoint)
            elif not self.State.Acq.acquisition[i]:
                self.n_time.append(0)
            else:
                self.n_time.append(1)
                
        if self.currentPage == 0:
            self.nChannels = self.State.Acq.nChannels
            self.width = self.State.Acq.pixelsPerLine
            self.height = self.State.Acq.linesPerFrame
            self.resolution = self.State.Spc.spcData.resolution
            self.FastZStack = self.State.Acq.FastZ_nSlices > 1 and self.State.Acq.fastZScan
            self.ZStack = self.State.Acq.ZStack
            if self.FastZStack:
                self.nFastZSlices = self.State.Acq.FastZ_nSlices
            else:
                self.nFastZSlices = 1
                    
    def decode_FLIM(self, flim):        
        image = []
        if (self.ImageFormat == 'ZLinear' or self.ImageFormat == 'Linear' or self.ImageFormat == 'ChTime_YX'):      
            flim1 = []
            if (self.ImageFormat == 'ZLinear'):
                flim1 = np.split(flim, self.nFastZSlices)
            else:
                flim1.append(flim)
            
            for flim_each in flim1:
                imageC = []                
                if self.ImageFormat == 'ChTime_YX':
                    image1 = np.reshape(flim_each, (self.height, self.width,  sum(self.n_time)), 'C')               
                    offset = 0
                    for i in range(0, self.nChannels):
                        offset2 = offset + self.n_time[i]
                        if self.n_time[i] > 0:
                            imageC.append(image1[:,:,offset:offset2])
                        else:
                            imageC.append(np.zeros(1)) #If not acquired, it return 0 value.
                        offset = offset2
                else:
                    
                    offset = 0
                    for i in range(0, self.nChannels):
                        offset2 = offset + self.height * self.width * self.n_time[i]

                        if self.n_time[i] > 0:
                            # imageC.append(np.reshape(flim_each[0, offset : offset2], (self.height, self.width,  self.n_time[i]), 'C'))
                            imageC.append(np.reshape(flim_each[offset : offset2], (self.height, self.width,  self.n_time[i]), 'C'))
                        else:
                            imageC.append(np.zeros(1)) #If not acquired, it return 0 value.
                        offset = offset2
                
                image.append(imageC)
        else: #I don't think there will any images with this format, but just in case.
            image1 = np.reshape(flim, (self.nChannels, self.height, self.width, self.n_time[0]), 'C')
            image = [np.split(image1, self.nChannels, 0)]
        return image
                    
    def read_imageFile(self, file_path, readImage = True):
        self.filename = file_path
        self.n_images = 1

        # These below require libtiff package
        # tif = TIFF.open(file_path, mode = 'r')
        #header = tif.GetField('ImageDescription')
        
        tif = tifffile.TiffFile(file_path, mode = 'r')

        for tag in tif.pages[0].tags:
            if len(str(tag.value))>500:
                header=str(tag.value)
        
        self.decode_header(header)        
        self.decode_acquired_time(file_path)
        
        tifarray=tif.asarray()
        
        if readImage:
            if len(tifarray.shape)==3:
                if (os.path.splitext(file_path)[-1] == '.flim'):
                    flim = np.array(tifarray[self.currentPage,0,:]).astype(np.ushort) #Sometimes image is stored in 8bit.
                    self.image.append(self.decode_FLIM(flim))
                    self.flim = True
                else:
                    self.image.append(np.array(tifarray[self.currentPage,0,:]))
                    self.flim = False
            else:
                if (os.path.splitext(file_path)[-1] == '.flim'):
                    flim = np.array(tifarray[0,:]).astype(np.ushort) #Sometimes image is stored in 8bit.
                    self.image.append(self.decode_FLIM(flim))
                    self.flim = True
                else:
                    self.image.append(np.array(tifarray[0,:]))
                    self.flim = False
                    
                
        self.currentPage += 1  
        
        # readdirectory do not work in tifflib
        # while tif.readdirectory():
        #
        while self.currentPage < tifarray.shape[0]:
            # print(self.currentPage ,tifarray.shape[0])
            # header = tif.GetField('ImageDescription')
            # self.decode_header(header, False)
            if readImage:
                if self.flim:
                    flim = np.array(tifarray[self.currentPage,0,:]).astype(np.ushort) #Sometimes image is stored in 8bit.
                    self.image.append(self.decode_FLIM(flim))            
                else:
                    self.image.append(tifarray[self.currentPage,0,:])
                    
            self.currentPage += 1    
            self.n_images = self.n_images + 1
            
        # self.currentPage = 0
        if self.flim:
            self.LoadFLIMFromMemory(0, 0, 0)
            
    def LoadFLIMFromMemory(self, page, fastZpage, channel):
        if self.pageValid(page, fastZpage, channel):
            if not self.FastZStack:
                fastZpage = 0
            self.FLIM3D = np.reshape(self.image[page][fastZpage][channel].astype(np.double), (self.height, self.width, self.n_time[channel]))
    
    def pageValid(self, page = 0, fastZpage = 0, channel = 0):
        fastZValid = (not self.FastZStack) or fastZpage >= 0 and fastZpage < self.nFastZSlices
        all_valid = self.flim and fastZValid and channel < self.nChannels and channel >= 0 and len(self.image) > page and len(self.image) > 0
        return all_valid
        
    def ifFLIMimage(self):
        if self.flim:
            sp = np.shape(self.FLIM3D)
            if np.size(sp) == 3 and sp[2] > 1:
                return True
            else:
                return False
        else:
            return False
        
    def calculateLifetimeCurve(self, page = 0, channel = 0, threshold = 0):
        if self.pageValid():
            img = self.FLIM3D
            intensity = self.intensity
            siz = np.shape(img)
            if threshold > 0:
                imgMask = np.reshape(np.repeat(intensity >= threshold, self.n_time[channel]), siz)
                img = img * imgMask
            self.lifetime = np.sum(np.sum(img, axis=0), axis=0)
            self.time = np.array(range(0, self.n_time[channel])) * self.resolution[channel] / 1000 # in nanoseconds
            
    def calculateIntensity(self):
        self.intensity = np.sum(self.FLIM3D ,2)
            
    def calculateLifetimeMap(self, lifetimeRange = [0, 64], lifetimeOffset = 0.5):
        if self.pageValid() and self.n_time[self.currentChannel] > 1:
            if lifetimeRange[0] < 0:
                lifetimeRange[0] = 0
            if lifetimeRange[1] > self.n_time[self.currentChannel]:
                lifetimeRange[1] = self.n_time[self.currentChannel]
                
            img = self.FLIM3D
            siz = np.shape(img)
            timeArray = np.array([range(0, self.n_time[self.currentChannel])])            
            timeMatrix = np.repeat(timeArray, siz[0] * siz[1], 0)
            timeMatrix = np.reshape(timeMatrix, siz)
            lt_range = range(lifetimeRange[0], lifetimeRange[1])
            sumImg = np.sum(img[:,:,lt_range], 2)
            sumImgZero = self.intensity == 0
            sumImg[sumImgZero] = 1 
            waitedSum = np.sum(img[:,:,lt_range] *  timeMatrix[:,:,lt_range], 2) / sumImg
            waitedSum[sumImgZero] = 0
            self.lifetimeMap = waitedSum * self.resolution[self.currentChannel] / 1000 - lifetimeOffset
            
    def calculateRGBLifetimeMap(self, lifetimeLimit = [1.6, 2.0], intensityLimit = [3, 25]):
        if self.pageValid() and self.n_time[self.currentChannel] > 1:
            gray = (self.lifetimeMap - lifetimeLimit[0]) / (lifetimeLimit[1] - lifetimeLimit[0])
            gray = 1 - gray
            gray[gray > 1] = 1
            gray[gray < 0] = 0
            part1 = np.bitwise_and(0 <= gray, gray < 1/3)
            part2 = np.bitwise_and(1/3 <= gray, gray < 2/3)
            part3 = np.bitwise_and(2/3 <= gray, gray <= 1)
            blue = part1 + part2 * (-3 * gray + 2)
            green = part1 * (3 * gray) + part2 + part3 * (-3 * gray + 3)
            red = part2 * (gray * 3 - 1) + part3
            alpha = (self.intensity - intensityLimit[0]) / (intensityLimit[1] - intensityLimit[0])
            alpha[alpha > 1] = 1
            alpha[alpha < 0] = 0
            rgbImage = np.array([red * alpha, green * alpha, blue * alpha])        
            self.rgbLifetime  = np.transpose(rgbImage, (1,2,0))
    
    def calculateAll(self, lifetimeRange = [0, 64], intensityLimit = [0, 20], lifetimeLimit = [1.6, 2.0], lifetimeOffset = 0.5):
            self.calculateIntensity()
            self.calculateLifetimeCurve(intensityLimit[0])
            self.calculateLifetimeMap(lifetimeRange, lifetimeOffset)
            self.calculateRGBLifetimeMap(lifetimeLimit, intensityLimit)
            
    def calculatePage(self, page = 0, fastZpage = 0, channel = 0, lifetimeRange = [0, 64], intensityLimit = [0, 20], lifetimeLimit = [1.6, 2.0], lifetimeOffset = 0.5):
        if self.pageValid(page, fastZpage, channel):
            self.LoadFLIMFromMemory(page, fastZpage, channel)
            self.calculateAll(lifetimeRange, intensityLimit, lifetimeLimit, lifetimeOffset)
    
    def export_statedict(self, export_txt_path):
        text = "FLIMimage parameters\n"
        for each_key in self.statedict:
            one_line = f"{each_key} = {self.statedict[each_key]};\n"
            text += one_line
        f = open(export_txt_path, "w")
        f.write(text)
        f.close()       
    
class microscope_parameters:
    def __init__(self):
        self.Acq = acquisition_parameters()
        self.Spc = spc_parameters()
        self.Uncaging = uncaging_parameters()
        
class acquisition_parameters:
    def __init__(self):
        self.pixelsPerLine = 128
        self.linesPerFrame = 128
        self.maxNFramePerFile = 4000
        self.aveFrame = False #obsolete for backward compatibility.
        self.aveFrameSeparately = False
        self.aveFrameA = [False, False]
        self.aveSlice = False
        self.ZStack = True
        self.acqFLIM = True #obsolete for backward compatibility.
        self.acqFLIMA = [True, True]
        self.acquisition = [True, True]
        self.nAveFrame = 4
        self.nAveragedFrames = 16
        self.nFrames = 64
        self.nSlices = 24
        self.nAveSlice = 4 #number of slices to be averaged.
        self.nAveragedSlices = 6 #number of "aveSlice"
        self.nImages = 1
        self.linesPerStripe = 32
        self.StripeDuringFocus = False
        self.BiDirectionalScan = False
        self.SineWaveScan = False
        self.flipXYScan = [False, False]
        self.switchXYScan = False
        self.LineClockDelay = 0.1
        self.nStripes = 4
        self.fastZScan = False
        self.fillFraction = 0.8
        self.scanFraction = 0.9
        self.ScanDelay = 0.074
        self.msPerLine = 2.0
        self.SliceMergin = 200
        self.FOV_default = [ 260.0, 260.0 ]
        self.object_magnification_default = 60
        self.field_of_view = [ 260.0, 260.0 ]
        # = FOV_default * object_magnification_default  / object_magnification
        self.object_magnification = 60
        self.scanVoltageMultiplier = [ 1.0, 1.0 ]
        self.zoom = 3
        self.XMaxVoltage = 5
        self.YMaxVoltage = 5.5
        self.Rotation = 0.0
        self.XOffset = 0.0
        self.YOffset = 0.0
        self.imageInterval = 120 #Time interval in s
        self.sliceInterval = 0 #Time interval in s
        self.sliceStep = 1.0 #Z step in um
        self.nChannels = 2
        self.outputRate = 250000
        self.inputRate = 250000
        self.triggerTime = datetime.now() #place holder
        self.power = [ 10, 10, 10, 10 ]
        self.externalTrigger = False
        self.ExpectedLaserPulseRate_MHz = 80.0
        self.FastZ_nSlices = 1
        self.FastZ_msPerLine = 0
        self.FastZ_Freq = 310000
        self.FastZ_Amp = 8
        self.FastZ_phase_detection_mode = False
        self.FastZ_Phase = [ 30.0, 30.0, 30.0 ]
        self.FastZ_PhaseRange = [ 35, 145 ]
        self.FastZ_umPerSlice = 1.0
        self.FastZ_degreePerSlice = 4.0

class uncaging_parameters:
    def __init__(self):
        self.name = "pulse set"
        self.pulse_number = 1
        self.uncage_whileImage = False
        self.sync_withFrame = False
        self.sync_withSlice = True
        self.FramesBeforeUncage = 32
        self.SlicesBeforeUncage = 32
        self.Uncage_FrameInterval = 4
        self.Uncage_SliceInterval = 1
        self.AnalogShutter_delay = 4 #ms
        self.DigitalShutter_delay = 8 #ms
        self.Mirror_delay = 4 #ms
        self.nPulses = 30
        self.pulseWidth = 6 #ms
        self.pulseISI = 2048
        self.pulseDelay = 4096
        self.sampleLength = 6000
        self.outputRate = 4000
        self.baselineBeforeTrain_forFrame = 2048
        self.pulseSetInterval_forFrame = 2048
        self.trainRepeat = 1
        self.trainInterval = 2048 #ms
        self.Power = 25 #percent.
        self.Position = [-1, -1] #Frac in image.
        self.PositionV = [ 0, 0 ] #Voltage
        self.CalibV = [ 0, 0 ] #Voltage
        
        #Multipositions::
        self.multiUncagingPosition = False
        self.rotatePosition = False
        self.currentPosition = 0 #0 = current position. 1, 2, 3... are the numbered positions.
        self.UncagingPositionsX = []
        self.UncagingPositionsY = []
        self.UncagingPositionsVX = [] #voltage
        self.UncagingPositionsVY = [] #voltage
        self.MoveMirrorsToUncagingPosition = True
        self.TurnOffImagingDuringUncaging = True
        
class spc_parameters:
    def __init__(self):
        self.datainfo = spc_datainfo()
        self.spcData = spc_spcData()

class spc_datainfo:
    def __init__(self):
        self.syncRate = [80000000, 80000000]
        self.countRate = [100, 100]

class spc_spcData:
    def __init__(self):
        self.n_dataPoint = 50
        self.device = 0
        self.time_per_unit = 1.24677e-08
        self.resolution = [ 250, 250 ] #ns
        self.sync_divider = [ 4, 4 ]
        self.sync_threshold = [ -50, -50 ]
        self.sync_zc_level = [ 0, 0 ]
        self.sync_offset = 7000
        self.ch_threshold = [ -5, -5 ]
        self.ch_zc_level = [ 0, 0 ]
        self.ch_offset = [ 0, 0 ]
        self.line_time_correction = 1.0
        self.measured_line_time_correction = 1.0
        
        #specific to Becker & Hickl
        self.BH_DLLDir = "C:\\Program Files (x86)\\BH\\SPCM\\DLL"
        self.BH_init_file = "C:\\Program Files (x86)\\BH\\SPCM\\DLL\\spcm.ini"            
        self.n_devicesBH = 2
        self.channelPerDeviceBH = 1
        self.acq_modeBH = 5
        self.lineID_BH = 1
        self.adc_res = 6 #6 bit
        self.tac_range = [ 50, 50 ]
        self.tac_gain = [ 4, 4 ]
        self.tac_offset = [ 5, 5 ] #%
        self.tac_limit_low = [ 5, 5 ] #%
        self.tac_limit_high = [ 95, 95 ] #%
        self.module_type = 150
        
        #parameters Specific to PicoQuant
        self.binning = 2  #PQ only.
        self.HW_Model = "THarp 260 N"
        self.n_devicesPQ = 1
        self.channelPerDevicePQ = 2
        self.TagID = 2
        self.acq_modePQ = 3
        self.lineID_PQ = 3 #M1, M2, M3, M4
        

if __name__ == "__main__":
    
    # If you want to select the path for .flim using GUI, use tkinter below.
    
    # import tkinter as tk
    # from tkinter import filedialog
    # plotWindow = tk.Tk()
    # plotWindow.wm_title('Fluorescence lifetime')                
    # plotWindow.withdraw()    
    # file_path = filedialog.askopenfilename()
    
    
    # If you will not use tkinter, declare path for .flim file.
    # file_path=r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20221129\concatenate_aligned_1122p8WTGFP_slice1_dendrites_.flim"
    # file_path=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230616\set1\pos1_high_039.flim"
    file_path = r"G:\ImagingData\Tetsuya\20241204\test_uncaging_010.flim"
    iminfo = FileReader()
    iminfo.read_imageFile(file_path, True) 
    
    print(np.array(iminfo.image).shape)
    
    
    # ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5]))
    # # Get intensity only data
    # imagearray=np.array(iminfo.image)
    # intensityarray=np.sum(imagearray,axis=-1)    
    # maxproj = np.max(intensityarray,axis=0)

    # ch=1
    
    # iminfo.export_statedict(file_path[:-5]+".txt")
    
    # # maxproj_singlech = np.max(intensityarray,axis=0)[0,ch,:,:]
    # # vmax = np.percentile(maxproj_singlech,99.5)
    # # plt.imshow(maxproj_singlech, vmin=0, vmax=vmax, cmap='gray')

    # # Showing intensity image and lifetime image
    # import matplotlib.pyplot as plt
    
    
    # intensity_range=[0,202]
    # for i in range(intensityarray.shape[0]):
    #     plt.imshow(intensityarray[i,0,ch,:,:],cmap="gray",
    #                vmin=intensity_range[0],vmax=intensity_range[1])
    #     plt.text(0,-10,str(i));plt.axis('off')
    #     plt.show()
    
    # for i in range(intensityarray.shape[0]):
    #     iminfo.calculatePage(page = i, fastZpage = 0, channel = 0, 
    #                   lifetimeRange = [5, 62], intensityLimit = [2, 100], 
    #                   lifetimeLimit = [1.6, 2.8], lifetimeOffset = 1.6)
        
    #     plt.imshow(iminfo.rgbLifetime)
    #     plt.text(0,-10,str(i));plt.axis('off')
    #     plt.show()
