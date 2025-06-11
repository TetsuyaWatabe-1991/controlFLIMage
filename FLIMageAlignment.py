# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:56:34 2022

@author: yasudalab
"""
import os,glob,math
from pathlib import Path
from FLIMageFileReader2 import FileReader
import matplotlib.pyplot as plt
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from scipy.signal import medfilt
from datetime import datetime
from skimage.transform import resize

def get_flimfile_list(one_file_path):
    filelist=glob.glob(one_file_path[:-8]+'[0-9][0-9][0-9].flim')
    # filelist=filelist[:-1]
    return filelist

def CenterPosGet(Tiff_MultiArray,ratio=0.5):
    shape=Tiff_MultiArray.shape
    z_start, z_to = NthDim_Center(shape,1,ratio)
    y_start, y_to = NthDim_Center(shape,2,ratio)
    x_start, x_to = NthDim_Center(shape,3,ratio)
    return z_start, z_to, y_start, y_to, x_start, x_to

def NthDim_Center(shape,NthDim,ratio=0.5):
    NthDimLen=shape[NthDim]
    center=NthDimLen/2
    start=int(center*(1-ratio))
    to=int(center*(1+ratio))
    return start,to

def plot_maxproj(flimpath, ch1or2, savefig=True):
    
    ch = ch1or2 - 1
    Tiff_MultiArray, _, _ = flim_files_to_nparray([flimpath],ch=ch)
    ZYXarray = Tiff_MultiArray[0]
    maxproj = np.max(ZYXarray, axis = 0)
    vmax = np.percentile(maxproj,99.5)

    plt.imshow(maxproj, cmap="gray", vmin = 0, vmax = vmax)    
    plt.axis('off')
    plt.title(flimpath[-8:-5])    
 
    if savefig==True:
        os.makedirs(flimpath[:-8],exist_ok=True)
        savepath = os.path.join(flimpath[:-8],"0.png")
        plt.savefig(savepath, dpi = 72, bbox_inches = 'tight')
    
    plt.show()

def fft_drift_3d(ref_array ,query_array,
                 MedianFilter = False, Ksize = 3):
    if MedianFilter==True:
        ref_array_for_correlation = medfilt(ref_array, kernel_size = Ksize)
        query_array_for_correlation  = medfilt(query_array, kernel_size = Ksize)
    else:
        ref_array_for_correlation = ref_array
        query_array_for_correlation  = query_array
    # shift, error, diffphase = phase_cross_correlation(ref_array, query_array)
    shift, error, diffphase = phase_cross_correlation(ref_array_for_correlation,
                                                      query_array_for_correlation,
                                                      upsample_factor=4)
    img_corr = fourier_shift(np.fft.fftn(query_array), shift)
    aligned_array = np.fft.ifftn(img_corr).real
    return aligned_array, shift


def Align_3d_array(Tiff_MultiArray, MedianFilter = False, Ksize = 3):
    shifts = []
    Aligned_3d = []
    
    for NthTime in range(Tiff_MultiArray.shape[0]):    
        aligned_array, shift = fft_drift_3d(Tiff_MultiArray[0],Tiff_MultiArray[NthTime],
                                            MedianFilter, Ksize)
        shifts.append(shift)
        Aligned_3d.append(aligned_array)
    # for plotting the xy shifts over time
    shifts = np.array(shifts)
    Aligned_3d_array = np.array(Aligned_3d)
    return shifts, Aligned_3d_array


def Align_4d_array(Tiff_MultiArray):
    shifts = []
    Aligned_4d = []
    
    for NthTime in range(Tiff_MultiArray.shape[0]):    
        aligned_array, shift = fft_drift_3d(Tiff_MultiArray[0],Tiff_MultiArray[NthTime])
        shifts.append(shift)
        Aligned_4d.append(aligned_array)
    # for plotting the xy shifts over time
    shifts = np.array(shifts)
    Aligned_4d_array = np.array(Aligned_4d)
    return shifts, Aligned_4d_array

def align_two_flimfile(flim_1, flim_2, ch, return_pixel = False):
    if False:
        filelist = [r"G:\ImagingData\Tetsuya\20250511\2ndslice\lowmag2__highmag_6_002.flim",
                    r"G:\ImagingData\Tetsuya\20250511\2ndslice\lowmag2__highmag_6_040.flim"]
        ch = 1#0 is for gfp, 1 is for RFP
    
    filelist = [flim_1, flim_2]
    print("filelist", filelist)
    Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray(filelist,ch=ch)
    shifts_zyx_pixel, Aligned_4d_array = Align_4d_array(Tiff_MultiArray)
    # print(shifts_zyx_pixel)
    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
    
    x_relative = x_um*shifts_zyx_pixel[-1][2]
    y_relative = y_um*shifts_zyx_pixel[-1][1]
    z_relative = z_um*shifts_zyx_pixel[-1][0]
    
    if return_pixel == True:
        return [z_relative, y_relative, x_relative], Aligned_4d_array, shifts_zyx_pixel
    else:
        return [z_relative, y_relative, x_relative], Aligned_4d_array


def align_two_flimfile_different_resolution(flim_1, flim_2, ch, return_pixel = False, debug = False):    
    Tiff_MultiArray_1, iminfo_1, _ = flim_files_to_nparray([flim_1],ch=ch)
    Tiff_MultiArray_2, iminfo_2, _ = flim_files_to_nparray([flim_2],ch=ch)
    threeD_array_1 = Tiff_MultiArray_1[0]
    threeD_array_2 = Tiff_MultiArray_2[0]
    assert threeD_array_1.shape[0] == threeD_array_2.shape[0]
    lower_x_dim = min(threeD_array_1.shape[2],threeD_array_2.shape[2])
    lower_y_dim = min(threeD_array_1.shape[1],threeD_array_2.shape[1])
    
    resized_array1 = resize(threeD_array_1, (11, lower_y_dim, lower_x_dim), anti_aliasing=False)
    resized_array2 = resize(threeD_array_2, (11, lower_y_dim, lower_x_dim), anti_aliasing=False)


    Tiff_MultiArray = np.array([resized_array1,resized_array2])
    shifts_zyx_pixel, Aligned_4d_array = Align_4d_array(Tiff_MultiArray)
    # print(shifts_zyx_pixel)
    x_um_1, y_um_1, z_um_1 = get_xyz_pixel_um(iminfo_1)
    x_um_2, y_um_2, z_um_2 = get_xyz_pixel_um(iminfo_2)
    x_um = max(x_um_1, x_um_2)
    y_um = max(y_um_1, y_um_2)
    z_um = max(z_um_1, z_um_2)
    
    x_relative = x_um*shifts_zyx_pixel[-1][2]
    y_relative = y_um*shifts_zyx_pixel[-1][1]
    z_relative = z_um*shifts_zyx_pixel[-1][0]
    
    if debug == True:
        print("x_um_1, y_um_1, z_um_1", x_um_1, y_um_1, z_um_1)
        print("x_um_2, y_um_2, z_um_2", x_um_2, y_um_2, z_um_2)
        print("x_um, y_um, z_um", x_um, y_um, z_um)
        print("lower_x_dim, lower_y_dim", lower_x_dim, lower_y_dim)
        print("threeD_array_1.shape, threeD_array_2.shape", threeD_array_1.shape, threeD_array_2.shape)
        vmax = threeD_array_1.max()
        plt.figure(figsize = (10,10))
        plt.subplot(2,2,1)
        plt.imshow(threeD_array_1[:,:,:].max(axis = 0), cmap = "gray", interpolation="nearest", vmax = vmax, aspect = "equal")
        plt.title("threeD_array_1  " + os.path.basename(flim_1))
        plt.subplot(2,2,2)
        plt.imshow(threeD_array_2[:,:,:].max(axis = 0), cmap = "gray", interpolation="nearest", vmax = vmax, aspect = "equal")
        plt.title("threeD_array_2  " + os.path.basename(flim_2))
        
        plt.subplot(2,2,3)
        plt.imshow(threeD_array_1[:,:,:].max(axis = 1), cmap = "gray", interpolation="nearest", vmax = vmax, aspect = "equal")
        plt.title("threeD_array_1  " + os.path.basename(flim_1))
        plt.subplot(2,2,4)
        plt.imshow(threeD_array_2[:,:,:].max(axis = 1), cmap = "gray", interpolation="nearest", vmax = vmax, aspect = "equal")
        plt.title("threeD_array_2  " + os.path.basename(flim_2))
        plt.show()


    if return_pixel == True:
        return [z_relative, y_relative, x_relative], Aligned_4d_array, shifts_zyx_pixel
    else:
        return [z_relative, y_relative, x_relative], Aligned_4d_array



def flim_files_to_nparray(filelist,ch=0,normalize_by_averageNum=True):
    FourDimList=[]
    timestamp_list,relative_sec_list = [],[]
    First=True
    for file_path in filelist:
        iminfo = FileReader()
        print(file_path)
        iminfo.read_imageFile(file_path, True) 
        # Get intensity only data
        imagearray=np.array(iminfo.image)
        
        nAveFrame = iminfo.State.Acq.nAveFrame
        if normalize_by_averageNum==False:
            DivBy = 1
        else:
            DivBy = nAveFrame
            
        if First:
            First=False
            imageshape=imagearray.shape


        if imagearray.shape == imageshape:
            intensityarray=(12*np.sum(imagearray,axis=-1))/DivBy
            FourDimList.append(intensityarray)
            timestamp_list.append(datetime.strptime(iminfo.acqTime[0],'%Y-%m-%dT%H:%M:%S.%f'))
            relative_sec_list.append((timestamp_list[-1] - timestamp_list[0]).seconds)
        else:
            print(file_path,'<- skipped read')
            
     
    print("ch",ch)
    Tiff_MultiArray=np.array(FourDimList,dtype=np.uint16)[:,:,0,ch,:,:]
    return Tiff_MultiArray, iminfo, relative_sec_list



def flim_files_to_nparray_uncaging(filelist,
                                  ch=0,
                                  normalize_by_averageNum=True):
    FourDimList=[]
    timestamp_list, relative_sec_list, uncaging_relative_sec_list = [],[],[]
    First=True
    for file_path in filelist:
        iminfo = FileReader()
        print(file_path)
        iminfo.read_imageFile(file_path, True) 
        # Get intensity only data
        imagearray=np.array(iminfo.image)
        
        nAveFrame = iminfo.State.Acq.nAveFrame
        if normalize_by_averageNum==False:
            DivBy = 1
        else:
            DivBy = nAveFrame
            
        if First:
            First=False
            imageshape=imagearray.shape


        if imagearray.shape == imageshape:
            intensityarray=(12*np.sum(imagearray,axis=-1))/DivBy
            FourDimList.append(intensityarray)
            timestamp_list.append(datetime.strptime(iminfo.acqTime[0],'%Y-%m-%dT%H:%M:%S.%f'))
            relative_sec_list.append((timestamp_list[-1] - timestamp_list[0]).seconds)
        else:
            print(imagearray.shape)
            if (imagearray.shape[0] > 29):
                print(file_path,'<- uncaging')
                uncaging_array = ((12*np.sum(imagearray,axis=-1))/DivBy)[:,0,ch,:,:]
                uncaging_iminfo = iminfo
                
                for each_acqtime in iminfo.acqTime:
                    uncaging_relative_sec_list.append(
                        (datetime.strptime(each_acqtime,'%Y-%m-%dT%H:%M:%S.%f') 
                         - timestamp_list[0]).seconds)
            else:
                print(file_path,'<- skipped read')
    print("ch",ch)
    Tiff_MultiArray=np.array(FourDimList,dtype=np.uint16)[:,:,0,ch,:,:]
    
    return Tiff_MultiArray, iminfo, relative_sec_list, uncaging_array, uncaging_iminfo, uncaging_relative_sec_list


def get_xyz_pixel_um(iminfo):
    field_len_x = iminfo.State.Acq.FOV_default[0]
    field_len_y = iminfo.State.Acq.FOV_default[1]
    zoom = iminfo.State.Acq.zoom
    pixels_x = iminfo.State.Acq.pixelsPerLine
    pixels_y = iminfo.State.Acq.linesPerFrame
    x_um = field_len_x/zoom/pixels_x
    y_um = field_len_y/zoom/pixels_y
    z_um = iminfo.State.Acq.sliceStep 
    return x_um, y_um, z_um

def make_save_folders(one_file_path):
    # saveFolder=os.path.join(path.Path(one_file_path).parent,"Analysis",path.Path(one_file_path).stem)
    saveFolder=os.path.join(Path(one_file_path).parent,"Analysis",Path(one_file_path).stem)
    os.makedirs(saveFolder,exist_ok=True)
    EachImgsaveFolder=os.path.join(saveFolder,"EachImg")
    os.makedirs(EachImgsaveFolder,exist_ok=True)
    return saveFolder, EachImgsaveFolder

def plot_alignment_shifts(shifts,iminfo,saveFolder='',savefigure=True,
                          relative_sec_list=False):
    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)

    x_drift = np.array([shifts[i][2] for i in range(0,shifts.shape[0])])*x_um
    y_drift = np.array([shifts[i][1] for i in range(0,shifts.shape[0])])*y_um
    z_drift = np.array([shifts[i][0] for i in range(0,shifts.shape[0])])*z_um
    
    if relative_sec_list==False:
        plt.plot(z_drift, '--g' , label = ' Z drift')
        plt.plot(x_drift, '--m' , label = ' X drfit')
        plt.plot(y_drift, '--k' , label = ' Y drfit')
        plt.ylabel("\u03BCm");plt.xlabel("Time")
    else:
        plt.plot(relative_sec_list, z_drift, '--g' , label = ' Z drift')
        plt.plot(relative_sec_list, x_drift, '--m' , label = ' X drfit')
        plt.plot(relative_sec_list, y_drift, '--k' , label = ' Y drfit')
        plt.ylabel("\u03BCm");plt.xlabel("Time (sec)")
    plt.legend()
    savepath=os.path.join(saveFolder,"XYZdrift.png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
   
    
def single_plane_align_with3dstack_flimfile(StackFilePath,SinglePlaneFilePath,ch=0,
                                            predefined_Z = False, Z_plane = -1):
    # StackFilePath=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221220\GFPneuron1_20_019.flim"
    # SinglePlaneFilePath=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221220\GFPneuron1_20_022.flim"  
    print('\n\n ---- \n',StackFilePath,'\n\n',SinglePlaneFilePath)
    Stack_file_list=[StackFilePath]
    SinglePlaneList=[SinglePlaneFilePath]
    ZYX_Stack_array, _ , _ = flim_files_to_nparray(Stack_file_list,ch=ch)
    YX_SinglePlane_array, _ , _ = flim_files_to_nparray(SinglePlaneList,ch=ch)

    if predefined_Z == True:
        print("\n\n  predefined_Z is True;  Using 2d align mode \n\n")
        Tiff_MultiArray = np.array([ZYX_Stack_array[0, Z_plane, :,:] , YX_SinglePlane_array[0,0,:,:]])
        shift, Aligned_TYX_array = Align_3d_array(Tiff_MultiArray, 
                                                  MedianFilter=True,
                                                  Ksize=3)
        single_shift = shift[-1]
    else:
        Z_plane, single_shift, Aligned_TYX_array = single_plane_align_with3dstack(ZYX_Stack_array[0,:,:,:],YX_SinglePlane_array[0,0,:,:])
    
    return Z_plane,single_shift, Aligned_TYX_array



    
# def single_plane_align_with3dstack_flimfile(StackFilePath,SinglePlaneFilePath,ch=1):
#     # StackFilePath=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221220\GFPneuron1_20_019.flim"
#     # SinglePlaneFilePath=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221220\GFPneuron1_20_022.flim"  
#     print('\n\n ---- \n',StackFilePath,'\n\n',SinglePlaneFilePath)
#     Stack_file_list=[StackFilePath]
#     SinglePlaneList=[SinglePlaneFilePath]
#     ZYX_Stack_array, _ , _ = flim_files_to_nparray(Stack_file_list,ch=ch)
#     YX_SinglePlane_array, _ , _ = flim_files_to_nparray(SinglePlaneList,ch=ch)
#     Z_plane,single_shift, Aligned_TXY_array = single_plane_align_with3dstack(ZYX_Stack_array[0,:,:,:],YX_SinglePlane_array[0,0,:,:])

#     return Z_plane,single_shift, Aligned_TXY_array
    
    

def mean_square_error(img1,img2,intensity_threshold=5,required_pixel=100):
     if math.prod(img1.shape)!=math.prod(img2.shape):        
         raise TypeError("Given images have different shapes")
         
     # Mask = (img1>intensity_threshold)|(img2>intensity_threshold)
     Mask = (img1>intensity_threshold)
     pixel_size = Mask.sum()
     
     if pixel_size<required_pixel:
         return 987654321
     
     diff= (img1.astype(float)-img2.astype(float))
     err = (Mask*(diff**2)).sum()
     mse = err/(float(pixel_size))
     # print("pixel_size,mse,",pixel_size,mse)
     
     return mse   
   
    
def single_plane_align_with3dstack(ZYX_Stack_array,YX_SinglePlane_array):    
    min_mse= 9876543210
    Z_plane= 9876543210
    for i in range(ZYX_Stack_array.shape[0]):
        ref_img = ZYX_Stack_array[i,:,:]
        moving_img = YX_SinglePlane_array
        shift, error, diffphase = phase_cross_correlation(ref_img, moving_img)
        img_corr = fourier_shift(np.fft.fftn(YX_SinglePlane_array[:,:]), shift)
        aligned_array = np.fft.ifftn(img_corr).real
        
        cut_region=np.abs(shift).astype(int)
        
        for j in range(2):
            if cut_region[j]==0:
                cut_region[j]=-YX_SinglePlane_array.shape[j]
        
        ref_array=ZYX_Stack_array[i,
                                  cut_region[0]:-cut_region[0],
                                  cut_region[1]:-cut_region[1]]
        
        query_array=aligned_array[cut_region[0]:-cut_region[0],
                                  cut_region[1]:-cut_region[1]]
        
        mse = mean_square_error(ref_array,query_array)
        
        if mse < min_mse:
            min_mse=mse
            Z_plane=i
            single_shift=shift
            aligned_array_Z_plane=aligned_array
        
    Aligned_TYX_array=np.array([ZYX_Stack_array[Z_plane,:,:],aligned_array_Z_plane])
    return Z_plane,single_shift, Aligned_TYX_array

#%%

if __name__ == "__main__":
    flim_1 = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250610\lowmag_002.flim"
    flim_2 = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250610\lowmag_001.flim"
    ch_1or2 = 2

    for i in range(1,4):
        flim_1 = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250610\lowmag_001.flim"
        flim_2 = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250610\lowmag_00" + str(i) + ".flim"
        ch_1or2 = 2
        xyz_relative, Aligned_4d_array, shifts_zyx_pixel = align_two_flimfile_different_resolution(flim_1, flim_2, ch_1or2 - 1, return_pixel = True, debug = True)
        print(xyz_relative)


# %%
