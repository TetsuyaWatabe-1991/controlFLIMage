# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:56:34 2022

@author: yasudalab
"""
import os,glob,path,math
from FLIMageFileReader2 import FileReader
import matplotlib.pyplot as plt
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from datetime import datetime

def get_flimfile_list(one_file_path):
    filelist=glob.glob(one_file_path[:-8]+'*.flim')
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

def fft_drift_3d(ref_array,query_array):
    # shift, error, diffphase = register_translation(ref_array, query_array)
    shift, error, diffphase = phase_cross_correlation(ref_array, query_array)
    img_corr = fourier_shift(np.fft.fftn(query_array), shift)
    aligned_array = np.fft.ifftn(img_corr).real
    return aligned_array, shift


def Align_3d_array(Tiff_MultiArray):
    shifts = []
    Aligned_4d = []
    
    for NthTime in range(Tiff_MultiArray.shape[0]):    
        aligned_array, shift = fft_drift_3d(Tiff_MultiArray[0],Tiff_MultiArray[NthTime])
        shifts.append(shift)
        Aligned_4d.append(aligned_array)
    # for plotting the xy shifts over time
    shifts = np.array(shifts)
    Aligned_3d_array = np.array(Aligned_4d)
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


def flim_files_to_nparray(filelist,ch=0):
    FourDimList=[]
    timestamp_list,relative_sec_list = [],[]
    First=True
    for file_path in filelist:
        iminfo = FileReader()
        print(file_path)
        iminfo.read_imageFile(file_path, True) 
        # Get intensity only data
        imagearray=np.array(iminfo.image)
        if First:
            First=False
            imageshape=imagearray.shape
        
        if imagearray.shape == imageshape:
            intensityarray=np.sum(imagearray,axis=-1)
            FourDimList.append(intensityarray)
        else:
            print(file_path,'<- skipped read')
            
        timestamp_list.append(datetime.strptime(iminfo.acqTime[0],'%Y-%m-%dT%H:%M:%S.%f'))
        relative_sec_list.append((timestamp_list[-1] - timestamp_list[0]).seconds) 
        
    RawArray=np.array(FourDimList,dtype=np.uint16)
    Tiff_MultiArray=RawArray[:,:,0,ch,:,:]
    return Tiff_MultiArray, iminfo, relative_sec_list


def get_xyz_pixel_um(iminfo):
    field_len = iminfo.State.Acq.FOV_default[0]
    zoom = iminfo.State.Acq.zoom
    pixels = iminfo.State.Acq.pixelsPerLine
    x_um = field_len/zoom/pixels
    y_um = field_len/zoom/pixels
    z_um = iminfo.State.Acq.sliceStep 
    return x_um, y_um, z_um

def make_save_folders(one_file_path):
    saveFolder=os.path.join(path.Path(one_file_path).parent,"Analysis",path.Path(one_file_path).stem)
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
   
    
def single_plane_align_with3dstack_flimfile(StackFilePath,SinglePlaneFilePath,ch=1):
    # StackFilePath=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221220\GFPneuron1_20_019.flim"
    # SinglePlaneFilePath=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221220\GFPneuron1_20_022.flim"  
    print('\n\n ---- \n',StackFilePath,'\n\n',SinglePlaneFilePath)
    Stack_file_list=[StackFilePath]
    SinglePlaneList=[SinglePlaneFilePath]
    ZYX_Stack_array, _ , _ = flim_files_to_nparray(Stack_file_list,ch=ch)
    YX_SinglePlane_array, _ , _ = flim_files_to_nparray(SinglePlaneList,ch=ch)
    Z_plane,single_shift, Aligned_TXY_array = single_plane_align_with3dstack(ZYX_Stack_array[0,:,:,:],YX_SinglePlane_array[0,0,:,:])

    return Z_plane,single_shift, Aligned_TXY_array
    

def mean_square_error(img1,img2,intensity_threshold=20,required_pixel=500):
     if math.prod(img1.shape)!=math.prod(img2.shape):        
         raise TypeError("Given images have different shapes")
         
     Mask = (img1>intensity_threshold)|(img2>intensity_threshold)
     pixel_size = Mask.sum()
     
     if pixel_size<required_pixel:
         return 9999999999
     
     diff= (img1.astype(float)-img2.astype(float))
     err = (Mask*(diff**2)).sum()
     mse = err/(float(pixel_size))
     
     return mse   
   
    
def single_plane_align_with3dstack(ZYX_Stack_array,YX_SinglePlane_array):    
    min_mse= 10000000000
    Z_plane= 10000000000
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

   
    
   
def sample():
    ch=0
    cmap='gray'
    vmin=0
    vmax_auto=True
    vmax_coefficient=0.8
    
    one_file_path=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221208\Test294_043.flim"
    saveFolder, EachImgsaveFolder = make_save_folders(one_file_path)
    
    iminfo = FileReader()
    iminfo.read_imageFile(one_file_path, True) 
    
    filelist=get_flimfile_list(one_file_path)
    Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray(filelist,ch=ch)
    
    ShowZ=int(Tiff_MultiArray.shape[1]/2)
    if vmax_auto==True:
        vmax=Tiff_MultiArray.max()*vmax_coefficient    
    
    shifts, Aligned_4d_array=Align_4d_array(Tiff_MultiArray)
    plot_alignment_shifts(shifts,iminfo, saveFolder=saveFolder,savefigure=True,
                          relative_sec_list=relative_sec_list)
    
    """
    
    # below code is for plotting aligned image        
    
    """
    
    
    for i in range(Aligned_4d_array.shape[0]):
        plt.figure()
    
        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(1,2) 
        
        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        axarr[0].imshow(Tiff_MultiArray[i,ShowZ,:,:],cmap=cmap, vmin=vmin, vmax=vmax)
        axarr[0].text(Tiff_MultiArray.shape[3],-10,str(i))
        axarr[0].text(0,-10,str("original"))
        
        axarr[1].imshow(Aligned_4d_array[i,ShowZ,:,:],cmap=cmap, vmin=vmin, vmax=vmax)
        axarr[1].text(0,-10,str("aligned"))
        
        # plt.imshow(Tiff_MultiArray[i,ShowZ,:,:],cmap=cmap, vmin=vmin, vmax=vmax)
        savepath=os.path.join(EachImgsaveFolder,f"{str(i).zfill(3)}.png")
        plt.savefig(savepath,dpi=150,transparent=True,bbox_inches='tight')
        plt.show()
    
    
    # Print the first position of the maximum value in the array
    # max_index_original=np.unravel_index(Tiff_MultiArray.argmax(), Tiff_MultiArray.shape)
    # From this code above, the brightest point might exist near edge. 
    # To avoid that,  I made codes below
    
    
    z_start, z_to, y_start, y_to, x_start, x_to=CenterPosGet(Tiff_MultiArray,ratio=0.5)
    
    FindBrightBase=Tiff_MultiArray[0,z_start:z_to,y_start:y_to,x_start:x_to]
    max_index_original=np.unravel_index(FindBrightBase.argmax(), FindBrightBase.shape)
    
    Slice_x=max_index_original[1]+y_start
    Slice_y=max_index_original[2]+x_start
    ShowZ=max_index_original[0]+z_start
    # Slice_y=60
    # Slice_x=155
    # Slice_x=64
    # Slice_y=40
    
    plt.imshow(Tiff_MultiArray[0,ShowZ,:,:],cmap=cmap, vmin=vmin, vmax=vmax)
    plt.plot([0,Tiff_MultiArray.shape[3]-1],[Slice_x,Slice_x],'r-')
    savepath=os.path.join(saveFolder,"XZimage_showingSliceX.png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
    
    plt.imshow(Tiff_MultiArray[:,ShowZ,Slice_x,:],cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("x");plt.ylabel('t')
    savepath=os.path.join(saveFolder,"XZimage_original.png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
    plt.imshow(Aligned_4d_array[:,ShowZ,Slice_x,:],cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("x");plt.ylabel('t')
    savepath=os.path.join(saveFolder,"XZimage_aligned.png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
    
    
    
    plt.imshow(Tiff_MultiArray[0,ShowZ,:,:],cmap=cmap, vmin=vmin, vmax=vmax)
    plt.plot([Slice_y,Slice_y],[0,Tiff_MultiArray.shape[2]-1],'r-')
    savepath=os.path.join(saveFolder,"YZimage_showingSliceX.png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
    
    plt.imshow(Tiff_MultiArray[:,ShowZ,:,Slice_y],cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("y");plt.ylabel('t')
    savepath=os.path.join(saveFolder,"YZimage_original.png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
    plt.imshow(Aligned_4d_array[:,ShowZ,:,Slice_y],cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("y");plt.ylabel('t')
    savepath=os.path.join(saveFolder,"YZimage_aligned.png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
    
    plt.imshow(Tiff_MultiArray[0,ShowZ,:,:],cmap=cmap, vmin=vmin, vmax=vmax)
    plt.plot([Slice_y+1,Slice_y-1,Slice_y-1,Slice_y+1,Slice_y+1],         
             [Slice_x+1,Slice_x+1,Slice_x-1,Slice_x-1,Slice_x+1],'r-')
    savepath=os.path.join(saveFolder,"TZimage_showingPoint.png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
    
    plt.imshow(Tiff_MultiArray[:,:,Slice_x,Slice_y],
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("z");plt.ylabel('t')
    savepath=os.path.join(saveFolder,"TZimage_original.png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
    
    
    plt.imshow(Aligned_4d_array[:,:,Slice_x,Slice_y],
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("z");plt.ylabel('t')
    savepath=os.path.join(saveFolder,"TZimage_aligned.png")
    plt.savefig(savepath,dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
    
    
    
    
