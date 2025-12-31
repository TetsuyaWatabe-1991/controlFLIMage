# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 08:56:14 2022

@author: yasudalab
"""
import math
import os
import copy
from time import sleep
import configparser
import PySimpleGUI as sg
import numpy as np
from io import BytesIO
from PIL import Image
from tifffile import imread
import cv2
import base64
import glob
import re
from FLIMageAlignment import flim_files_to_nparray
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from FLIMageFileReader2 import FileReader


def read_multiple_uncagingpos(flimpath):
    
    txtpath = flimpath[:-5]+".txt"
    z = -99
    ylist, xlist = [], []
    with open(txtpath, 'r') as f:
        num_pos = int(f.readline())
        for nth_pos in range(num_pos):
            zyx = (f.readline()).split(",")
            z = int(zyx[0])
            ylist.append(float(zyx[1]))
            xlist.append(float(zyx[2]))
    if len(ylist)<1 or z ==-99:
        raise Exception(f"{txtpath} do not have any uncaging position")
    else:
        return z, ylist, xlist

def read_dendriteinfo(flimpath):
    txtpath = flimpath[:-5]+"dendrite.txt"    
    direction_list, orientation_list, dendylist, dendxlist = [], [], [], []
    with open(txtpath, 'r') as f:
        num_pos = int(f.readline())
        for nth_pos in range(num_pos):
            dir_ori_y_x = (f.readline()).split(",")
            #{direction},{orientation},{y_moved},{x_moved
            direction_list.append(int(dir_ori_y_x[0]))
            orientation_list.append(float(dir_ori_y_x[1]))
            dendylist.append(float(dir_ori_y_x[2]))
            dendxlist.append(float(dir_ori_y_x[3]))
    if len(direction_list) < 1:
        raise Exception(f"{txtpath} do not have any uncaging position")
    else:
        return direction_list, orientation_list, dendylist, dendxlist 
    
    
def dend_props_forEach(flimpath, ch1or2=1,
                       square_side_half_len = 20,
                       threshold_coordinate = 0.1, Gaussian_pixel = 3,
                       plot_img = False):
    ch = ch1or2 - 1
    Tiff_MultiArray, _, _ = flim_files_to_nparray([flimpath],ch=ch)
    ZYXarray = Tiff_MultiArray[0]
    

    z, ylist, xlist = read_multiple_uncagingpos(flimpath)
    
    maxproj = np.max(ZYXarray[z-1:z+2,:,:], axis = 0)
    ylist = list(map(int,ylist))
    xlist = list(map(int,xlist))
    txtpath = flimpath[:-5]+"dendrite.txt"
    with open(txtpath, 'w') as f:
        f.write(str(len(ylist))+'\n')
        for y, x in zip(ylist, xlist):
            maxproj_aroundYX = maxproj[y - square_side_half_len : y + square_side_half_len + 1,
                                       x - square_side_half_len : x + square_side_half_len + 1]
            blur = cv2.GaussianBlur(maxproj_aroundYX,(Gaussian_pixel,Gaussian_pixel),0)
            Threshold = blur[square_side_half_len,square_side_half_len]*threshold_coordinate
            print(f"x, y, Threshold - {x},{y},{Threshold}")
            ret3,th3 = cv2.threshold(blur,Threshold,255,cv2.THRESH_BINARY)
            label_img = label(th3)
            
            spineregion = -1
            for each_label in range(1,label_img.max()+1):
                if label_img[square_side_half_len, square_side_half_len] == each_label:
                    spineregion = each_label
                    
            if spineregion < 1:
                print("\n\n ERROR 102,  Cannot find dendrite \n No update in uncaging position. \n")
                # return None
            
            else:
                spinebinimg = np.zeros(label_img.shape)
                spinebinimg[label_img == spineregion] = 1
                dendprops = regionprops(label(spinebinimg))[0]
    
            orientation = round(dendprops.orientation,3)
            y0,x0 = dendprops.centroid 
            x_moved = round(x0 - square_side_half_len,1)
            y_moved = round(y0 - square_side_half_len,1)
            x_rotated = x_moved*math.cos(orientation) - y_moved*math.sin(orientation)
            if x_rotated<=0:
                direction = 1
            else:
                direction = -1
            f.write(f'{direction},{orientation},{y_moved},{x_moved}\n')
            
            if plot_img == True:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(blur, cmap='gray')
                ax2.imshow(spinebinimg, cmap='gray')
                ax2.scatter([x0],[y0],c="r")
                plt.show()
    
    if plot_img == True:            
        direction_list, orientation_list, dendylist, dendxlist  = read_dendriteinfo(flimpath)
        dend_center_x = np.array(dendxlist) + np.array(xlist)
        dend_center_y = np.array(dendylist) + np.array(ylist)
        
        plt.imshow(maxproj,cmap="gray")
        plt.scatter(xlist,ylist,c='cyan',s=10)
        plt.scatter(dend_center_x,dend_center_y,c='r',s=10)
        
        for y0, x0, orientation in zip(dend_center_y, dend_center_x, orientation_list):
            x0,x1,x2,x1_1,x2_1,y0,y1,y2,y1_1,y2_1 = get_axis_position(y0,x0,orientation, HalfLen_c=10)
            plt.plot((x2_1, x2), (y2_1, y2), '-b', linewidth=1.5)
        
        plt.axis('off')
        plt.title("Uncaging position")
            
        plt.savefig(flimpath[:-5]+".png", dpi=150, bbox_inches='tight')
        
        os.makedirs(flimpath[:-8],exist_ok=True)
        savepath = os.path.join(flimpath[:-8],"uncagingpos.png")
        plt.savefig(savepath, dpi = 72, bbox_inches = 'tight')
        
        plt.show()
                            
    print(f"Dendrite information was saved as {txtpath}")

def PILimg_to_data(im):
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data    


def OpenPNG(pngpath):
    try:
        with open(pngpath, "rb") as image_file:
            data = base64.b64encode(image_file.read())
    except:
        print("Could not open - ",pngpath)
        data = None
    return data

def tiff_size_read(tiffpath):
    return imread(tiffpath).shape

def first_tiff_read(tiffpath):
    stack_array = np.array(imread(tiffpath))

    if len(stack_array.shape)>3:
         raise TypeError("Only 2D or 3D tiff is allowed")
    else:
        return stack_array.shape

def tiff16bit_to_PIL(tiffpath,Percent=100,show_size_xy=[512,512],
                     return_ratio=False,NthSlice=1):
    # This can be used for two D image also.
    stack_array = np.array(imread(tiffpath))
    if return_ratio==True:    
        im_PIL,resize_ratio_yx  = tiffarray_to_PIL(stack_array,Percent=Percent,show_size_xy=show_size_xy,
                                                   return_ratio=return_ratio,NthSlice=NthSlice)
        return im_PIL,resize_ratio_yx
    else:
        im_PIL = tiffarray_to_PIL(stack_array,Percent=Percent,show_size_xy=show_size_xy,
                                                   return_ratio=return_ratio,NthSlice=NthSlice)
        return im_PIL 

def tiffarray_to_PIL(stack_array,Percent=100,show_size_xy=[512,512],
                     return_ratio=False,NthSlice=1):
    # This can be used for two D image also.
    
    if Percent<1 or Percent>100:
        Percent=100
        
    if len(stack_array.shape)== 3:
        im_array = stack_array[NthSlice-1,:,:]
    else:
        im_array = stack_array
        
    norm_array = (100/Percent)*255*(im_array/stack_array.max())
    # print(norm_array)
    norm_array[norm_array>255]=255
    rgb_array = cv2.cvtColor(norm_array.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    im_PIL = Image.fromarray(rgb_array)
    
    im_PIL = im_PIL.resize(show_size_xy)
    resize_ratio_yx = (show_size_xy[1]/im_array.shape[0],show_size_xy[0]/im_array.shape[1])
    if return_ratio==True:    
        return im_PIL,resize_ratio_yx
    else:
        return im_PIL




def tiffarray_to_PIL_8bit(stack_array, vmax = 255, show_size_yx=[512,512],
                          return_ratio=False,NthSlice=1):
    # This can be used for two D image also.
    
    if vmax<1 or vmax>255:
        vmax=100
        
    if len(stack_array.shape)== 3:
        im_array = stack_array[NthSlice-1,:,:]
    else:
        im_array = stack_array
        
    norm_array = (255/vmax)*(im_array)
    # print(norm_array)
    norm_array[norm_array>255]=255
    rgb_array = cv2.cvtColor(norm_array.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    im_PIL = Image.fromarray(rgb_array)
    
    im_PIL = im_PIL.resize(show_size_yx[::-1])
    resize_ratio_yx = (show_size_yx[0]/im_array.shape[0],show_size_yx[1]/im_array.shape[1])
    if return_ratio==True:    
        return im_PIL,resize_ratio_yx
    else:
        return im_PIL

def calc_zoom_rate_based_on_maxsize(yx_shape: tuple, 
                                    show_size_YX_max: list, 
                                    show_size_YX_min: list) -> tuple:
    candidate_yxshape = np.array(yx_shape)
    for axis_ind in range(len(candidate_yxshape)):
        if candidate_yxshape[axis_ind] > show_size_YX_max[axis_ind]:
            candidate_yxshape = (candidate_yxshape / candidate_yxshape[axis_ind]) * show_size_YX_max[axis_ind]

    for axis_ind in range(len(candidate_yxshape)):
        if candidate_yxshape[axis_ind] < show_size_YX_min[axis_ind]:
            candidate_yxshape = (candidate_yxshape / candidate_yxshape[axis_ind]) * show_size_YX_min[axis_ind]

    resize_yx_shape = list(candidate_yxshape.astype(np.uint16))
    return resize_yx_shape



def z_stack_multi_z_click(stack_array, pre_assigned_pix_zyx_list=[], show_text = ""):
    first_text_at_upper = "Click each position and click assign"
    col_dict = {
        "clicked_currentZ": "white",
        "clicked_differentZ": "cyan",
        "clicked_now": "magenta",
        "pos_now": "red",
        }
    font = ("Courier New", 15)
    show_size_YX_max = [768, 1024]
    show_size_YX_min = [512, 512]
    
    TiffShape= stack_array.shape
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        Z_change=[sg.Text("Z position", size=(20, 1)),
                  sg.Slider(orientation ='horizontal', key='Z',
                            default_value=int(NumOfZ/2), range=(1,NumOfZ),enable_events = True)            
                  ]
        yx_shape = TiffShape[1:3]
        resize_yx_shape = calc_zoom_rate_based_on_maxsize(yx_shape, show_size_YX_max,show_size_YX_min)
        im_PIL,resize_ratio_yx = tiffarray_to_PIL_8bit(stack_array,vmax = 255, show_size_yx = resize_yx_shape,
                                                  return_ratio=True,NthSlice=int(NumOfZ/2))
    else:
        NumOfZ = 1
        Z_change=[]
        resize_yx_shape = calc_zoom_rate_based_on_maxsize(TiffShape, show_size_YX_max,show_size_YX_min)
        im_PIL,resize_ratio_yx = tiffarray_to_PIL_8bit(stack_array, vmax = 255,  show_size_yx = resize_yx_shape,
                                                  return_ratio=True,NthSlice=1)
    
    print("resize_ratio_yx",resize_ratio_yx)
    
    ZYX_pixel_clicked_list = []
    if type(pre_assigned_pix_zyx_list)==list:
        print(pre_assigned_pix_zyx_list)
        for each_zyx in pre_assigned_pix_zyx_list:
            each_z_resized = each_zyx[0]
            each_y_resized = resize_yx_shape[0] - each_zyx[1]*resize_ratio_yx[0]
            each_x_resized = each_zyx[2]*resize_ratio_yx[1]
            ZYX_pixel_clicked_list.append([each_z_resized,
                                           each_y_resized,
                                           each_x_resized])
    print("ZYX_pixel_clicked_list ",ZYX_pixel_clicked_list)
    data =  PILimg_to_data(im_PIL)    
   
    sg.theme('Dark Blue 3')

    layout = [
                [
                    sg.Text(show_text, font='Arial 8', size=(90, 1))
                    ],
                [
                    sg.Text(first_text_at_upper, key='notification', font='Arial 10', 
                            text_color='black', background_color='white', size=(60, 2))
                    ],
                [
                    sg.Graph(
                    canvas_size=resize_yx_shape[::-1], 
                    graph_bottom_left=(0, 0),
                    graph_top_right=resize_yx_shape[::-1],
                    key="-GRAPH-",
                    enable_events=True,background_color='lightblue',
                    drag_submits=True,motion_events=True,
                    )
                    ],
                [
                    sg.Text("Intensity", size=(20, 1)),
                    sg.Slider(orientation ='horizontal', key='Intensity',default_value=stack_array.max(),
                         range=(1,255),enable_events = True),
                    ],
                Z_change,
                [
                    sg.Text("Assigned pos", size=(20, 1)),
                    sg.Slider(orientation ='horizontal', key='pos',default_value=0,
                         range=(0,20),enable_events = True),
                    ],

                [
                    sg.Text(key='-INFO-', size=(30, 1)),
                    sg.Button('Assign', size=(20, 2)),
                    sg.Button('Reset', size=(20, 2)),
                    sg.Button('OK', size=(20, 2))
                    ]
            ]
    
    img_paste_loc = [0,resize_yx_shape[0]]
    window = sg.Window("Z stack viewer", layout, finalize=True)
    graph = window["-GRAPH-"]  # type: sg.Graph
    graph.draw_image(data=data, location = img_paste_loc)

    x=-1
    y=-1
    NthSlice = 1

    First = True
    while True:
        if First:
            First = False
            event, values = window.read(timeout=1)
            event = "-GRAPH-"
        else:
            event, values = window.read()
        ShowIntensityMax = values['Intensity']

        if len(TiffShape)==3:
            NthSlice = int(values['Z'])

        if event == sg.WIN_CLOSED:
            break
        
        if event == "pos":
            if values['pos']<1:
                continue
            if len(ZYX_pixel_clicked_list)>=values['pos']:
                nthpos = int(values['pos']-1)
                Z = ZYX_pixel_clicked_list[nthpos][0] + 1
                window['Z'].update(Z)
                NthSlice = Z
                window['notification'].update(f"Go to pos id: {int(values['pos'])}")
            else:
                window['notification'].update(f"pos id: {int(values['pos'])} is not defined.")
            
        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
                
        if event ==  'Assign':
            z,y,x = NthSlice - 1, y, x
            ZYX_pixel_clicked_list.append([z, y, x])
            x=-1; y=-1
            
        if event == "-GRAPH-" or "pos" or 'Update' or 'Z' or "Intensity" or 'Assign':
            im_PIL = tiffarray_to_PIL_8bit(stack_array,
                                           vmax = ShowIntensityMax,  
                                           show_size_yx = resize_yx_shape,
                                           return_ratio = False, 
                                           NthSlice = NthSlice)
            data = PILimg_to_data(im_PIL)
            graph.draw_image(data=data, location=img_paste_loc)
            graph.draw_point((x,y), size=5, color = col_dict["clicked_now"])
                
            halfsize = min(resize_yx_shape)//8
            for nth, EachZYX in enumerate(ZYX_pixel_clicked_list):
                if EachZYX[0] == NthSlice -1:
                    color = col_dict["clicked_currentZ"]
                    if nth == int(values['pos']-1):
                        color = col_dict["pos_now"]
                    graph.DrawRectangle(
                        (EachZYX[2]-halfsize, EachZYX[1]-halfsize), 
                        (EachZYX[2]+halfsize, EachZYX[1]+halfsize), 
                        line_color=color)
                    text_x = max(EachZYX[2] - halfsize*0.90, resize_yx_shape[1]*0.007)
                    text_y = min(EachZYX[1] + halfsize*0.80, resize_yx_shape[0]*0.990)
                    graph.DrawText(str(nth+1), (text_x, text_y),
                                    font=font, color = color)
                    
                else:
                    color = col_dict["clicked_differentZ"]
                    graph.DrawText(str(nth+1), (EachZYX[2],EachZYX[1]),
                                    font=font, color = color)

        if event ==  'Reset':
            ZYX_pixel_clicked_list = []
            graph.draw_image(data=data, location=img_paste_loc)
            
        if event ==  'OK':
            window.close()
            pix_zyx_list = []
            for each_zyx in ZYX_pixel_clicked_list:
                each_z_pix = each_zyx[0]
                each_y_pix = round((resize_yx_shape[0]-each_zyx[1])/resize_ratio_yx[0])
                each_x_pix = round(each_zyx[2]/resize_ratio_yx[1])
                pix_zyx_list.append([each_z_pix,
                                     each_y_pix,
                                     each_x_pix])
            break
        
    return pix_zyx_list

def z_stack_multi_z_click_with_delete(stack_array, pre_assigned_pix_zyx_list=[], show_text = ""):
    first_text_at_upper = "Click each position and click assign"
    col_dict = {
        "clicked_currentZ": "white",
        "clicked_differentZ": "cyan",
        "clicked_now": "magenta",
        "pos_now": "red",
        }
    font = ("Courier New", 15)
    show_size_YX_max = [768, 1024]
    show_size_YX_min = [512, 512]
    
    TiffShape= stack_array.shape
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        Z_change=[sg.Text("Z position", size=(20, 1)),
                  sg.Slider(orientation ='horizontal', key='Z',
                            default_value=int(NumOfZ/2), range=(1,NumOfZ),enable_events = True)            
                  ]
        yx_shape = TiffShape[1:3]
        resize_yx_shape = calc_zoom_rate_based_on_maxsize(yx_shape, show_size_YX_max,show_size_YX_min)
        im_PIL,resize_ratio_yx = tiffarray_to_PIL_8bit(stack_array,vmax = 255, show_size_yx = resize_yx_shape,
                                                  return_ratio=True,NthSlice=int(NumOfZ/2))
    else:
        NumOfZ = 1
        Z_change=[]
        resize_yx_shape = calc_zoom_rate_based_on_maxsize(TiffShape, show_size_YX_max,show_size_YX_min)
        im_PIL,resize_ratio_yx = tiffarray_to_PIL_8bit(stack_array, vmax = 255,  show_size_yx = resize_yx_shape,
                                                  return_ratio=True,NthSlice=1)
    
    print("resize_ratio_yx",resize_ratio_yx)
    
    ZYX_pixel_clicked_list = []
    if type(pre_assigned_pix_zyx_list)==list:
        print(pre_assigned_pix_zyx_list)
        for each_zyx in pre_assigned_pix_zyx_list:
            each_z_resized = each_zyx[0]
            each_y_resized = resize_yx_shape[0] - each_zyx[1]*resize_ratio_yx[0]
            each_x_resized = each_zyx[2]*resize_ratio_yx[1]
            ZYX_pixel_clicked_list.append([each_z_resized,
                                           each_y_resized,
                                           each_x_resized])
    print("ZYX_pixel_clicked_list ",ZYX_pixel_clicked_list)
    data =  PILimg_to_data(im_PIL)    
   
    sg.theme('Dark Blue 3')

    layout = [
                [
                    sg.Text(show_text, font='Arial 8', size=(90, 1))
                    ],
                [
                    sg.Text(first_text_at_upper, key='notification', font='Arial 10', 
                            text_color='black', background_color='white', size=(60, 2))
                    ],
                [
                    sg.Graph(
                    canvas_size=resize_yx_shape[::-1], 
                    graph_bottom_left=(0, 0),
                    graph_top_right=resize_yx_shape[::-1],
                    key="-GRAPH-",
                    enable_events=True,background_color='lightblue',
                    drag_submits=True,motion_events=True,
                    )
                    ],
                [
                    sg.Text("Intensity", size=(20, 1)),
                    sg.Slider(orientation ='horizontal', key='Intensity',default_value=stack_array.max(),
                         range=(1,255),enable_events = True),
                    ],
                Z_change,
                [
                    sg.Text("Assigned pos", size=(20, 1)),
                    sg.Slider(orientation ='horizontal', key='pos',default_value=0,
                         range=(0,20),enable_events = True),
                    ],

                [
                    sg.Text(key='-INFO-', size=(30, 1)),
                    sg.Button('Assign', size=(20, 2)),
                    sg.Button('Delete', size=(20, 2)),
                    sg.Button('Reset', size=(20, 2)),
                    sg.Button('OK', size=(20, 2))
                    ]
            ]
    
    img_paste_loc = [0,resize_yx_shape[0]]
    window = sg.Window("Z stack viewer with delete", layout, finalize=True)
    graph = window["-GRAPH-"]  # type: sg.Graph
    graph.draw_image(data=data, location = img_paste_loc)

    x=-1
    y=-1
    NthSlice = 1

    First = True
    while True:
        if First:
            First = False
            event, values = window.read(timeout=1)
            event = "-GRAPH-"
        else:
            event, values = window.read()
        ShowIntensityMax = values['Intensity']

        if len(TiffShape)==3:
            NthSlice = int(values['Z'])

        if event == sg.WIN_CLOSED:
            break
        
        if event == "pos":
            if values['pos']<1:
                continue
            if len(ZYX_pixel_clicked_list)>=values['pos']:
                nthpos = int(values['pos']-1)
                Z = ZYX_pixel_clicked_list[nthpos][0] + 1
                window['Z'].update(Z)
                NthSlice = Z
                window['notification'].update(f"Go to pos id: {int(values['pos'])}")
            else:
                window['notification'].update(f"pos id: {int(values['pos'])} is not defined.")
            
        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
                
        if event ==  'Assign':
            z,y,x = NthSlice - 1, y, x
            ZYX_pixel_clicked_list.append([z, y, x])
            x=-1; y=-1
            
        if event == 'Delete':
            if values['pos'] >= 1 and len(ZYX_pixel_clicked_list) >= values['pos']:
                nthpos = int(values['pos'] - 1)
                deleted_pos = ZYX_pixel_clicked_list.pop(nthpos)
                window['notification'].update(f"Deleted position {int(values['pos'])} at Z={deleted_pos[0]+1}, Y={deleted_pos[1]}, X={deleted_pos[2]}")
                # Reset position slider if it's now out of range
                if values['pos'] > len(ZYX_pixel_clicked_list):
                    window['pos'].update(0)
            else:
                window['notification'].update("No position selected to delete")
            
        if event == "-GRAPH-" or "pos" or 'Update' or 'Z' or "Intensity" or 'Assign' or 'Delete':
            im_PIL = tiffarray_to_PIL_8bit(stack_array,
                                           vmax = ShowIntensityMax,  
                                           show_size_yx = resize_yx_shape,
                                           return_ratio = False, 
                                           NthSlice = NthSlice)
            data = PILimg_to_data(im_PIL)
            graph.draw_image(data=data, location=img_paste_loc)
            graph.draw_point((x,y), size=5, color = col_dict["clicked_now"])
                
            halfsize = min(resize_yx_shape)//8
            for nth, EachZYX in enumerate(ZYX_pixel_clicked_list):
                if EachZYX[0] == NthSlice -1:
                    color = col_dict["clicked_currentZ"]
                    if nth == int(values['pos']-1):
                        color = col_dict["pos_now"]
                    graph.DrawRectangle(
                        (EachZYX[2]-halfsize, EachZYX[1]-halfsize), 
                        (EachZYX[2]+halfsize, EachZYX[1]+halfsize), 
                        line_color=color)
                    text_x = max(EachZYX[2] - halfsize*0.90, resize_yx_shape[1]*0.007)
                    text_y = min(EachZYX[1] + halfsize*0.80, resize_yx_shape[0]*0.990)
                    graph.DrawText(str(nth+1), (text_x, text_y),
                                    font=font, color = color)
                    
                else:
                    color = col_dict["clicked_differentZ"]
                    graph.DrawText(str(nth+1), (EachZYX[2],EachZYX[1]),
                                    font=font, color = color)

        if event ==  'Reset':
            ZYX_pixel_clicked_list = []
            graph.draw_image(data=data, location=img_paste_loc)
            
        if event ==  'OK':
            window.close()
            pix_zyx_list = []
            for each_zyx in ZYX_pixel_clicked_list:
                each_z_pix = each_zyx[0]
                each_y_pix = round((resize_yx_shape[0]-each_zyx[1])/resize_ratio_yx[0])
                each_x_pix = round(each_zyx[2]/resize_ratio_yx[1])
                pix_zyx_list.append([each_z_pix,
                                     each_y_pix,
                                     each_x_pix])
            break
        
    return pix_zyx_list

def threeD_array_click(stack_array,Text="Click",SampleImg=None,
                       ShowPoint=False,ShowPoint_YX=[0,0],
                       predefined = False, predefined_ZYX = [0,0,0],
                       existing_spines_list=None, maxproj=None):
    TiffShape= stack_array.shape
    col_list=['red','cyan']
    
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        if predefined == True:    
            default_Z = predefined_ZYX[0] + 1
        else:
            default_Z = int(NumOfZ/2 + 1)
        print("default_Z",default_Z)
        Z_change=[sg.Text("Z position", size=(20, 1)),
                  sg.Slider(orientation ='horizontal', key='Z',
                            default_value=default_Z, range=(1,NumOfZ),enable_events = True)
                  ]
        im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array,Percent=100, show_size_xy=[512,512],
                                                  return_ratio=True, NthSlice = default_Z)
    else:
        NumOfZ = 1
        Z_change=[]
        im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array,Percent=100, show_size_xy=[512,512],
                                                  return_ratio=True,NthSlice=1)

    data =  PILimg_to_data(im_PIL)
    
    # Prepare right panel with max proj and existing spines
    data_maxproj_right = None
    resize_ratio_maxproj_right = None
    right_panel_graph = None
    data_sample = None
    
    # Calculate max projection if not provided
    if maxproj is None and len(TiffShape) == 3:
        # Calculate max projection from stack_array
        intensityarray_temp = np.sum(np.sum(np.sum(stack_array, axis=-1), axis=1), axis=1)
        maxproj = np.sum(intensityarray_temp, axis=0)
    
    if maxproj is not None:
        im_PIL_maxproj_right, resize_ratio_maxproj_right = tiffarray_to_PIL(
            stack_array=maxproj, Percent=100, show_size_xy=[512, 512], return_ratio=True)
        data_maxproj_right = PILimg_to_data(im_PIL_maxproj_right)
        right_panel_graph = sg.Graph(
            canvas_size=(512, 512), graph_bottom_left=(0, 0),
            graph_top_right=(512, 512),
            key="-MAXPROJ-",
            enable_events=False, background_color='black',
        )
    
    # Fallback to Sample image if maxproj is not available
    if right_panel_graph is None:
        data_sample = OpenPNG(pngpath=SampleImg)
        right_panel_graph = sg.Graph(
            canvas_size=(512, 512), graph_bottom_left=(0, 0),
            graph_top_right=(512, 512),
            key="-Sample-",
            enable_events=True, background_color='black',
            drag_submits=True, motion_events=True,
        )

    sg.theme('Dark Blue 3')

    layout = [
                [sg.Text(Text, font='Arial 10', text_color='black', background_color='white', size=(80, 2))],
                [
                sg.Graph(
                canvas_size=(512, 512), graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-GRAPH-",
                enable_events=True,background_color='lightblue',
                drag_submits=True,motion_events=True,
                ),
                right_panel_graph
                ],
              [
               sg.Text("Contrast", size=(20, 1)),
               sg.Slider(orientation ='horizontal', key='Intensity',default_value=100,
                         range=(1,100),enable_events = True),
              ],
               Z_change
              ,
            [sg.Text(key='-INFO-', size=(60, 1)),sg.Button('OK', size=(20, 2)),sg.Button('Exclude', size=(20, 2))],
            ]
    
    window = sg.Window("Spine selection", layout, finalize=True)
    graph = window["-GRAPH-"]       # type: sg.Graph
    graph.draw_image(data=data, location=(0,512))
    
    # Draw right panel: max proj or sample image
    if data_maxproj_right is not None and "-MAXPROJ-" in window.AllKeysDict:
        graph_maxproj = window["-MAXPROJ-"]
        graph_maxproj.draw_image(data=data_maxproj_right, location=(0, 512))
        
        # Draw existing spines if provided
        if existing_spines_list is not None and len(existing_spines_list) > 0:
            for spine_info in existing_spines_list:
                if isinstance(spine_info, tuple) and len(spine_info) >= 3:
                    spine_zyx, dend_slope, dend_intercept = spine_info[0], spine_info[1], spine_info[2]
                    # Draw spine point
                    y_max = 512 - spine_zyx[1] * resize_ratio_maxproj_right[0]
                    x_max = spine_zyx[2] * resize_ratio_maxproj_right[1]
                    if 0 <= x_max < 512 and 0 <= y_max < 512:
                        graph_maxproj.draw_point((x_max, y_max), size=6, color='cyan')
                        
                        # Draw dendrite line
                        if maxproj is not None:
                            dend_x_list = []
                            dend_y_list = []
                            for x in range(0, maxproj.shape[1]):
                                y = dend_slope * x + dend_intercept
                                if 0 <= y < maxproj.shape[0]:
                                    dend_x_list.append(x)
                                    dend_y_list.append(y)
                            
                            if len(dend_x_list) > 1:
                                points = []
                                for x_pix, y_pix in zip(dend_x_list, dend_y_list):
                                    x_graph = x_pix * resize_ratio_maxproj_right[1]
                                    y_graph = 512 - y_pix * resize_ratio_maxproj_right[0]
                                    if 0 <= x_graph < 512 and 0 <= y_graph < 512:
                                        points.append((x_graph, y_graph))
                                if len(points) > 1:
                                    for i in range(len(points)-1):
                                        graph_maxproj.draw_line(points[i], points[i+1], color='cyan', width=1)
    elif data_sample is not None and "-Sample-" in window.AllKeysDict:
        window["-Sample-"].draw_image(data=data_sample, location=(0,512))
    
    if ShowPoint==True:
        col_list=['cyan','red']
        y = 512 - ShowPoint_YX[0]*resize_ratio_yx[0]
        x = ShowPoint_YX[1]*resize_ratio_yx[1]
        graph.draw_point((x,y), size=5, color = col_list[1])
    elif predefined == True:    
        y = 512 - predefined_ZYX[1]*resize_ratio_yx[0]
        x = predefined_ZYX[2]*resize_ratio_yx[1]
        graph.draw_point((x,y), size=5, color = col_list[0])
    else:
        x=-1;y=-1
    

    NthSlice = 1
    
    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']

        if len(TiffShape)==3:
            NthSlice = int(values['Z'])

        if event == sg.WIN_CLOSED:
            break

        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            im_PIL = tiffarray_to_PIL(stack_array,Percent=ShowIntensityMax,NthSlice=NthSlice)
            data =  PILimg_to_data(im_PIL)
            graph.erase()
            graph.draw_image(data=data, location=(0,512))
            graph.draw_point((x,y), size=5, color = col_list[0])

        if event ==  'Update' or 'Z' or "Intensity":
            im_PIL = tiffarray_to_PIL(stack_array,Percent=ShowIntensityMax,NthSlice=NthSlice)
            data =  PILimg_to_data(im_PIL)
            graph.draw_image(data=data, location=(0,512))
            graph.draw_point((x,y), size=5, color = col_list[0])

        if event ==  'OK':
            if x==-1:
                window["-INFO-"].update(value="Please click")
                continue
            else:
                z,y,x=NthSlice-1, y, x
                window.close()
                return z,int((512-y)/resize_ratio_yx[0]),int(x/resize_ratio_yx[1])
                break
            
        if event == "Exclude":
            window.close()
            return -1, -1, -1
            break
        

def multiple_uncaging_click(stack_array,Text="Click",SampleImg=None,ShowPoint=False,ShowPoint_YX=[0,0]):

    TiffShape= stack_array.shape
    col_list=['red','cyan']
    ShowPointsYXlist = []
    
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        Z_change=[sg.Text("Z position", size=(20, 1)),
                  sg.Slider(orientation ='horizontal', key='Z',
                            default_value=int(NumOfZ/2), range=(1,NumOfZ),enable_events = True)            
                  ]
        im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array,Percent=100, show_size_xy=[512,512],
                                                  return_ratio=True,NthSlice=int(NumOfZ/2))
    else:
        NumOfZ = 1
        Z_change=[]
        im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array,Percent=100, show_size_xy=[512,512],
                                                  return_ratio=True,NthSlice=1)

    data =  PILimg_to_data(im_PIL)    
    
    data_sample = OpenPNG(pngpath=SampleImg)

    sg.theme('Dark Blue 3')

    layout = [
                [sg.Text(Text, font='Arial 10', text_color='black', background_color='white', size=(60, 2))],
                [
                sg.Graph(
                canvas_size=(512, 512), graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-GRAPH-",
                enable_events=True,background_color='lightblue',
                drag_submits=True,motion_events=True,
                ),
                sg.Graph(
                canvas_size=(512, 512), graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-Sample-",
                enable_events=True,background_color='black',
                drag_submits=True,motion_events=True,
                )
                ],
              [
               sg.Text("Contrast", size=(20, 1)),
               sg.Slider(orientation ='horizontal', key='Intensity',default_value=100,
                         range=(1,100),enable_events = True),
              ],
               Z_change
              ,
            [sg.Text(key='-INFO-', size=(60, 1)),sg.Button('Assign', size=(20, 2)),sg.Button('Reset', size=(20, 2)),sg.Button('OK', size=(20, 2))]
            ]
    
    window = sg.Window("Spine selection", layout, finalize=True)
    graph = window["-GRAPH-"]       # type: sg.Graph
    graph.draw_image(data=data, location=(0,512))
    
    window["-Sample-"].draw_image(data=data_sample, location=(0,512))
   
    if len(ShowPointsYXlist)>0:
        col_list=['cyan','red']
        y_2 = 512 - ShowPoint_YX[0]*resize_ratio_yx[0]
        x_2 = ShowPoint_YX[1]*resize_ratio_yx[1]
        graph.draw_point((x_2,y_2), size=5, color = col_list[1]) 
   
    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']
        NthCh = int(values['Ch'])
        
        if NthCh == 1:
            tiffpath = transparent_tiffpath
        else:
            tiffpath = fluorescent_tiffpath
        
        if event == sg.WIN_CLOSED:
            break

        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            im_PIL = tiff16bit_to_PIL(tiffpath,Percent=ShowIntensityMax,show_size_xy=show_size_xy)
            data =  PILimg_to_data(im_PIL)
            graph.erase()
            graph.draw_image(data=data, location=(0,show_size_xy[1]))
            graph.draw_point((x,y), size=5, color = col_list[0])

        if event ==  'Update' or "Intensity" or "Ch":
            im_PIL = tiff16bit_to_PIL(tiffpath,Percent=ShowIntensityMax,show_size_xy=show_size_xy)
            data =  PILimg_to_data(im_PIL)
            graph.draw_image(data=data, location=(0,show_size_xy[1]))
            graph.draw_point((x,y), size=5, color = col_list[0])

        if event ==  'OK':
            if x==-1:
                window["-INFO-"].update(value="Please click")
                continue
            else:
                y,x = y, x
                window.close()
                return ShowPointsYXlist
                break



def multiple_uncaging_click_savetext(flimpath, ch1or2=1):
    ch = ch1or2 - 1
    Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray([flimpath],
                                                                       ch=ch)
    FirstStack = Tiff_MultiArray[0]
    z, ylist, xlist = multiple_uncaging_click(FirstStack,SampleImg=None,
                                              ShowPoint=False,ShowPoint_YX=[110,134])
    print(z, ylist, xlist)
    
    num_pos = len(ylist)
    
    txtpath = flimpath[:-5]+".txt"
    
    with open(txtpath, 'w') as f:
        f.write(str(num_pos)+'\n')
        for nth_pos in range(num_pos):
            f.write(f'{z},{ylist[nth_pos]},{xlist[nth_pos]}\n')
    print(f"Uncaging pos file was saved as {txtpath}")
    

def threeD_img_click(tiffpath,Text="Click",SampleImg=None,ShowPoint=False,ShowPoint_YX=[0,0]):
    TiffShape= first_tiff_read(tiffpath)
    col_list=['red','cyan']
    
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        Z_change=[sg.Text("Z position", size=(20, 1)),
                  sg.Slider(orientation ='horizontal', key='Z',
                            default_value=int(NumOfZ/2), range=(1,NumOfZ),enable_events = True)            
                  ]
        im_PIL,resize_ratio_yx = tiff16bit_to_PIL(tiffpath,Percent=100, show_size_xy=[512,512],
                                                  return_ratio=True,NthSlice=int(NumOfZ/2))
    else:
        NumOfZ = 1
        Z_change=[]
        im_PIL,resize_ratio_yx = tiff16bit_to_PIL(tiffpath,Percent=100, show_size_xy=[512,512],
                                                  return_ratio=True,NthSlice=1)

    data =  PILimg_to_data(im_PIL)    
    
    data_sample = OpenPNG(pngpath=SampleImg)

    sg.theme('Dark Blue 3')

    layout = [
                [sg.Text(Text, font='Arial 10', text_color='black', background_color='white', size=(60, 2))],
                [
                sg.Graph(
                canvas_size=(512, 512), graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-GRAPH-",
                enable_events=True,background_color='lightblue',
                drag_submits=True,motion_events=True,
                ),
                sg.Graph(
                canvas_size=(512, 512), graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-Sample-",
                enable_events=True,background_color='black',
                drag_submits=True,motion_events=True,
                )
                ],
              [
                sg.Text("Contrast", size=(20, 1)),
                sg.Slider(orientation ='horizontal', key='Intensity',default_value=100,
                          range=(1,100),enable_events = True),
              ],
                Z_change
              ,
            [sg.Text(key='-INFO-', size=(60, 1)),sg.Button('OK', size=(20, 2))]
            ]
    
    window = sg.Window("Spine selection", layout, finalize=True)
    graph = window["-GRAPH-"]       # type: sg.Graph
    graph.draw_image(data=data, location=(0,512))
    
    window["-Sample-"].draw_image(data=data_sample, location=(0,512))
    
    if ShowPoint==True:
        col_list=['cyan','red']
        y_2 = 512 - ShowPoint_YX[0]*resize_ratio_yx[0]
        x_2 = ShowPoint_YX[1]*resize_ratio_yx[1]
        graph.draw_point((x_2,y_2), size=5, color = col_list[1])
        
    x=-1;y=-1

    NthSlice = 1
    
    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']

        if len(TiffShape)==3:
            NthSlice = int(values['Z'])

        if event == sg.WIN_CLOSED:
            break

        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            im_PIL = tiff16bit_to_PIL(tiffpath,Percent=ShowIntensityMax,NthSlice=NthSlice)
            data =  PILimg_to_data(im_PIL)
            graph.erase()
            graph.draw_image(data=data, location=(0,512))
            graph.draw_point((x,y), size=5, color = col_list[0])
            if ShowPoint==True:
                graph.draw_point((x_2,y_2), size=5, color = col_list[1])

        if event ==  'Update' or 'Z' or "Intensity":
            im_PIL = tiff16bit_to_PIL(tiffpath,Percent=ShowIntensityMax,NthSlice=NthSlice)
            data =  PILimg_to_data(im_PIL)
            graph.draw_image(data=data, location=(0,512))
            graph.draw_point((x,y), size=5, color = col_list[0])
            if ShowPoint==True:
                graph.draw_point((x_2,y_2), size=5, color = col_list[1])

            
        if event ==  'OK':
            if x==-1:
                window["-INFO-"].update(value="Please click")
                continue
            else:
                z,y,x=NthSlice-1, y, x
                window.close()
                return z,int((512-y)/resize_ratio_yx[0]),int(x/resize_ratio_yx[1])
                break


def TwoD_2ch_img_click(transparent_tiffpath, fluorescent_tiffpath, Text="Click",
                       max_img_xwidth = 600, max_img_ywidth = 600):
    y_pix,x_pix=tiff_size_read(transparent_tiffpath)
    showratio = max(x_pix/max_img_xwidth, y_pix/max_img_ywidth)
    show_size_xy = [int(x_pix/showratio),int(y_pix/showratio) ]
        
    col_list=['red']
    im_PIL,resize_ratio_yx = tiff16bit_to_PIL(transparent_tiffpath,Percent=100, show_size_xy=show_size_xy,
                                              return_ratio=True)
 
    data =  PILimg_to_data(im_PIL)   
    
    sg.theme('Dark Blue 3')

    layout = [
              [
                sg.Text(Text, font='Arial 10', text_color='black', background_color='white', size=(60, 2))
              ],
              [
                sg.Graph(
                canvas_size=(show_size_xy), 
                graph_bottom_left=(0, 0),
                graph_top_right=(show_size_xy),
                key="-GRAPH-",
                enable_events=True,background_color='lightblue',
                drag_submits=True,motion_events=True,
                )
              ],
              [
                sg.Text("Contrast", size=(20, 1)),
                sg.Slider(orientation ='horizontal', key='Intensity',default_value=100,
                          range=(1,100),enable_events = True),
              ],
              [
                sg.Text("Ch", size=(20, 1)),
                sg.Slider(orientation ='horizontal', key='Ch',
                          default_value=1, range=(1,2),enable_events = True)
              ],
              [
               sg.Text(key='-INFO-', size=(60, 1)),sg.Button('OK', size=(20, 2))
              ]
            ]
    
    window = sg.Window("Neuron selection", layout, finalize=True)
    graph = window["-GRAPH-"]       # type: sg.Graph
    graph.draw_image(data=data, location=(0,show_size_xy[1]))

    x=-1;y=-1
   
    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']
        NthCh = int(values['Ch'])
        
        if NthCh == 1:
            tiffpath = transparent_tiffpath
        else:
            tiffpath = fluorescent_tiffpath
        
        if event == sg.WIN_CLOSED:
            break

        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            im_PIL = tiff16bit_to_PIL(tiffpath,Percent=ShowIntensityMax,show_size_xy=show_size_xy)
            data =  PILimg_to_data(im_PIL)
            graph.erase()
            graph.draw_image(data=data, location=(0,show_size_xy[1]))
            graph.draw_point((x,y), size=5, color = col_list[0])

        if event ==  'Update' or "Intensity" or "Ch":
            im_PIL = tiff16bit_to_PIL(tiffpath,Percent=ShowIntensityMax,show_size_xy=show_size_xy)
            data =  PILimg_to_data(im_PIL)
            graph.draw_image(data=data, location=(0,show_size_xy[1]))
            graph.draw_point((x,y), size=5, color = col_list[0])

        if event ==  'OK':
            if x==-1:
                window["-INFO-"].update(value="Please click")
                continue
            else:
                y,x = y, x
                window.close()
                return int((show_size_xy[0]-y)/resize_ratio_yx[0]),int(x/resize_ratio_yx[1])
                break





def TwoD_multiple_click(transparent_tiffpath, fluorescent_tiffpath, Text="Click",
                       max_img_xwidth = 600, max_img_ywidth = 600, ShowPoint_YX=[0,0]):
    y_pix,x_pix=tiff_size_read(transparent_tiffpath)
    showratio = max(x_pix/max_img_xwidth, y_pix/max_img_ywidth)
    show_size_xy = [int(x_pix/showratio),int(y_pix/showratio) ]
        
    col_list=['red','cyan']
    ShowPointsYXlist = []
    
    im_PIL,resize_ratio_yx = tiff16bit_to_PIL(transparent_tiffpath,Percent=100, show_size_xy=show_size_xy,
                                              return_ratio=True)
 
    data =  PILimg_to_data(im_PIL)   
    
    sg.theme('Dark Blue 3')

    layout = [
              [
                sg.Text(Text, font='Arial 10', text_color='black', background_color='white', size=(60, 2))
              ],
              [
                sg.Graph(
                canvas_size=(show_size_xy), 
                graph_bottom_left=(0, 0),
                graph_top_right=(show_size_xy),
                key="-GRAPH-",
                enable_events=True,background_color='lightblue',
                drag_submits=True,motion_events=True,
                )
              ],
              [
                sg.Text("Contrast", size=(20, 1)),
                sg.Slider(orientation ='horizontal', key='Intensity',default_value=100,
                          range=(1,100),enable_events = True),
              ],
              [
                sg.Text("Ch", size=(20, 1)),
                sg.Slider(orientation ='horizontal', key='Ch',
                          default_value=1, range=(1,2),enable_events = True)
              ],
              [
               sg.Text(key='-INFO-', size=(60, 1)),sg.Button('Assign', size=(20, 2)),sg.Button('Reset', size=(20, 2)),sg.Button('OK', size=(20, 2))
              ]
            ]
    
    window = sg.Window("Neuron selection", layout, finalize=True)
    graph = window["-GRAPH-"]       # type: sg.Graph
    graph.draw_image(data=data, location=(0,show_size_xy[1]))

    x=-1;y=-1
   
    if len(ShowPointsYXlist)>0:
         col_list=['cyan','red']
         y_2 = 512 - ShowPoint_YX[0]*resize_ratio_yx[0]
         x_2 = ShowPoint_YX[1]*resize_ratio_yx[1]
         graph.draw_point((x_2,y_2), size=5, color = col_list[1]) 
   
    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']
        NthCh = int(values['Ch'])
        
        if NthCh == 1:
            tiffpath = transparent_tiffpath
        else:
            tiffpath = fluorescent_tiffpath
        
        if event == sg.WIN_CLOSED:
            break

        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            im_PIL = tiff16bit_to_PIL(tiffpath,Percent=ShowIntensityMax,show_size_xy=show_size_xy)
            data =  PILimg_to_data(im_PIL)
            graph.erase()
            graph.draw_image(data=data, location=(0,show_size_xy[1]))
            graph.draw_point((x,y), size=5, color = col_list[0])
            
            if len(ShowPointsYXlist)>0:
                for EachYX in ShowPointsYXlist:
                    graph.draw_point((EachYX[1],EachYX[0]), size=5, color = col_list[1])
            

        if event ==  'Update' or "Intensity" or "Ch":
            im_PIL = tiff16bit_to_PIL(tiffpath,Percent=ShowIntensityMax,show_size_xy=show_size_xy)
            data =  PILimg_to_data(im_PIL)
            graph.draw_image(data=data, location=(0,show_size_xy[1]))
            graph.draw_point((x,y), size=5, color = col_list[0])
            if len(ShowPointsYXlist)>0:
                for EachYX in ShowPointsYXlist:
                    graph.draw_point((EachYX[1],EachYX[0]), size=5, color = col_list[1])
                        
        if event ==  'Assign':
            ShowPointsYXlist.append([y,x])
            print(ShowPointsYXlist)
            
        if event ==  'Reset':
            ShowPointsYXlist = []
    

        if event ==  'OK':
            if x==-1:
                window["-INFO-"].update(value="Please click")
                continue
            else:
                y,x = y, x
                window.close()
                return ShowPointsYXlist
                break



def twoD_click_tiff(twoD_numpy, Text="Click",
                    max_img_xwidth = 600, max_img_ywidth = 600, 
                    ShowPoint_YX=[0,0],
                    predefined = False, predefied_yx_list = [],
                    existing_spines_list=None):
    y_pix,x_pix= twoD_numpy.shape
    showratio = max(x_pix/max_img_xwidth, y_pix/max_img_ywidth)
    show_size_xy = [int(x_pix/showratio),int(y_pix/showratio) ]

    col_list=['magenta', 'green']

    
    im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array = twoD_numpy,
                                              Percent=100,
                                              show_size_xy=show_size_xy,
                                              return_ratio=True)
   
    ShowPointsYXlist = []
    if predefined == True:
        for each_yx in predefied_yx_list:
            pre_def_y = show_size_xy[1] - each_yx[0] * resize_ratio_yx[0]
            pre_def_x = each_yx[1] * resize_ratio_yx[1]
            ShowPointsYXlist.append([pre_def_y,pre_def_x])
            
    showpoint_y = show_size_xy[1] - ShowPoint_YX[0] * resize_ratio_yx[0]
    showpoint_x = ShowPoint_YX[1] * resize_ratio_yx[1]
     
    data =  PILimg_to_data(im_PIL)   
    
    # Prepare right panel with max proj and existing spines
    # Always create right panel to show max projection
    im_PIL_maxproj_right, resize_ratio_maxproj_right = tiffarray_to_PIL(
        stack_array=twoD_numpy, Percent=100, show_size_xy=show_size_xy, return_ratio=True)
    data_maxproj_right = PILimg_to_data(im_PIL_maxproj_right)
    right_panel = [
        sg.Graph(
            canvas_size=(show_size_xy), 
            graph_bottom_left=(0, 0),
            graph_top_right=(show_size_xy),
            key="-MAXPROJ-",
            enable_events=False, background_color='black',
        )
    ]
    
    sg.theme('Dark Blue 3')

    layout = [
              [
                sg.Text(Text, font='Arial 10', text_color='black', background_color='white', size=(60, 2))
              ],
              [
                sg.Graph(
                canvas_size=(show_size_xy), 
                graph_bottom_left=(0, 0),
                graph_top_right=(show_size_xy),
                key="-GRAPH-",
                enable_events=True,background_color='lightblue',
                drag_submits=True,motion_events=True,
                )
              ] + right_panel,
              [
                sg.Text("Contrast", size=(20, 1)),
                sg.Slider(orientation ='horizontal', key='Intensity',default_value=100,
                          range=(1,100),enable_events = True),
              ],
              
              [
               sg.Text(key='-INFO-', size=(60, 1)), 
               sg.Button('Reset', size=(20, 2)),
               sg.Button('OK', size=(20, 2))
              ]
            ]
    
    window = sg.Window("Neuron selection", layout, finalize=True)
    graph = window["-GRAPH-"]       # type: sg.Graph
    graph.draw_image(data=data, location=(0,show_size_xy[1]))
    graph.draw_point((showpoint_x,showpoint_y), size=10, color = col_list[0])
    if len(ShowPointsYXlist)>0:
        for EachYX in ShowPointsYXlist:
            graph.draw_point((EachYX[1],EachYX[0]), 
                             size=5, color = col_list[1])
    
    # Draw right panel with max proj and existing spines
    if "-MAXPROJ-" in window.AllKeysDict:
        graph_maxproj = window["-MAXPROJ-"]
        graph_maxproj.draw_image(data=data_maxproj_right, location=(0, show_size_xy[1]))
        
        # Draw existing spines if provided
        if existing_spines_list is not None and len(existing_spines_list) > 0:
            for spine_info in existing_spines_list:
                if isinstance(spine_info, tuple) and len(spine_info) >= 3:
                    spine_zyx, dend_slope, dend_intercept = spine_info[0], spine_info[1], spine_info[2]
                    # Draw spine point
                    y_max = show_size_xy[1] - spine_zyx[1] * resize_ratio_maxproj_right[0]
                    x_max = spine_zyx[2] * resize_ratio_maxproj_right[1]
                    if 0 <= x_max < show_size_xy[0] and 0 <= y_max < show_size_xy[1]:
                        graph_maxproj.draw_point((x_max, y_max), size=6, color='cyan')
                        
                        # Draw dendrite line
                        dend_x_list = []
                        dend_y_list = []
                        for x in range(0, twoD_numpy.shape[1]):
                            y = dend_slope * x + dend_intercept
                            if 0 <= y < twoD_numpy.shape[0]:
                                dend_x_list.append(x)
                                dend_y_list.append(y)
                        
                        if len(dend_x_list) > 1:
                            points = []
                            for x_pix, y_pix in zip(dend_x_list, dend_y_list):
                                x_graph = x_pix * resize_ratio_maxproj_right[1]
                                y_graph = show_size_xy[1] - y_pix * resize_ratio_maxproj_right[0]
                                if 0 <= x_graph < show_size_xy[0] and 0 <= y_graph < show_size_xy[1]:
                                    points.append((x_graph, y_graph))
                            if len(points) > 1:
                                for i in range(len(points)-1):
                                    graph_maxproj.draw_line(points[i], points[i+1], color='cyan', width=1)
            
    x=-1;y=-1

    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']
                        
        if event == sg.WIN_CLOSED:
            break
        
        if event ==  'Reset':
            ShowPointsYXlist = []
        
        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            ShowPointsYXlist.append([y,x])
        
        if event in ["-GRAPH-", 'Update',"Intensity","Reset"]:
            im_PIL = tiffarray_to_PIL(stack_array = twoD_numpy,
                                      Percent=ShowIntensityMax,
                                      show_size_xy=show_size_xy,
                                      return_ratio=False)
            data =  PILimg_to_data(im_PIL)
            graph.erase()
            graph.draw_image(data=data, location=(0,show_size_xy[1]))
            graph.draw_point((showpoint_x,showpoint_y), size=10, color = col_list[0])
    
            if len(ShowPointsYXlist)>0:
                for EachYX in ShowPointsYXlist:
                    graph.draw_point((EachYX[1],EachYX[0]), 
                                     size=5, color = col_list[1])
            
            # Redraw right panel if exists
            if data_maxproj_right is not None and "-MAXPROJ-" in window.AllKeysDict:
                graph_maxproj = window["-MAXPROJ-"]
                graph_maxproj.erase()
                graph_maxproj.draw_image(data=data_maxproj_right, location=(0, show_size_xy[1]))
                
                # Redraw existing spines
                if existing_spines_list is not None:
                    for spine_info in existing_spines_list:
                        if isinstance(spine_info, tuple) and len(spine_info) >= 3:
                            spine_zyx, dend_slope, dend_intercept = spine_info[0], spine_info[1], spine_info[2]
                            y_max = show_size_xy[1] - spine_zyx[1] * resize_ratio_maxproj_right[0]
                            x_max = spine_zyx[2] * resize_ratio_maxproj_right[1]
                            if 0 <= x_max < show_size_xy[0] and 0 <= y_max < show_size_xy[1]:
                                graph_maxproj.draw_point((x_max, y_max), size=6, color='cyan')
                                
                                # Draw dendrite line
                                dend_x_list = []
                                dend_y_list = []
                                for x in range(0, twoD_numpy.shape[1]):
                                    y = dend_slope * x + dend_intercept
                                    if 0 <= y < twoD_numpy.shape[0]:
                                        dend_x_list.append(x)
                                        dend_y_list.append(y)
                                
                                if len(dend_x_list) > 1:
                                    points = []
                                    for x_pix, y_pix in zip(dend_x_list, dend_y_list):
                                        x_graph = x_pix * resize_ratio_maxproj_right[1]
                                        y_graph = show_size_xy[1] - y_pix * resize_ratio_maxproj_right[0]
                                        if 0 <= x_graph < show_size_xy[0] and 0 <= y_graph < show_size_xy[1]:
                                            points.append((x_graph, y_graph))
                                    if len(points) > 1:
                                        for i in range(len(points)-1):
                                            graph_maxproj.draw_line(points[i], points[i+1], color='cyan', width=1)
            
            sleep(0.1)

        if event ==  'OK':
            if len(ShowPointsYXlist)<1:
                window["-INFO-"].update(value="Please click")
                
            else:                
                window.close()
                yx_list = []
                for EachYX in ShowPointsYXlist:
                    each_yx = [(show_size_xy[1]-EachYX[0])/resize_ratio_yx[0],
                               EachYX[1]/resize_ratio_yx[1]]
                    yx_list.append(each_yx)
                return yx_list
                break
    


def tiffarray_to_PIL2(stack_array,Percent=100,show_size_xy=[512,512],
                     return_ratio=False,NthSlice=1):
    # This can be used for two D image also.
    
    if Percent<1 or Percent>100:
        Percent=100
        
    if len(stack_array.shape)== 3:
        im_array = stack_array[NthSlice-1,:,:]
    else:
        im_array = stack_array
    
    norm_array = (100/Percent)*255*(im_array/im_array.max())
    # print(norm_array)
    norm_array[norm_array>255]=255
    rgb_array = cv2.cvtColor(norm_array.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    im_PIL = Image.fromarray(rgb_array)
    
    im_PIL = im_PIL.resize(show_size_xy)
    resize_ratio_yx = (show_size_xy[0]/im_array.shape[0],show_size_xy[1]/im_array.shape[1])
    if return_ratio==True:    
        return im_PIL,resize_ratio_yx
    else:
        return im_PIL 



    
def get_axis_position(y0,x0,orientation, HalfLen_c=10):
    # x0,x1,x2,x1_1,x2_1,y0,y1,y2,y1_1,y2_1 = get_axis_position(y0,x0,orientation, HalfLen_c=10)
    
    x1 = x0 + math.cos(orientation) * HalfLen_c #* props.minor_axis_length
    y1 = y0 - math.sin(orientation) * HalfLen_c #* props.minor_axis_length
    x2 = x0 - math.sin(orientation) * HalfLen_c #* props.major_axis_length
    y2 = y0 - math.cos(orientation) * HalfLen_c #* props.major_axis_length
    x1_1 = x0 - math.cos(orientation) * HalfLen_c #* props.minor_axis_length
    y1_1 = y0 + math.sin(orientation) * HalfLen_c #* props.minor_axis_length
    x2_1 = x0 + math.sin(orientation) * HalfLen_c #* props.major_axis_length
    y2_1 = y0 + math.cos(orientation) * HalfLen_c #* props.major_axis_length
    return x0,x1,x2,x1_1,x2_1,y0,y1,y2,y1_1,y2_1


def multiuncaging():
    flimpath = r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230606\test\pos2_high_001.flim"
    
    # flimpath = r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230606\test2\pos4_high_003.flim"
    flimpath = r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230606\test2\pos3_high_001.flim"
    # flimpath = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230606\test2\pos3_high_001.flim"
    # multiple_uncaging_click_savetext(flimpath, ch1or2=1)
    # dend_props_forEach(flimpath,square_side_half_len = 25, plot_img=True)
    
    list_of_fileset = [['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos1_neu1_low_002.flim', 'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos1_neu1_high_001.flim'], ['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos2_neu1_low_004.flim', 'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos2_neu1_high_003.flim'], ['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos3_neu2_low_006.flim', 'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos3_neu2_high_005.flim'], ['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos4_neu2_low_008.flim', 'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos4_neu2_high_007.flim'], ['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos5_neu3_low_010.flim', 'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos5_neu3_high_009.flim'], ['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos6_neu3_low_012.flim', 'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230621\\set2\\pos6_neu3_high_011.flim']]
    # list_of_fileset =     [ ['C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230616\\set2\\pos3_low_006.flim', 'C:\\Users\\Yasudalab\\Documents\\Tetsuya_Imaging\\20230616\\set2\\pos3_high_005.flim']]
    ch1or2=2
    for eachlowhigh in list_of_fileset:
        multiple_uncaging_click_savetext(eachlowhigh[1], ch1or2=ch1or2)
        dend_props_forEach(eachlowhigh[1], ch1or2=ch1or2, square_side_half_len = 30, plot_img=True)
        


def define_uncagingPoint_dend_click_multiple(flim_file_path,
                                             read_ini = False,
                                             inipath = "",
                                             SampleImg = None,
                                             only_for_exclusion = False,
                                             existing_spines_list=None):    
    iminfo = FileReader()
    print(flim_file_path)
    iminfo.read_imageFile(flim_file_path, True)
    imagearray=np.array(iminfo.image)
    intensityarray=np.sum(np.sum(np.sum(imagearray,axis=-1),axis=1),axis=1)
    text = "Click the center of the spine you want to stimulate. (Not the uncaging position itself)"
    maxproj = np.sum(intensityarray,axis=0)
    text2 = "Click the dendrite near the selected spine"
    
    # Load existing spines if not provided
    if existing_spines_list is None:
        existing_spines_list = []
        # Try to find existing .ini files for this flim file
        try:
            savefolder = flim_file_path[:-9]  # Remove last 9 chars
            base_name = os.path.basename(flim_file_path[:-9])
            existing_inilist = glob.glob(os.path.join(savefolder, base_name + "*.ini"))
            for existing_inipath in existing_inilist:
                if os.path.exists(existing_inipath):
                    try:
                        spine_zyx, dend_slope, dend_intercept, excluded = read_xyz_single(existing_inipath, return_excluded=True)
                        if not excluded:
                            existing_spines_list.append((spine_zyx, dend_slope, dend_intercept))
                    except:
                        pass
        except:
            pass

    if (read_ini == True) * (os.path.exists(inipath)):
        spine_zyx, dend_slope, dend_intercept = read_xyz_single(inipath = inipath)
        spine_zyx = threeD_array_click(intensityarray, text,
                                 SampleImg = SampleImg, ShowPoint=False,
                                 predefined = True, predefined_ZYX = spine_zyx,
                                 existing_spines_list=existing_spines_list,
                                 maxproj=maxproj)
        if spine_zyx[0] < 0:
            return spine_zyx, 0, 0
            
        predefied_yx_list = []
        for x in range(1,intensityarray.shape[-1]-1):
            y = dend_slope*x + dend_intercept
            if (1<y) * (y < intensityarray.shape[-1]-1):
                predefied_yx_list.append([y,x])
                
        if only_for_exclusion:
            return spine_zyx, 0, 0
        
        yx_list = twoD_click_tiff(twoD_numpy = maxproj, Text=text2,
                                  max_img_xwidth = 600, max_img_ywidth = 600,
                                  ShowPoint_YX=[spine_zyx[1],spine_zyx[2]],
                                  predefined = True,
                                  predefied_yx_list = predefied_yx_list,
                                  existing_spines_list=existing_spines_list)
    else:
        spine_zyx = threeD_array_click(intensityarray,text,
                                 SampleImg=SampleImg,ShowPoint=False,
                                 existing_spines_list=existing_spines_list,
                                 maxproj=maxproj)
        if spine_zyx[0] < 0:
            return spine_zyx, 0, 0
            
        yx_list = twoD_click_tiff(twoD_numpy = maxproj, Text=text2,
                                  max_img_xwidth = 600, max_img_ywidth = 600,
                                  ShowPoint_YX=[spine_zyx[1],spine_zyx[2]],
                                  existing_spines_list=existing_spines_list)

    dend_slope, dend_intercept = np.polyfit(np.array(yx_list)[:,1], np.array(yx_list)[:,0], 1)
    print("dend_slope, dend_intercept :", dend_slope, dend_intercept)
    return spine_zyx, dend_slope, dend_intercept


def define_multiple_files_spine_manager(flim_file_list, savefolder_func_param=None, base_name_func_param=None, 
                                       file_list_update_func=None):
    """
    GUI to manage spines for multiple .flim files.
    Left: 3D stack view for spine selection
    Center: Maximum projection for dendrite selection and existing spine display
    Right: File list and spine list for selected file
    
    Parameters:
    -----------
    flim_file_list : list
        List of .flim file paths
    savefolder_func_param : function, optional
        Function that takes flim_path and returns savefolder path
        If None, uses flim_path[:-9]
    base_name_func_param : function, optional
        Function that takes flim_path and returns base_name for .ini files
        If None, uses os.path.basename(flim_path[:-9])
        
    Returns:
    --------
    dict : Dictionary with flim_file_path as key and list of (spine_zyx, dend_slope, dend_intercept) as value
    """
    if savefolder_func_param is None:
        def savefolder_func(flim_path):
            return flim_path[:-9]
    else:
        savefolder_func = savefolder_func_param
    
    if base_name_func_param is None:
        def base_name_func(flim_path):
            return os.path.basename(flim_path[:-9])
    else:
        base_name_func = base_name_func_param
    
    # Store spine data for each file
    file_spines_data = {}  # {flim_path: {inipath: (spine_zyx, dend_slope, dend_intercept, excluded), ...}}
    
    # Store original file list for refresh functionality
    original_flim_file_list = flim_file_list.copy()
    
    # Initialize data structure
    for flim_path in flim_file_list:
        file_spines_data[flim_path] = {}
        savefolder = savefolder_func(flim_path)
        base_name = base_name_func(flim_path)
        existing_inilist = glob.glob(os.path.join(savefolder, base_name + "*.ini"))
        for inipath in existing_inilist:
            if os.path.exists(inipath):
                try:
                    spine_zyx, dend_slope, dend_intercept, excluded = read_xyz_single(inipath, return_excluded=True)
                    file_spines_data[flim_path][inipath] = (spine_zyx, dend_slope, dend_intercept, excluded)
                except:
                    pass
    
    # Create GUI
    sg.theme('Dark Blue 3')
    
    # File list (short names for display) - keep as module-level variable
    file_display_names = [os.path.basename(f) for f in flim_file_list]
    
    def get_file_display_names():
        """Get current file display names"""
        return [os.path.basename(f) for f in flim_file_list]
    
    def refresh_file_list():
        """Refresh file list using file_list_update_func if provided"""
        nonlocal flim_file_list, file_display_names, current_flim_path
        
        if file_list_update_func is not None:
            new_file_list = file_list_update_func()
            # Add new files to the list
            for new_file in new_file_list:
                if new_file not in flim_file_list:
                    flim_file_list.append(new_file)
                    # Initialize spine data for new file
                    file_spines_data[new_file] = {}
                    savefolder = savefolder_func(new_file)
                    base_name = base_name_func(new_file)
                    existing_inilist = glob.glob(os.path.join(savefolder, base_name + "*.ini"))
                    for inipath in existing_inilist:
                        if os.path.exists(inipath):
                            try:
                                spine_zyx, dend_slope, dend_intercept, excluded = read_xyz_single(inipath, return_excluded=True)
                                file_spines_data[new_file][inipath] = (spine_zyx, dend_slope, dend_intercept, excluded)
                            except:
                                pass
            
            # Update display
            file_display_names = get_file_display_names()
            window["-FILELIST-"].update(values=file_display_names)
            
            # Try to maintain current selection
            if current_flim_path and current_flim_path in flim_file_list:
                try:
                    idx = flim_file_list.index(current_flim_path)
                    window["-FILELIST-"].update(set_to_index=[idx])
                except:
                    pass
    
    layout = [
        [sg.Text("Multiple File Spine Manager", font='Arial 14', text_color='black', background_color='white', size=(100, 1))],
        [
            # Left panel: 3D view for spine selection (will be populated when file is selected)
            sg.Column([
                [sg.Text("3D Stack View", size=(30, 1))],
                [sg.Graph(canvas_size=(400, 400), graph_bottom_left=(0, 0),
                         graph_top_right=(400, 400), key="-GRAPH3D-",
                         enable_events=True, background_color='lightblue',
                         drag_submits=True, motion_events=True)],
                [sg.Text("Contrast", size=(20, 1)),
                 sg.Slider(orientation='horizontal', key='Intensity', default_value=100,
                          range=(1, 100), enable_events=True)],
                [sg.Text("Z position", size=(20, 1)),
                 sg.Slider(orientation='horizontal', key='Z', default_value=1,
                          range=(1, 100), enable_events=True)]
            ], vertical_alignment='top'),
            # Center panel: Maximum projection for dendrite and existing spines
            sg.Column([
                [sg.Text("Maximum Projection", size=(30, 1))],
                [sg.Graph(canvas_size=(400, 400), graph_bottom_left=(0, 0),
                         graph_top_right=(400, 400), key="-GRAPHMAX-",
                         enable_events=True, background_color='black',
                         drag_submits=True, motion_events=True)],
            ], vertical_alignment='top'),
            # Right panel: File list and spine list
            sg.Column([
                [sg.Text("Files", size=(30, 1)),
                 sg.Button('Refresh File List', size=(15, 1))],
                [sg.Listbox(values=file_display_names, size=(40, 15), key="-FILELIST-",
                           enable_events=True, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)],
                [sg.Text("Spines for selected file:", size=(40, 1))],
                [sg.Listbox(values=[], size=(40, 10), key="-SPINELIST-",
                           enable_events=True, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)],
                [sg.Button('Add New Spine', size=(20, 2)),
                 sg.Button('Edit Selected Spine', size=(20, 2))],
                [sg.Button('Delete Selected Spine', size=(20, 2)),
                 sg.Button('Done', size=(20, 2))]
            ], vertical_alignment='top')
        ],
        [sg.Text(key='-INFO-', size=(100, 2))]
    ]
    
    window = sg.Window("Multiple File Spine Manager", layout, finalize=True, resizable=True)
    
    # Current state
    current_flim_path = None
    current_intensityarray = None
    current_maxproj = None
    current_im_PIL_maxproj = None
    resize_ratio_maxproj = None
    data_maxproj = None
    current_im_PIL = None
    resize_ratio_yx = None
    data_3d = None
    NumOfZ = 1
    NthSlice = 1
    ShowIntensityMax = 100
    selected_spine_inipath_global = None  # Store selected spine inipath globally
    
    def update_file_display():
        """Update the file list display"""
        file_display_names = [os.path.basename(f) for f in flim_file_list]
        window["-FILELIST-"].update(values=file_display_names)
        if current_flim_path:
            try:
                idx = flim_file_list.index(current_flim_path)
                window["-FILELIST-"].update(set_to_index=[idx])
            except:
                pass
    
    def update_spine_list_display():
        """Update the spine list for current file"""
        if current_flim_path is None:
            window["-SPINELIST-"].update(values=[])
            return
        
        spine_list = []
        for inipath in sorted(file_spines_data[current_flim_path].keys()):
            spine_zyx, dend_slope, dend_intercept, excluded = file_spines_data[current_flim_path][inipath]
            spine_basename = os.path.basename(inipath)
            status = "EXCLUDED" if excluded else "OK"
            spine_list.append(f"{spine_basename} - {status}")
        window["-SPINELIST-"].update(values=spine_list)
    
    def load_file(flim_path):
        """Load a .flim file and update displays"""
        nonlocal current_flim_path, current_intensityarray, current_maxproj
        nonlocal current_im_PIL_maxproj, resize_ratio_maxproj, data_maxproj
        nonlocal current_im_PIL, resize_ratio_yx, data_3d, NumOfZ, NthSlice
        nonlocal selected_spine_inipath_global
        
        try:
            # Update current file path first
            current_flim_path = flim_path
            # Reset selected spine when loading new file
            selected_spine_inipath_global = None
            
            # Reload spine data for this file from disk to ensure it's up to date
            savefolder = savefolder_func(flim_path)
            base_name = base_name_func(flim_path)
            existing_inilist = glob.glob(os.path.join(savefolder, base_name + "*.ini"))
            
            # Update file_spines_data for current file
            file_spines_data[flim_path] = {}
            for inipath in existing_inilist:
                if os.path.exists(inipath):
                    try:
                        spine_zyx, dend_slope, dend_intercept, excluded = read_xyz_single(inipath, return_excluded=True)
                        file_spines_data[flim_path][inipath] = (spine_zyx, dend_slope, dend_intercept, excluded)
                    except:
                        pass
            
            iminfo = FileReader()
            iminfo.read_imageFile(flim_path, True)
            imagearray = np.array(iminfo.image)
            current_intensityarray = np.sum(np.sum(np.sum(imagearray, axis=-1), axis=1), axis=1)
            current_maxproj = np.sum(current_intensityarray, axis=0)
            
            # Prepare maximum projection
            current_im_PIL_maxproj, resize_ratio_maxproj = tiffarray_to_PIL(
                stack_array=current_maxproj, Percent=100, show_size_xy=[400, 400], 
                return_ratio=True, NthSlice=1)
            data_maxproj = PILimg_to_data(current_im_PIL_maxproj)
            
            # Prepare 3D view
            TiffShape = current_intensityarray.shape
            if len(TiffShape) == 3:
                NumOfZ = TiffShape[0]
                NthSlice = int(NumOfZ/2 + 1)
                window["Z"].update(range=(1, NumOfZ), value=NthSlice)
            else:
                NumOfZ = 1
                window["Z"].update(range=(1, 1), value=1)
            
            current_im_PIL, resize_ratio_yx = tiffarray_to_PIL(
                current_intensityarray, Percent=100, show_size_xy=[400, 400],
                return_ratio=True, NthSlice=NthSlice)
            data_3d = PILimg_to_data(current_im_PIL)
            
            # Update graphs
            graph_3d = window["-GRAPH3D-"]
            graph_max = window["-GRAPHMAX-"]
            
            graph_3d.erase()
            graph_3d.draw_image(data=data_3d, location=(0, 400))
            graph_max.erase()
            graph_max.draw_image(data=data_maxproj, location=(0, 400))
            
            # Draw existing spines (now current_flim_path is already updated)
            # Selected spine will be determined inside draw_existing_spines function
            draw_existing_spines()
            
            update_spine_list_display()
            window["-INFO-"].update(value=f"Loaded: {os.path.basename(flim_path)}")
        except Exception as e:
            window["-INFO-"].update(value=f"Error loading file: {str(e)}")
    
    def draw_existing_spines():
        """Draw existing spines on both graphs"""
        nonlocal selected_spine_inipath_global
        
        if current_flim_path is None or current_flim_path not in file_spines_data:
            return
        
        graph_3d = window["-GRAPH3D-"]
        graph_max = window["-GRAPHMAX-"]
        
        # Use global selected spine inipath
        selected_spine_inipath = selected_spine_inipath_global
        
        for idx, (inipath, (spine_zyx, dend_slope, dend_intercept, excluded)) in enumerate(file_spines_data[current_flim_path].items()):
            if excluded:
                continue
            
            # Color: red for selected spine, cyan for others
            if inipath == selected_spine_inipath:
                color = 'red'
            else:
                color = 'cyan'
            
            # Draw on 3D view
            if current_intensityarray is not None and len(current_intensityarray.shape) == 3:
                if spine_zyx[0] >= 0 and spine_zyx[0] < current_intensityarray.shape[0]:
                    y_3d = 400 - spine_zyx[1] * resize_ratio_yx[0]
                    x_3d = spine_zyx[2] * resize_ratio_yx[1]
                    if 0 <= x_3d < 400 and 0 <= y_3d < 400:
                        graph_3d.draw_point((x_3d, y_3d), size=6, color=color)
            
            # Draw on max projection
            y_max = 400 - spine_zyx[1] * resize_ratio_maxproj[0]
            x_max = spine_zyx[2] * resize_ratio_maxproj[1]
            if 0 <= x_max < 400 and 0 <= y_max < 400:
                graph_max.draw_point((x_max, y_max), size=6, color=color)
                
                # Draw dendrite line
                dend_x_list = []
                dend_y_list = []
                for x in range(0, current_maxproj.shape[1]):
                    y = dend_slope * x + dend_intercept
                    if 0 <= y < current_maxproj.shape[0]:
                        dend_x_list.append(x)
                        dend_y_list.append(y)
                
                if len(dend_x_list) > 1:
                    points = []
                    for x_pix, y_pix in zip(dend_x_list, dend_y_list):
                        x_graph = x_pix * resize_ratio_maxproj[1]
                        y_graph = 400 - y_pix * resize_ratio_maxproj[0]
                        if 0 <= x_graph < 400 and 0 <= y_graph < 400:
                            points.append((x_graph, y_graph))
                    if len(points) > 1:
                        for i in range(len(points)-1):
                            graph_max.draw_line(points[i], points[i+1], color=color, width=1)
    
    # Initialize: load first file if available
    if len(flim_file_list) > 0:
        load_file(flim_file_list[0])
    
    # Event loop
    while True:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED:
            break
        
        if event == "Refresh File List":
            refresh_file_list()
            window["-INFO-"].update(value="File list refreshed.")
        
        if event == "-SPINELIST-" and len(values.get("-SPINELIST-", [])) > 0:
            # Update global selected spine
            selected_spine_display = values["-SPINELIST-"][0]
            spine_basename = selected_spine_display.split(" - ")[0]
            savefolder = savefolder_func(current_flim_path) if current_flim_path else ""
            selected_spine_inipath_global = os.path.join(savefolder, spine_basename) if savefolder else None
            
            # Get selected spine's Z coordinate and update Z slider
            if (current_flim_path is not None and 
                current_flim_path in file_spines_data and 
                selected_spine_inipath_global in file_spines_data[current_flim_path]):
                spine_zyx, dend_slope, dend_intercept, excluded = file_spines_data[current_flim_path][selected_spine_inipath_global]
                if len(current_intensityarray.shape) == 3 and spine_zyx[0] >= 0:
                    # Z coordinate is 0-indexed, slider is 1-indexed
                    selected_z = spine_zyx[0] + 1
                    NthSlice = selected_z  # Update NthSlice variable
                    window["Z"].update(value=selected_z)
                    # Update 3D view to show selected Z plane
                    ShowIntensityMax = values['Intensity']
                    current_im_PIL, new_resize_ratio_yx = tiffarray_to_PIL(
                        current_intensityarray, Percent=ShowIntensityMax, show_size_xy=[400, 400],
                        return_ratio=True, NthSlice=selected_z)
                    data_3d = PILimg_to_data(current_im_PIL)
                    # Update global resize_ratio_yx for draw_existing_spines
                    resize_ratio_yx = new_resize_ratio_yx
                    graph_3d = window["-GRAPH3D-"]
                    graph_3d.erase()
                    graph_3d.draw_image(data=data_3d, location=(0, 400))
                    draw_existing_spines()
                else:
                    # Redraw graphs without Z change
                    graph_3d = window["-GRAPH3D-"]
                    graph_max = window["-GRAPHMAX-"]
                    graph_3d.erase()
                    graph_3d.draw_image(data=data_3d, location=(0, 400))
                    graph_max.erase()
                    graph_max.draw_image(data=data_maxproj, location=(0, 400))
                    draw_existing_spines()
            else:
                # Redraw graphs
                graph_3d = window["-GRAPH3D-"]
                graph_max = window["-GRAPHMAX-"]
                graph_3d.erase()
                graph_3d.draw_image(data=data_3d, location=(0, 400))
                graph_max.erase()
                graph_max.draw_image(data=data_maxproj, location=(0, 400))
                draw_existing_spines()
        
        if event == "-FILELIST-" and len(values["-FILELIST-"]) > 0:
            # Reset selected spine when switching files
            selected_spine_inipath_global = None
            
            selected_file_display = values["-FILELIST-"][0]
            try:
                current_display_names = get_file_display_names()
                idx = current_display_names.index(selected_file_display)
                selected_file = flim_file_list[idx]
                load_file(selected_file)
            except Exception as e:
                window["-INFO-"].update(value=f"Error selecting file: {str(e)}")
        
        if event == "Add New Spine":
            if current_flim_path is None:
                window["-INFO-"].update(value="Please select a file first.")
                continue
            
            # Use define_uncagingPoint_dend_click_multiple to add new spine
            try:
                spine_zyx, dend_slope, dend_intercept = define_uncagingPoint_dend_click_multiple(
                    current_flim_path, read_ini=False, inipath="", SampleImg=None, only_for_exclusion=False)
                
                if spine_zyx[0] < 0:
                    window["-INFO-"].update(value="Spine selection cancelled or excluded.")
                    continue
                
                # Determine save path
                savefolder = savefolder_func(current_flim_path)
                os.makedirs(savefolder, exist_ok=True)
                base_name = base_name_func(current_flim_path)
                
                # Find next available index
                existing_inilist = glob.glob(os.path.join(savefolder, base_name + "*.ini"))
                existing_indices = []
                for inipath in existing_inilist:
                    match = re.search(r'_(\d{3})\.ini$', os.path.basename(inipath))
                    if match:
                        existing_indices.append(int(match.group(1)))
                
                next_idx = max(existing_indices) + 1 if existing_indices else 0
                new_inipath = os.path.join(savefolder, base_name + f"_{str(next_idx).zfill(3)}.ini")
                
                # Save and add to data structure
                save_spine_dend_info(spine_zyx, dend_slope, dend_intercept, new_inipath, excluded=0)
                file_spines_data[current_flim_path][new_inipath] = (spine_zyx, dend_slope, dend_intercept, 0)
                
                # Reload file to refresh display
                load_file(current_flim_path)
                window["-INFO-"].update(value=f"New spine saved: {os.path.basename(new_inipath)}")
            except Exception as e:
                window["-INFO-"].update(value=f"Error adding spine: {str(e)}")
        
        if event == "Edit Selected Spine":
            if current_flim_path is None or len(values["-SPINELIST-"]) == 0:
                window["-INFO-"].update(value="Please select a spine to edit.")
                continue
            
            try:
                selected_spine_display = values["-SPINELIST-"][0]
                # Extract inipath from display string
                spine_basename = selected_spine_display.split(" - ")[0]
                savefolder = savefolder_func(current_flim_path)
                inipath = os.path.join(savefolder, spine_basename)
                
                if inipath not in file_spines_data[current_flim_path]:
                    window["-INFO-"].update(value="Selected spine not found in data.")
                    continue
                
                # Use define_uncagingPoint_dend_click_multiple to edit
                spine_zyx, dend_slope, dend_intercept = define_uncagingPoint_dend_click_multiple(
                    current_flim_path, read_ini=True, inipath=inipath, SampleImg=None, only_for_exclusion=False)
                
                if spine_zyx[0] < 0:
                    # Mark as excluded
                    spine_zyx_old, dend_slope_old, dend_intercept_old, excluded_old = file_spines_data[current_flim_path][inipath]
                    save_spine_dend_info(spine_zyx_old, dend_slope_old, dend_intercept_old, inipath, excluded=1)
                    file_spines_data[current_flim_path][inipath] = (spine_zyx_old, dend_slope_old, dend_intercept_old, 1)
                else:
                    # Update spine
                    save_spine_dend_info(spine_zyx, dend_slope, dend_intercept, inipath, excluded=0)
                    file_spines_data[current_flim_path][inipath] = (spine_zyx, dend_slope, dend_intercept, 0)
                
                # Reload file to refresh display
                load_file(current_flim_path)
                window["-INFO-"].update(value=f"Spine updated: {spine_basename}")
            except Exception as e:
                window["-INFO-"].update(value=f"Error editing spine: {str(e)}")
        
        if event == "Delete Selected Spine":
            if current_flim_path is None or len(values["-SPINELIST-"]) == 0:
                window["-INFO-"].update(value="Please select a spine to delete.")
                continue
            
            try:
                selected_spine_display = values["-SPINELIST-"][0]
                spine_basename = selected_spine_display.split(" - ")[0]
                savefolder = savefolder_func(current_flim_path)
                inipath = os.path.join(savefolder, spine_basename)
                
                if inipath in file_spines_data[current_flim_path]:
                    # Delete file
                    if os.path.exists(inipath):
                        os.remove(inipath)
                    # Remove from data structure
                    del file_spines_data[current_flim_path][inipath]
                    # Reload file to refresh display
                    load_file(current_flim_path)
                    window["-INFO-"].update(value=f"Spine deleted: {spine_basename}")
                else:
                    window["-INFO-"].update(value="Selected spine not found.")
            except Exception as e:
                window["-INFO-"].update(value=f"Error deleting spine: {str(e)}")
        
        if event in ["Z", "Intensity"]:
            if current_flim_path is None:
                continue
            
            ShowIntensityMax = values['Intensity']
            if len(current_intensityarray.shape) == 3:
                NthSlice = int(values['Z'])
            
            # Update 3D view
            current_im_PIL, resize_ratio_yx = tiffarray_to_PIL(
                current_intensityarray, Percent=ShowIntensityMax, show_size_xy=[400, 400],
                return_ratio=True, NthSlice=NthSlice)
            data_3d = PILimg_to_data(current_im_PIL)
            graph_3d = window["-GRAPH3D-"]
            graph_3d.erase()
            graph_3d.draw_image(data=data_3d, location=(0, 400))
            draw_existing_spines()
        
        if event == "Done":
            break
    
    window.close()
    
    # Return results in the format: {flim_path: [(spine_zyx, dend_slope, dend_intercept), ...]}
    result_dict = {}
    for flim_path, spines_dict in file_spines_data.items():
        result_list = []
        for inipath, (spine_zyx, dend_slope, dend_intercept, excluded) in spines_dict.items():
            if not excluded:
                result_list.append((spine_zyx, dend_slope, dend_intercept))
        result_dict[flim_path] = result_list
    
    return result_dict


def define_multiple_uncagingPoint_dend_click(flim_file_path,
                                             existing_spines_list=[],
                                             existing_inipaths=[]):
    """
    Allow clicking multiple spine-dendrite pairs from a single .flim file.
    Right panel shows maximum projection with already selected spines.
    
    Parameters:
    -----------
    flim_file_path : str
        Path to the .flim file
    existing_spines_list : list
        List of existing spine ZYX coordinates [[z1,y1,x1], [z2,y2,x2], ...]
    existing_inipaths : list
        List of existing .ini file paths for reading existing spine info
        
    Returns:
    --------
    result_list : list
        List of tuples (spine_zyx, dend_slope, dend_intercept) for each clicked pair
    """
    iminfo = FileReader()
    print(flim_file_path)
    iminfo.read_imageFile(flim_file_path, True)
    imagearray = np.array(iminfo.image)
    intensityarray = np.sum(np.sum(np.sum(imagearray, axis=-1), axis=1), axis=1)
    maxproj = np.sum(intensityarray, axis=0)
    
    # Read existing spine information if provided
    existing_spines_zyx = []
    existing_dend_slopes = []
    existing_dend_intercepts = []
    if len(existing_inipaths) > 0:
        for inipath in existing_inipaths:
            if os.path.exists(inipath):
                spine_zyx, dend_slope, dend_intercept = read_xyz_single(inipath=inipath)
                existing_spines_zyx.append(spine_zyx)
                existing_dend_slopes.append(dend_slope)
                existing_dend_intercepts.append(dend_intercept)
    elif len(existing_spines_list) > 0:
        existing_spines_zyx = existing_spines_list
    
    # Prepare maximum projection image for right panel
    im_PIL_maxproj, resize_ratio_maxproj = tiffarray_to_PIL(
        stack_array=maxproj, Percent=100, show_size_xy=[512, 512], 
        return_ratio=True, NthSlice=1)
    data_maxproj = PILimg_to_data(im_PIL_maxproj)
    
    # Initialize result list
    result_list = []
    col_list = ['red', 'cyan', 'yellow', 'green', 'magenta']
    
    TiffShape = intensityarray.shape
    if len(TiffShape) == 3:
        NumOfZ = TiffShape[0]
        default_Z = int(NumOfZ/2 + 1)
        Z_change = [sg.Text("Z position", size=(20, 1)),
                    sg.Slider(orientation='horizontal', key='Z',
                             default_value=default_Z, range=(1, NumOfZ), enable_events=True)]
        im_PIL, resize_ratio_yx = tiffarray_to_PIL(
            intensityarray, Percent=100, show_size_xy=[512, 512],
            return_ratio=True, NthSlice=default_Z)
    else:
        NumOfZ = 1
        Z_change = []
        im_PIL, resize_ratio_yx = tiffarray_to_PIL(
            intensityarray, Percent=100, show_size_xy=[512, 512],
            return_ratio=True, NthSlice=1)
    
    data = PILimg_to_data(im_PIL)
    
    sg.theme('Dark Blue 3')
    
    layout = [
        [sg.Text("Click spine center, then click dendrite. Click 'Add' to save current pair, 'Done' when finished.", 
                 font='Arial 10', text_color='black', background_color='white', size=(100, 2))],
        [
            sg.Graph(
                canvas_size=(512, 512), graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-GRAPH-",
                enable_events=True, background_color='lightblue',
                drag_submits=True, motion_events=True,
            ),
            sg.Graph(
                canvas_size=(512, 512), graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-MAXPROJ-",
                enable_events=True, background_color='black',
                drag_submits=True, motion_events=True,
            )
        ],
        [
            sg.Text("Contrast", size=(20, 1)),
            sg.Slider(orientation='horizontal', key='Intensity', default_value=100,
                     range=(1, 100), enable_events=True),
        ],
        Z_change,
        [sg.Text(key='-INFO-', size=(60, 1)),
         sg.Button('Add', size=(15, 2)),
         sg.Button('Reset Current', size=(15, 2)),
         sg.Button('Done', size=(15, 2))]
    ]
    
    window = sg.Window("Multiple Spine Selection", layout, finalize=True)
    graph = window["-GRAPH-"]
    graph_maxproj = window["-MAXPROJ-"]
    
    # Draw initial images
    graph.draw_image(data=data, location=(0, 512))
    graph_maxproj.draw_image(data=data_maxproj, location=(0, 512))
    
    # Draw existing spines on both panels
    for idx, existing_spine in enumerate(existing_spines_zyx):
        color = col_list[idx % len(col_list)]
        # Draw on 3D view (left)
        y_3d = 512 - existing_spine[1] * resize_ratio_yx[0]
        x_3d = existing_spine[2] * resize_ratio_yx[1]
        graph.draw_point((x_3d, y_3d), size=8, color=color)
        # Draw on max projection (right)
        y_max = 512 - existing_spine[1] * resize_ratio_maxproj[0]
        x_max = existing_spine[2] * resize_ratio_maxproj[1]
        graph_maxproj.draw_point((x_max, y_max), size=8, color=color)
    
    # Current selection variables
    current_spine_zyx = None
    current_yx_list = []
    x = -1
    y = -1
    NthSlice = default_Z if len(TiffShape) == 3 else 1
    ShowIntensityMax = 100
    
    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']
        
        if len(TiffShape) == 3:
            NthSlice = int(values['Z'])
        
        if event == sg.WIN_CLOSED:
            window.close()
            return result_list
        
        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            if current_spine_zyx is None:
                # First click: select spine
                z, y_pix, x_pix = NthSlice - 1, y, x
                y_pix = int((512 - y) / resize_ratio_yx[0])
                x_pix = int(x / resize_ratio_yx[1])
                current_spine_zyx = [z, y_pix, x_pix]
                window["-INFO-"].update(value=f"Spine selected at Z={z}, Y={y_pix}, X={x_pix}. Now click dendrite on right panel.")
                
                # Update display
                im_PIL = tiffarray_to_PIL(intensityarray, Percent=ShowIntensityMax, 
                                         NthSlice=NthSlice, show_size_xy=[512, 512])
                data = PILimg_to_data(im_PIL)
                graph.erase()
                graph.draw_image(data=data, location=(0, 512))
                
                # Redraw existing spines
                for idx, existing_spine in enumerate(existing_spines_zyx):
                    color = col_list[idx % len(col_list)]
                    y_ex = 512 - existing_spine[1] * resize_ratio_yx[0]
                    x_ex = existing_spine[2] * resize_ratio_yx[1]
                    graph.draw_point((x_ex, y_ex), size=8, color=color)
                
                # Draw current spine
                graph.draw_point((x, y), size=10, color='magenta')
                # Also draw on max projection
                y_max = 512 - current_spine_zyx[1] * resize_ratio_maxproj[0]
                x_max = current_spine_zyx[2] * resize_ratio_maxproj[1]
                graph_maxproj.draw_point((x_max, y_max), size=10, color='magenta')
        
        if event == "-MAXPROJ-":
            x, y = values["-MAXPROJ-"]
            if current_spine_zyx is not None:
                # Add dendrite points on maximum projection
                y_pix = int((512 - y) / resize_ratio_maxproj[0])
                x_pix = int(x / resize_ratio_maxproj[1])
                current_yx_list.append([y_pix, x_pix])
                # Draw dendrite point
                graph_maxproj.draw_point((x, y), size=5, color='cyan')
                window["-INFO-"].update(value=f"Dendrite point added. Total: {len(current_yx_list)}. Click 'Add' to save pair or click more points.")
        
        if event == 'Update' or event == 'Z' or event == "Intensity":
            im_PIL = tiffarray_to_PIL(intensityarray, Percent=ShowIntensityMax, NthSlice=NthSlice, show_size_xy=[512, 512])
            data = PILimg_to_data(im_PIL)
            graph.erase()
            graph.draw_image(data=data, location=(0, 512))
            
            # Redraw existing spines
            for idx, existing_spine in enumerate(existing_spines_zyx):
                color = col_list[idx % len(col_list)]
                y_ex = 512 - existing_spine[1] * resize_ratio_yx[0]
                x_ex = existing_spine[2] * resize_ratio_yx[1]
                graph.draw_point((x_ex, y_ex), size=8, color=color)
            
            # Redraw current selection if exists
            if current_spine_zyx is not None:
                y_curr = 512 - current_spine_zyx[1] * resize_ratio_yx[0]
                x_curr = current_spine_zyx[2] * resize_ratio_yx[1]
                graph.draw_point((x_curr, y_curr), size=10, color='magenta')
        
        if event == 'Add':
            if current_spine_zyx is None or len(current_yx_list) < 2:
                window["-INFO-"].update(value="Please select spine and at least 2 dendrite points first.")
            else:
                # Calculate dendrite slope and intercept
                yx_array = np.array(current_yx_list)
                dend_slope, dend_intercept = np.polyfit(yx_array[:, 1], yx_array[:, 0], 1)
                
                # Add to result list
                result_list.append((current_spine_zyx, dend_slope, dend_intercept))
                
                # Update existing spines list for display
                existing_spines_zyx.append(current_spine_zyx)
                
                # Update max projection display
                graph_maxproj.erase()
                graph_maxproj.draw_image(data=data_maxproj, location=(0, 512))
                for idx, existing_spine in enumerate(existing_spines_zyx):
                    color = col_list[idx % len(col_list)]
                    y_max = 512 - existing_spine[1] * resize_ratio_maxproj[0]
                    x_max = existing_spine[2] * resize_ratio_maxproj[1]
                    graph_maxproj.draw_point((x_max, y_max), size=8, color=color)
                
                window["-INFO-"].update(value=f"Pair {len(result_list)} saved. Total: {len(result_list)} pairs.")
                current_spine_zyx = None
                current_yx_list = []
                x = -1
                y = -1
                
                # Update 3D view to remove current selection markers
                im_PIL = tiffarray_to_PIL(intensityarray, Percent=ShowIntensityMax, NthSlice=NthSlice, show_size_xy=[512, 512])
                data = PILimg_to_data(im_PIL)
                graph.erase()
                graph.draw_image(data=data, location=(0, 512))
                for idx, existing_spine in enumerate(existing_spines_zyx):
                    color = col_list[idx % len(col_list)]
                    y_ex = 512 - existing_spine[1] * resize_ratio_yx[0]
                    x_ex = existing_spine[2] * resize_ratio_yx[1]
                    graph.draw_point((x_ex, y_ex), size=8, color=color)
        
        if event == 'Reset Current':
            current_spine_zyx = None
            current_yx_list = []
            x = -1
            y = -1
            window["-INFO-"].update(value="Current selection reset.")
            
            # Redraw without current selection
            im_PIL = tiffarray_to_PIL(intensityarray, Percent=ShowIntensityMax, NthSlice=NthSlice, show_size_xy=[512, 512])
            data = PILimg_to_data(im_PIL)
            graph.erase()
            graph.draw_image(data=data, location=(0, 512))
            for idx, existing_spine in enumerate(existing_spines_zyx):
                color = col_list[idx % len(col_list)]
                y_ex = 512 - existing_spine[1] * resize_ratio_yx[0]
                x_ex = existing_spine[2] * resize_ratio_yx[1]
                graph.draw_point((x_ex, y_ex), size=8, color=color)
            
            # Reset max projection display
            graph_maxproj.erase()
            graph_maxproj.draw_image(data=data_maxproj, location=(0, 512))
            for idx, existing_spine in enumerate(existing_spines_zyx):
                color = col_list[idx % len(col_list)]
                y_max = 512 - existing_spine[1] * resize_ratio_maxproj[0]
                x_max = existing_spine[2] * resize_ratio_maxproj[1]
                graph_maxproj.draw_point((x_max, y_max), size=8, color=color)
        
        if event == 'Done':
            window.close()
            return result_list
    
    window.close()
    return result_list


def save_spine_dend_info(spine_zyx, dend_slope, dend_intercept, inipath,
                         excluded = 0):
    config = configparser.ConfigParser()
    config['uncaging_settings'] = {}
    config['uncaging_settings']['spine_z'] = str(spine_zyx[0])
    config['uncaging_settings']['spine_y'] = str(spine_zyx[1])
    config['uncaging_settings']['spine_x'] = str(spine_zyx[2])
    config['uncaging_settings']['dend_slope'] = str(dend_slope)
    config['uncaging_settings']['dend_intercept'] = str(dend_intercept)
    config['uncaging_settings']['excluded'] = str(excluded)
    with open(inipath, 'w') as configfile:
        config.write(configfile)
    print("spine and dend info saved as",inipath)
        
def read_xyz_single(inipath, return_excluded = False):
    config = configparser.ConfigParser()
    # print(inipath)
    if os.path.exists(inipath) == False:
        print(inipath, "do not exist")
        assert False
    # assert(os.path.exists(inipath))
    config.read(inipath,encoding='cp932')
    if len(config.sections()) != 1:
        raise Exception('The inifile does not have single section.')
    else:
        z = int(config['uncaging_settings']['spine_z'])
        y = int(config['uncaging_settings']['spine_y'])
        x = int(config['uncaging_settings']['spine_x'])
        dend_slope = float(config['uncaging_settings']['dend_slope'])
        dend_intercept = float(config['uncaging_settings']['dend_intercept'])
        spine_zyx = (z, y, x)
        if return_excluded:
            try:
                excluded = int(config['uncaging_settings']['excluded'])
            except:
                if spine_zyx[0] < -1:
                    excluded = 1
                else:
                    print("\n"*3,"could not read exclusion info","\n"*3)
                    excluded = 0
            return spine_zyx, dend_slope, dend_intercept, excluded
        return spine_zyx, dend_slope, dend_intercept

    
    

def main():
        # tiffpath=r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\Intensity\A1_dendrite_10_20um_40h__Ch1_001.tif"
    # Spine_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Spine_example.png"
    # Dendrite_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Dendrite_example.png"
    flim_file_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20241029\24well_1011_1016_FLEXGFP\highmag_GFP200ms55p\tpem\A1_72h_dend5_23um_001.flim"
    spine_zyx, dend_slope, dend_intercept = define_uncagingPoint_dend_click_multiple(flim_file_path)
    
    # z, ylist, xlist = MultipleUncaging_click(tiffpath,SampleImg=Spine_example,ShowPoint=False,ShowPoint_YX=[110,134])
    # Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray([r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20230508\Rab10CY_slice1_dend1_timelapse2_001.flim"],
    #                                                                    ch=0)
    # FirstStack=Tiff_MultiArray[0]
    # stack_array = np.array(imread(tiffpath))
    
    # tiffarray_to_PIL2(FirstStack,NthSlice=4)
    
    # z,y,x = threeD_array_click(FirstStack,SampleImg=Spine_example,ShowPoint=True,ShowPoint_YX=[110,134])
    # # transparent_tiffpath = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\micromanager\20230118\20230118_142934.tif"
    # # fluorescent_tiffpath = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\micromanager\20230118\20230118_143344_99.99norm.tif"
    # transparent_tiffpath = r"G:\ImagingData\Tetsuya\20231215\multiwell_tiling3\F2\tiled_img.tif"
    # fluorescent_tiffpath = r"G:\ImagingData\Tetsuya\20231215\multiwell_tiling3\F2\tiled_img.tif"
    # ShowPointsYXlist = TwoD_multiple_click(transparent_tiffpath, transparent_tiffpath, Text="Click")
    # print(ShowPointsYXlist)
    
    # z, ylist, xlist = multiple_uncaging_click(FirstStack,SampleImg=Spine_example,ShowPoint=False,ShowPoint_YX=[110,134])
    # print(z, ylist, xlist)   
    
    
    
if __name__=="__main__":
    # pass
    main()
    





def check_spinepos_analyzer():
    flimpath = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20230606\test2\pos3_high_001.flim"
    ch = 0
    Tiff_MultiArray, _, _ = flim_files_to_nparray([flimpath],ch=ch)
    ZYXarray = Tiff_MultiArray[0]
    maxproj = np.max(ZYXarray, axis = 0)
    z, ylist, xlist = read_multiple_uncagingpos(flimpath)
    ylist = list(map(int,ylist))
    xlist = list(map(int,xlist))
    
    txtpath = flimpath[:-5]+"dendrite.txt"
    direction_list, orientation_list, dendylist, dendxlist  = read_dendriteinfo(flimpath)
    
    dend_center_x = np.array(dendxlist) + np.array(xlist)
    dend_center_y = np.array(dendylist) + np.array(ylist)
    
    plt.imshow(maxproj,cmap="gray")
    plt.scatter(xlist,ylist,c='cyan',s=10)
    plt.scatter(dend_center_x,dend_center_y,c='r',s=10)
    
    for y0, x0, orientation in zip(dend_center_y, dend_center_x, orientation_list):
        x0,x1,x2,x1_1,x2_1,y0,y1,y2,y1_1,y2_1 = get_axis_position(y0,x0,orientation, HalfLen_c=10)
        plt.plot((x2_1, x2), (y2_1, y2), '-b', linewidth=2.5)
    
    plt.show()
    
