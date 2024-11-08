# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 08:56:14 2022

@author: yasudalab
"""
import math
import os
import copy
import PySimpleGUI as sg
import numpy as np
from io import BytesIO
from PIL import Image
from tifffile import imread
import cv2
import base64
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
    resize_ratio_yx = (show_size_xy[0]/im_array.shape[0],show_size_xy[1]/im_array.shape[1])
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

def calc_zoom_rate_based_on_maxsize(yx_shape: tuple, show_size_YX_max: int) -> tuple:
    candidate_yxshape = np.array(yx_shape)
    for axis_ind in range(len(candidate_yxshape)):
        if candidate_yxshape[axis_ind] > show_size_YX_max[axis_ind]:
            candidate_yxshape = (candidate_yxshape / candidate_yxshape[axis_ind]) * show_size_YX_max[axis_ind]
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
    
    TiffShape= stack_array.shape
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        Z_change=[sg.Text("Z position", size=(20, 1)),
                  sg.Slider(orientation ='horizontal', key='Z',
                            default_value=int(NumOfZ/2), range=(1,NumOfZ),enable_events = True)            
                  ]
        yx_shape = TiffShape[1:3]
        resize_yx_shape = calc_zoom_rate_based_on_maxsize(yx_shape, show_size_YX_max)
        im_PIL,resize_ratio_yx = tiffarray_to_PIL_8bit(stack_array,vmax = 255, show_size_yx = resize_yx_shape,
                                                  return_ratio=True,NthSlice=int(NumOfZ/2))
    else:
        NumOfZ = 1
        Z_change=[]
        resize_yx_shape = calc_zoom_rate_based_on_maxsize(TiffShape, show_size_YX_max)
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
                    sg.Text(show_text, font='Arial 10', size=(60, 1))
                    ],
                [
                    sg.Text(first_text_at_upper, key='notification', font='Arial 10', text_color='black', background_color='white', size=(60, 2))
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



def threeD_array_click(stack_array,Text="Click",SampleImg=None,
                       ShowPoint=False,ShowPoint_YX=[0,0]):
    
    TiffShape= stack_array.shape
    col_list=['red','cyan']
    
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
            [sg.Text(key='-INFO-', size=(60, 1)),sg.Button('OK', size=(20, 2))]
            ]
    
    window = sg.Window("Spine selection", layout, finalize=True)
    graph = window["-GRAPH-"]       # type: sg.Graph
    graph.draw_image(data=data, location=(0,512))
    
    window["-Sample-"].draw_image(data=data_sample, location=(0,512))
    
    if ShowPoint==True:
        col_list=['cyan','red']
        y = 512 - ShowPoint_YX[0]*resize_ratio_yx[0]
        x = ShowPoint_YX[1]*resize_ratio_yx[1]
        graph.draw_point((x,y), size=5, color = col_list[1])
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
                window["-INFO-"].update(value="Please click",text_color='red', background_color='white')
                continue
            else:
                z,y,x=NthSlice-1, y, x
                window.close()
                return z,int((512-y)/resize_ratio_yx[0]),int(x/resize_ratio_yx[1])
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
        
    x=-1;y=-1;fixedSlice=-1;

    NthSlice = 1
    fixZ = False
    
    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']

        if len(TiffShape)==3:
            if fixZ == False:
                NthSlice = int(values['Z'])
            else:
                NthSlice = fixedSlice          

        if event == sg.WIN_CLOSED:
            break

        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            im_PIL = tiffarray_to_PIL(stack_array,Percent=ShowIntensityMax,NthSlice=NthSlice)
            data =  PILimg_to_data(im_PIL)
            graph.erase()
            graph.draw_image(data=data, location=(0,512))
            graph.draw_point((x,y), size=5, color = col_list[0])

            if len(ShowPointsYXlist)>0:
                for EachYX in ShowPointsYXlist:
                    graph.draw_point((EachYX[1],EachYX[0]), size=5, color = col_list[1])

        if event ==  'Update' or 'Z' or "Intensity":
            im_PIL = tiffarray_to_PIL(stack_array,Percent=ShowIntensityMax,NthSlice=NthSlice)
            data =  PILimg_to_data(im_PIL)
            graph.draw_image(data=data, location=(0,512))
            graph.draw_point((x,y), size=5, color = col_list[0])
            if len(ShowPointsYXlist)>0:
                for EachYX in ShowPointsYXlist:
                    graph.draw_point((EachYX[1],EachYX[0]), size=5, color = col_list[1])

        if event ==  'Assign':
            z,y,x=NthSlice-1, y, x
            ShowPointsYXlist.append([y,x])
            fixZ = True
            fixedSlice = NthSlice
            
            
        if event ==  'Reset':
            ShowPointsYXlist = []
            fixZ = False
            
        if event ==  'OK':
            if x==-1:
                window["-INFO-"].update(value="Please click",text_color='red', background_color='white')
                continue
            else:
                z = fixedSlice-1
                window.close()
                xlist, ylist = [], []
                for EachYX in ShowPointsYXlist:
                    xlist.append(EachYX[1]/resize_ratio_yx[1])
                    ylist.append((512-EachYX[0])/resize_ratio_yx[0])

                return z,ylist,xlist
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
                    ShowPoint_YX=[0,0]):
    y_pix,x_pix= twoD_numpy.shape
    showratio = max(x_pix/max_img_xwidth, y_pix/max_img_ywidth)
    show_size_xy = [int(x_pix/showratio),int(y_pix/showratio) ]

    col_list=['magenta', 'green']
    ShowPointsYXlist = []
    
    im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array = twoD_numpy,
                                              Percent=100,
                                              show_size_xy=show_size_xy,
                                              return_ratio=True)
    
    showpoint_y = show_size_xy[1] - ShowPoint_YX[0] * resize_ratio_yx[0]
    showpoint_x = ShowPoint_YX[1] * resize_ratio_yx[1]
     
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
               sg.Text(key='-INFO-', size=(60, 1)), 
               sg.Button('Reset', size=(20, 2)),
               sg.Button('OK', size=(20, 2))
              ]
            ]
    
    window = sg.Window("Neuron selection", layout, finalize=True)
    graph = window["-GRAPH-"]       # type: sg.Graph
    graph.draw_image(data=data, location=(0,show_size_xy[1]))
    graph.draw_point((showpoint_x,showpoint_y), size=10, color = col_list[0])

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
        


def define_uncagingPoint_dend_click_multiple(flim_file_path):    
    iminfo = FileReader()
    print(flim_file_path)
    iminfo.read_imageFile(flim_file_path, True) 
    imagearray=np.array(iminfo.image)
    intensityarray=np.sum(np.sum(np.sum(imagearray,axis=-1),axis=1),axis=1)
    
    text = "Click the center of the spine you want to stimulate. (Not the uncaging position itself)"
    spine_zyx = threeD_array_click(intensityarray,text,
                             SampleImg=None,ShowPoint=False)
    maxproj = np.sum(intensityarray,axis=0)
    text2 = "Click the dendrite near the selected spine"
    yx_list = twoD_click_tiff(twoD_numpy = maxproj, Text=text2,
                              max_img_xwidth = 600, max_img_ywidth = 600, 
                              ShowPoint_YX=[spine_zyx[1],spine_zyx[2]])
    
    dend_slope, dend_intercept = np.polyfit(np.array(yx_list)[:,1], np.array(yx_list)[:,0], 1)

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
    
