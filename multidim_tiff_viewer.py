# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 08:56:14 2022

@author: yasudalab
"""
import PySimpleGUI as sg
import numpy as np
from io import BytesIO
from PIL import Image
from tifffile import imread
import cv2
import base64

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


def first_tiff_read(tiffpath):
    stack_array = np.array(imread(tiffpath))

    if len(stack_array.shape)>3:
         raise TypeError("Only 2D or 3D tiff is allowed")
    else:
        return stack_array.shape

def tiff16bit_to_PIL(tiffpath,Percent=100,show_size_yx=[512,512],
                     return_ratio=False,NthSlice=1):
    stack_array = np.array(imread(tiffpath))
    if return_ratio==True:    
        im_PIL,resize_ratio_yx  = tiffarray_to_PIL(stack_array,Percent=Percent,show_size_yx=show_size_yx,
                                                   return_ratio=return_ratio,NthSlice=NthSlice)
        return im_PIL,resize_ratio_yx
    else:
        im_PIL = tiffarray_to_PIL(stack_array,Percent=Percent,show_size_yx=show_size_yx,
                                                   return_ratio=return_ratio,NthSlice=NthSlice)
        return im_PIL 

def tiffarray_to_PIL(stack_array,Percent=100,show_size_yx=[512,512],
                     return_ratio=False,NthSlice=1):
    if Percent<1 or Percent>100:
        Percent=100
        
    if len(stack_array.shape)== 3:
        im_array = stack_array[NthSlice-1,:,:]
    else:
        im_array = stack_array
    
    norm_array = (100/Percent)*255*(im_array/im_array.max())
    norm_array[norm_array>255]=255
    rgb_array = cv2.cvtColor(norm_array.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    im_PIL = Image.fromarray(rgb_array)
    
    im_PIL = im_PIL.resize(show_size_yx)
    resize_ratio_yx = (show_size_yx[0]/im_array.shape[0],show_size_yx[1]/im_array.shape[1])
    if return_ratio==True:    
        return im_PIL,resize_ratio_yx
    else:
        return im_PIL 


def threeD_array_click(stack_array,Text="Click",SampleImg=None,ShowPoint=False,ShowPoint_YX=[0,0]):
    TiffShape= stack_array.shape
    col_list=['red','cyan']
    
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        Z_change=[sg.Text("Z position", size=(20, 1)),
                  sg.Slider(orientation ='horizontal', key='Z',
                            default_value=int(NumOfZ/2), range=(1,NumOfZ),enable_events = True)            
                  ]
        im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array,Percent=100, show_size_yx=[512,512],
                                                  return_ratio=True,NthSlice=int(NumOfZ/2))
    else:
        NumOfZ = 1
        Z_change=[]
        im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array,Percent=100, show_size_yx=[512,512],
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
            im_PIL = tiffarray_to_PIL(stack_array,Percent=ShowIntensityMax,NthSlice=NthSlice)
            data =  PILimg_to_data(im_PIL)
            graph.erase()
            graph.draw_image(data=data, location=(0,512))
            graph.draw_point((x,y), size=5, color = col_list[0])
            if ShowPoint==True:
                graph.draw_point((x_2,y_2), size=5, color = col_list[1])

        if event ==  'Update' or 'Z' or "Intensity":
            im_PIL = tiffarray_to_PIL(stack_array,Percent=ShowIntensityMax,NthSlice=NthSlice)
            data =  PILimg_to_data(im_PIL)
            graph.draw_image(data=data, location=(0,512))
            graph.draw_point((x,y), size=5, color = col_list[0])
            if ShowPoint==True:
                graph.draw_point((x_2,y_2), size=5, color = col_list[1])

        if event ==  'OK':
            if x==-1:
                window["-INFO-"].update(value="Please click",text_color='red', background_color='white')
                continue
            else:
                z,y,x=NthSlice-1, y, x
                window.close()
                return z,int((512-y)/resize_ratio_yx[0]),int(x/resize_ratio_yx[1])
                break

def threeD_img_click(tiffpath,Text="Click",SampleImg=None,ShowPoint=False,ShowPoint_YX=[0,0]):
    TiffShape= first_tiff_read(tiffpath)
    col_list=['red','cyan']
    
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        Z_change=[sg.Text("Z position", size=(20, 1)),
                  sg.Slider(orientation ='horizontal', key='Z',
                            default_value=int(NumOfZ/2), range=(1,NumOfZ),enable_events = True)            
                  ]
        im_PIL,resize_ratio_yx = tiff16bit_to_PIL(tiffpath,Percent=100, show_size_yx=[512,512],
                                                  return_ratio=True,NthSlice=int(NumOfZ/2))
    else:
        NumOfZ = 1
        Z_change=[]
        im_PIL,resize_ratio_yx = tiff16bit_to_PIL(tiffpath,Percent=100, show_size_yx=[512,512],
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

def main():
    tiffpath=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20221215\Intensity\CAGGFP_Slice2_dendrite1__Ch1_018.tif"
    Spine_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Spine_example.png"
    Dendrite_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Dendrite_example.png"
    z,y,x = threeD_img_click(tiffpath,SampleImg=Spine_example,ShowPoint=True,ShowPoint_YX=[110,134])
    
    stack_array = np.array(imread(tiffpath))
    z,y,x = threeD_array_click(stack_array,SampleImg=Spine_example,ShowPoint=True,ShowPoint_YX=[110,134])

# print(z,y,x)