import math
import os
import numpy as np
from io import BytesIO
from PIL import Image
from tifffile import imread
import cv2
import base64
from FLIMageAlignment import flim_files_to_nparray
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import TkEasyGUI as tg
from FLIMageFileReader2 import FileReader
import configparser

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
    if os.path.exists(inipath) == False:
        print(inipath, "do not exist")
    assert(os.path.exists(inipath))
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
    halfsize = min(show_size_YX_min)//8  # Define halfsize at the top level
    
    TiffShape = stack_array.shape
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        Z_change=[tg.Text("Z position", size=(20, 1)),
                  tg.Slider(orientation ='horizontal', key='Z',
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
    data = PILimg_to_data(im_PIL)    

    layout = [
                [
                    tg.Text(show_text, font='Arial 8', size=(90, 1))
                    ],
                [
                    tg.Text(first_text_at_upper, key='notification', font='Arial 10', 
                            text_color='black', background_color='white', size=(60, 2))
                    ],
                [
                    tg.Graph(
                    canvas_size=resize_yx_shape[::-1], 
                    graph_bottom_left=(0, 0),
                    graph_top_right=resize_yx_shape[::-1],
                    key="-GRAPH-",
                    background_color='lightblue'
                    )
                    ],
                [
                    tg.Text("Intensity", size=(20, 1)),
                    tg.Slider(orientation ='horizontal', key='Intensity',default_value=stack_array.max(),
                         range=(1,255),enable_events = True),
                    ],
                Z_change,
                [
                    tg.Text("Assigned pos", size=(20, 1)),
                    tg.Slider(orientation ='horizontal', key='pos',default_value=0,
                         range=(0,20),enable_events = True),
                    ],
                [
                    tg.Text(key='-INFO-', size=(30, 1)),
                    tg.Button('Assign', size=(20, 2)),
                    tg.Button('Reset', size=(20, 2)),
                    tg.Button('OK', size=(20, 2))
                    ]
            ]
    
    window = tg.Window("Z stack viewer", layout, finalize=True)
    graph = window["-GRAPH-"]

    # Initialize variables
    x = -1
    y = -1
    NthSlice = int(NumOfZ/2)
    ShowIntensityMax = stack_array.max()
    current_pos = 0  # Track position separately

    # Bind mouse click event
    def on_click(event):
        nonlocal x, y
        x, y = event.x, event.y
        redraw_all()
        
    graph.widget.bind('<Button-1>', on_click)

    def draw_point_with_text(x, y, text, color, size=5):
        """Helper function to draw a point with text safely"""
        try:
            graph.draw_point((float(x), float(y)), size=size, color=color)
            if text:
                text_x = max(float(x) - float(halfsize)*0.90, resize_yx_shape[1]*0.007)
                text_y = min(float(y) + float(halfsize)*0.80, resize_yx_shape[0]*0.990)
                graph.draw_text(str(text), (text_x, text_y), font=font, color=color)
        except (ValueError, TypeError) as e:
            print(f"Error drawing point: {e}")

    def redraw_all():
        im_PIL = tiffarray_to_PIL_8bit(stack_array, vmax=ShowIntensityMax, show_size_yx=resize_yx_shape,
                                      return_ratio=False, NthSlice=NthSlice)
        data = PILimg_to_data(im_PIL)
        graph.erase()
        graph.draw_image(data=data, location=(0, 0))
        
        # Draw current point
        if x != -1:
            draw_point_with_text(x, y, None, col_dict["clicked_now"])
            
        # Draw all assigned points
        for nth, EachZYX in enumerate(ZYX_pixel_clicked_list):
            try:
                if EachZYX[0] == NthSlice - 1:
                    color = col_dict["clicked_currentZ"]
                    if current_pos > 0 and nth == current_pos - 1:
                        color = col_dict["pos_now"]
                    # Draw rectangle
                    x1, y1 = float(EachZYX[2]-halfsize), float(EachZYX[1]-halfsize)
                    x2, y2 = float(EachZYX[2]+halfsize), float(EachZYX[1]+halfsize)
                    graph.draw_rectangle((x1, y1), (x2, y2), line_color=color)
                    # Draw point number
                    draw_point_with_text(EachZYX[2], EachZYX[1], nth+1, color)
                else:
                    # Draw only point number for points in different Z planes
                    draw_point_with_text(EachZYX[2], EachZYX[1], nth+1, col_dict["clicked_differentZ"])
            except (ValueError, TypeError) as e:
                print(f"Error drawing point {nth}: {e}")

    # Initial draw
    redraw_all()

    while True:
        event, values = window.read()
        if values:  # Update values only if we got new values
            try:
                ShowIntensityMax = float(values['Intensity'])
                if len(TiffShape)==3:
                    NthSlice = int(values['Z'])
                if 'pos' in values:
                    current_pos = int(float(values['pos']))  # Convert through float to handle slider values
            except (ValueError, TypeError):
                continue  # Skip invalid values

        if event == tg.WIN_CLOSED:
            break
        
        if event == "pos":
            if current_pos < 1:
                continue
            if len(ZYX_pixel_clicked_list) >= current_pos:
                nthpos = current_pos - 1
                Z = ZYX_pixel_clicked_list[nthpos][0] + 1
                window['Z'].update(Z)
                NthSlice = Z
                window['notification'].update(f"Go to pos id: {current_pos}")
            else:
                window['notification'].update(f"pos id: {current_pos} is not defined.")
            
        if event == 'Assign':
            z,y,x = NthSlice - 1, y, x
            ZYX_pixel_clicked_list.append([z, y, x])
            x=-1; y=-1
            
        if event in ['Update', 'Z', "Intensity", 'Assign', "pos"]:
            redraw_all()
            
        if event == 'Reset':
            ZYX_pixel_clicked_list = []
            current_pos = 0
            redraw_all()
            
        if event == 'OK':
            window.close()
            pix_zyx_list = []
            for each_zyx in ZYX_pixel_clicked_list:
                each_z_pix = each_zyx[0]
                each_y_pix = round(each_zyx[1]/resize_ratio_yx[0])
                each_x_pix = round(each_zyx[2]/resize_ratio_yx[1])
                pix_zyx_list.append([each_z_pix,
                                     each_y_pix,
                                     each_x_pix])
            break
        
    return pix_zyx_list

def threeD_array_click(stack_array,Text="Click",SampleImg=None,
                       ShowPoint=False,ShowPoint_YX=[0,0],
                       predefined = False, predefined_ZYX = [0,0,0]):
    TiffShape= stack_array.shape
    col_list=['red','cyan']
    
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        Z_change=[tg.Text("Z position", size=(20, 1)),
                  tg.Slider(orientation ='horizontal', key='Z',
                            default_value=int(NumOfZ/2), range=(1,NumOfZ),enable_events = True)            
                  ]
        im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array,Percent=100, show_size_xy=[512,512],
                                                  return_ratio=True,NthSlice=int(NumOfZ/2))
    else:
        NumOfZ = 1
        Z_change=[]
        im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array,Percent=100, show_size_xy=[512,512],
                                                  return_ratio=True,NthSlice=1)

    data = PILimg_to_data(im_PIL)    
    data_sample = OpenPNG(pngpath=SampleImg)

    layout = [
                [tg.Text(Text, font='Arial 10', text_color='black', background_color='white', size=(60, 2))],
                [
                tg.Graph(
                canvas_size=(512, 512), 
                graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-GRAPH-",
                background_color='lightblue'
                ),
                tg.Graph(
                canvas_size=(512, 512), 
                graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-Sample-",
                background_color='black'
                )
                ],
              [
               tg.Text("Contrast", size=(20, 1)),
               tg.Slider(orientation ='horizontal', key='Intensity',default_value=100,
                         range=(1,100),enable_events = True),
              ],
               Z_change
              ,
            [tg.Text(key='-INFO-', size=(60, 1)),tg.Button('OK', size=(20, 2))]
            ]
    
    window = tg.Window("Spine selection", layout, finalize=True)
    graph = window["-GRAPH-"]
    graph_sample = window["-Sample-"]
    
    # Initialize variables
    ShowIntensityMax = 100
    NthSlice = int(NumOfZ/2)
    x = -1
    y = -1
    
    # Draw initial images
    graph.draw_image(data=data, location=(0, 0))
    if data_sample:
        graph_sample.draw_image(data=data_sample, location=(0, 0))
    
    if ShowPoint==True:
        col_list=['cyan','red']
        y_2 = 512 - ShowPoint_YX[0]*resize_ratio_yx[0]
        x_2 = ShowPoint_YX[1]*resize_ratio_yx[1]
        graph.draw_point((x_2,y_2), size=5, color = col_list[1])
    
    # Bind mouse click event
    def on_click(event):
        nonlocal x, y
        x, y = event.x, event.y
        im_PIL = tiffarray_to_PIL(stack_array,Percent=ShowIntensityMax,NthSlice=NthSlice)
        data = PILimg_to_data(im_PIL)
        graph.erase()
        graph.draw_image(data=data, location=(0, 0))
        graph.draw_point((x,y), size=5, color = col_list[0])
        if ShowPoint==True:
            graph.draw_point((x_2,y_2), size=5, color = col_list[1])
            
    graph.widget.bind('<Button-1>', on_click)
    
    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']

        if len(TiffShape)==3:
            NthSlice = int(values['Z'])

        if event == tg.WIN_CLOSED:
            break

        if event in ['Update', 'Z', "Intensity"]:
            im_PIL = tiffarray_to_PIL(stack_array,Percent=ShowIntensityMax,NthSlice=NthSlice)
            data = PILimg_to_data(im_PIL)
            graph.draw_image(data=data, location=(0, 0))
            if x != -1:
                graph.draw_point((x,y), size=5, color = col_list[0])
            if ShowPoint==True:
                graph.draw_point((x_2,y_2), size=5, color = col_list[1])

        if event == 'OK':
            if x==-1:
                window["-INFO-"].update(value="Please click")
                continue
            else:
                z,y,x=NthSlice-1, y, x
                window.close()
                return z,int((y)/resize_ratio_yx[0]),int(x/resize_ratio_yx[1])
                break

def multiple_uncaging_click(stack_array, Text="Click", SampleImg=None, ShowPoint=False, ShowPoint_YX=[0,0]):
    TiffShape = stack_array.shape
    col_list = ['red', 'cyan']
    ShowPointsYXlist = []
    
    if len(TiffShape)==3:
        NumOfZ = TiffShape[0]
        Z_change=[tg.Text("Z position", size=(20, 1)),
                  tg.Slider(orientation ='horizontal', key='Z',
                            default_value=int(NumOfZ/2), range=(1,NumOfZ),enable_events = True)            
                  ]
        im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array,Percent=100, show_size_xy=[512,512],
                                                  return_ratio=True,NthSlice=int(NumOfZ/2))
    else:
        NumOfZ = 1
        Z_change=[]
        im_PIL,resize_ratio_yx = tiffarray_to_PIL(stack_array,Percent=100, show_size_xy=[512,512],
                                                  return_ratio=True,NthSlice=1)

    data = PILimg_to_data(im_PIL)    
    data_sample = OpenPNG(pngpath=SampleImg)

    layout = [
                [tg.Text(Text, font='Arial 10', text_color='black', background_color='white', size=(60, 2))],
                [
                tg.Graph(
                canvas_size=(512, 512), 
                graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-GRAPH-",
                background_color='lightblue'
                ),
                tg.Graph(
                canvas_size=(512, 512), 
                graph_bottom_left=(0, 0),
                graph_top_right=(512, 512),
                key="-Sample-",
                background_color='black'
                )
                ],
              [
               tg.Text("Contrast", size=(20, 1)),
               tg.Slider(orientation ='horizontal', key='Intensity',default_value=100,
                         range=(1,100),enable_events = True),
              ],
               Z_change
              ,
            [tg.Text(key='-INFO-', size=(60, 1)),tg.Button('Assign', size=(20, 2)),tg.Button('Reset', size=(20, 2)),tg.Button('OK', size=(20, 2))]
            ]
    
    window = tg.Window("Spine selection", layout, finalize=True)
    graph = window["-GRAPH-"]
    graph_sample = window["-Sample-"]
    
    # Initialize variables
    ShowIntensityMax = 100
    NthSlice = int(NumOfZ/2)
    x = -1
    y = -1
    fixZ = False
    fixedSlice = -1
    
    # Draw initial images
    graph.draw_image(data=data, location=(0, 0))
    if data_sample:
        graph_sample.draw_image(data=data_sample, location=(0, 0))
    
    if len(ShowPointsYXlist)>0:
        col_list=['cyan','red']
        y_2 = 512 - ShowPoint_YX[0]*resize_ratio_yx[0]
        x_2 = ShowPoint_YX[1]*resize_ratio_yx[1]
        graph.draw_point((x_2,y_2), size=5, color = col_list[1])
    
    def redraw_all():
        im_PIL = tiffarray_to_PIL(stack_array,Percent=ShowIntensityMax,NthSlice=NthSlice)
        data = PILimg_to_data(im_PIL)
        graph.erase()
        graph.draw_image(data=data, location=(0, 0))
        
        # Draw current point
        if x != -1:
            graph.draw_point((x,y), size=5, color = col_list[0])
            
        # Draw all assigned points
        if len(ShowPointsYXlist)>0:
            for EachYX in ShowPointsYXlist:
                graph.draw_point((EachYX[1],EachYX[0]), size=5, color = col_list[1])
    
    # Bind mouse click event
    def on_click(event):
        nonlocal x, y
        x, y = event.x, event.y
        redraw_all()
            
    graph.widget.bind('<Button-1>', on_click)
    
    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']

        if len(TiffShape)==3:
            if not fixZ:
                NthSlice = int(values['Z'])
            else:
                NthSlice = fixedSlice

        if event == tg.WIN_CLOSED:
            break

        if event == 'Assign':
            z,y,x = NthSlice-1, y, x
            ShowPointsYXlist.append([y,x])
            fixZ = True
            fixedSlice = NthSlice
            
        if event == 'Reset':
            ShowPointsYXlist = []
            fixZ = False
            
        if event in ['Update', 'Z', "Intensity", 'Assign']:
            redraw_all()
            
        if event == 'OK':
            if len(ShowPointsYXlist)<1:
                window["-INFO-"].update(value="Please click")
                continue
            else:                
                window.close()
                xlist, ylist = [], []
                for EachYX in ShowPointsYXlist:
                    xlist.append(EachYX[1]/resize_ratio_yx[1])
                    ylist.append(EachYX[0]/resize_ratio_yx[0])
                return z,ylist,xlist
                break
                

def twoD_click_tiff(twoD_numpy, Text="Click",
                    max_img_xwidth = 600, max_img_ywidth = 600, 
                    ShowPoint_YX=[0,0],
                    predefined = False, predefied_yx_list = []):
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
     
    data = PILimg_to_data(im_PIL)    

    layout = [
                [tg.Text(Text, font='Arial 10', text_color='black', background_color='white', size=(60, 2))],
                [
                tg.Graph(
                canvas_size=show_size_xy, 
                graph_bottom_left=(0, 0),
                graph_top_right=show_size_xy,
                key="-GRAPH-",
                background_color='lightblue'
                )
                ],
              [
               tg.Text("Contrast", size=(20, 1)),
               tg.Slider(orientation ='horizontal', key='Intensity',default_value=100,
                         range=(1,100),enable_events = True),
              ],
              [
               tg.Text(key='-INFO-', size=(60, 1)), 
               tg.Button('Reset', size=(20, 2)),
               tg.Button('OK', size=(20, 2))
              ]
            ]
    
    window = tg.Window("Neuron selection", layout, finalize=True)
    graph = window["-GRAPH-"]
    
    # Initialize variables
    ShowIntensityMax = 100
    x = -1
    y = -1
    
    # Draw initial images
    graph.draw_image(data=data, location=(0, 0))
    graph.draw_point((showpoint_x,showpoint_y), size=10, color = col_list[0])
    if len(ShowPointsYXlist)>0:
        for EachYX in ShowPointsYXlist:
            graph.draw_point((EachYX[1],EachYX[0]), size=5, color = col_list[1])
    
    def redraw_all():
        im_PIL = tiffarray_to_PIL(stack_array = twoD_numpy,
                                  Percent=ShowIntensityMax,
                                  show_size_xy=show_size_xy,
                                  return_ratio=False)
        data = PILimg_to_data(im_PIL)
        graph.erase()
        graph.draw_image(data=data, location=(0, 0))
        graph.draw_point((showpoint_x,showpoint_y), size=10, color = col_list[0])
        
        # Draw all assigned points
        if len(ShowPointsYXlist)>0:
            for EachYX in ShowPointsYXlist:
                graph.draw_point((EachYX[1],EachYX[0]), size=5, color = col_list[1])
    
    # Bind mouse click event
    def on_click(event):
        nonlocal x, y
        x, y = event.x, event.y
        ShowPointsYXlist.append([y, x])
        redraw_all()
            
    graph.widget.bind('<Button-1>', on_click)
    
    while True:
        event, values = window.read()
        ShowIntensityMax = values['Intensity']

        if event == tg.WIN_CLOSED:
            break

        if event == 'Reset':
            ShowPointsYXlist = []
            redraw_all()
            
        if event in ['Update', "Intensity"]:
            redraw_all()
            
        if event == 'OK':
            if len(ShowPointsYXlist)<1:
                window["-INFO-"].update(value="Please click")
                continue
            else:                
                window.close()
                yx_list = []
                for EachYX in ShowPointsYXlist:
                    each_yx = [(show_size_xy[1]-EachYX[0])/resize_ratio_yx[0],
                               EachYX[1]/resize_ratio_yx[1]]
                    yx_list.append(each_yx)
                return yx_list
                break



def define_uncagingPoint_dend_click_multiple(flim_file_path,
                                             read_ini = False,
                                             inipath = "",
                                             SampleImg = None,
                                             only_for_exclusion = False):    
    iminfo = FileReader()
    print(flim_file_path)
    iminfo.read_imageFile(flim_file_path, True)
    imagearray=np.array(iminfo.image)
    intensityarray=np.sum(np.sum(np.sum(imagearray,axis=-1),axis=1),axis=1)
    text = "Click the center of the spine you want to stimulate. (Not the uncaging position itself)"
    maxproj = np.sum(intensityarray,axis=0)
    text2 = "Click the dendrite near the selected spine"

    if (read_ini == True) * (os.path.exists(inipath)):
        spine_zyx, dend_slope, dend_intercept = read_xyz_single(inipath = inipath)
        spine_zyx = threeD_array_click(intensityarray, text,
                                 SampleImg = SampleImg, ShowPoint=False,
                                 predefined = True, predefined_ZYX = spine_zyx)
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
                                  predefied_yx_list = predefied_yx_list)
    else:
        spine_zyx = threeD_array_click(intensityarray,text,
                                 SampleImg=SampleImg,ShowPoint=False)
        if spine_zyx[0] < 0:
            return spine_zyx, 0, 0
            
        yx_list = twoD_click_tiff(twoD_numpy = maxproj, Text=text2,
                                  max_img_xwidth = 600, max_img_ywidth = 600,
                                  ShowPoint_YX=[spine_zyx[1],spine_zyx[2]])

    dend_slope, dend_intercept = np.polyfit(np.array(yx_list)[:,1], np.array(yx_list)[:,0], 1)
    print("dend_slope, dend_intercept :", dend_slope, dend_intercept)
    return spine_zyx, dend_slope, dend_intercept



def test_all_click_functions():
    """Test function to compare all TkEasyGUI versions of click functions"""
    # Load test data
    flim_file_path = r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging\20250520\lowmag1__highmag_1_003.flim"
    Tiff_MultiArray, iminfo, relative_sec_list = flim_files_to_nparray([flim_file_path], ch=1)
    FirstStack = Tiff_MultiArray[0]
    
    # Test multiple_uncaging_click
    print("\n=== Testing multiple_uncaging_click ===")
    z1, ylist1, xlist1 = multiple_uncaging_click(FirstStack, Text="Click points")
    print(f"Results - Z: {z1}, Y points: {ylist1}, X points: {xlist1}")
    
    # Test z_stack_multi_z_click
    print("\n=== Testing z_stack_multi_z_click ===")
    pix_zyx_list1 = z_stack_multi_z_click(FirstStack, show_text="Click points")
    print(f"Results - Points: {pix_zyx_list1}")
        
    # Test threeD_img_click with a sample tiff file
    print("\n=== Testing threeD_img_click ===")
    z3, y3, x3 = threeD_array_click(FirstStack, Text="Click")
    print(f" Results - Z: {z3}, Y: {y3}, X: {x3}")


if __name__=="__main__":
    # Test all click functions
    test_all_click_functions()
    



