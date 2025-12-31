import datetime
import math
import PySimpleGUI as sg
import numpy as np
from io import BytesIO
from PIL import Image
from tifffile import imread
import cv2

def PILimg_to_data(im):
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data    

def tiff_size_read(tiffpath):
    return imread(tiffpath).shape

def trim_area_return(twoD_tiffshape, 
                     center_yx_abs, 
                     nextzoom):
    yshape, xshape = twoD_tiffshape[0], twoD_tiffshape[1]
    
    trim_xsize = math.floor(xshape/nextzoom)
    trim_ysize = math.floor(yshape/nextzoom)
    half_trim_x = math.ceil(trim_xsize/2)  
    half_trim_y = math.ceil(trim_ysize/2)  
    
    trim_x_from = max(0, center_yx_abs[1] - half_trim_x)
    trim_x_to = min(xshape, trim_x_from + trim_xsize)
    if trim_x_to == xshape:
        trim_x_from = xshape - trim_xsize
    
    trim_y_from = max(0, center_yx_abs[0] - half_trim_y)
    trim_y_to = min(yshape, trim_y_from + trim_ysize)
    if trim_y_to == yshape:
        trim_y_from = yshape - trim_ysize
    next_x_range = [trim_x_from, trim_x_to]
    next_y_range = [trim_y_from, trim_y_to]
    return next_x_range, next_y_range



class ImageViewer():
    def __init__(self, ch1_tiffpath, ch2_tiffpath = ""):
        self.ch1_tiffpath = ch1_tiffpath
        if ch2_tiffpath == "":
            self.ch2_tiffpath = ch1_tiffpath
            self.num_ch = 1
        else:
            assert tiff_size_read(ch1_tiffpath)== tiff_size_read(ch2_tiffpath),\
                "Ch1 tiff and Ch2 tiff have different image shape"
            self.ch2_tiffpath = ch2_tiffpath
            self.num_ch = 2
            
        self.text = "click and 'Assign'"
        self.max_img_xwidth = 1000
        self.max_img_ywidth = 700
        self.zoom = 1
        self.minzoom = 1
        self.maxzoom = 20
        self.zoom_bin = 1
        self.zoom_samepos_millisec = 500
        self.ch1_array = np.array(imread(self.ch1_tiffpath))
        self.ch2_array = np.array(imread(self.ch2_tiffpath))
        self.plot_color = 'magenta'
        self.circle_size = 15
        self.plot_fontsize = 36
        self.plot_font_name = "Arial"
        self.plotfont = f"{self.plot_font_name} {self.plot_fontsize}"
        
    
    def click_start(self):
        y_pix, x_pix = tiff_size_read(self.ch1_tiffpath)
        self.showratio = max(x_pix/self.max_img_xwidth, y_pix/self.max_img_ywidth)
        self.show_size_xy = [int(x_pix/self.showratio),int(y_pix/self.showratio) ]
        
        return self.TwoD_multiple_click()
        
    
    def calc_absolute_pos_from_zoomed(self, 
                                      PILcoord_YX,
                                      return_int = True):
        # print("PILcoord_YX ", PILcoord_YX)
        
        original_Y = self.current_y_range[0] + \
            (self.current_y_range[1] - self.current_y_range[0])* \
                ((self.show_size_xy[1] - PILcoord_YX[0])/self.show_size_xy[1])
        
        original_X = self.current_x_range[0] + \
            (self.current_x_range[1] - self.current_x_range[0])* \
                ((PILcoord_YX[1])/self.show_size_xy[0])
                
        if return_int:
            original_Y = int(original_Y)
            original_X = int(original_X)
            
        # print("original YX", original_Y, original_X)
        return original_Y, original_X


    def calc_relative_pos_from_absolute(self, 
                                      original_Y, original_X,
                                      return_int = True):
        PILcoord_Y = int(
                        self.show_size_xy[1] - 
                        self.show_size_xy[1] * 
                        (original_Y -self.current_y_range[0])/
                        (self.current_y_range[1] - self.current_y_range[0])
                        )

        PILcoord_X = int(
                        (original_X - self.current_x_range[0])*
                        self.show_size_xy[0]/
                        (self.current_x_range[1] - self.current_x_range[0])
                        )
 
        return PILcoord_Y, PILcoord_X

    
    
    def return_zoomed_data(self):
        center_yx_abs = 1
        next_x_range, next_y_range = trim_area_return(self.ch1_array.shape, 
                                                     center_yx_abs, 
                                                     self.nextzoom)
        
    def trim_stack_to_PIL(self,
                         stack_array, center_yx):
        center_yx_abs = self.calc_absolute_pos_from_zoomed(center_yx)
        # print("center_yx, center_yx_abs ",center_yx, center_yx_abs)
        im_PIL = self.trim_stack_to_PIL_use_abs_pos(stack_array,center_yx_abs)
        return im_PIL
    
    def trim_stack_to_PIL_use_abs_pos(self,stack_array,center_yx_abs):
        next_x_range, next_y_range = \
            trim_area_return(twoD_tiffshape = stack_array.shape, 
                             center_yx_abs= center_yx_abs, 
                             nextzoom = self.nextzoom)
        self.trim_array = stack_array[next_y_range[0]:next_y_range[1],
                                      next_x_range[0]:next_x_range[1]]
        im_PIL  = self.tiffarray_to_PIL(self.trim_array)
        self.current_y_range = next_y_range
        self.current_x_range = next_x_range
        self.current_center_yx_abs = center_yx_abs
        return im_PIL



    def tiff16bit_to_PIL(self,
                         stack_array,
                         return_ratio=False,NthSlice=1):
        
        if return_ratio==True:    
            im_PIL,resize_ratio_yx = self.tiffarray_to_PIL(stack_array,
                                                       return_ratio=return_ratio,
                                                       NthSlice=NthSlice)
            return im_PIL,resize_ratio_yx
        else:
            im_PIL = self.tiffarray_to_PIL(stack_array,
                                           return_ratio=return_ratio,
                                           NthSlice=NthSlice)
            return im_PIL 
    
    def tiffarray_to_PIL(self,
                         stack_array,
                         return_ratio=False,NthSlice=1):
        # This can be used for two D image also.
        Percent = self.IntensityPercent
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
        
        im_PIL = im_PIL.resize(self.show_size_xy)
        resize_ratio_yx = (self.show_size_xy[1]/im_array.shape[0],self.show_size_xy[0]/im_array.shape[1])
        if return_ratio==True:    
            return im_PIL,resize_ratio_yx
        else:
            return im_PIL 
   

    def plot_assigned_points(self):
        roinum = 0
        for EachYX in self.ShowPointsAbsYXlist:
            roinum+=1
            PIL_Y, PIL_X = self.calc_relative_pos_from_absolute(EachYX[0],EachYX[1])
            if 0<PIL_Y<self.show_size_xy[1] and 0<PIL_X<self.show_size_xy[0]:                    
                text = str(roinum)
                self.graph.draw_text(text, (PIL_X,PIL_Y), 
                                font = self.plotfont,
                                color = self.plot_color,
                                text_location = "center")
    
    def update_image(self,  y = -100, x = -100, preassigned_center = False):
        if preassigned_center == True:
            im_PIL = self.trim_stack_to_PIL_use_abs_pos(stack_array = self.show_array, 
                                                        center_yx_abs= self.current_center_yx_abs)
        else:
            assert x > -1; assert y > -1; 
            im_PIL = self.trim_stack_to_PIL(stack_array = self.show_array, 
                                            center_yx= [y,x])
            
        data =  PILimg_to_data(im_PIL)
        self.graph.draw_image(data=data, location=(0,self.show_size_xy[1]))
        self.plot_assigned_points()
                
    def TwoD_multiple_click(self):
        sg.theme('DarkBlack1')
    
        layout = [
                  [
                    sg.Text(self.text, key='-Top-INFO-', font='Arial 14', 
                            text_color='black', background_color='white', size=(90, 2))
                  ],
                  [
                    sg.Graph(
                    canvas_size=(self.show_size_xy), 
                    graph_bottom_left=(0, 0),
                    graph_top_right=(self.show_size_xy),
                    key="-GRAPH-",
                    enable_events=True,background_color='lightblue',
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
                              default_value=1, range=(1,self.num_ch),enable_events = True)
                  ],
                  [
                   sg.Text(key='-INFO-', size=(30, 1)),
                   sg.Button('Assign', size=(10, 2)),
                   sg.Button('Zoom1x', size=(10, 2)),
                   sg.Button('Cancel last ROI', size=(10, 2)),
                   sg.Button('Reset', size=(10, 2)),
                   sg.Button('OK', size=(10, 2))
                  ]
                ]
        
        window = sg.Window("Neuron selection", layout, 
                           finalize=True)
        
        self.graph = window["-GRAPH-"]
        self.graph.bind('<MouseWheel>', '__SCROLL')
        
        self.ShowPointsAbsYXlist = []
        self.nextzoom = 1
        self.current_x_range = [0, self.ch1_array.shape[1]]
        self.current_y_range = [0, self.ch1_array.shape[0]]   
        self.current_center_yx_abs = [self.ch1_array.shape[0]//2, self.ch1_array.shape[1]//2]
        self.IntensityPercent = 100
        self.ch1_im_PIL, self.resize_ratio_yx = self.tiff16bit_to_PIL(self.ch1_array,
                                                                      return_ratio=True)
        data =  PILimg_to_data(self.ch1_im_PIL)
        self.graph.draw_image(data=data, location=(0,self.show_size_xy[1]))
        x=-1;y=-1
        lastzoom = datetime.datetime.now() - datetime.timedelta(seconds = self.zoom_samepos_millisec/1000)
        
        while True:
            event, values = window.read()
            self.IntensityPercent = values['Intensity']
            self.NthCh = int(values['Ch'])
            
            # print(event, values)
            
            if self.NthCh == 1:
                self.show_array = self.ch1_array
            else:
                self.show_array = self.ch2_array
            
            if event == sg.WIN_CLOSED:
                return self.ShowPointsAbsYXlist
                break
            
            if event == "-GRAPH-__SCROLL":
                if self.graph.user_bind_event.delta > 0:
                    self.nextzoom = min(self.maxzoom, self.zoom + self.zoom_bin)
                elif self.graph.user_bind_event.delta < 0:
                    self.nextzoom = max(self.minzoom, self.zoom - self.zoom_bin)

                millisec_after_lastzoom = (1000*(datetime.datetime.now() - lastzoom).seconds 
                                      + 0.001*(datetime.datetime.now() - lastzoom).microseconds)
                
                if millisec_after_lastzoom > self.zoom_samepos_millisec:
                    x, y = values["-GRAPH-"]
                    window['-Top-INFO-'].update(f"Zoom {self.nextzoom}x")
                else:
                    x, y = self.show_size_xy[0]//2, self.show_size_xy[1]//2
                    window['-Top-INFO-'].update(f"Zoom {self.nextzoom}x, without changing center\n"+
                                                f"because you changed zoom within {self.zoom_samepos_millisec} ms after the last change")
                    
                self.update_image(y, x)
                # im_PIL = self.trim_stack_to_PIL(stack_array = self.show_array, 
                #                                 center_yx= [y,x])
                # data =  PILimg_to_data(im_PIL)
                # self.graph.draw_image(data=data, location=(0,self.show_size_xy[1]))
                # self.plot_assigned_points()
                self.zoom = self.nextzoom
                lastzoom = datetime.datetime.now()
            
            if event == "Zoom1x":
                self.nextzoom = 1
                im_PIL = self.trim_stack_to_PIL(stack_array = self.show_array, 
                                                center_yx= [y,x])
                data =  PILimg_to_data(im_PIL)
                window['-Top-INFO-'].update(f"Zoom {self.nextzoom}x")
                self.graph.draw_image(data=data, location=(0,self.show_size_xy[1]))
                self.plot_assigned_points()
                self.zoom = self.nextzoom  
    
            if event == "-GRAPH-":
                x, y = values["-GRAPH-"]
                self.update_image(y, x)
                plot_radius = min(self.circle_size * self.zoom, min(self.show_size_xy)/2.3)
                self.graph.draw_circle((x,y), radius=plot_radius,
                                  fill_color=None, line_color = self.plot_color,
                                  line_width=1)
                
            if event ==  "Intensity" or event == "Ch":
                self.update_image(preassigned_center = True)
                window['-INFO-'].update(f"The number of ROI(s): {len(self.ShowPointsAbsYXlist )}")
                
            if event ==  'Assign':
                if x == -1:
                    window['-Top-INFO-'].update("Click on the image before 'Assign'")
                    continue
                
                original_Y, original_X = self.calc_absolute_pos_from_zoomed([y,x])
                if [original_Y, original_X] in self.ShowPointsAbsYXlist:
                    window['-Top-INFO-'].update("Do not assign exactly same position")
                    continue
                
                self.ShowPointsAbsYXlist.append([original_Y, original_X])
                print(self.ShowPointsAbsYXlist)
                window['-INFO-'].update(f"The number of ROI(s): {len(self.ShowPointsAbsYXlist )}")
                self.update_image(y, x, preassigned_center = True)
                # self.graph.draw_image(data=data, location=(0,self.show_size_xy[1]))
                # self.plot_assigned_points()
                window['-Top-INFO-'].update(f"ROI {len(self.ShowPointsAbsYXlist)} assigned")
                x=-1;y=-1
                
            if event ==  'Reset':
                self.ShowPointsAbsYXlist  = []
                self.update_image(y, x, preassigned_center = False)
                # self.graph.draw_image(data=data, location=(0,self.show_size_xy[1]))
                window['-INFO-'].update(f"The number of ROI(s): {len(self.ShowPointsAbsYXlist)}")
                window['-Top-INFO-'].update("Reset")
            
            if event ==  'Cancel last ROI':
                if len(self.ShowPointsAbsYXlist)<1:
                    window['-Top-INFO-'].update("ROI has not yet been assigned")
                else:
                    window['-Top-INFO-'].update("Cancel the last ROI assignment")
                    _ = self.ShowPointsAbsYXlist.pop()
                    
                    self.update_image(y, x, preassigned_center = False)
                    # self.graph.draw_image(data=data, location=(0,self.show_size_xy[1]))
                    # self.plot_assigned_points()
                    window['-INFO-'].update(f"The number of ROI(s): {len(self.ShowPointsAbsYXlist)}")
                
            if event ==  'OK':
                window.close()
                return self.ShowPointsAbsYXlist

if __name__=="__main__":
    # fluorescent_tiffpath = r"C:\Users\WatabeT\Desktop\tiled_img.tif"
    # ch2 = r"C:\Users\WatabeT\Desktop\rotate.tif"
    fluorescent_tiffpath = r"G:\ImagingData\Tetsuya\20240613\96well\B5\tiled_img.tif"
    ch2 = r"G:\ImagingData\Tetsuya\20240613\96well\B5\tiled_img.tif"
    
    ClickImage = ImageViewer(fluorescent_tiffpath,ch2)
    ShowPointsYXlist_original_coord = ClickImage.click_start()
    print(ShowPointsYXlist_original_coord)
    