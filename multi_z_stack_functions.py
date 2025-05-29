# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:59:28 2024

@author: yasudalab
"""

from time import sleep
import datetime
import os
from pathlib import Path
import glob
import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
import json
import winsound
import tifffile
import numpy as np
from click_image_multiple_scroll import ImageViewer
from after_click_image_func import get_abs_mm_pos_from_click_list, \
    export_pos_dict_as_csv, save_image_with_assigned_pos, get_abs_mm_pos_3d_from_click_list, \
    save_pix_pos_from_click_list, save_image_with_assigned_pos_3d, get_ZYX_pix_list_from_csv, \
    get_abs_um_pos_from_center_3d, save_um_pos_from_click_list
from multidim_tiff_viewer import z_stack_multi_z_click
from travelling_salesman import nearest_neighbor
from FLIMageFileReader2 import FileReader


def add_header_in_posid(header, poslist):
    result_poslist = []
    for eachpos_oneline in poslist:
        pos_id, x_mm, y_mm, z_mm = get_xyz_mm_from_oneline(eachpos_oneline)
        pos_id_with_header = header + "_" + pos_id
        each_pos_oneline_with_header = f"{pos_id_with_header}, 0, {x_mm}, {y_mm}, {z_mm}"
        result_poslist.append(each_pos_oneline_with_header)
    return result_poslist
        
def get_xyz_mm_from_oneline(eachpos_oneline):
    with_id_objxyz = eachpos_oneline
    pos_id = with_id_objxyz.split(",")[0]
    objxyzstr = with_id_objxyz[with_id_objxyz.find(",")+1:]
    x_mm, y_mm, z_mm = [float(val) for val in objxyzstr.split(",")[1:4]] 
    return pos_id, x_mm, y_mm, z_mm

def make_poslist_from_dict(modified_posdict):
    poslist = []
    for each_posid in modified_posdict:
        x_mm = modified_posdict[each_posid]["x_mm"]
        y_mm = modified_posdict[each_posid]["y_mm"]
        z_mm = modified_posdict[each_posid]["z_mm"]
        poslist.append(f"{each_posid}, 0, {x_mm}, {y_mm}, {z_mm}")
    return poslist

def save_poslist_as_csv(csv_path, poslist):
    first_line = 'pos_id, obj_pos, x_pos_mm, y_pos_mm, z_pos_mm'
    result = ""
    result += first_line + "\r"
    for eachline in poslist:
        result += eachline + "\r"
    f = open(csv_path, "w")
    f.write(result)
    f.close()
    
def save_poslist_tsp_calc(csv_path, poslist):
    first_line = 'pos_id, obj_pos, x_pos_mm, y_pos_mm, z_pos_mm'
    result = ""
    result += first_line + "\r"
    zyx_list = []
    for eachpos_oneline in poslist:
        _, x_mm, y_mm, z_mm = get_xyz_mm_from_oneline(eachpos_oneline)
        zyx_list.append([z_mm, y_mm, x_mm])
    better_order = nearest_neighbor(zyx_list)
    for nth in better_order:
        result += poslist[nth] + "\r"
    f = open(csv_path, "w")
    f.write(result)
    f.close()
    
def hardware_z_focus_setting(FLIMageCont, Zfocus_laser_pow, led_comport):
    FLIMageCont.flim.sendCommand("State.Acq.zoom = 100")
    FLIMageCont.flim.sendCommand("State.Acq.XOffset = 0")
    FLIMageCont.flim.sendCommand("State.Acq.YOffset = 0")
    FLIMageCont.set_laser1_pow(Zfocus_laser_pow)
    FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 1")
    FLIMageCont.flim.sendCommand('MotorDisconnect')
    dict_command_and_sleeptime={
                        "LED x=0 y=0":0.1,
                        "MOVE Z=1":1,
                        "MOVE F=2":1
                      }
    sendcommand_list(led_comport, dict_command_and_sleeptime)

def sleep_countdown(sleep_sec, 
                    define_starttime = False,
                    start_time = -1):
    count_dict = {301:20,
                  61:10,
                  16:5,
                  0:1}
    
    if define_starttime == True:
        if type(start_time) != datetime.datetime:
            print("\n\n  please use datetime.datetime type for start_time. \n")
            start_time = datetime.datetime.now()
    else:
        start_time = datetime.datetime.now()
    while True:
        now = datetime.datetime.now()
        elapsed_time_sec = (now - start_time).total_seconds()
        if elapsed_time_sec >= sleep_sec:
            break
        remaining_sec = sleep_sec - elapsed_time_sec
        print(int(remaining_sec), end = " ")
        for each_threshold_sec in count_dict:            
            if remaining_sec > each_threshold_sec:
                sleep(count_dict[each_threshold_sec])
                break

def beep(frequency, duration):
    winsound.Beep(frequency, duration)
    
def nearest_tone(freq):
    C2 = 131
    octave = 5
    tonelist = [int(C2*(2**(1/12))**i) for i in range(octave*12)]   
    takeClosest = lambda num,tonelist:min(tonelist,key=lambda x:abs(x-num))
    return takeClosest(freq, tonelist)

def surface_difference_beep(array):
    lower = np.percentile(array,10)
    C5  = 523
    beep(C5, 500)
    if surface_found(array):
        times = 4
    else:
        times = 1 + (array.max()/lower)/10
    beep(nearest_tone(C5 * times), 500)

def surface_found(array) -> bool:
    lower = np.percentile(array,10)
    times = array.max()/lower
    if times > 10:
        return True
    else:
        return False
   
def read_obj_diff_json(obj_setting_json_path):
    with open(obj_setting_json_path, "r") as f:
        obj_diff_dict = json.load(f)
    obj_diff_onetenth_um = {}
    obj_diff_onetenth_um["x"] = obj_diff_dict["x_obj_diff"]
    obj_diff_onetenth_um["y"] = obj_diff_dict["y_obj_diff"]
    obj_diff_onetenth_um["z"] = obj_diff_dict["z_obj_diff"]
    return obj_diff_onetenth_um


def convert_pos_low_to_high(lowmag_pos_csv_path,
                            highmag_pos_save_path,
                            obj_setting_json_path = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\ongoing\ASIcontroller\objective_diff_settings\obj_setting.json",
                            ):
    convert_position_using_json(sourse_pos_csv_path = lowmag_pos_csv_path,
                                result_pos_save_path = highmag_pos_save_path,
                                obj_setting_json_path = obj_setting_json_path)
    
    
def convert_position_using_json(sourse_pos_csv_path,
                            result_pos_save_path,
                            obj_setting_json_path = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\ongoing\ASIcontroller\objective_diff_settings\obj_setting.json",
                            ):
    obj_diff_onetenth_um = read_obj_diff_json(obj_setting_json_path)
    source_poslist = import_from_csv(sourse_pos_csv_path)
    obj_diff_mm = {}
    for each_ax in ["x", "y", "z"]:
        obj_diff_mm[each_ax] = obj_diff_onetenth_um[each_ax]/(10**4)
    result_pos_list = []
    for one_line_pos in source_poslist:
        each_posid, low_x_mm, low_y_mm, low_z_mm = get_xyz_mm_from_oneline(one_line_pos)
        x_mm = round(low_x_mm - obj_diff_mm["x"], 4)
        y_mm = round(low_y_mm - obj_diff_mm["y"], 4)
        z_mm = round(low_z_mm - obj_diff_mm["z"], 4)
        result_pos_list.append(f"{each_posid}, 0, {x_mm}, {y_mm}, {z_mm}")        
    save_poslist_as_csv(result_pos_save_path, result_pos_list)

def surface_difference_beep_notify_error(array):
    lower = np.percentile(array,10)
    times = 1 + (array.max()/lower)/10
    C5  = 523
    beep(C5, 500)
    
    if times < 2:
        line_notification("glass surface error, low {lower},  max {array.max()}")
        beep(nearest_tone(C5 * times), 500)
    else:
        times = 4
        beep(nearest_tone(C5 * times), 500)


def laser1_shutter_close(FLIMageCont):
    FLIMageCont.flim.sendCommand("SetDIOPanel, 1, 0")

        
def flimage_motor_reconnect(FLIMageCont):
    FLIMageCont.flim.sendCommand('MotorReopen') 

def click_img_export_pos(ch1_dir,ch2_dir):
    ch1_tiffpath = os.path.join(ch1_dir, "stitched_z_proj.tif")
    ch2_tiffpath = os.path.join(ch2_dir, "stitched_z_proj.tif")
    setting_jsonpath = os.path.join(ch1_dir, "setting.json")
    ClickImage = ImageViewer(ch1_tiffpath,ch2_tiffpath)
    ShowPointsYXlist_original_coord = ClickImage.click_start()
    get_abs_mm_pos_dict = get_abs_mm_pos_from_click_list(setting_jsonpath, 
                                                         ShowPointsYXlist_original_coord)
    
    for each_ch_dir in [ch1_dir, ch2_dir]:        
        export_pos_dict_as_csv(get_abs_mm_pos_dict, 
                               csv_savepath = os.path.join(each_ch_dir, "assigned_pos_lowmag.csv"))
        save_image_with_assigned_pos(os.path.join(each_ch_dir, "stitched_z_proj.tif"),
                                     ShowPointsYXlist_original_coord,
                                     os.path.join(each_ch_dir, "clicked_pos.png"))
        convert_pos_low_to_high(lowmag_pos_csv_path = os.path.join(each_ch_dir, "assigned_pos_lowmag.csv"),
                                highmag_pos_save_path = os.path.join(each_ch_dir, "assigned_pos_converted_to_highmag.csv"))

def click_img_for_all_in_dir(parent_path):
    dir_list = glob.glob(os.path.join(parent_path,"*\\"))
    for each_dir in dir_list:
        click_img_export_pos(each_dir,each_dir)

def click_img_2ch(grand_parent_path, ch1_dir_stem, ch2_dir_stem):
    ch1_dir_list = glob.glob(os.path.join(grand_parent_path,ch1_dir_stem,"*\\"))
    print("ch1_dir_list ", ch1_dir_list)
    for each_ch1_dir in ch1_dir_list:
        each_ch1_dir_stem = Path(each_ch1_dir).stem
        each_ch2_dir = os.path.join(grand_parent_path, ch2_dir_stem, each_ch1_dir_stem)
        click_img_export_pos(each_ch1_dir, each_ch2_dir)

def click_zstack_for_all_in_dir(parent_path):
    dir_list = glob.glob(os.path.join(parent_path,"*\\"))
    for each_dir in dir_list:
        click_zstack_export_pos(each_dir)

def click_zstack_export_pos(dir_name, use_predefined_pos = True):
    tif_path = os.path.join(dir_name, "stitched_each_z.tif")
    tiling_setting_jsonpath = os.path.join(dir_name, "setting.json")
    stack_array = tifffile.imread(tif_path)
    
    pos_csv_path = os.path.join(dir_name, "assigned_pixel_pos.csv")
    ZYX = []
    if use_predefined_pos:
        if os.path.exists(pos_csv_path):
            ZYX = get_ZYX_pix_list_from_csv(pos_csv_path)
    
    pix_zyx_list = z_stack_multi_z_click(stack_array = stack_array, 
                                         pre_assigned_pix_zyx_list=ZYX,
                                         show_text=dir_name)
    
    get_abs_mm_pos_dict = get_abs_mm_pos_3d_from_click_list(tiling_setting_jsonpath, 
                                                            pix_zyx_list)
    export_pos_dict_as_csv(get_abs_mm_pos_dict, 
                           csv_savepath = os.path.join(dir_name, "assigned_pos_highmag_widefield.csv"))
    save_pix_pos_from_click_list(pix_zyx_list, csv_savepath = pos_csv_path)
    
    eachpos_export_path = os.path.join(dir_name, "each_pos_export")
    os.makedirs(eachpos_export_path, exist_ok=True)
    for eachpng in glob.glob(os.path.join(eachpos_export_path, "*.png")):
        os.remove(eachpng)
    save_image_with_assigned_pos_3d(tif_path = tif_path,
                                    pix_pos_csv_path = os.path.join(dir_name, "assigned_pixel_pos.csv"),
                                    png_savefolder = eachpos_export_path)

def click_zstack_flimfile(flim_file_path, use_predefined_pos = True):
    iminfo = FileReader()
    iminfo.read_imageFile(flim_file_path, True)
    ZYXarray = np.array(iminfo.image).sum(axis=tuple([1,2,5]))

    eachpos_export_path = os.path.join(Path(flim_file_path).parent,
                                       Path(flim_file_path).stem)    
    os.makedirs(eachpos_export_path, exist_ok=True)
    for eachpng in glob.glob(os.path.join(eachpos_export_path, "*.png")):
        os.remove(eachpng)
    pos_csv_path = os.path.join(eachpos_export_path, "assigned_pixel_pos.csv")
    ZYX = []
    if use_predefined_pos:
        if os.path.exists(pos_csv_path):
            ZYX = get_ZYX_pix_list_from_csv(pos_csv_path)
        else:
            print("could not find file pos info file.")
    
    pix_zyx_list = z_stack_multi_z_click(stack_array = ZYXarray, 
                                         pre_assigned_pix_zyx_list=ZYX,
                                         show_text=flim_file_path)
    
    save_pix_pos_from_click_list(pix_zyx_list, csv_savepath = pos_csv_path)
    
    ZYX_um_dict = get_abs_um_pos_from_center_3d(statedict = iminfo.statedict,
                                                pix_zyx_list = pix_zyx_list)
    pos_rel_um_csv_path = os.path.join(eachpos_export_path, "assigned_relative_um_pos.csv")
    save_um_pos_from_click_list(ZYX_um_dict = ZYX_um_dict, 
                                csv_savepath = pos_rel_um_csv_path)
    save_image_with_assigned_pos_3d(tif_path = "",
                                    pix_pos_csv_path = pos_csv_path,
                                    png_savefolder = eachpos_export_path,
                                    input_arr=True, array=ZYXarray)


def combine_pos_highmag(parent_path, solve_tsp = True):
    dir_list = glob.glob(os.path.join(parent_path,"*\\"))
    combined_highmag_poslist = []
    for each_dir in dir_list:
        wellname = os.path.basename(os.path.dirname(each_dir))
        print("wellname", wellname)
        highmag_pos_csv_savepath = os.path.join(each_dir, "assigned_pos_highmag_widefield.csv")
        each_highmag_poslist = import_from_csv(highmag_pos_csv_savepath)
        each_highmag_poslist_with_header = add_header_in_posid(header = wellname, 
                                                               poslist = each_highmag_poslist)
        combined_highmag_poslist += each_highmag_poslist_with_header
        
    combined_highmag_poslist_savepath = os.path.join(parent_path, "combined_pos_highmag_widefield.csv")
    save_poslist_as_csv(combined_highmag_poslist_savepath, combined_highmag_poslist)
    if solve_tsp:
        combined_tsp_pos_savepath = os.path.join(parent_path, "combined_pos_highmag_tsp_widefield.csv")
        save_poslist_tsp_calc(combined_tsp_pos_savepath, combined_highmag_poslist)    
    print("Position list was saved as") 
    print(combined_highmag_poslist_savepath)    


def combine_pos_from_child_dir(parent_path):
    dir_list = glob.glob(os.path.join(parent_path,"*\\"))
    combined_highmag_poslist = []
    combined_lowmag_poslist = []
    for each_dir in dir_list:
        wellname = os.path.basename(os.path.dirname(each_dir))
        print("wellname", wellname)
        highmag_pos_csv_savepath = os.path.join(each_dir, "assigned_pos_converted_to_highmag.csv")
        lowmag_pos_csv_savepath = os.path.join(each_dir, "assigned_pos_lowmag.csv")
        each_highmag_poslist = import_from_csv(highmag_pos_csv_savepath)
        each_lowmag_poslist = import_from_csv(lowmag_pos_csv_savepath)
        each_highmag_poslist_with_header = add_header_in_posid(header = wellname, 
                                                               poslist = each_highmag_poslist)
        each_lowmag_poslist_with_header = add_header_in_posid(header = wellname, 
                                                              poslist = each_lowmag_poslist)
        combined_highmag_poslist += each_highmag_poslist_with_header
        combined_lowmag_poslist += each_lowmag_poslist_with_header
    combined_lowmag_poslist_savepath = os.path.join(parent_path, "combined_pos_lowmag.csv")
    combined_highmag_poslist_savepath = os.path.join(parent_path, "combined_pos_highmag.csv")
    save_poslist_as_csv(combined_lowmag_poslist_savepath, combined_lowmag_poslist)
    save_poslist_as_csv(combined_highmag_poslist_savepath, combined_highmag_poslist)
    print("Position list was saved as")
    print(combined_highmag_poslist_savepath)
    
if __name__ == "__main__":
    # filepath = r"G:\ImagingData\Tetsuya\20240626\multipos_3\e_001.flim"
    # click_zstack_flimfile(filepath)
    
    lowmag_pos_csv_path = r"G:\ImagingData\Tetsuya\20250512\lowmag_pos.csv"
    highmag_pos_csv_path = r"G:\ImagingData\Tetsuya\20250512\highmag_pos.csv"
    convert_pos_low_to_high(lowmag_pos_csv_path,highmag_pos_csv_path )
    
    
    
    
    
    