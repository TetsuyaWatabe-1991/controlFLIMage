import os
import json
import matplotlib.pyplot as plt
from tifffile import imread
import pandas as pd

def get_abs_mm_pos_3d_from_click_list(tiling_setting_jsonpath: str, 
                                      pix_zyx_list: list) -> dict:
    with open(tiling_setting_jsonpath, "r") as f:
        tiling_setting = json.load(f)
    xsize_um = tiling_setting["img_xsize"] * tiling_setting["pix_um"] * tiling_setting["UseXarea"]
    ysize_um = tiling_setting["img_ysize"] * tiling_setting["pix_um"] * tiling_setting["UseYarea"]
    binned_pixel_um = tiling_setting["pix_um"] * tiling_setting["binning"]
    zero_x_mm = tiling_setting["center_x_mm"] - (xsize_um/1000)*tiling_setting["xnum"]/2
    zero_y_mm = tiling_setting["center_y_mm"] + (ysize_um/1000)*tiling_setting["ynum"]/2
    zero_z_mm = tiling_setting["center_z_mm"] - ((tiling_setting["Znumber"] - 1) / 2) * tiling_setting["Zevery_um"]/1000
    
    ZYX_mm_dict = {}
    for ind, each_ZYX_pix_coord in enumerate(pix_zyx_list):
        ZYX_mm_dict[ind] = {}
        ZYX_mm_dict[ind]["z_mm"] = round(zero_z_mm + each_ZYX_pix_coord[0]*tiling_setting["Zevery_um"]/1000,4)
        ZYX_mm_dict[ind]["y_mm"] = round(zero_y_mm - each_ZYX_pix_coord[1]*binned_pixel_um/1000,4)
        ZYX_mm_dict[ind]["x_mm"] = round(zero_x_mm + each_ZYX_pix_coord[2]*binned_pixel_um/1000,4)

    return ZYX_mm_dict

def get_abs_mm_pos_from_click_list(tiling_setting_jsonpath, ShowPointsYXlist_original_coord):
    with open(tiling_setting_jsonpath, "r") as f:
        tiling_setting = json.load(f)
    xsize_um = tiling_setting["img_xsize"] * tiling_setting["pix_um"] * tiling_setting["UseXarea"]
    ysize_um = tiling_setting["img_ysize"] * tiling_setting["pix_um"] * tiling_setting["UseYarea"]
    binned_pixel_um = tiling_setting["pix_um"] * tiling_setting["binning"]
    zero_x_mm = tiling_setting["center_x_mm"] - (xsize_um/1000)*tiling_setting["xnum"]/2
    zero_y_mm = tiling_setting["center_y_mm"] + (ysize_um/1000)*tiling_setting["ynum"]/2

    YXZ_mm_coord = {}
    for ind, each_YX_pix_coord in enumerate(ShowPointsYXlist_original_coord):
        YXZ_mm_coord[ind] = {}
        YXZ_mm_coord[ind]["x_mm"] = round(zero_x_mm + each_YX_pix_coord[1]*binned_pixel_um/1000,4)
        YXZ_mm_coord[ind]["y_mm"] = round(zero_y_mm - each_YX_pix_coord[0]*binned_pixel_um/1000,4)
        YXZ_mm_coord[ind]["z_mm"] = tiling_setting["center_z_mm"]

    return YXZ_mm_coord


def get_pixel_from_abs_mm(tiling_setting_jsonpath, y_mm, x_mm):
    with open(tiling_setting_jsonpath, "r") as f:
        tiling_setting = json.load(f)
    xsize_um = tiling_setting["img_xsize"] * tiling_setting["pix_um"] * tiling_setting["UseXarea"]
    ysize_um = tiling_setting["img_ysize"] * tiling_setting["pix_um"] * tiling_setting["UseYarea"]
    ysize_um = tiling_setting["img_ysize"] * tiling_setting["pix_um"] * tiling_setting["UseYarea"]
    binned_pixel_um = tiling_setting["pix_um"] * tiling_setting["binning"]
    zero_x_mm = tiling_setting["center_x_mm"] - (xsize_um/1000)*tiling_setting["xnum"]/2
    zero_y_mm = tiling_setting["center_y_mm"] + (ysize_um/1000)*tiling_setting["ynum"]/2
    
    x_mm_from_zero_x_mm = x_mm - zero_x_mm
    y_mm_from_zero_y_mm = y_mm - zero_y_mm
    
    x_pixel = x_mm_from_zero_x_mm*1000 / binned_pixel_um
    y_pixel = y_mm_from_zero_y_mm*1000 / binned_pixel_um
    
    return y_pixel, x_pixel


def get_abs_um_pos_from_center_3d(statedict: dict, 
                                  pix_zyx_list: list) -> dict:
    side_len = statedict['State.Acq.FOV_default'][0]/statedict['State.Acq.zoom']
    x_len_pix = statedict['State.Acq.pixelsPerLine']
    y_len_pix = statedict['State.Acq.linesPerFrame']
    z_len_pix = statedict['State.Acq.nSlices']
    longer_pix = max(x_len_pix, y_len_pix)
    um_per_pix = side_len/longer_pix
    
    y_side_len = statedict['State.Acq.FOV_default'][1]/statedict['State.Acq.zoom']
    y_um_per_pix = y_side_len/longer_pix
    
    um_per_z = statedict['State.Acq.sliceStep']
    ZYX_um_dict = {}
    for nth, pix_zyx in enumerate(pix_zyx_list):
        ZYX_um_dict[nth] = {}
        ZYX_um_dict[nth]["z_um"] = um_per_z * (pix_zyx[0] - z_len_pix//2)
        ZYX_um_dict[nth]["y_um"] = round(y_um_per_pix * (pix_zyx[1] - y_len_pix/2), 1)
        ZYX_um_dict[nth]["x_um"] = round(um_per_pix * (pix_zyx[2] - x_len_pix/2), 1)
    return ZYX_um_dict


def save_pix_pos_from_click_list(pix_zyx_list: list,
                                 csv_savepath: str) -> None:
    text = "pos_id,x_pix,y_pix,z_pix\n"
    for ind, each_zyx in enumerate(pix_zyx_list):
        text += f"{ind + 1}, {each_zyx[2]}, {each_zyx[1]}, {each_zyx[0]}\n"
    with open(csv_savepath, 'w') as f:
        f.write(text)
    print(f"pixel position data saved as: {csv_savepath}")     


def save_um_pos_from_click_list(ZYX_um_dict: list,
                                 csv_savepath: str) -> None:
    text = "pos_id,x_um,y_um,z_um\n"
    for ind in ZYX_um_dict:
        text += f"{ind + 1},{ZYX_um_dict[ind]['x_um']},{ZYX_um_dict[ind]['y_um']},{ZYX_um_dict[ind]['z_um']}\n"
    with open(csv_savepath, 'w') as f:
        f.write(text)
    print(f"pixel position data saved as: {csv_savepath}")   



def get_ZYX_pix_list_from_csv(csv_path: str) -> list:
    ZYXlist = []
    f = open(csv_path,"r")
    lines = f.readlines()
    for nth, each_line in enumerate(lines):
        if nth==0:
            assert each_line == "pos_id,x_pix,y_pix,z_pix\n"
        else:
            ZYXlist.append(
                [int(xyz) for xyz in each_line.replace("\n","").split(",")[1:]][::-1]
                )
    return ZYXlist

def export_pos_dict_as_csv(YXZ_mm_coord, csv_savepath):
    text="pos_id, obj_pos, x_pos_mm, y_pos_mm, z_pos_mm\n"
    for ind in YXZ_mm_coord:        
        abs_x_mm = YXZ_mm_coord[ind]["x_mm"]
        abs_y_mm = YXZ_mm_coord[ind]["y_mm"]
        abs_z_mm = YXZ_mm_coord[ind]["z_mm"] 
        text += f"{ind + 1}, 0, {abs_x_mm}, {abs_y_mm}, {abs_z_mm}\n"
    with open(csv_savepath, 'w') as f:
        f.write(text)
    print(f"absolute position (unit: mm) data saved as: {csv_savepath}")

def save_image_with_assigned_pos(ch1_tiffpath,
                                 ShowPointsYXlist_original_coord,
                                 png_savepath,
                                 dpi = 300):
    tiffarray = imread(ch1_tiffpath)
    plt.imshow(tiffarray, cmap = 'gray')
    for ind, each_coord in enumerate(ShowPointsYXlist_original_coord):
        plt.text(x = ShowPointsYXlist_original_coord[ind][1],
                 y = ShowPointsYXlist_original_coord[ind][0],
                 s = str(ind + 1),
                 color = "m",
                 ha = "center", va = "center", 
                 fontsize = 'x-large')
    plt.savefig(png_savepath, dpi = dpi, bbox_inches = "tight")
    plt.show()

    plt.imshow(tiffarray, cmap = 'gray')
    for ind, each_coord in enumerate(ShowPointsYXlist_original_coord):
        plt.scatter([ShowPointsYXlist_original_coord[ind][1]],
                    [ShowPointsYXlist_original_coord[ind][0]],
                    facecolors='none', edgecolors="m")
        
    plt.savefig(png_savepath[:-4]+"_dots.png", dpi = dpi, bbox_inches = "tight")
    plt.show()
    
    plt.imshow(tiffarray, cmap = 'gray')
    plt.savefig(png_savepath[:-4]+"_no_position.png", dpi = dpi, bbox_inches = "tight")
    plt.show()
             

def save_image_with_assigned_pos_3d(tif_path,
                                    pix_pos_csv_path,
                                    png_savefolder,
                                    dpi = 300, 
                                    input_arr = False,
                                    array = None):
    if input_arr == True:
        tiffarray = array
    else:
        tiffarray = imread(tif_path)
            
    half_square = min(tiffarray.shape[1:])//10
    
    if len(tiffarray.shape)!=3:
        raise Exception(f"{tif_path}\n  Tiff array shape is not 3d, {tiffarray.shape} ")
    df = pd.read_csv(pix_pos_csv_path)
    
    for each_ind in df.index:
        pos_id = df.at[each_ind, "pos_id"]
        x = df.at[each_ind, "x_pix"]
        y = df.at[each_ind, "y_pix"]
        z = df.at[each_ind, "z_pix"]
        
        plt_x_list = [x - half_square, x - half_square, x + half_square, x + half_square, x - half_square]
        plt_y_list = [y - half_square, y + half_square, y + half_square, y - half_square, y - half_square]
        upper_left_x = max(x - half_square + tiffarray.shape[2]*0.01, 0)
        upper_left_y = max(y - half_square + tiffarray.shape[1]*0.01, 0)
        savepath = os.path.join(png_savefolder, f"{str(pos_id).zfill(3)}.png")
        
        plt.imshow(tiffarray[z,:,:], cmap = 'gray')
        
        plt.plot(plt_x_list, plt_y_list, c = 'w')
        plt.text(upper_left_x, upper_left_y, s = pos_id, fontsize=10,
                 ha = "left", va = "top", color = 'cyan')
        plt.ylim([0, tiffarray.shape[1]])
        plt.xlim([0, tiffarray.shape[2]])
        plt.gca().invert_yaxis()
        plt.gca().set_axis_off()
        plt.savefig(savepath, dpi = dpi, bbox_inches = "tight", pad_inches = 0)
        plt.close();plt.clf();

    return None
    # '''
    # tif_path = r"G:\ImagingData\Tetsuya\sample\GFP100ms47p\B1_00_1\stitched_each_z.tif"
    # pix_pos_csv_path = r"G:\ImagingData\Tetsuya\sample\GFP100ms47p\B1_00_1\assigned_pixel_pos.csv"
    # png_savefolder = r"G:\ImagingData\Tetsuya\sample\GFP100ms47p\B1_00_1\each_pos_export"
    # dpi = 300
    # square_len_pix = 100
    # '''
        

    # for ind, each_coord in enumerate(ShowPointsYXlist_original_coord):
    #     plt.text(x = ShowPointsYXlist_original_coord[ind][1],
    #              y = ShowPointsYXlist_original_coord[ind][0],
    #              s = str(ind + 1),
    #              color = "m",
    #              ha = "center", va = "center", 
    #              fontsize = 'x-large')
    # plt.savefig(png_savepath, dpi = dpi, bbox_inches = "tight")
    # plt.show()

    # plt.imshow(tiffarray, cmap = 'gray')
    # for ind, each_coord in enumerate(ShowPointsYXlist_original_coord):
    #     plt.scatter([ShowPointsYXlist_original_coord[ind][1]],
    #                 [ShowPointsYXlist_original_coord[ind][0]],
    #                 facecolors='none', edgecolors="m")
        
    # plt.savefig(png_savepath[:-4]+"_dots.png", dpi = dpi, bbox_inches = "tight")
    # plt.show()
    
    # plt.imshow(tiffarray, cmap = 'gray')
    # plt.savefig(png_savepath[:-4]+"_no_position.png", dpi = dpi, bbox_inches = "tight")
    # plt.show()
             



