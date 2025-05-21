#%%
import os
import pathlib
from time import sleep
from FLIMageAlignment import  align_two_flimfile
from multipos_upright import Multiarea_from_lowmag

def prepare_lowmag_instance_list(lowmag_path_list, high_mag_setting_path):
    lowmag_instance_list = []
    for each_lowmag in lowmag_path_list:
        rel_pos_um_csv_path = os.path.join(pathlib.Path(each_lowmag).parent, 
                                        pathlib.Path(each_lowmag).stem,
                                        "assigned_relative_um_pos.csv")
        with open(rel_pos_um_csv_path) as f:
            text = f.read()
        if len(text.split("\n"))<3:
            continue
        lowmag_instance_list.append(Multiarea_from_lowmag(lowmag_path = each_lowmag,
                                                        rel_pos_um_csv_path = rel_pos_um_csv_path,
                                                        high_mag_setting_path = high_mag_setting_path))
    return lowmag_instance_list

def acq_lowmag_instance_list(lowmag_instance_list, FLIMageCont, ch_1or2, zoom_highmag):
    FLIMageCont.set_param(RepeatNum = 1, interval_sec = 30, 
                        ch_1or2 = ch_1or2,
                        expected_grab_duration_sec= 15)
    for Each_lowmag_instance in lowmag_instance_list:
        Each_lowmag_instance.go_to_lowmag_center_pos(FLIMageCont)
        Each_lowmag_instance.send_lowmag_acq_info(FLIMageCont)
        FLIMageCont.xy_zoom1_setting()
        FLIMageCont.acquisition_include_connect_wait()

        FLIMageCont.relative_zyx_um, _ = align_two_flimfile(
                                            Each_lowmag_instance.lowmag_path,
                                            Each_lowmag_instance.low_max_plus1_flim,
                                            Each_lowmag_instance.ch)
        FLIMageCont.go_to_relative_pos_motor_checkstate()
        Each_lowmag_instance.update_pos_fromcurrent(FLIMageCont)
        
        for each_high_mag_id in Each_lowmag_instance.high_mag_relpos_dict:
            Each_lowmag_instance.go_to_lowmag_center_pos(FLIMageCont)
            Each_lowmag_instance.high_mag_zoom = zoom_highmag
            Each_lowmag_instance.send_highmag_acq_info(FLIMageCont, each_high_mag_id,
                                                    use_galvo = False)
            FLIMageCont.xy_zoom1_setting()
            FLIMageCont.acquisition_include_connect_wait()

    print("acquisition done")
    return None

if __name__ == "__main__":
    from controlflimage_threading import Control_flimage
    high_mag_setting_path = r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\highmag_128_sum8_z1um_x7.txt"
    ini_path = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"
    
    ch_1or2 = 2
    zoom_highmag = 15

    lowmag_path_list = [        
    r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20250520\E4bufferMNI4Ca\lowmag1_001.flim",
    r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20250520\E4bufferMNI4Ca\lowmag2_001.flim",
    r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20250520\E4bufferMNI4Ca\lowmag3_001.flim",
                        ]
    
    FLIMageCont = Control_flimage(ini_path)
    FLIMageCont.XYsize_ini_path = r"C:\Users\Yasudalab\Documents\Tetsuya_GIT\controlFLIMage\XYsize.ini"
    FLIMageCont.xy_zoom1_setting()

    lowmag_instance_list = prepare_lowmag_instance_list(lowmag_path_list, high_mag_setting_path)
    acq_lowmag_instance_list(lowmag_instance_list, FLIMageCont, ch_1or2, zoom_highmag)

# %%
