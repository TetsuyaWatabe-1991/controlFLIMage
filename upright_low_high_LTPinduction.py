import glob
import os
import pathlib
from time import sleep
import datetime
from FLIMageAlignment import  align_two_flimfile
from controlflimage_threading import Control_flimage
from multidim_tiff_viewer import read_xyz_single
from multipos_upright import Multiarea_from_lowmag

uncaging_setting_path = r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\uncagingLTP.txt"
high_mag_setting_path = r"C:\Users\Yasudalab\Documents\FLIMage\Init_Files\highmag_128_sum8_z1um_x7.txt"
ini_path = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini"

uncaging_power = 24

uncaging_margin = 0.1
uncaging_at_nth = 0
each_instance_len = 1
acquisition_number = 30
minutes_after_uncaging = 25

ch_1or2 = 2
zoom_highmag = 15

lowmag_path_list = [        
r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20250520\E4bufferMNI4Ca\lowmag1_001.flim",
r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20250520\E4bufferMNI4Ca\lowmag2_001.flim",
r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\20250520\E4bufferMNI4Ca\lowmag3_001.flim"
                        ]

FLIMageCont = Control_flimage(ini_path)

lowmag_instance_list = []
for each_lowmag in lowmag_path_list:
    rel_pos_um_csv_path = os.path.join(pathlib.Path(each_lowmag).parent, 
                                      pathlib.Path(each_lowmag).stem,
                                      "assigned_relative_um_pos.csv")
    with open(rel_pos_um_csv_path) as f:
        text = f.read()
    if len(text.split("\n"))<3:
        continue
    lowmag_instance = Multiarea_from_lowmag(lowmag_path = each_lowmag,
                                            rel_pos_um_csv_path = rel_pos_um_csv_path,
                                            high_mag_setting_path = high_mag_setting_path,
                                            preassigned_spine = False)
    if len(lowmag_instance.high_mag_relpos_dict) > 0:
        lowmag_instance_list.append(lowmag_instance)

list_of_list = []
for nthinst in range(len(lowmag_instance_list)):
    if nthinst % each_instance_len ==0:
        list_of_list.append([])
    list_of_list[-1].append(lowmag_instance_list[nthinst])


First = True
for nth_spine in range(12):        

    for each_lowmag_instance_list in list_of_list:
        time_count_started_flag = False
             
        for nth_acq in range(acquisition_number):
            track_uncaging_TF = (nth_acq <= uncaging_at_nth)
            
            if (track_uncaging_TF==False)*(time_count_started_flag == False):
                time_count_started_flag = True
                datetime_last_uncaging = datetime.datetime.now()
                
            for Each_lowmag_instance in each_lowmag_instance_list:    
                Each_lowmag_instance.go_to_lowmag_center_pos(FLIMageCont)
                Each_lowmag_instance.send_lowmag_acq_info(FLIMageCont)
                FLIMageCont.set_param(RepeatNum = 1, interval_sec = 30, 
                    ch_1or2 = ch_1or2,
                    expected_grab_duration_sec= 15)
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
                    FLIMageCont.set_param(RepeatNum = 1,
                                          interval_sec = 500,
                                          ch_1or2 = ch_1or2,
                                          expected_grab_duration_sec = 5,
                                          track_uncaging = track_uncaging_TF,
                                          drift_control = False,
                                          drift_cont_galvo = False,
                                          defined_dendrite = True,
                                          ShowUncagingDetection = track_uncaging_TF
                                          )
    
                    ini_path_list = glob.glob(os.path.join(
                                            f"{Each_lowmag_instance.lowmag_path[:-8]}_highmag_{each_high_mag_id}",
                                            f"{os.path.basename(Each_lowmag_instance.lowmag_path[:-8])}_highmag_{each_high_mag_id}_*.ini"                           
                                            ))
                    
                    non_rejected_inipath = []
        
                    for each_ini in ini_path_list:
                        spine_zyx, dend_slope, dend_intercept, excluded = read_xyz_single(each_ini,
                                                                                return_excluded = True)
                        if excluded == 0:
                            non_rejected_inipath.append(each_ini)
                    
                    if nth_spine < len(non_rejected_inipath):
                        spine_inipath = non_rejected_inipath[nth_spine]
                    else:
                        print("SKIPPED")
                        continue
                    
                    # FLIMageCont.set_spine_dendrite_from_ini(inipath = spine_inipath)
                    FLIMageCont.SpineHeadToUncaging_um = uncaging_margin
                    FLIMageCont.start_repeat(spine_inipath = spine_inipath)
                    
                    if nth_acq == uncaging_at_nth:
                        FLIMageCont.go_to_uncaging_plane()
                        FLIMageCont.set_param(RepeatNum = 1, interval_sec = 200,
                                              ch_1or2 = ch_1or2, LoadSetting = True,
                                              SettingPath = uncaging_setting_path,
                                              track_uncaging = False,
                                              drift_control = False,
                                              ShowUncagingDetection = False,
                                              expected_grab_duration_sec = 64,
                                              )
                        FLIMageCont.set_uncaging_power(uncaging_power)
                        FLIMageCont.start_repeat()
                        
            if time_count_started_flag == True:
                elapsed_time_sec = (datetime.datetime.now() - datetime_last_uncaging).total_seconds()
                print(round(elapsed_time_sec/60,1), "minutes passed after the last uncaging.")
                if minutes_after_uncaging < elapsed_time_sec/60:
                    print("go to next lowmag instances")
                    time_count_started_flag = False
                    break

