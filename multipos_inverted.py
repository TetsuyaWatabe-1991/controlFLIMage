# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:19:07 2024

@author: yasudalab
"""

import time
import os
import glob
import pathlib
import tempfile
import pandas as pd
import copy
from FLIMageAlignment import  align_two_flimfile
from FLIMageFileReader2 import FileReader
from controlflimage_threading import Control_flimage
from multidim_tiff_viewer import read_xyz_single

class Multiarea_from_lowmag():
    def __init__(self, lowmag_path,
                 rel_pos_um_csv_path,
                 high_mag_setting_path,
                 high_mag_zoom=None,
                 ch_1or2 = 1,
                 preassigned_spine = False
                 ):
        self.lowmag_path = lowmag_path
        self.lowmag_basename = pathlib.Path(lowmag_path).stem[:-3]        
        self.lowmag_iminfo = FileReader()
        self.lowmag_iminfo.read_imageFile(self.lowmag_path, True)
        self.lowmag_magnification = self.lowmag_iminfo.statedict['State.Acq.zoom']
        self._files_folder = None
        # latestpath = self.latest_path()
        # self.lowmag_iminfo = FileReader()
        # self.lowmag_iminfo.read_imageFile(latestpath, True)
        
        self.high_mag_setting_path = high_mag_setting_path
        self.high_mag_relpos_dict = {}
        self.high_mag_zoom = high_mag_zoom
        self.rel_pos_um_csv_path = rel_pos_um_csv_path
        
        self.preassigned_spine = preassigned_spine
        
        self.read_rel_pos_um_csv()
        
        self._set_corrected_lowmag_from_iminfo(self.lowmag_iminfo)
        self.ch = ch_1or2 -1
        self.Spine_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Spine_example.png"
        self.Dendrite_example=r"C:\Users\Yasudalab\Documents\Tetsuya_Imaging\Dendrite_example.png"
        self.cuboid_ZYX=[2,20,20]

        assert type(self.high_mag_setting_path) == str

        if os.path.exists(self.high_mag_setting_path):
            self.high_mag_setting_path = self.high_mag_setting_path
        else:
            self.first_highmag_flim = self.get_first_high_mag_flim()
            if self.first_highmag_flim != "":
                self.high_mag_setting_path = self.first_highmag_flim
                print(f"Using {self.first_highmag_flim} as high mag setting path")

    def _set_corrected_lowmag_from_iminfo(self, iminfo) -> None:
        bottom_lowmag_xyz_um = list(copy.copy(iminfo.statedict['State.Motor.motorPosition']))
        slice_step = iminfo.statedict['State.Acq.sliceStep']
        n_slices = iminfo.statedict['State.Acq.nSlices']
        addition_z_um = slice_step * (n_slices - 1) / 2
        corrected_lowmag_xyz_um = copy.copy(bottom_lowmag_xyz_um)
        corrected_lowmag_xyz_um[2] += addition_z_um
        self.bottom_lowmag_xyz_um = copy.copy(bottom_lowmag_xyz_um)
        self.corrected_lowmag_xyz_um = copy.copy(corrected_lowmag_xyz_um)

    def set_corrected_lowmag_from_flim_path(self, flim_path: str) -> None:
        """Update corrected_lowmag_xyz_um from motor position stored in a lowmag FLIM."""
        iminfo = FileReader()
        iminfo.read_imageFile(flim_path, True)
        self._set_corrected_lowmag_from_iminfo(iminfo)

    def _files_folder_fallback(self) -> str:
        """Directory containing the reference lowmag file on disk."""
        return os.path.dirname(os.path.abspath(self.lowmag_path))

    def files_folder(self, FLIMageCont=None) -> str:
        """Output folder for lowmag/highmag series.

        Prefer FLIMage's current ``State.Files.pathName`` so save location is not
        locked to the reference .flim tag. Fall back to the reference file folder.
        """
        if FLIMageCont is not None:
            try:
                path = FLIMageCont.get_val_sendCommand("State.Files.pathName")
                if path:
                    path = str(path).strip().strip('"').strip("'")
                    if os.path.isdir(path):
                        self._files_folder = path
            except Exception:
                pass
        if self._files_folder:
            return self._files_folder
        return self._files_folder_fallback()

    def first_lowmag_path(self) -> str:
        """Reference lowmag FLIM passed at construction (e.g. *_001.flim)."""
        return self.lowmag_path

    
    def read_rel_pos_um_csv(self):
        self.rel_pos_df = pd.read_csv(self.rel_pos_um_csv_path)
        for ind in self.rel_pos_df.index:
            pos_id = self.rel_pos_df.loc[ind,"pos_id"]
            x_um = self.rel_pos_df.loc[ind,"x_um"]
            y_um = self.rel_pos_df.loc[ind,"y_um"]
            z_um = self.rel_pos_df.loc[ind,"z_um"]
            
            if self.preassigned_spine == True:
                inipath = f"{self.lowmag_path[:-8]}_highmag_{pos_id}.ini"
                spine_zyx, dend_slope, dend_intercept = read_xyz_single(inipath)
                if spine_zyx[0]<0:
                    print(f"Rejected highmag, {inipath}")
                    continue
                
            self.high_mag_relpos_dict[pos_id] = {}
            self.high_mag_relpos_dict[pos_id]["x_um"] = x_um
            self.high_mag_relpos_dict[pos_id]["y_um"] = y_um
            self.high_mag_relpos_dict[pos_id]["z_um"] = z_um
    
    def get_max_flimfiles(self, flimlist):
        counter = 1
        for eachflim in flimlist:
            try:   
                num = int(eachflim[-8:-5])
                if num > counter:
                    counter = num
            except:
                pass
        return counter    
            
    def get_max_plus_one_flimfiles(self, flimlist):
        counter = self.get_max_flimfiles(flimlist)
        counter+=1
        return counter
    
    def latest_path(self):
        folder = self.files_folder()
        low_flimlist = glob.glob(os.path.join(folder,
                                              self.lowmag_basename+"[0-9][0-9][0-9].flim"))        
        low_maxcount = self.get_max_flimfiles(low_flimlist)
        latestpath = os.path.join(folder, 
                                  self.lowmag_basename + str(low_maxcount).zfill(3) + ".flim")
        return latestpath

    def use_latest_lowmag_position(self) -> str:
        """
        Set corrected_lowmag_xyz_um from the most recently acquired lowmag FLIM.

        Falls back to the constructor reference file when no series exists yet.
        """
        latest_path = self.latest_path()
        if os.path.isfile(latest_path):
            self.set_corrected_lowmag_from_flim_path(latest_path)
            print(f"Initial lowmag move target: latest file {latest_path}")
            return latest_path
        print(
            f"Initial lowmag move target: reference file {self.lowmag_path} "
            f"(latest not found: {latest_path})"
        )
        return self.lowmag_path

    def first_highmag_flim_for_pos(self, pos_id) -> str:
        """Earliest existing highmag FLIM for one position (e.g. *_highmag_1_001.flim)."""
        folder = self.files_folder()
        pattern = os.path.join(
            folder,
            f"{self.lowmag_basename}_highmag_{pos_id}_[0-9][0-9][0-9].flim",
        )
        flims = sorted(glob.glob(pattern))
        return flims[0] if flims else ""

    @staticmethod
    def zoom_from_flim(flim_path: str) -> float:
        iminfo = FileReader()
        iminfo.read_imageFile(flim_path, True)
        return float(iminfo.statedict["State.Acq.zoom"])

    def resolve_high_mag_zoom(self, pos_id) -> float:
        """Prefer explicit script zoom; else zoom from the position's reference FLIM."""
        if self.high_mag_zoom is not None:
            print(
                f"high_mag zoom from script setting pos_id={pos_id}: "
                f"{self.high_mag_zoom}"
            )
            return float(self.high_mag_zoom)
        ref_flim = self.first_highmag_flim_for_pos(pos_id)
        if ref_flim:
            zoom = self.zoom_from_flim(ref_flim)
            print(
                f"high_mag zoom from reference FLIM pos_id={pos_id}: "
                f"{zoom} ({os.path.basename(ref_flim)})"
            )
            return zoom
        raise ValueError(
            f"No existing highmag FLIM for pos_id={pos_id} "
            f"({self.lowmag_basename}_highmag_{pos_id}_*.flim) and high_mag_zoom "
            "was not set. Set zoom_highmag in the acquisition script for the "
            "first highmag acquisition."
        )

    def get_first_high_mag_flim(self):
        folder = self.files_folder()
        pattern = os.path.join(
            folder,
            f"{self.lowmag_basename}_highmag_*_[0-9][0-9][0-9].flim",
        )
        highmag_flimlist = sorted(glob.glob(pattern))
        if len(highmag_flimlist) == 0:
            return ""
        return highmag_flimlist[0]

    def count_flimfiles(self, FLIMageCont=None) -> int:
        folder = self.files_folder(FLIMageCont)
        low_flimlist = glob.glob(os.path.join(folder,
                                              self.lowmag_basename+"[0-9][0-9][0-9].flim"))
        self.low_counter = self.get_max_plus_one_flimfiles(low_flimlist)    
        self.low_max_plus1_flim = os.path.join(folder, 
                                             self.lowmag_basename + str(self.low_counter).zfill(3) + ".flim")
        return self.low_counter

    def count_high_mag_flimfiles(self, pos_id, return_first_flim = False, FLIMageCont=None) -> int:
        folder = self.files_folder(FLIMageCont)
        highmag_flimlist = glob.glob(os.path.join(folder,
                                              f"{self.lowmag_basename}_highmag_{pos_id}_"+"[0-9][0-9][0-9].flim"))
        counter = self.get_max_plus_one_flimfiles(highmag_flimlist)    
        return counter
        # self.high_max_plus1_flim = os.path.join(self.lowmag_iminfo.statedict["State.Files.pathName"], 
                                              # f"{self.lowmag_basename}_highmag_{pos_id}_" + str(self.low_counter).zfill(3) + ".flim")


    def send_lowmag_acq_info(
        self,
        FLIMageCont,
        use_load_setting_lite=True,
        use_bulk_overlay=True,
        n_frames_per_slice=1,
    ):
        """
        Restore lowmag acquisition settings before an align/lowmag grab.

        Fast path (default): LoadSettingLite + one ApplySetupOverlayLite txt
        that sets power/baseName/counter/zoom/nSlices/sliceStep and forces
        nFrames/nAveFrame for single-frame-per-slice Z stacks.

        Note: some lowmag .flim files store nFrames==nSlices (bad metadata).
        Loading that file alone would acquire nFrames frames at every Z.
        Always override nFrames to n_frames_per_slice (default 1).
        """
        self.files_folder(FLIMageCont)
        low_counter = self.count_flimfiles(FLIMageCont)
        d = self.lowmag_iminfo.statedict

        load_cmd = "LoadSettingLite" if use_load_setting_lite else "LoadSetting"
        FLIMageCont.flim.sendCommand(f"{load_cmd}, {self.lowmag_path}")

        power = d["State.Acq.power"]
        zoom = d["State.Acq.zoom"]
        n_slices = d["State.Acq.nSlices"]
        slice_step = d["State.Acq.sliceStep"]
        # Never trust nFrames from the template .flim for this workflow.
        n_frames = int(n_frames_per_slice)

        if use_bulk_overlay:
            overlay_path = os.path.join(
                tempfile.gettempdir(),
                f"flimage_lowmag_overlay_{os.getpid()}.txt",
            )
            with open(overlay_path, "w", encoding="utf-8") as f:
                f.write("FLIMimage parameters\n")
                f.write(f"State.Acq.power = {power}\n")
                f.write(f'State.Files.baseName = "{self.lowmag_basename}"\n')
                f.write(f"State.Files.fileCounter = {low_counter}\n")
                f.write(f"State.Acq.zoom = {zoom}\n")
                f.write(f"State.Acq.nSlices = {n_slices}\n")
                f.write(f"State.Acq.sliceStep = {slice_step}\n")
                f.write("State.Acq.ZStack = True\n")
                f.write(f"State.Acq.nFrames = {n_frames}\n")
                f.write("State.Acq.nAveFrame = 1\n")
                f.write("State.Acq.nAveragedFrames = 1\n")
                f.write("State.Acq.aveFrame = False\n")
                f.write("State.Acq.aveFrameA = [False, False]\n")
                f.write("State.Acq.nAveSlice = 1\n")
                f.write("State.Acq.aveSlice = False\n")
            FLIMageCont.flim.sendCommand(f"ApplySetupOverlayLite, {overlay_path}")
        else:
            FLIMageCont.flim.sendCommand(f"State.Acq.power = {power}")
            FLIMageCont.flim.sendCommand(
                f'State.Files.baseName = "{self.lowmag_basename}"'
            )
            FLIMageCont.flim.sendCommand(f"State.Files.fileCounter = {low_counter}")
            FLIMageCont.flim.sendCommand(f"State.Acq.zoom = {zoom}")
            FLIMageCont.flim.sendCommand(f"State.Acq.nSlices = {n_slices}")
            FLIMageCont.flim.sendCommand(f"State.Acq.sliceStep = {slice_step}")
            FLIMageCont.flim.sendCommand("State.Acq.ZStack = True")
            FLIMageCont.flim.sendCommand(f"State.Acq.nFrames = {n_frames}")
            FLIMageCont.flim.sendCommand("State.Acq.nAveFrame = 1")
            FLIMageCont.flim.sendCommand("State.Acq.nAveragedFrames = 1")
            FLIMageCont.flim.sendCommand("State.Acq.aveFrame = False")
            FLIMageCont.flim.sendCommand("State.Acq.aveFrameA = [False, False]")

        FLIMageCont.flim.sendCommand("SetScanMirrorXY_um, 0, 0")
        FLIMageCont.flim.sendCommand("SetCenter")
    
    def highmag_motor_destination_um(self, FLIMageCont, pos_id) -> tuple[float, float, float]:
        """Absolute motor XYZ for a highmag field from corrected lowmag center + CSV offset."""
        cx, cy, cz = self.corrected_lowmag_xyz_um
        off = self.high_mag_relpos_dict[pos_id]
        dest_x = cx + FLIMageCont.directionMotorX * off["x_um"]
        dest_y = cy + FLIMageCont.directionMotorY * off["y_um"]
        dest_z = cz + FLIMageCont.directionMotorZ * off["z_um"]
        return dest_x, dest_y, dest_z

    def go_to_highmag_motor_pos(
        self,
        FLIMageCont,
        pos_id,
        first_wait_sec=None,
        iter_wait_sec=None,
        fast=False,
    ):
        """Move the stage to the absolute highmag motor position for pos_id."""
        dest_x, dest_y, dest_z = self.highmag_motor_destination_um(FLIMageCont, pos_id)
        kwargs = {"fast": fast}
        if first_wait_sec is not None:
            kwargs["first_wait_sec"] = first_wait_sec
        if iter_wait_sec is not None:
            kwargs["iter_wait_sec"] = iter_wait_sec
        FLIMageCont.go_to_absolute_pos_motor_checkstate(
            dest_x, dest_y, dest_z, **kwargs
        )
        return dest_x, dest_y, dest_z

    def update_corrected_lowmag_from_highmag_pos(self, FLIMageCont, pos_id) -> None:
        """Infer corrected lowmag center from the current aligned highmag motor position."""
        x, y, z = FLIMageCont.get_position()
        off = self.high_mag_relpos_dict[pos_id]
        self.corrected_lowmag_xyz_um = [
            x - FLIMageCont.directionMotorX * off["x_um"],
            y - FLIMageCont.directionMotorY * off["y_um"],
            z - FLIMageCont.directionMotorZ * off["z_um"],
        ]

    def warn_mismatched_reference_highmag_positions(
        self,
        FLIMageCont,
        *,
        tolerance_um: float = 5.0,
    ) -> list[int]:
        """
        Warn when an existing first highmag FLIM motor position disagrees with CSV offsets.

        Returns pos_id values that exceed tolerance_um (re-acquire reference FLIMs for these).
        """
        mismatched: list[int] = []
        for pos_id in self.high_mag_relpos_dict:
            pattern = os.path.join(
                self.files_folder(),
                f"{self.lowmag_basename}_highmag_{pos_id}_*.flim",
            )
            ref_paths = sorted(
                p for p in glob.glob(pattern) if "for_align" not in os.path.basename(p)
            )
            if not ref_paths:
                continue
            iminfo = FileReader()
            iminfo.read_imageFile(ref_paths[0], True)
            bottom_xyz = list(copy.copy(iminfo.statedict["State.Motor.motorPosition"]))
            slice_step = iminfo.statedict["State.Acq.sliceStep"]
            n_slices = iminfo.statedict["State.Acq.nSlices"]
            bottom_xyz[2] += slice_step * (n_slices - 1) / 2
            actual = bottom_xyz
            expected_x, expected_y, expected_z = self.highmag_motor_destination_um(
                FLIMageCont, pos_id
            )
            err = (
                (actual[0] - expected_x) ** 2
                + (actual[1] - expected_y) ** 2
                + (actual[2] - expected_z) ** 2
            ) ** 0.5
            if err > tolerance_um:
                mismatched.append(pos_id)
                print(
                    "WARNING reference highmag motor mismatch:",
                    f"pos_id={pos_id}",
                    f"ref={os.path.basename(ref_paths[0])}",
                    f"expected=({expected_x:.1f}, {expected_y:.1f}, {expected_z:.1f})",
                    f"actual=({actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f})",
                    f"err={err:.1f} um",
                )
        return mismatched

    def send_highmag_acq_info(
        self,
        FLIMageCont,
        pos_id,
        use_galvo=True,
        load_setting=True,
        use_load_setting_lite=False,
        first_wait_sec=None,
        iter_wait_sec=None,
        fast_motor=False,
    ):
        """
        Apply highmag acquisition settings and move to the field.

        Args:
            load_setting: If False, skip LoadSetting (reuse current FLIMage Acq).
            use_load_setting_lite: If True, use remote LoadSettingLite (skip GUI).
            first_wait_sec / iter_wait_sec: Optional shorter motor settle waits
                when use_galvo is False and fast_motor is False.
            fast_motor: Use go_to_absolute_pos_motor_checkstate(fast=True).

        Returns:
            (dest_xyz_or_None, file_counter). Absolute motor XYZ when use_galvo
            is False; file_counter is the next FLIM index for this highmag id.
        """
        self.files_folder(FLIMageCont)
        if load_setting:
            cmd = "LoadSettingLite" if use_load_setting_lite else "LoadSetting"
            FLIMageCont.flim.sendCommand(f"{cmd}, {self.high_mag_setting_path}")

        namestem = f"{self.lowmag_basename}_highmag_{pos_id}_"
        zoom = self.resolve_high_mag_zoom(pos_id)
        counter = self.count_high_mag_flimfiles(pos_id=pos_id, FLIMageCont=FLIMageCont)

        # One ReSetup/FillGUI for baseName + zoom + counter (not three).
        FLIMageCont.flim.sendCommand("BeginStateBatch")
        FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{namestem}"')
        FLIMageCont.flim.sendCommand(f"State.Acq.zoom = {zoom}")
        FLIMageCont.flim.sendCommand(f"State.Files.fileCounter = {counter}")
        FLIMageCont.flim.sendCommand("EndStateBatch")

        FLIMageCont.relative_zyx_um = [
            (-1) * self.high_mag_relpos_dict[pos_id]["z_um"],
            (-1) * self.high_mag_relpos_dict[pos_id]["y_um"],
            (-1) * self.high_mag_relpos_dict[pos_id]["x_um"],
        ]

        dest = None
        if use_galvo:
            FLIMageCont.go_to_absolute_pos_um_galvo(z_move=True)
            FLIMageCont.flim.sendCommand("SetCenter")
        else:
            # Absolute motor path already calls SetCenter inside checkstate.
            dest = self.go_to_highmag_motor_pos(
                FLIMageCont,
                pos_id,
                first_wait_sec=first_wait_sec,
                iter_wait_sec=iter_wait_sec,
                fast=fast_motor,
            )
        return dest, counter
    
    def update_pos_fromcurrent(self, FLIMageCont):
        self.corrected_lowmag_xyz_um = FLIMageCont.get_position()
    
    # def go_to_lowmag_center_pos(self, FLIMageCont, FastMS2k):
    #     dest_x,dest_y,dest_z = self.corrected_lowmag_xyz_um
    #     FLIMageCont.flim.sendCommand('MotorDisconnect')
    #     # FastMS2k.move_pos_mm(self, 
    #     #                      x_mm = dest_x*10**3,
    #     #                      y_mm = dest_y*10**3, 
    #     #                      z_mm = dest_z*10**3)
    #     FastMS2k.move_pos_mm(
    #                          x_mm = dest_x/10**3,
    #                          y_mm = dest_y/10**3, 
    #                          z_mm = dest_z/10**3)
    #     #20250212 not sure... but try adding sleep
    #     time.sleep(1)
    #     FLIMageCont.flim.sendCommand('MotorReopen')
        
    # def go_to_relative_pos_after_(self, relative_zyx_um, FLIMageCont, FastMS2k):
    #     x,y,z=FLIMageCont.get_position()        
    #     dest_x = x - FLIMageCont.directionMotorX * relative_zyx_um[2]
    #     dest_y = y - FLIMageCont.directionMotorY * relative_zyx_um[1]
    #     dest_z = z - FLIMageCont.directionMotorZ * relative_zyx_um[0]
    #     FLIMageCont.flim.sendCommand('MotorDisconnect')
    #     FastMS2k.move_pos_mm(x_mm = dest_x/10**3,
    #                          y_mm = dest_y/10**3, 
    #                          z_mm = dest_z/10**3)
    #     FLIMageCont.flim.sendCommand('MotorReopen')

    def go_to_lowmag_center_pos(self, FLIMageCont):
        dest_x,dest_y,dest_z = self.corrected_lowmag_xyz_um        
        FLIMageCont.go_to_absolute_pos_motor_checkstate(dest_x, dest_y, dest_z)
        # time.sleep(1)
        
    def go_to_relative_pos_after_(self, relative_zyx_um, FLIMageCont):
        x,y,z=FLIMageCont.get_position()        
        dest_x = x - FLIMageCont.directionMotorX * relative_zyx_um[2]
        dest_y = y - FLIMageCont.directionMotorY * relative_zyx_um[1]
        dest_z = z - FLIMageCont.directionMotorZ * relative_zyx_um[0]
        FLIMageCont.go_to_absolute_pos_motor_checkstate(dest_x, dest_y, dest_z)





if __name__ == "__main__":
    high_mag_setting_path = r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\z7_10_kal8.txt"
    FLIMageCont = Control_flimage()
    FLIMageCont.interval_sec = 600
    FLIMageCont.expected_grab_duration_sec = 5
    ch_1or2 = 2
    lowmag_path_list = [r"G:\ImagingData\Tetsuya\20240626\multipos_3\a_001.flim",
                        r"G:\ImagingData\Tetsuya\20240626\multipos_3\b_001.flim",
                        r"G:\ImagingData\Tetsuya\20240626\multipos_3\c_001.flim",
                        r"G:\ImagingData\Tetsuya\20240626\multipos_3\d_001.flim",
                        r"G:\ImagingData\Tetsuya\20240626\multipos_3\e_001.flim"
                        ]
    
    lowmag_instance_list = []
    for each_lowmag in lowmag_path_list:
        # lowmag_path = r"G:\ImagingData\Tetsuya\20240626\b_001.flim"
        # rel_pos_um_csv_path = r"G:\ImagingData\Tetsuya\20240626\b_001\assigned_relative_um_pos.csv"
        
        rel_pos_um_csv_path = os.path.join(pathlib.Path(each_lowmag).parent, 
                                          pathlib.Path(each_lowmag).stem,
                                          "assigned_relative_um_pos.csv")
        
        lowmag_instance_list.append(Multiarea_from_lowmag(lowmag_path = each_lowmag,
                                                          rel_pos_um_csv_path = rel_pos_um_csv_path,
                                                          high_mag_setting_path = high_mag_setting_path))
    for nth_acq in range(100):
        
        for Each_lowmag_instance in lowmag_instance_list:    
            Each_lowmag_instance.go_to_lowmag_center_pos(FLIMageCont)
            Each_lowmag_instance.send_lowmag_acq_info(FLIMageCont)
            FLIMageCont.acquisition_include_connect_wait()
            FLIMageCont.set_param(RepeatNum = 1, interval_sec = 500, ch_1or2 = ch_1or2)
            FLIMageCont.relative_zyx_um, _ = align_two_flimfile(
                                                Each_lowmag_instance.lowmag_path, 
                                                Each_lowmag_instance.low_max_plus1_flim,
                                                Each_lowmag_instance.ch)
            FLIMageCont.go_to_relative_pos_motor_checkstate()
            Each_lowmag_instance.update_pos_fromcurrent(FLIMageCont)
            
            for each_high_mag_id in Each_lowmag_instance.high_mag_relpos_dict:   
                Each_lowmag_instance.go_to_lowmag_center_pos(FLIMageCont)
                Each_lowmag_instance.send_highmag_acq_info(FLIMageCont, each_high_mag_id)
                FLIMageCont.acquisition_include_connect_wait()
                
                
                
                