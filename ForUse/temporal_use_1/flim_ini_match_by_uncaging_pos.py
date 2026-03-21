# -*- coding: utf-8 -*-
"""
Match FLIM files to spine INI files by uncaging position.

When multiple INI files exist for a flim group (e.g. _000.ini, _001.ini, _002.ini),
this module finds the INI whose spine position is closest to the uncaging position
stored in the FLIM file's State.Uncaging.Position.

iPython / Jupyter usage:
    %cd C:\\Users\\WatabeT\\Documents\\Git\\controlFLIMage\\ForUse\\temporal_use_1
    from flim_ini_match_by_uncaging_pos import get_matching_ini_for_flim, batch_calc_spine_dend_with_matching

    # Single file
    inipath = get_matching_ini_for_flim(flim_path, max_distance_pix=20)

    # Batch
    df = batch_calc_spine_dend_with_matching(folder_path, save_csv=True)
"""
import os
import sys
import glob
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from FLIMageFileReader2 import FileReader
from multidim_tiff_viewer import read_xyz_single


def get_uncaging_position_pixels(flim_path):
    """
    Get uncaging position from FLIM file in pixel coordinates (x, y).

    Parameters
    ----------
    flim_path : str
        Path to .flim file

    Returns
    -------
    tuple or None
        (center_x, center_y) in pixels, or None if uncaging position cannot be read
    """
    if not os.path.exists(flim_path):
        return None
    iminfo = FileReader()
    try:
        iminfo.read_imageFile(flim_path, False)
    except Exception as e:
        print(f"Could not read flim file {flim_path}: {e}")
        return None
    try:
        uncaging_x_y_0to1 = iminfo.statedict["State.Uncaging.Position"]
    except KeyError:
        print(f"No State.Uncaging.Position in {flim_path}")
        return None
    x_pix = iminfo.statedict["State.Acq.pixelsPerLine"]
    y_pix = iminfo.statedict["State.Acq.linesPerFrame"]
    center_x = x_pix * uncaging_x_y_0to1[0]
    center_y = y_pix * uncaging_x_y_0to1[1]
    return (center_x, center_y)


def get_ini_folder_and_base(flim_path):
    """
    Get the folder and base name for INI files from a flim path.

    For "folder/1_pos1__highmag_1_001.flim", returns:
        (folder/1_pos1__highmag_1, 1_pos1__highmag_1)

    Parameters
    ----------
    flim_path : str
        Path to .flim file

    Returns
    -------
    tuple
        (ini_folder, base_name)
    """
    folder = os.path.dirname(flim_path)
    basename = os.path.basename(flim_path)
    # Remove _XXX.flim (9 chars for _001.flim, _002.flim, etc.)
    if basename.endswith(".flim"):
        # Match _001, _002, ..., _999
        match = re.match(r"^(.+)_(\d{3})\.flim$", basename)
        if match:
            prefix = match.group(1)
        else:
            prefix = basename[:-5]  # remove .flim
    else:
        prefix = basename
    ini_folder = os.path.join(folder, prefix)
    return ini_folder, prefix


def get_matching_ini_for_flim(
    flim_path,
    exclude_excluded=True,
    max_distance_pix=None,
):
    """
    Find the INI file whose spine position is closest to the uncaging position in the FLIM file.

    Uses 2D Euclidean distance between State.Uncaging.Position (x, y) and spine (x, y) from INI.

    Parameters
    ----------
    flim_path : str
        Path to .flim file (typically an uncaging frame)
    exclude_excluded : bool, optional
        If True, skip INI files with excluded=1. Default True.
    max_distance_pix : float or None, optional
        Maximum allowed distance in pixels. If the closest INI is farther, return None.
        Default None (no limit).

    Returns
    -------
    str or None
        Path to the best-matching INI file, or None if no match found
    """
    uncaging_xy = get_uncaging_position_pixels(flim_path)
    if uncaging_xy is None:
        return None
    center_x, center_y = uncaging_xy

    ini_folder, base_name = get_ini_folder_and_base(flim_path)
    pattern = os.path.join(ini_folder, base_name + "*.ini")
    inilist = glob.glob(pattern)
    if not inilist:
        return None

    best_inipath = None
    best_distance = float("inf")

    for inipath in inilist:
        if not os.path.exists(inipath):
            continue
        try:
            spine_zyx, dend_slope, dend_intercept, excluded = read_xyz_single(
                inipath, return_excluded=True
            )
        except Exception as e:
            print(f"Could not read INI {inipath}: {e}")
            continue
        if exclude_excluded and excluded:
            continue
        spine_z, spine_y, spine_x = spine_zyx
        # 2D distance (x, y); z from motor could be used for use_3d but is often unreliable
        dist = ((center_x - spine_x) ** 2 + (center_y - spine_y) ** 2) ** 0.5
        if dist < best_distance:
            best_distance = dist
            best_inipath = inipath

    if best_inipath is None:
        return None
    if max_distance_pix is not None and best_distance > max_distance_pix:
        print(
            f"Closest INI {best_inipath} is {best_distance:.1f} px away "
            f"(max {max_distance_pix}). Skipping."
        )
        return None
    return best_inipath


def batch_calc_spine_dend_with_matching(
    folder_or_file_list,
    save_csv=True,
    csv_name="result_F_F0.csv",
    max_distance_pix=20.0,
    min_seconds_since_modified=20,
):
    """
    Batch process FLIM files using uncaging-position-based INI matching.

    Parameters
    ----------
    folder_or_file_list : str or list
        Either a folder path (glob *_highmag_*.flim) or a list of flim file paths
    save_csv : bool, optional
        Save results to CSV. Default True.
    csv_name : str, optional
        CSV filename. Default "result_F_F0.csv"
    max_distance_pix : float, optional
        Max distance for INI match. Default 20.0
    min_seconds_since_modified : float, optional
        Skip files modified within this many seconds (avoid processing incomplete files).
        Default 20.

    Returns
    -------
    pandas.DataFrame
        Results with file_path, spineF_F0, shaftF_F0, inipath
    """
    import datetime
    import pandas as pd

    from flimage_graph_func import calc_spine_dend_GCaMP

    if isinstance(folder_or_file_list, str):
        flimlist = glob.glob(
            os.path.join(folder_or_file_list, "*_highmag_*.flim")
        )
    else:
        flimlist = list(folder_or_file_list)

    results = []
    cutoff = datetime.datetime.now() - datetime.timedelta(seconds=min_seconds_since_modified)

    for each_file in sorted(flimlist):
        if not os.path.exists(each_file):
            continue
        if datetime.datetime.fromtimestamp(os.path.getmtime(each_file)) > cutoff:
            continue
        inipath = get_matching_ini_for_flim(
            each_file, exclude_excluded=True, max_distance_pix=max_distance_pix
        )
        if inipath is None:
            print(f"No matching INI for {each_file}, skip")
            continue
        print(f"Processing {each_file} -> {inipath}")
        spineF_F0, shaftF_F0 = calc_spine_dend_GCaMP(
            each_file=each_file, each_ini=inipath, save_img=True
        )
        if spineF_F0 == -1:
            continue
        results.append({
            "file_path": each_file,
            "inipath": inipath,
            "spineF_F0": spineF_F0,
            "shaftF_F0": shaftF_F0,
        })

    df = pd.DataFrame(results)
    if save_csv and len(df) > 0:
        out_folder = os.path.dirname(df["file_path"].iloc[0])
        save_folder = os.path.join(out_folder, "plot")
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, csv_name)
        df.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")
    return df


# --- iPython / Jupyter usage ---
# %cd C:\Users\WatabeT\Documents\Git\controlFLIMage\ForUse\temporal_use_1
# %run flim_ini_match_by_uncaging_pos  # or: import flim_ini_match_by_uncaging_pos
#
# from flim_ini_match_by_uncaging_pos import get_matching_ini_for_flim, batch_calc_spine_dend_with_matching
#
# # Single file: get matching INI
# inipath = get_matching_ini_for_flim(r"G:\ImagingData\Tetsuya\20260225\auto1\1_pos1__highmag_1_003.flim", max_distance_pix=20)
#
# # Batch: process all flim files in folder
# df = batch_calc_spine_dend_with_matching(r"G:\ImagingData\Tetsuya\20260225\auto1", save_csv=True, max_distance_pix=20)

if __name__ == "__main__":
    folder_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260225\auto1"
    df = batch_calc_spine_dend_with_matching(folder_path, save_csv=True, max_distance_pix=20)

    print(df)
    csv_path = os.path.join(folder_path, "plot", "result_F_F0.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")