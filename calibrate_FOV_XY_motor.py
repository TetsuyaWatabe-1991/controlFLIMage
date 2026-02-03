# -*- coding: utf-8 -*-
"""
Calibration script for two-photon image field of view (FOV) size and XY motor movement.

This script:
1. Acquires a reference image at the initial position
2. Moves the motor X direction by a specified distance (default 30 um) and acquires an image
3. Returns to the initial position
4. Moves the motor Y direction by a specified distance (default 30 um) and acquires an image
5. Aligns the images to measure pixel shifts
6. Calculates FOV size from motor movement distance and pixel shifts
7. Saves the calibration results to XYsize.ini

Created: 2026-01-27
@author: yasudalab
"""

import os
import sys
import configparser
from time import sleep
from pathlib import Path

# Add controlFLIMage to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controlflimage_threading import Control_flimage
from FLIMageAlignment import align_two_flimfile, get_xyz_pixel_um
from FLIMageAlignment import get_flimfile_list


def calibrate_fov_xy_motor(
    ini_path: str = r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage\DirectionSetting.ini",
    move_distance_um: float = 30.0,
    ch_1or2: int = 1,
    xy_size_ini_path: str = None,
    base_name: str = "calibration_FOV"
) -> tuple[float, float]:
    """
    Calibrate FOV size by moving motor in X and Y directions and measuring pixel shifts.

    Args:
        ini_path: Path to DirectionSetting.ini file
        move_distance_um: Distance to move motor in X and Y directions (default: 30.0 um)
        ch_1or2: Channel to use for imaging (1 or 2, default: 1)
        xy_size_ini_path: Path to save XYsize.ini file. If None, uses default location.
        base_name: Base name for acquired images (default: "calibration_FOV")

    Returns:
        tuple: (X_zoom1_um, Y_zoom1_um) - Calculated FOV sizes for zoom=1
    """
    print("=" * 60)
    print("FOV and XY Motor Calibration")
    print("=" * 60)
    print(f"Move distance: {move_distance_um} um")
    print(f"Channel: {ch_1or2}")
    print()

    # Initialize FLIMage controller
    FLIMageCont = Control_flimage(ini_path=ini_path, debug_mode=False)

    # Set acquisition parameters
    FLIMageCont.set_param(
        RepeatNum=1,
        interval_sec=30,
        ch_1or2=ch_1or2,
        expected_grab_duration_sec=3
    )

    # Set base name for images
    FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{base_name}"')
    FLIMageCont.flim.sendCommand('State.Files.fileCounter = 0')

    # Get current position as reference
    x_ref, y_ref, z_ref = FLIMageCont.get_position()
    print(f"Reference position: X={x_ref:.2f}, Y={y_ref:.2f}, Z={z_ref:.2f} um")
    print()

    # Get folder path for saving images
    folder = FLIMageCont.get_val_sendCommand("State.Files.pathName")
    name_stem = FLIMageCont.get_val_sendCommand("State.Files.baseName")

    # Step 1: Acquire reference image
    print("Step 1: Acquiring reference image...")
    FLIMageCont.acquisition_include_connect_wait()
    sleep(1)

    # Get reference image path
    flimlist = get_flimfile_list(os.path.join(folder, f"{name_stem}001.flim"))
    if len(flimlist) == 0:
        raise FileNotFoundError("Reference image not found after acquisition")
    ref_image_path = flimlist[0]
    print(f"Reference image: {ref_image_path}")
    print()

    # Step 2: Move X direction and acquire image
    print(f"Step 2: Moving X direction by {move_distance_um} um and acquiring image...")
    FLIMageCont.relative_zyx_um = [0, 0, move_distance_um]  # [z, y, x] in um
    FLIMageCont.go_to_relative_pos_motor_checkstate()
    sleep(1)

    FLIMageCont.flim.sendCommand('State.Files.fileCounter = 1')
    FLIMageCont.acquisition_include_connect_wait()
    sleep(1)

    # Get X-moved image path
    flimlist = get_flimfile_list(os.path.join(folder, f"{name_stem}002.flim"))
    if len(flimlist) == 0:
        raise FileNotFoundError("X-moved image not found after acquisition")
    x_image_path = flimlist[-1]  # Get the latest one
    print(f"X-moved image: {x_image_path}")
    print()

    # Step 3: Return to reference position
    print("Step 3: Returning to reference position...")
    FLIMageCont.go_to_absolute_pos_motor_checkstate(x_ref, y_ref, z_ref)
    sleep(1)
    print()

    # Step 4: Move Y direction and acquire image
    print(f"Step 4: Moving Y direction by {move_distance_um} um and acquiring image...")
    FLIMageCont.relative_zyx_um = [0, move_distance_um, 0]  # [z, y, x] in um
    FLIMageCont.go_to_relative_pos_motor_checkstate()
    sleep(1)

    FLIMageCont.flim.sendCommand('State.Files.fileCounter = 2')
    FLIMageCont.acquisition_include_connect_wait()
    sleep(1)

    # Get Y-moved image path
    flimlist = get_flimfile_list(os.path.join(folder, f"{name_stem}003.flim"))
    if len(flimlist) == 0:
        raise FileNotFoundError("Y-moved image not found after acquisition")
    y_image_path = flimlist[-1]  # Get the latest one
    print(f"Y-moved image: {y_image_path}")
    print()

    # Step 5: Align images and measure pixel shifts
    print("Step 5: Aligning images and measuring pixel shifts...")
    
    # Align reference and X-moved images
    print("  Aligning reference and X-moved images...")
    relative_zyx_um_x, _, shifts_zyx_pixel_x = align_two_flimfile(
        ref_image_path, x_image_path, ch=ch_1or2 - 1, return_pixel=True
    )
    print(f"  X shift: {relative_zyx_um_x[2]:.2f} um (pixel shift: {shifts_zyx_pixel_x[-1][2]:.2f} pixels)")
    
    # Align reference and Y-moved images
    print("  Aligning reference and Y-moved images...")
    relative_zyx_um_y, _, shifts_zyx_pixel_y = align_two_flimfile(
        ref_image_path, y_image_path, ch=ch_1or2 - 1, return_pixel=True
    )
    print(f"  Y shift: {relative_zyx_um_y[1]:.2f} um (pixel shift: {shifts_zyx_pixel_y[-1][1]:.2f} pixels)")
    print()

    # Step 6: Calculate FOV size
    print("Step 6: Calculating FOV size...")
    
    # Get current zoom and pixel dimensions
    zoom = FLIMageCont.zoom
    pixels_x = FLIMageCont.pixelsPerLine
    pixels_y = FLIMageCont.get_val_sendCommand("State.Acq.linesPerFrame")
    
    print(f"  Current zoom: {zoom}")
    print(f"  Pixels per line: {pixels_x}")
    print(f"  Lines per frame: {pixels_y}")
    
    # Calculate pixel shift in pixels
    pixel_shift_x = abs(shifts_zyx_pixel_x[-1][2])
    pixel_shift_y = abs(shifts_zyx_pixel_y[-1][1])
    
    # Calculate FOV size for zoom=1
    # FOV_size_zoom1 = (move_distance_um / pixel_shift) * pixels * zoom
    if pixel_shift_x > 0:
        fov_x_zoom1 = (move_distance_um / pixel_shift_x) * pixels_x * zoom
    else:
        raise ValueError("X pixel shift is zero or negative. Cannot calculate FOV.")
    
    if pixel_shift_y > 0:
        fov_y_zoom1 = (move_distance_um / pixel_shift_y) * pixels_y * zoom
    else:
        raise ValueError("Y pixel shift is zero or negative. Cannot calculate FOV.")
    
    print(f"  Calculated FOV X (zoom=1): {fov_x_zoom1:.2f} um")
    print(f"  Calculated FOV Y (zoom=1): {fov_y_zoom1:.2f} um")
    print()

    # Step 7: Save to XYsize.ini
    print("Step 7: Saving calibration results to XYsize.ini...")
    
    if xy_size_ini_path is None:
        xy_size_ini_path = os.path.join(
            os.path.dirname(ini_path),
            "XYsize.ini"
        )
    
    # Create or update XYsize.ini
    config = configparser.ConfigParser()
    
    # Read existing file if it exists
    if os.path.exists(xy_size_ini_path):
        config.read(xy_size_ini_path)
    
    # Set values
    if 'Size' not in config:
        config.add_section('Size')
    
    config['Size']['X_zoom1_um'] = str(fov_x_zoom1)
    config['Size']['Y_zoom1_um'] = str(fov_y_zoom1)
    
    # Write to file
    with open(xy_size_ini_path, 'w') as configfile:
        config.write(configfile)
    
    print(f"  Saved to: {xy_size_ini_path}")
    print(f"  X_zoom1_um = {fov_x_zoom1:.2f}")
    print(f"  Y_zoom1_um = {fov_y_zoom1:.2f}")
    print()

    print("=" * 60)
    print("Calibration completed successfully!")
    print("=" * 60)

    #close the FLIMage controller
    # FLIMageCont.flim.close()
    return fov_x_zoom1, fov_y_zoom1


def apply_calibration_to_txt(
    source_adjusted_text_path: str,
    target_txt_path_list: list,
    parameter_list: list = [
        "State.Acq.fillFraction",
        "State.Acq.scanFraction",
        "State.Acq.ScanDelay",
        "State.Acq.EOMDelay",
        "State.Acq.msPerLine",
        "State.Acq.FOV_default",
    ]
    ) -> None:
    """
    Copy specified parameters from source_adjusted_text_path to files in target_txt_path_list.

    This function reads parameter values from the source file and replaces
    corresponding lines in target files.

    """

    print("=" * 60)
    print("Applying calibration parameters to FLIMage setting files")
    print("=" * 60)
    print(f"Source file: {source_adjusted_text_path}")
    print(f"Target files: {len(target_txt_path_list)} file(s)")
    print(f"Parameters to copy: {len(parameter_list)} parameter(s)")
    print()
    
    # Step 1: Read parameter values from source file
    print("Step 1: Reading parameters from source file...")
    if not os.path.exists(source_adjusted_text_path):
        raise FileNotFoundError(f"Source file not found: {source_adjusted_text_path}")
    
    parameter_values = {}
    with open(source_adjusted_text_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Check if this line contains any of the parameters we're looking for
        for param in parameter_list:
            if line.startswith(param + ' = '):
                # Extract the value after ' = '
                value_str = line.split(' = ', 1)[1]
                parameter_values[param] = value_str
                print(f"  Found {param} = {value_str}")
                break
    
    # Check if all parameters were found
    missing_params = [p for p in parameter_list if p not in parameter_values]
    if missing_params:
        print(f"\nWarning: The following parameters were not found in source file:")
        for param in missing_params:
            print(f"  - {param}")
        print()
    
    if not parameter_values:
        raise ValueError("No parameters found in source file. Check parameter names.")
    
    print(f"  Found {len(parameter_values)} parameter(s)")
    print()
    
    # Step 2: Apply parameters to each target file
    print("Step 2: Applying parameters to target files...")
    for target_path in target_txt_path_list:
        print(f"\n  Processing: {os.path.basename(target_path)}")
        
        if not os.path.exists(target_path):
            print(f"    Warning: File not found, skipping: {target_path}")
            continue
        
        # Read target file
        with open(target_path, 'r', encoding='utf-8') as f:
            target_lines = f.readlines()
        
        # Replace matching lines
        modified = False
        for i, line in enumerate(target_lines):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                continue
            
            # Check if this line contains any parameter we want to replace
            for param in parameter_values.keys():
                if line_stripped.startswith(param + ' = '):
                    # Replace the line
                    new_line = f"{param} = {parameter_values[param]}\n"
                    if target_lines[i] != new_line:
                        target_lines[i] = new_line
                        modified = True
                        print(f"    Updated {param}")
                    break
        
        # Write back to file if modified
        if modified:
            with open(target_path, 'w', encoding='utf-8') as f:
                f.writelines(target_lines)
            print(f"    Saved: {target_path}")
        else:
            print(f"    No changes needed")
    
    print()
    print("=" * 60)
    print("Parameter application completed!")
    print("=" * 60)



if __name__ == "__main__":
    import glob
    target_txt_path_candidates = glob.glob(r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\*.txt")
    exclude_file_head_list = ["Default", "FLIM_deviceFile", "FLIM_init"]
    target_txt_path_list = [path for path in target_txt_path_candidates if not any(head in path for head in exclude_file_head_list)]
    print(f"Target files: {len(target_txt_path_list)} file(s)")
    #print file names, no parent folder path. new line for each file
    print(f"Target files: \n")
    
    additional_target_txt_path_list = [
        r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\Default-4_0_33_working_20260127.txt"
        r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\Default-4_0_36_working_20260127.txt"
        ]
    target_txt_path_list.extend(additional_target_txt_path_list)

    for each_file in [os.path.basename(path) for path in target_txt_path_list]:
        print(f"  {each_file}")

    apply_calibration_to_txt(
        source_adjusted_text_path=r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\Default-4_0_33_working_20260127.txt",
        target_txt_path_list=target_txt_path_list,
    )


if __name__ == "__main__2":
    # Example usage in IPython interactive window:
    # 
    # from calibrate_FOV_XY_motor import calibrate_fov_xy_motor
    # 
    # # Use default settings (30 um movement, channel 1)
    # fov_x, fov_y = calibrate_fov_xy_motor()
    # 
    # # Or with custom settings:
    # fov_x, fov_y = calibrate_fov_xy_motor(
    #     move_distance_um=50.0,
    #     ch_1or2=2,
    #     base_name="my_calibration"
    # )
    
    # Default execution for testing
    import datetime
    yyyymmdd_hhmmss = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        fov_x, fov_y = calibrate_fov_xy_motor(
            move_distance_um=40.0,
            ch_1or2=1,
            base_name=f"calibration_FOV_1_{yyyymmdd_hhmmss}"
        )
        print(f"\nFinal results:")
        print(f"  X_zoom1_um = {fov_x:.2f}")
        print(f"  Y_zoom1_um = {fov_y:.2f}")
    except Exception as e:
        print(f"\nError during calibration: {e}")
        import traceback

        traceback.print_exc()

