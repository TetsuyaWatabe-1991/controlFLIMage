#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Uncaging Titration GUI Analysis

This script performs uncaging titration analysis with enhanced GUI features
using functions from GCaMPanalysis_tkinter_file_selection.py.

Features:
- ROI visualization and confirmation
- Automatic ROI loading if previously saved
- GUI-based user interaction
- Quality detection for automatic ROI fallback
- CSV output with comprehensive results

Created on: 2025
@author: Enhanced Analysis Tool
"""

import sys
sys.path.append(r"..\\")
from datetime import datetime
import os
import glob
import pandas as pd
import numpy as np
from FLIMageAlignment import get_flimfile_list
from FLIMageFileReader2 import FileReader
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from skimage.draw import polygon
import re
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
import json
from typing import Tuple, List, Dict, Optional


# Functions copied from GCaMPanalysis_tkinter_file_selection.py to avoid import issues
class NextFileDialog:
    def __init__(self, roi_image_path, results_dict):
        self.result = None  # True for continue, False for redo, None for finish
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("ROI Analysis Results")
        
        # Get the position of the current matplotlib window
        try:
            mng = plt.get_current_fig_manager()
            if hasattr(mng, 'window'):
                # Get the position of the matplotlib window
                fig_x = mng.window.winfo_x()
                fig_y = mng.window.winfo_y()
                # Use this position for our dialog
                self.root.geometry(f"+{fig_x}+{fig_y}")
        except:
            # If we can't get the matplotlib window position, center the dialog
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'+{x}+{y}')
        
        # Make sure window stays on top and gets focus
        self.root.attributes('-topmost', True)
        self.root.lift()
        self.root.focus_force()
        
        # Bind keyboard events
        self.root.bind('y', lambda e: self.set_result(True))
        self.root.bind('Y', lambda e: self.set_result(True))
        self.root.bind('n', lambda e: self.set_result(False))
        self.root.bind('N', lambda e: self.set_result(False))
        self.root.bind('f', lambda e: self.set_result(None))
        self.root.bind('F', lambda e: self.set_result(None))
        
        # Load and display ROI image
        img = Image.open(roi_image_path)
        # Resize image if too large
        max_size = (799, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        
        # Create image label
        img_label = ttk.Label(self.root, image=self.photo)
        img_label.pack(padx=9, pady=10)
        
        # Create results text
        results_text = f"""
Analysis Results:
----------------
Power: {results_dict['power_mw']} mW
Pulse Width: {results_dict['pulse_width']} ms

Shaft ROI (Uncaging):
- Pre intensity: {results_dict['pre_shaft']:.0f}
- Unc intensity: {results_dict['unc_shaft']:.0f}
- F/F-1: {results_dict['shaft_f_f0']:.2f}

Spine ROI (Uncaging):
- Pre intensity: {results_dict['pre_spine']:.0f}
- Unc intensity: {results_dict['unc_spine']:.0f}
- F/F-1: {results_dict['spine_f_f0']:.2f}

Press 'Y' to continue, 'N' to redo ROI, or 'F' to finish analysis
"""
        text_label = ttk.Label(self.root, text=results_text, justify=tk.LEFT)
        text_label.pack(padx=9, pady=10)
        
        # Create buttons frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=9)
        
        # Create buttons
        ttk.Button(button_frame, text="Accept and Continue (Y)", 
                  command=lambda: self.set_result(True)).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="Redo ROI (N)", 
                  command=lambda: self.set_result(False)).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="Finish Analysis (F)", 
                  command=lambda: self.set_result(None)).pack(side=tk.LEFT, padx=4)
        
    def set_result(self, value):
        self.result = value
        self.root.quit()
        
    def show(self):
        # Ensure window is on top and has focus when shown
        self.root.attributes('-topmost', True)
        self.root.lift()
        self.root.focus_force()
        
        self.root.mainloop()
        self.root.destroy()
        return self.result

def detect_image_quality(image: np.ndarray, threshold: float = 0.3, print_info: bool = True) -> bool:
    """
    Detect if an image is too noisy/low quality.
    
    Args:
        image: Input image array
        threshold: Threshold for quality detection (higher means stricter quality requirement)
        
    Returns:
        bool: True if image quality is acceptable, False if too noisy
    """
    # Calculate mean and standard deviation
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    # Calculate coefficient of variation (CV = std/mean)
    # Lower CV typically indicates better signal-to-noise ratio
    cv = std_intensity / (mean_intensity + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Calculate histogram-based metrics
    hist, bins = np.histogram(image.flatten(), bins=50)
    hist = hist / np.sum(hist)  # Normalize histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Image entropy
    
    # Combine metrics (you can adjust these weights)
    quality_score = 1.0 / (cv + 0.1) + entropy * 0.1
    
    if print_info:
        print(f"CV: {cv:.2f}, Entropy: {entropy:.2f}, Quality Score: {quality_score:.2f}")  
    
    return quality_score > threshold


def save_roi_info_enhanced(savefolder: str, basename: str, roi_points: np.ndarray, roi_points_spine: np.ndarray) -> str:
    """Save ROI coordinates to a JSON file with both shaft and spine ROIs"""
    roi_info = {
        'shaft': roi_points.tolist(),
        'spine': roi_points_spine.tolist()
    }
    roi_file = os.path.join(savefolder, basename[:-5] + "_roi.json")
    with open(roi_file, 'w') as f:
        json.dump(roi_info, f)
    return roi_file

def load_roi_info_enhanced(roi_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ROI coordinates from a JSON file with both shaft and spine ROIs"""
    with open(roi_file, 'r') as f:
        roi_info = json.load(f)
    return np.array(roi_info['shaft']), np.array(roi_info['spine'])

def get_roi_masks(roi_points: np.ndarray, roi_points_spine: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create masks from ROI points for both shaft and spine"""
    # Shaft mask
    r, c = roi_points[:, 0], roi_points[:, 1]
    mask_rows, mask_cols = polygon(c, r, image_shape)
    mask = np.zeros(image_shape, dtype=bool)
    mask[mask_rows, mask_cols] = True
    
    # Spine mask
    r_spine, c_spine = roi_points_spine[:, 0], roi_points_spine[:, 1]
    mask_rows_spine, mask_cols_spine = polygon(c_spine, r_spine, image_shape)
    mask_spine = np.zeros(image_shape, dtype=bool)
    mask_spine[mask_rows_spine, mask_cols_spine] = True
    
    return mask, mask_spine

def visualize_rois(fig: plt.Figure, ax: plt.Axes, roi_points: np.ndarray, roi_points_spine: np.ndarray) -> None:
    """Visualize both shaft and spine ROIs on the given figure"""
    # Shaft ROI
    r, c = roi_points[:, 0], roi_points[:, 1]
    ax.plot(list(r) + [list(r)[0]], list(c) + [list(c)[0]], "m", lw=2)
    plt.text(.05, .95, 'Shaft', c="m", size=10, ha='left', va='top', transform=ax.transAxes)
    
    # Spine ROI
    r_spine, c_spine = roi_points_spine[:, 0], roi_points_spine[:, 1]
    ax.plot(list(r_spine) + [list(r_spine)[0]], list(c_spine) + [list(c_spine)[0]], "c", lw=2)
    plt.text(.05, .85, 'Spine', c="c", size=10, ha='left', va='top', transform=ax.transAxes)
    
    ax.set_title("ROI")
    ax.axis("off")

def create_default_rois(image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create default shaft and spine ROIs in the center of the image.
    
    Args:
        image_shape: Shape of the image (height, width)
        
    Returns:
        Tuple of (shaft_roi_points, spine_roi_points) as numpy arrays
    """
    height, width = image_shape
    center_y, center_x = height // 2, width // 2
    roi_size = min(height, width) // 6
    
    # Create shaft ROI (left side)
    shaft_points = np.array([
        [center_y - roi_size//2, center_x - roi_size],
        [center_y - roi_size//2, center_x - roi_size//2],
        [center_y + roi_size//2, center_x - roi_size//2],
        [center_y + roi_size//2, center_x - roi_size]
    ])
    
    # Create spine ROI (right side)
    spine_points = np.array([
        [center_y - roi_size//2, center_x + roi_size//2],
        [center_y - roi_size//2, center_x + roi_size],
        [center_y + roi_size//2, center_x + roi_size],
        [center_y + roi_size//2, center_x + roi_size//2]
    ])
    
    return shaft_points, spine_points

def analyze_uncaging_titration(filelist: List[str], pow_slope: float, pow_intcpt: float, quality_threshold: float = 0.3):
    """
    Analyze uncaging titration data with enhanced GUI features.
    
    Args:
        filelist: List of file paths to analyze
        pow_slope: Power calibration slope
        pow_intcpt: Power calibration intercept
        quality_threshold: Threshold for image quality detection
    """
    # Group files by pattern
    grouped_files = defaultdict(list)
    pattern = re.compile(r"(.*)_\d{3}\.flim$")
    
    for filepath in filelist:
        filename = os.path.basename(filepath)
        match = pattern.search(filename)
        if match:
            group_key = match.group(1)
            grouped_files[group_key].append(filepath)
    
    print("Found file groups:")
    for group, files in grouped_files.items():
        print(f"\nGroup: {group}")
        for each_file in files:
            print(f"  {each_file}")
    
    # Initialize results storage
    results_dict = {}
    should_continue_analysis = True
    
    # CSV header
    result_txt = "group,pow_mw_round,pre_mean_intensity,post_mean_intensity,dend_F_F0,pre_spine_intensity,post_spine_intensity,spine_F_F0,roi_file,roi_image\n"
    
    for group, files in grouped_files.items():
        if not should_continue_analysis:
            break
            
        print(f"\nProcessing group: {group}")
        first_file_in_group = True
        roi_points = None
        roi_points_spine = None
        mask = None
        mask_spine = None
        roi_file = None
        roi_image_file = None
        
        for each_file in files:
            if not should_continue_analysis:
                break
                
            print(f"Processing file: {each_file}")
            
            folder = os.path.dirname(each_file)
            savefolder = os.path.join(folder, "plot")
            os.makedirs(savefolder, exist_ok=True)
            basename = os.path.basename(each_file)
            
            try:
                # Load and process image data
                uncaging_iminfo = FileReader()
                uncaging_iminfo.read_imageFile(each_file, True)
                imagearray = np.array(uncaging_iminfo.image)
                uncaging_x_y_0to1 = uncaging_iminfo.statedict["State.Uncaging.Position"]
                uncaging_pow = uncaging_iminfo.statedict["State.Uncaging.Power"]
                pulseWidth = int(uncaging_iminfo.statedict["State.Uncaging.pulseWidth"])
                center_y = imagearray.shape[-2] * uncaging_x_y_0to1[1]
                center_x = imagearray.shape[-3] * uncaging_x_y_0to1[0]
                
                # Get images for pre and uncaging time points
                GCpre = imagearray[0,0,0,:,:,:].sum(axis=-1)  # First frame
                GCunc = imagearray[3,0,0,:,:,:].sum(axis=-1)  # Uncaging frame
                Tdpre = imagearray[0,0,1,:,:,:].sum(axis=-1)
                
                GC_pre_med = median_filter(GCpre, size=3)
                GC_unc_med = median_filter(GCunc, size=3)
                
                pow_mw = pow_slope * uncaging_pow + pow_intcpt
                pow_mw_coherent = pow_mw/3
                pow_mw_round = round(pow_mw_coherent, 1)
                
                # Handle ROI definition only for the first file in each group
                if first_file_in_group:
                    roi_file = os.path.join(savefolder, group + "_roi.json")  # Use group name for ROI file
                    roi_image_file = os.path.join(savefolder, group + "_roi.png")
                    
                    re_define_roi = True
                    while re_define_roi and should_continue_analysis:
                        is_good_quality = detect_image_quality(GC_pre_med, quality_threshold)
                        
                        # Check if we can reuse existing ROIs
                        if os.path.exists(roi_file) and os.path.exists(roi_image_file):
                            print(f"Found existing ROI for group {group}")
                            roi_points, roi_points_spine = load_roi_info_enhanced(roi_file)
                            
                            try:
                                plt.close('all')
                                fig, ax = plt.subplots()
                                ax.imshow(Tdpre, cmap='gray')
                                ax.plot(center_x, center_y, 'ro', markersize=2)
                                visualize_rois(fig, ax, roi_points, roi_points_spine)
                                plt.savefig(roi_image_file, dpi=150, bbox_inches="tight")
                                plt.close(fig)
                            except Exception as e:
                                print(f"Warning: Could not update ROI visualization: {str(e)}")
                            
                            re_define_roi = False
                        else:
                            if not is_good_quality:
                                print(f"Low quality image detected for group {group}. Using default ROIs.")
                                roi_points, roi_points_spine = create_default_rois(GC_pre_med.shape)
                            else:
                                try:
                                    plt.close('all')
                                    fig, ax = plt.subplots()
                                    ax.imshow(Tdpre, cmap='gray')
                                    ax.plot(center_x, center_y, 'ro', markersize=2)
                                    mng = plt.get_current_fig_manager()
                                    try:
                                        mng.window.state('zoomed')
                                    except:
                                        pass
                                    ax.set_title("shaft")
                                    
                                    roi_points = np.array(plt.ginput(n=4, timeout=0))
                                    r, c = roi_points[:, 0], roi_points[:, 1]
                                    ax.plot(list(r) + [list(r)[0]], list(c) + [list(c)[0]], "m", lw=2)
                                    
                                    ax.set_title("SPINE")
                                    roi_points_spine = np.array(plt.ginput(n=4, timeout=0))
                                    r_spine, c_spine = roi_points_spine[:, 0], roi_points_spine[:, 1]
                                    ax.plot(list(r_spine) + [list(r_spine)[0]], 
                                           list(c_spine) + [list(c_spine)[0]], "c", lw=2)
                                    plt.close(fig)
                                except Exception as e:
                                    print(f"Error during manual ROI definition: {str(e)}")
                                    raise
                            
                            try:
                                plt.close('all')
                                fig, ax = plt.subplots()
                                ax.imshow(Tdpre, cmap='gray')
                                ax.plot(center_x, center_y, 'ro', markersize=2)
                                visualize_rois(fig, ax, roi_points, roi_points_spine)
                                plt.savefig(roi_image_file, dpi=150, bbox_inches="tight")
                                plt.close(fig)
                                
                                save_roi_info_enhanced(savefolder, group + ".flim", roi_points, roi_points_spine)
                            except Exception as e:
                                print(f"Error saving ROI information: {str(e)}")
                                raise
                        
                        # Calculate masks once for the group
                        mask, mask_spine = get_roi_masks(roi_points, roi_points_spine, GC_pre_med.shape)
                        
                        # Calculate results for the first file
                        pre_mean_intensity = round(GC_pre_med[mask].sum(), 1)
                        post_mean_intensity = round(GC_unc_med[mask].sum(), 1)
                        dend_F_F0 = post_mean_intensity / pre_mean_intensity if pre_mean_intensity > 0 else 0
                        
                        pre_spine_intensity = round(GC_pre_med[mask_spine].sum(), 1)
                        post_spine_intensity = round(GC_unc_med[mask_spine].sum(), 1)
                        spine_F_F0 = post_spine_intensity / pre_spine_intensity if pre_spine_intensity > 0 else 0
                        
                        # Store results in dictionary for GUI display
                        results_dict[each_file] = {
                            'group': group,
                            'power_mw': pow_mw_round,
                            'pulse_width': pulseWidth,
                            'pre_shaft': int(pre_mean_intensity),
                            'unc_shaft': int(post_mean_intensity),
                            'shaft_f_f0': float(f"{dend_F_F0:.2f}"),
                            'pre_spine': int(pre_spine_intensity),
                            'unc_spine': int(post_spine_intensity),
                            'spine_f_f0': float(f"{spine_F_F0:.2f}"),
                            'roi_file': roi_file,
                            'roi_image': roi_image_file,
                            'is_low_quality': not is_good_quality
                        }
                        
                        # Show dialog for confirmation
                        dialog = NextFileDialog(roi_image_file, results_dict[each_file])
                        result = dialog.show()
                        
                        if result is None:
                            print(f"\nAnalysis terminated by user.")
                            should_continue_analysis = False
                            break
                        elif result:
                            re_define_roi = False
                            first_file_in_group = False
                        else:
                            re_define_roi = True
                            if os.path.exists(roi_file):
                                os.remove(roi_file)
                            if os.path.exists(roi_image_file):
                                os.remove(roi_image_file)
                            results_dict.pop(each_file, None)
                            continue
                        
                        # Add to CSV result string
                        result_txt += f"{group},{pow_mw_round},{pre_mean_intensity},{post_mean_intensity},{dend_F_F0:.2f},{pre_spine_intensity},{post_spine_intensity},{spine_F_F0:.2f},{roi_file},{roi_image_file}\n"
                        
                        print(f"Processed: {basename}")
                        print(f"  Shaft F/F0: {dend_F_F0:.2f}")
                        print(f"  Spine F/F0: {spine_F_F0:.2f}")
                
                # For subsequent files in the same group, use the same ROI
                else:
                    if roi_points is None or roi_points_spine is None or mask is None or mask_spine is None:
                        print(f"Error: ROI not defined for group {group}")
                        continue
                    
                    # Calculate results using the same ROI
                    pre_mean_intensity = round(GC_pre_med[mask].sum(), 1)
                    post_mean_intensity = round(GC_unc_med[mask].sum(), 1)
                    dend_F_F0 = post_mean_intensity / pre_mean_intensity if pre_mean_intensity > 0 else 0
                    
                    pre_spine_intensity = round(GC_pre_med[mask_spine].sum(), 1)
                    post_spine_intensity = round(GC_unc_med[mask_spine].sum(), 1)
                    spine_F_F0 = post_spine_intensity / pre_spine_intensity if pre_spine_intensity > 0 else 0
                    
                    # Store results in dictionary
                    results_dict[each_file] = {
                        'group': group,
                        'power_mw': pow_mw_round,
                        'pulse_width': pulseWidth,
                        'pre_shaft': int(pre_mean_intensity),
                        'unc_shaft': int(post_mean_intensity),
                        'shaft_f_f0': float(f"{dend_F_F0:.2f}"),
                        'pre_spine': int(pre_spine_intensity),
                        'unc_spine': int(post_spine_intensity),
                        'spine_f_f0': float(f"{spine_F_F0:.2f}"),
                        'roi_file': roi_file,
                        'roi_image': roi_image_file,
                        'is_low_quality': False
                    }
                    
                    # Add to CSV result string
                    result_txt += f"{group},{pow_mw_round},{pre_mean_intensity},{post_mean_intensity},{dend_F_F0:.2f},{pre_spine_intensity},{post_spine_intensity},{spine_F_F0:.2f},{roi_file},{roi_image_file}\n"
                    
                    print(f"Processed: {basename}")
                    print(f"  Shaft F/F0: {dend_F_F0:.2f}")
                    print(f"  Spine F/F0: {spine_F_F0:.2f}")
                    
            except Exception as e:
                print(f"\nError processing {each_file}: {str(e)}")
                plt.close('all')
                continue
    
    # Save results to CSV
    if results_dict:
        # Use the last processed file's folder for saving results
        res_save_path = os.path.join(savefolder, "titration_result.csv")
        with open(res_save_path, "w") as file:
            file.write(result_txt)
        
        print(f"\nResults saved to: {res_save_path}")
        print(f"Total files analyzed: {len(results_dict)}")
        print("\nFinal Results:")
        print(result_txt)
    else:
        print("No results to save.")


if __name__ == "__main__":
    # Configuration parameters
    pow_slope = 0.198
    pow_intcpt = 0.154
    quality_threshold = 0.1  # Adjust based on your image quality requirements
    
    # File list - modify this path as needed
    filelist = glob.glob(r"\\RY-LAB-WS04\ImagingData\Tetsuya\20250606\*um_*.flim")
    
    # Alternative example paths (uncomment as needed):
    # filelist = glob.glob(r"G:\ImagingData\Tetsuya\20250506\E4_roomair_*.flim")
    # filelist = glob.glob(r"G:\ImagingData\Tetsuya\20250508\titration_*.flim")
    
    print(f"Found {len(filelist)} files to analyze")
    
    if len(filelist) > 0:
        analyze_uncaging_titration(filelist, pow_slope, pow_intcpt, quality_threshold)
    else:
        print("No files found. Please check the file path pattern.") 