# %%
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:46:20 2025

@author: yasudalab
"""

import sys
sys.path.append(r"C:\Users\yasudalab\Documents\Tetsuya_GIT\controlFLIMage")
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

Shaft ROI:
- Pre intensity: {results_dict['pre_shaft']:.0f}
- Post intensity: {results_dict['post_shaft']:.0f}
- F/F-1: {results_dict['shaft_f_f0']:.2f}

Spine ROI:
- Pre intensity: {results_dict['pre_spine']:.0f}
- Post intensity: {results_dict['post_spine']:.0f}
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

def save_roi_info(savefolder: str, basename: str, roi_points: np.ndarray, roi_points_spine: np.ndarray) -> str:
    """Save ROI coordinates to a JSON file"""
    roi_info = {
        'shaft': roi_points.tolist(),
        'spine': roi_points_spine.tolist()
    }
    roi_file = os.path.join(savefolder, basename[:-5] + "_roi.json")
    with open(roi_file, 'w') as f:
        json.dump(roi_info, f)
    return roi_file

def load_roi_info(roi_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ROI coordinates from a JSON file"""
    with open(roi_file, 'r') as f:
        roi_info = json.load(f)
    return np.array(roi_info['shaft']), np.array(roi_info['spine'])

def get_roi_masks(roi_points: np.ndarray, roi_points_spine: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Create masks from ROI points"""
    r, c = roi_points[:, 0], roi_points[:, 1]
    r_spine, c_spine = roi_points_spine[:, 0], roi_points_spine[:, 1]
    
    mask_rows, mask_cols = polygon(c, r, image_shape)
    mask = np.zeros(image_shape, dtype=bool)
    mask[mask_rows, mask_cols] = True
    
    mask_rows_spine, mask_cols_spine = polygon(c_spine, r_spine, image_shape)
    mask_spine = np.zeros(image_shape, dtype=bool)
    mask_spine[mask_rows_spine, mask_cols_spine] = True
    
    return mask, mask_spine

def visualize_rois(fig: plt.Figure, ax: plt.Axes, roi_points: np.ndarray, roi_points_spine: np.ndarray) -> None:
    """Visualize ROIs on the given figure"""
    r, c = roi_points[:, 0], roi_points[:, 1]
    r_spine, c_spine = roi_points_spine[:, 0], roi_points_spine[:, 1]
    
    ax.plot(list(r) + [list(r)[0]], list(c) + [list(c)[0]], "m", lw=2)
    ax.plot(list(r_spine) + [list(r_spine)[0]], list(c_spine) + [list(c_spine)[0]], "c", lw=2)
    
    plt.text(.05, .95, 'Shaft', c="m", size=10, ha='left', va='top', transform=ax.transAxes)
    plt.text(.05, .85, 'Spine', c="c", size=10, ha='left', va='top', transform=ax.transAxes)
    ax.set_title("ROI")
    ax.axis("off")

def create_pre_post_sets_from_filelist(filelist: List[str]) -> List[Dict[str, str]]:
    """
    Create sets of pre, uncaging, and post files from a file list.
    
    Args:
        filelist: List of file paths
        
    Returns:
        List of dictionaries containing pre, uncaging, and post file paths
    """
    uncaging_nth_list = []
    First = True
    imageshape = None
    
    # Detect uncaging frames
    for nth, file_path in enumerate(filelist):
        iminfo = FileReader()
        try:
            iminfo.read_imageFile(file_path, True) 
            imagearray = np.array(iminfo.image)
        except:
            print(f"\n\nCould not read {file_path}\n")
            continue
        
        if First:
            First = False
            imageshape = imagearray.shape
        
        if imagearray.shape == imageshape:
            pass
        else:
            if (imagearray.shape[0] > 29):
                print(f"{file_path} <- uncaging")
                uncaging_nth_list.append(nth)
    
    print(uncaging_nth_list)
    
    # Create pre/post sets
    pre_post_set_list = []
    for number, nth in enumerate(uncaging_nth_list[:-1]):
        pre_post_set_list.append({
            "pre": filelist[nth-1],
            "unc": filelist[nth],
            "post": filelist[uncaging_nth_list[number + 1] - 2],
        })
    
    if len(uncaging_nth_list) > 0:
        pre_post_set_list.append({
            "pre": filelist[uncaging_nth_list[-1] - 1],
            "unc": filelist[uncaging_nth_list[-1]],
            "post": filelist[-1],
        })
    
    return pre_post_set_list

def get_all_pre_post_sets(base_path: str, filename_sample: str) -> List[Dict[str, str]]:
    """
    Get all pre/post sets from files matching the pattern in the given directory.
    
    Args:
        base_path: Path to the directory containing the files
        filename_sample: Pattern to match files (e.g., "*002.flim")
        
    Returns:
        List of dictionaries containing pre, uncaging, and post file paths
    """
    # Get list of files matching the pattern
    one_of_file_list = glob.glob(os.path.join(os.path.dirname(base_path), filename_sample))
    all_pre_post_sets = []
    
    # Process each file to create pre/post sets
    for each_firstfilepath in one_of_file_list:
        filelist = get_flimfile_list(each_firstfilepath)
        pre_post_sets = create_pre_post_sets_from_filelist(filelist)
        all_pre_post_sets.extend(pre_post_sets)
    
    return all_pre_post_sets

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

def create_default_roi(image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create default ROIs (1/4 size squares) in the center of the image.
    
    Args:
        image_shape: Shape of the image (height, width)
        
    Returns:
        Tuple of (shaft_roi_points, spine_roi_points) as numpy arrays
    """
    height, width = image_shape
    center_y, center_x = height // 2, width // 2
    roi_size = min(height, width) // 4
    
    # Create shaft ROI (slightly larger)
    shaft_points = np.array([
        [center_y - roi_size//2, center_x - roi_size//2],
        [center_y - roi_size//2, center_x + roi_size//2],
        [center_y + roi_size//2, center_x + roi_size//2],
        [center_y + roi_size//2, center_x - roi_size//2]
    ])
    
    # Create spine ROI (slightly smaller and offset)
    spine_offset = roi_size // 3
    spine_points = np.array([
        [center_y - roi_size//4, center_x - roi_size//4 + spine_offset],
        [center_y - roi_size//4, center_x + roi_size//4 + spine_offset],
        [center_y + roi_size//4, center_x + roi_size//4 + spine_offset],
        [center_y + roi_size//4, center_x - roi_size//4 + spine_offset]
    ])
    
    return shaft_points, spine_points

def get_pre_post_set_list(all_pre_post_sets: List[Dict[str, str]], pow_slope: float, pow_intcpt: float, quality_threshold: float = 0.3) -> None:
    """
    Process pre/post sets and perform analysis.
    """
    # Initialize results dictionary
    results_dict = {}
    
    # Add a flag variable before the loops
    should_continue_analysis = True
    
    # Process each pre/post set
    for each_set in all_pre_post_sets:
        if not should_continue_analysis:
            break
            
        re_define_roi = True
        
        while re_define_roi and should_continue_analysis:
            each_file = each_set["unc"]
            folder = os.path.dirname(each_file)
            savefolder = os.path.join(folder, "plot")
            os.makedirs(savefolder, exist_ok=True)
            basename = os.path.basename(each_file)
            
            # Check if ROI file exists
            roi_file = os.path.join(savefolder, basename[:-5] + "_roi.json")
            roi_image_file = os.path.join(savefolder, basename[:-5] + "_roi.png")
            
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
                
                GCpre = imagearray[0,0,0,:,:,:].sum(axis=-1)
                GCunc = imagearray[3,0,0,:,:,:].sum(axis=-1)
                Tdpre = imagearray[0,0,1,:,:,:].sum(axis=-1)
                
                GC_pre_med = median_filter(GCpre, size=3)
                GC_unc_med = median_filter(GCunc, size=3)
                
                GC_noZeros = GC_pre_med.copy()
                GC_noZeros[GC_pre_med == 0] = 1
                GCF_F0 = (GC_unc_med/GC_noZeros)
                GCF_F0[GC_pre_med == 0] = 0
                
                pow_mw = pow_slope * uncaging_pow + pow_intcpt
                pow_mw_coherent = pow_mw/3
                pow_mw_round = round(pow_mw_coherent, 1)
                
                # Check image quality
                is_good_quality = detect_image_quality(GC_pre_med, quality_threshold)
                
                # Check if we can reuse existing ROIs
                if os.path.exists(roi_file) and os.path.exists(roi_image_file):
                    print(f"\nFound existing ROI for {basename}")
                    roi_points, roi_points_spine = load_roi_info(roi_file)
                    mask, mask_spine = get_roi_masks(roi_points, roi_points_spine, GC_pre_med.shape)
                    
                    try:
                        # Create visualization
                        plt.close('all')  # Close any existing figures
                        fig, ax = plt.subplots()
                        ax.imshow(Tdpre, cmap='gray')
                        ax.plot(center_x, center_y, 'ro', markersize=2)
                        visualize_rois(fig, ax, roi_points, roi_points_spine)
                        plt.savefig(roi_image_file, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                    except Exception as e:
                        print(f"Warning: Could not update ROI visualization: {str(e)}")
                        # Continue with existing ROI image if available
                    
                    re_define_roi = False
                else:
                    # Always start with default ROIs for low quality images
                    if not is_good_quality:
                        print(f"\nLow quality image detected for {basename}. Using default ROIs.")
                        roi_points, roi_points_spine = create_default_roi(GC_pre_med.shape)
                    else:
                        try:
                            # Let user define new ROIs
                            plt.close('all')  # Close any existing figures
                            fig, ax = plt.subplots()
                            ax.imshow(Tdpre, cmap='gray')
                            ax.plot(center_x, center_y, 'ro', markersize=2)
                            mng = plt.get_current_fig_manager()
                            try:
                                mng.window.state('zoomed')
                            except:
                                pass
                            ax.set_title("shaft")
                            
                            # Get shaft ROI
                            roi_points = np.array(plt.ginput(n=4, timeout=0))
                            r, c = roi_points[:, 0], roi_points[:, 1]
                            ax.plot(list(r) + [list(r)[0]], list(c) + [list(c)[0]], "m", lw=2)
                            
                            # Get spine ROI
                            ax.set_title("SPINE")
                            roi_points_spine = np.array(plt.ginput(n=4, timeout=0))
                            r_spine, c_spine = roi_points_spine[:, 0], roi_points_spine[:, 1]
                            ax.plot(list(r_spine) + [list(r_spine)[0]], 
                                   list(c_spine) + [list(c_spine)[0]], "c", lw=2)
                            plt.close(fig)
                        except Exception as e:
                            print(f"Error during manual ROI definition: {str(e)}")
                            raise  # Re-raise the exception to be caught by the outer try-except
                    
                    try:
                        # Create masks
                        mask, mask_spine = get_roi_masks(roi_points, roi_points_spine, GC_pre_med.shape)
                        
                        # Save ROI visualization and coordinates
                        plt.close('all')  # Close any existing figures
                        fig, ax = plt.subplots()
                        ax.imshow(Tdpre, cmap='gray')
                        ax.plot(center_x, center_y, 'ro', markersize=2)
                        visualize_rois(fig, ax, roi_points, roi_points_spine)
                        plt.savefig(roi_image_file, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        
                        # Save ROI coordinates
                        save_roi_info(savefolder, basename, roi_points, roi_points_spine)
                    except Exception as e:
                        print(f"Error saving ROI information: {str(e)}")
                        raise  # Re-raise the exception to be caught by the outer try-except
                
                # Calculate results
                pre_spine = GC_pre_med[mask_spine].sum()
                post_spine = GC_unc_med[mask_spine].sum()
                pre_shaft = GC_pre_med[mask].sum()
                post_shaft = GC_unc_med[mask].sum()
                
                # Calculate F/F0 ratios, handling zero denominators
                spine_f_f0 = post_spine / pre_spine if pre_spine > 0 else 0
                shaft_f_f0 = post_shaft / pre_shaft if pre_shaft > 0 else 0
                
                # Store results in dictionary
                results_dict[each_file] = {
                    'power_mw': pow_mw_round,
                    'pulse_width': pulseWidth,
                    'pre_shaft': int(pre_shaft),
                    'post_shaft': int(post_shaft),
                    'shaft_f_f0': float(f"{shaft_f_f0:.2f}"),
                    'pre_spine': int(pre_spine),
                    'post_spine': int(post_spine),
                    'spine_f_f0': float(f"{spine_f_f0:.2f}"),
                    'roi_file': roi_file,
                    'roi_image': roi_image_file,
                    'is_low_quality': not is_good_quality
                }
                
                # Show dialog
                dialog = NextFileDialog(roi_image_file, results_dict[each_file])
                result = dialog.show()

                if result in [None, True]:
                    # Save results to CSV
                    save_results_to_csv(results_dict, savefolder)
                    
                if result is None:  # User clicked "Finish Analysis"
                    print(f"\nAnalysis terminated by user. Results saved to: {os.path.join(savefolder, 'result.csv')}")
                    print(f"Total files analyzed: {len(results_dict)}")
                    should_continue_analysis = False
                    break
                elif result:  # User clicked "Accept and Continue"
                    re_define_roi = False
                else:  # User clicked "Redo ROI"
                    re_define_roi = True
                    if os.path.exists(roi_file):
                        os.remove(roi_file)
                    if os.path.exists(roi_image_file):
                        os.remove(roi_image_file)
                    results_dict.pop(each_file, None)
                    
                    try:
                        # If redoing ROI, always allow manual ROI definition regardless of quality
                        print("\nRedefining ROIs manually...")
                        plt.close('all')  # Close any existing figures
                        fig, ax = plt.subplots()
                        ax.imshow(Tdpre, cmap='gray')
                        ax.plot(center_x, center_y, 'ro', markersize=2)
                        mng = plt.get_current_fig_manager()
                        try:
                            mng.window.state('zoomed')
                        except:
                            pass
                        ax.set_title("shaft")
                        
                        # Get shaft ROI
                        roi_points = np.array(plt.ginput(n=4, timeout=0))
                        r, c = roi_points[:, 0], roi_points[:, 1]
                        ax.plot(list(r) + [list(r)[0]], list(c) + [list(c)[0]], "m", lw=2)
                        
                        # Get spine ROI
                        ax.set_title("SPINE")
                        roi_points_spine = np.array(plt.ginput(n=4, timeout=0))
                        r_spine, c_spine = roi_points_spine[:, 0], roi_points_spine[:, 1]
                        ax.plot(list(r_spine) + [list(r_spine)[0]], 
                               list(c_spine) + [list(c_spine)[0]], "c", lw=2)
                        plt.close(fig)
                        
                        # Create masks
                        mask, mask_spine = get_roi_masks(roi_points, roi_points_spine, GC_pre_med.shape)
                        
                        # Save ROI visualization and coordinates
                        plt.close('all')  # Close any existing figures
                        fig, ax = plt.subplots()
                        ax.imshow(Tdpre, cmap='gray')
                        ax.plot(center_x, center_y, 'ro', markersize=2)
                        visualize_rois(fig, ax, roi_points, roi_points_spine)
                        plt.savefig(roi_image_file, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        
                        # Save ROI coordinates
                        save_roi_info(savefolder, basename, roi_points, roi_points_spine)
                    except Exception as e:
                        print(f"Error during manual ROI redefinition: {str(e)}")
                        raise  # Re-raise the exception to be caught by the outer try-except
                    
            except Exception as e:
                print(f"\nError processing {each_file}: {str(e)}")
                plt.close('all')  # Make sure to close any remaining figures
                continue

def save_results_to_csv(results_dict: Dict[str, Dict], savefolder: str) -> None:
    """
    Save results to a CSV file.
    
    Args:
        results_dict: Dictionary of results with filepaths as keys
        savefolder: Folder to save the CSV file
    """
    # Create CSV header
    header = "filepath,power_mw,pulse_width,pre_shaft,post_shaft,shaft_f_f0,pre_spine,post_spine,spine_f_f0,roi_file,roi_image\n"
    
    # Create CSV content
    csv_lines = [header]
    for filepath, results in results_dict.items():
        line = f"{filepath},{results['power_mw']},{results['pulse_width']},"
        line += f"{results['pre_shaft']},{results['post_shaft']},{results['shaft_f_f0']},"
        line += f"{results['pre_spine']},{results['post_spine']},{results['spine_f_f0']},"
        line += f"{results['roi_file']},{results['roi_image']}\n"
        csv_lines.append(line)
    
    # Save to file
    csv_path = os.path.join(savefolder, "result.csv")
    with open(csv_path, "w") as f:
        f.writelines(csv_lines)

def save_pre_post_sets(base_path: str, all_pre_post_sets: List[Dict[str, str]]) -> str:
    """
    Save pre/post sets to a JSON file.
    
    Args:
        base_path: Path to the directory containing the files
        all_pre_post_sets: List of dictionaries containing pre, uncaging, and post file paths
        
    Returns:
        Path to the saved JSON file
    """
    save_dir = os.path.dirname(base_path)
    save_path = os.path.join(save_dir, "pre_post_sets.json")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_sets = []
    for set_dict in all_pre_post_sets:
        serializable_sets.append({
            'pre': str(set_dict['pre']),  # Convert Path objects to strings
            'unc': str(set_dict['unc']),
            'post': str(set_dict['post'])
        })
    
    with open(save_path, 'w') as f:
        json.dump(serializable_sets, f, indent=2)
    
    return save_path

def load_pre_post_sets(base_path: str) -> Optional[List[Dict[str, str]]]:
    """
    Load pre/post sets from a JSON file if it exists.
    
    Args:
        base_path: Path to the directory containing the files
        
    Returns:
        List of dictionaries containing pre, uncaging, and post file paths, or None if file doesn't exist
    """
    save_dir = os.path.dirname(base_path)
    save_path = os.path.join(save_dir, "pre_post_sets.json")
    
    if not os.path.exists(save_path):
        return None
    
    try:
        with open(save_path, 'r') as f:
            sets = json.load(f)
            # Convert string paths back to Path objects
            return [{
                'pre': set_dict['pre'],
                'unc': set_dict['unc'],
                'post': set_dict['post']
            } for set_dict in sets]
    except Exception as e:
        print(f"Error loading pre/post sets: {e}")
        return None

if __name__ == "__main__":
    # one_of_filepath = r"G:\ImagingData\Tetsuya\20250227\0131Cas9_GC6stdTom_neuron1_dend1_002.flim"
    # one_of_filepath = r"G:\ImagingData\Tetsuya\20250325\B6wt_GC6stdTom_tony_cut0303_2ndslice\tpem\14_1__highmag_1_002.flim"
    # one_of_filepath = r"G:\ImagingData\Tetsuya\20250227\automation\highmag_highmag_list\tpem\14_1__highmag_1_003.flim"
    # one_of_filepath = r"G:\ImagingData\Tetsuya\20250331\B6_cut0319_FlxGC6sTom_0322\highmag_RFP50ms100p\tpem2\lwomag1__highmag_1_080.flim"
    # one_of_filepath = r"G:\ImagingData\Tetsuya\20250408\tpem2\lowmag_pos2__highmag_5_002.flim"
    # one_of_filepath = r"G:\ImagingData\Tetsuya\20250415\B6_cut0326_FlxGC6s_tdTomato\highmag_Trans5ms\tpem\C1_00_1_1__highmag_1_002.flim"
    # one_of_filepath = r"G:\ImagingData\Tetsuya\20250416\B6_cut0326_FlxGC6s_tdTomato0330\highmag_Trans5ms\tpem\C2_00_1_1__highmag_1_002.flim"
    one_of_filepath = r"G:\ImagingData\Tetsuya\20250417\B6_cut0326_FlxGC6s_tdTomato0330\highmag_Trans5ms\tpem\C4_00_2_1__highmag_3_025.flim"
    
    filename_sample = "*highmag_*002.flim"
    pow_slope = 0.192
    pow_intcpt = 0.2990
    
    # Try to load existing pre/post sets
    all_pre_post_sets = load_pre_post_sets(one_of_filepath)
    
    if all_pre_post_sets is None:
        print("No saved pre/post sets found. Creating new sets...")
        all_pre_post_sets = get_all_pre_post_sets(one_of_filepath, filename_sample)
        # Save the newly created sets
        save_path = save_pre_post_sets(one_of_filepath, all_pre_post_sets)
        print(f"Saved pre/post sets to: {save_path}")
    else:
        print("Loaded existing pre/post sets")
    
    # Add quality threshold parameter
    quality_threshold = 0.1  # Adjust this value based on your needs
    
    # Process the sets with quality threshold
    get_pre_post_set_list(all_pre_post_sets, pow_slope, pow_intcpt, quality_threshold)


# %%

