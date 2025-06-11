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

Shaft ROI (Uncaging):
- Pre intensity: {results_dict['pre_shaft']:.0f}
- Unc intensity: {results_dict['unc_shaft']:.0f}
- F/F-1: {results_dict['shaft_f_f0']:.2f}

Spine ROI (Uncaging):
- Pre intensity: {results_dict['pre_spine']:.0f}
- Unc intensity: {results_dict['unc_spine']:.0f}
- F/F-1: {results_dict['spine_f_f0']:.2f}

Spine ROI (Pre/Post):
- Pre intensity: {results_dict['pre_spine']:.0f}
- Post intensity: {results_dict['post_spine']:.0f}
- F/F-1: {results_dict['spine_pre_post_f_f0']:.2f}

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

def save_roi_info(savefolder: str, basename: str, roi_points_spine: np.ndarray) -> str:
    """Save ROI coordinates to a JSON file"""
    roi_info = {
        'spine': roi_points_spine.tolist()
    }
    roi_file = os.path.join(savefolder, basename[:-5] + "_roi.json")
    with open(roi_file, 'w') as f:
        json.dump(roi_info, f)
    return roi_file

def load_roi_info(roi_file: str) -> np.ndarray:
    """Load ROI coordinates from a JSON file"""
    with open(roi_file, 'r') as f:
        roi_info = json.load(f)
    return np.array(roi_info['spine'])

def get_roi_mask(roi_points_spine: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """Create mask from ROI points"""
    r_spine, c_spine = roi_points_spine[:, 0], roi_points_spine[:, 1]
    mask_rows_spine, mask_cols_spine = polygon(c_spine, r_spine, image_shape)
    mask_spine = np.zeros(image_shape, dtype=bool)
    mask_spine[mask_rows_spine, mask_cols_spine] = True
    return mask_spine

def visualize_roi(fig: plt.Figure, ax: plt.Axes, roi_points_spine: np.ndarray) -> None:
    """Visualize ROI on the given figure"""
    r_spine, c_spine = roi_points_spine[:, 0], roi_points_spine[:, 1]
    ax.plot(list(r_spine) + [list(r_spine)[0]], list(c_spine) + [list(c_spine)[0]], "c", lw=2)
    plt.text(.05, .95, 'Spine', c="c", size=10, ha='left', va='top', transform=ax.transAxes)
    ax.set_title("Spine ROI")
    ax.axis("off")

def create_default_roi(image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create default spine ROI (1/4 size square) in the center of the image.
    
    Args:
        image_shape: Shape of the image (height, width)
        
    Returns:
        spine_roi_points as numpy array
    """
    height, width = image_shape
    center_y, center_x = height // 2, width // 2
    roi_size = min(height, width) // 4
    
    # Create spine ROI
    spine_points = np.array([
        [center_y - roi_size//4, center_x - roi_size//4],
        [center_y - roi_size//4, center_x + roi_size//4],
        [center_y + roi_size//4, center_x + roi_size//4],
        [center_y + roi_size//4, center_x - roi_size//4]
    ])
    
    return spine_points

def get_pre_post_set_list(all_pre_post_sets: List[Dict[str, str]], pow_slope: float, pow_intcpt: float, quality_threshold: float = 0.3) -> None:
    """
    Process pre/post sets and perform analysis.
    """
    results_dict = {}
    should_continue_analysis = True
    
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
                
                # Get images for all time points
                GCpre = imagearray[0,0,0,:,:,:].sum(axis=-1)  # First frame
                GCunc = imagearray[3,0,0,:,:,:].sum(axis=-1)  # Uncaging frame
                GCpost = imagearray[-1,0,0,:,:,:].sum(axis=-1)  # Last frame
                Tdpre = imagearray[0,0,1,:,:,:].sum(axis=-1)
                
                GC_pre_med = median_filter(GCpre, size=3)
                GC_unc_med = median_filter(GCunc, size=3)
                GC_post_med = median_filter(GCpost, size=3)
                
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
                        print(f"\nLow quality image detected for {basename}. Using default ROIs.")
                        roi_points, roi_points_spine = create_default_roi(GC_pre_med.shape)
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
                        mask, mask_spine = get_roi_masks(roi_points, roi_points_spine, GC_pre_med.shape)
                        
                        plt.close('all')
                        fig, ax = plt.subplots()
                        ax.imshow(Tdpre, cmap='gray')
                        ax.plot(center_x, center_y, 'ro', markersize=2)
                        visualize_rois(fig, ax, roi_points, roi_points_spine)
                        plt.savefig(roi_image_file, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        
                        save_roi_info(savefolder, basename, roi_points, roi_points_spine)
                    except Exception as e:
                        print(f"Error saving ROI information: {str(e)}")
                        raise
                
                # Calculate results for uncaging time point
                pre_spine = GC_pre_med[mask_spine].sum()
                unc_spine = GC_unc_med[mask_spine].sum()
                pre_shaft = GC_pre_med[mask].sum()
                unc_shaft = GC_unc_med[mask].sum()
                
                # Calculate results for pre/post time points using spine ROI
                post_spine = GC_post_med[mask_spine].sum()
                
                # Calculate F/F0 ratios
                spine_f_f0 = unc_spine / pre_spine if pre_spine > 0 else 0
                shaft_f_f0 = unc_shaft / pre_shaft if pre_shaft > 0 else 0
                spine_pre_post_f_f0 = post_spine / pre_spine if pre_spine > 0 else 0
                
                # Store results in dictionary
                results_dict[each_file] = {
                    'power_mw': pow_mw_round,
                    'pulse_width': pulseWidth,
                    'pre_shaft': int(pre_shaft),
                    'unc_shaft': int(unc_shaft),
                    'shaft_f_f0': float(f"{shaft_f_f0:.2f}"),
                    'pre_spine': int(pre_spine),
                    'unc_spine': int(unc_spine),
                    'spine_f_f0': float(f"{spine_f_f0:.2f}"),
                    'post_spine': int(post_spine),
                    'spine_pre_post_f_f0': float(f"{spine_pre_post_f_f0:.2f}"),
                    'roi_file': roi_file,
                    'roi_image': roi_image_file,
                    'is_low_quality': not is_good_quality
                }
                
                # Show dialog with updated results
                dialog = NextFileDialog(roi_image_file, results_dict[each_file])
                result = dialog.show()

                if result in [None, True]:
                    save_results_to_csv(results_dict, savefolder)
                    
                if result is None:
                    print(f"\nAnalysis terminated by user. Results saved to: {os.path.join(savefolder, 'result.csv')}")
                    print(f"Total files analyzed: {len(results_dict)}")
                    should_continue_analysis = False
                    break
                elif result:
                    re_define_roi = False
                else:
                    re_define_roi = True
                    if os.path.exists(roi_file):
                        os.remove(roi_file)
                    if os.path.exists(roi_image_file):
                        os.remove(roi_image_file)
                    results_dict.pop(each_file, None)
                    
                    try:
                        print("\nRedefining ROIs manually...")
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
                        
                        mask, mask_spine = get_roi_masks(roi_points, roi_points_spine, GC_pre_med.shape)
                        
                        plt.close('all')
                        fig, ax = plt.subplots()
                        ax.imshow(Tdpre, cmap='gray')
                        ax.plot(center_x, center_y, 'ro', markersize=2)
                        visualize_rois(fig, ax, roi_points, roi_points_spine)
                        plt.savefig(roi_image_file, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        
                        save_roi_info(savefolder, basename, roi_points, roi_points_spine)
                    except Exception as e:
                        print(f"Error during manual ROI redefinition: {str(e)}")
                        raise
                    
            except Exception as e:
                print(f"\nError processing {each_file}: {str(e)}")
                plt.close('all')
                continue

def save_results_to_csv(results_dict: Dict[str, Dict], savefolder: str) -> None:
    """
    Save results to a CSV file.
    """
    header = "filepath,power_mw,pulse_width,pre_shaft,unc_shaft,shaft_f_f0,pre_spine,unc_spine,spine_f_f0,post_spine,spine_pre_post_f_f0,roi_file,roi_image\n"
    
    csv_lines = [header]
    for filepath, results in results_dict.items():
        line = f"{filepath},{results['power_mw']},{results['pulse_width']},"
        line += f"{results['pre_shaft']},{results['unc_shaft']},{results['shaft_f_f0']},"
        line += f"{results['pre_spine']},{results['unc_spine']},{results['spine_f_f0']},"
        line += f"{results['post_spine']},{results['spine_pre_post_f_f0']},"
        line += f"{results['roi_file']},{results['roi_image']}\n"
        csv_lines.append(line)
    
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

