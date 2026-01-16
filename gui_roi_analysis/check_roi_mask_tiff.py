# %%
"""
ROI Mask TIFF File Checker and Visualizer

This script allows you to check and visualize ROI masks saved as TIFF files.
It can display individual frames, create overlays with original images, and show statistics.
"""

import os
import sys
sys.path.append('..\\')
import numpy as np
import pandas as pd
import tifffile
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Polygon
import argparse
from pathlib import Path


def load_roi_mask_tiff(tiff_path):
    """Load ROI mask from TIFF file.
    
    Args:
        tiff_path: Path to the TIFF file containing ROI masks
        
    Returns:
        numpy.ndarray: 3D array of ROI masks (time, height, width) as boolean
    """
    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
    
    roi_stack = tifffile.imread(tiff_path)
    print(f"Loaded ROI mask TIFF: {tiff_path}")
    print(f"  Shape: {roi_stack.shape} (time, height, width)")
    print(f"  Dtype: {roi_stack.dtype}")
    
    # Convert to boolean if needed
    if roi_stack.dtype != bool:
        roi_stack = roi_stack.astype(bool)
    
    return roi_stack


def get_roi_statistics(roi_stack):
    """Calculate statistics for ROI masks.
    
    Args:
        roi_stack: 3D array of ROI masks (time, height, width)
        
    Returns:
        dict: Dictionary containing statistics
    """
    num_frames = roi_stack.shape[0]
    areas = [np.sum(roi_stack[i]) for i in range(num_frames)]
    
    stats = {
        'num_frames': num_frames,
        'image_shape': roi_stack.shape[1:],
        'total_pixels': roi_stack.shape[1] * roi_stack.shape[2],
        'roi_areas': areas,
        'mean_area': np.mean(areas),
        'std_area': np.std(areas),
        'min_area': np.min(areas),
        'max_area': np.max(areas),
        'roi_coverage_percent': (np.mean(areas) / (roi_stack.shape[1] * roi_stack.shape[2])) * 100
    }
    
    return stats


def visualize_roi_mask_frames(roi_stack, save_path=None, max_frames_to_show=10):
    """Visualize ROI masks for multiple frames.
    
    Args:
        roi_stack: 3D array of ROI masks (time, height, width)
        save_path: Optional path to save the figure
        max_frames_to_show: Maximum number of frames to display
    """
    num_frames = roi_stack.shape[0]
    frames_to_show = min(num_frames, max_frames_to_show)
    
    # Calculate grid size
    cols = 5
    rows = (frames_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(frames_to_show):
        frame_idx = int(i * num_frames / frames_to_show) if frames_to_show < num_frames else i
        roi_mask = roi_stack[frame_idx]
        
        ax = axes[i]
        ax.imshow(roi_mask, cmap='gray', interpolation='nearest')
        ax.set_title(f'Frame {frame_idx}\nArea: {np.sum(roi_mask)} px')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(frames_to_show, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def visualize_roi_overlay(roi_mask, original_image=None, frame_idx=0, save_path=None):
    """Create overlay visualization of ROI mask on original image.
    
    Args:
        roi_mask: 2D boolean array or 3D array (if 3D, uses frame_idx)
        original_image: Optional original image to overlay ROI on
        frame_idx: Frame index if roi_mask is 3D
        save_path: Optional path to save the figure
    """
    # Extract single frame if 3D
    if len(roi_mask.shape) == 3:
        roi_mask_2d = roi_mask[frame_idx]
    else:
        roi_mask_2d = roi_mask
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image (if provided) or empty
    if original_image is not None:
        if len(original_image.shape) == 3:
            orig_2d = original_image[frame_idx] if frame_idx < original_image.shape[0] else original_image[0]
        else:
            orig_2d = original_image
        axes[0].imshow(orig_2d, cmap='gray', interpolation='nearest')
        axes[0].set_title('Original Image')
    else:
        axes[0].imshow(np.zeros_like(roi_mask_2d), cmap='gray', interpolation='nearest')
        axes[0].set_title('Original Image (not provided)')
    axes[0].axis('off')
    
    # ROI mask
    axes[1].imshow(roi_mask_2d, cmap='gray', interpolation='nearest')
    axes[1].set_title(f'ROI Mask (Frame {frame_idx})\nArea: {np.sum(roi_mask_2d)} px')
    axes[1].axis('off')
    
    # Overlay
    if original_image is not None:
        overlay = orig_2d.copy()
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay, overlay, overlay], axis=-1)
        overlay[roi_mask_2d, 0] = np.clip(overlay[roi_mask_2d, 0] * 0.7 + 255 * 0.3, 0, 255)
        axes[2].imshow(overlay.astype(np.uint8), interpolation='nearest')
        axes[2].set_title('Overlay (Red = ROI)')
    else:
        overlay = np.zeros((*roi_mask_2d.shape, 3), dtype=np.uint8)
        overlay[roi_mask_2d] = [255, 0, 0]  # Red
        axes[2].imshow(overlay, interpolation='nearest')
        axes[2].set_title('ROI Mask (Red)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved overlay to: {save_path}")
    
    plt.show()


def compare_multiple_roi_types(tiff_paths_dict, original_image_path=None, frame_idx=0, save_path=None):
    """Compare multiple ROI types side by side.
    
    Args:
        tiff_paths_dict: Dictionary mapping ROI type names to TIFF file paths
        original_image_path: Optional path to original image TIFF
        frame_idx: Frame index to display
        save_path: Optional path to save the figure
    """
    num_roi_types = len(tiff_paths_dict)
    fig, axes = plt.subplots(2, num_roi_types + 1, figsize=(5*(num_roi_types+1), 10))
    
    # Load original image if provided
    original_image = None
    if original_image_path and os.path.exists(original_image_path):
        original_image = tifffile.imread(original_image_path)
        if len(original_image.shape) == 4:
            orig_2d = original_image[frame_idx].max(axis=0) if frame_idx < original_image.shape[0] else original_image[0].max(axis=0)
        elif len(original_image.shape) == 3:
            orig_2d = original_image[frame_idx] if frame_idx < original_image.shape[0] else original_image[0]
        else:
            orig_2d = original_image
    else:
        orig_2d = None
    
    # Display original image
    if orig_2d is not None:
        axes[0, 0].imshow(orig_2d, cmap='gray', interpolation='nearest')
        axes[0, 0].set_title('Original Image')
    else:
        axes[0, 0].text(0.5, 0.5, 'Original Image\n(not provided)', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
    axes[0, 0].axis('off')
    
    if orig_2d is not None:
        overlay_all = orig_2d.copy()
        if len(overlay_all.shape) == 2:
            overlay_all = np.stack([overlay_all, overlay_all, overlay_all], axis=-1)
    else:
        overlay_all = None
    
    # Display each ROI type
    colors = {'Spine': 'red', 'DendriticShaft': 'blue', 'Background': 'green'}
    for idx, (roi_type, tiff_path) in enumerate(tiff_paths_dict.items(), 1):
        if not os.path.exists(tiff_path):
            axes[0, idx].text(0.5, 0.5, f'{roi_type}\nFile not found', 
                            ha='center', va='center', transform=axes[0, idx].transAxes)
            axes[0, idx].axis('off')
            axes[1, idx].axis('off')
            continue
        
        roi_stack = load_roi_mask_tiff(tiff_path)
        if frame_idx >= roi_stack.shape[0]:
            frame_idx = 0
        
        roi_mask = roi_stack[frame_idx]
        
        # Display ROI mask
        axes[0, idx].imshow(roi_mask, cmap='gray', interpolation='nearest')
        axes[0, idx].set_title(f'{roi_type} ROI\nArea: {np.sum(roi_mask)} px')
        axes[0, idx].axis('off')
        
        # Create overlay
        if orig_2d is not None:
            overlay = orig_2d.copy()
            if len(overlay.shape) == 2:
                overlay = np.stack([overlay, overlay, overlay], axis=-1)
            color = colors.get(roi_type, [255, 255, 0])
            if isinstance(color, str):
                color_map = {'red': [255, 0, 0], 'blue': [0, 0, 255], 'green': [0, 255, 0]}
                color = color_map.get(color, [255, 255, 0])
            overlay[roi_mask, 0] = np.clip(overlay[roi_mask, 0] * 0.7 + color[0] * 0.3, 0, 255)
            overlay[roi_mask, 1] = np.clip(overlay[roi_mask, 1] * 0.7 + color[1] * 0.3, 0, 255)
            overlay[roi_mask, 2] = np.clip(overlay[roi_mask, 2] * 0.7 + color[2] * 0.3, 0, 255)
            axes[1, idx].imshow(overlay.astype(np.uint8), interpolation='nearest')
            axes[1, idx].set_title(f'{roi_type} Overlay')
            
            # Add to combined overlay
            if overlay_all is not None:
                overlay_all[roi_mask, 0] = np.clip(overlay_all[roi_mask, 0] * 0.7 + color[0] * 0.3, 0, 255)
                overlay_all[roi_mask, 1] = np.clip(overlay_all[roi_mask, 1] * 0.7 + color[1] * 0.3, 0, 255)
                overlay_all[roi_mask, 2] = np.clip(overlay_all[roi_mask, 2] * 0.7 + color[2] * 0.3, 0, 255)
        else:
            overlay = np.zeros((*roi_mask.shape, 3), dtype=np.uint8)
            color = colors.get(roi_type, [255, 255, 0])
            if isinstance(color, str):
                color_map = {'red': [255, 0, 0], 'blue': [0, 0, 255], 'green': [0, 255, 0]}
                color = color_map.get(color, [255, 255, 0])
            overlay[roi_mask] = color
            axes[1, idx].imshow(overlay, interpolation='nearest')
            axes[1, idx].set_title(f'{roi_type} Mask')
        axes[1, idx].axis('off')
    
    # Display combined overlay
    if overlay_all is not None:
        axes[1, 0].imshow(overlay_all.astype(np.uint8), interpolation='nearest')
        axes[1, 0].set_title('All ROIs Overlay')
    else:
        axes[1, 0].text(0.5, 0.5, 'All ROIs Overlay\n(original image needed)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


def print_roi_statistics(tiff_path):
    """Print detailed statistics about ROI mask.
    
    Args:
        tiff_path: Path to the ROI mask TIFF file
    """
    roi_stack = load_roi_mask_tiff(tiff_path)
    stats = get_roi_statistics(roi_stack)
    
    print("\n" + "="*50)
    print(f"ROI Mask Statistics: {os.path.basename(tiff_path)}")
    print("="*50)
    print(f"Number of frames: {stats['num_frames']}")
    print(f"Image shape: {stats['image_shape']} (height, width)")
    print(f"Total pixels per frame: {stats['total_pixels']}")
    print(f"\nROI Area Statistics:")
    print(f"  Mean area: {stats['mean_area']:.1f} pixels")
    print(f"  Std area: {stats['std_area']:.1f} pixels")
    print(f"  Min area: {stats['min_area']} pixels")
    print(f"  Max area: {stats['max_area']} pixels")
    print(f"  Coverage: {stats['roi_coverage_percent']:.2f}% of image")
    print("\nArea per frame:")
    for i, area in enumerate(stats['roi_areas']):
        print(f"  Frame {i}: {area} pixels")
    print("="*50 + "\n")


def find_roi_mask_files(base_directory, roi_types=None):
    """Find ROI mask TIFF files in a directory.
    
    Args:
        base_directory: Directory to search in
        roi_types: List of ROI types to search for (default: ["Spine", "DendriticShaft", "Background"])
        
    Returns:
        dict: Dictionary mapping ROI types to file paths
    """
    if roi_types is None:
        roi_types = ["Spine", "DendriticShaft", "Background"]
    
    found_files = {}
    base_path = Path(base_directory)
    
    # Search for ROI mask files
    for roi_type in roi_types:
        pattern = f"*{roi_type}_roi_mask.tif"
        files = list(base_path.glob(pattern))
        if files:
            found_files[roi_type] = str(files[0])  # Take first match
            print(f"Found {roi_type} ROI mask: {files[0]}")
        else:
            # Also try .tiff extension
            pattern = f"*{roi_type}_roi_mask.tiff"
            files = list(base_path.glob(pattern))
            if files:
                found_files[roi_type] = str(files[0])
                print(f"Found {roi_type} ROI mask: {files[0]}")
            else:
                print(f"No {roi_type} ROI mask file found")
    
    return found_files


# %%
# Example usage functions

def example_check_single_roi():
    """Example: Check a single ROI mask file"""
    # Specify the path to your ROI mask TIFF file
    roi_mask_path = r"C:\path\to\your\roi_mask.tif"
    
    # Load and print statistics
    print_roi_statistics(roi_mask_path)
    
    # Visualize frames
    roi_stack = load_roi_mask_tiff(roi_mask_path)
    visualize_roi_mask_frames(roi_stack, max_frames_to_show=10)
    
    # Visualize overlay (if you have the original image)
    # original_image_path = r"C:\path\to\original_image.tif"
    # original_image = tifffile.imread(original_image_path)
    # visualize_roi_overlay(roi_stack, original_image, frame_idx=0)


def example_check_multiple_rois():
    """Example: Check multiple ROI types"""
    # Specify the directory containing ROI mask files
    base_dir = r"C:\path\to\your\data\directory"
    
    # Find ROI mask files
    roi_files = find_roi_mask_files(base_dir)
    
    if not roi_files:
        print("No ROI mask files found!")
        return
    
    # Check each ROI type
    for roi_type, file_path in roi_files.items():
        print(f"\n{'='*60}")
        print(f"Checking {roi_type} ROI")
        print(f"{'='*60}")
        print_roi_statistics(file_path)
    
    # Compare all ROI types side by side
    original_image_path = None  # Set this if you have the original image
    # original_image_path = os.path.join(base_dir, "after_align_set0.tif")
    
    compare_multiple_roi_types(roi_files, original_image_path, frame_idx=0)


def example_check_from_dataframe():
    """Example: Check ROI masks based on dataframe information"""
    # Load your combined dataframe
    combined_df_path = r"C:\path\to\combined_df.pkl"
    combined_df = pd.read_pickle(combined_df_path)
    
    # Get a specific group and set
    each_filepath = combined_df['filepath_without_number'].unique()[0]
    each_group_df = combined_df[combined_df['filepath_without_number'] == each_filepath]
    each_set_label = each_group_df['nth_set_label'].unique()[0]
    
    # Get the original TIFF path
    original_tiff_path = each_group_df[each_group_df['nth_set_label'] == each_set_label]["after_align_save_path"].values[0]
    tiff_dir = os.path.dirname(original_tiff_path)
    tiff_basename = os.path.basename(original_tiff_path)
    tiff_basename_no_ext = os.path.splitext(tiff_basename)[0]
    
    # Find ROI mask files
    roi_types = ["Spine", "DendriticShaft", "Background"]
    roi_files = {}
    for roi_type in roi_types:
        roi_path = os.path.join(tiff_dir, f"{tiff_basename_no_ext}_{roi_type}_roi_mask.tif")
        if os.path.exists(roi_path):
            roi_files[roi_type] = roi_path
            print(f"Found {roi_type} ROI: {roi_path}")
        else:
            print(f"Not found: {roi_path}")
    
    if roi_files:
        # Compare all ROI types
        compare_multiple_roi_types(roi_files, original_tiff_path, frame_idx=0)
        
        # Print statistics for each
        for roi_type, file_path in roi_files.items():
            print_roi_statistics(file_path)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check and visualize ROI mask TIFF files')
    parser.add_argument('--roi_path', type=str, help='Path to a single ROI mask TIFF file')
    parser.add_argument('--directory', type=str, help='Directory to search for ROI mask files')
    parser.add_argument('--original', type=str, help='Path to original image TIFF for overlay')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to display (default: 0)')
    parser.add_argument('--stats', action='store_true', help='Print statistics only')
    parser.add_argument('--compare', action='store_true', help='Compare multiple ROI types')
    
    args = parser.parse_args()
    
    if args.roi_path:
        if args.stats:
            print_roi_statistics(args.roi_path)
        else:
            roi_stack = load_roi_mask_tiff(args.roi_path)
            if args.original:
                original_image = tifffile.imread(args.original)
                visualize_roi_overlay(roi_stack, original_image, frame_idx=args.frame)
            else:
                visualize_roi_mask_frames(roi_stack, max_frames_to_show=10)
    
    elif args.directory:
        roi_files = find_roi_mask_files(args.directory)
        if args.compare:
            compare_multiple_roi_types(roi_files, args.original, frame_idx=args.frame)
        else:
            for roi_type, file_path in roi_files.items():
                if args.stats:
                    print_roi_statistics(file_path)
                else:
                    roi_stack = load_roi_mask_tiff(file_path)
                    visualize_roi_mask_frames(roi_stack, max_frames_to_show=10)
    
    else:
        print("Please specify either --roi_path or --directory")
        print("\nExample usage:")
        print("  python check_roi_mask_tiff.py --roi_path path/to/roi_mask.tif")
        print("  python check_roi_mask_tiff.py --directory path/to/directory --compare")
        print("  python check_roi_mask_tiff.py --roi_path path/to/roi_mask.tif --original path/to/original.tif --frame 0")
