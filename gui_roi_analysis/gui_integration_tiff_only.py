# %%
"""
TIFF-only ROI saving version of gui_integration.py
This module extends gui_integration.py to support saving ROI masks to TIFF files only,
without saving to dataframe.
"""

import os
import sys
sys.path.append('..\\')
import numpy as np
import pandas as pd
import tifffile
from gui_integration import (
    create_roi_mask_from_params,
    SHIFT_DIRECTION
)


def save_roi_mask_from_gui_to_tiff(gui_instance, save_path, header="ROI"):
    """Save ROI masks from GUI instance directly to a 3D 1-bit TIFF file.
    
    Args:
        gui_instance: Instance of ROIAnalysisGUI containing ROI data
        save_path: Path where the TIFF file should be saved
        header: Header string used for column names (default: "ROI")
    
    Returns:
        str: Path to the saved TIFF file, or None if saving failed
    """
    print(f"Saving {header} ROI masks from GUI to TIFF: {save_path}")
    
    # Get the image shape from the GUI data
    frame_data = gui_instance.after_align_tiff_data[0]
    if len(frame_data.shape) == 3:
        image_shape = frame_data.max(axis=0).shape  # (y, x)
    else:
        image_shape = frame_data.shape[-2:]  # Get last 2 dimensions
    
    print(f"Image shape for ROI masks: {image_shape}")
    
    # Check if GUI instance has frame-specific ROI parameters
    has_frame_specific_params = hasattr(gui_instance, 'frame_roi_parameters') and gui_instance.frame_roi_parameters
    
    # Collect all ROI masks
    roi_masks = []
    num_frames = len(gui_instance.intensity_data['mean'])
    
    for frame_idx in range(num_frames):
        # Get ROI parameters for this frame
        if has_frame_specific_params and frame_idx in gui_instance.frame_roi_parameters:
            frame_roi_params = gui_instance.frame_roi_parameters[frame_idx]
        else:
            frame_roi_params = gui_instance.roi_parameters
        
        # Create ROI mask for this frame
        roi_mask = create_roi_mask_from_params(
            frame_roi_params, gui_instance.roi_shape, image_shape
        )
        
        # Ensure roi_mask is a numpy array
        if not isinstance(roi_mask, np.ndarray):
            roi_mask = np.array(roi_mask, dtype=bool)
        
        roi_masks.append(roi_mask)
    
    if len(roi_masks) == 0:
        print("Error: No ROI masks generated")
        return None
    
    # Stack masks into 3D array (time, height, width)
    roi_stack = np.stack(roi_masks, axis=0)
    
    # Convert boolean to uint8 (1-bit: 0 or 1)
    roi_stack_uint8 = roi_stack.astype(np.uint8)
    
    print(f"ROI stack shape: {roi_stack_uint8.shape} (time, height, width)")
    print(f"ROI stack dtype: {roi_stack_uint8.dtype}")
    print(f"Number of frames: {len(roi_masks)}")
    
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Save as TIFF file
    try:
        tifffile.imwrite(save_path, roi_stack_uint8, photometric='minisblack')
        print(f"Successfully saved ROI masks to: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error saving TIFF file: {e}")
        import traceback
        traceback.print_exc()
        return None


def launch_roi_analysis_gui_tiff_only(combined_df, tiff_data_path, each_group, each_set_label, header="ROI", save_tiff_path=None):
    """Launch the ROI analysis GUI with TIFF-only saving option.
    
    This function is a wrapper around the original launch_roi_analysis_gui that adds
    TIFF saving functionality without saving to dataframe.

    Args:
        combined_df: The combined dataframe containing analysis data
        tiff_data_path: Path to the after_align TIFF file
        each_group: Group identifier (could be 'group' or 'filepath_without_number' value)
        each_set_label: Set label identifier
        header: Header string to use for column names and GUI display
        save_tiff_path: If provided, save ROI masks to TIFF file at this path (default: None)
    
    Returns:
        int: Result code (0 for success, 1 for error)
    """
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt, QEventLoop
    from roi_analysis_gui import ROIAnalysisGUI
    import sys
    
    # Load the TIFF data
    if os.path.exists(tiff_data_path):
        after_align_tiff_data = tifffile.imread(tiff_data_path)
    else:
        print(f"Error: TIFF file not found at {tiff_data_path}")
        return None

    # Create max projection properly handling different data dimensions
    print(f"TIFF data shape: {after_align_tiff_data.shape}")

    if len(after_align_tiff_data.shape) == 4:
        # 4D data: (time, z, y, x) - take max over both time and z axes
        max_proj_image = after_align_tiff_data.max(axis=(0, 1))
    elif len(after_align_tiff_data.shape) == 3:
        # 3D data: could be (time, y, x) or (z, y, x) - take max over first axis
        max_proj_image = after_align_tiff_data.max(axis=0)
    else:
        # 2D data: use as is
        max_proj_image = after_align_tiff_data

    print(f"Max projection shape: {max_proj_image.shape}")

    # Ensure we have a 2D image for display
    if len(max_proj_image.shape) != 2:
        print(f"Warning: Max projection has unexpected shape {max_proj_image.shape}")
        # Take max over all but last two dimensions
        while len(max_proj_image.shape) > 2:
            max_proj_image = max_proj_image.max(axis=0)
        print(f"Adjusted max projection shape: {max_proj_image.shape}")

    # Filter dataframe for the specific group and set
    # Try filtering by 'group' first, then fall back to 'filepath_without_number'
    filtered_df = combined_df[
        (combined_df['group'] == each_group) &
        (combined_df['nth_set_label'] == each_set_label)
    ].copy()

    # If no data found with 'group', try with 'filepath_without_number'
    if len(filtered_df) == 0:
        print(f"No data found using 'group'={each_group}. Trying 'filepath_without_number'...")
        if 'filepath_without_number' in combined_df.columns:
            # First find the filepath_without_number that corresponds to this group
            filepath_without_number_candidates = combined_df[
                combined_df['group'] == each_group
            ]['filepath_without_number'].unique()

            if len(filepath_without_number_candidates) > 0:
                for filepath_without_number in filepath_without_number_candidates:
                    temp_filtered_df = combined_df[
                        (combined_df['filepath_without_number'] == filepath_without_number) &
                        (combined_df['nth_set_label'] == each_set_label)
                    ].copy()
                    if len(temp_filtered_df) > 0:
                        filtered_df = temp_filtered_df
                        print(f"Found data using 'filepath_without_number'={filepath_without_number}")
                        break

            # If still no data, try direct match with filepath_without_number
            if len(filtered_df) == 0:
                filtered_df = combined_df[
                    (combined_df['filepath_without_number'] == each_group) &
                    (combined_df['nth_set_label'] == each_set_label)
                ].copy()
                if len(filtered_df) > 0:
                    print(f"Found data using direct 'filepath_without_number' match")

    if len(filtered_df) == 0:
        print(f"Error: No data found for group={each_group}, set_label={each_set_label}")
        print(f"Available groups in 'group' column: {combined_df['group'].unique()}")
        if 'filepath_without_number' in combined_df.columns:
            print(f"Available groups in 'filepath_without_number' column: {combined_df['filepath_without_number'].unique()}")
        print(f"Available set_labels: {combined_df['nth_set_label'].unique()}")
        return None

    print(f"Successfully filtered data: {len(filtered_df)} rows found")

    # Get uncaging position information from the filtered dataframe
    uncaging_info = {}
    first_row = filtered_df.iloc[0]
    
    if 'corrected_uncaging_x' in filtered_df.columns and 'corrected_uncaging_y' in filtered_df.columns:
        # Get uncaging position from the first row (same for all rows in a set)
        if pd.notna(first_row.get('corrected_uncaging_x')) and pd.notna(first_row.get('corrected_uncaging_y')):
            # If uncaging_display_x/y are set (e.g. full-size ROI TIFF), use them directly; else convert to small region
            if ('uncaging_display_x' in filtered_df.columns and 'uncaging_display_y' in filtered_df.columns and
                    pd.notna(first_row.get('uncaging_display_x')) and pd.notna(first_row.get('uncaging_display_y'))):
                uncaging_info = {
                    'x': float(first_row.get('uncaging_display_x', 0)),
                    'y': float(first_row.get('uncaging_display_y', 0)),
                    'has_uncaging': True
                }
            else:
                small_x_from = first_row.get('small_x_from', 0)
                small_y_from = first_row.get('small_y_from', 0)
                corrected_uncaging_x = first_row.get('corrected_uncaging_x', 0)
                corrected_uncaging_y = first_row.get('corrected_uncaging_y', 0)
                uncaging_info = {
                    'x': corrected_uncaging_x - small_x_from,
                    'y': corrected_uncaging_y - small_y_from,
                    'has_uncaging': True
                }
        else:
            uncaging_info = {'has_uncaging': False}
    elif 'center_x' in filtered_df.columns and 'center_y' in filtered_df.columns:
        # For transient data: use center_x / center_y directly (no small region cropping)
        if pd.notna(first_row.get('center_x')) and pd.notna(first_row.get('center_y')):
            uncaging_info = {
                'x': first_row.get('center_x', 0),
                'y': first_row.get('center_y', 0),
                'has_uncaging': True
            }
            print(f"  Uncaging position: ({uncaging_info['x']:.1f}, {uncaging_info['y']:.1f}) pixels")
        else:
            uncaging_info = {'has_uncaging': False}
    else:
        uncaging_info = {'has_uncaging': False}

    # Get file information for display
    file_info = {}
    if 'file_path' in filtered_df.columns:
        first_file_path = filtered_df.iloc[0].get('file_path', '')
        if first_file_path:
            file_info['filename'] = os.path.basename(first_file_path)
            file_info['directory'] = os.path.dirname(first_file_path)
        else:
            file_info['filename'] = 'Unknown'
            file_info['directory'] = 'Unknown'
    else:
        file_info['filename'] = 'Unknown'
        file_info['directory'] = 'Unknown'

    # Add group and set information
    file_info['group'] = each_group
    file_info['set_label'] = each_set_label

    # Check if QApplication already exists
    app = QApplication.instance()
    if app is None:
        # Create new application if none exists
        app = QApplication(sys.argv)
        app_created = True
    else:
        # Use existing application
        app_created = False
        print("Using existing QApplication instance")

    # Create GUI window with additional information and header
    window = ROIAnalysisGUI(filtered_df, after_align_tiff_data, max_proj_image, uncaging_info, file_info, header=header)

    # Store the filtered dataframe in the GUI instance for verification during saving
    window.filtered_df = filtered_df

    # Set window flags to ensure proper cleanup
    window.setAttribute(Qt.WA_DeleteOnClose, True)

    window.show()

    # Run application only if we created it
    if app_created:
        result = app.exec_()
    else:
        # For existing application, use a different approach
        # Set up event loop for the window
        loop = QEventLoop()

        # Connect window close event to loop quit
        def on_window_close():
            try:
                loop.quit()
            except RuntimeError:
                # Handle case where loop is already finished
                pass

        # Override closeEvent to ensure proper cleanup
        original_close_event = window.closeEvent
        def enhanced_close_event(event):
            try:
                original_close_event(event)
                on_window_close()
            except Exception as e:
                print(f"Warning: Error during window close: {e}")
                event.accept()
                on_window_close()

        window.closeEvent = enhanced_close_event

        # Connect analysis complete to window close
        original_complete = window.complete_analysis
        def complete_and_close():
            try:
                original_complete()
                window.close()
            except Exception as e:
                print(f"Warning: Error during analysis completion: {e}")
                window.close()
        window.complete_analysis = complete_and_close

        # Run the event loop with error handling
        try:
            loop.exec_()
            result = 0
        except Exception as e:
            print(f"Warning: Event loop error: {e}")
            result = 1

    # Save ROI data if analysis was completed
    if hasattr(window, 'analysis_completed') and window.analysis_completed:
        print("=== ROI DATA SAVING (TIFF ONLY) ===")
        
        # Save to TIFF file if path is provided
        if save_tiff_path is not None:
            print(f"Saving {header} ROI masks to TIFF file...")
            try:
                save_roi_mask_from_gui_to_tiff(window, save_tiff_path, header)
            except Exception as tiff_error:
                print(f"Error saving ROI masks to TIFF: {tiff_error}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: save_tiff_path not provided. ROI masks will not be saved.")
        
        print("Skipping dataframe save (TIFF-only mode)")
    else:
        print("Analysis not completed or analysis_completed flag not set")
        if hasattr(window, 'analysis_completed'):
            print(f"analysis_completed = {window.analysis_completed}")
        else:
            print("analysis_completed attribute not found")

    # Clean up
    try:
        window.deleteLater()

        # Process events to ensure cleanup
        if app:
            app.processEvents()

        # Force garbage collection
        import gc
        gc.collect()

    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

    return result


def load_roi_mask_from_tiff_and_create_shifted_mask(combined_df, tiff_path, each_group, each_set_label, 
                                                     roi_type, z_plus_minus, image_shape=(128, 128)):
    """Load ROI masks from TIFF file and create shifted_mask in dataframe.
    
    Args:
        combined_df: The combined dataframe
        tiff_path: Path to the TIFF file containing ROI masks (3D array: time, height, width)
        each_group: Group identifier
        each_set_label: Set label identifier
        roi_type: ROI type (e.g., "Spine", "DendriticShaft", "Background")
        z_plus_minus: z_plus_minus parameter for shift calculation
        image_shape: Shape of the full-size image (height, width)
    
    Returns:
        combined_df: Updated dataframe with shifted_mask
    """
    print(f"Loading {roi_type} ROI masks from TIFF: {tiff_path}")
    
    if not os.path.exists(tiff_path):
        print(f"Error: TIFF file not found: {tiff_path}")
        return combined_df
    
    # Load ROI masks from TIFF
    roi_stack = tifffile.imread(tiff_path)
    print(f"Loaded ROI stack shape: {roi_stack.shape}")
    
    # Convert to boolean if needed
    if roi_stack.dtype != bool:
        roi_stack = roi_stack.astype(bool)
    
    # Filter dataframe to get the relevant rows
    base_mask = (combined_df['group'] == each_group) & (combined_df['nth_set_label'] == each_set_label)
    filtered_df = combined_df[base_mask]
    
    # If no data found with 'group', try with 'filepath_without_number'
    if len(filtered_df) == 0:
        if 'filepath_without_number' in combined_df.columns:
            filepath_without_number_candidates = combined_df[
                combined_df['group'] == each_group
            ]['filepath_without_number'].unique()
            
            if len(filepath_without_number_candidates) > 0:
                for filepath_without_number in filepath_without_number_candidates:
                    temp_base_mask = (combined_df['filepath_without_number'] == filepath_without_number) & (combined_df['nth_set_label'] == each_set_label)
                    temp_filtered_df = combined_df[temp_base_mask]
                    if len(temp_filtered_df) > 0:
                        base_mask = temp_base_mask
                        filtered_df = temp_filtered_df
                        break
    
    # Filter to only include rows that were actually analyzed (nth_omit_induction >= 0)
    analysis_mask = base_mask & (combined_df['nth_omit_induction'] >= 0)
    analysis_filtered_df = combined_df[analysis_mask]
    
    if len(analysis_filtered_df) == 0:
        print(f"Error: No data found for group {each_group}, set {each_set_label}")
        return combined_df
    
    # Ensure shifted_mask column exists
    shifted_mask_col = f"{roi_type}_shifted_mask"
    if shifted_mask_col not in combined_df.columns:
        combined_df[shifted_mask_col] = None
    combined_df[shifted_mask_col] = combined_df[shifted_mask_col].astype(object)
    
    # Sort by nth_omit_induction to ensure correct frame order
    set_df_sorted = analysis_filtered_df.sort_values('nth_omit_induction')
    
    # Process each frame
    for frame_idx, (df_idx, row) in enumerate(set_df_sorted.iterrows()):
        if frame_idx >= roi_stack.shape[0]:
            print(f"Warning: Frame {frame_idx} exceeds ROI stack size {roi_stack.shape[0]}")
            continue
        
        # Get ROI mask for this frame
        roi_mask = roi_stack[frame_idx]
        
        # Get shift parameters
        shift_x = row.get('shift_x', 0)
        shift_y = row.get('shift_y', 0)
        small_shift_x = row.get('small_shift_x', 0)
        small_shift_y = row.get('small_shift_y', 0)
        small_x_from = row.get('small_x_from', 0)
        small_y_from = row.get('small_y_from', 0)
        
        total_shift_x = shift_x + small_shift_x
        total_shift_y = shift_y + small_shift_y
        
        # Get coordinates of ROI pixels
        coords = np.argwhere(roi_mask)
        shifted_coords = coords.copy()
        shifted_coords[:, 0] = coords[:, 0] + small_y_from + total_shift_y * SHIFT_DIRECTION
        shifted_coords[:, 1] = coords[:, 1] + small_x_from + total_shift_x * SHIFT_DIRECTION
        
        # Check if the shifted_coords is out of the image
        height, width = image_shape
        valid_mask = (
            (shifted_coords[:, 0] >= 0) & 
            (shifted_coords[:, 0] < height) & 
            (shifted_coords[:, 1] >= 0) & 
            (shifted_coords[:, 1] < width)
        )
        shifted_coords = shifted_coords[valid_mask]
        
        # Create shifted mask
        shifted_mask = np.zeros(image_shape, dtype=bool)
        if len(shifted_coords) > 0:
            shifted_mask[shifted_coords[:, 0], shifted_coords[:, 1]] = True
        else:
            print(f"Warning: No valid shifted coordinates for frame {frame_idx}")
            # Fallback: use original coordinates
            if len(coords) > 0:
                shifted_mask[coords[:, 0], coords[:, 1]] = True
        
        # Save to dataframe
        combined_df.at[df_idx, shifted_mask_col] = shifted_mask.copy()
    
    print(f"Successfully created {roi_type}_shifted_mask from TIFF file")
    return combined_df


def load_roi_masks_from_tiff_and_create_shifted_masks(combined_df, tiff_path_dict, each_group, each_set_label,
                                                      roi_types, z_plus_minus, image_shape=(128, 128)):
    """Load ROI masks from TIFF files and create shifted_mask for multiple ROI types.
    
    Args:
        combined_df: The combined dataframe
        tiff_path_dict: Dictionary mapping roi_type to TIFF file path
        each_group: Group identifier
        each_set_label: Set label identifier
        roi_types: List of ROI types (e.g., ["Spine", "DendriticShaft", "Background"])
        z_plus_minus: z_plus_minus parameter for shift calculation
        image_shape: Shape of the full-size image (height, width)
    
    Returns:
        combined_df: Updated dataframe with shifted_mask for all ROI types
    """
    for roi_type in roi_types:
        if roi_type in tiff_path_dict:
            tiff_path = tiff_path_dict[roi_type]
            combined_df = load_roi_mask_from_tiff_and_create_shifted_mask(
                combined_df, tiff_path, each_group, each_set_label,
                roi_type, z_plus_minus, image_shape
            )
    
    return combined_df
