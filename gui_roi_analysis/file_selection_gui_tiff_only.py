# %%
"""
TIFF-only ROI saving version of file_selection_gui.py
This module extends file_selection_gui.py to support saving ROI masks to TIFF files only,
without saving to dataframe.
"""

import sys
import os
# Import the original file_selection_gui module
from gui_roi_analysis.file_selection_gui import FileSelectionGUI as OriginalFileSelectionGUI
from PyQt5.QtWidgets import QMessageBox


class FileSelectionGUITiffOnly(OriginalFileSelectionGUI):
    """Extended FileSelectionGUI that saves ROI masks to TIFF files only."""
    
    def __init__(self, combined_df, df_save_path_2=None, additional_columns=None, save_auto=True, parent=None):
        """Initialize with TIFF-only saving enabled."""
        super().__init__(combined_df, df_save_path_2, additional_columns, save_auto, parent)
        self.save_to_dataframe = False  # Disable dataframe saving
        self.save_roi_to_tiff = True   # Enable TIFF saving
        self.log_message("TIFF-only mode enabled: ROI masks will be saved to TIFF files, not dataframe")
    
    def get_roi_status_and_date(self, group_df, header):
        """Check if ROI is defined by checking TIFF file existence (TIFF-only mode)."""
        try:
            import datetime
            
            has_roi = False
            date_str = ""
            
            # Get TIFF path from the dataframe
            tiff_path = None
            if 'after_align_save_path' in group_df.columns:
                tiff_paths = group_df['after_align_save_path'].dropna()
                if len(tiff_paths) > 0:
                    tiff_path = tiff_paths.iloc[0]
            
            if tiff_path and os.path.exists(tiff_path):
                # Generate expected ROI mask TIFF path
                tiff_dir = os.path.dirname(tiff_path)
                tiff_basename = os.path.basename(tiff_path)
                tiff_basename_no_ext = os.path.splitext(tiff_basename)[0]
                roi_tiff_path = os.path.join(tiff_dir, f"{tiff_basename_no_ext}_{header}_roi_mask.tif")
                
                # Check if ROI mask TIFF file exists
                if os.path.exists(roi_tiff_path):
                    has_roi = True
                    
                    # Get file modification time as date
                    try:
                        mtime = os.path.getmtime(roi_tiff_path)
                        mod_date = datetime.datetime.fromtimestamp(mtime)
                        date_str = mod_date.strftime("%m-%d %H:%M")
                    except Exception as e:
                        print(f"Warning: Could not get modification time for {roi_tiff_path}: {e}")
                        date_str = ""
            
            return has_roi, date_str
            
        except Exception as e:
            print(f"Error getting ROI status for {header} (TIFF-only mode): {e}")
            import traceback
            traceback.print_exc()
            return False, ""
    
    def launch_all_roi_analysis(self, group, set_label, tiff_path):
        """Launch ROI analysis for all three types with TIFF-only saving."""
        if not tiff_path or not os.path.exists(tiff_path):
            QMessageBox.warning(self, "Error", f"TIFF file not found: {tiff_path}")
            self.log_message(f"ERROR: TIFF file not found for {group}, Set {set_label}: {tiff_path}")
            return
        
        self.status_label.setText("Running ROI analysis...")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.log_message(f"Starting all ROI analysis for {group}, Set {set_label} (TIFF-only mode)")
        
        try:
            # Import TIFF-only version
            from gui_integration_tiff_only import launch_roi_analysis_gui_tiff_only
            
            roi_types = ["Spine", "DendriticShaft", "Background"]
            for roi_type in roi_types:
                self.log_message(f"Launching {roi_type} ROI analysis for {group}, Set {set_label}")
                
                # Generate TIFF save path
                tiff_dir = os.path.dirname(tiff_path)
                tiff_basename = os.path.basename(tiff_path)
                # Remove extension
                tiff_basename_no_ext = os.path.splitext(tiff_basename)[0]
                # Add ROI type suffix
                tiff_save_path = os.path.join(tiff_dir, f"{tiff_basename_no_ext}_{roi_type}_roi_mask.tif")
                self.log_message(f"ROI mask will be saved to: {tiff_save_path}")
                
                launch_roi_analysis_gui_tiff_only(
                    self.combined_df, 
                    tiff_path, 
                    group, 
                    set_label, 
                    header=roi_type,
                    save_tiff_path=tiff_save_path
                )
                self.roi_analysis_completed.emit(group, set_label, roi_type)
            
            # Note: No dataframe auto-save in TIFF-only mode
            self.log_message("ROI analysis completed. Masks saved to TIFF files only (not to dataframe).")
            
            # Refresh table immediately to show updated ROI status
            self.refresh_table()
            
            # Schedule delayed refresh if auto-refresh is enabled (as backup)
            if self.auto_refresh_check.isChecked():
                self.refresh_timer.start(1000)  # Refresh after 1 second as backup
            
            self.status_label.setText("ROI analysis completed (TIFF saved)")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.log_message(f"All ROI analysis completed for {group}, Set {set_label}")
            
        except Exception as e:
            error_msg = f"Failed to launch ROI analysis: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.log_message(f"ERROR: {error_msg}")
            self.status_label.setText("Error occurred")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def launch_individual_roi_analysis(self, group, set_label, tiff_path, roi_type):
        """Launch ROI analysis for individual type with TIFF-only saving."""
        if not tiff_path or not os.path.exists(tiff_path):
            QMessageBox.warning(self, "Error", f"TIFF file not found: {tiff_path}")
            self.log_message(f"ERROR: TIFF file not found for {group}, Set {set_label}: {tiff_path}")
            return
        
        self.status_label.setText(f"Running {roi_type} ROI analysis...")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.log_message(f"Starting {roi_type} ROI analysis for {group}, Set {set_label} (TIFF-only mode)")
        
        try:
            # Import TIFF-only version
            from gui_integration_tiff_only import launch_roi_analysis_gui_tiff_only
            
            # Generate TIFF save path
            tiff_dir = os.path.dirname(tiff_path)
            tiff_basename = os.path.basename(tiff_path)
            # Remove extension
            tiff_basename_no_ext = os.path.splitext(tiff_basename)[0]
            # Add ROI type suffix
            tiff_save_path = os.path.join(tiff_dir, f"{tiff_basename_no_ext}_{roi_type}_roi_mask.tif")
            self.log_message(f"ROI mask will be saved to: {tiff_save_path}")
            
            launch_roi_analysis_gui_tiff_only(
                self.combined_df, 
                tiff_path, 
                group, 
                set_label, 
                header=roi_type,
                save_tiff_path=tiff_save_path
            )
            self.roi_analysis_completed.emit(group, set_label, roi_type)
            
            # Note: No dataframe auto-save in TIFF-only mode
            self.log_message(f"{roi_type} ROI analysis completed. Mask saved to TIFF file only (not to dataframe).")
            
            # Refresh table immediately to show updated ROI status
            self.refresh_table()
            
            # Schedule delayed refresh if auto-refresh is enabled (as backup)
            if self.auto_refresh_check.isChecked():
                self.refresh_timer.start(1000)  # Refresh after 1 second as backup
            
            self.status_label.setText(f"{roi_type} ROI analysis completed (TIFF saved)")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.log_message(f"{roi_type} ROI analysis completed for {group}, Set {set_label}")
            
        except Exception as e:
            error_msg = f"Failed to launch {roi_type} ROI analysis: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.log_message(f"ERROR: {error_msg}")
            self.status_label.setText("Error occurred")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def _create_central_roi_for_rejected(self, group_id, set_label, mask):
        """Create a central rectangular ROI for rejected items and save to TIFF files (TIFF-only mode)."""
        try:
            import datetime
            import tifffile
            import numpy as np
            from gui_integration import create_roi_mask_from_params
            
            # Get group/set specific data
            group_set_df = self.combined_df[mask]
            
            # Get TIFF path
            tiff_path = ""
            if 'after_align_save_path' in group_set_df.columns:
                tiff_paths = group_set_df['after_align_save_path'].dropna()
                if len(tiff_paths) > 0:
                    tiff_path = tiff_paths.iloc[0]
            
            if not tiff_path or not os.path.exists(tiff_path):
                self.log_message(f"Warning: TIFF file not found for {group_id}, Set {set_label}. Cannot create ROI.")
                return
            
            # Load TIFF data to get image size
            try:
                after_align_tiff_data = tifffile.imread(tiff_path)
                
                # Get image shape (handle different dimensions)
                if len(after_align_tiff_data.shape) == 4:
                    # 4D data: (time, z, y, x) - use first time point and max projection over z
                    image_shape = after_align_tiff_data[0].max(axis=0).shape  # (y, x)
                    num_frames = after_align_tiff_data.shape[0]
                elif len(after_align_tiff_data.shape) == 3:
                    # 3D data: could be (time, y, x) or (z, y, x) - use max projection over first axis
                    image_shape = after_align_tiff_data.max(axis=0).shape  # (y, x)
                    num_frames = after_align_tiff_data.shape[0]
                else:
                    # 2D data: use as is
                    image_shape = after_align_tiff_data.shape  # (y, x)
                    num_frames = 1
                
                height, width = image_shape
                
                # Calculate central ROI: half of each side length
                roi_width = width // 2
                roi_height = height // 2
                roi_x = (width - roi_width) // 2
                roi_y = (height - roi_height) // 2
                
                # Create ROI parameters for rectangle
                roi_params = {
                    'x': roi_x,
                    'y': roi_y,
                    'width': roi_width,
                    'height': roi_height
                }
                
                # Create ROI mask
                roi_mask = create_roi_mask_from_params(roi_params, 'rectangle', image_shape)
                
                # Check ROI status for each type and create ROI if not defined
                roi_types = ["Spine", "DendriticShaft", "Background"]
                
                for roi_type in roi_types:
                    has_roi, _ = self.get_roi_status_and_date(group_set_df, roi_type)
                    
                    if not has_roi:
                        # Generate TIFF save path
                        tiff_dir = os.path.dirname(tiff_path)
                        tiff_basename = os.path.basename(tiff_path)
                        tiff_basename_no_ext = os.path.splitext(tiff_basename)[0]
                        roi_tiff_path = os.path.join(tiff_dir, f"{tiff_basename_no_ext}_{roi_type}_roi_mask.tif")
                        
                        # Create 3D ROI stack (same mask for all frames)
                        roi_stack = np.stack([roi_mask] * num_frames, axis=0)
                        roi_stack_uint8 = roi_stack.astype(np.uint8)
                        
                        # Save to TIFF file
                        try:
                            tifffile.imwrite(roi_tiff_path, roi_stack_uint8, photometric='minisblack')
                            self.log_message(f"Created central rectangular ROI for {roi_type} (rejected) and saved to: {roi_tiff_path}")
                            self.log_message(f"  ROI size: {roi_width}x{roi_height} at ({roi_x}, {roi_y}), {num_frames} frames")
                            
                            # Refresh table to show updated status
                            self.refresh_table()
                        except Exception as e:
                            self.log_message(f"Error saving ROI mask to TIFF for {roi_type}: {e}")
                            print(f"Error saving ROI mask to TIFF: {e}")
                            import traceback
                            traceback.print_exc()
                
            except Exception as e:
                self.log_message(f"Error creating central ROI for rejected item: {e}")
                print(f"Error creating central ROI: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            self.log_message(f"Error in _create_central_roi_for_rejected: {e}")
            print(f"Error in _create_central_roi_for_rejected: {e}")
            import traceback
            traceback.print_exc()


def launch_file_selection_gui_tiff_only(combined_df, df_save_path_2=None, additional_columns=None, save_auto=True):
    """Launch the file selection GUI with TIFF-only ROI saving
    
    Args:
        combined_df: DataFrame containing the analysis data
        df_save_path_2: Optional path to save the DataFrame automatically (not used for ROI masks)
        additional_columns: Optional list of column names to display in addition to the standard columns
        save_auto: Whether to auto-save dataframe (not used for ROI masks)
    
    Returns:
        FileSelectionGUITiffOnly instance
    """
    try:
        print("Starting file selection GUI launcher (TIFF-only mode)...")
        
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            print("Created new QApplication instance")
        
        print("Creating FileSelectionGUITiffOnly instance...")
        gui = FileSelectionGUITiffOnly(combined_df, df_save_path_2, additional_columns, save_auto)
        print("Showing GUI window...")
        gui.show()
        
        print("File selection GUI launched successfully (TIFF-only mode)")
        return gui
        
    except Exception as e:
        print(f"Error in launch_file_selection_gui_tiff_only: {e}")
        import traceback
        traceback.print_exc()
        raise e
