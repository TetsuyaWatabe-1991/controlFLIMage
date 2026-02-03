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
        # Store row mapping for quick lookup: (group_id, set_label) -> row_idx
        self.row_mapping = {}
        self.log_message("TIFF-only mode enabled: ROI masks will be saved to TIFF files, not dataframe")
    
    # =========================================================================
    # File-based reject status management
    # =========================================================================
    
    def _get_reject_flag_path(self, tiff_path):
        """Get the path to the reject flag file for a given TIFF path."""
        if not tiff_path:
            return None
        tiff_dir = os.path.dirname(tiff_path)
        tiff_basename = os.path.basename(tiff_path)
        tiff_basename_no_ext = os.path.splitext(tiff_basename)[0]
        return os.path.join(tiff_dir, f"{tiff_basename_no_ext}_rejected.flag")
    
    def get_reject_status_from_file(self, tiff_path):
        """Check if reject flag file exists (file-based reject status)."""
        flag_path = self._get_reject_flag_path(tiff_path)
        if flag_path and os.path.exists(flag_path):
            return True
        return False
    
    def save_reject_status_to_file(self, tiff_path, is_rejected):
        """Save or remove reject flag file (file-based reject status)."""
        flag_path = self._get_reject_flag_path(tiff_path)
        if not flag_path:
            return False
        
        try:
            if is_rejected:
                # Create flag file
                with open(flag_path, 'w') as f:
                    import datetime
                    f.write(f"Rejected at: {datetime.datetime.now().isoformat()}\n")
                self.log_message(f"Created reject flag: {flag_path}")
            else:
                # Remove flag file if it exists
                if os.path.exists(flag_path):
                    os.remove(flag_path)
                    self.log_message(f"Removed reject flag: {flag_path}")
            return True
        except Exception as e:
            self.log_message(f"Error managing reject flag file: {e}")
            return False
    
    def on_reject_changed(self, group_id, set_label, state):
        """Handle reject checkbox state change - file-based version."""
        from PyQt5.QtCore import Qt
        try:
            # Get TIFF path for this group/set
            if 'filepath_without_number' in self.combined_df.columns:
                mask = (self.combined_df['filepath_without_number'] == group_id) & (self.combined_df['nth_set_label'] == set_label)
            else:
                mask = (self.combined_df['group'] == group_id) & (self.combined_df['nth_set_label'] == set_label)
            
            group_set_df = self.combined_df[mask]
            
            # Get TIFF path
            tiff_path = ""
            if 'after_align_save_path' in group_set_df.columns:
                tiff_paths = group_set_df['after_align_save_path'].dropna()
                if len(tiff_paths) > 0:
                    tiff_path = tiff_paths.iloc[0]
            
            is_rejected = (state == Qt.Checked)
            
            # Save reject status to file (file-based)
            self.save_reject_status_to_file(tiff_path, is_rejected)
            
            # Also update dataframe for compatibility
            self.combined_df.loc[mask, 'reject'] = is_rejected
            
            self.log_message(f"Reject status changed for {group_id}, Set {set_label}: {'Rejected' if is_rejected else 'Accepted'}")
            
            # If reject is checked and ROI is not defined, create a central rectangular ROI
            if is_rejected:
                self._create_central_roi_for_rejected(group_id, set_label, mask)
            
            # Auto-save dataframe (for compatibility)
            self.auto_save_dataframe()
            
        except Exception as e:
            print(f"Error handling reject change: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # Row-level update methods (for performance)
    # =========================================================================
    
    def _find_row_index(self, group_id, set_label):
        """Find the table row index for a given group_id and set_label."""
        # Check cached mapping first
        key = (group_id, set_label)
        if key in self.row_mapping:
            return self.row_mapping[key]
        
        # Search through table
        for row_idx in range(self.table.rowCount()):
            item = self.table.item(row_idx, 1)  # TIFF Path column
            if item:
                # Get tooltip which contains the full path info
                tooltip = item.toolTip()
                # Check if this row matches by comparing with expected pattern
                if f"set{set_label}" in item.text() or f"set{set_label}" in tooltip:
                    # Verify by checking the data
                    self.row_mapping[key] = row_idx
                    return row_idx
        
        return -1
    
    def update_single_row_roi_status(self, row_idx, tiff_path):
        """Update only the ROI status columns for a single row."""
        from PyQt5.QtWidgets import QTableWidgetItem
        try:
            if row_idx < 0 or row_idx >= self.table.rowCount():
                return
            
            # Find the column indices for ROI status
            # Headers: "View Plot", "TIFF Path", "Reject", "Comment", [z columns], [additional], 
            #          "Spine", "date", "Shaft", "date", "BG", "date", "All ROIs", "Individual ROIs"
            
            # Calculate base column index for ROI status
            base_col = 4  # After View Plot, TIFF Path, Reject, Comment
            
            # Check if z columns exist
            z_columns_exist = all(col in self.combined_df.columns for col in ['z_from', 'z_to', 'len_z'])
            if z_columns_exist:
                base_col += 3
            
            # Add additional columns count
            available_additional_columns = [col for col in self.additional_columns if col in self.combined_df.columns]
            base_col += len(available_additional_columns)
            
            # Now base_col points to "Spine" column
            # Update ROI status for each type
            roi_types = ["Spine", "DendriticShaft", "Background"]
            
            # Create a temporary DataFrame for get_roi_status_and_date
            # We need to pass a DataFrame that has 'after_align_save_path'
            temp_df = self.combined_df[self.combined_df['after_align_save_path'] == tiff_path].head(1)
            if len(temp_df) == 0:
                # Create a minimal DataFrame
                import pandas as pd
                temp_df = pd.DataFrame({'after_align_save_path': [tiff_path]})
            
            for i, roi_type in enumerate(roi_types):
                col_roi = base_col + (i * 2)      # ROI status column
                col_date = base_col + (i * 2) + 1  # Date column
                
                has_roi, date_str = self.get_roi_status_and_date(temp_df, roi_type)
                
                # Update ROI status cell
                roi_item = QTableWidgetItem("✓" if has_roi else "")
                if has_roi:
                    roi_item.setBackground(self._get_green_color())
                self.table.setItem(row_idx, col_roi, roi_item)
                
                # Update date cell
                date_item = QTableWidgetItem(date_str)
                self.table.setItem(row_idx, col_date, date_item)
            
            print(f"Updated ROI status for row {row_idx}")
            
        except Exception as e:
            print(f"Error updating single row ROI status: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_green_color(self):
        """Get light green background color."""
        from PyQt5.QtGui import QColor
        return QColor(144, 238, 144)  # Light green
    
    def populate_table(self):
        """Override populate_table to use file-based reject status."""
        # First, migrate old DataFrame reject status to flag files (one-time migration)
        self._migrate_reject_status_to_files()
        # Then sync file-based reject status to DataFrame
        self._sync_reject_status_from_files()
        # Then call parent's populate_table
        super().populate_table()
        # Clear row mapping cache (will be rebuilt on next lookup)
        self.row_mapping = {}
    
    def _migrate_reject_status_to_files(self):
        """
        Migrate reject status from DataFrame to flag files.
        
        This handles backward compatibility:
        - If DataFrame has reject=True but no flag file exists, create the flag file.
        - This ensures old DataFrames work correctly with the new file-based system.
        """
        try:
            if 'after_align_save_path' not in self.combined_df.columns:
                return
            if 'reject' not in self.combined_df.columns:
                return
            
            migrated_count = 0
            
            # Find rows where reject=True in DataFrame
            rejected_rows = self.combined_df[self.combined_df['reject'] == True]
            
            for idx, row in rejected_rows.iterrows():
                tiff_path = row.get('after_align_save_path', None)
                if not tiff_path or not os.path.exists(str(tiff_path)):
                    continue
                
                # Check if flag file already exists
                flag_path = self._get_reject_flag_path(tiff_path)
                if flag_path and not os.path.exists(flag_path):
                    # Create flag file (migrate from DataFrame)
                    try:
                        with open(flag_path, 'w') as f:
                            import datetime
                            f.write(f"Migrated from DataFrame at: {datetime.datetime.now().isoformat()}\n")
                        migrated_count += 1
                    except Exception as e:
                        print(f"Error creating flag file during migration: {e}")
            
            if migrated_count > 0:
                self.log_message(f"Migrated {migrated_count} reject status(es) from DataFrame to flag files")
            
        except Exception as e:
            print(f"Error migrating reject status to files: {e}")
            import traceback
            traceback.print_exc()
    
    def _sync_reject_status_from_files(self):
        """Sync reject status from flag files to DataFrame."""
        try:
            if 'after_align_save_path' not in self.combined_df.columns:
                return
            
            # Get unique TIFF paths
            unique_paths = self.combined_df['after_align_save_path'].dropna().unique()
            
            for tiff_path in unique_paths:
                if not tiff_path or not os.path.exists(str(tiff_path)):
                    continue
                
                # Check file-based reject status
                is_rejected = self.get_reject_status_from_file(tiff_path)
                
                # Update DataFrame for rows with this TIFF path
                mask = self.combined_df['after_align_save_path'] == tiff_path
                self.combined_df.loc[mask, 'reject'] = is_rejected
            
            self.log_message("Synced reject status from flag files")
            
        except Exception as e:
            print(f"Error syncing reject status from files: {e}")
            import traceback
            traceback.print_exc()
    
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
            
            # Update only the affected row instead of full refresh
            row_idx = self._find_row_by_tiff_path(tiff_path)
            if row_idx >= 0:
                self.update_single_row_roi_status(row_idx, tiff_path)
                self.log_message(f"Updated row {row_idx} for {group}, Set {set_label}")
            else:
                # Fallback to full refresh if row not found
                self.refresh_table()
            
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
            
            # Update only the affected row instead of full refresh
            row_idx = self._find_row_by_tiff_path(tiff_path)
            if row_idx >= 0:
                self.update_single_row_roi_status(row_idx, tiff_path)
                self.log_message(f"Updated row {row_idx} for {group}, Set {set_label}")
            else:
                # Fallback to full refresh if row not found
                self.refresh_table()
            
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
                roi_created = False
                
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
                            roi_created = True
                        except Exception as e:
                            self.log_message(f"Error saving ROI mask to TIFF for {roi_type}: {e}")
                            print(f"Error saving ROI mask to TIFF: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Update only the affected row instead of refreshing entire table
                if roi_created:
                    row_idx = self._find_row_by_tiff_path(tiff_path)
                    if row_idx >= 0:
                        self.update_single_row_roi_status(row_idx, tiff_path)
                        self.log_message(f"Updated row {row_idx} for {group_id}, Set {set_label}")
                    else:
                        # Fallback to full refresh if row not found
                        self.log_message(f"Row not found, falling back to full refresh")
                        self.refresh_table()
                
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
    
    def _find_row_by_tiff_path(self, tiff_path):
        """Find the table row index for a given TIFF path."""
        if not tiff_path:
            return -1
        
        tiff_basename = os.path.basename(tiff_path)
        
        for row_idx in range(self.table.rowCount()):
            item = self.table.item(row_idx, 1)  # TIFF Path column
            if item:
                # Check if the displayed text or tooltip matches
                if tiff_basename in item.text() or tiff_path in item.toolTip():
                    return row_idx
        
        return -1


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
