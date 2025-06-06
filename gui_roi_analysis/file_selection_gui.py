import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QTableWidget, QTableWidgetItem, QPushButton,
                            QLabel, QHeaderView, QCheckBox, QMessageBox, QProgressBar,
                            QSplitter, QTextEdit, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor


class FileSelectionGUI(QMainWindow):
    roi_analysis_completed = pyqtSignal(str, int, str)  # group, set_label, header
    
    def __init__(self, combined_df, df_save_path_2=None, additional_columns=None, parent=None):
        super().__init__(parent)
        
        print("Initializing FileSelectionGUI...")
        
        try:
            self.combined_df = combined_df
            self.df_save_path_2 = df_save_path_2  # Store the save path for auto-saving
            self.additional_columns = additional_columns or []  # Store additional columns to display
            
            # Initialize reject column if it doesn't exist
            if 'reject' not in self.combined_df.columns:
                self.combined_df['reject'] = False
            
            # Initialize comment column if it doesn't exist
            if 'comment' not in self.combined_df.columns:
                self.combined_df['comment'] = ""
            
            print(f"DataFrame shape: {combined_df.shape if combined_df is not None else 'None'}")
            if self.additional_columns:
                print(f"Additional columns to display: {self.additional_columns}")
            
            self.setWindowTitle("ROI Analysis File Selection")
            self.setGeometry(100, 100, 1600, 900)  # Made wider for comment column
            
            # Create central widget and layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            
            # Title
            title_label = QLabel("Select Files for ROI Analysis")
            title_label.setFont(QFont("Arial", 16, QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(title_label)
            
            # Create info label
            info_label = QLabel("✓ = ROI defined, ✗ = ROI not defined. Green buttons = already defined, click to redefine.")
            info_label.setFont(QFont("Arial", 10))
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("color: gray; margin: 5px;")
            main_layout.addWidget(info_label)
            
            # Create splitter for table and log
            splitter = QSplitter(Qt.Vertical)
            main_layout.addWidget(splitter)
            
            # Create table container
            table_widget = QWidget()
            table_layout = QVBoxLayout(table_widget)
            
            # Create table
            self.create_table()
            table_layout.addWidget(self.table)
            splitter.addWidget(table_widget)
            
            # Create log area
            log_widget = QWidget()
            log_layout = QVBoxLayout(log_widget)
            log_label = QLabel("Activity Log:")
            log_label.setFont(QFont("Arial", 12, QFont.Bold))
            log_layout.addWidget(log_label)
            
            self.log_text = QTextEdit()
            self.log_text.setMaximumHeight(150)
            self.log_text.setReadOnly(True)
            self.log_text.setStyleSheet("background-color: #f5f5f5; font-family: monospace;")
            log_layout.addWidget(self.log_text)
            splitter.addWidget(log_widget)
            
            # Set splitter sizes (table gets more space)
            splitter.setSizes([600, 150])
            
            # Button layout
            button_layout = QHBoxLayout()
            
            # Refresh button
            refresh_btn = QPushButton("🔄 Refresh Table")
            refresh_btn.setToolTip("Refresh the table to show latest ROI status")
            refresh_btn.clicked.connect(self.refresh_table)
            button_layout.addWidget(refresh_btn)
            
            # Auto-refresh checkbox
            self.auto_refresh_check = QCheckBox("Auto-refresh after ROI analysis")
            self.auto_refresh_check.setChecked(True)
            self.auto_refresh_check.setToolTip("Automatically refresh table after completing ROI analysis")
            button_layout.addWidget(self.auto_refresh_check)
            
            button_layout.addStretch()
            
            # Status label
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            button_layout.addWidget(self.status_label)
            
            # Close button
            close_btn = QPushButton("❌ Close")
            close_btn.setToolTip("Close the file selection window")
            close_btn.clicked.connect(self.close)
            button_layout.addWidget(close_btn)
            
            main_layout.addLayout(button_layout)
            
            # Timer for auto-refresh delay
            self.refresh_timer = QTimer()
            self.refresh_timer.setSingleShot(True)
            self.refresh_timer.timeout.connect(self.delayed_refresh)
            
            # Populate table
            self.log_message("File Selection GUI initialized successfully.")
            self.populate_table()
            
            print("FileSelectionGUI initialization completed successfully.")
            
        except Exception as e:
            print(f"Error initializing FileSelectionGUI: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
    def log_message(self, message):
        """Add a timestamped message to the log"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"
            self.log_text.append(formatted_message)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
            print(f"LOG: {formatted_message}")  # Also print to console
        except Exception as e:
            print(f"Error logging message: {e}")
        
    def create_table(self):
        """Create the main table widget"""
        try:
            print("Creating table widget...")
            self.table = QTableWidget()
            
            # Basic table setup - columns will be configured in populate_table
            # based on actual DataFrame structure
            
            # Set table properties
            self.table.setSortingEnabled(True)
            self.table.setAlternatingRowColors(True)
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            
            print("Table widget created successfully.")
            
        except Exception as e:
            print(f"Error creating table: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
    def get_roi_status_and_date(self, group_df, header):
        """Check if ROI is defined and get the date"""
        try:
            roi_column = f"{header}_roi_mask"
            date_column = f"{header}_roi_analysis_timestamp"  # Updated to match gui_integration.py
            
            # Check if any row in this group/set has the ROI defined
            has_roi = False
            date_str = ""
            
            if roi_column in group_df.columns:
                roi_masks = group_df[roi_column].dropna()
                if len(roi_masks) > 0:
                    # Check if any mask is not None and not empty
                    for mask in roi_masks:
                        if mask is not None and np.any(mask):
                            has_roi = True
                            break
            
            if has_roi and date_column in group_df.columns:
                dates = group_df[date_column].dropna()
                if len(dates) > 0:
                    # Get the most recent date
                    latest_date = dates.iloc[-1]
                    if pd.notna(latest_date):
                        if isinstance(latest_date, str):
                            date_str = latest_date
                        else:
                            date_str = str(latest_date)
            
            return has_roi, date_str
            
        except Exception as e:
            print(f"Error getting ROI status for {header}: {e}")
            return False, ""
    
    def populate_table(self):
        """Populate the table with data from combined_df"""
        try:
            print("Populating table...")
            self.status_label.setText("Loading data...")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            
            if self.combined_df is None or len(self.combined_df) == 0:
                self.log_message("No data available to display")
                self.status_label.setText("No data")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                return
            
            # Check if z columns exist in the DataFrame
            z_columns_exist = all(col in self.combined_df.columns for col in ['z_from', 'z_to', 'len_z'])
            
            # Check which additional columns actually exist in the DataFrame
            available_additional_columns = [col for col in self.additional_columns if col in self.combined_df.columns]
            if self.additional_columns:
                missing_columns = [col for col in self.additional_columns if col not in self.combined_df.columns]
                if missing_columns:
                    print(f"Warning: Additional columns not found in DataFrame: {missing_columns}")
                if available_additional_columns:
                    print(f"Available additional columns: {available_additional_columns}")
            
            # Recreate table with proper column structure now that we know z column availability
            headers = ["TIFF Path", "Reject", "Comment"]
            if z_columns_exist:
                headers.extend(["z_from", "z_to", "len_z"])
            
            # Add available additional columns
            headers.extend(available_additional_columns)
            
            headers.extend([
                "Spine ROI", "Spine Date",
                "DendriticShaft ROI", "DendriticShaft Date", 
                "Background ROI", "Background Date",
                "All ROIs", "Individual ROIs"
            ])
            
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            
            # Update column widths based on new structure
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)          # TIFF Path (immutable)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Reject
            header.setSectionResizeMode(2, QHeaderView.Interactive)      # Comment - make it resizable
            header.resizeSection(2, 150)  # Set initial width for comment column
            
            col_idx = 3
            if z_columns_exist:
                header.setSectionResizeMode(col_idx, QHeaderView.ResizeToContents)      # z_from
                header.setSectionResizeMode(col_idx + 1, QHeaderView.ResizeToContents)  # z_to
                header.setSectionResizeMode(col_idx + 2, QHeaderView.ResizeToContents)  # len_z
                col_idx += 3
            
            # Set resize mode for additional columns
            for i, _ in enumerate(available_additional_columns):
                header.setSectionResizeMode(col_idx + i, QHeaderView.ResizeToContents)
            col_idx += len(available_additional_columns)
            
            header.setSectionResizeMode(col_idx, QHeaderView.ResizeToContents)      # Spine ROI
            header.setSectionResizeMode(col_idx + 1, QHeaderView.ResizeToContents)  # Spine Date
            header.setSectionResizeMode(col_idx + 2, QHeaderView.ResizeToContents)  # DendriticShaft ROI
            header.setSectionResizeMode(col_idx + 3, QHeaderView.ResizeToContents)  # DendriticShaft Date
            header.setSectionResizeMode(col_idx + 4, QHeaderView.ResizeToContents)  # Background ROI
            header.setSectionResizeMode(col_idx + 5, QHeaderView.ResizeToContents)  # Background Date
            header.setSectionResizeMode(col_idx + 6, QHeaderView.ResizeToContents)  # All ROIs
            header.setSectionResizeMode(col_idx + 7, QHeaderView.ResizeToContents)  # Individual ROIs
            
            # Check required columns
            required_columns = ['group', 'nth_set_label']
            missing_columns = [col for col in required_columns if col not in self.combined_df.columns]
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                self.log_message(f"ERROR: {error_msg}")
                self.status_label.setText("Data error")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                QMessageBox.critical(self, "Data Error", error_msg)
                return
            
            # Get unique combinations of group and set_label
            valid_data = self.combined_df[
                (self.combined_df['nth_set_label'] >= 0) & 
                (self.combined_df['nth_set_label'].notna())
            ]
            
            if len(valid_data) == 0:
                self.log_message("No valid data entries found (nth_set_label >= 0)")
                self.status_label.setText("No valid data")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                return
            
            unique_combinations = valid_data.groupby(['group', 'nth_set_label']).first().reset_index()
            print(f"Found {len(unique_combinations)} unique group/set combinations")
            
            self.table.setRowCount(len(unique_combinations))
            
            for row_idx, (_, row_data) in enumerate(unique_combinations.iterrows()):
                try:
                    group = row_data['group']
                    set_label = row_data['nth_set_label']
                    
                    print(f"Processing row {row_idx}: Group={group}, Set={set_label}")
                    
                    # Get group/set specific data
                    group_set_df = self.combined_df[
                        (self.combined_df['group'] == group) & 
                        (self.combined_df['nth_set_label'] == set_label)
                    ]
                    
                    # Get TIFF path
                    tiff_path = ""
                    if 'after_align_save_path' in group_set_df.columns:
                        tiff_paths = group_set_df['after_align_save_path'].dropna()
                        if len(tiff_paths) > 0:
                            tiff_path = tiff_paths.iloc[0]
                    
                    # TIFF path with tooltip showing full path (immutable)
                    tiff_item = QTableWidgetItem(os.path.basename(str(tiff_path)) if tiff_path else "No TIFF file")
                    tiff_item.setToolTip(str(tiff_path))
                    tiff_item.setFlags(tiff_item.flags() & ~Qt.ItemIsEditable)  # Make immutable
                    self.table.setItem(row_idx, 0, tiff_item)
                    
                    col_idx = 1  # Start after TIFF Path
                    
                    # Reject checkbox
                    reject_checkbox = QCheckBox()
                    # Get current reject status for this group/set
                    current_reject = group_set_df['reject'].iloc[0] if len(group_set_df) > 0 else False
                    reject_checkbox.setChecked(current_reject)
                    reject_checkbox.stateChanged.connect(
                        lambda state, g=group, s=set_label: self.on_reject_changed(g, s, state)
                    )
                    
                    # Center the checkbox
                    reject_widget = QWidget()
                    reject_layout = QHBoxLayout(reject_widget)
                    reject_layout.addWidget(reject_checkbox)
                    reject_layout.setAlignment(Qt.AlignCenter)
                    reject_layout.setContentsMargins(0, 0, 0, 0)
                    
                    self.table.setCellWidget(row_idx, col_idx, reject_widget)
                    col_idx += 1
                    
                    # Comment input field
                    comment_line_edit = QLineEdit()
                    current_comment = group_set_df['comment'].iloc[0] if len(group_set_df) > 0 else ""
                    comment_line_edit.setText(str(current_comment))
                    comment_line_edit.setPlaceholderText("Enter comment...")
                    comment_line_edit.editingFinished.connect(
                        lambda g=group, s=set_label, widget=comment_line_edit: self.on_comment_changed(g, s, widget.text())
                    )
                    
                    self.table.setCellWidget(row_idx, col_idx, comment_line_edit)
                    col_idx += 1
                    
                    # Add z columns if they exist
                    if z_columns_exist:
                        for z_col in ['z_from', 'z_to', 'len_z']:
                            z_value = group_set_df[z_col].iloc[0] if len(group_set_df) > 0 and z_col in group_set_df.columns else ""
                            z_item = QTableWidgetItem(str(z_value) if pd.notna(z_value) else "")
                            z_item.setTextAlignment(Qt.AlignCenter)
                            z_item.setFlags(z_item.flags() & ~Qt.ItemIsEditable)  # Make immutable
                            self.table.setItem(row_idx, col_idx, z_item)
                            col_idx += 1
                    
                    # Add additional columns if they exist
                    for add_col in available_additional_columns:
                        add_value = group_set_df[add_col].iloc[0] if len(group_set_df) > 0 and add_col in group_set_df.columns else ""
                        add_item = QTableWidgetItem(str(add_value) if pd.notna(add_value) else "")
                        add_item.setTextAlignment(Qt.AlignCenter)
                        add_item.setFlags(add_item.flags() & ~Qt.ItemIsEditable)  # Make immutable by default
                        add_item.setToolTip(f"{add_col}: {add_value}")  # Add tooltip showing column name and value
                        self.table.setItem(row_idx, col_idx, add_item)
                        col_idx += 1
                    
                    # Check ROI status for each type
                    roi_types = ["Spine", "DendriticShaft", "Background"]
                    all_defined = True
                    
                    for i, roi_type in enumerate(roi_types):
                        has_roi, date_str = self.get_roi_status_and_date(group_set_df, roi_type)
                        
                        # Status column
                        status_item = QTableWidgetItem("✓" if has_roi else "✗")
                        status_item.setTextAlignment(Qt.AlignCenter)
                        if has_roi:
                            status_item.setBackground(QColor(200, 255, 200))  # Light green
                            status_item.setToolTip(f"{roi_type} ROI is defined")
                        else:
                            status_item.setBackground(QColor(255, 200, 200))  # Light red
                            status_item.setToolTip(f"{roi_type} ROI is not defined")
                            all_defined = False
                        
                        self.table.setItem(row_idx, col_idx, status_item)
                        col_idx += 1
                        
                        # Date column
                        date_item = QTableWidgetItem(date_str)
                        date_item.setTextAlignment(Qt.AlignCenter)
                        if date_str:
                            date_item.setToolTip(f"ROI defined on: {date_str}")
                        else:
                            date_item.setToolTip("No ROI definition date available")
                        self.table.setItem(row_idx, col_idx, date_item)
                        col_idx += 1
                    
                    # All ROIs button
                    all_roi_btn = QPushButton("🎯 Define All ROIs")
                    all_roi_btn.setToolTip(f"Define Spine, DendriticShaft, and Background ROIs for {group}, Set {set_label}")
                    all_roi_btn.clicked.connect(
                        lambda checked, g=group, s=set_label, p=tiff_path: 
                        self.launch_all_roi_analysis(g, s, p)
                    )
                    if all_defined:
                        all_roi_btn.setStyleSheet("background-color: lightgreen;")
                        all_roi_btn.setText("🔄 Redefine All ROIs")
                        all_roi_btn.setToolTip(f"All ROIs are defined. Click to redefine for {group}, Set {set_label}")
                    
                    self.table.setCellWidget(row_idx, col_idx, all_roi_btn)
                    col_idx += 1
                    
                    # Individual ROI buttons
                    individual_widget = QWidget()
                    individual_layout = QHBoxLayout(individual_widget)  # Changed from QVBoxLayout to QHBoxLayout
                    individual_layout.setContentsMargins(1, 1, 1, 1)  # Reduced margins
                    individual_layout.setSpacing(2)  # Reduced spacing
                    
                    roi_icons = {"Spine": "🔴", "DendriticShaft": "🔵", "Background": "🟢"}
                    roi_short_names = {"Spine": "Sp", "DendriticShaft": "Ds", "Background": "Bg"}  # Short names
                    
                    for roi_type in roi_types:
                        icon = roi_icons.get(roi_type, "⭕")
                        short_name = roi_short_names.get(roi_type, roi_type[:2])
                        btn = QPushButton(f"{icon}{short_name}")  # Use icon + short name
                        btn.setMaximumHeight(20)  # Reduced height
                        btn.setMaximumWidth(35)   # Set maximum width to keep buttons small
                        btn.setMinimumWidth(30)   # Set minimum width
                        btn.setToolTip(f"Define {roi_type} ROI for {group}, Set {set_label}")
                        btn.clicked.connect(
                            lambda checked, g=group, s=set_label, p=tiff_path, h=roi_type: 
                            self.launch_individual_roi_analysis(g, s, p, h)
                        )
                        
                        # Check if this ROI is already defined
                        has_roi, _ = self.get_roi_status_and_date(group_set_df, roi_type)
                        if has_roi:
                            btn.setStyleSheet("background-color: lightgreen; font-size: 9px;")  # Smaller font
                            btn.setToolTip(f"{roi_type} ROI is already defined. Click to redefine for {group}, Set {set_label}")
                        else:
                            btn.setStyleSheet("font-size: 9px;")  # Smaller font for undefined ROIs too
                        
                        individual_layout.addWidget(btn)
                    
                    self.table.setCellWidget(row_idx, col_idx, individual_widget)
                    
                except Exception as e:
                    print(f"Error processing row {row_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.log_message(f"Table populated with {len(unique_combinations)} file entries.")
            print("Table populated successfully.")
            
        except Exception as e:
            error_msg = f"Error populating table: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.log_message(f"ERROR: {error_msg}")
            self.status_label.setText("Error")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def launch_all_roi_analysis(self, group, set_label, tiff_path):
        """Launch ROI analysis for all three types"""
        if not tiff_path or not os.path.exists(tiff_path):
            QMessageBox.warning(self, "Error", f"TIFF file not found: {tiff_path}")
            self.log_message(f"ERROR: TIFF file not found for {group}, Set {set_label}: {tiff_path}")
            return
        
        self.status_label.setText("Running ROI analysis...")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.log_message(f"Starting all ROI analysis for {group}, Set {set_label}")
        
        try:
            # Import here to avoid circular imports
            from gui_integration import launch_roi_analysis_gui
            
            roi_types = ["Spine", "DendriticShaft", "Background"]
            for roi_type in roi_types:
                self.log_message(f"Launching {roi_type} ROI analysis for {group}, Set {set_label}")
                launch_roi_analysis_gui(self.combined_df, tiff_path, group, set_label, header=roi_type)
                self.roi_analysis_completed.emit(group, set_label, roi_type)
            
            # Auto-save after ROI analysis completion
            self.auto_save_dataframe()
            
            # Schedule delayed refresh if auto-refresh is enabled
            if self.auto_refresh_check.isChecked():
                self.refresh_timer.start(1000)  # Refresh after 1 second
            
            self.status_label.setText("ROI analysis completed")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.log_message(f"All ROI analysis completed for {group}, Set {set_label}")
            
        except Exception as e:
            error_msg = f"Failed to launch ROI analysis: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.log_message(f"ERROR: {error_msg}")
            self.status_label.setText("Error occurred")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def launch_individual_roi_analysis(self, group, set_label, tiff_path, roi_type):
        """Launch ROI analysis for individual type"""
        if not tiff_path or not os.path.exists(tiff_path):
            QMessageBox.warning(self, "Error", f"TIFF file not found: {tiff_path}")
            self.log_message(f"ERROR: TIFF file not found for {group}, Set {set_label}: {tiff_path}")
            return
        
        self.status_label.setText(f"Running {roi_type} ROI analysis...")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        self.log_message(f"Starting {roi_type} ROI analysis for {group}, Set {set_label}")
        
        try:
            # Import here to avoid circular imports
            from gui_integration import launch_roi_analysis_gui
            
            launch_roi_analysis_gui(self.combined_df, tiff_path, group, set_label, header=roi_type)
            self.roi_analysis_completed.emit(group, set_label, roi_type)
            
            # Auto-save after ROI analysis completion
            self.auto_save_dataframe()
            
            # Schedule delayed refresh if auto-refresh is enabled
            if self.auto_refresh_check.isChecked():
                self.refresh_timer.start(1000)  # Refresh after 1 second
            
            self.status_label.setText(f"{roi_type} ROI analysis completed")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.log_message(f"{roi_type} ROI analysis completed for {group}, Set {set_label}")
            
        except Exception as e:
            error_msg = f"Failed to launch {roi_type} ROI analysis: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.log_message(f"ERROR: {error_msg}")
            self.status_label.setText("Error occurred")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def delayed_refresh(self):
        """Refresh table with a slight delay"""
        self.refresh_table()
    
    def refresh_table(self):
        """Refresh the table data"""
        self.log_message("Refreshing table...")
        self.populate_table()

    def on_reject_changed(self, group, set_label, state):
        """Handle reject checkbox state change"""
        try:
            self.combined_df.loc[(self.combined_df['group'] == group) & (self.combined_df['nth_set_label'] == set_label), 'reject'] = state == Qt.Checked
            self.log_message(f"Reject status changed for {group}, Set {set_label}: {'Rejected' if state == Qt.Checked else 'Accepted'}")
            self.auto_save_dataframe()
        except Exception as e:
            print(f"Error handling reject change: {e}")
            import traceback
            traceback.print_exc()
    
    def on_comment_changed(self, group, set_label, comment):
        """Handle comment text change"""
        try:
            self.combined_df.loc[(self.combined_df['group'] == group) & (self.combined_df['nth_set_label'] == set_label), 'comment'] = comment
            self.log_message(f"Comment changed for {group}, Set {set_label}: {comment}")
            self.auto_save_dataframe()
        except Exception as e:
            print(f"Error handling comment change: {e}")
            import traceback
            traceback.print_exc()
    
    def auto_save_dataframe(self):
        """Auto-save the combined_df to the specified path"""
        if self.df_save_path_2:
            try:
                self.combined_df.to_pickle(self.df_save_path_2)
                self.log_message(f"DataFrame auto-saved to {self.df_save_path_2}")
            except Exception as e:
                self.log_message(f"ERROR: Failed to auto-save DataFrame: {e}")
                print(f"Auto-save error: {e}")
        else:
            self.log_message("No save path specified for auto-save")


def launch_file_selection_gui(combined_df, df_save_path_2=None, additional_columns=None):
    """Launch the file selection GUI
    
    Args:
        combined_df: DataFrame containing the analysis data
        df_save_path_2: Optional path to save the DataFrame automatically
        additional_columns: Optional list of column names to display in addition to the standard columns
                          Example: ['relative_time_min', 'nth_omit_induction', 'phase']
    
    Returns:
        FileSelectionGUI instance
    """
    try:
        print("Starting file selection GUI launcher...")
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            print("Created new QApplication instance")
        
        print("Creating FileSelectionGUI instance...")
        gui = FileSelectionGUI(combined_df, df_save_path_2, additional_columns)
        print("Showing GUI window...")
        gui.show()
        
        print("File selection GUI launched successfully")
        return gui
        
    except Exception as e:
        print(f"Error in launch_file_selection_gui: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    # Example usage
    print("Running file selection GUI test...")
    
    try:
        app = QApplication(sys.argv)
        
        # Create dummy data for testing
        dummy_df = pd.DataFrame({
            'group': ['Group1', 'Group1', 'Group2', 'Group2'],
            'nth_set_label': [0, 1, 0, 1],
            'after_align_save_path': [
                'test1.tiff', 'test2.tiff', 'test3.tiff', 'test4.tiff'
            ],
            'Spine_roi_mask': [None, np.ones((10, 10)), None, None],
            'Spine_roi_analysis_timestamp': [None, '2024-01-01 12:00:00', None, None],
            'DendriticShaft_roi_mask': [None, None, None, None],
            'DendriticShaft_roi_analysis_timestamp': [None, None, None, None],
            'Background_roi_mask': [None, None, None, None],
            'Background_roi_analysis_timestamp': [None, None, None, None],
        })
        
        print("Creating test GUI with dummy data...")
        gui = FileSelectionGUI(dummy_df)
        gui.show()
        
        print("Starting event loop...")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error in main test: {e}")
        import traceback
        traceback.print_exc() 