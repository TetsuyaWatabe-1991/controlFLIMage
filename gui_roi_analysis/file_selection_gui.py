import sys
import os
import pandas as pd
import numpy as np
import subprocess
import platform
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QTableWidget, QTableWidgetItem, QPushButton,
                            QLabel, QHeaderView, QCheckBox, QMessageBox, QProgressBar,
                            QSplitter, QTextEdit, QLineEdit, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor


class FileSelectionGUI(QMainWindow):
    roi_analysis_completed = pyqtSignal(str, int, str)  # group, set_label, header
    
    def __init__(self, combined_df, df_save_path_2=None, additional_columns=None, save_auto = True, parent=None):
        super().__init__(parent)
        
        print("Initializing FileSelectionGUI...")
        
        try:
            self.combined_df = combined_df
            self.df_save_path_2 = df_save_path_2  # Store the save path for auto-saving
            self.additional_columns = additional_columns or []  # Store additional columns to display
            self.save_auto = save_auto
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
            
            # Save button
            save_btn = QPushButton("💾 Save")
            save_btn.setToolTip("Manually save the DataFrame")
            save_btn.clicked.connect(self.manual_save_dataframe)
            button_layout.addWidget(save_btn)
            
            # Close button
            close_btn = QPushButton("❌ Close")
            close_btn.setToolTip("Close the file selection window")
            close_btn.clicked.connect(self.close_and_save)
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
        
    def open_file_with_os_viewer(self, file_path):
        """Open file with OS standard viewer"""
        try:
            if not file_path or pd.isna(file_path) or str(file_path).strip() == "":
                QMessageBox.warning(self, "Error", "No file path available")
                return
            
            file_path = str(file_path).strip()
            
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "File Not Found", f"File does not exist:\n{file_path}")
                self.log_message(f"ERROR: File not found: {file_path}")
                return
            
            # Open file with OS default viewer
            system = platform.system()
            if system == "Windows":
                os.startfile(file_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", file_path])
            else:  # Linux and other Unix-like systems
                subprocess.run(["xdg-open", file_path])
            
            self.log_message(f"Opened file: {os.path.basename(file_path)}")
            
        except Exception as e:
            error_msg = f"Failed to open file: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.log_message(f"ERROR: {error_msg}")
    
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
                        try:
                            # Try to parse as datetime and format to short format
                            if isinstance(latest_date, str):
                                # Try to parse string datetime
                                parsed_date = pd.to_datetime(latest_date)
                            else:
                                # Already a datetime object
                                parsed_date = pd.to_datetime(latest_date)
                            
                            # Format as MM-DD HH:MM (year and seconds removed)
                            date_str = parsed_date.strftime("%m-%d %H:%M")
                            
                        except (ValueError, TypeError):
                            # Fallback to original string if parsing fails
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
            
            # Check if full_plot_with_roi_path column exists
            plot_path_column_exists = 'full_plot_with_roi_path' in self.combined_df.columns
            print(f"full_plot_with_roi_path column exists: {plot_path_column_exists}")
            if plot_path_column_exists:
                non_null_plot_paths = self.combined_df['full_plot_with_roi_path'].dropna()
                print(f"Number of non-null plot paths: {len(non_null_plot_paths)}")
                if len(non_null_plot_paths) > 0:
                    print(f"Sample plot paths: {non_null_plot_paths.head(3).tolist()}")
            
            # Check which additional columns actually exist in the DataFrame
            available_additional_columns = [col for col in self.additional_columns if col in self.combined_df.columns]
            if self.additional_columns:
                missing_columns = [col for col in self.additional_columns if col not in self.combined_df.columns]
                if missing_columns:
                    print(f"Warning: Additional columns not found in DataFrame: {missing_columns}")
                if available_additional_columns:
                    print(f"Available additional columns: {available_additional_columns}")
            
            # Recreate table with proper column structure now that we know z column availability
            headers = ["View Plot", "TIFF Path", "Reject", "Comment"]
            if z_columns_exist:
                headers.extend(["z_from", "z_to", "len_z"])
            
            # Add available additional columns
            headers.extend(available_additional_columns)
            
            headers.extend([
                # "Spine ROI", "Spine Date",
                # "DendriticShaft ROI", "DendriticShaft Date", 
                # "Background ROI", "Background Date",
                "Spine", "date",
                "Shaft", "date", 
                "BG", "date",
                "All ROIs", "Individual ROIs"
            ])
            
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            
            print(f"Table columns set to: {len(headers)}")
            print(f"Headers: {headers}")
            
            # Update column widths based on new structure
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)   # View Plot button
            header.setSectionResizeMode(1, QHeaderView.Stretch)          # TIFF Path (immutable)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Reject
            header.setSectionResizeMode(3, QHeaderView.Interactive)      # Comment - make it resizable
            header.resizeSection(3, 150)  # Set initial width for comment column
            
            col_idx = 4
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
            
            # Check required columns - use filepath_without_number as primary key
            if 'filepath_without_number' in self.combined_df.columns:
                primary_grouping_col = 'filepath_without_number'
                required_columns = ['filepath_without_number', 'nth_set_label']
                print("Using 'filepath_without_number' as primary grouping column")
            else:
                primary_grouping_col = 'group'
                required_columns = ['group', 'nth_set_label']
                print("Falling back to 'group' as primary grouping column")
            
            missing_columns = [col for col in required_columns if col not in self.combined_df.columns]
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                self.log_message(f"ERROR: {error_msg}")
                self.status_label.setText("Data error")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                QMessageBox.critical(self, "Data Error", error_msg)
                return
            
            # Get unique combinations using the appropriate primary column
            valid_data = self.combined_df[
                (self.combined_df['nth_set_label'] >= 0) & 
                (self.combined_df['nth_set_label'].notna())
            ]
            
            if len(valid_data) == 0:
                self.log_message("No valid data entries found (nth_set_label >= 0)")
                self.status_label.setText("No valid data")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                return
            
            unique_combinations = valid_data.groupby([primary_grouping_col, 'nth_set_label']).first().reset_index()
            print(f"Found {len(unique_combinations)} unique {primary_grouping_col}/set combinations")
            
            self.table.setRowCount(len(unique_combinations))
            
            for row_idx, (_, row_data) in enumerate(unique_combinations.iterrows()):
                try:
                    # Use the primary grouping column value
                    group_id = row_data[primary_grouping_col]
                    set_label = row_data['nth_set_label']
                    
                    # Get display name (show 'group' if available, otherwise use the primary identifier)
                    if 'group' in row_data and primary_grouping_col == 'filepath_without_number':
                        display_group = row_data['group']
                        print(f"Processing row {row_idx}: Filepath={group_id}, Group={display_group}, Set={set_label}")
                    else:
                        display_group = group_id
                        print(f"Processing row {row_idx}: Group={group_id}, Set={set_label}")
                    
                    # Get group/set specific data using the primary grouping column
                    group_set_df = self.combined_df[
                        (self.combined_df[primary_grouping_col] == group_id) & 
                        (self.combined_df['nth_set_label'] == set_label)
                    ]
                    
                    # Get TIFF path
                    tiff_path = ""
                    if 'after_align_save_path' in group_set_df.columns:
                        tiff_paths = group_set_df['after_align_save_path'].dropna()
                        if len(tiff_paths) > 0:
                            tiff_path = tiff_paths.iloc[0]
                    
                    # Get full_plot_with_roi_path for View Plot button
                    plot_path = ""
                    if 'full_plot_with_roi_path' in group_set_df.columns:
                        plot_paths = group_set_df['full_plot_with_roi_path'].dropna()
                        if len(plot_paths) > 0:
                            plot_path = plot_paths.iloc[0]
                            print(f"Row {row_idx}: Found plot path: {plot_path}")
                        else:
                            print(f"Row {row_idx}: No plot path found in data")
                    else:
                        print(f"Row {row_idx}: full_plot_with_roi_path column not found in group_set_df")
                    
                    # View Plot button (first column) - ALWAYS create this button
                    view_plot_btn = QPushButton("📊 View")
                    view_plot_btn.setToolTip("Open full region plot with OS default viewer")
                    
                    if plot_path and os.path.exists(str(plot_path)):
                        view_plot_btn.setEnabled(True)
                        view_plot_btn.setStyleSheet("background-color: lightblue;")
                        view_plot_btn.clicked.connect(
                            lambda checked, path=plot_path: self.open_file_with_os_viewer(path)
                        )
                        print(f"Row {row_idx}: Created enabled View button for {plot_path}")
                    else:
                        view_plot_btn.setEnabled(False)
                        if plot_path:
                            view_plot_btn.setToolTip(f"Plot file not found: {plot_path}")
                            print(f"Row {row_idx}: Plot file not found: {plot_path}")
                        else:
                            view_plot_btn.setToolTip("Plot file not available")
                            print(f"Row {row_idx}: No plot path available")
                        view_plot_btn.setStyleSheet("background-color: lightgray;")
                    
                    # CRITICAL: Always set the button in the table
                    self.table.setCellWidget(row_idx, 0, view_plot_btn)
                    print(f"Row {row_idx}: View Plot button set in column 0")
                    
                    # TIFF path with tooltip showing full path (immutable)
                    # Show the display group name for better readability
                    tiff_display_name = f"{os.path.basename(str(tiff_path))}" if tiff_path else f"NoTIFF_{display_group}_set{set_label}"
                    tiff_item = QTableWidgetItem(tiff_display_name)
                    tiff_item.setToolTip(str(tiff_path))
                    tiff_item.setFlags(tiff_item.flags() & ~Qt.ItemIsEditable)  # Make immutable
                    self.table.setItem(row_idx, 1, tiff_item)
                    
                    col_idx = 2  # Start after TIFF Path
                    
                    # Reject checkbox
                    reject_checkbox = QCheckBox()
                    # Get current reject status for this group/set
                    current_reject = group_set_df['reject'].iloc[0] if len(group_set_df) > 0 else False
                    reject_checkbox.setChecked(current_reject)
                    reject_checkbox.stateChanged.connect(
                        lambda state, g=group_id, s=set_label: self.on_reject_changed(g, s, state)
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
                        lambda g=group_id, s=set_label, widget=comment_line_edit: self.on_comment_changed(g, s, widget.text())
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
                        
                        # Format datetime values to YY/MM/DD HH:MM format
                        if pd.notna(add_value):
                            try:
                                # Check if the column is datetime type (not string)
                                if pd.api.types.is_datetime64_any_dtype(group_set_df[add_col]):
                                    # Convert to datetime and format
                                    parsed_date = pd.to_datetime(add_value)
                                    formatted_value = parsed_date.strftime("%y/%m/%d %H:%M")  # YY/MM/DD HH:MM format
                                else:
                                    # For non-datetime columns, use original value
                                    formatted_value = str(add_value)
                            except (ValueError, TypeError):
                                formatted_value = str(add_value)
                        else:
                            formatted_value = ""
                        
                        add_item = QTableWidgetItem(formatted_value)
                        add_item.setTextAlignment(Qt.AlignCenter)
                        add_item.setFlags(add_item.flags() & ~Qt.ItemIsEditable)  # Make immutable by default
                        add_item.setToolTip(f"{add_col}: {add_value}")  # Add tooltip showing column name and original value
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
                        date_item.setFlags(date_item.flags() & ~Qt.ItemIsEditable)  # Make immutable
                        self.table.setItem(row_idx, col_idx, date_item)
                        col_idx += 1
                    
                    #for debug, print tiff_path 20250627
                    print("tiff_path", tiff_path)
                    print("os.path.exists(tiff_path)", os.path.exists(tiff_path))
                    print("bool(tiff_path and os.path.exists(tiff_path))", bool(tiff_path and os.path.exists(tiff_path)))
                    #till here

                    # All ROIs button
                    all_button = QPushButton("Launch All")
                    all_button.setEnabled(bool(tiff_path and os.path.exists(tiff_path)))
                    all_button.clicked.connect(
                        lambda checked, g=group_id, s=set_label, path=tiff_path: 
                        self.launch_all_roi_analysis(g, s, path)
                    )
                    
                    if all_defined:
                        all_button.setStyleSheet("background-color: lightgreen;")
                        all_button.setToolTip("All ROIs are defined - click to reanalyze")
                    else:
                        all_button.setToolTip("Launch ROI analysis for all three types")
                    
                    self.table.setCellWidget(row_idx, col_idx, all_button)
                    col_idx += 1
                    
                    # Individual ROIs buttons
                    individual_buttons_widget = QWidget()
                    individual_buttons_layout = QHBoxLayout(individual_buttons_widget)
                    individual_buttons_layout.setContentsMargins(2, 2, 2, 2)
                    individual_buttons_layout.setSpacing(2)
                    
                    # Create three buttons for ROI types
                    spine_button = QPushButton("Sp")
                    dendrite_button = QPushButton("Dn")
                    background_button = QPushButton("Bg")
                    
                    # Set button properties
                    for button in [spine_button, dendrite_button, background_button]:
                        button.setFixedSize(30, 25)
                        button.setEnabled(bool(tiff_path and os.path.exists(tiff_path)))
                    
                    # Connect button click events
                    spine_button.clicked.connect(
                        lambda checked, g=group_id, s=set_label, path=tiff_path: 
                        self.launch_individual_roi_analysis(g, s, path, "Spine")
                    )
                    dendrite_button.clicked.connect(
                        lambda checked, g=group_id, s=set_label, path=tiff_path: 
                        self.launch_individual_roi_analysis(g, s, path, "DendriticShaft")
                    )
                    background_button.clicked.connect(
                        lambda checked, g=group_id, s=set_label, path=tiff_path: 
                        self.launch_individual_roi_analysis(g, s, path, "Background")
                    )
                    
                    # Add buttons to layout
                    individual_buttons_layout.addWidget(spine_button)
                    individual_buttons_layout.addWidget(dendrite_button)
                    individual_buttons_layout.addWidget(background_button)
                    individual_buttons_layout.addStretch()
                    
                    self.table.setCellWidget(row_idx, col_idx, individual_buttons_widget)
                    col_idx += 1
                    
                except Exception as row_error:
                    print(f"Error processing row {row_idx}: {row_error}")
                    import traceback
                    traceback.print_exc()
                    self.log_message(f"ERROR processing row {row_idx}: {row_error}")
                    continue
            
            self.status_label.setText(f"Loaded {len(unique_combinations)} entries")
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
            
            # DEBUG: Log the state of ROI masks after analysis
            self.debug_roi_mask_state(group, set_label)
            
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
            
            # DEBUG: Log the state of ROI masks after analysis
            self.debug_roi_mask_state(group, set_label, roi_type)
            
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

    def on_reject_changed(self, group_id, set_label, state):
        """Handle reject checkbox state change"""
        try:
            # Determine which column to use for filtering
            if 'filepath_without_number' in self.combined_df.columns:
                mask = (self.combined_df['filepath_without_number'] == group_id) & (self.combined_df['nth_set_label'] == set_label)
            else:
                mask = (self.combined_df['group'] == group_id) & (self.combined_df['nth_set_label'] == set_label)
            
            self.combined_df.loc[mask, 'reject'] = state == Qt.Checked
            self.log_message(f"Reject status changed for {group_id}, Set {set_label}: {'Rejected' if state == Qt.Checked else 'Accepted'}")
            self.auto_save_dataframe()
        except Exception as e:
            print(f"Error handling reject change: {e}")
            import traceback
            traceback.print_exc()
    
    def on_comment_changed(self, group_id, set_label, comment):
        """Handle comment text change"""
        try:
            # Determine which column to use for filtering
            if 'filepath_without_number' in self.combined_df.columns:
                mask = (self.combined_df['filepath_without_number'] == group_id) & (self.combined_df['nth_set_label'] == set_label)
            else:
                mask = (self.combined_df['group'] == group_id) & (self.combined_df['nth_set_label'] == set_label)
            
            self.combined_df.loc[mask, 'comment'] = comment
            self.log_message(f"Comment changed for {group_id}, Set {set_label}: {comment}")
            self.auto_save_dataframe()
        except Exception as e:
            print(f"Error handling comment change: {e}")
            import traceback
            traceback.print_exc()
    
    def auto_save_dataframe(self):
        """Auto-save the combined_df to the specified path"""
        if self.df_save_path_2 and self.save_auto:
            try:
                self.combined_df.to_pickle(self.df_save_path_2)
                self.log_message(f"DataFrame auto-saved to {self.df_save_path_2}")
            except Exception as e:
                self.log_message(f"ERROR: Failed to auto-save DataFrame: {e}")
                print(f"Auto-save error: {e}")
        else:
            self.log_message("No save path specified for auto-save or save_auto is False")
    
    def manual_save_dataframe(self):
        """Manually save the combined_df to the specified path (ignores save_auto setting)"""
        if self.df_save_path_2:
            try:
                self.combined_df.to_pickle(self.df_save_path_2)
                self.log_message(f"DataFrame manually saved to {self.df_save_path_2}")
                self.status_label.setText("Saved")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
            except Exception as e:
                self.log_message(f"ERROR: Failed to manually save DataFrame: {e}")
                print(f"Manual save error: {e}")
                self.status_label.setText("Save failed")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.log_message("ERROR: No save path specified for manual save")
            self.status_label.setText("No save path")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

    def close_and_save(self):
        """Save the DataFrame manually, then close the window."""
        self.manual_save_dataframe()        
        self.close()
    
    def debug_roi_mask_state(self, group, set_label, roi_type=None):
        """Debug method to log ROI mask state after analysis"""
        try:
            # Find the relevant rows
            if 'filepath_without_number' in self.combined_df.columns:
                mask = (self.combined_df['filepath_without_number'] == group) & (self.combined_df['nth_set_label'] == set_label)
            else:
                mask = (self.combined_df['group'] == group) & (self.combined_df['nth_set_label'] == set_label)
            
            analysis_mask = mask & (self.combined_df['nth_omit_induction'] >= 0)
            analysis_rows = self.combined_df[analysis_mask]
            
            if len(analysis_rows) == 0:
                self.log_message(f"DEBUG: No analysis rows found for {group}, Set {set_label}")
                return
            
            if roi_type:
                # Check specific ROI type
                roi_columns = [f"{roi_type}_roi_mask"]
                self.log_message(f"DEBUG: Checking {roi_type} ROI masks for {group}, Set {set_label}")
            else:
                # Check all ROI types
                roi_columns = [col for col in self.combined_df.columns if col.endswith('_roi_mask')]
                self.log_message(f"DEBUG: Checking all ROI masks for {group}, Set {set_label}")
            
            for roi_col in roi_columns:
                if roi_col in self.combined_df.columns:
                    non_null_count = analysis_rows[roi_col].notna().sum()
                    total_count = len(analysis_rows)
                    self.log_message(f"DEBUG: {roi_col}: {non_null_count}/{total_count} rows have masks")
                    
                    # Check a sample mask
                    sample_masks = analysis_rows[analysis_rows[roi_col].notna()]
                    if len(sample_masks) > 0:
                        sample_mask = sample_masks.iloc[0][roi_col]
                        if isinstance(sample_mask, np.ndarray):
                            mask_pixels = np.sum(sample_mask)
                            self.log_message(f"DEBUG: Sample {roi_col} shape: {sample_mask.shape}, pixels: {mask_pixels}")
                        else:
                            self.log_message(f"DEBUG: Sample {roi_col} type: {type(sample_mask)}")
                else:
                    self.log_message(f"DEBUG: Column {roi_col} not found in DataFrame")
                    
        except Exception as e:
            self.log_message(f"ERROR in debug_roi_mask_state: {e}")
            print(f"Debug error: {e}")
            import traceback
            traceback.print_exc()


def launch_file_selection_gui(combined_df, df_save_path_2=None, additional_columns=None, save_auto = True):
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
        gui = FileSelectionGUI(combined_df, df_save_path_2, additional_columns, save_auto)
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