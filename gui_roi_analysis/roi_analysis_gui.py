import sys
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse, Polygon
from matplotlib.path import Path
import matplotlib.patches as patches
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QPushButton, QSlider, QLabel, QButtonGroup,
    QFrame, QSplitter, QSizePolicy, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

class ROIAnalysisGUI(QMainWindow):
    """Main GUI application for ROI analysis with time series data."""
    
    def __init__(self, combined_df, after_align_tiff_data, max_proj_image, uncaging_info=None, file_info=None, header="ROI"):
        super().__init__()
        self.combined_df = combined_df
        self.after_align_tiff_data = after_align_tiff_data
        self.max_proj_image = max_proj_image
        self.uncaging_info = uncaging_info or {'has_uncaging': False}
        self.file_info = file_info or {'filename': 'Unknown', 'group': 'Unknown', 'set_label': 'Unknown'}
        self.frame_info_df = self.file_info.get('frame_info_df')  # per-frame CSV if available
        self.header = header  # ROI analysis header for column names and display
        
        # Set matplotlib parameters for consistent thin lines
        plt.rcParams['axes.linewidth'] = 0.5
        plt.rcParams['lines.linewidth'] = 0.5
        plt.rcParams['patch.linewidth'] = 0.5
        
        # Debug information
        print(f"GUI initialized with:")
        print(f"  TIFF data shape: {after_align_tiff_data.shape}")
        print(f"  Max proj shape: {max_proj_image.shape}")
        print(f"  Combined df shape: {combined_df.shape}")
        print(f"  Header: {header}")
        
        # Current state variables
        self.current_frame = 0
        self.total_frames = after_align_tiff_data.shape[0]
        self.current_roi = None
        
        # Set default ROI shape based on header (ROI type)
        if header == "Background":
            self.roi_shape = 'rectangle'  # Use rectangle for Background ROI
        else:
            self.roi_shape = 'polygon'    # Use polygon for Spine and Dendrite ROI
        
        self.is_defining_roi = True  # True for max proj, False for time series
        self.roi_parameters = {}
        
        # ROI interaction state
        self.is_drawing = False
        self.is_moving_roi = False
        self.roi_start_pos = None
        self.drag_start_pos = None
        self.temp_roi = None  # For real-time drawing preview
        self.polygon_points = []  # For polygon creation
        
        # Double-click detection for polygon
        self.last_click_time = 0
        self.last_click_pos = None
        self.double_click_threshold = 0.3  # seconds
        
        # Intensity data storage
        self.intensity_data = {
            'mean': [0.0] * self.total_frames,
            'max': [0.0] * self.total_frames,
            'sum': [0.0] * self.total_frames
        }
        
        # ROI position storage for each frame
        self.roi_positions_per_frame = {}
        
        # Frame-specific ROI parameters storage
        self.frame_roi_parameters = {}  # Dictionary to store ROI parameters for each frame
        self.use_frame_specific_rois = True  # Always enable frame-specific ROI mode
        
        # Analysis state
        self.analysis_completed = False
        self.roi_moved_in_current_frame = False
        
        # ROI coordinate snapping option
        self.snap_to_integer = True  # New: Enable integer coordinate snapping
        
        # Display intensity: vmax scale in percent (100 = auto from 99th percentile)
        self.vmax_scale = 100
        self._current_base_vmax = None
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(f"{self.header} Analysis Tool")
        self.setGeometry(100, 100, 1200, 900)  # Increased height for header
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QGridLayout(central_widget)
        
        # Create header display
        self.create_header_display()
        
        # Create GUI components
        self.create_image_display()
        self.create_roi_control_panel()
        self.create_plot_display()
        self.create_navigation_panel()
        
        # Add components to layout
        main_layout.addWidget(self.header_frame, 0, 0, 1, 2)  # Header spans both columns
        main_layout.addWidget(self.image_frame, 1, 0, 1, 1)
        main_layout.addWidget(self.roi_control_frame, 1, 1, 1, 1)
        main_layout.addWidget(self.plot_frame, 2, 0, 1, 1)
        main_layout.addWidget(self.plot_control_frame, 2, 1, 1, 1)
        main_layout.addWidget(self.navigation_frame, 3, 0, 1, 2)  # Navigation spans both columns
        
        # Set column and row stretch
        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 1)
        main_layout.setRowStretch(0, 0)  # Header row doesn't stretch
        main_layout.setRowStretch(1, 2)  # Image/control row
        main_layout.setRowStretch(2, 2)  # Plot row
        main_layout.setRowStretch(3, 0)  # Navigation row doesn't stretch
        
    def create_header_display(self):
        """Create the header display area."""
        self.header_frame = QFrame()
        self.header_frame.setFrameStyle(QFrame.StyledPanel)
        self.header_frame.setFixedHeight(80)
        self.header_frame.setStyleSheet("QFrame { background-color: lightblue; border: 2px solid darkblue; }")
        
        layout = QVBoxLayout(self.header_frame)
        
        # Main header label
        header_label = QLabel(f"{self.header} Analysis")
        header_label.setFont(QFont("Arial", 24, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("QLabel { color: darkblue; background-color: transparent; }")
        layout.addWidget(header_label)
        
        # Sub-header with file info
        file_info_text = f"Group: {self.file_info.get('group', 'Unknown')} | Set: {self.file_info.get('set_label', 'Unknown')}"
        sub_header_label = QLabel(file_info_text)
        sub_header_label.setFont(QFont("Arial", 12))
        sub_header_label.setAlignment(Qt.AlignCenter)
        sub_header_label.setStyleSheet("QLabel { color: darkblue; background-color: transparent; }")
        layout.addWidget(sub_header_label)
        
    def create_image_display(self):
        """Create the main image display area."""
        self.image_frame = QFrame()
        self.image_frame.setFrameStyle(QFrame.StyledPanel)
        
        main_layout = QHBoxLayout(self.image_frame)
        
        # Left: vmax (intensity) slider
        slider_panel = QFrame()
        slider_panel.setFixedWidth(56)
        slider_layout = QVBoxLayout(slider_panel)
        vmax_label = QLabel("vmax\n%")
        vmax_label.setFont(QFont("Arial", 8))
        vmax_label.setAlignment(Qt.AlignCenter)
        slider_layout.addWidget(vmax_label)
        self.vmax_slider = QSlider(Qt.Vertical)
        self.vmax_slider.setRange(10, 200)
        self.vmax_slider.setValue(100)
        self.vmax_slider.setToolTip("Display intensity scale (percent of auto vmax)")
        slider_layout.addWidget(self.vmax_slider, 1)
        main_layout.addWidget(slider_panel)
        
        # Right: matplotlib figure for image display
        layout = QVBoxLayout()
        self.image_figure = Figure(figsize=(8, 6))
        self.image_canvas = FigureCanvas(self.image_figure)
        self.image_ax = self.image_figure.add_subplot(111)
        layout.addWidget(self.image_canvas)
        main_layout.addLayout(layout, 1)
        
        # Display initial max projection image
        self.display_max_proj()
        
    def create_roi_control_panel(self):
        """Create the ROI control panel."""
        self.roi_control_frame = QFrame()
        self.roi_control_frame.setFrameStyle(QFrame.StyledPanel)
        self.roi_control_frame.setFixedWidth(300)
        
        layout = QVBoxLayout(self.roi_control_frame)
        
        # ROI shape selection
        shape_label = QLabel("ROI Shape:")
        shape_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(shape_label)
        
        shape_layout = QHBoxLayout()
        self.shape_button_group = QButtonGroup()
        
        self.rect_button = QPushButton("□")
        self.ellipse_button = QPushButton("○")
        self.polygon_button = QPushButton("△")
        
        self.rect_button.setCheckable(True)
        self.ellipse_button.setCheckable(True)
        self.polygon_button.setCheckable(True)
        
        # Set default button selection based on ROI type
        if self.header == "Background":
            self.rect_button.setChecked(True)      # Rectangle for Background
        else:
            self.polygon_button.setChecked(True)   # Polygon for Spine and Dendrite
        
        self.shape_button_group.addButton(self.rect_button, 0)
        self.shape_button_group.addButton(self.ellipse_button, 1)
        self.shape_button_group.addButton(self.polygon_button, 2)
        
        shape_layout.addWidget(self.rect_button)
        shape_layout.addWidget(self.ellipse_button)
        shape_layout.addWidget(self.polygon_button)
        layout.addLayout(shape_layout)
        
        # Integer coordinate snapping option
        coord_snap_layout = QHBoxLayout()
        snap_label = QLabel("Snap to pixels:")
        snap_label.setFont(QFont("Arial", 9))
        self.snap_checkbox = QCheckBox()
        self.snap_checkbox.setChecked(self.snap_to_integer)
        self.snap_checkbox.setToolTip("Round coordinates to integer pixel values for precise positioning")
        coord_snap_layout.addWidget(snap_label)
        coord_snap_layout.addWidget(self.snap_checkbox)
        coord_snap_layout.addStretch()
        layout.addLayout(coord_snap_layout)
        
        # ROI parameters display (hidden per user request)
        params_label = QLabel("ROI Parameters:")
        params_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(params_label)
        params_label.setVisible(False)
        
        self.params_display = QLabel("No ROI defined")
        layout.addWidget(self.params_display)
        self.params_display.setVisible(False)
        
        # Back to max proj button
        self.back_to_maxproj_button = QPushButton("Back to Max Proj")
        self.back_to_maxproj_button.setEnabled(False)
        layout.addWidget(self.back_to_maxproj_button)
        
        # Complete analysis button
        self.complete_analysis_button = QPushButton("Analysis Complete")
        self.complete_analysis_button.setEnabled(False)
        self.complete_analysis_button.setStyleSheet("QPushButton { background-color: lightgreen; }")
        layout.addWidget(self.complete_analysis_button)
        
        # Intensity values display
        intensity_label = QLabel("Intensity Values:")
        intensity_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(intensity_label)
        
        self.intensity_display = QLabel("Mean: -\nMax: -\nSum: -")
        layout.addWidget(self.intensity_display)
        
        # File information display
        file_info_label = QLabel("File Information:")
        file_info_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(file_info_label)
        
        # Display file info with small font (per-frame if frame_info_df available)
        file_info_text = self._build_file_info_text_for_frame(0)
        self.file_info_display = QLabel(file_info_text)
        self.file_info_display.setFont(QFont("Arial", 8))
        self.file_info_display.setWordWrap(True)
        layout.addWidget(self.file_info_display)
        
        layout.addStretch()

    def _build_file_info_text_for_frame(self, frame_index):
        """Build file info text for the right panel. Uses frame_info_df if available (per-frame)."""
        group = self.file_info.get('group', 'Unknown')
        set_label = self.file_info.get('set_label', 'Unknown')
        if self.frame_info_df is None or frame_index < 0 or frame_index >= len(self.frame_info_df):
            filename = self.file_info.get('filename', 'Unknown')
            return f"File: {filename}\nGroup: {group}\nSet: {set_label}"
        row = self.frame_info_df.iloc[frame_index]
        lines = [
            f"File: {row.get('filename', '') or '—'}",
            f"Group: {group}",
            f"Set: {set_label}",
            f"Phase: {self._phase_display(row.get('phase', ''))}",
        ]
        if 'elapsed_time_sec' in self.frame_info_df.columns:
            v = row.get('elapsed_time_sec')
            if pd.isna(v):
                lines.append("elapsed_time_sec: — sec")
            else:
                try:
                    lines.append(f"elapsed_time_sec: {float(v):.2f} sec")
                except (TypeError, ValueError):
                    lines.append("elapsed_time_sec: — sec")
        if 'Z_projection' in self.frame_info_df.columns and row.get('Z_projection') and 'z_from' in self.frame_info_df.columns and 'z_to' in self.frame_info_df.columns:
            zf, zt = row.get('z_from'), row.get('z_to')
            if not (pd.isna(zf) or pd.isna(zt)):
                try:
                    z_from = int(zf)
                    z_to_last = int(zt) - 1  # Python slice z_from:z_to uses indices up to z_to-1
                    lines.append(f"Z projection, slice {z_from} to {z_to_last}")
                except (TypeError, ValueError):
                    lines.append("Single slice")
            else:
                lines.append("Single slice")
        else:
            lines.append("Single slice")
        return "\n".join(lines)

    def _phase_display(self, phase):
        """Display phase as pre / post / unc."""
        if phase == 'uncaging':
            return 'unc'
        return phase if phase else '—'

    def create_plot_display(self):
        """Create the time series plot display."""
        self.plot_frame = QFrame()
        self.plot_frame.setFrameStyle(QFrame.StyledPanel)
        self.plot_frame.setMinimumHeight(220)
        
        layout = QVBoxLayout(self.plot_frame)
        
        # Create matplotlib figure for plot
        self.plot_figure = Figure(figsize=(8, 4))
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_ax = self.plot_figure.add_subplot(111)
        
        layout.addWidget(self.plot_canvas)
        
        # Set plot style
        self.setup_plot_style()
        self.plot_figure.tight_layout()
        
    def create_navigation_panel(self):
        """Create the navigation panel."""
        self.navigation_frame = QFrame()
        self.navigation_frame.setFrameStyle(QFrame.StyledPanel)
        self.navigation_frame.setFixedHeight(80)
        
        layout = QHBoxLayout(self.navigation_frame)
        
        # Previous button
        self.prev_button = QPushButton("←")
        self.prev_button.setFixedSize(40, 40)
        self.prev_button.setEnabled(False)
        
        # Next button
        self.next_button = QPushButton("→")
        self.next_button.setFixedSize(40, 40)
        self.next_button.setEnabled(False)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        
        # Frame info
        self.frame_info_label = QLabel(f"Frame 1/{self.total_frames}")
        self.frame_info_label.setFont(QFont("Arial", 12))
        
        layout.addWidget(self.prev_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.frame_slider)
        layout.addWidget(self.frame_info_label)
        
        # Plot control frame
        self.plot_control_frame = QFrame()
        self.plot_control_frame.setFrameStyle(QFrame.StyledPanel)
        self.plot_control_frame.setFixedWidth(300)
        
        plot_control_layout = QVBoxLayout(self.plot_control_frame)
        
        plot_label = QLabel("Plot Display:")
        plot_label.setFont(QFont("Arial", 10, QFont.Bold))
        plot_control_layout.addWidget(plot_label)
        
        # Plot type buttons
        plot_button_layout = QVBoxLayout()
        self.plot_button_group = QButtonGroup()
        
        self.mean_plot_button = QPushButton("Average")
        self.max_plot_button = QPushButton("Maximum")
        self.sum_plot_button = QPushButton("Sum")
        
        self.mean_plot_button.setCheckable(True)
        self.max_plot_button.setCheckable(True)
        self.sum_plot_button.setCheckable(True)
        self.mean_plot_button.setChecked(True)
        
        self.plot_button_group.addButton(self.mean_plot_button, 0)
        self.plot_button_group.addButton(self.max_plot_button, 1)
        self.plot_button_group.addButton(self.sum_plot_button, 2)
        
        plot_button_layout.addWidget(self.mean_plot_button)
        plot_button_layout.addWidget(self.max_plot_button)
        plot_button_layout.addWidget(self.sum_plot_button)
        plot_control_layout.addLayout(plot_button_layout)
        
        plot_control_layout.addStretch()
        
    def setup_connections(self):
        """Set up signal-slot connections."""
        # Mouse events
        self.image_canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.image_canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.image_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Shape selection
        self.shape_button_group.buttonClicked.connect(self.on_shape_changed)
        
        # Navigation buttons
        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button.clicked.connect(self.next_frame)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        
        # Plot control
        self.plot_button_group.buttonClicked.connect(self.update_plot)
        
        # Control buttons
        self.back_to_maxproj_button.clicked.connect(self.back_to_max_proj)
        self.complete_analysis_button.clicked.connect(self.complete_analysis)
        
        # Snap to pixels checkbox
        self.snap_checkbox.stateChanged.connect(self.on_snap_changed)
        
        # vmax (intensity) slider
        self.vmax_slider.valueChanged.connect(self.on_vmax_slider_changed)
        
    def on_vmax_slider_changed(self, value):
        """Update display intensity scale from vmax slider (percent)."""
        self.vmax_scale = value
        self._refresh_image_display()
    
    def _refresh_image_display(self):
        """Redraw current image (max proj or time-series frame) with current vmax scale."""
        if self.is_defining_roi:
            self.display_max_proj()
        else:
            self.display_time_series_frame(self.current_frame)
        
    def setup_plot_style(self):
        """Setup the plot style for scientific appearance."""
        self.plot_ax.set_facecolor('white')
        self.plot_ax.grid(True, color='lightgray', linewidth=0.5)
        self.plot_ax.spines['top'].set_color('black')
        self.plot_ax.spines['bottom'].set_color('black')
        self.plot_ax.spines['left'].set_color('black')
        self.plot_ax.spines['right'].set_color('black')
        
    def display_max_proj(self):
        """Display the max projection image."""
        img = np.asarray(self.max_proj_image).astype(float)
        base = float(np.nanpercentile(img, 99)) if np.any(np.isfinite(img)) else 1.0
        if base <= 0:
            base = float(np.nanmax(img)) if np.any(np.isfinite(img)) else 1.0
        self._current_base_vmax = base
        vmax = self._current_base_vmax * self.vmax_scale / 100.0
        self.image_ax.clear()
        self.image_ax.imshow(self.max_proj_image, cmap='gray', vmin=0, vmax=vmax)
        self.image_ax.set_title("Max Projection - ROI Definition")
        self.image_ax.axis('off')
        
        # Display uncaging position if available
        if self.uncaging_info.get('has_uncaging', False):
            uncaging_x = self.uncaging_info.get('x', 0)
            uncaging_y = self.uncaging_info.get('y', 0)
            # Display uncaging position as red scatter point (same as save_small_region_plots)
            self.image_ax.scatter(uncaging_x, uncaging_y, color="red", s=100, 
                                 marker='o', edgecolor='white', linewidth=1, zorder=10)
        
        # Display ROI if available
        if self.current_roi is not None and self.roi_parameters:
            # Use copy of current_roi to avoid modifying the original
            roi_copy = None
            if self.roi_shape == 'rectangle':
                if 'x' in self.roi_parameters and 'y' in self.roi_parameters:
                    x = self.roi_parameters['x']
                    y = self.roi_parameters['y']
                    width = self.roi_parameters['width']
                    height = self.roi_parameters['height']
                    roi_copy = Rectangle((x, y), width, height, 
                                       linewidth=2, edgecolor='red', facecolor='none')
            elif self.roi_shape == 'ellipse':
                if 'center_x' in self.roi_parameters and 'center_y' in self.roi_parameters:
                    center_x = self.roi_parameters['center_x']
                    center_y = self.roi_parameters['center_y']
                    width = self.roi_parameters['width']
                    height = self.roi_parameters['height']
                    roi_copy = Ellipse((center_x, center_y), width, height, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            elif self.roi_shape == 'polygon':
                if 'points' in self.roi_parameters:
                    points = self.roi_parameters['points']
                    roi_copy = Polygon(points, closed=True, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            
            if roi_copy:
                self.image_ax.add_patch(roi_copy)
        
        self.image_canvas.draw()
        
    def display_time_series_frame(self, frame_idx):
        """Display a specific frame from the time series."""
        frame_data = self.after_align_tiff_data[frame_idx]
        
        # Handle different dimensions for frame data
        if len(frame_data.shape) == 3:
            frame_2d = frame_data.max(axis=0)
        elif len(frame_data.shape) == 2:
            frame_2d = frame_data
        else:
            frame_2d = frame_data
            while len(frame_2d.shape) > 2:
                frame_2d = frame_2d.max(axis=0)
        
        img = np.asarray(frame_2d).astype(float)
        base = float(np.nanpercentile(img, 99)) if np.any(np.isfinite(img)) else 1.0
        if base <= 0:
            base = float(np.nanmax(img)) if np.any(np.isfinite(img)) else 1.0
        self._current_base_vmax = base
        vmax = self._current_base_vmax * self.vmax_scale / 100.0
        self.image_ax.clear()
        self.image_ax.imshow(frame_2d, cmap='gray', aspect='equal', vmin=0, vmax=vmax)
        self.image_ax.set_title(f"Frame {frame_idx + 1}/{self.total_frames}")
        
        # Display ROI if available
        if self.current_roi is not None and self.roi_parameters:
            # Use copy of current_roi to avoid modifying the original
            roi_copy = None
            if self.roi_shape == 'rectangle':
                if 'x' in self.roi_parameters and 'y' in self.roi_parameters:
                    x = self.roi_parameters['x']
                    y = self.roi_parameters['y']
                    width = self.roi_parameters['width']
                    height = self.roi_parameters['height']
                    roi_copy = Rectangle((x, y), width, height, 
                                       linewidth=2, edgecolor='red', facecolor='none')
            elif self.roi_shape == 'ellipse':
                if 'center_x' in self.roi_parameters and 'center_y' in self.roi_parameters:
                    center_x = self.roi_parameters['center_x']
                    center_y = self.roi_parameters['center_y']
                    width = self.roi_parameters['width']
                    height = self.roi_parameters['height']
                    roi_copy = Ellipse((center_x, center_y), width, height, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            elif self.roi_shape == 'polygon':
                if 'points' in self.roi_parameters:
                    points = self.roi_parameters['points']
                    roi_copy = Polygon(points, closed=True, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            
            if roi_copy:
                self.image_ax.add_patch(roi_copy)
        
        self.image_canvas.draw()
        
    def display_roi(self):
        """Display the current ROI on the image."""
        if self.current_roi is None:
            return
            
        # Create a copy of the ROI for display with pixel-aligned edges
        if self.roi_shape == 'rectangle':
            x = round(self.roi_parameters['x']) - 0.5
            y = round(self.roi_parameters['y']) - 0.5
            width = round(self.roi_parameters['width'])
            height = round(self.roi_parameters['height'])
            display_roi = Rectangle((x, y), width, height)
        elif self.roi_shape == 'ellipse':
            center_x = round(self.roi_parameters['center_x'])
            center_y = round(self.roi_parameters['center_y'])
            width = round(self.roi_parameters['width'])
            height = round(self.roi_parameters['height'])
            display_roi = Ellipse((center_x, center_y), width, height)
        elif self.roi_shape == 'polygon':
            rounded_points = [(round(p[0]), round(p[1])) for p in self.roi_parameters['points']]
            display_roi = Polygon(rounded_points)
        else:
            return
            
        # Apply styling consistently for all ROI shapes
        display_roi.set_edgecolor('red')
        display_roi.set_facecolor('none')
        display_roi.set_linewidth(1.5)
        display_roi.set_antialiased(True)
        
        self.image_ax.add_patch(display_roi)
        
    def on_shape_changed(self, button):
        """Handle ROI shape selection change."""
        shape_map = {0: 'rectangle', 1: 'ellipse', 2: 'polygon'}
        self.roi_shape = shape_map[self.shape_button_group.id(button)]
        
    def on_mouse_press(self, event):
        """Handle mouse press events for ROI creation and interaction."""
        if event.inaxes != self.image_ax:
            return
            
        # Round coordinates to integers for pixel-perfect positioning
        x, y = self.round_to_int(event.xdata, event.ydata)
        
        if event.button == 1:  # Left click
            if self.is_defining_roi:
                if self.roi_shape == 'polygon':
                    self.handle_polygon_click(x, y)
                else:
                    self.start_roi_creation(x, y)
            else:
                # Check if clicking on existing ROI for moving
                if self.is_point_in_roi(x, y):
                    self.start_roi_move(x, y)
                    
    def on_mouse_release(self, event):
        """Handle mouse release events for ROI creation and interaction."""
        if event.inaxes != self.image_ax:
            return
            
        x, y = self.round_to_int(event.xdata, event.ydata)
            
        if event.button == 1:  # Left click
            if self.is_defining_roi and self.is_drawing:
                if self.roi_shape != 'polygon':
                    self.finish_roi_creation(x, y)
            elif self.is_moving_roi:
                self.finish_roi_move()
                
    def on_mouse_move(self, event):
        """Handle mouse move events for ROI creation and interaction."""
        if event.inaxes != self.image_ax:
            return
            
        x, y = self.round_to_int(event.xdata, event.ydata)
            
        if self.is_drawing and self.roi_start_pos:
            # Show temporary ROI during drawing
            self.redraw_image_with_temp_roi(x, y)
        elif self.is_moving_roi:
            # Move the ROI
            self.update_roi_position(x, y)
            
    def start_roi_creation(self, x, y):
        """Start creating a new ROI."""
        self.roi_start_pos = (x, y)
        self.is_drawing = True
        
    def finish_roi_creation(self, x, y):
        """Finish creating a new ROI."""
        if self.roi_shape == 'rectangle':
            self.create_rectangle_roi(self.roi_start_pos, (x, y))
        elif self.roi_shape == 'ellipse':
            self.create_ellipse_roi(self.roi_start_pos, (x, y))
        
        self.is_drawing = False
        self.roi_start_pos = None
        
    def create_rectangle_roi(self, start_pos, end_pos):
        """Create a rectangular ROI."""
        x1, y1 = self.round_to_int(*start_pos)
        x2, y2 = self.round_to_int(*end_pos)
        
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        x = min(x1, x2)
        y = min(y1, y2)
        
        # Ensure minimum size
        width = max(1, width)
        height = max(1, height)
        
        # Constrain ROI to image boundaries
        image_height, image_width = self.max_proj_image.shape
        x = max(0, min(x, image_width - width))
        y = max(0, min(y, image_height - height))
        width = min(width, image_width - x)
        height = min(height, image_height - y)
        
        self.current_roi = Rectangle((x, y), width, height)
        self.roi_parameters = {
            'x': float(x), 'y': float(y), 'width': float(width), 'height': float(height)
        }
        
        self.update_roi_display()
        self.enable_time_series_mode()
        
    def create_ellipse_roi(self, start_pos, end_pos):
        """Create an elliptical ROI."""
        x1, y1 = self.round_to_int(*start_pos)
        x2, y2 = self.round_to_int(*end_pos)
        
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Ensure minimum size
        width = max(1, width)
        height = max(1, height)
        
        # Constrain ROI to image boundaries
        image_height, image_width = self.max_proj_image.shape
        half_width = width / 2.0
        half_height = height / 2.0
        
        # Adjust center to keep ellipse within bounds
        center_x = max(half_width, min(center_x, image_width - half_width))
        center_y = max(half_height, min(center_y, image_height - half_height))
        
        # Adjust size if needed
        max_width = min(width, 2 * center_x, 2 * (image_width - center_x))
        max_height = min(height, 2 * center_y, 2 * (image_height - center_y))
        width = max_width
        height = max_height
        
        self.current_roi = Ellipse((center_x, center_y), width, height)
        self.roi_parameters = {
            'center_x': float(center_x), 'center_y': float(center_y), 
            'width': float(width), 'height': float(height)
        }
        
        self.update_roi_display()
        self.enable_time_series_mode()
        
    def handle_polygon_click(self, x, y):
        """Handle polygon creation clicks."""
        x, y = self.round_to_int(x, y)
        
        current_time = time.time()
        
        # Check for double-click
        if (self.last_click_time > 0 and 
            current_time - self.last_click_time < self.double_click_threshold):
            # Double-click detected - finish polygon
            if len(self.polygon_points) >= 3:
                self.finish_polygon_creation()
            return
        
        # Single click - add point
        self.polygon_points.append((x, y))
        self.redraw_image_with_temp_roi(x, y)
        
        # Update click tracking
        self.last_click_time = current_time
        
    def finish_polygon_creation(self):
        """Finish creating a polygon ROI."""
        if len(self.polygon_points) >= 3:
            self.current_roi = Polygon(self.polygon_points)
            self.roi_parameters = {'points': self.polygon_points.copy()}
            self.polygon_points = []
            self.update_roi_display()
            self.enable_time_series_mode()
        
    def update_roi_display(self):
        """Update the ROI display and parameters."""
        self.display_max_proj()
        
        # Update parameters display using helper method
        self.update_roi_display_params()
        
    def enable_time_series_mode(self):
        """Enable time series analysis mode."""
        self.is_defining_roi = False
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self.frame_slider.setEnabled(True)
        self.back_to_maxproj_button.setEnabled(True)
        self.complete_analysis_button.setEnabled(True)
        
        # Initialize frame tracking for frame-specific ROI mode
        self.previous_frame = 0
        
        # Set ROI parameters for the first frame only (others will inherit when visited)
        if self.roi_parameters:
            self.frame_roi_parameters[0] = self.roi_parameters.copy()
            print(f"Initialized ROI parameters for frame 0: {self.roi_parameters}")
        
        # Calculate intensity for all frames
        self.calculate_all_intensities()
        
        # Display first frame
        self.display_time_series_frame(0)
        self.update_plot()
        
    def calculate_all_intensities(self):
        """Calculate intensity values for all frames."""
        self.intensity_data = {'mean': [], 'max': [], 'sum': []}
        
        for frame_idx in range(self.total_frames):
            frame_data = self.after_align_tiff_data[frame_idx]
            
            # Handle different dimensions for frame data
            if len(frame_data.shape) == 3:
                frame_2d = frame_data.max(axis=0)
            elif len(frame_data.shape) == 2:
                frame_2d = frame_data
            else:
                frame_2d = frame_data
                while len(frame_2d.shape) > 2:
                    frame_2d = frame_2d.max(axis=0)
            
            # Get ROI parameters for this frame
            if self.use_frame_specific_rois and frame_idx in self.frame_roi_parameters:
                # Use frame-specific ROI parameters
                roi_params_for_frame = self.frame_roi_parameters[frame_idx]
            else:
                # Use global ROI parameters
                roi_params_for_frame = self.roi_parameters
            
            # Calculate intensity with frame-specific or global ROI
            roi_mask = self.create_roi_mask_with_params(frame_2d.shape, roi_params_for_frame)
            
            if np.any(roi_mask):
                roi_values = frame_2d[roi_mask]
                self.intensity_data['mean'].append(np.mean(roi_values))
                self.intensity_data['max'].append(np.max(roi_values))
                self.intensity_data['sum'].append(np.sum(roi_values))
            else:
                self.intensity_data['mean'].append(0)
                self.intensity_data['max'].append(0)
                self.intensity_data['sum'].append(0)
        
        self.update_intensity_display()
        
    def create_roi_mask_with_params(self, image_shape, roi_params):
        """Create a boolean mask for ROI with specific parameters."""
        mask = np.zeros(image_shape, dtype=bool)
        
        if not roi_params:
            return mask
        
        if self.roi_shape == 'rectangle':
            if all(key in roi_params for key in ['x', 'y', 'width', 'height']):
                x, y, width, height = (
                    int(round(roi_params['x'])),
                    int(round(roi_params['y'])),
                    int(round(roi_params['width'])),
                    int(round(roi_params['height']))
                )
                # Ensure bounds are within image
                x = max(0, min(x, image_shape[1]-1))
                y = max(0, min(y, image_shape[0]-1))
                x_end = max(0, min(x + width, image_shape[1]))
                y_end = max(0, min(y + height, image_shape[0]))
                mask[y:y_end, x:x_end] = True
            
        elif self.roi_shape == 'ellipse':
            if all(key in roi_params for key in ['center_x', 'center_y', 'width', 'height']):
                center_x = round(roi_params['center_x'])
                center_y = round(roi_params['center_y'])
                width = round(roi_params['width'])
                height = round(roi_params['height'])
                
                y_coords, x_coords = np.ogrid[:image_shape[0], :image_shape[1]]
                mask = (((x_coords - center_x) / (width/2))**2 + 
                       ((y_coords - center_y) / (height/2))**2) <= 1
                       
        elif self.roi_shape == 'polygon':
            if 'points' in roi_params:
                points = roi_params['points']
                if len(points) >= 3:
                    from matplotlib.path import Path
                    path = Path(points)
                    y_coords, x_coords = np.mgrid[0:image_shape[0], 0:image_shape[1]]
                    coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
                    mask_flat = path.contains_points(coords)
                    mask = mask_flat.reshape(image_shape)
                   
        return mask
        
    def create_roi_mask(self, image_shape):
        """Create a boolean mask for the current ROI (compatibility method)."""
        return self.create_roi_mask_with_params(image_shape, self.roi_parameters)
        
    def update_intensity_display(self):
        """Update the intensity values display."""
        if self.current_frame < len(self.intensity_data['mean']):
            mean_val = self.intensity_data['mean'][self.current_frame]
            max_val = self.intensity_data['max'][self.current_frame]
            sum_val = self.intensity_data['sum'][self.current_frame]
            
            display_text = f"Mean: {mean_val:.2f}\nMax: {max_val:.2f}\nSum: {sum_val:.2f}"
            self.intensity_display.setText(display_text)
            
    def update_plot(self):
        """Update the time series plot."""
        if not self.intensity_data['mean']:
            return
            
        self.plot_ax.clear()
        
        # Get selected plot type
        selected_id = self.plot_button_group.checkedId()
        plot_types = ['mean', 'max', 'sum']
        plot_labels = ['Average', 'Maximum', 'Sum']
        
        if selected_id >= 0:
            plot_type = plot_types[selected_id]
            plot_label = plot_labels[selected_id]
            
            frames = range(1, self.total_frames + 1)
            self.plot_ax.plot(frames, self.intensity_data[plot_type], 
                            '-', color='black', linewidth=1.5, label=plot_label)
            
            # Mark current frame
            if self.current_frame < len(self.intensity_data[plot_type]):
                self.plot_ax.plot(self.current_frame + 1, 
                                self.intensity_data[plot_type][self.current_frame],
                                'ko', markersize=8)
        
        self.plot_ax.set_xlabel('Frame')
        self.plot_ax.set_ylabel('Intensity')
        self.plot_ax.set_title(f"Frame {self.current_frame + 1}/{self.total_frames}")
        self.plot_ax.grid(True, color='lightgray', linewidth=0.5)
        self.plot_figure.tight_layout(pad=0.5)
        self.plot_canvas.draw()
        
    def prev_frame(self):
        """Go to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame_display()
            
    def next_frame(self):
        """Go to next frame."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_frame_display()
            
    def on_frame_changed(self, value):
        """Handle frame slider change."""
        # Prevent recursive calls
        if hasattr(self, '_updating_slider') and self._updating_slider:
            return
            
        self.current_frame = value
        self.update_frame_display()
        
    def update_frame_display(self):
        """Update the frame display."""
        # Save current ROI parameters for previous frame (frame-specific mode always enabled)
        if (hasattr(self, 'previous_frame') and 
            self.roi_parameters and 
            self.previous_frame is not None):
            self.frame_roi_parameters[self.previous_frame] = self.roi_parameters.copy()
            print(f"Saved ROI parameters for frame {self.previous_frame}: {self.roi_parameters}")
        
        # Store current frame as previous frame for next update
        self.previous_frame = self.current_frame
        
        # Update slider without triggering on_frame_changed
        self._updating_slider = True
        self.frame_slider.setValue(self.current_frame)
        self._updating_slider = False
        
        self.frame_info_label.setText(f"Frame {self.current_frame + 1}/{self.total_frames}")

        if self.frame_info_df is not None and hasattr(self, 'file_info_display'):
            self.file_info_display.setText(self._build_file_info_text_for_frame(self.current_frame))
        
        # Load ROI parameters for current frame (frame-specific mode always enabled)
        if self.current_frame in self.frame_roi_parameters:
            # Check if we should inherit the latest ROI position instead of using saved position
            if (hasattr(self, 'previous_frame') and 
                self.previous_frame is not None and 
                self.previous_frame in self.frame_roi_parameters and
                self.roi_parameters):
                # If we have a current ROI position that might be more recent, use it
                self.frame_roi_parameters[self.current_frame] = self.roi_parameters.copy()
                print(f"Updated ROI parameters for frame {self.current_frame} with latest position: {self.roi_parameters}")
            else:
                # Use saved ROI parameters for this frame
                self.roi_parameters = self.frame_roi_parameters[self.current_frame].copy()
                print(f"Loaded saved ROI parameters for frame {self.current_frame}: {self.roi_parameters}")
            
            # Recreate ROI object with loaded/updated parameters
            self.recreate_roi_from_parameters()
        else:
            # If no saved ROI parameters for this frame, inherit from current position
            if self.roi_parameters:
                # Save current ROI parameters to this frame (inherit from previous position)
                self.frame_roi_parameters[self.current_frame] = self.roi_parameters.copy()
                print(f"Inherited ROI parameters for frame {self.current_frame}: {self.roi_parameters}")
                # Recalculate intensity immediately for inherited ROI
                if not self.is_defining_roi:
                    self.update_current_frame_intensity()
            else:
                print(f"No ROI parameters available for frame {self.current_frame}")
        
        if not self.is_defining_roi:
            self.display_time_series_frame(self.current_frame)
            self.update_roi_display_params()
            # Always recalculate intensity for current frame (important for frame switching)
            self.update_current_frame_intensity()
            self.update_intensity_display()
            self.update_plot()
            
    def recreate_roi_from_parameters(self):
        """Recreate the ROI object from saved parameters."""
        if not self.roi_parameters:
            self.current_roi = None
            return
            
        if self.roi_shape == 'rectangle':
            x = self.roi_parameters['x']
            y = self.roi_parameters['y']
            width = self.roi_parameters['width']
            height = self.roi_parameters['height']
            self.current_roi = Rectangle((x, y), width, height)
        elif self.roi_shape == 'ellipse':
            center_x = self.roi_parameters['center_x']
            center_y = self.roi_parameters['center_y']
            width = self.roi_parameters['width']
            height = self.roi_parameters['height']
            self.current_roi = Ellipse((center_x, center_y), width, height)
        elif self.roi_shape == 'polygon':
            points = self.roi_parameters['points']
            self.current_roi = Polygon(points, closed=True)
            
        print(f"Recreated {self.roi_shape} ROI for frame {self.current_frame}")
        
    def back_to_max_proj(self):
        """Return to max projection ROI definition mode."""
        self.is_defining_roi = True
        self.current_roi = None
        self.roi_parameters = {}
        self.polygon_points = []
        
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.back_to_maxproj_button.setEnabled(False)
        self.complete_analysis_button.setEnabled(False)
        
        self.params_display.setText("No ROI defined")
        self.intensity_display.setText("Mean: -\nMax: -\nSum: -")
        
        self.display_max_proj()
        self.plot_ax.clear()
        self.plot_canvas.draw()
        
    def redraw_image_with_temp_roi(self, current_x, current_y):
        """Redraw the image with temporary ROI during creation."""
        if self.is_defining_roi:
            self.display_max_proj()
        else:
            self.display_time_series_frame(self.current_frame)
            
        # Display temporary ROI during creation
        if self.is_drawing and self.roi_start_pos:
            start_x, start_y = self.roi_start_pos
            
            if self.roi_shape == 'rectangle':
                # Draw temporary rectangle
                width = abs(current_x - start_x)
                height = abs(current_y - start_y)
                x = min(start_x, current_x)
                y = min(start_y, current_y)
                temp_rect = Rectangle((x, y), width, height, 
                                    linewidth=1, edgecolor='yellow', 
                                    facecolor='none', linestyle='--')
                self.image_ax.add_patch(temp_rect)
                
            elif self.roi_shape == 'ellipse':
                # Draw temporary ellipse
                center_x = (start_x + current_x) / 2
                center_y = (start_y + current_y) / 2
                width = abs(current_x - start_x)
                height = abs(current_y - start_y)
                temp_ellipse = Ellipse((center_x, center_y), width, height,
                                     linewidth=1, edgecolor='yellow',
                                     facecolor='none', linestyle='--')
                self.image_ax.add_patch(temp_ellipse)
            
        # Display polygon points if drawing polygon
        if self.roi_shape == 'polygon' and self.polygon_points:
            for i, point in enumerate(self.polygon_points):
                self.image_ax.plot(point[0], point[1], 'yo', markersize=3)
                if i > 0:
                    prev_point = self.polygon_points[i-1]
                    self.image_ax.plot([prev_point[0], point[0]], [prev_point[1], point[1]], 'y-')
            
            # Draw line to current mouse position
            if len(self.polygon_points) > 0:
                last_point = self.polygon_points[-1]
                self.image_ax.plot([last_point[0], current_x], [last_point[1], current_y], 'y--')
                
        self.image_canvas.draw()
        
    def is_point_in_roi(self, x, y):
        """Check if a point is inside the current ROI."""
        if self.current_roi is None:
            return False
            
        if self.roi_shape == 'rectangle':
            rx, ry = self.roi_parameters['x'], self.roi_parameters['y']
            rw, rh = self.roi_parameters['width'], self.roi_parameters['height']
            return rx <= x <= rx + rw and ry <= y <= ry + rh
        elif self.roi_shape == 'ellipse':
            cx, cy = self.roi_parameters['center_x'], self.roi_parameters['center_y']
            w, h = self.roi_parameters['width'], self.roi_parameters['height']
            dx, dy = (x - cx) / (w/2), (y - cy) / (h/2)
            return dx*dx + dy*dy <= 1
        elif self.roi_shape == 'polygon':
            points = self.roi_parameters['points']
            if len(points) < 3:
                return False
            polygon_path = Path(points)
            return polygon_path.contains_point((x, y))
        return False
        
    def start_roi_move(self, x, y):
        """Start moving an existing ROI."""
        self.is_moving_roi = True
        self.drag_start_pos = (x, y)
        
    def finish_roi_move(self):
        """Finish moving the ROI."""
        self.is_moving_roi = False
        self.drag_start_pos = None
        
    def update_roi_position(self, x, y):
        """Update ROI position during drag."""
        if self.drag_start_pos is None or self.current_roi is None:
            return
            
        # Calculate movement delta
        dx = x - self.drag_start_pos[0]
        dy = y - self.drag_start_pos[1]
        
        # Apply snapping if enabled
        if self.snap_to_integer:
            dx = round(dx)
            dy = round(dy)
        
        # Get image boundaries
        image_height, image_width = self.max_proj_image.shape
        
        # Update ROI parameters based on shape type
        if self.roi_shape == 'rectangle':
            # Calculate new position
            new_x = self.roi_parameters['x'] + dx
            new_y = self.roi_parameters['y'] + dy
            width = self.roi_parameters['width']
            height = self.roi_parameters['height']
            
            # Constrain to image boundaries
            new_x = max(0, min(new_x, image_width - width))
            new_y = max(0, min(new_y, image_height - height))
            
            # Update rectangle position
            self.roi_parameters['x'] = new_x
            self.roi_parameters['y'] = new_y
            
            # Update the matplotlib rectangle
            self.current_roi.set_x(new_x)
            self.current_roi.set_y(new_y)
            
        elif self.roi_shape == 'ellipse':
            # Calculate new center position
            new_center_x = self.roi_parameters['center_x'] + dx
            new_center_y = self.roi_parameters['center_y'] + dy
            width = self.roi_parameters['width']
            height = self.roi_parameters['height']
            half_width = width / 2.0
            half_height = height / 2.0
            
            # Constrain to image boundaries
            new_center_x = max(half_width, min(new_center_x, image_width - half_width))
            new_center_y = max(half_height, min(new_center_y, image_height - half_height))
            
            # Update ellipse center position
            self.roi_parameters['center_x'] = new_center_x
            self.roi_parameters['center_y'] = new_center_y
            
            # Update the matplotlib ellipse
            self.current_roi.center = (new_center_x, new_center_y)
            
        elif self.roi_shape == 'polygon':
            # Calculate new polygon position while preserving shape
            current_points = self.roi_parameters['points']
            
            # Calculate proposed new points
            proposed_points = []
            for point in current_points:
                new_x = point[0] + dx
                new_y = point[1] + dy
                proposed_points.append((new_x, new_y))
            
            # Check if all proposed points are within image boundaries
            all_points_valid = True
            for point in proposed_points:
                if point[0] < 0 or point[0] >= image_width or point[1] < 0 or point[1] >= image_height:
                    all_points_valid = False
                    break
            
            # Only update if all points remain within boundaries
            if all_points_valid:
                self.roi_parameters['points'] = proposed_points
                # Update the matplotlib polygon
                self.current_roi.set_xy(proposed_points)
            else:
                # Calculate constrained movement
                # Find the maximum allowed movement in each direction
                min_x = min(point[0] for point in current_points)
                max_x = max(point[0] for point in current_points)
                min_y = min(point[1] for point in current_points)
                max_y = max(point[1] for point in current_points)
                
                # Calculate maximum allowed movement
                max_dx_positive = image_width - 1 - max_x
                max_dx_negative = -min_x
                max_dy_positive = image_height - 1 - max_y
                max_dy_negative = -min_y
                
                # Constrain the movement
                constrained_dx = max(max_dx_negative, min(dx, max_dx_positive))
                constrained_dy = max(max_dy_negative, min(dy, max_dy_positive))
                
                # Apply constrained movement
                if abs(constrained_dx) > 0.1 or abs(constrained_dy) > 0.1:  # Only move if movement is significant
                    constrained_points = []
                    for point in current_points:
                        new_x = point[0] + constrained_dx
                        new_y = point[1] + constrained_dy
                        constrained_points.append((new_x, new_y))
                    
                    self.roi_parameters['points'] = constrained_points
                    # Update the matplotlib polygon
                    self.current_roi.set_xy(constrained_points)
        
        # Update drag start position for next iteration
        self.drag_start_pos = (x, y)
        
        # Redraw the image with updated ROI
        if self.is_defining_roi:
            self.display_max_proj()
        else:
            self.display_time_series_frame(self.current_frame)
            
        # Update parameters display
        self.update_roi_display_params()
        
        # Update intensity for current frame only (frame-specific mode always enabled)
        if not self.is_defining_roi:
            # Update intensity for current frame only
            self.update_current_frame_intensity()
            self.update_intensity_display()
            self.update_plot()
    
    def update_roi_display_params(self):
        """Update the ROI parameters display text."""
        param_text = f"Shape: {self.roi_shape}\n"
        
        if self.roi_parameters:
            for key, value in self.roi_parameters.items():
                if key == 'points':
                    param_text += f"points: {len(value)} vertices\n"
                else:
                    if self.snap_to_integer:
                        if isinstance(value, (int, float)):
                            param_text += f"{key}: {int(value)}\n"
                        else:
                            param_text += f"{key}: {value}\n"
                    else:
                        if isinstance(value, (int, float)):
                            param_text += f"{key}: {value:.1f}\n"
                        else:
                            param_text += f"{key}: {value}\n"
        
        self.params_display.setText(param_text)

    # Helper methods...
    def round_to_int(self, x, y):
        """Round coordinates to integer values for pixel-perfect ROI positioning."""
        if self.snap_to_integer:
            return (round(x), round(y))
        else:
            return (x, y)

    def on_snap_changed(self, state):
        """Handle changes to the snap to pixels checkbox."""
        self.snap_to_integer = state == Qt.Checked
        print(f"Coordinate snapping {'enabled' if self.snap_to_integer else 'disabled'}")

    def get_phase_for_frame(self, frame_idx):
        """Get phase information for the current frame from combined_df."""
        try:
            # Find row corresponding to current frame
            if 'phase' in self.combined_df.columns and 'nth_omit_induction' in self.combined_df.columns:
                # Get valid rows and sort by nth_omit_induction
                valid_rows = self.combined_df[self.combined_df['nth_omit_induction'] >= 0].copy()
                if len(valid_rows) > 0:
                    sorted_rows = valid_rows.sort_values('nth_omit_induction').reset_index(drop=True)
                    if frame_idx < len(sorted_rows):
                        phase_value = sorted_rows.iloc[frame_idx]['phase']
                        return str(phase_value) if pd.notna(phase_value) else ""
            return ""
        except Exception as e:
            print(f"Warning: Could not get phase for frame {frame_idx}: {e}")
            return ""

    def complete_analysis(self):
        """Complete the analysis and close the window."""
        if self.roi_parameters:
            # Save current frame ROI parameters (frame-specific mode always enabled)
            self.frame_roi_parameters[self.current_frame] = self.roi_parameters.copy()
            print(f"Saved final ROI parameters for frame {self.current_frame}")
            
            # Set analysis completed flag
            self.analysis_completed = True
            
            # Final intensity calculation to ensure all data is up to date
            self.calculate_all_intensities()
            
            print(f"Analysis completed for {self.header}")
            print(f"Total frames analyzed: {self.total_frames}")
            print(f"Frame-specific ROI parameters saved for frames: {list(self.frame_roi_parameters.keys())}")
            print(f"Intensity data length: mean={len(self.intensity_data['mean'])}, max={len(self.intensity_data['max'])}, sum={len(self.intensity_data['sum'])}")
            
            self.close()
        else:
            print("Cannot complete analysis - no ROI defined")

    def update_current_frame_intensity(self):
        """Update intensity for current frame only (for frame-specific ROI mode)."""
        # Get current frame data
        frame_data = self.after_align_tiff_data[self.current_frame]
        
        # Handle different dimensions for frame data
        if len(frame_data.shape) == 3:
            frame_2d = frame_data.max(axis=0)
        elif len(frame_data.shape) == 2:
            frame_2d = frame_data
        else:
            frame_2d = frame_data
            while len(frame_2d.shape) > 2:
                frame_2d = frame_2d.max(axis=0)
        
        # Calculate intensity with current ROI
        roi_mask = self.create_roi_mask_with_params(frame_2d.shape, self.roi_parameters)
        
        if np.any(roi_mask):
            roi_values = frame_2d[roi_mask]
            mean_val = np.mean(roi_values)
            max_val = np.max(roi_values)
            sum_val = np.sum(roi_values)
        else:
            mean_val = max_val = sum_val = 0
        
        # Update intensity data for current frame
        if self.current_frame < len(self.intensity_data['mean']):
            self.intensity_data['mean'][self.current_frame] = mean_val
            self.intensity_data['max'][self.current_frame] = max_val
            self.intensity_data['sum'][self.current_frame] = sum_val
        
        print(f"[INTENSITY UPDATE] Frame {self.current_frame}: mean={mean_val:.2f}, max={max_val:.2f}, sum={sum_val:.2f}, ROI area={np.sum(roi_mask)} pixels")

def main():
    """Main function to run the GUI application."""
    app = QApplication(sys.argv)
    
    print("ROI Analysis GUI module created successfully!")
    print("To use this module, import it and create an instance with your data:")
    print("from roi_analysis_gui import ROIAnalysisGUI")
    print("app = QApplication(sys.argv)")
    print("window = ROIAnalysisGUI(combined_df, after_align_tiff_data, max_proj_image, header='Spine')")
    print("window.show()")
    print("app.exec_()")

if __name__ == "__main__":
    main() 
