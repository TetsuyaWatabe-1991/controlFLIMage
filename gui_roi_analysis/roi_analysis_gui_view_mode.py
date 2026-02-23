# -*- coding: utf-8 -*-
"""
ROI Analysis GUI with View / Define mode toggle.

Extends ROIAnalysisGUI without modifying the original:
- Define mode: same as base (draw/move ROI on max proj and per frame).
- View mode: frame slider works, existing ROI is displayed read-only; mouse cannot move/draw ROI.
  Switch to View mode is only allowed when at least one ROI is already defined (on max projection).

Usage (standalone):
    from roi_analysis_gui_view_mode import ROIAnalysisGUIWithViewMode
    window = ROIAnalysisGUIWithViewMode(filtered_df, after_align_tiff_data, max_proj_image,
                                        uncaging_info, file_info, header=header)

To try with existing TIFF-only flow (minimal change):
    In gui_integration_tiff_only.py, replace:
      from roi_analysis_gui import ROIAnalysisGUI
      ...
      window = ROIAnalysisGUI(...)
    with:
      from roi_analysis_gui_view_mode import ROIAnalysisGUIWithViewMode
      ...
      window = ROIAnalysisGUIWithViewMode(...)
"""

import os
import numpy as np
import tifffile
from matplotlib.patches import Polygon as MplPolygon
from skimage.measure import find_contours

from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from roi_analysis_gui import ROIAnalysisGUI


class ROIAnalysisGUIWithViewMode(ROIAnalysisGUI):
    """ROI Analysis GUI with View mode (read-only ROI display) and Define mode (edit ROI)."""

    def __init__(self, combined_df, after_align_tiff_data, max_proj_image,
                 uncaging_info=None, file_info=None, header="ROI"):
        self.view_mode = False  # False = Define, True = View (read-only)
        super().__init__(combined_df, after_align_tiff_data, max_proj_image,
                         uncaging_info, file_info, header=header)
        self._try_load_existing_roi_and_start_in_view_mode()

    def _has_roi_defined(self):
        """Return True if at least one ROI is defined (on max proj or any frame)."""
        return bool(self.roi_parameters) or len(self.frame_roi_parameters) > 0

    @staticmethod
    def _mask_to_roi_parameters(mask_2d):
        """Convert binary 2D mask to polygon roi_parameters (points as (x, y)). Returns None if empty."""
        if mask_2d is None or not np.any(mask_2d):
            return None
        mask_uint = np.asarray(mask_2d, dtype=np.uint8)
        contours = find_contours(mask_uint, 0.5)
        if not contours:
            return None
        # Take largest contour by area
        largest = max(contours, key=lambda c: len(c))
        # find_contours returns (row, col); GUI uses (x, y) = (col, row)
        points = [(float(col), float(row)) for row, col in largest]
        if len(points) > 200:
            step = max(1, len(points) // 200)
            points = points[::step]
        if len(points) < 3:
            return None
        return {"points": points}

    def _try_load_existing_roi_and_start_in_view_mode(self):
        """If ROI mask TIFF exists for this header, load it and start in View mode.
        For 3D stacks, loads ROI per frame into frame_roi_parameters so the time bar
        shows the correct ROI for each frame.
        """
        tiff_path = (self.file_info or {}).get("tiff_data_path")
        if not tiff_path or not os.path.exists(tiff_path):
            return
        tiff_dir = os.path.dirname(tiff_path)
        tiff_basename_no_ext = os.path.splitext(os.path.basename(tiff_path))[0]
        roi_mask_path = os.path.join(
            tiff_dir, f"{tiff_basename_no_ext}_{self.header}_roi_mask.tif"
        )
        if not os.path.exists(roi_mask_path):
            return
        try:
            roi_stack = tifffile.imread(roi_mask_path)
            if roi_stack.ndim == 3:
                n_frames = roi_stack.shape[0]
                for i in range(n_frames):
                    mask_2d = roi_stack[i] > 0
                    params = self._mask_to_roi_parameters(mask_2d)
                    if params:
                        self.frame_roi_parameters[i] = params.copy()
                if not self.frame_roi_parameters:
                    return
                # Use first frame's ROI for initial display and roi_parameters
                first_frame = min(self.frame_roi_parameters.keys())
                params = self.frame_roi_parameters[first_frame].copy()
                # Show first frame that has ROI (in case frame 0 has no ROI)
                if hasattr(self, "frame_slider") and self.frame_slider is not None:
                    self.current_frame = first_frame
                    self.frame_slider.setValue(first_frame)
            else:
                mask_2d = np.asarray(roi_stack) > 0
                params = self._mask_to_roi_parameters(mask_2d)
                if not params:
                    return
                self.frame_roi_parameters[0] = params.copy()
            self.roi_shape = "polygon"
            self.roi_parameters = params.copy()
            self.current_roi = MplPolygon(
                params["points"], closed=True,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            self.enable_time_series_mode()
            self.view_mode = True
            if hasattr(self, "view_mode_checkbox"):
                self.view_mode_checkbox.blockSignals(True)
                self.view_mode_checkbox.setChecked(True)
                self.view_mode_checkbox.setEnabled(True)
                self.view_mode_checkbox.blockSignals(False)
        except Exception as e:
            print(f"Could not load existing ROI for View mode: {e}")

    def create_roi_control_panel(self):
        """Add View/Define mode toggle at top of control panel; rest is same as base."""
        super().create_roi_control_panel()
        layout = self.roi_control_frame.layout()
        if layout is None:
            return

        mode_frame = QFrame()
        mode_frame.setFrameStyle(QFrame.StyledPanel)
        mode_layout = QVBoxLayout(mode_frame)
        mode_label = QLabel("Mode:")
        mode_label.setFont(QFont("Arial", 10, QFont.Bold))
        mode_layout.addWidget(mode_label)
        self.view_mode_checkbox = QCheckBox("View mode (read-only)")
        self.view_mode_checkbox.setChecked(False)
        self.view_mode_checkbox.setToolTip(
            "When checked: only view ROI on each frame; cannot move or draw ROI. "
            "Available only after at least one ROI is defined on max projection."
        )
        self.view_mode_checkbox.setEnabled(self._has_roi_defined())
        self.view_mode_checkbox.stateChanged.connect(self._on_view_mode_toggled)
        mode_layout.addWidget(self.view_mode_checkbox)
        layout.insertWidget(0, mode_frame)

    def _on_view_mode_toggled(self, state):
        """Handle View mode checkbox: allow only if ROI is defined."""
        if state == Qt.Checked:
            if not self._has_roi_defined():
                self.view_mode_checkbox.blockSignals(True)
                self.view_mode_checkbox.setChecked(False)
                self.view_mode_checkbox.blockSignals(False)
                QMessageBox.information(
                    self,
                    "View mode",
                    "Define at least one ROI on the max projection first, then you can switch to View mode.",
                )
                return
            self.view_mode = True
        else:
            self.view_mode = False

    def enable_time_series_mode(self):
        """Enable time series mode and enable View mode checkbox when ROI is defined."""
        super().enable_time_series_mode()
        if hasattr(self, "view_mode_checkbox"):
            self.view_mode_checkbox.setEnabled(self._has_roi_defined())

    def update_frame_display(self):
        """In View mode: do not save/overwrite ROI; only load per-frame ROI and display.
        In Define mode: delegate to base implementation.
        """
        if not self.view_mode:
            super().update_frame_display()
            return
        # View mode: update UI and load ROI for current frame only (never overwrite)
        self.previous_frame = self.current_frame
        self._updating_slider = True
        self.frame_slider.setValue(self.current_frame)
        self._updating_slider = False
        self.frame_info_label.setText(f"Frame {self.current_frame + 1}/{self.total_frames}")
        if self.frame_info_df is not None and hasattr(self, "file_info_display"):
            self.file_info_display.setText(self._build_file_info_text_for_frame(self.current_frame))
        if self.current_frame in self.frame_roi_parameters:
            self.roi_parameters = self.frame_roi_parameters[self.current_frame].copy()
            self.recreate_roi_from_parameters()
        if not self.is_defining_roi:
            self.display_time_series_frame(self.current_frame)
            self.update_roi_display_params()
            self.update_current_frame_intensity()
            self.update_intensity_display()
            self.update_plot()

    def on_mouse_press(self, event):
        """In View mode, ignore mouse press (no ROI edit)."""
        if self.view_mode:
            return
        super().on_mouse_press(event)

    def on_mouse_release(self, event):
        """In View mode, ignore mouse release (no ROI edit)."""
        if self.view_mode:
            return
        super().on_mouse_release(event)

    def on_mouse_move(self, event):
        """In View mode, ignore mouse move (no ROI drag)."""
        if self.view_mode:
            return
        super().on_mouse_move(event)
