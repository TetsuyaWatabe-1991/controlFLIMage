# -*- coding: utf-8 -*-
"""Interactive bead FWHM tool for .flim Z and Y max projections.

Open a .flim file, view Channel 1 Z-projection (XY) and Y-projection (XZ),
draw multiple axis-aligned lines, and save intensity profiles plus FWHM.

Launch (use the same Python env as other controlFLIMage GUIs):
    C:\\Users\\yasudalab\\Documents\\Tetsuya_GIT\\deepd3\\Scripts\\python.exe bead_fwhm_gui.py

Headless self-test (no GUI):
    set BEAD_FWHM_SELF_TEST=1
    set BEAD_FWHM_DATA_DIR=C:\\path\\to\\out
    python bead_fwhm_gui.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_CALIB_DIR = Path(__file__).resolve().parent
if str(_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIB_DIR))

from bead_fwhm_core import (  # noqa: E402
    DEFAULT_SAVE_DIR,
    SELF_TEST_DATA_DIR_ENV,
    SELF_TEST_ENV,
    run_self_test,
)


def _run_env_self_test() -> int:
    out_dir = os.environ.get(SELF_TEST_DATA_DIR_ENV, str(DEFAULT_SAVE_DIR / "self_test_output"))
    result = run_self_test(out_dir)
    print("SELF-TEST PASS")
    print(f"  output: {result['saved']['out_dir']}")
    print(f"  FWHM CSV rows: {result['n_fwhm_rows']}")
    print(f"  profile CSV rows: {result['n_profile_rows']}")
    for label, err in result["errors"].items():
        print(
            f"  {label}: measured={result['measured_um'][label]:.4f} um  "
            f"expected={result['expected_um'][label]:.4f} um  rel_err={err:.4f}"
        )
    return 0


if os.environ.get(SELF_TEST_ENV, "").strip() == "1" and __name__ == "__main__":
    raise SystemExit(_run_env_self_test())

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from bead_fwhm_core import (  # noqa: E402
    MIN_LINE_PIXELS,
    DrawnLine,
    LineMeasurement,
    LoadedVolume,
    PixelSizeUm,
    compute_projections,
    default_contrast,
    load_flim_channel_zyx,
    make_drawn_line,
    line_length_px,
    make_output_dir,
    measure_line,
    next_line_style,
    save_bead_fwhm_outputs,
    snap_to_axis,
)

TABLE_HEADERS = [
    "Line",
    "Projection",
    "Axis",
    "FWHM (um)",
    "Gauss FWHM (um)",
    "Peak",
    "Background",
]


class ContrastControls(QGroupBox):
    """Min/max intensity sliders and spin boxes for one projection."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self._updating = False
        self.min_spin = QDoubleSpinBox()
        self.max_spin = QDoubleSpinBox()
        for spin in (self.min_spin, self.max_spin):
            spin.setDecimals(1)
            spin.setRange(-1e9, 1e9)
            spin.setSingleStep(1.0)
        self.min_slider = QSlider(Qt.Horizontal)
        self.max_slider = QSlider(Qt.Horizontal)
        for slider in (self.min_slider, self.max_slider):
            slider.setRange(0, 1000)
        layout = QFormLayout(self)
        layout.addRow("Min", self.min_spin)
        layout.addRow("", self.min_slider)
        layout.addRow("Max", self.max_spin)
        layout.addRow("", self.max_slider)
        self.min_spin.valueChanged.connect(self._spin_changed)
        self.max_spin.valueChanged.connect(self._spin_changed)
        self.min_slider.valueChanged.connect(self._slider_changed)
        self.max_slider.valueChanged.connect(self._slider_changed)
        self._range = (0.0, 1.0)
        self._callback = None

    def set_callback(self, callback) -> None:
        self._callback = callback

    def set_data_range(self, vmin: float, vmax: float) -> None:
        if vmax <= vmin:
            vmax = vmin + 1.0
        lo_limit = min(0.0, float(vmin))
        hi_limit = float(vmax)
        self._range = (lo_limit, hi_limit)
        self._updating = True
        self.min_spin.setRange(lo_limit, hi_limit)
        self.max_spin.setRange(lo_limit, hi_limit)
        self.min_spin.setValue(float(vmin))
        self.max_spin.setValue(hi_limit)
        self.min_slider.setValue(self._value_to_fraction(float(vmin)))
        self.max_slider.setValue(1000)
        self._updating = False

    def values(self) -> tuple[float, float]:
        lo = float(self.min_spin.value())
        hi = float(self.max_spin.value())
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _fraction_to_value(self, slider_val: int) -> float:
        lo, hi = self._range
        return lo + (hi - lo) * (slider_val / 1000.0)

    def _value_to_fraction(self, value: float) -> int:
        lo, hi = self._range
        if hi <= lo:
            return 0
        frac = (value - lo) / (hi - lo)
        return int(np.clip(round(frac * 1000.0), 0, 1000))

    def _spin_changed(self) -> None:
        if self._updating:
            return
        self._updating = True
        lo, hi = self.values()
        if self.max_spin.value() < lo + 1.0:
            self.max_spin.setValue(lo + 1.0)
            hi = lo + 1.0
        self.min_slider.setValue(self._value_to_fraction(lo))
        self.max_slider.setValue(self._value_to_fraction(hi))
        self._updating = False
        if self._callback:
            self._callback()

    def _slider_changed(self) -> None:
        if self._updating:
            return
        self._updating = True
        lo = self._fraction_to_value(self.min_slider.value())
        hi = self._fraction_to_value(self.max_slider.value())
        if hi <= lo:
            hi = lo + (self._range[1] - self._range[0]) / 1000.0
            self.max_slider.setValue(self._value_to_fraction(hi))
        self.min_spin.setValue(lo)
        self.max_spin.setValue(hi)
        self._updating = False
        if self._callback:
            self._callback()


class BeadFwhmWindow(QMainWindow):
    """Main window: two projections, line drawing, FWHM table."""

    def __init__(self, save_dir: Path | None = None):
        super().__init__()
        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.loaded: LoadedVolume | None = None
        self.z_proj = np.zeros((8, 8), dtype=np.float64)
        self.y_proj = np.zeros((8, 8), dtype=np.float64)
        self.pixel_size = PixelSizeUm(1.0, 1.0, 1.0)
        self.measurements: list[LineMeasurement] = []
        self._drag = None
        self._preview_line = None
        self.setWindowTitle("Bead FWHM (Ch1 Z/Y projections)")
        self.resize(1480, 920)
        self._build_ui()
        self._connect_canvas_events()
        self._redraw_all()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        self.open_btn = QPushButton("Open .flim")
        self.save_btn = QPushButton("Save plots")
        self.undo_btn = QPushButton("Undo last line")
        self.clear_btn = QPushButton("Clear lines")
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Ch1", "Ch2"])
        self.channel_combo.setCurrentIndex(0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 11)
        self.width_spin.setSingleStep(2)
        self.width_spin.setValue(1)
        self.width_spin.setToolTip("Odd number of pixels averaged perpendicular to the line")
        toolbar.addWidget(self.open_btn)
        toolbar.addWidget(QLabel("Channel"))
        toolbar.addWidget(self.channel_combo)
        toolbar.addWidget(self.save_btn)
        toolbar.addWidget(self.undo_btn)
        toolbar.addWidget(self.clear_btn)
        toolbar.addWidget(QLabel("Profile width (px)"))
        toolbar.addWidget(self.width_spin)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        axis_row = QHBoxLayout()
        axis_box = QGroupBox("Line direction (axis-parallel only)")
        axis_layout = QHBoxLayout(axis_box)
        self.axis_group = QButtonGroup(self)
        self.radio_auto = QRadioButton("Auto snap")
        self.radio_h = QRadioButton("Horizontal only")
        self.radio_v = QRadioButton("Vertical only")
        self.radio_auto.setChecked(True)
        for i, radio in enumerate((self.radio_auto, self.radio_h, self.radio_v)):
            self.axis_group.addButton(radio, i)
            axis_layout.addWidget(radio)
        axis_row.addWidget(axis_box)
        self.info_label = QLabel("Open a .flim file, then click-drag on a projection.")
        self.info_label.setWordWrap(True)
        axis_row.addWidget(self.info_label, 1)
        root.addLayout(axis_row)

        splitter = QSplitter(Qt.Vertical)
        top_split = QSplitter(Qt.Horizontal)

        z_widget = QWidget()
        z_layout = QVBoxLayout(z_widget)
        z_layout.setContentsMargins(0, 0, 0, 0)
        self.z_fig = Figure(figsize=(5.5, 5.0), dpi=100)
        self.z_ax = self.z_fig.add_subplot(111)
        self.z_canvas = FigureCanvas(self.z_fig)
        self.z_contrast = ContrastControls("Z projection contrast")
        z_layout.addWidget(QLabel("Z max projection (XY)"))
        z_layout.addWidget(self.z_canvas, 1)
        z_layout.addWidget(self.z_contrast)

        y_widget = QWidget()
        y_layout = QVBoxLayout(y_widget)
        y_layout.setContentsMargins(0, 0, 0, 0)
        self.y_fig = Figure(figsize=(5.5, 5.0), dpi=100)
        self.y_ax = self.y_fig.add_subplot(111)
        self.y_canvas = FigureCanvas(self.y_fig)
        self.y_contrast = ContrastControls("Y projection contrast")
        y_layout.addWidget(QLabel("Y max projection (XZ)"))
        y_layout.addWidget(self.y_canvas, 1)
        y_layout.addWidget(self.y_contrast)

        top_split.addWidget(z_widget)
        top_split.addWidget(y_widget)
        top_split.setSizes([700, 700])

        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.p_fig = Figure(figsize=(8.0, 3.2), dpi=100)
        self.p_ax = self.p_fig.add_subplot(111)
        self.p_canvas = FigureCanvas(self.p_fig)
        self.table = QTableWidget(0, len(TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.table.horizontalHeader().setStretchLastSection(True)
        bottom_layout.addWidget(self.p_canvas, 3)
        bottom_layout.addWidget(self.table, 2)

        splitter.addWidget(top_split)
        splitter.addWidget(bottom)
        splitter.setSizes([620, 280])
        root.addWidget(splitter, 1)

        self.open_btn.clicked.connect(self.open_flim)
        self.save_btn.clicked.connect(self.save_plots)
        self.undo_btn.clicked.connect(self.undo_last_line)
        self.clear_btn.clicked.connect(self.clear_lines)
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        self.z_contrast.set_callback(self._redraw_images)
        self.y_contrast.set_callback(self._redraw_images)
        QShortcut(QKeySequence("Ctrl+O"), self, self.open_flim)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_plots)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_last_line)

    def _connect_canvas_events(self) -> None:
        self.z_canvas.mpl_connect("button_press_event", lambda e: self._on_press(e, "z_proj"))
        self.z_canvas.mpl_connect("motion_notify_event", lambda e: self._on_motion(e, "z_proj"))
        self.z_canvas.mpl_connect("button_release_event", lambda e: self._on_release(e, "z_proj"))
        self.y_canvas.mpl_connect("button_press_event", lambda e: self._on_press(e, "y_proj"))
        self.y_canvas.mpl_connect("motion_notify_event", lambda e: self._on_motion(e, "y_proj"))
        self.y_canvas.mpl_connect("button_release_event", lambda e: self._on_release(e, "y_proj"))

    def _axis_mode(self) -> str:
        if self.radio_h.isChecked():
            return "horizontal"
        if self.radio_v.isChecked():
            return "vertical"
        return "auto"

    def _channel_1based(self) -> int:
        return self.channel_combo.currentIndex() + 1

    def _image_for(self, projection: str) -> np.ndarray:
        return self.z_proj if projection == "z_proj" else self.y_proj

    def _axes_canvas(self, projection: str):
        if projection == "z_proj":
            return self.z_ax, self.z_canvas
        return self.y_ax, self.y_canvas

    def open_flim(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open FLIM file",
            "",
            "FLIM files (*.flim);;All files (*.*)",
        )
        if not path:
            return
        self.load_path(path)

    def load_path(self, path: str) -> None:
        try:
            loaded = load_flim_channel_zyx(path, channel_1based=self._channel_1based())
        except Exception as exc:
            QMessageBox.critical(self, "Failed to read FLIM", str(exc))
            return
        self._apply_loaded(loaded)

    def _apply_loaded(self, loaded: LoadedVolume) -> None:
        self.loaded = loaded
        self.pixel_size = loaded.pixel_size
        self.z_proj, self.y_proj = compute_projections(loaded.volume_zyx)
        self.measurements = []
        z_lo, z_hi = default_contrast(self.z_proj)
        y_lo, y_hi = default_contrast(self.y_proj)
        self.z_contrast.set_data_range(z_lo, z_hi)
        self.y_contrast.set_data_range(y_lo, y_hi)
        nz, ny, nx = loaded.volume_zyx.shape
        name = Path(loaded.path).name
        self.setWindowTitle(f"Bead FWHM  -  {name}")
        self.info_label.setText(
            f"{name}  Ch{loaded.channel_1based}  ZYX={nz}x{ny}x{nx}  "
            f"xy={loaded.pixel_size.x_um:.4g} um/px  "
            f"z={loaded.pixel_size.z_um:.4g} um/px  zoom={loaded.zoom:g}  "
            f"{'FastZ' if loaded.fast_z else 'Z-stack'}  |  "
            "Click-drag to draw an axis-parallel line (repeat as needed)."
        )
        self._redraw_all()

    def _channel_changed(self) -> None:
        if self.loaded is None:
            return
        self.load_path(self.loaded.path)

    def _on_press(self, event, projection: str) -> None:
        if event.button == 3:
            self.undo_last_line()
            return
        if event.button != 1 or event.inaxes is None or event.xdata is None:
            return
        if self.loaded is None:
            QMessageBox.information(self, "No image", "Open a .flim file first.")
            return
        self._drag = {
            "projection": projection,
            "x0": float(event.xdata),
            "y0": float(event.ydata),
        }

    def _on_motion(self, event, projection: str) -> None:
        if self._drag is None or self._drag["projection"] != projection:
            return
        if event.inaxes is None or event.xdata is None:
            return
        x0, y0, x1, y1, _ = snap_to_axis(
            self._drag["x0"],
            self._drag["y0"],
            float(event.xdata),
            float(event.ydata),
            mode=self._axis_mode(),
        )
        self._preview_line = (projection, x0, y0, x1, y1)
        self._redraw_images()

    def _on_release(self, event, projection: str) -> None:
        if self._drag is None or self._drag["projection"] != projection:
            return
        if event.xdata is None or event.ydata is None:
            self._drag = None
            self._preview_line = None
            self._redraw_images()
            return
        x0, y0, x1, y1, _ = snap_to_axis(
            self._drag["x0"],
            self._drag["y0"],
            float(event.xdata),
            float(event.ydata),
            mode=self._axis_mode(),
        )
        self._drag = None
        self._preview_line = None
        self.add_line_from_pixels(
            projection,
            x0,
            y0,
            x1,
            y1,
            mode=self._axis_mode(),
            width_px=int(self.width_spin.value()),
        )

    def add_line_from_pixels(
        self,
        projection: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        mode: str | None = None,
        width_px: int | None = None,
    ) -> LineMeasurement | None:
        """Add one axis-aligned line and measure FWHM. Returns None if too short."""
        image = self._image_for(projection)
        label, color = next_line_style([m.line for m in self.measurements])
        line = make_drawn_line(
            projection=projection,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            image_shape=image.shape,
            label=label,
            color=color,
            mode=mode or self._axis_mode(),
            width_px=int(self.width_spin.value() if width_px is None else width_px),
        )
        if line_length_px(line) < MIN_LINE_PIXELS:
            self.info_label.setText("Line too short; drag farther along one axis.")
            self._redraw_images()
            return None
        meas = measure_line(image, line, self.pixel_size)
        self.measurements.append(meas)
        txt = (
            f"{line.label}: FWHM {meas.fwhm.measured_axis} = {meas.fwhm.fwhm_um:.3f} um"
            if meas.fwhm.ok
            else f"{line.label}: FWHM could not be computed (no half-max crossing)."
        )
        self.info_label.setText(txt)
        self._redraw_all()
        return meas

    def undo_last_line(self) -> None:
        if not self.measurements:
            return
        self.measurements.pop()
        self._relabel_lines()
        self._redraw_all()

    def clear_lines(self) -> None:
        self.measurements = []
        self._redraw_all()

    def _relabel_lines(self) -> None:
        relabeled: list[LineMeasurement] = []
        for i, meas in enumerate(self.measurements, start=1):
            new_line = DrawnLine(
                projection=meas.line.projection,
                x0_px=meas.line.x0_px,
                y0_px=meas.line.y0_px,
                x1_px=meas.line.x1_px,
                y1_px=meas.line.y1_px,
                orientation=meas.line.orientation,
                color=meas.line.color,
                label=f"L{i}",
                width_px=meas.line.width_px,
            )
            relabeled.append(LineMeasurement(line=new_line, fwhm=meas.fwhm))
        self.measurements = relabeled

    def _redraw_all(self) -> None:
        self._redraw_images()
        self._redraw_profiles()
        self._refresh_table()

    def _redraw_images(self) -> None:
        self._draw_one_image(
            self.z_ax,
            self.z_canvas,
            self.z_proj,
            "z_proj",
            self.z_contrast.values(),
            "Z max projection (XY)",
            self.pixel_size.y_um / self.pixel_size.x_um if self.pixel_size.x_um else 1.0,
            "X (px)",
            "Y (px)",
        )
        self._draw_one_image(
            self.y_ax,
            self.y_canvas,
            self.y_proj,
            "y_proj",
            self.y_contrast.values(),
            "Y max projection (XZ)",
            self.pixel_size.z_um / self.pixel_size.x_um if self.pixel_size.x_um else 1.0,
            "X (px)",
            "Z (px)",
        )

    def _draw_one_image(
        self,
        ax,
        canvas,
        image: np.ndarray,
        projection: str,
        contrast: tuple[float, float],
        title: str,
        aspect: float,
        xlabel: str,
        ylabel: str,
    ) -> None:
        ax.clear()
        vmin, vmax = contrast
        ax.imshow(
            image,
            cmap="gray",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            aspect=aspect if np.isfinite(aspect) and aspect > 0 else "auto",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for meas in self.measurements:
            if meas.line.projection != projection:
                continue
            line = meas.line
            ax.plot(
                [line.x0_px, line.x1_px],
                [line.y0_px, line.y1_px],
                color=line.color,
                lw=1.6,
            )
            txt = line.label
            if meas.fwhm.ok:
                txt += f" {meas.fwhm.fwhm_um:.2f} um"
            ax.text(
                0.5 * (line.x0_px + line.x1_px),
                0.5 * (line.y0_px + line.y1_px),
                txt,
                color=line.color,
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
        if self._preview_line and self._preview_line[0] == projection:
            _, x0, y0, x1, y1 = self._preview_line
            ax.plot([x0, x1], [y0, y1], color="yellow", lw=1.2, ls="--")
        canvas.draw_idle()

    def _redraw_profiles(self) -> None:
        self.p_ax.clear()
        if not self.measurements:
            self.p_ax.set_title("Intensity profiles")
            self.p_ax.set_xlabel("Position (um)")
            self.p_ax.set_ylabel("Intensity")
            self.p_canvas.draw_idle()
            return
        for meas in self.measurements:
            fwhm = meas.fwhm
            label = meas.line.label
            if fwhm.ok:
                label += f"  {fwhm.measured_axis} FWHM={fwhm.fwhm_um:.3f} um"
            self.p_ax.plot(
                fwhm.positions_um,
                fwhm.intensities,
                color=meas.line.color,
                lw=1.5,
                label=label,
            )
            if fwhm.ok:
                self.p_ax.axhline(
                    fwhm.half_max, color=meas.line.color, ls="--", lw=0.7, alpha=0.6
                )
                self.p_ax.axvline(fwhm.x_left_um, color=meas.line.color, ls=":", lw=0.7)
                self.p_ax.axvline(fwhm.x_right_um, color=meas.line.color, ls=":", lw=0.7)
        self.p_ax.set_title("Intensity profiles")
        self.p_ax.set_xlabel("Position (um)")
        self.p_ax.set_ylabel("Intensity")
        self.p_ax.legend(fontsize=8)
        self.p_ax.grid(True, alpha=0.3)
        self.p_fig.tight_layout()
        self.p_canvas.draw_idle()

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.measurements))
        for row, meas in enumerate(self.measurements):
            fwhm = meas.fwhm
            vals = [
                meas.line.label,
                meas.line.projection,
                fwhm.measured_axis,
                f"{fwhm.fwhm_um:.4f}" if fwhm.ok else "n/a",
                (
                    f"{fwhm.gaussian_fwhm_um:.4f}"
                    if np.isfinite(fwhm.gaussian_fwhm_um)
                    else "n/a"
                ),
                f"{fwhm.peak_intensity:.1f}",
                f"{fwhm.background:.1f}",
            ]
            for col, val in enumerate(vals):
                self.table.setItem(row, col, QTableWidgetItem(val))
        self.table.resizeColumnsToContents()

    def export_outputs(self, out_dir: Path | None = None) -> dict[str, Path]:
        """Write plots and CSV without showing a dialog."""
        if self.loaded is None:
            raise RuntimeError("Open a .flim file first.")
        if not self.measurements:
            raise RuntimeError("Draw at least one line before saving.")
        if out_dir is None:
            out_dir = make_output_dir(self.save_dir, self.loaded.path)
        else:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        saved = save_bead_fwhm_outputs(
            out_dir,
            self.z_proj,
            self.y_proj,
            self.pixel_size,
            self.measurements,
            self.z_contrast.values(),
            self.y_contrast.values(),
            source_name=Path(self.loaded.path).name,
            channel_1based=self.loaded.channel_1based,
        )
        self.info_label.setText(f"Saved {len(self.measurements)} line(s) to {out_dir}")
        return saved

    def save_plots(self) -> None:
        try:
            saved = self.export_outputs()
        except RuntimeError as exc:
            QMessageBox.information(self, "Cannot save", str(exc))
            return
        QMessageBox.information(
            self,
            "Saved",
            "Wrote:\n"
            + "\n".join(
                str(saved[k])
                for k in (
                    "z_projection",
                    "y_projection",
                    "intensity_profiles",
                    "combined",
                    "fwhm_csv",
                )
            ),
        )


def load_synthetic_demo() -> LoadedVolume:
    """Build a synthetic bead volume so the GUI can be opened without a file."""
    from bead_fwhm_core import make_synthetic_bead_volume

    volume = make_synthetic_bead_volume()
    return LoadedVolume(
        volume_zyx=volume,
        pixel_size=PixelSizeUm(z_um=0.50, y_um=0.10, x_um=0.10),
        channel_1based=1,
        path="synthetic_bead.flim",
        zoom=10.0,
        fast_z=False,
        n_pages=41,
    )


def launch_window(
    flim_path: str | None = None,
    channel_1based: int = 1,
    save_dir: Path | None = None,
    show: bool = True,
    argv: list[str] | None = None,
) -> int:
    """Create the bead FWHM window, optionally load a .flim file, and run Qt."""
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    win = BeadFwhmWindow(save_dir=save_dir)
    ch_idx = max(0, min(win.channel_combo.count() - 1, int(channel_1based) - 1))
    win.channel_combo.blockSignals(True)
    win.channel_combo.setCurrentIndex(ch_idx)
    win.channel_combo.blockSignals(False)
    if flim_path:
        win.load_path(str(flim_path))
        if win.loaded is None:
            return 1
    if show:
        win.show()
        return app.exec_()
    win.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if os.environ.get(SELF_TEST_ENV, "").strip() == "1":
        return _run_env_self_test()

    flim_path = None
    channel_1based = 1
    positional = [a for a in argv[1:] if not a.startswith("-")]
    if positional:
        flim_path = positional[0]
    if "--demo" in argv:
        app = QApplication.instance() or QApplication(argv)
        win = BeadFwhmWindow(save_dir=DEFAULT_SAVE_DIR)
        win._apply_loaded(load_synthetic_demo())
        win.show()
        return app.exec_()
    return launch_window(flim_path=flim_path, channel_1based=channel_1based, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
