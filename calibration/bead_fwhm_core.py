# -*- coding: utf-8 -*-
"""Bead PSF resolution: Z/Y projections, axis-aligned line profiles, FWHM."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_CALIB_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _CALIB_DIR.parent
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

FWHM_SIGMA_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
LINE_COLORS = (
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#ff7f00",
    "#984ea3",
    "#a65628",
    "#f781bf",
    "#17becf",
)
DEFAULT_SAVE_DIR = _CALIB_DIR
SELF_TEST_ENV = "BEAD_FWHM_SELF_TEST"
SELF_TEST_DATA_DIR_ENV = "BEAD_FWHM_DATA_DIR"
MIN_LINE_PIXELS = 3


@dataclass
class PixelSizeUm:
    """Voxel size along Z, Y, X (micrometers per pixel)."""

    z_um: float
    y_um: float
    x_um: float


@dataclass
class LoadedVolume:
    """Channel intensity volume in ZYX order."""

    volume_zyx: np.ndarray
    pixel_size: PixelSizeUm
    channel_1based: int
    path: str
    zoom: float = 1.0
    fast_z: bool = False
    n_pages: int = 1


@dataclass
class DrawnLine:
    """Axis-aligned line on a 2D projection, in pixel coordinates."""

    projection: str  # "z_proj" or "y_proj"
    x0_px: int
    y0_px: int
    x1_px: int
    y1_px: int
    orientation: str  # "horizontal" or "vertical"
    color: str
    label: str
    width_px: int = 1


@dataclass
class FwhmResult:
    """FWHM measurement for one line profile."""

    fwhm_um: float
    fwhm_px: float
    x_left_um: float
    x_right_um: float
    peak_intensity: float
    background: float
    half_max: float
    peak_position_um: float
    gaussian_fwhm_um: float
    gaussian_sigma_um: float
    measured_axis: str
    pixel_size_um: float
    positions_um: np.ndarray
    intensities: np.ndarray
    ok: bool


@dataclass
class LineMeasurement:
    """A drawn line plus its intensity profile and FWHM."""

    line: DrawnLine
    fwhm: FwhmResult


def snap_to_axis(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    mode: str = "auto",
) -> tuple[float, float, float, float, str]:
    """Snap a drag to a horizontal or vertical segment.

    Args:
        mode: ``auto``, ``horizontal``, or ``vertical``.

    Returns:
        (x0, y0, x1, y1, orientation).
    """
    mode = str(mode).lower()
    if mode == "horizontal":
        return x0, y0, x1, y0, "horizontal"
    if mode == "vertical":
        return x0, y0, x0, y1, "vertical"
    if abs(x1 - x0) >= abs(y1 - y0):
        return x0, y0, x1, y0, "horizontal"
    return x0, y0, x0, y1, "vertical"


def measured_axis_name(projection: str, orientation: str) -> str:
    """Physical axis sampled by an axis-aligned line."""
    if orientation == "horizontal":
        return "X"
    if projection == "z_proj":
        return "Y"
    return "Z"


def pixel_size_along_line(projection: str, orientation: str, pixel_size: PixelSizeUm) -> float:
    """Micrometers per pixel along a drawn line."""
    axis = measured_axis_name(projection, orientation)
    if axis == "X":
        return float(pixel_size.x_um)
    if axis == "Y":
        return float(pixel_size.y_um)
    return float(pixel_size.z_um)


def _pixel_sizes_from_iminfo(iminfo) -> PixelSizeUm:
    from FLIMageAlignment import get_xyz_pixel_um

    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
    fast_z = bool(getattr(iminfo, "FastZStack", False))
    acq = iminfo.State.Acq
    if fast_z:
        z_um = float(getattr(acq, "FastZ_umPerSlice", z_um) or z_um)
    return PixelSizeUm(z_um=float(z_um), y_um=float(y_um), x_um=float(x_um))


def load_flim_channel_zyx(flim_path: str | Path, channel_1based: int = 1) -> LoadedVolume:
    """Read a .flim file and return Ch intensity as a (Z, Y, X) array.

    Regular Z-stacks use TIFF pages as Z. FastZ volumes use FastZ slices as Z
    and average remaining pages (time) when more than one page is present.
    Intensity is the sum over the lifetime histogram (photon counts).
    """
    from FLIMageFileReader2 import FileReader

    flim_path = Path(flim_path)
    if not flim_path.is_file():
        raise FileNotFoundError(f"FLIM file not found: {flim_path}")

    iminfo = FileReader()
    iminfo.read_imageFile(str(flim_path), True)
    if not iminfo.image:
        raise ValueError(f"No image pages in {flim_path}")

    ch = int(channel_1based) - 1
    if ch < 0:
        raise ValueError("channel_1based must be >= 1")

    pages: list[np.ndarray] = []
    for page in iminfo.image:
        z_slices: list[np.ndarray] = []
        for fast_z_page in page:
            if ch >= len(fast_z_page):
                raise ValueError(
                    f"Channel {channel_1based} is not present in {flim_path.name}"
                )
            arr = np.asarray(fast_z_page[ch])
            if arr.size <= 1:
                raise ValueError(
                    f"Channel {channel_1based} was not acquired in {flim_path.name}"
                )
            if arr.ndim == 3:
                intensity = arr.sum(axis=-1).astype(np.float64)
            else:
                intensity = arr.astype(np.float64)
            z_slices.append(intensity)
        if not z_slices:
            continue
        pages.append(np.stack(z_slices, axis=0))

    if not pages:
        raise ValueError(f"Could not decode intensity volume from {flim_path}")

    stacked = np.stack(pages, axis=0)
    fast_z = bool(getattr(iminfo, "FastZStack", False)) and stacked.shape[1] > 1
    if fast_z:
        volume = stacked.mean(axis=0)
    elif stacked.shape[1] == 1:
        volume = stacked[:, 0, :, :]
    else:
        volume = stacked.reshape(-1, stacked.shape[2], stacked.shape[3])

    pixel_size = _pixel_sizes_from_iminfo(iminfo)
    zoom = float(getattr(iminfo.State.Acq, "zoom", 1.0) or 1.0)
    return LoadedVolume(
        volume_zyx=np.asarray(volume, dtype=np.float64),
        pixel_size=pixel_size,
        channel_1based=int(channel_1based),
        path=str(flim_path),
        zoom=zoom,
        fast_z=fast_z,
        n_pages=len(iminfo.image),
    )


def compute_projections(volume_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Z-max-projection YX, Y-max-projection ZX)."""
    vol = np.asarray(volume_zyx, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Expected a ZYX volume, got shape {vol.shape}")
    z_proj = np.max(vol, axis=0)
    y_proj = np.max(vol, axis=1)
    return z_proj, y_proj


def _round_clamp(value: float, n: int) -> int:
    return int(np.clip(int(round(value)), 0, max(n - 1, 0)))


def make_drawn_line(
    projection: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    image_shape: Sequence[int],
    label: str,
    color: str,
    mode: str = "auto",
    width_px: int = 1,
) -> DrawnLine:
    """Build an integer, axis-aligned line clipped to the image."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x0s, y0s, x1s, y1s, orientation = snap_to_axis(x0, y0, x1, y1, mode=mode)
    x0i = _round_clamp(x0s, width)
    x1i = _round_clamp(x1s, width)
    y0i = _round_clamp(y0s, height)
    y1i = _round_clamp(y1s, height)
    if orientation == "horizontal":
        y1i = y0i
    else:
        x1i = x0i
    width_px = max(1, int(width_px))
    if width_px % 2 == 0:
        width_px += 1
    return DrawnLine(
        projection=projection,
        x0_px=x0i,
        y0_px=y0i,
        x1_px=x1i,
        y1_px=y1i,
        orientation=orientation,
        color=color,
        label=label,
        width_px=width_px,
    )


def line_length_px(line: DrawnLine) -> int:
    """Inclusive pixel length of an axis-aligned line."""
    if line.orientation == "horizontal":
        return abs(line.x1_px - line.x0_px) + 1
    return abs(line.y1_px - line.y0_px) + 1


def extract_line_profile(
    image: np.ndarray,
    line: DrawnLine,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pixel indices along the line, mean intensity).

    ``width_px`` averages an odd number of pixels perpendicular to the line.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}")
    height, width = img.shape
    half = line.width_px // 2
    if line.orientation == "horizontal":
        c0, c1 = sorted((line.x0_px, line.x1_px))
        c0 = int(np.clip(c0, 0, width - 1))
        c1 = int(np.clip(c1, 0, width - 1))
        r0 = int(np.clip(line.y0_px - half, 0, height - 1))
        r1 = int(np.clip(line.y0_px + half, 0, height - 1))
        strip = img[r0 : r1 + 1, c0 : c1 + 1]
        profile = np.mean(strip, axis=0)
        indices = np.arange(c0, c1 + 1, dtype=np.float64)
    else:
        r0, r1 = sorted((line.y0_px, line.y1_px))
        r0 = int(np.clip(r0, 0, height - 1))
        r1 = int(np.clip(r1, 0, height - 1))
        c0 = int(np.clip(line.x0_px - half, 0, width - 1))
        c1 = int(np.clip(line.x0_px + half, 0, width - 1))
        strip = img[r0 : r1 + 1, c0 : c1 + 1]
        profile = np.mean(strip, axis=1)
        indices = np.arange(r0, r1 + 1, dtype=np.float64)
    return indices, np.asarray(profile, dtype=np.float64)


def _crossing_position(
    positions: np.ndarray,
    intensities: np.ndarray,
    peak_idx: int,
    half: float,
    direction: int,
) -> float:
    n = len(intensities)
    i = peak_idx
    while 0 <= i + direction < n and intensities[i] > half:
        i += direction
    if not (0 <= i < n) or intensities[i] > half:
        return float("nan")
    j = i - direction
    if j < 0 or j >= n:
        return float(positions[i])
    i0 = float(intensities[i])
    i1 = float(intensities[j])
    if i1 == i0:
        return float(positions[i])
    frac = (half - i0) / (i1 - i0)
    return float(positions[i] + frac * (positions[j] - positions[i]))


def _fit_gaussian_fwhm(positions: np.ndarray, intensities: np.ndarray) -> tuple[float, float]:
    """Return (gaussian_fwhm_um, sigma_um), or (nan, nan) on failure."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return float("nan"), float("nan")

    y = np.asarray(intensities, dtype=np.float64)
    x = np.asarray(positions, dtype=np.float64)
    if x.size < 5:
        return float("nan"), float("nan")
    background = float(np.min(y))
    peak = float(np.max(y))
    amp = peak - background
    if amp <= 0:
        return float("nan"), float("nan")
    mu0 = float(x[int(np.argmax(y))])
    sigma0 = max(float(np.std(x)) / 4.0, (x[1] - x[0]) if x.size > 1 else 1.0)

    def _gauss(xx, amp_, mu_, sigma_, offset_):
        return offset_ + amp_ * np.exp(-0.5 * ((xx - mu_) / np.maximum(sigma_, 1e-12)) ** 2)

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _gauss,
                x,
                y,
                p0=(amp, mu0, sigma0, background),
                bounds=(
                    (0.0, float(np.min(x)), 1e-9, -np.inf),
                    (np.inf, float(np.max(x)), np.inf, np.inf),
                ),
                maxfev=5000,
            )
    except (RuntimeError, ValueError, TypeError):
        return float("nan"), float("nan")
    sigma = abs(float(popt[2]))
    return float(FWHM_SIGMA_FACTOR * sigma), sigma


def compute_fwhm(
    positions_um: np.ndarray,
    intensities: np.ndarray,
    measured_axis: str,
    pixel_size_um: float,
) -> FwhmResult:
    """Background-subtracted FWHM by linear interpolation around the peak."""
    pos = np.asarray(positions_um, dtype=np.float64)
    inten = np.asarray(intensities, dtype=np.float64)
    nan = float("nan")
    empty = FwhmResult(
        fwhm_um=nan,
        fwhm_px=nan,
        x_left_um=nan,
        x_right_um=nan,
        peak_intensity=float(np.max(inten)) if inten.size else nan,
        background=float(np.min(inten)) if inten.size else nan,
        half_max=nan,
        peak_position_um=nan,
        gaussian_fwhm_um=nan,
        gaussian_sigma_um=nan,
        measured_axis=measured_axis,
        pixel_size_um=float(pixel_size_um),
        positions_um=pos,
        intensities=inten,
        ok=False,
    )
    if pos.size < MIN_LINE_PIXELS or inten.size != pos.size:
        return empty

    background = float(np.min(inten))
    peak = float(np.max(inten))
    amp = peak - background
    if amp <= 0:
        return empty

    half = background + 0.5 * amp
    peak_idx = int(np.argmax(inten))
    x_left = _crossing_position(pos, inten, peak_idx, half, direction=-1)
    x_right = _crossing_position(pos, inten, peak_idx, half, direction=1)
    ok = np.isfinite(x_left) and np.isfinite(x_right) and x_right > x_left
    fwhm_um = float(x_right - x_left) if ok else nan
    fwhm_px = fwhm_um / pixel_size_um if ok and pixel_size_um else nan
    g_fwhm, g_sigma = _fit_gaussian_fwhm(pos, inten)
    return FwhmResult(
        fwhm_um=fwhm_um,
        fwhm_px=float(fwhm_px) if np.isfinite(fwhm_px) else nan,
        x_left_um=x_left,
        x_right_um=x_right,
        peak_intensity=peak,
        background=background,
        half_max=half,
        peak_position_um=float(pos[peak_idx]),
        gaussian_fwhm_um=g_fwhm,
        gaussian_sigma_um=g_sigma,
        measured_axis=measured_axis,
        pixel_size_um=float(pixel_size_um),
        positions_um=pos,
        intensities=inten,
        ok=bool(ok),
    )


def measure_line(
    image: np.ndarray,
    line: DrawnLine,
    pixel_size: PixelSizeUm,
) -> LineMeasurement:
    """Extract the profile of one line and compute FWHM."""
    indices, intensities = extract_line_profile(image, line)
    dx = pixel_size_along_line(line.projection, line.orientation, pixel_size)
    positions_um = indices * dx
    axis = measured_axis_name(line.projection, line.orientation)
    fwhm = compute_fwhm(positions_um, intensities, axis, dx)
    return LineMeasurement(line=line, fwhm=fwhm)


def next_line_style(existing: Sequence[DrawnLine]) -> tuple[str, str]:
    """Return (label, color) for the next line."""
    n = len(existing) + 1
    return f"L{n}", LINE_COLORS[(n - 1) % len(LINE_COLORS)]


def make_synthetic_bead_volume(
    shape_zyx: tuple[int, int, int] = (41, 64, 64),
    center_zyx: tuple[float, float, float] = (20.0, 32.0, 32.0),
    sigma_zyx_px: tuple[float, float, float] = (6.0, 3.0, 3.0),
    amplitude: float = 1000.0,
    background: float = 10.0,
) -> np.ndarray:
    """3D Gaussian bead used for tests and headless self-test."""
    nz, ny, nx = shape_zyx
    z, y, x = np.ogrid[:nz, :ny, :nx]
    cz, cy, cx = center_zyx
    sz, sy, sx = sigma_zyx_px
    gauss = np.exp(
        -0.5
        * (
            ((z - cz) / sz) ** 2
            + ((y - cy) / sy) ** 2
            + ((x - cx) / sx) ** 2
        )
    )
    return background + amplitude * gauss


def theoretical_gaussian_fwhm_um(sigma_px: float, pixel_um: float) -> float:
    """FWHM of a Gaussian with the given sigma in pixels."""
    return float(FWHM_SIGMA_FACTOR * sigma_px * pixel_um)


def _imshow_projection(
    ax,
    image: np.ndarray,
    pixel_size: PixelSizeUm,
    projection: str,
    vmin: float,
    vmax: float,
    title: str,
) -> None:
    if projection == "z_proj":
        aspect = pixel_size.y_um / pixel_size.x_um if pixel_size.x_um else 1.0
        xlabel, ylabel = "X (pixels)", "Y (pixels)"
        xlabel_um, ylabel_um = pixel_size.x_um, pixel_size.y_um
    else:
        aspect = pixel_size.z_um / pixel_size.x_um if pixel_size.x_um else 1.0
        xlabel, ylabel = "X (pixels)", "Z (pixels)"
        xlabel_um, ylabel_um = pixel_size.x_um, pixel_size.z_um
    ax.imshow(
        image,
        cmap="gray",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        aspect=aspect if np.isfinite(aspect) and aspect > 0 else "auto",
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(f"{xlabel}; 1 px = {xlabel_um:.4g} um")
    ax.set_ylabel(f"{ylabel}; 1 px = {ylabel_um:.4g} um")


def _draw_lines_on_axes(ax, lines: Iterable[DrawnLine]) -> None:
    for line in lines:
        ax.plot(
            [line.x0_px, line.x1_px],
            [line.y0_px, line.y1_px],
            color=line.color,
            lw=1.5,
            solid_capstyle="butt",
        )
        mid_x = 0.5 * (line.x0_px + line.x1_px)
        mid_y = 0.5 * (line.y0_px + line.y1_px)
        ax.text(
            mid_x,
            mid_y,
            line.label,
            color=line.color,
            fontsize=8,
            ha="left",
            va="bottom",
            fontweight="bold",
        )


def _plot_one_profile(ax, measurement: LineMeasurement) -> None:
    fwhm = measurement.fwhm
    line = measurement.line
    ax.plot(
        fwhm.positions_um,
        fwhm.intensities,
        color=line.color,
        lw=1.5,
        label=line.label,
    )
    if np.isfinite(fwhm.half_max):
        ax.axhline(fwhm.half_max, color=line.color, ls="--", lw=0.8, alpha=0.8)
    if fwhm.ok:
        ax.axvline(fwhm.x_left_um, color=line.color, ls=":", lw=0.8)
        ax.axvline(fwhm.x_right_um, color=line.color, ls=":", lw=0.8)
        fwhm_txt = f"FWHM {fwhm.measured_axis} = {fwhm.fwhm_um:.3f} um"
        if np.isfinite(fwhm.gaussian_fwhm_um):
            fwhm_txt += f"  (Gauss {fwhm.gaussian_fwhm_um:.3f} um)"
    else:
        fwhm_txt = "FWHM: n/a"
    ax.set_title(f"{line.label}  {line.projection}  {fwhm_txt}", fontsize=9)
    ax.set_xlabel(f"Position along {fwhm.measured_axis} (um)")
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.3)


def default_contrast(image: np.ndarray) -> tuple[float, float]:
    """Return (vmin, vmax) for display; vmax is the image maximum."""
    img = np.asarray(image, dtype=np.float64)
    if img.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def make_output_dir(save_root: str | Path | None, flim_path: str | Path) -> Path:
    """Return the folder for plots/CSV.

    Defaults to the directory that contains the .flim file. ``save_root``
    overrides that (used by tests and headless self-test).
    """
    if save_root is not None:
        out = Path(save_root)
    else:
        flim = Path(flim_path).expanduser()
        out = flim.parent
        if str(out) in ("", "."):
            out = Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    return out


def output_stem_prefix(source_name: str) -> str:
    """Filename prefix so outputs next to the .flim are identifiable."""
    stem = Path(source_name).stem or "bead"
    return f"{stem}_"


def save_bead_fwhm_outputs(
    out_dir: str | Path,
    z_proj: np.ndarray,
    y_proj: np.ndarray,
    pixel_size: PixelSizeUm,
    measurements: Sequence[LineMeasurement],
    z_contrast: tuple[float, float],
    y_contrast: tuple[float, float],
    source_name: str = "bead",
    channel_1based: int = 1,
) -> dict[str, Path]:
    """Save projection overlays, profile plots, and CSV tables."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_stem_prefix(source_name)
    z_lines = [m.line for m in measurements if m.line.projection == "z_proj"]
    y_lines = [m.line for m in measurements if m.line.projection == "y_proj"]

    z_path = out_dir / f"{prefix}z_projection_with_lines.png"
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=120)
    _imshow_projection(
        ax,
        z_proj,
        pixel_size,
        "z_proj",
        z_contrast[0],
        z_contrast[1],
        f"{source_name}  Ch{channel_1based}  Z max projection (XY)",
    )
    _draw_lines_on_axes(ax, z_lines)
    fig.tight_layout()
    fig.savefig(z_path, dpi=150)
    plt.close(fig)

    y_path = out_dir / f"{prefix}y_projection_with_lines.png"
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=120)
    _imshow_projection(
        ax,
        y_proj,
        pixel_size,
        "y_proj",
        y_contrast[0],
        y_contrast[1],
        f"{source_name}  Ch{channel_1based}  Y max projection (XZ)",
    )
    _draw_lines_on_axes(ax, y_lines)
    fig.tight_layout()
    fig.savefig(y_path, dpi=150)
    plt.close(fig)

    n_prof = max(len(measurements), 1)
    fig, axes = plt.subplots(n_prof, 1, figsize=(8.0, 2.4 * n_prof), dpi=120, squeeze=False)
    if measurements:
        for ax, meas in zip(axes[:, 0], measurements):
            _plot_one_profile(ax, meas)
    else:
        axes[0, 0].text(0.5, 0.5, "No lines", ha="center", va="center")
        axes[0, 0].set_axis_off()
    fig.tight_layout()
    profiles_path = out_dir / f"{prefix}intensity_profiles.png"
    fig.savefig(profiles_path, dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(12.0, 8.5), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax_z = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, :])
    _imshow_projection(
        ax_z,
        z_proj,
        pixel_size,
        "z_proj",
        z_contrast[0],
        z_contrast[1],
        "Z max projection (XY)",
    )
    _draw_lines_on_axes(ax_z, z_lines)
    _imshow_projection(
        ax_y,
        y_proj,
        pixel_size,
        "y_proj",
        y_contrast[0],
        y_contrast[1],
        "Y max projection (XZ)",
    )
    _draw_lines_on_axes(ax_y, y_lines)
    for meas in measurements:
        fwhm = meas.fwhm
        ax_p.plot(
            fwhm.positions_um,
            fwhm.intensities,
            color=meas.line.color,
            lw=1.4,
            label=(
                f"{meas.line.label} {fwhm.measured_axis} "
                f"FWHM={fwhm.fwhm_um:.3f} um"
                if fwhm.ok
                else f"{meas.line.label} FWHM=n/a"
            ),
        )
        if fwhm.ok:
            ax_p.axhline(fwhm.half_max, color=meas.line.color, ls="--", lw=0.7, alpha=0.6)
    ax_p.set_xlabel("Position (um)")
    ax_p.set_ylabel("Intensity")
    ax_p.set_title("Line intensity profiles")
    if measurements:
        ax_p.legend(fontsize=8, loc="best")
    ax_p.grid(True, alpha=0.3)
    fig.suptitle(f"{source_name}  Ch{channel_1based} bead FWHM", fontsize=11)
    fig.tight_layout()
    combined_path = out_dir / f"{prefix}combined.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    fwhm_csv = out_dir / f"{prefix}fwhm_results.csv"
    with fwhm_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "projection",
                "orientation",
                "measured_axis",
                "x0_px",
                "y0_px",
                "x1_px",
                "y1_px",
                "width_px",
                "fwhm_um",
                "fwhm_px",
                "gaussian_fwhm_um",
                "peak_intensity",
                "background",
                "half_max",
                "pixel_size_um",
                "ok",
            ],
        )
        writer.writeheader()
        for meas in measurements:
            line, fwhm = meas.line, meas.fwhm
            writer.writerow(
                {
                    "label": line.label,
                    "projection": line.projection,
                    "orientation": line.orientation,
                    "measured_axis": fwhm.measured_axis,
                    "x0_px": line.x0_px,
                    "y0_px": line.y0_px,
                    "x1_px": line.x1_px,
                    "y1_px": line.y1_px,
                    "width_px": line.width_px,
                    "fwhm_um": f"{fwhm.fwhm_um:.6f}" if fwhm.ok else "",
                    "fwhm_px": f"{fwhm.fwhm_px:.6f}" if fwhm.ok else "",
                    "gaussian_fwhm_um": (
                        f"{fwhm.gaussian_fwhm_um:.6f}"
                        if np.isfinite(fwhm.gaussian_fwhm_um)
                        else ""
                    ),
                    "peak_intensity": f"{fwhm.peak_intensity:.6f}",
                    "background": f"{fwhm.background:.6f}",
                    "half_max": f"{fwhm.half_max:.6f}" if np.isfinite(fwhm.half_max) else "",
                    "pixel_size_um": f"{fwhm.pixel_size_um:.6f}",
                    "ok": int(fwhm.ok),
                }
            )

    profile_csv = out_dir / f"{prefix}intensity_profiles.csv"
    with profile_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["label", "projection", "measured_axis", "position_um", "intensity"],
        )
        writer.writeheader()
        for meas in measurements:
            for pos, inten in zip(meas.fwhm.positions_um, meas.fwhm.intensities):
                writer.writerow(
                    {
                        "label": meas.line.label,
                        "projection": meas.line.projection,
                        "measured_axis": meas.fwhm.measured_axis,
                        "position_um": f"{float(pos):.6f}",
                        "intensity": f"{float(inten):.6f}",
                    }
                )

    return {
        "z_projection": z_path,
        "y_projection": y_path,
        "intensity_profiles": profiles_path,
        "combined": combined_path,
        "fwhm_csv": fwhm_csv,
        "profile_csv": profile_csv,
        "out_dir": out_dir,
    }


def measure_center_lines_on_synthetic(
    volume_zyx: np.ndarray,
    pixel_size: PixelSizeUm,
    center_zyx: tuple[float, float, float],
    width_px: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[LineMeasurement]]:
    """Draw X/Y/Z lines through a bead center and measure FWHM."""
    z_proj, y_proj = compute_projections(volume_zyx)
    cz, cy, cx = center_zyx
    ny, nx = z_proj.shape
    nz = y_proj.shape[0]
    specs = [
        ("z_proj", z_proj, 0, cy, nx - 1, cy, "horizontal", "L1"),
        ("z_proj", z_proj, cx, 0, cx, ny - 1, "vertical", "L2"),
        ("y_proj", y_proj, 0, cz, nx - 1, cz, "horizontal", "L3"),
        ("y_proj", y_proj, cx, 0, cx, nz - 1, "vertical", "L4"),
    ]
    measurements: list[LineMeasurement] = []
    for i, (proj, img, x0, y0, x1, y1, mode, label) in enumerate(specs):
        line = make_drawn_line(
            projection=proj,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            image_shape=img.shape,
            label=label,
            color=LINE_COLORS[i],
            mode=mode,
            width_px=width_px,
        )
        measurements.append(measure_line(img, line, pixel_size))
    return z_proj, y_proj, measurements


def run_self_test(output_dir: str | Path | None = None, rel_tol: float = 0.08) -> dict:
    """Headless synthetic-bead test. Raises AssertionError on failure.

    Environment:
        BEAD_FWHM_DATA_DIR: optional output folder for saved plots/CSV.
    """
    if output_dir is None:
        output_dir = os.environ.get(SELF_TEST_DATA_DIR_ENV, str(_CALIB_DIR / "self_test_output"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_size = PixelSizeUm(z_um=0.50, y_um=0.10, x_um=0.10)
    sigma_zyx = (6.0, 3.0, 3.0)
    center = (20.0, 32.0, 32.0)
    volume = make_synthetic_bead_volume(
        shape_zyx=(41, 64, 64),
        center_zyx=center,
        sigma_zyx_px=sigma_zyx,
        amplitude=1000.0,
        background=25.0,
    )
    z_proj, y_proj, measurements = measure_center_lines_on_synthetic(
        volume, pixel_size, center
    )
    expected = {
        "L1": theoretical_gaussian_fwhm_um(sigma_zyx[2], pixel_size.x_um),
        "L2": theoretical_gaussian_fwhm_um(sigma_zyx[1], pixel_size.y_um),
        "L3": theoretical_gaussian_fwhm_um(sigma_zyx[2], pixel_size.x_um),
        "L4": theoretical_gaussian_fwhm_um(sigma_zyx[0], pixel_size.z_um),
    }
    by_label = {m.line.label: m for m in measurements}
    errors: dict[str, float] = {}
    for label, exp in expected.items():
        meas = by_label[label]
        assert meas.fwhm.ok, f"{label} FWHM was not computed"
        rel = abs(meas.fwhm.fwhm_um - exp) / exp
        errors[label] = rel
        assert rel <= rel_tol, (
            f"{label} FWHM {meas.fwhm.fwhm_um:.4f} um vs expected {exp:.4f} um "
            f"(rel err {rel:.3f} > {rel_tol})"
        )

    z_c = default_contrast(z_proj)
    y_c = default_contrast(y_proj)
    saved = save_bead_fwhm_outputs(
        output_dir,
        z_proj,
        y_proj,
        pixel_size,
        measurements,
        z_c,
        y_c,
        source_name="synthetic_bead",
        channel_1based=1,
    )
    for key in ("z_projection", "y_projection", "intensity_profiles", "combined", "fwhm_csv"):
        path = saved[key]
        assert path.is_file() and path.stat().st_size > 0, f"Missing output: {key}"

    with saved["fwhm_csv"].open("r", encoding="utf-8") as fh:
        n_rows = sum(1 for _ in csv.DictReader(fh))
    assert n_rows == 4, f"Expected 4 FWHM CSV rows, got {n_rows}"

    with saved["profile_csv"].open("r", encoding="utf-8") as fh:
        n_prof = sum(1 for _ in csv.DictReader(fh))
    assert n_prof > 20, f"Profile CSV too short: {n_prof} rows"

    return {
        "errors": errors,
        "expected_um": expected,
        "measured_um": {k: float(by_label[k].fwhm.fwhm_um) for k in expected},
        "saved": saved,
        "z_proj_shape": tuple(z_proj.shape),
        "y_proj_shape": tuple(y_proj.shape),
        "n_fwhm_rows": n_rows,
        "n_profile_rows": n_prof,
    }
