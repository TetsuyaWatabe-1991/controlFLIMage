"""Gentle quadratic sample surface from a Z stack.

A column has reached the surface when its signal ends before the last Z
slice. Signal still present on the last slice means the stack stopped
before the surface. Empty background is not counted as unreached.
The intensity cutoff is Otsu's threshold on this stack's maximum
projection, not a fixed count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, label, median_filter
from scipy.optimize import minimize

XY_SMOOTH_SIGMA = 1.0
MIN_BLOB_PX = 4
OTSU_BINS = 256
BIN_PX = 32
CURVATURE_WEIGHT = 2.0
TILT_WEIGHT = 0.15
MEDIAN_PARABOLA_THRESHOLD = 2.0
PARABOLA_COEF_LIMIT_PER_UM = 1.0

STATUS_REACHED = "reached"
STATUS_PARTIAL = "partial"
STATUS_NOT_REACHED = "not_reached"
STATUS_NO_SIGNAL = "no_signal"

WARNING_NOT_REACHED = (
    "Surface was not reached anywhere in the field. Returning the top Z plane."
)
WARNING_PARTIAL = (
    "Surface was not reached in part of the field. "
    "Those pixels are marked in not_reached."
)
WARNING_NO_SIGNAL = "No sample signal. Returning the top Z plane."


@dataclass
class GentleSurfaceResult:
    """Fitted surface and where the stack did not reach it.

    surface_z is in Z-slice index, shape (Y, X).
    not_reached is True where the last slice still contains sample signal.
    """

    surface_z: np.ndarray
    not_reached: np.ndarray
    warning: str | None
    status: str
    threshold: float


def findsurface_stack_plan(
    below_um: float,
    above_um: float,
    step_um: float,
) -> tuple[int, float]:
    """Return n_slices and the motor offset that centers that stack.

    The stack then runs from csv_z - below_um to csv_z + above_um.
    FLIMage centers a Z stack on the current motor position.
    """
    if step_um <= 0:
        raise ValueError("step_um must be positive")
    if below_um < 0 or above_um < 0:
        raise ValueError("below_um and above_um must be non-negative")
    span_um = float(below_um) + float(above_um)
    n_intervals = int(round(span_um / float(step_um)))
    if abs(n_intervals * float(step_um) - span_um) > 1e-6:
        raise ValueError("below_um + above_um must be a multiple of step_um")
    n_slices = n_intervals + 1
    center_offset_um = (float(above_um) - float(below_um)) / 2.0
    return n_slices, center_offset_um


def slice_index_to_motor_um(
    csv_z_um: float,
    slice_index: float,
    below_um: float,
    step_um: float,
) -> float:
    """Motor Z in um for a slice index. Index 0 is the bottom of the stack."""
    return float(csv_z_um) - float(below_um) + float(slice_index) * float(step_um)


def um_down_to_stack_center(slice_step_um: float, n_slices: int) -> float:
    """Microns from the top of a centered Z stack down to its center.

    Equal to slice_step * (n_slices - 1) / 2.
    """
    if int(n_slices) < 1:
        raise ValueError("n_slices must be at least 1")
    return float(slice_step_um) * (int(n_slices) - 1) / 2.0


def reference_motor_z_um(
    csv_z_um: float,
    surface_z: np.ndarray,
    below_um: float,
    step_um: float,
    rel_z_um: float,
) -> float:
    """Stage Z for the real acquisition.

    The reference is the highest point of the surface, where only a sliver
    of the sheet is in plane. rel_z_um is added to that motor position.
    """
    highest = float(np.max(surface_z))
    return slice_index_to_motor_um(csv_z_um, highest, below_um, step_um) + float(rel_z_um)


def _design(x: np.ndarray, y: np.ndarray, n_x: int, n_y: int) -> np.ndarray:
    x_scale = max((n_x - 1) / 2.0, 1.0)
    y_scale = max((n_y - 1) / 2.0, 1.0)
    xn = (np.asarray(x, dtype=float) - (n_x - 1) / 2.0) / x_scale
    yn = (np.asarray(y, dtype=float) - (n_y - 1) / 2.0) / y_scale
    return np.column_stack([xn ** 2, xn * yn, yn ** 2, xn, yn, np.ones(xn.shape[0])])


def _otsu_threshold(image: np.ndarray) -> float:
    """Otsu cutoff. A flat image returns its only value, so nothing is above it."""
    finite = np.asarray(image, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    low = float(finite.min())
    high = float(finite.max())
    if high <= low:
        return high
    hist, edges = np.histogram(finite, bins=OTSU_BINS, range=(low, high))
    centers = (edges[:-1] + edges[1:]) * 0.5
    total = float(hist.sum())
    weight = hist.astype(np.float64)
    sum_all = float((weight * centers).sum())
    weight0 = 0.0
    sum0 = 0.0
    best = -1.0
    threshold = low
    for index in range(len(weight) - 1):
        weight0 += weight[index]
        if weight0 <= 0.0:
            continue
        weight1 = total - weight0
        if weight1 <= 0.0:
            break
        sum0 += weight[index] * centers[index]
        mean0 = sum0 / weight0
        mean1 = (sum_all - sum0) / weight1
        score = weight0 * weight1 * (mean0 - mean1) ** 2
        if score > best:
            best = score
            threshold = float(edges[index + 1])
    return threshold


def intensity_threshold(zyx: np.ndarray) -> float:
    """Cutoff for one stack, taken from that stack rather than a fixed count.

    The XY maximum projection of the smoothed volume is split with Otsu's
    method. A brighter acquisition gets a higher cutoff.
    """
    volume = np.asarray(zyx, dtype=np.float32)
    smoothed = gaussian_filter(volume, sigma=(0.0, XY_SMOOTH_SIGMA, XY_SMOOTH_SIGMA))
    projection = np.max(smoothed, axis=0)
    return _otsu_threshold(projection)


def _signal_mask(smoothed: np.ndarray, threshold: float, min_blob_px: int) -> np.ndarray:
    signal = np.zeros(smoothed.shape, dtype=bool)
    for z_index in range(smoothed.shape[0]):
        labels, n_labels = label(smoothed[z_index] > threshold)
        if n_labels == 0:
            continue
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_blob_px
        keep[0] = False
        signal[z_index] = keep[labels]
    return signal


def _noise_ceiling_z(signal: np.ndarray) -> np.ndarray:
    reversed_signal = signal[::-1]
    has_signal = reversed_signal.any(axis=0)
    first_from_top = reversed_signal.argmax(axis=0)
    z_top = signal.shape[0] - 1 - first_from_top
    return np.where(has_signal, z_top, -1).astype(np.int16)


def _bin_ceilings(z_top: np.ndarray, bin_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _n_z, n_y, n_x = (0, *z_top.shape)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for y0 in range(0, n_y - bin_px + 1, bin_px):
        for x0 in range(0, n_x - bin_px + 1, bin_px):
            patch = z_top[y0 : y0 + bin_px, x0 : x0 + bin_px]
            if not np.any(patch >= 0):
                continue
            xs.append(x0 + (bin_px - 1) / 2.0)
            ys.append(y0 + (bin_px - 1) / 2.0)
            zs.append(float(patch.max()))
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def _fit_lowest_gentle_sheet(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_x: int,
    n_y: int,
) -> np.ndarray:
    design = _design(x, y, n_x, n_y)
    constraints = [
        {
            "type": "ineq",
            "fun": lambda coef, row=design[i], floor=z[i]: float(row @ coef - floor),
        }
        for i in range(len(z))
    ]

    def objective(coef: np.ndarray) -> float:
        height = float(np.mean(design @ coef))
        curvature = float(np.sum(coef[:3] ** 2))
        tilt = float(np.sum(coef[3:5] ** 2))
        return height + CURVATURE_WEIGHT * curvature + TILT_WEIGHT * tilt

    start = np.zeros(6, dtype=float)
    start[5] = float(np.max(z))
    result = minimize(
        objective,
        start,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Surface fit failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _top_plane(n_z: int, n_y: int, n_x: int) -> np.ndarray:
    return np.full((n_y, n_x), float(n_z - 1), dtype=float)


def fit_gentle_surface(zyx: np.ndarray) -> GentleSurfaceResult:
    """Fit the noise-ceiling surface of one intensity stack, shape (Z, Y, X)."""
    volume = np.asarray(zyx, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("zyx must have shape (Z, Y, X)")
    n_z, n_y, n_x = volume.shape
    if n_z < 2:
        raise ValueError("zyx needs at least 2 Z slices")

    smoothed = gaussian_filter(volume, sigma=(0.0, XY_SMOOTH_SIGMA, XY_SMOOTH_SIGMA))
    threshold = _otsu_threshold(np.max(smoothed, axis=0))
    signal = _signal_mask(smoothed, threshold, MIN_BLOB_PX)
    z_top = _noise_ceiling_z(signal)
    not_reached = signal[-1].copy()
    has_signal = z_top >= 0
    top_index = n_z - 1

    if not np.any(has_signal):
        return GentleSurfaceResult(
            surface_z=_top_plane(n_z, n_y, n_x),
            not_reached=np.zeros((n_y, n_x), dtype=bool),
            warning=WARNING_NO_SIGNAL,
            status=STATUS_NO_SIGNAL,
            threshold=threshold,
        )

    if np.all(z_top[has_signal] == top_index):
        return GentleSurfaceResult(
            surface_z=_top_plane(n_z, n_y, n_x),
            not_reached=not_reached,
            warning=WARNING_NOT_REACHED,
            status=STATUS_NOT_REACHED,
            threshold=threshold,
        )

    bin_x, bin_y, bin_z = _bin_ceilings(z_top, BIN_PX)
    if len(bin_z) < 1:
        return GentleSurfaceResult(
            surface_z=_top_plane(n_z, n_y, n_x),
            not_reached=not_reached,
            warning=WARNING_NO_SIGNAL,
            status=STATUS_NO_SIGNAL,
            threshold=threshold,
        )

    coef = _fit_lowest_gentle_sheet(bin_x, bin_y, bin_z, n_x, n_y)
    yy, xx = np.indices((n_y, n_x))
    surface_z = (
        _design(xx.ravel().astype(float), yy.ravel().astype(float), n_x, n_y) @ coef
    ).reshape(n_y, n_x)

    if np.any(not_reached):
        warning: str | None = WARNING_PARTIAL
        status = STATUS_PARTIAL
    else:
        warning = None
        status = STATUS_REACHED
    return GentleSurfaceResult(
        surface_z=surface_z,
        not_reached=not_reached,
        warning=warning,
        status=status,
        threshold=threshold,
    )


def _design_um(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    return np.column_stack([xn ** 2, xn * yn, yn ** 2, xn, yn, np.ones(xn.shape[0])])


def fit_surface_median_parabola(
    zyx: np.ndarray,
    x_um_per_px: float,
    y_um_per_px: float,
    z_um_per_slice: float,
) -> GentleSurfaceResult:
    """Noise-ceiling sheet after a 3x3 XY median and a fixed threshold of 2.

    x, y, and z are in micrometers. Quadratic coefficients stay within
    +/- PARABOLA_COEF_LIMIT_PER_UM. The sheet is the lowest one that stays
    at or above every 32 px bin ceiling.
    """
    filtered = median_filter(np.asarray(zyx, dtype=np.float32), size=(1, 3, 3))
    n_z, n_y, n_x = filtered.shape
    signal = _signal_mask(filtered, MEDIAN_PARABOLA_THRESHOLD, MIN_BLOB_PX)
    z_top = _noise_ceiling_z(signal)
    not_reached = signal[-1].copy()
    has_signal = z_top >= 0
    top = _top_plane(n_z, n_y, n_x)
    if not np.any(has_signal):
        return GentleSurfaceResult(
            surface_z=top,
            not_reached=np.zeros((n_y, n_x), dtype=bool),
            warning=WARNING_NO_SIGNAL,
            status=STATUS_NO_SIGNAL,
            threshold=MEDIAN_PARABOLA_THRESHOLD,
        )
    if np.all(z_top[has_signal] == n_z - 1):
        return GentleSurfaceResult(
            surface_z=top,
            not_reached=not_reached,
            warning=WARNING_NOT_REACHED,
            status=STATUS_NOT_REACHED,
            threshold=MEDIAN_PARABOLA_THRESHOLD,
        )
    bin_x, bin_y, bin_z = _bin_ceilings(z_top, BIN_PX)
    if len(bin_z) < 1:
        return GentleSurfaceResult(
            surface_z=top,
            not_reached=not_reached,
            warning=WARNING_NO_SIGNAL,
            status=STATUS_NO_SIGNAL,
            threshold=MEDIAN_PARABOLA_THRESHOLD,
        )
    half_x = max(0.5 * (n_x - 1) * float(x_um_per_px), 1e-6)
    half_y = max(0.5 * (n_y - 1) * float(y_um_per_px), 1e-6)
    xn = (bin_x * float(x_um_per_px) - half_x) / half_x
    yn = (bin_y * float(y_um_per_px) - half_y) / half_y
    z_um_bins = bin_z * float(z_um_per_slice)
    design = _design_um(xn, yn)
    a_max = PARABOLA_COEF_LIMIT_PER_UM * half_x ** 2
    b_max = PARABOLA_COEF_LIMIT_PER_UM * half_x * half_y
    c_max = PARABOLA_COEF_LIMIT_PER_UM * half_y ** 2
    bounds = [(-a_max, a_max), (-b_max, b_max), (-c_max, c_max), (None, None), (None, None), (None, None)]
    constraints = [
        {"type": "ineq", "fun": lambda coef, row=design[i], floor=z_um_bins[i]: float(row @ coef - floor)}
        for i in range(len(z_um_bins))
    ]
    start = np.zeros(6, dtype=float)
    start[5] = float(np.max(z_um_bins))
    result = minimize(
        lambda coef: float(np.mean(design @ coef)),
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 800, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Surface fit failed: {result.message}")
    coef = np.asarray(result.x, dtype=float)
    yy, xx = np.indices((n_y, n_x))
    xnf = (xx.ravel() * float(x_um_per_px) - half_x) / half_x
    ynf = (yy.ravel() * float(y_um_per_px) - half_y) / half_y
    surface_um = (_design_um(xnf, ynf) @ coef).reshape(n_y, n_x)
    surface_z = surface_um / float(z_um_per_slice)
    if np.any(not_reached):
        warning: str | None = WARNING_PARTIAL
        status = STATUS_PARTIAL
    else:
        warning = None
        status = STATUS_REACHED
    return GentleSurfaceResult(
        surface_z=surface_z,
        not_reached=not_reached,
        warning=warning,
        status=status,
        threshold=MEDIAN_PARABOLA_THRESHOLD,
    )
