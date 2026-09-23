"""Gentle quadratic sample surface from a Z stack.

A column has reached the surface when its signal ends before the last Z
slice. Signal still present on the last slice means the stack stopped
before the surface. Empty background is not counted as unreached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import minimize

NOISE_THRESHOLD = 8.0
XY_SMOOTH_SIGMA = 1.0
MIN_BLOB_PX = 4
BIN_PX = 32
CURVATURE_WEIGHT = 2.0
TILT_WEIGHT = 0.15

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
    signal = _signal_mask(smoothed, NOISE_THRESHOLD, MIN_BLOB_PX)
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
        )

    if np.all(z_top[has_signal] == top_index):
        return GentleSurfaceResult(
            surface_z=_top_plane(n_z, n_y, n_x),
            not_reached=not_reached,
            warning=WARNING_NOT_REACHED,
            status=STATUS_NOT_REACHED,
        )

    bin_x, bin_y, bin_z = _bin_ceilings(z_top, BIN_PX)
    if len(bin_z) < 1:
        return GentleSurfaceResult(
            surface_z=_top_plane(n_z, n_y, n_x),
            not_reached=not_reached,
            warning=WARNING_NO_SIGNAL,
            status=STATUS_NO_SIGNAL,
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
    )
