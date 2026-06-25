# -*- coding: utf-8 -*-
"""Drift alignment method variants for comparison with FLIMageAlignment baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import fourier_shift, shift as ndimage_shift
from scipy.signal import medfilt, savgol_filter
from skimage.filters import laplace
from skimage.registration import phase_cross_correlation


ShiftZYx = Tuple[float, float, float]
AlignResult = Tuple[np.ndarray, np.ndarray]  # (shifts [T,3], aligned_stack [T,Z,Y,X])


@dataclass
class RoiCropSpec:
    """Center (Z, Y, X) and half-extents for local alignment."""

    z: int = 6
    y: int = 35
    x: int = 33
    # Defaults aligned with gui_integration.process_small_region (60 px, Z±2).
    half_z: int = 2
    half_y: int = 30
    half_x: int = 30


def _to_float(stack: np.ndarray) -> np.ndarray:
    return stack.astype(np.float64)


def crop_zyx(
    volume: np.ndarray,
    spec: RoiCropSpec,
) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    """Return cropped volume and slice bounds (z0, z1, y0, y1, x0, x1)."""
    z0 = max(0, spec.z - spec.half_z)
    z1 = min(volume.shape[0], spec.z + spec.half_z + 1)
    y0 = max(0, spec.y - spec.half_y)
    y1 = min(volume.shape[1], spec.y + spec.half_y + 1)
    x0 = max(0, spec.x - spec.half_x)
    x1 = min(volume.shape[2], spec.x + spec.half_x + 1)
    return volume[z0:z1, y0:y1, x0:x1], (z0, z1, y0, y1, x0, x1)


def apply_shift_3d(
    volume: np.ndarray,
    shift_zyx: ShiftZYx,
    *,
    mode: str = "fourier",
    order: int = 3,
) -> np.ndarray:
    """Apply sub-pixel shift to a 3D volume."""
    if mode == "fourier":
        img_corr = fourier_shift(np.fft.fftn(volume), shift_zyx)
        return np.fft.ifftn(img_corr).real
    if mode == "spline":
        return ndimage_shift(volume, shift=shift_zyx, order=order, mode="nearest")
    if mode == "constant":
        return ndimage_shift(volume, shift=shift_zyx, order=order, mode="constant", cval=0.0)
    raise ValueError(f"Unknown shift mode: {mode}")


def estimate_shift_3d(
    ref: np.ndarray,
    query: np.ndarray,
    *,
    upsample_factor: int = 4,
    ref_for_corr: Optional[np.ndarray] = None,
    query_for_corr: Optional[np.ndarray] = None,
    median_filter: bool = False,
    ksize: int = 3,
) -> Tuple[ShiftZYx, float]:
    """Phase cross-correlation shift (Z, Y, X) from ref to query."""
    if ref_for_corr is None:
        ref_for_corr = ref
    if query_for_corr is None:
        query_for_corr = query

    if median_filter:
        ref_for_corr = medfilt(ref_for_corr, kernel_size=ksize)
        query_for_corr = medfilt(query_for_corr, kernel_size=ksize)

    shift, error, _ = phase_cross_correlation(
        ref_for_corr,
        query_for_corr,
        upsample_factor=upsample_factor,
    )
    return (float(shift[0]), float(shift[1]), float(shift[2])), float(error)


def estimate_shift_xy_z_split(
    ref: np.ndarray,
    query: np.ndarray,
    *,
    upsample_factor: int = 10,
    ref_for_corr: Optional[np.ndarray] = None,
    query_for_corr: Optional[np.ndarray] = None,
) -> Tuple[ShiftZYx, float]:
    """Estimate XY from max-Z projection, then Z from max-Y projection."""
    if ref_for_corr is None:
        ref_for_corr = ref
    if query_for_corr is None:
        query_for_corr = query

    ref_xy = np.max(ref_for_corr, axis=0)
    query_xy = np.max(query_for_corr, axis=0)
    shift_yx, error_xy, _ = phase_cross_correlation(
        ref_xy,
        query_xy,
        upsample_factor=upsample_factor,
    )
    shift_y = float(shift_yx[0])
    shift_x = float(shift_yx[1])

    query_xy_corr = apply_shift_3d(query, (0.0, shift_y, shift_x), mode="spline", order=1)
    if query_for_corr is not query:
        query_corr_for_z = apply_shift_3d(query_for_corr, (0.0, shift_y, shift_x), mode="spline", order=1)
    else:
        query_corr_for_z = query_xy_corr

    ref_zx = np.max(ref_for_corr, axis=1)
    query_zx = np.max(query_corr_for_z, axis=1)
    shift_zx, error_zx, _ = phase_cross_correlation(
        ref_zx,
        query_zx,
        upsample_factor=upsample_factor,
    )
    shift_z = float(shift_zx[0])
    combined_error = float(error_xy + error_zx)
    return (shift_z, shift_y, shift_x), combined_error


def structure_image_3d(volume: np.ndarray) -> np.ndarray:
    """Laplacian-based structural image for correlation (reduces bleach bias)."""
    vol = _to_float(volume)
    lap = np.abs(laplace(vol))
    return lap / (lap.max() + 1e-6)


def align_stack_from_shifts(
    stack: np.ndarray,
    shifts: np.ndarray,
    *,
    shift_mode: str = "fourier",
) -> np.ndarray:
    """Apply per-frame shifts to a time series stack."""
    aligned = []
    for t in range(stack.shape[0]):
        aligned.append(
            apply_shift_3d(stack[t], tuple(shifts[t]), mode=shift_mode)
        )
    return np.array(aligned)


def _align_series(
    stack: np.ndarray,
    estimator: Callable[[np.ndarray, np.ndarray], Tuple[ShiftZYx, float]],
    *,
    reference: str = "first",
    shift_mode: str = "fourier",
    roi_spec: Optional[RoiCropSpec] = None,
) -> AlignResult:
    """Generic alignment loop with selectable reference strategy."""
    n = stack.shape[0]
    raw_shifts = np.zeros((n, 3), dtype=np.float64)

    if reference == "first":
        ref_idx = 0
        for t in range(n):
            ref_vol = stack[ref_idx]
            query_vol = stack[t]
            if roi_spec is not None:
                ref_vol, _ = crop_zyx(ref_vol, roi_spec)
                query_vol, _ = crop_zyx(query_vol, roi_spec)
            shift, _ = estimator(ref_vol, query_vol)
            raw_shifts[t] = shift
    elif reference == "adjacent":
        for t in range(1, n):
            ref_vol = stack[t - 1]
            query_vol = stack[t]
            if roi_spec is not None:
                ref_vol, _ = crop_zyx(ref_vol, roi_spec)
                query_vol, _ = crop_zyx(query_vol, roi_spec)
            delta, _ = estimator(ref_vol, query_vol)
            raw_shifts[t] = raw_shifts[t - 1] + np.array(delta)
    else:
        raise ValueError(f"Unknown reference mode: {reference}")

    aligned = align_stack_from_shifts(stack, raw_shifts, shift_mode=shift_mode)
    return raw_shifts, aligned


def align_traditional(
    stack: np.ndarray,
    *,
    upsample_factor: int = 4,
    median_filter: bool = False,
) -> AlignResult:
    """Baseline: 3D PCC vs frame 0, same as FLIMageAlignment.Align_3d_array."""

    def estimator(ref, query):
        return estimate_shift_3d(
            ref,
            query,
            upsample_factor=upsample_factor,
            median_filter=median_filter,
        )

    return _align_series(stack, estimator, reference="first", shift_mode="fourier")


def align_xy_z_split(stack: np.ndarray, *, upsample_factor: int = 10) -> AlignResult:
    """XY from max projection, Z from ZX projection after XY correction."""

    def estimator(ref, query):
        return estimate_shift_xy_z_split(ref, query, upsample_factor=upsample_factor)

    return _align_series(stack, estimator, reference="first", shift_mode="spline")


def align_smoothed(
    stack: np.ndarray,
    *,
    upsample_factor: int = 4,
    savgol_window: int = 5,
    savgol_poly: int = 2,
) -> AlignResult:
    """Traditional shifts with Savitzky-Golay temporal smoothing."""
    shifts, _ = align_traditional(stack, upsample_factor=upsample_factor)
    n = shifts.shape[0]
    window = min(savgol_window, n if n % 2 == 1 else n - 1)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window -= 1
    smoothed = np.zeros_like(shifts)
    for axis in range(3):
        smoothed[:, axis] = savgol_filter(
            shifts[:, axis],
            window_length=window,
            polyorder=min(savgol_poly, window - 1),
        )
    # Keep frame 0 as reference (Savitzky-Golay can pull it away from zero).
    smoothed -= smoothed[0]
    aligned = align_stack_from_shifts(stack, smoothed, shift_mode="fourier")
    return smoothed, aligned


def align_structure_laplacian(stack: np.ndarray, *, upsample_factor: int = 10) -> AlignResult:
    """Correlate Laplacian structure images; apply shift to raw intensity."""

    def estimator(ref, query):
        ref_s = structure_image_3d(ref)
        query_s = structure_image_3d(query)
        return estimate_shift_3d(
            ref,
            query,
            upsample_factor=upsample_factor,
            ref_for_corr=ref_s,
            query_for_corr=query_s,
        )

    return _align_series(stack, estimator, reference="first", shift_mode="spline")


def align_adjacent_cumulative(stack: np.ndarray, *, upsample_factor: int = 10) -> AlignResult:
    """Pairwise t vs t-1 shifts accumulated over time."""

    def estimator(ref, query):
        return estimate_shift_3d(ref, query, upsample_factor=upsample_factor)

    return _align_series(stack, estimator, reference="adjacent", shift_mode="spline")


def align_high_upsample(stack: np.ndarray, *, upsample_factor: int = 20) -> AlignResult:
    """Higher sub-pixel upsampling with spline interpolation."""

    def estimator(ref, query):
        return estimate_shift_3d(ref, query, upsample_factor=upsample_factor)

    return _align_series(stack, estimator, reference="first", shift_mode="spline")


def align_roi_crop(
    stack: np.ndarray,
    roi_spec: RoiCropSpec,
    *,
    upsample_factor: int = 10,
) -> AlignResult:
    """Estimate shift from a small ROI around a spine; apply to full volume."""

    def estimator(ref, query):
        return estimate_shift_3d(ref, query, upsample_factor=upsample_factor)

    return _align_series(
        stack,
        estimator,
        reference="first",
        shift_mode="spline",
        roi_spec=roi_spec,
    )


def align_roi_crop_xy_z_split(
    stack: np.ndarray,
    roi_spec: RoiCropSpec,
    *,
    upsample_factor: int = 10,
) -> AlignResult:
    """ROI-cropped XY/Z split alignment."""

    def estimator(ref, query):
        return estimate_shift_xy_z_split(ref, query, upsample_factor=upsample_factor)

    return _align_series(
        stack,
        estimator,
        reference="first",
        shift_mode="spline",
        roi_spec=roi_spec,
    )


METHOD_REGISTRY: Dict[str, Callable[..., AlignResult]] = {
    "01_traditional": align_traditional,
    "02_xy_z_split": align_xy_z_split,
    "03_smoothed": align_smoothed,
    "04_structure_laplacian": align_structure_laplacian,
    "05_adjacent_cumulative": align_adjacent_cumulative,
    "06_high_upsample": align_high_upsample,
}


def run_all_methods(
    stack: np.ndarray,
    roi_spec: Optional[RoiCropSpec] = None,
) -> Dict[str, AlignResult]:
    """Run all registered methods plus optional ROI variants."""
    results: Dict[str, AlignResult] = {}
    for name, func in METHOD_REGISTRY.items():
        results[name] = func(stack)

    if roi_spec is not None:
        results["07_roi_crop"] = align_roi_crop(stack, roi_spec)
        results["08_roi_crop_xy_z_split"] = align_roi_crop_xy_z_split(stack, roi_spec)

    return results


def run_selected_methods(
    stack: np.ndarray,
    method_names: List[str],
) -> Dict[str, AlignResult]:
    """Run only the requested alignment methods."""
    results: Dict[str, AlignResult] = {}
    for name in method_names:
        if name not in METHOD_REGISTRY:
            raise KeyError(f"Unknown alignment method: {name}")
        results[name] = METHOD_REGISTRY[name](stack)
    return results


def shifts_to_um(
    shifts: np.ndarray,
    z_um: float,
    y_um: float,
    x_um: float,
) -> np.ndarray:
    """Convert pixel shifts [T,3] to microns (Z, Y, X)."""
    scale = np.array([z_um, y_um, x_um], dtype=np.float64)
    return shifts * scale


def alignment_mse_vs_ref(
    aligned_stack: np.ndarray,
    ref_index: int = 0,
    intensity_threshold: float = 5.0,
    *,
    roi_bounds: Optional[Tuple[int, int, int, int, int, int]] = None,
    exclude_shift_margin_px: int = 0,
) -> float:
    """Masked MSE of max projection vs reference frame (lower is better)."""
    ref_vol = aligned_stack[ref_index]
    if roi_bounds is not None:
        z0, z1, y0, y1, x0, x1 = roi_bounds
        ref_mp = np.max(ref_vol[z0:z1, y0:y1, x0:x1], axis=0).astype(np.float64)
    else:
        ref_mp = np.max(ref_vol, axis=0).astype(np.float64)

    mask = ref_mp > intensity_threshold
    if exclude_shift_margin_px > 0:
        m = exclude_shift_margin_px
        inner = np.zeros_like(mask, dtype=bool)
        inner[m:-m, m:-m] = True
        mask &= inner
    if mask.sum() < 100:
        return float("nan")
    errors = []
    for t in range(aligned_stack.shape[0]):
        vol = aligned_stack[t]
        if roi_bounds is not None:
            z0, z1, y0, y1, x0, x1 = roi_bounds
            mp = np.max(vol[z0:z1, y0:y1, x0:x1], axis=0).astype(np.float64)
        else:
            mp = np.max(vol, axis=0).astype(np.float64)
        diff = (mp - ref_mp)[mask]
        errors.append(np.mean(diff ** 2))
    return float(np.mean(errors))


def central_crop_3d(volume: np.ndarray, margin_yx: int) -> np.ndarray:
    """Crop XY margins before correlation to reduce edge artifacts."""
    if margin_yx <= 0:
        return volume
    m = margin_yx
    h, w = volume.shape[1], volume.shape[2]
    if h <= 2 * m or w <= 2 * m:
        return volume
    return volume[:, m:-m, m:-m]
