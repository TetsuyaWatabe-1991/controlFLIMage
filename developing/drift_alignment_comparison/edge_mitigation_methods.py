# -*- coding: utf-8 -*-
"""Edge-artifact mitigation variants for drift alignment comparison."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from align_methods import (
    AlignResult,
    RoiCropSpec,
    _align_series,
    align_adjacent_cumulative,
    align_roi_crop,
    align_roi_crop_xy_z_split,
    align_stack_from_shifts,
    align_traditional,
    central_crop_3d,
    crop_zyx,
    estimate_shift_3d,
    estimate_shift_xy_z_split,
)


def align_spline_constant(
    stack: np.ndarray,
    *,
    upsample_factor: int = 10,
) -> AlignResult:
    """Same shift estimation as traditional; apply with zero-filled spline."""

    def estimator(ref, query):
        return estimate_shift_3d(ref, query, upsample_factor=upsample_factor)

    return _align_series(stack, estimator, reference="first", shift_mode="constant")


def align_central_margin(
    stack: np.ndarray,
    *,
    margin_yx: int = 15,
    upsample_factor: int = 10,
) -> AlignResult:
    """Estimate shift on central crop (exclude image borders)."""

    def estimator(ref, query):
        ref_c = central_crop_3d(ref, margin_yx)
        query_c = central_crop_3d(query, margin_yx)
        return estimate_shift_3d(ref_c, query_c, upsample_factor=upsample_factor)

    return _align_series(stack, estimator, reference="first", shift_mode="constant")


def align_two_stage_roi(
    stack: np.ndarray,
    roi_spec: RoiCropSpec,
    *,
    upsample_factor: int = 10,
) -> AlignResult:
    """Global constant-fill align, then add ROI-local shifts (AlignSmallRegion pattern)."""
    global_shifts, globally_aligned = align_spline_constant(stack, upsample_factor=upsample_factor)

    n = stack.shape[0]
    local_shifts = np.zeros((n, 3), dtype=np.float64)
    for t in range(n):
        ref_crop, _ = crop_zyx(globally_aligned[0], roi_spec)
        query_crop, _ = crop_zyx(globally_aligned[t], roi_spec)
        delta, _ = estimate_shift_3d(ref_crop, query_crop, upsample_factor=upsample_factor)
        local_shifts[t] = delta

    combined = global_shifts + local_shifts
    aligned = align_stack_from_shifts(stack, combined, shift_mode="constant")
    return combined, aligned


def motor_shifts_from_positions(
    motor_positions: List[Tuple[float, float, float]],
    z_um: float,
    y_um: float,
    x_um: float,
) -> np.ndarray:
    """Convert motor XYZ (um) per frame to pixel shifts vs frame 0."""
    m0 = np.array(motor_positions[0], dtype=np.float64)
    shifts = np.zeros((len(motor_positions), 3), dtype=np.float64)
    for t, pos in enumerate(motor_positions):
        delta_um = np.array(pos, dtype=np.float64) - m0
        shifts[t] = np.array([
            delta_um[2] / z_um,
            delta_um[1] / y_um,
            delta_um[0] / x_um,
        ])
    return shifts


def align_motor_only(
    stack: np.ndarray,
    motor_positions: List[Tuple[float, float, float]],
    z_um: float,
    y_um: float,
    x_um: float,
) -> AlignResult:
    """Apply shifts derived from stage motor metadata only."""
    shifts = motor_shifts_from_positions(motor_positions, z_um, y_um, x_um)
    aligned = align_stack_from_shifts(stack, shifts, shift_mode="constant")
    return shifts, aligned


def align_motor_plus_roi_refine(
    stack: np.ndarray,
    motor_positions: List[Tuple[float, float, float]],
    roi_spec: RoiCropSpec,
    z_um: float,
    y_um: float,
    x_um: float,
    *,
    upsample_factor: int = 10,
) -> AlignResult:
    """Motor-based coarse shift + ROI phase-correlation refinement."""
    motor_shifts, motor_aligned = align_motor_only(
        stack, motor_positions, z_um, y_um, x_um
    )
    refine = np.zeros_like(motor_shifts)
    for t in range(stack.shape[0]):
        ref_crop, _ = crop_zyx(motor_aligned[0], roi_spec)
        query_crop, _ = crop_zyx(motor_aligned[t], roi_spec)
        delta, _ = estimate_shift_3d(ref_crop, query_crop, upsample_factor=upsample_factor)
        refine[t] = delta
    combined = motor_shifts + refine
    aligned = align_stack_from_shifts(stack, combined, shift_mode="constant")
    return combined, aligned


def run_edge_mitigation_methods(
    stack: np.ndarray,
    roi_spec: RoiCropSpec,
    motor_positions: Optional[List[Tuple[float, float, float]]] = None,
    z_um: float = 1.0,
    y_um: float = 0.13,
    x_um: float = 0.13,
) -> Dict[str, AlignResult]:
    """Run baseline + edge-mitigation alignment variants."""
    results: Dict[str, AlignResult] = {
        "A_baseline_fourier": align_traditional(stack),
        "B_spline_constant": align_spline_constant(stack),
        "C_roi_corr_spline": align_roi_crop(stack, roi_spec),
        "D_roi_xy_z_split": align_roi_crop_xy_z_split(stack, roi_spec),
        "E_roi_adjacent": _align_series(
            stack,
            lambda ref, q: estimate_shift_3d(ref, q, upsample_factor=10),
            reference="adjacent",
            shift_mode="constant",
            roi_spec=roi_spec,
        ),
        "F_central_margin": align_central_margin(stack, margin_yx=15),
        "G_two_stage_roi": align_two_stage_roi(stack, roi_spec),
    }
    if motor_positions is not None and len(motor_positions) == stack.shape[0]:
        results["H_motor_only"] = align_motor_only(
            stack, motor_positions, z_um, y_um, x_um
        )
        results["I_motor_roi_refine"] = align_motor_plus_roi_refine(
            stack, motor_positions, roi_spec, z_um, y_um, x_um
        )
    return results
