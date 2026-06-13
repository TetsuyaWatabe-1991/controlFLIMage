# -*- coding: utf-8 -*-
"""
Detect axon bouton / crossing-fiber false positives from dendrite-axis intensity.

Scores each spine candidate using:
  1. Axial brightness continuity along the fitted dendrite axis
  2. Bright-structure axis angle vs dendrite axis (crossing axon)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

DEFAULT_PROFILE_HALF_UM = 5.0
DEFAULT_PROFILE_STEP_UM = 0.15
DEFAULT_STRIP_HALF_WIDTH_UM = 0.35
DEFAULT_BRIGHT_FRAC = 0.45
DEFAULT_MIN_COMPONENT_AREA_PX = 8


def _z_window_indices(z_idx: int, n_z: int, half: int) -> tuple[int, int]:
    z0 = max(0, z_idx - half)
    z1 = min(n_z, z_idx + half + 1)
    return z0, z1


def raw_local_z_mip(
    raw_zyx: np.ndarray,
    z_pix: float,
    *,
    z_half_window: int = 2,
) -> np.ndarray:
    z_idx = int(np.clip(round(z_pix), 0, raw_zyx.shape[0] - 1))
    z0, z1 = _z_window_indices(z_idx, raw_zyx.shape[0], z_half_window)
    return raw_zyx[z0:z1].max(axis=0).astype(np.float64)


def _dendrite_unit_vector(slope: float) -> np.ndarray:
    """Unit vector along dendrite in (x, y) pixel coordinates."""
    dx = 1.0
    dy = slope
    norm = math.hypot(dx, dy)
    return np.array([dx / norm, dy / norm], dtype=np.float64)


def _sample_strip_max(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    tangent_xy: np.ndarray,
    xy_um: float,
    *,
    strip_half_width_um: float,
) -> float:
    """Max intensity in a strip perpendicular to the dendrite tangent."""
    height, width = image.shape
    perp = np.array([-tangent_xy[1], tangent_xy[0]], dtype=np.float64)
    half_px = strip_half_width_um / xy_um
    offsets = np.linspace(-half_px, half_px, max(3, int(2 * half_px) + 1))
    values: list[float] = []
    for off in offsets:
        x = center_x + perp[0] * off
        y = center_y + perp[1] * off
        xi = int(np.clip(round(x), 0, width - 1))
        yi = int(np.clip(round(y), 0, height - 1))
        values.append(float(image[yi, xi]))
    return float(max(values))


def compute_axial_intensity_profile(
    raw_mip: np.ndarray,
    head_x: float,
    head_y: float,
    dend_slope: float,
    xy_um: float,
    *,
    profile_half_um: float = DEFAULT_PROFILE_HALF_UM,
    profile_step_um: float = DEFAULT_PROFILE_STEP_UM,
    strip_half_width_um: float = DEFAULT_STRIP_HALF_WIDTH_UM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample max intensity along the dendrite axis through the head.

    Returns:
        s_um: signed arc-length coordinates (0 = head)
        intensity: sampled values
    """
    tangent = _dendrite_unit_vector(dend_slope)
    n_steps = int(profile_half_um / profile_step_um)
    s_vals: list[float] = []
    i_vals: list[float] = []
    for step in range(-n_steps, n_steps + 1):
        s_um = step * profile_step_um
        s_px = s_um / xy_um
        cx = head_x + tangent[0] * s_px
        cy = head_y + tangent[1] * s_px
        s_vals.append(s_um)
        i_vals.append(
            _sample_strip_max(
                raw_mip,
                cx,
                cy,
                tangent,
                xy_um,
                strip_half_width_um=strip_half_width_um,
            )
        )
    return np.asarray(s_vals, dtype=np.float64), np.asarray(i_vals, dtype=np.float64)


def _extent_above_threshold(
    s_um: np.ndarray,
    intensity: np.ndarray,
    *,
    center_idx: int,
    peak: float,
    bright_frac: float,
    direction: int,
) -> float:
    """Distance from center where intensity stays above bright_frac * peak."""
    threshold = peak * bright_frac
    extent = 0.0
    idx = center_idx
    while 0 <= idx < len(intensity):
        if intensity[idx] < threshold:
            break
        extent = abs(float(s_um[idx] - s_um[center_idx]))
        idx += direction
    return extent


def compute_axial_continuity_metrics(
    s_um: np.ndarray,
    intensity: np.ndarray,
    *,
    bright_frac: float = DEFAULT_BRIGHT_FRAC,
) -> dict[str, float]:
    """Derive bouton-relevant metrics from an axial intensity profile."""
    center_idx = int(np.argmin(np.abs(s_um)))
    peak = float(intensity[center_idx])
    if peak <= 0:
        return {
            "axial_peak_intensity": 0.0,
            "bright_span_um": 0.0,
            "bright_left_um": 0.0,
            "bright_right_um": 0.0,
            "neck_contrast": 0.0,
            "bouton_axis_score": 0.0,
        }

    left_um = _extent_above_threshold(
        s_um, intensity, center_idx=center_idx, peak=peak,
        bright_frac=bright_frac, direction=-1,
    )
    right_um = _extent_above_threshold(
        s_um, intensity, center_idx=center_idx, peak=peak,
        bright_frac=bright_frac, direction=1,
    )
    span_um = left_um + right_um

    shaft_side = s_um < 0
    if shaft_side.any():
        neck_min = float(intensity[shaft_side].min())
    else:
        neck_min = peak
    neck_contrast = max(0.0, (peak - neck_min) / peak)

    # Higher = more bouton-like (long bright run, shallow neck, both sides lit).
    bilateral = min(left_um, right_um)
    bouton_axis_score = (
        0.45 * min(span_um / 3.0, 1.0)
        + 0.30 * min(bilateral / 1.5, 1.0)
        + 0.25 * (1.0 - min(neck_contrast / 0.5, 1.0))
    )

    return {
        "axial_peak_intensity": peak,
        "bright_span_um": span_um,
        "bright_left_um": left_um,
        "bright_right_um": right_um,
        "neck_contrast": neck_contrast,
        "bouton_axis_score": float(np.clip(bouton_axis_score, 0.0, 1.0)),
    }


def compute_bright_structure_axis_angle_deg(
    raw_mip: np.ndarray,
    head_x: float,
    head_y: float,
    dend_slope: float,
    *,
    bright_frac: float = 0.5,
    min_component_area_px: int = DEFAULT_MIN_COMPONENT_AREA_PX,
) -> dict[str, float]:
    """
    Angle between the bright blob PCA axis and the dendrite axis.

    Small angle => structure runs along / across dendrite (axon bouton-like).
    Large angle => protrusion roughly normal to dendrite (mushroom-like).
    """
    yi = int(np.clip(round(head_y), 0, raw_mip.shape[0] - 1))
    xi = int(np.clip(round(head_x), 0, raw_mip.shape[1] - 1))
    peak = float(raw_mip[yi, xi])
    if peak <= 0:
        return {
            "bright_axis_angle_deg": np.nan,
            "bright_structure_score": 0.0,
            "bright_elongation": np.nan,
        }

    mask = raw_mip >= peak * bright_frac
    if not mask.any():
        return {
            "bright_axis_angle_deg": np.nan,
            "bright_structure_score": 0.0,
            "bright_elongation": np.nan,
        }

    from scipy import ndimage

    labeled, n_lab = ndimage.label(mask)
    head_label = int(labeled[yi, xi])
    if head_label == 0:
        return {
            "bright_axis_angle_deg": np.nan,
            "bright_structure_score": 0.0,
            "bright_elongation": np.nan,
        }

    ys, xs = np.where(labeled == head_label)
    if len(xs) < min_component_area_px:
        return {
            "bright_axis_angle_deg": np.nan,
            "bright_structure_score": 0.0,
            "bright_elongation": np.nan,
        }

    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    coords -= coords.mean(axis=0)
    cov = coords.T @ coords / max(len(coords) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.sort(np.maximum(eigvals, 1e-6))
    major = eigvecs[:, int(np.argmax(eigvals))]
    struct_angle = math.degrees(math.atan2(major[1], major[0]))
    dend_angle = math.degrees(math.atan(dend_slope))
    delta = abs(struct_angle - dend_angle) % 180.0
    delta = min(delta, 180.0 - delta)
    elongation = float(math.sqrt(eigvals[-1] / eigvals[0]))

    # 0 deg = parallel (bouton-like), 90 deg = protruding (mushroom-like)
    bright_structure_score = float(np.clip(1.0 - delta / 90.0, 0.0, 1.0))
    return {
        "bright_axis_angle_deg": float(delta),
        "bright_structure_score": bright_structure_score,
        "bright_elongation": elongation,
    }


def compute_bouton_scores(
    raw_mip: np.ndarray,
    head_x: float,
    head_y: float,
    dend_slope: float,
    xy_um: float,
    *,
    bright_frac: float = DEFAULT_BRIGHT_FRAC,
) -> dict[str, float]:
    """Combined bouton / crossing-axon scores for one spine candidate."""
    s_um, intensity = compute_axial_intensity_profile(
        raw_mip,
        head_x,
        head_y,
        dend_slope,
        xy_um,
    )
    axial = compute_axial_continuity_metrics(
        s_um,
        intensity,
        bright_frac=bright_frac,
    )
    axis = compute_bright_structure_axis_angle_deg(
        raw_mip,
        head_x,
        head_y,
        dend_slope,
        bright_frac=bright_frac,
    )
    combined = float(np.clip(
        0.65 * axial["bouton_axis_score"] + 0.35 * axis["bright_structure_score"],
        0.0,
        1.0,
    ))
    return {
        **axial,
        **axis,
        "bouton_combined_score": combined,
    }


def passes_bouton_reject(
    scores: dict[str, float],
    *,
    combined_threshold: float = 0.55,
    span_threshold_um: float = 2.5,
    bilateral_min_um: float = 0.8,
    crossing_angle_deg: float = 12.0,
    crossing_span_um: float = 0.70,
    crossing_elongation: float = 2.8,
    crossing_structure_score: float = 0.85,
    crossing_neck_contrast: float = 0.95,
) -> tuple[bool, str]:
    """
    Return True when candidate looks like axon bouton / crossing fiber.

    True means REJECT (bouton-like).
    """
    span = scores.get("bright_span_um", 0.0)
    left = scores.get("bright_left_um", 0.0)
    right = scores.get("bright_right_um", 0.0)
    combined = scores.get("bouton_combined_score", 0.0)
    bilateral = min(left, right)
    angle = scores.get("bright_axis_angle_deg", np.nan)
    elongation = scores.get("bright_elongation", np.nan)
    structure = scores.get("bright_structure_score", 0.0)
    neck_contrast = scores.get("neck_contrast", 0.0)
    reasons: list[str] = []
    if combined >= combined_threshold:
        reasons.append(f"bouton_combined {combined:.2f} (>= {combined_threshold})")
    if span >= span_threshold_um and bilateral >= bilateral_min_um:
        reasons.append(
            f"bright_span {span:.2f} um with bilateral >= {bilateral_min_um} um"
        )
    if (
        np.isfinite(angle)
        and np.isfinite(elongation)
        and angle <= crossing_angle_deg
        and span <= crossing_span_um
        and elongation >= crossing_elongation
        and structure >= crossing_structure_score
        and neck_contrast >= crossing_neck_contrast
    ):
        reasons.append(
            f"crossing fiber (angle {angle:.1f} deg, span {span:.2f} um, "
            f"elong {elongation:.2f}, neck {neck_contrast:.2f})"
        )
    if reasons:
        return True, "; ".join(reasons)
    return False, ""
