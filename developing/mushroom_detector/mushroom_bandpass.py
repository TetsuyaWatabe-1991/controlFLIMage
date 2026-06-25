# -*- coding: utf-8 -*-
"""Band-pass gates and tier scores for mushroom spine morphology."""

from __future__ import annotations

from typing import Any

# Hard accept bands (inclusive). Calibrated from 20260515 manual ratings (n=156).
HEAD_VOL_UM3_BAND = (0.25, 1.00)
# nnU-Net class dendrite distance (shaft_to_head_um from nearest_shaft_anchor_xy).
SHAFT_TO_HEAD_UM_BAND = (0.50, 2.50)
# Legacy RESPAN labeled_dendrites metric (head_euclidean_dist_to_dend).
HEAD_TO_DEND_UM_BAND = (0.50, 1.10)
HEAD_AREA_UM2_BAND = (0.18, 0.60)
SEG_AREA_UM2_BAND = (0.20, 0.50)

# Trapezoid tier bands: (hard_min, soft_min, soft_max, hard_max).
# Score peaks between soft_min and soft_max; penalize both tails.
HEAD_VOL_TIER_BAND = (0.25, 0.40, 0.75, 1.05)
HEAD_TO_DEND_TIER_BAND = (0.45, 0.70, 1.00, 1.25)
HEAD_AREA_TIER_BAND = (0.18, 0.28, 0.42, 0.58)
SEG_AREA_TIER_BAND = (0.18, 0.26, 0.38, 0.50)


def value_in_band(value: float, band: tuple[float, float]) -> bool:
    """Return True when value lies inside the inclusive (min, max) band."""
    return band[0] <= value <= band[1]


def band_reject_reason(
    value: float,
    band: tuple[float, float],
    label: str,
    unit: str,
) -> str:
    """Human-readable reject reason when value is outside band."""
    if value < band[0]:
        return f"{label} {value:.3f} {unit} (< {band[0]} {unit})"
    if value > band[1]:
        return f"{label} {value:.3f} {unit} (> {band[1]} {unit})"
    return ""


def band_tier_score(
    value: float,
    band: tuple[float, float, float, float],
    *,
    prefer_lower: bool = False,
) -> float:
    """
    Map a scalar to [-1, 2] with a plateau between soft bounds.

    By default the score peaks when soft_min <= value <= soft_max.
    With prefer_lower=True the score still peaks in the soft band but the
    ramp above soft_max falls off faster (for distance-like features).
    """
    hard_min, soft_min, soft_max, hard_max = band
    if value <= hard_min or value >= hard_max:
        return -1.0
    if soft_min <= value <= soft_max:
        return 2.0
    if value < soft_min:
        return -1.0 + 3.0 * (value - hard_min) / max(soft_min - hard_min, 1e-9)
    # value > soft_max
    span = max(hard_max - soft_max, 1e-9)
    if prefer_lower:
        return max(-1.0, 2.0 - 3.0 * (value - soft_max) / span)
    return max(-1.0, 2.0 - 2.0 * (value - soft_max) / span)


def format_band(band: tuple[float, float], unit: str) -> str:
    return f"{band[0]:g}-{band[1]:g} {unit}"


def passes_head_to_dend_bandpass(
    row: dict[str, Any],
    *,
    head_to_dend_band: tuple[float, float] = HEAD_TO_DEND_UM_BAND,
) -> tuple[bool, str]:
    """Check head-to-dendrite distance before per-spine geometry is built."""
    head_to_dend_um = float(row["head_euclidean_dist_to_dend"])
    reason = band_reject_reason(head_to_dend_um, head_to_dend_band, "head-to-dendrite", "um")
    return (not reason), reason


def passes_shaft_to_head_bandpass(
    shaft_to_head_um: float,
    *,
    shaft_to_head_band: tuple[float, float] = SHAFT_TO_HEAD_UM_BAND,
) -> tuple[bool, str]:
    """Check nnU-Net class dendrite distance after per-spine geometry is built."""
    reason = band_reject_reason(
        shaft_to_head_um,
        shaft_to_head_band,
        "shaft-to-head",
        "um",
    )
    return (not reason), reason


def passes_pre_geometry_bandpass(
    row: dict[str, Any],
    *,
    head_to_dend_band: tuple[float, float] = HEAD_TO_DEND_UM_BAND,
) -> tuple[bool, str]:
    """Pre-geometry bandpass (legacy RESPAN head-to-dendrite only)."""
    return passes_head_to_dend_bandpass(row, head_to_dend_band=head_to_dend_band)


def passes_seg_area_bandpass(
    seg_area_um2: float,
    *,
    seg_area_band: tuple[float, float] = SEG_AREA_UM2_BAND,
) -> tuple[bool, str]:
    reason = band_reject_reason(seg_area_um2, seg_area_band, "seg area", "um^2")
    return (not reason), reason
