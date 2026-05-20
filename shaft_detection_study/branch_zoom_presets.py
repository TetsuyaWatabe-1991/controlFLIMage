# -*- coding: utf-8 -*-
"""
Low-mag branch segment scoring presets keyed by FLIM acquisition zoom.

Zoom is read from iminfo.State.Acq.zoom (same as FLIMageAlignment.get_xyz_pixel_um).
"""
from __future__ import annotations

from typing import Any

from FLIMageAlignment import get_xyz_pixel_um  # noqa: E402
from FLIMageFileReader2 import FileReader  # noqa: E402

# Tuned on Tetsuya low-mag stacks (2x: CM_1_pos1; 4x: AP5_pos6_256_4x).
ZOOM_PRESETS: dict[int, dict[str, float]] = {
    2: {
        "min_branch_width_um": 1.0,
        "max_branch_width_um": 1.8,
        "dendrite_percentile": 98.0,
        "min_branch_length_um": 20.0,
        "segment_length_um": 15.0,
        "overlap_um": 5.0,
    },
    4: {
        "min_branch_width_um": 0.6,
        "max_branch_width_um": 1.0,
        "dendrite_percentile": 94.0,
        "min_branch_length_um": 20.0,
        "segment_length_um": 15.0,
        "overlap_um": 5.0,
    },
}


def read_flim_acq_info(flim_path: str) -> dict[str, Any]:
    """Load acquisition metadata from a .flim file (lightweight: header only if possible)."""
    iminfo = FileReader()
    iminfo.read_imageFile(flim_path, True)
    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
    zoom = float(iminfo.State.Acq.zoom)
    return {
        "flim_path": flim_path,
        "zoom": zoom,
        "x_um": float(x_um),
        "y_um": float(y_um),
        "z_um": float(z_um),
        "pixels_per_line": int(iminfo.State.Acq.pixelsPerLine),
        "lines_per_frame": int(iminfo.State.Acq.linesPerFrame),
    }


def preset_for_zoom(zoom: float) -> tuple[int, dict[str, float]]:
    """
    Return (canonical_zoom_key, preset dict).
    Uses exact match when zoom is 2 or 4; otherwise nearest low-mag preset.
    """
    z_int = int(round(zoom))
    if z_int in ZOOM_PRESETS:
        return z_int, dict(ZOOM_PRESETS[z_int])

    # Nearest known low-mag zoom (2 or 4)
    keys = sorted(ZOOM_PRESETS.keys())
    nearest = min(keys, key=lambda k: abs(k - zoom))
    return nearest, dict(ZOOM_PRESETS[nearest])


def resolve_scoring_params(
    flim_path: str,
    *,
    use_zoom_presets: bool = True,
    overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Merge zoom-based defaults with optional explicit overrides.

    Returns acq info + resolved scoring kwargs for score_all_branch_segments / run_one.
    """
    acq = read_flim_acq_info(flim_path)
    zoom_key, preset = preset_for_zoom(acq["zoom"])

    resolved = dict(preset)
    resolved["zoom_key"] = zoom_key
    resolved["zoom_read"] = acq["zoom"]
    resolved["x_um"] = acq["x_um"]
    resolved["y_um"] = acq["y_um"]
    resolved["z_um"] = acq["z_um"]

    if not use_zoom_presets:
        resolved["preset_source"] = "manual"
        if overrides:
            resolved.update(overrides)
        return resolved

    resolved["preset_source"] = f"zoom_{zoom_key}"
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                resolved[k] = v
    return resolved


def format_preset_summary(params: dict[str, Any]) -> str:
    src = params.get("preset_source", "?")
    zr = params.get("zoom_read", "?")
    wlo = params.get("min_branch_width_um")
    whi = params.get("max_branch_width_um")
    pct = params.get("dendrite_percentile")
    return (
        f"zoom={zr} ({src})  width=({wlo}, {whi}] um  "
        f"dendrite_pct={pct}  xy_um={params.get('x_um', 0):.3f}"
    )
