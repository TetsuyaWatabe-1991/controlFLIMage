# -*- coding: utf-8 -*-
"""DeepD3 shaft (dendrite) prediction only — no spine ROI pipeline."""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

_CONTROLFLIMAGE = Path(__file__).resolve().parents[1]
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

from deepd3.core.analysis import Stack
from deepd3_spine_head_detector import (
    fuse_dendrite_with_image_mask,
    local_z_mip_stack,
    preprocess_stack_for_thin_branches,
)
from FLIMageAlignment import get_xyz_pixel_um
from FLIMageFileReader2 import FileReader


def load_flim_stack(flim_path: str, ch_1or2: int | None = None) -> tuple[np.ndarray, float, float]:
    """Load ZYX stack and pixel sizes (um) from a .flim file."""
    iminfo = FileReader()
    iminfo.read_imageFile(flim_path, True)
    if ch_1or2 in (1, 2):
        zyx = np.array(iminfo.image)[:, :, ch_1or2 - 1, :, :, :].sum(axis=(1, 4))
    else:
        zyx = np.array(iminfo.image).sum(axis=(1, 2, 5))
    x_um, _, z_um = get_xyz_pixel_um(iminfo)
    return np.asarray(zyx, dtype=np.float32), float(x_um), float(z_um)


def raw_z_mip(zyx: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(zyx, dtype=np.float32), axis=0)


def predict_deepd3_stack(
    zyx: np.ndarray,
    xy_pixel_um: float,
    z_pixel_um: float,
    model_path: str,
    stack_preprocess: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Run DeepD3 plane-by-plane inference; return (shaft, spine) ZYX predictions."""
    stack_input = preprocess_stack_for_thin_branches(zyx, mode=stack_preprocess)
    buf = BytesIO()
    tifffile.imwrite(buf, stack_input)
    s = Stack(
        buf,
        dimensions=dict(xy=xy_pixel_um, z=z_pixel_um),
    )
    s.predictWholeImage(model_path)
    shaft = np.asarray(s.prediction[..., 0], dtype=np.float32)
    spine = np.asarray(s.prediction[..., 1], dtype=np.float32)
    return shaft, spine


def predict_dendrite_raw(
    zyx: np.ndarray,
    xy_pixel_um: float,
    z_pixel_um: float,
    model_path: str,
    stack_preprocess: str = "none",
) -> np.ndarray:
    """Run DeepD3 inference; return dendrite (shaft) channel before fusion."""
    shaft, _ = predict_deepd3_stack(
        zyx, xy_pixel_um, z_pixel_um, model_path, stack_preprocess=stack_preprocess
    )
    return shaft


def apply_shaft_fusion(
    dendrite_pred_raw: np.ndarray,
    raw_zyx: np.ndarray,
    *,
    enhance: bool,
    image_fusion_percentile: float = 92.0,
    image_fusion_weight: float = 0.5,
    dendrite_closing_iterations: int = 1,
) -> np.ndarray:
    if not enhance:
        return dendrite_pred_raw.copy()
    return fuse_dendrite_with_image_mask(
        dendrite_pred_raw,
        raw_zyx,
        image_percentile=image_fusion_percentile,
        fusion_weight=image_fusion_weight,
        closing_iterations=int(dendrite_closing_iterations),
    )


def run_shaft_combo(
    zyx: np.ndarray,
    xy_pixel_um: float,
    z_pixel_um: float,
    model_path: str,
    combo: dict[str, Any],
    dendrite_pred_raw_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    Run one parameter combination.

    combo keys: stack_preprocess, enhance_thin_branches,
    image_fusion_percentile, image_fusion_weight, dendrite_closing_iterations
    """
    preprocess = combo.get("stack_preprocess", "none")
    if dendrite_pred_raw_cache is not None and preprocess in dendrite_pred_raw_cache:
        dendrite_raw = dendrite_pred_raw_cache[preprocess]
    else:
        dendrite_raw = predict_dendrite_raw(
            zyx, xy_pixel_um, z_pixel_um, model_path, stack_preprocess=preprocess
        )
        if dendrite_pred_raw_cache is not None:
            dendrite_pred_raw_cache[preprocess] = dendrite_raw

    dendrite_fused = apply_shaft_fusion(
        dendrite_raw,
        zyx,
        enhance=bool(combo.get("enhance_thin_branches", False)),
        image_fusion_percentile=float(combo.get("image_fusion_percentile", 92.0)),
        image_fusion_weight=float(combo.get("image_fusion_weight", 0.5)),
        dendrite_closing_iterations=int(combo.get("dendrite_closing_iterations", 1)),
    )
    return {
        "dendrite_pred_raw": dendrite_raw,
        "dendrite_pred_fused": dendrite_fused,
        "stack_preprocess": preprocess,
    }
