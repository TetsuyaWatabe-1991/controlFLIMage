# -*- coding: utf-8 -*-
"""
Parameter grids for shaft (dendrite) identification tuning.

Shaft-only knobs (change dendrite pred / skeleton, NOT spine ROI count):
  - stack_preprocess: input to DeepD3 ('none' | 'median' | 'tophat_clahe')
  - enhance_thin_branches: fuse raw image mask into dendrite pred
  - image_fusion_percentile, image_fusion_weight, dendrite_closing_iterations

Spine ROI knobs (NOT swept here): roi_areaThreshold, roi_peakThreshold, ...

Post-hoc skeleton display (metrics only in sweep): dendrite_skeleton_threshold
"""
from __future__ import annotations

import itertools
from typing import Any


def combo_id(combo: dict[str, Any]) -> str:
    """Short filesystem-safe label."""
    pp = combo.get("stack_preprocess", "none")
    if not combo.get("enhance_thin_branches", False):
        return f"pp_{pp}_nofusion"
    p = int(combo.get("image_fusion_percentile", 92))
    w = int(round(float(combo.get("image_fusion_weight", 0.5)) * 100))
    c = int(combo.get("dendrite_closing_iterations", 1))
    return f"pp_{pp}_fus_p{p}_w{w}_c{c}"


def _dict_product(**kwargs: Any) -> list[dict[str, Any]]:
    keys = list(kwargs.keys())
    values = [kwargs[k] if isinstance(kwargs[k], (list, tuple)) else [kwargs[k]] for k in keys]
    out = []
    for combo_vals in itertools.product(*values):
        out.append(dict(zip(keys, combo_vals)))
    return out


def grid_quick() -> list[dict[str, Any]]:
    """Few combinations for a first look (~8 runs + preprocess cache)."""
    combos = []
    for pp in ("none", "tophat_clahe"):
        combos.append(
            {
                "stack_preprocess": pp,
                "enhance_thin_branches": False,
            }
        )
    for percentile, weight in ((92, 0.55), (90, 0.55), (88, 0.5)):
        combos.append(
            {
                "stack_preprocess": "tophat_clahe",
                "enhance_thin_branches": True,
                "image_fusion_percentile": float(percentile),
                "image_fusion_weight": float(weight),
                "dendrite_closing_iterations": 1,
            }
        )
    combos.append(
        {
            "stack_preprocess": "median",
            "enhance_thin_branches": True,
            "image_fusion_percentile": 92.0,
            "image_fusion_weight": 0.55,
            "dendrite_closing_iterations": 1,
        }
    )
    return combos


def grid_preprocess() -> list[dict[str, Any]]:
    """All preprocess modes without fusion (3 DeepD3 runs per file)."""
    return [
        {"stack_preprocess": pp, "enhance_thin_branches": False}
        for pp in ("none", "median", "tophat_clahe")
    ]


def grid_fusion(
    stack_preprocess: str = "tophat_clahe",
) -> list[dict[str, Any]]:
    """Fusion sweep with fixed preprocess (1 DeepD3 run per file if cached)."""
    base = {"stack_preprocess": stack_preprocess, "enhance_thin_branches": True}
    combos = []
    for p, w, c in itertools.product(
        (88.0, 90.0, 92.0, 94.0),
        (0.4, 0.55, 0.7),
        (0, 1, 2),
    ):
        cmb = dict(base)
        cmb["image_fusion_percentile"] = p
        cmb["image_fusion_weight"] = w
        cmb["dendrite_closing_iterations"] = c
        combos.append(cmb)
    return combos


def grid_full() -> list[dict[str, Any]]:
    """Preprocess x a small fusion subset (~15 DeepD3 runs worst-case)."""
    combos = grid_preprocess()
    for pp in ("tophat_clahe", "median"):
        for p, w in itertools.product((88.0, 92.0), (0.45, 0.6)):
            combos.append(
                {
                    "stack_preprocess": pp,
                    "enhance_thin_branches": True,
                    "image_fusion_percentile": p,
                    "image_fusion_weight": w,
                    "dendrite_closing_iterations": 1,
                }
            )
    return combos


GRID_BUILDERS = {
    "quick": grid_quick,
    "preprocess": grid_preprocess,
    "fusion": grid_fusion,
    "full": grid_full,
}


def get_parameter_grid(name: str, **kwargs: Any) -> list[dict[str, Any]]:
    if name not in GRID_BUILDERS:
        raise ValueError(f"Unknown grid {name!r}. Choose from: {list(GRID_BUILDERS)}")
    builder = GRID_BUILDERS[name]
    if name == "fusion":
        return builder(stack_preprocess=kwargs.get("stack_preprocess", "tophat_clahe"))
    return builder()


def shaft_parameter_help() -> str:
    return """
Shaft identification parameters (this study folder):

  Affects DeepD3 dendrite map (shaft pred):
    stack_preprocess          — before inference
    enhance_thin_branches     — post-inference image fusion
    image_fusion_percentile   — lower -> more thin branches from raw
    image_fusion_weight       — higher → stronger fusion
    dendrite_closing_iterations — bridge gaps (1–2)

  Does NOT change dendrite pred (spine / tracing only):
    roi_areaThreshold, roi_peakThreshold
    dendrite_skeleton_threshold — only skeleton overlay after ROIs

  Suggested workflow:
    1) grid preprocess  -> pick best stack_preprocess
    2) grid fusion --stack-preprocess <chosen>
    3) Compare PNGs + CSV metrics (frac_mip_gt_0_2, skel2d_len_0_3)
"""
