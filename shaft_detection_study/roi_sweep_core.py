# -*- coding: utf-8 -*-
"""Run DeepD3 once, rebuild ROIs per parameter set (no per-spine ini/png)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

_CONTROLFLIMAGE = Path(__file__).resolve().parents[1]
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

from deepd3.core.analysis import ROI3D_Creator, Stack
from deepd3_spine_head_detector import (
    fuse_dendrite_with_image_mask,
    local_z_mip_stack,
    preprocess_stack_for_thin_branches,
)
from FLIMageAlignment import get_xyz_pixel_um
from FLIMageFileReader2 import FileReader


@dataclass
class CachedDeepD3Prediction:
    S: Any
    zyx_raw: np.ndarray
    xy_pixel_um: float
    z_pixel_um: float


def load_flim_zyx(flim_path: str, ch_1or2: int | None = None) -> tuple[np.ndarray, float, float]:
    iminfo = FileReader()
    iminfo.read_imageFile(flim_path, True)
    if ch_1or2 in (1, 2):
        zyx = np.array(iminfo.image)[:, :, ch_1or2 - 1, :, :, :].sum(axis=(1, 4))
    else:
        zyx = np.array(iminfo.image).sum(axis=(1, 2, 5))
    x_um, _, z_um = get_xyz_pixel_um(iminfo)
    return np.asarray(zyx, dtype=np.float32), float(x_um), float(z_um)


def run_deepd3_prediction(
    flim_path: str,
    model_path: str,
    *,
    use_local_z_mip: bool = True,
    local_z_mip_radius: int = 1,
    stack_preprocess: str = "tophat_clahe",
    enhance_thin_branches: bool = False,
    image_fusion_percentile: float = 94.0,
    image_fusion_weight: float = 0.4,
    dendrite_closing_iterations: int = 1,
    ch_1or2: int | None = None,
) -> CachedDeepD3Prediction:
    """Single DeepD3 inference; ROI params applied later."""
    zyx_raw, xy_um, z_um = load_flim_zyx(flim_path, ch_1or2=ch_1or2)
    if use_local_z_mip:
        zyx_for_deepd3 = local_z_mip_stack(zyx_raw, radius=int(local_z_mip_radius))
    else:
        zyx_for_deepd3 = zyx_raw

    stack_input = preprocess_stack_for_thin_branches(zyx_for_deepd3, mode=stack_preprocess)
    buf = BytesIO()
    tifffile.imwrite(buf, stack_input)
    s = Stack(buf, dimensions=dict(xy=xy_um, z=z_um))
    s.predictWholeImage(model_path)

    if enhance_thin_branches:
        fused = fuse_dendrite_with_image_mask(
            s.prediction[..., 0],
            zyx_raw,
            image_percentile=image_fusion_percentile,
            fusion_weight=image_fusion_weight,
            closing_iterations=int(dendrite_closing_iterations),
        )
        s.prediction[..., 0] = fused

    return CachedDeepD3Prediction(S=s, zyx_raw=zyx_raw, xy_pixel_um=xy_um, z_pixel_um=z_um)


def build_roi_maps(
    cache: CachedDeepD3Prediction,
    roi_params: dict[str, Any],
    *,
    roi_mode: str = "thresholded",
) -> tuple[Any, int]:
    """Return (ROI3D_Creator, n_labels)."""
    s = cache.S
    r = ROI3D_Creator(
        dendrite_prediction=s.prediction[..., 0],
        spine_prediction=s.prediction[..., 1],
        mode=roi_mode,
        areaThreshold=roi_params["roi_areaThreshold"],
        peakThreshold=roi_params["roi_peakThreshold"],
        seedDelta=roi_params.get("roi_seedDelta", 0.1),
        distanceToSeed=roi_params.get("roi_distanceToSeed", 10),
        dimensions=dict(xy=cache.xy_pixel_um, z=cache.z_pixel_um),
    )
    n = r.create(
        int(roi_params["min_roi_size"]),
        int(roi_params["max_roi_size"]),
        int(roi_params.get("min_planes", 1)),
    )
    return r, int(n)


def summarize_rois(
    cache: CachedDeepD3Prediction,
    r: Any,
    spine_filters: dict[str, float],
) -> dict[str, Any]:
    """prop_dict + cand_spines counts (same filters as mushroom pipeline)."""
    prop_table = regionprops_table(
        r.roi_map,
        properties=["label", "centroid", "num_pixels", "equivalent_diameter_area"],
    )
    if len(prop_table["label"]) == 0:
        return {
            "n_roi": 0,
            "n_cand": 0,
            "prop_dict": {},
            "cand_spines": pd.DataFrame(),
        }

    prop_dict: dict[str, dict] = {}
    intensity_list = []
    for nth in range(len(prop_table["label"])):
        label = str(prop_table["label"][nth])
        prop_dict[label] = {
            "z": round(prop_table["centroid-0"][nth]),
            "y": round(prop_table["centroid-1"][nth]),
            "x": round(prop_table["centroid-2"][nth]),
            "num_pixels": prop_table["num_pixels"][nth],
            "intensity": float(
                cache.zyx_raw[r.roi_map == prop_table["label"][nth]].sum()
            ),
            "equivalent_diameter_area": prop_table["equivalent_diameter_area"][nth],
        }
        intensity_list.append(prop_dict[label]["intensity"])

    upper_px = np.percentile(
        prop_table["num_pixels"], spine_filters["upper_spine_pixel_percentile"]
    )
    lower_px = np.percentile(
        prop_table["num_pixels"], spine_filters["lower_spine_pixel_percentile"]
    )
    upper_int = np.percentile(
        intensity_list, spine_filters["upper_spine_intensity_percentile"]
    )
    lower_int = np.percentile(
        intensity_list, spine_filters["lower_spine_intensity_percentile"]
    )

    prop_df = pd.DataFrame.from_dict(prop_dict, orient="index")
    cand = prop_df[
        (prop_df["num_pixels"] <= upper_px)
        & (prop_df["num_pixels"] >= lower_px)
        & (prop_df["intensity"] <= upper_int)
        & (prop_df["intensity"] >= lower_int)
    ]
    return {
        "n_roi": len(prop_dict),
        "n_cand": len(cand),
        "prop_dict": prop_dict,
        "cand_spines": cand,
    }
