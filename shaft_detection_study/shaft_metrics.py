# -*- coding: utf-8 -*-
"""Quantitative metrics for shaft (dendrite) prediction maps."""
from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops_table
from skimage.morphology import skeletonize


def fraction_above(dendrite_zyx: np.ndarray, level: float) -> float:
    arr = np.asarray(dendrite_zyx, dtype=np.float32)
    return float(np.mean(arr > level))


def mip_fraction_above(dendrite_zyx: np.ndarray, level: float) -> float:
    mip = np.max(dendrite_zyx, axis=0)
    return float(np.mean(mip > level))


def connected_components_mip(dendrite_zyx: np.ndarray, level: float) -> tuple[int, float]:
    """Return (n_components, largest_component_area_px) on Z-MIP binary mask."""
    mip = np.max(np.asarray(dendrite_zyx, dtype=np.float32), axis=0)
    binary = mip > level
    if not np.any(binary):
        return 0, 0.0
    labeled = label(binary)
    props = regionprops_table(labeled, properties=("area",))
    areas = np.asarray(props["area"], dtype=np.int64)
    return int(len(areas)), float(np.max(areas))


def skeleton_length_2d_mip(dendrite_zyx: np.ndarray, level: float) -> int:
    mip = np.max(np.asarray(dendrite_zyx, dtype=np.float32), axis=0)
    sk = skeletonize(mip > level)
    return int(np.sum(sk))


def skeleton_length_3d(dendrite_zyx: np.ndarray, level: float) -> int:
    sk = skeletonize(np.asarray(dendrite_zyx, dtype=np.float32) > level)
    return int(np.sum(sk))


def summarize_shaft_map(
    dendrite_zyx: np.ndarray,
    *,
    tag: str = "fused",
    skeleton_thresholds: tuple[float, ...] = (0.2, 0.3, 0.5),
) -> dict[str, float | int | str]:
    """Metrics that depend only on dendrite prediction (not spine ROIs)."""
    d = np.asarray(dendrite_zyx, dtype=np.float32)
    out: dict[str, float | int | str] = {"map_tag": tag}
    out["pred_min"] = float(d.min())
    out["pred_max"] = float(d.max())
    out["pred_mean"] = float(d.mean())
    for thr in (0.1, 0.2, 0.3):
        key = str(thr).replace(".", "_")
        out[f"frac3d_gt_{key}"] = fraction_above(d, thr)
        out[f"frac_mip_gt_{key}"] = mip_fraction_above(d, thr)

    n_comp, largest = connected_components_mip(d, 0.2)
    out["n_components_mip_0_2"] = n_comp
    out["largest_component_px_0_2"] = largest

    for sk_thr in skeleton_thresholds:
        sk_key = str(sk_thr).replace(".", "_")
        out[f"skel2d_len_{sk_key}"] = skeleton_length_2d_mip(d, sk_thr)
        out[f"skel3d_len_{sk_key}"] = skeleton_length_3d(d, sk_thr)

    return out


def compare_raw_vs_fused(
    dendrite_raw: np.ndarray,
    dendrite_fused: np.ndarray,
) -> dict[str, float]:
    """Simple deltas to spot fusion gain on thin branches."""
    return {
        "delta_frac_mip_0_2": mip_fraction_above(dendrite_fused, 0.2)
        - mip_fraction_above(dendrite_raw, 0.2),
        "delta_skel2d_0_3": skeleton_length_2d_mip(dendrite_fused, 0.3)
        - skeleton_length_2d_mip(dendrite_raw, 0.3),
    }
