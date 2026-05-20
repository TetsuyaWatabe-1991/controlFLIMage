# -*- coding: utf-8 -*-
"""PNG outputs for shaft parameter comparison."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_triple_mip_png(
    raw_mip: np.ndarray,
    dendrite_raw_zyx: np.ndarray,
    dendrite_fused_zyx: np.ndarray,
    savepath: str,
    *,
    title_suffix: str = "",
    raw_vmax_percentile: float = 95.0,
    contour_level: float = 0.2,
) -> None:
    raw_mip = np.asarray(raw_mip, dtype=np.float32)
    shaft_raw_mip = np.max(dendrite_raw_zyx, axis=0)
    shaft_fused_mip = np.max(dendrite_fused_zyx, axis=0)
    if np.any(raw_mip > 0):
        vmax_raw = float(np.percentile(raw_mip[raw_mip > 0], raw_vmax_percentile))
    else:
        vmax_raw = float(raw_mip.max()) if raw_mip.size else 1.0
    vmax_raw = max(vmax_raw, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw_mip, cmap="gray", vmin=0, vmax=vmax_raw)
    axes[0].set_title(f"Raw Z-MIP p{raw_vmax_percentile:g}")
    axes[0].axis("off")

    axes[1].imshow(shaft_raw_mip, cmap="gray", vmin=0, vmax=1.0)
    if contour_level > 0:
        axes[1].contour(shaft_raw_mip, levels=[contour_level], colors="cyan", linewidths=0.4)
    axes[1].set_title("DeepD3 shaft (raw)")
    axes[1].axis("off")

    axes[2].imshow(shaft_fused_mip, cmap="gray", vmin=0, vmax=1.0)
    if contour_level > 0:
        axes[2].contour(shaft_fused_mip, levels=[contour_level], colors="lime", linewidths=0.4)
    axes[2].set_title("Shaft used (fused)")
    axes[2].axis("off")

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=10)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_montage(
    thumbnails: list[np.ndarray],
    labels: list[str],
    savepath: str,
    *,
    ncol: int = 4,
    thumb_size: int = 128,
) -> None:
    """Grid of fused shaft MIPs for quick visual scan."""
    from skimage.transform import resize

    n = len(thumbnails)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.2 * nrow))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat[n:]:
        ax.axis("off")

    for i, (img, lab) in enumerate(zip(thumbnails, labels)):
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        small = resize(arr, (thumb_size, thumb_size), preserve_range=True, anti_aliasing=True)
        axes_flat[i].imshow(small, cmap="gray", vmin=0, vmax=1)
        axes_flat[i].set_title(lab, fontsize=7)
        axes_flat[i].axis("off")

    fig.tight_layout()
    fig.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_single_vs_localzmip_compare(
    raw_mip: np.ndarray,
    shaft_single_zyx: np.ndarray,
    shaft_local_zyx: np.ndarray,
    spine_single_zyx: np.ndarray,
    spine_local_zyx: np.ndarray,
    savepath: str,
    *,
    z_radius: int = 2,
    raw_vmax_percentile: float = 95.0,
    contour_level: float = 0.2,
) -> None:
    """2x3: raw | single-plane shaft/spine MIP | local-Z-MIP shaft/spine MIP."""
    raw_mip = np.asarray(raw_mip, dtype=np.float32)
    if np.any(raw_mip > 0):
        vmax_raw = float(np.percentile(raw_mip[raw_mip > 0], raw_vmax_percentile))
    else:
        vmax_raw = float(raw_mip.max()) if raw_mip.size else 1.0
    vmax_raw = max(vmax_raw, 1e-6)

    shaft_single_mip = np.max(shaft_single_zyx, axis=0)
    shaft_local_mip = np.max(shaft_local_zyx, axis=0)
    spine_single_mip = np.max(spine_single_zyx, axis=0)
    spine_local_mip = np.max(spine_local_zyx, axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for r in range(2):
        axes[r, 0].imshow(raw_mip, cmap="gray", vmin=0, vmax=vmax_raw)
        axes[r, 0].set_title(f"Raw global Z-MIP (p{raw_vmax_percentile:g})")
        axes[r, 0].axis("off")

    for ax, img, title, color in (
        (axes[0, 1], shaft_single_mip, "Shaft: single Z / plane", "cyan"),
        (axes[0, 2], shaft_local_mip, f"Shaft: local Z-MIP (r={z_radius})", "lime"),
        (axes[1, 1], spine_single_mip, "Spine: single Z / plane", "cyan"),
        (axes[1, 2], spine_local_mip, f"Spine: local Z-MIP (r={z_radius})", "lime"),
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1.0)
        if contour_level > 0:
            ax.contour(img, levels=[contour_level], colors=color, linewidths=0.4)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"DeepD3: single-plane vs local Z-MIP (2r+1={2 * z_radius + 1} planes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
