# -*- coding: utf-8 -*-
"""Save DeepD3-style _mip.png overview (6-panel) for ROI parameter comparison."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deepd3_spine_head_detector import SpinePosDeepD3

N_PANELS = 6


def _figure_size_for_panels(n_cols: int, image_hw: tuple[int, int], *, title_band: float = 0.14) -> tuple[float, float]:
    """Match figure aspect to image stack so savefig does not pad huge vertical margins."""
    h, w = image_hw
    if h <= 0:
        h = 1
    aspect = float(w) / float(h)
    row_h = 1.05
    return (aspect * n_cols * row_h, row_h + title_band)


def save_mip_overview_png(
    S,
    r,
    prop_dict: dict,
    cand_spines,
    savepath: str,
    *,
    title: str = "",
    dpi: int = 400,
) -> None:
    """
    Same layout as SpinePosDeepD3.plot_uncaging_pos MIP figure ({stem}_mip.png).
    result_dict is omitted (no uncaging markers).
    """
    assigner = SpinePosDeepD3()
    cand_index = (
        set(cand_spines.index.astype(str))
        if hasattr(cand_spines, "index") and len(cand_spines)
        else set()
    )

    stack_mip = np.amax(S.stack, axis=0)
    roi_mip = np.amax(r.roi_map, axis=0)
    colors = assigner.glasbey_colors()

    fig_w, fig_h = _figure_size_for_panels(N_PANELS, stack_mip.shape)
    fig, axs = plt.subplots(1, N_PANELS, figsize=(fig_w, fig_h))
    for ax in axs:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0)

    axs[0].imshow(stack_mip, cmap="gray", interpolation="nearest")

    axs[1].imshow(stack_mip, cmap="gray", interpolation="nearest")

    axs[2].imshow(roi_mip, cmap=colors, interpolation="nearest")

    axs[3].imshow(roi_mip, cmap=colors, interpolation="nearest")

    axs[4].imshow(stack_mip, cmap="gray", interpolation="nearest")
    axs[4].imshow(
        roi_mip,
        cmap=colors,
        alpha=(roi_mip > 0).astype(np.float32),
        interpolation="nearest",
    )

    axs[5].imshow(stack_mip, cmap="gray", interpolation="nearest")

    for label, info in prop_dict.items():
        x, y = info["x"], info["y"]
        npix = info["num_pixels"]
        intensity = info["intensity"]
        axs[2].text(x, y, str(npix), fontsize=2, color="white")
        axs[3].text(x, y, str(intensity), fontsize=2, color="white")
        if label in cand_index:
            axs[1].plot(x, y, "r.", ms=1)
        else:
            axs[1].plot(x, y, "c.", ms=1)

    top = 0.90 if title else 0.99
    fig.subplots_adjust(left=0.001, right=0.999, top=top, bottom=0.001, wspace=0.02)
    if title:
        fig.suptitle(title, fontsize=4, y=0.97)
    fig.savefig(
        savepath,
        dpi=dpi,
        pad_inches=0.01,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_montage_thumbnails(
    image_paths: list[str],
    labels: list[str],
    savepath: str,
    *,
    ncol: int = 4,
    thumb_size: int = 200,
) -> None:
    from skimage.transform import resize

    n = len(image_paths)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.35 * ncol, 1.05 * nrow))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat[n:]:
        ax.axis("off")

    for i, (path, lab) in enumerate(zip(image_paths, labels)):
        img = plt.imread(path)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=-1)
        small = resize(
            img.astype(np.float32),
            (thumb_size, thumb_size),
            preserve_range=True,
            anti_aliasing=True,
        )
        axes_flat[i].imshow(small, cmap="gray")
        axes_flat[i].set_title(lab, fontsize=6)
        axes_flat[i].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.08, hspace=0.25)
    fig.savefig(savepath, dpi=120, pad_inches=0.02, bbox_inches="tight")
    plt.close(fig)


def save_two_panel_compare(
    path_left: str,
    path_right: str,
    label_left: str,
    label_right: str,
    savepath: str,
) -> None:
    """Full-width side-by-side of two _mip.png panels."""
    left = plt.imread(path_left)
    fig_w, fig_h = _figure_size_for_panels(2, left.shape[:2], title_band=0.2)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    for ax, path, lab in zip(
        axes,
        (path_left, path_right),
        (label_left, label_right),
    ):
        ax.imshow(plt.imread(path))
        ax.set_title(lab, fontsize=7)
        ax.axis("off")
        ax.margins(0)
    fig.subplots_adjust(left=0.002, right=0.998, top=0.88, bottom=0.002, wspace=0.02)
    fig.savefig(savepath, dpi=150, pad_inches=0.02, bbox_inches="tight")
    plt.close(fig)
