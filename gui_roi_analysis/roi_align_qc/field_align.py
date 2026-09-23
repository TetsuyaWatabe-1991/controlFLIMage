"""Rigid alignment using the dendrite and the rest of the field, not the spine.

Spine-sized crops have too little texture for phase correlation. These
estimators use the whole 2D projection or its edges.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel

from tracking import content_shift_yx, phase_only_shift_yx


def shift_vs_first(stack: np.ndarray, pair_shift) -> np.ndarray:
    """Object displacement of every frame relative to frame 0."""
    reference = np.asarray(stack[0], dtype=np.float32)
    shifts = np.zeros((len(stack), 2), dtype=np.float64)
    for t in range(1, len(stack)):
        shifts[t] = pair_shift(reference, np.asarray(stack[t], dtype=np.float32))
    return shifts


def full_spatial(stack: np.ndarray) -> np.ndarray:
    """Spatial cross-correlation of the whole frame against frame 0."""
    return shift_vs_first(stack, content_shift_yx)


def full_phase(stack: np.ndarray) -> np.ndarray:
    """Phase-only correlation of the whole frame against frame 0."""

    def _pair(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
        dy, dx, _ = phase_only_shift_yx(reference, moving)
        return dy, dx

    return shift_vs_first(stack, _pair)


def _highpass(image: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    blurred = gaussian_filter(image, sigma=sigma)
    return np.asarray(image - blurred, dtype=np.float32)


def highpass_spatial(stack: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Spatial correlation after removing the slow background.

    The dendrite and other sharp processes remain. A uniform glow does not.
    """
    filtered = np.stack([_highpass(frame, sigma) for frame in stack])
    return full_spatial(filtered)


def edge_spatial(stack: np.ndarray) -> np.ndarray:
    """Spatial correlation of Sobel edge magnitude.

    Edge magnitude follows the dendrite shaft even when spine brightness changes.
    """
    edges = np.stack([sobel(np.asarray(frame, dtype=np.float32)) for frame in stack])
    return full_spatial(edges)


def tile_median_spatial(stack: np.ndarray, n_tiles: int = 3) -> np.ndarray:
    """Median of spatial shifts from a grid of tiles.

    A tile with almost no texture is ignored. One bright spine cannot outvote
    the other tiles.
    """
    reference = np.asarray(stack[0], dtype=np.float32)
    height, width = reference.shape
    shifts = np.zeros((len(stack), 2), dtype=np.float64)
    for t in range(1, len(stack)):
        moving = np.asarray(stack[t], dtype=np.float32)
        samples: list[tuple[float, float]] = []
        for i in range(n_tiles):
            for j in range(n_tiles):
                y0, y1 = i * height // n_tiles, (i + 1) * height // n_tiles
                x0, x1 = j * width // n_tiles, (j + 1) * width // n_tiles
                ref_tile = reference[y0:y1, x0:x1]
                mov_tile = moving[y0:y1, x0:x1]
                if float(np.std(ref_tile)) < 1e-3 or float(np.std(mov_tile)) < 1e-3:
                    continue
                samples.append(content_shift_yx(ref_tile, mov_tile))
        if samples:
            shifts[t] = np.median(np.asarray(samples, dtype=np.float64), axis=0)
    return shifts


FIELD_ALIGNERS = {
    "full_spatial": full_spatial,
    "full_phase": full_phase,
    "highpass_spatial": highpass_spatial,
    "edge_spatial": edge_spatial,
    "tile_median": tile_median_spatial,
}
