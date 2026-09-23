"""XYZ registration from highpass-filtered XY and YZ projections.

XY gives dy and dx. YZ gives dz. The Y shift from the YZ projection is not
used, because the in-plane match on the XY projection is the one checked
against the 2D highpass result.

Registration shifts are the amount to apply to the moving volume so it lands
on the reference. They are the opposite of the object displacement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndimage_shift

_CONTROL = str(Path(__file__).resolve().parents[2])
if _CONTROL not in sys.path:
    sys.path.insert(0, _CONTROL)

from FLIMageAlignment import highpass_registration_zyx  # noqa: E402

XY_SIGMA = 6.0
# Z is only ~15 slices, so the Z blur is much narrower than the XY blur.
YZ_SIGMA_ZY = (1.5, 6.0)


def highpass_2d(image: np.ndarray, sigma: float | tuple[float, float]) -> np.ndarray:
    """Subtract a Gaussian blur. Sharp processes remain."""
    blurred = gaussian_filter(np.asarray(image, dtype=np.float32), sigma=sigma)
    return np.asarray(image, dtype=np.float32) - blurred


def apply_registration_zyx(volume: np.ndarray, shift_zyx: np.ndarray) -> np.ndarray:
    """Shift a ZYX volume by a registration vector. Edges are filled with 0."""
    return ndimage_shift(
        np.asarray(volume, dtype=np.float32),
        shift=np.asarray(shift_zyx, dtype=np.float64),
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)


def cumulative_adjacent(volumes: list[np.ndarray], pair_shift) -> np.ndarray:
    """Frame-0-relative registration shift from adjacent pairs.

    Each step compares the raw volume with the previous raw volume, matching
    the quantifier, which does not feed the warped crop back in.
    """
    shifts = np.zeros((len(volumes), 3), dtype=np.float64)
    for index in range(1, len(volumes)):
        incremental = np.asarray(pair_shift(volumes[index - 1], volumes[index]), dtype=np.float64)
        shifts[index] = shifts[index - 1] + incremental
    return shifts
