"""Call a first-pre max projection obviously dead.

A dead field has a round bleb larger than a spine and no long dendrite shaft.
Beads sitting on a continuous shaft are not called dead.

Lengths are micrometers. Pass the in-plane micrometers per pixel so the same
physical sizes apply at every zoom.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace
from skimage.feature import peak_local_max
from skimage.filters import frangi
from skimage.morphology import skeletonize

# Dimensionless peak of the scale-normalized LoG after a 0-1 stretch.
BLEB_MIN = 0.45
# Disk erased around a bleb, in units of the LoG sigma.
BLEB_RADIUS_FACTOR = 1.8

# Tuned on zoom-14 high-mag (FOV 273 x 271 um, 128 px). At that sampling the
# shaft cutoff is 28 px and the blob sigmas are 3, 4.5, 6, 8, and 11 px.
_TUNED_FOV_X_UM = 273.0
_TUNED_FOV_Y_UM = 271.0
_TUNED_ZOOM = 14.0
_TUNED_PIXELS = 128.0
TUNED_XY_UM = 0.5 * (
    _TUNED_FOV_X_UM / _TUNED_ZOOM / _TUNED_PIXELS
    + _TUNED_FOV_Y_UM / _TUNED_ZOOM / _TUNED_PIXELS
)


def _um(pixels: float) -> float:
    return float(pixels) * TUNED_XY_UM


BLEB_SIGMA_UM = tuple(_um(sigma) for sigma in (3.0, 4.5, 6.0, 8.0, 11.0))
RIDGE_SIGMA_UM = tuple(_um(sigma) for sigma in (1.0, 2.0, 3.0))
SHAFT_MAX_UM = _um(28.0)


def xy_um_from_state(statedict: dict) -> float:
    """Mean in-plane micrometers per pixel from a FLIMage acquisition state."""
    fov = statedict["State.Acq.FOV_default"]
    zoom = float(statedict["State.Acq.zoom"])
    pixels_x = float(statedict["State.Acq.pixelsPerLine"])
    pixels_y = float(statedict["State.Acq.linesPerFrame"])
    if zoom <= 0 or pixels_x <= 0 or pixels_y <= 0:
        raise ValueError("zoom and pixel counts must be positive")
    x_um = float(fov[0]) / zoom / pixels_x
    y_um = float(fov[1]) / zoom / pixels_y
    return 0.5 * (x_um + y_um)


def _to_px(length_um: float, xy_um: float) -> float:
    if xy_um <= 0:
        raise ValueError("xy_um must be positive")
    return float(length_um) / float(xy_um)


def _unit_interval(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99.5])
    scaled = (values - low) / max(float(high - low), 1e-6)
    scaled = np.clip(scaled, 0, 1)
    scaled[~np.isfinite(values)] = 0
    return scaled.astype(np.float32)


def _response(scaled: np.ndarray, sigma_px: float) -> np.ndarray:
    """Scale-normalized LoG. Bright round objects are positive."""
    return (-(sigma_px ** 2) * gaussian_laplace(scaled, sigma_px)).astype(np.float32)


@dataclass
class DeadAnalysis:
    """Intermediates for one field, with lengths in micrometers."""

    scaled: np.ndarray
    bleb_response: np.ndarray
    bleb_centers: list[tuple[int, int, float]]
    residual: np.ndarray
    ridges: np.ndarray
    skeleton: np.ndarray
    shaft_mask: np.ndarray
    bleb: float
    shaft_um: float
    xy_um: float


def analyze(
    image: np.ndarray,
    xy_um: float,
    bleb_sigma_um: tuple[float, ...] = BLEB_SIGMA_UM,
    ridge_sigma_um: tuple[float, ...] = RIDGE_SIGMA_UM,
) -> DeadAnalysis:
    """Bleb map, bleb-suppressed image, and the longest remaining shaft."""
    scaled = _unit_interval(image)
    sigmas = tuple(_to_px(sigma, xy_um) for sigma in bleb_sigma_um)
    responses = [_response(scaled, sigma) for sigma in sigmas]
    bleb_response = np.maximum.reduce(responses) if responses else np.zeros_like(scaled)
    bleb = max((float(np.max(response)) for response in responses), default=0.0)

    residual = scaled.copy()
    centers: list[tuple[int, int, float]] = []
    for sigma, response in zip(sigmas, responses):
        threshold = max(0.5, 0.65 * float(np.max(response)))
        peaks = peak_local_max(
            response,
            min_distance=max(int(sigma), 1),
            threshold_abs=threshold,
        )
        radius_px = max(int(round(sigma * BLEB_RADIUS_FACTOR)), 2)
        for y_coord, x_coord in peaks:
            y_index = int(y_coord)
            x_index = int(x_coord)
            y0 = max(0, y_index - radius_px)
            y1 = min(scaled.shape[0], y_index + radius_px + 1)
            x0 = max(0, x_index - radius_px)
            x1 = min(scaled.shape[1], x_index + radius_px + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disk = (yy - y_index) ** 2 + (xx - x_index) ** 2 <= radius_px ** 2
            residual[y0:y1, x0:x1][disk] = 0
            centers.append((y_index, x_index, radius_px * xy_um))

    ridge_px = tuple(_to_px(sigma, xy_um) for sigma in ridge_sigma_um)
    ridges = np.asarray(frangi(residual, sigmas=ridge_px, black_ridges=False), dtype=np.float32)
    peak = float(np.max(ridges)) if ridges.size else 0.0
    shaft_mask = np.zeros(scaled.shape, dtype=bool)
    if peak <= 0:
        skeleton = np.zeros(scaled.shape, dtype=bool)
        shaft_px = 0
    else:
        mask = ridges >= max(0.15 * peak, 1e-4)
        skeleton = np.asarray(skeletonize(mask), dtype=bool)
        count, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
        shaft_px = 0
        for label in range(1, count):
            component = labels == label
            length = _longest_component(component)
            if length > shaft_px:
                shaft_px = length
                shaft_mask = component
    return DeadAnalysis(
        scaled=scaled,
        bleb_response=bleb_response,
        bleb_centers=centers,
        residual=residual,
        ridges=ridges,
        skeleton=skeleton,
        shaft_mask=shaft_mask,
        bleb=bleb,
        shaft_um=float(shaft_px) * float(xy_um),
        xy_um=float(xy_um),
    )


def _longest_component(mask: np.ndarray) -> int:
    """Pixel steps along the longest path in one skeleton component.

    Each 8-connected step counts as one pixel, including diagonals.
    """
    ys, xs = np.nonzero(mask)
    count = len(ys)
    if count < 2:
        return count
    index = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    neighbors: list[list[int]] = [[] for _ in range(count)]
    for i, (y, x) in enumerate(zip(ys.tolist(), xs.tolist())):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                other = index.get((y + dy, x + dx))
                if other is not None and other > i:
                    neighbors[i].append(other)
                    neighbors[other].append(i)

    def farthest(start: int) -> tuple[int, int]:
        distance = [-1] * count
        distance[start] = 0
        stack = [start]
        while stack:
            current = stack.pop()
            for nxt in neighbors[current]:
                if distance[nxt] < 0:
                    distance[nxt] = distance[current] + 1
                    stack.append(nxt)
        end = int(np.argmax(distance))
        return end, int(distance[end])

    far, _length = farthest(0)
    _other, length = farthest(far)
    return length


def dead_scores(
    image: np.ndarray,
    xy_um: float,
    bleb_sigma_um: tuple[float, ...] = BLEB_SIGMA_UM,
    ridge_sigma_um: tuple[float, ...] = RIDGE_SIGMA_UM,
) -> tuple[float, float]:
    """Bleb strength and shaft length in micrometers."""
    result = analyze(image, xy_um, bleb_sigma_um=bleb_sigma_um, ridge_sigma_um=ridge_sigma_um)
    return result.bleb, result.shaft_um


def obviously_dead(
    image: np.ndarray,
    xy_um: float,
    bleb_min: float = BLEB_MIN,
    shaft_max_um: float = SHAFT_MAX_UM,
    bleb_sigma_um: tuple[float, ...] = BLEB_SIGMA_UM,
    ridge_sigma_um: tuple[float, ...] = RIDGE_SIGMA_UM,
) -> bool:
    """True when a large bleb is present and no long shaft crosses the field."""
    bleb, shaft_um = dead_scores(
        image,
        xy_um,
        bleb_sigma_um=bleb_sigma_um,
        ridge_sigma_um=ridge_sigma_um,
    )
    return bleb >= bleb_min and shaft_um <= shaft_max_um
