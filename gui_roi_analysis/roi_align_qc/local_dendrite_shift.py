"""Per-frame integer ROI shift from the dendrite around the spine, not the spine.

The auto spine ROI is a RESPAN seg_mask drawn on the group's reference
volume (the first high-mag stack, stored shift 0). Every frame of the
aligned TIFF is matched to that reference inside a window around the
spine, with the spine pixels removed from the match. The match is a
weighted normalized cross-correlation on highpass images over an integer
search grid, which is the kind of move a human makes in the ROI editor.

Shifts are content displacement in TIFF pixels: a (dy, dx) of (+2, 0)
means the dendrite next to the spine sits two rows lower than in the
reference, and the ROI should move two rows down.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTROL = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (_THIS_DIR, _CONTROL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from manual_shift_dataset import translate_mask  # noqa: E402

HIGHPASS_SIGMA = 3.0
WINDOW_RADIUS = 20
EXCLUDE_DILATION = 3
MAX_SHIFT = 6
MIN_GAIN = 0.02


@dataclass
class ShiftEstimate:
    """Integer shift per frame plus the NCC at that shift and at zero."""

    shift: np.ndarray
    score: np.ndarray
    zero_score: np.ndarray


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def highpass(image: np.ndarray, sigma: float = HIGHPASS_SIGMA) -> np.ndarray:
    """Subtract a Gaussian blur so slow background and brightness drift drop out."""
    image = np.asarray(image, dtype=np.float32)
    if sigma <= 0:
        return image
    return image - ndi.gaussian_filter(image, sigma)


def dendrite_weight(
    spine_mask: np.ndarray,
    *,
    window_radius: int = WINDOW_RADIUS,
    exclude_dilation: int = EXCLUDE_DILATION,
    center_yx: tuple[float, float] | None = None,
) -> np.ndarray:
    """Window around the spine with the dilated spine removed.

    ``window_radius`` <= 0 uses the whole frame (still minus the spine).
    """
    spine = np.asarray(spine_mask, dtype=bool)
    height, width = spine.shape
    if center_yx is None:
        ys, xs = np.nonzero(spine)
        center_yx = (float(ys.mean()), float(xs.mean())) if ys.size else (height / 2.0, width / 2.0)
    if window_radius > 0:
        yy, xx = np.ogrid[:height, :width]
        window = ((yy - center_yx[0]) ** 2 + (xx - center_yx[1]) ** 2) <= window_radius**2
    else:
        window = np.ones((height, width), dtype=bool)
    excluded = ndi.binary_dilation(spine, structure=disk(exclude_dilation)) if exclude_dilation > 0 else spine
    return window & ~excluded


def weighted_ncc_grid(
    reference: np.ndarray,
    moving: np.ndarray,
    weight: np.ndarray,
    max_shift: int = MAX_SHIFT,
) -> np.ndarray:
    """NCC for every integer content shift in [-max_shift, max_shift]^2.

    The weight lives on the reference grid. A content shift (dy, dx) means
    ``moving[y + dy, x + dx]`` is compared with ``reference[y, x]``. Pixels
    whose shifted sample leaves the frame are dropped from that shift.
    """
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    height, width = ref.shape
    size = 2 * max_shift + 1
    grid = np.full((size, size), np.nan, dtype=np.float64)
    ys, xs = np.nonzero(w > 0)
    if ys.size < 16:
        return grid
    wv = w[ys, xs]
    rv = ref[ys, xs]
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            sy = ys + dy
            sx = xs + dx
            inside = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
            if int(inside.sum()) < 16:
                continue
            ww = wv[inside]
            wsum = ww.sum()
            a = rv[inside]
            b = mov[sy[inside], sx[inside]]
            am = (a * ww).sum() / wsum
            bm = (b * ww).sum() / wsum
            ac = a - am
            bc = b - bm
            denom = np.sqrt((ww * ac * ac).sum() * (ww * bc * bc).sum())
            if denom <= 0:
                continue
            grid[dy + max_shift, dx + max_shift] = (ww * ac * bc).sum() / denom
    return grid


def best_shift(grid: np.ndarray, max_shift: int, *, min_gain: float = MIN_GAIN) -> tuple[tuple[int, int], float, float]:
    """Peak of the NCC grid. Falls back to zero when the gain over zero is small."""
    zero = grid[max_shift, max_shift]
    zero = float(zero) if np.isfinite(zero) else float("-inf")
    if not np.isfinite(grid).any():
        return (0, 0), zero, zero
    flat = np.nanargmax(grid)
    py, px = np.unravel_index(int(flat), grid.shape)
    peak = float(grid[py, px])
    shift = (int(py) - max_shift, int(px) - max_shift)
    if shift != (0, 0) and peak < zero + min_gain:
        return (0, 0), zero, zero
    return shift, peak, zero


def subpixel_peak(grid: np.ndarray, peak: tuple[int, int]) -> tuple[float, float]:
    """Parabolic refinement of an integer NCC peak. Returns (dy, dx) offsets of the peak."""
    py, px = peak
    out = [float(py), float(px)]
    for axis, (lo, hi) in enumerate(((py - 1, py + 1), (px - 1, px + 1))):
        if lo < 0 or hi >= grid.shape[axis]:
            continue
        if axis == 0:
            left, center, right = grid[lo, px], grid[py, px], grid[hi, px]
        else:
            left, center, right = grid[py, lo], grid[py, px], grid[py, hi]
        if not (np.isfinite(left) and np.isfinite(center) and np.isfinite(right)):
            continue
        denom = left - 2.0 * center + right
        if denom >= 0:
            continue
        offset = 0.5 * (left - right) / denom
        out[axis] += float(np.clip(offset, -0.5, 0.5))
    return out[0], out[1]


def frame_groups(phases: list[str] | None, n_frames: int) -> list[list[int]]:
    """Frames that share one shift estimate.

    Without phases every frame is its own group. With phases, one uncaging
    run (single-slice frames taken within seconds) is one group whose images
    are averaged before matching. Pre and post frames stay separate.
    """
    if phases is None:
        return [[t] for t in range(n_frames)]
    groups: list[list[int]] = []
    t = 0
    while t < n_frames:
        name = str(phases[t]).strip().lower()
        if name in ("unc", "uncaging"):
            run = [t]
            while t + 1 < n_frames and str(phases[t + 1]).strip().lower() in ("unc", "uncaging"):
                t += 1
                run.append(t)
            groups.append(run)
        else:
            groups.append([t])
        t += 1
    return groups


def estimate_shifts(
    reference: np.ndarray,
    stack: np.ndarray,
    spine_mask: np.ndarray,
    *,
    window_radius: int = WINDOW_RADIUS,
    exclude_dilation: int = EXCLUDE_DILATION,
    max_shift: int = MAX_SHIFT,
    sigma: float = HIGHPASS_SIGMA,
    min_gain: float = MIN_GAIN,
    subpixel: bool = False,
    phases: list[str] | None = None,
) -> ShiftEstimate:
    """Integer dendrite shift of every frame relative to the reference image.

    ``subpixel`` refines the NCC peak with a parabola before rounding, which
    removes the bias of a flat-topped peak. ``phases`` lets one uncaging run
    share one estimate made on the averaged uncaging image.
    """
    ref_hp = highpass(reference, sigma)
    n_frames = len(stack)
    shifts = np.zeros((n_frames, 2), dtype=np.int64)
    scores = np.full(n_frames, np.nan)
    zeros = np.full(n_frames, np.nan)
    weight = dendrite_weight(spine_mask, window_radius=window_radius, exclude_dilation=exclude_dilation)
    for group in frame_groups(phases, n_frames):
        image = stack[group[0]] if len(group) == 1 else np.mean(stack[group], axis=0)
        grid = weighted_ncc_grid(ref_hp, highpass(image, sigma), weight, max_shift)
        shift, peak, zero = best_shift(grid, max_shift, min_gain=min_gain)
        if subpixel and shift != (0, 0):
            py, px = subpixel_peak(grid, (shift[0] + max_shift, shift[1] + max_shift))
            shift = (int(round(py)) - max_shift, int(round(px)) - max_shift)
        for t in group:
            shifts[t] = shift
            scores[t] = peak
            zeros[t] = zero
    return ShiftEstimate(shifts, scores, zeros)


def _single_shift(
    reference: np.ndarray,
    image: np.ndarray,
    weight: np.ndarray,
    *,
    max_shift: int,
    sigma: float,
    min_gain: float,
    subpixel: bool,
    min_px: float = 0.0,
) -> tuple[tuple[int, int], float, float]:
    """One integer shift with optional subpixel refinement and a magnitude gate.

    ``min_px`` keeps the auto position (zero shift) unless the refined peak is
    at least that far from zero. Manual corrections below ~1.5 px are mostly
    annotator noise, so a gate near 1.5-2.5 px avoids trading one 1-px error
    for another.
    """
    grid = weighted_ncc_grid(highpass(reference, sigma), highpass(image, sigma), weight, max_shift)
    shift, peak, zero = best_shift(grid, max_shift, min_gain=min_gain)
    if shift == (0, 0):
        return shift, peak, zero
    fy, fx = float(shift[0]), float(shift[1])
    if subpixel or min_px > 0:
        py, px = subpixel_peak(grid, (shift[0] + max_shift, shift[1] + max_shift))
        fy, fx = py - max_shift, px - max_shift
    if min_px > 0 and float(np.hypot(fy, fx)) < min_px:
        return (0, 0), zero, zero
    if subpixel:
        shift = (int(round(fy)), int(round(fx)))
    return shift, peak, zero


def estimate_two_stage(
    reference: np.ndarray,
    stack: np.ndarray,
    spine_mask: np.ndarray,
    phases: list[str],
    *,
    static_radius: int = 0,
    static_sigma: float = HIGHPASS_SIGMA,
    static_gain: float = MIN_GAIN,
    local_radius: int = WINDOW_RADIUS,
    local_sigma: float = HIGHPASS_SIGMA,
    local_gain: float = MIN_GAIN,
    exclude_dilation: int = EXCLUDE_DILATION,
    max_shift: int = MAX_SHIFT,
    subpixel: bool = False,
    pre_only_static: bool = True,
    use_static: bool = True,
    static_min_px: float = 0.0,
    local_min_px: float = 0.0,
) -> ShiftEstimate:
    """Static offset from the reference plus a per-frame drift from the pre mean.

    Stage 1 places the auto ROI on the pre frames: the mean pre image is
    matched to the reference stack inside ``static_radius`` (0 = whole
    frame). Stage 2 follows the dendrite from that pre mean to every frame
    inside ``local_radius``. Both stages leave the spine out of the match.
    The per-frame shift is the sum, so pre frames receive the static offset
    (plus their own small drift when ``pre_only_static`` is False).

    With ``use_static=False`` stage 1 is skipped (the auto ROI is trusted on
    the pre frames) and only the drift from the pre mean is estimated; the
    ``reference`` argument is then ignored. ``static_min_px`` and
    ``local_min_px`` gate each stage on the subpixel magnitude of its peak so
    that sub-threshold shifts are not applied.
    """
    n_frames = len(stack)
    phase_names = [str(p).strip().lower() for p in phases]
    pre = [t for t, p in enumerate(phase_names) if p == "pre"]
    if not pre:
        pre = [0]
    pre_mean = np.mean(stack[pre], axis=0)
    if use_static:
        static_weight = dendrite_weight(spine_mask, window_radius=static_radius, exclude_dilation=exclude_dilation)
        static, static_peak, static_zero = _single_shift(
            reference,
            pre_mean,
            static_weight,
            max_shift=max_shift,
            sigma=static_sigma,
            min_gain=static_gain,
            subpixel=subpixel,
            min_px=static_min_px,
        )
    else:
        static, static_peak, static_zero = (0, 0), float("nan"), float("nan")
    moved_spine = translate_mask(spine_mask, *static)
    local_weight = dendrite_weight(moved_spine, window_radius=local_radius, exclude_dilation=exclude_dilation)
    shifts = np.zeros((n_frames, 2), dtype=np.int64)
    scores = np.full(n_frames, np.nan)
    zeros = np.full(n_frames, np.nan)
    for group in frame_groups(phase_names, n_frames):
        if pre_only_static and phase_names[group[0]] == "pre":
            for t in group:
                shifts[t] = static
                scores[t] = static_peak
                zeros[t] = static_zero
            continue
        image = stack[group[0]] if len(group) == 1 else np.mean(stack[group], axis=0)
        local, peak, zero = _single_shift(
            pre_mean,
            image,
            local_weight,
            max_shift=max_shift,
            sigma=local_sigma,
            min_gain=local_gain,
            subpixel=subpixel,
            min_px=local_min_px,
        )
        for t in group:
            shifts[t] = (static[0] + local[0], static[1] + local[1])
            scores[t] = peak
            zeros[t] = zero
    return ShiftEstimate(shifts, scores, zeros)


def smooth_within_phase(shifts: np.ndarray, phases: list[str], *, size: int = 3) -> np.ndarray:
    """Median filter the integer shifts inside each phase run.

    Uncaging frames are many and nearly simultaneous, so one median is used
    for the whole uncaging run. Pre and post get a running median of ``size``.
    """
    out = np.array(shifts, dtype=np.int64, copy=True)
    n = len(phases)
    start = 0
    while start < n:
        end = start
        while end < n and phases[end] == phases[start]:
            end += 1
        run = out[start:end]
        name = str(phases[start]).strip().lower()
        if name in ("unc", "uncaging"):
            run[:] = np.round(np.median(run, axis=0)).astype(np.int64)
        elif len(run) >= size:
            for k in range(2):
                run[:, k] = ndi.median_filter(run[:, k], size=size, mode="nearest")
        out[start:end] = run
        start = end
    return out


def masks_from_shifts(spine_mask: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Translate one mask by each row of integer shifts."""
    out = np.zeros((len(shifts),) + spine_mask.shape, dtype=bool)
    for t, (dy, dx) in enumerate(shifts):
        out[t] = translate_mask(spine_mask, int(dy), int(dx))
    return out
