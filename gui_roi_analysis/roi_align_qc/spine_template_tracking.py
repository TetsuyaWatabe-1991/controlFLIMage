"""Track the uncaged spine itself with a brightness-normalized template.

The manual ROI follows the spine head: when the head swells, shrinks or
creeps along the shaft after uncaging the annotator moves the mask even
though the surrounding field is still. The dendrite-only estimators in
``local_dendrite_shift`` cannot see this. Here the pre-mean image inside a
slightly dilated spine mask (shaft pixels removed) is used as a template and
matched to every frame with weighted normalized cross-correlation. NCC is
invariant to the spine brightening, so the F/F0 increase itself does not pull
the ROI; only the spatial pattern does. A dendrite NCC term regularizes the
match, candidate positions that would put the ROI on the shaft or on another
RESPAN spine are rejected, and small moves are gated away because manual
corrections below ~1.5 px are mostly annotator noise.

All outputs are integer translations of the auto mask, so the ROI shape is
identical on every frame of a set as the quantification requires.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from local_dendrite_shift import (
    HIGHPASS_SIGMA,
    MAX_SHIFT,
    ShiftEstimate,
    dendrite_weight,
    disk,
    frame_groups,
    highpass,
    subpixel_peak,
    weighted_ncc_grid,
)
from manual_shift_dataset import translate_mask

SPINE_DILATION = 2
SHAFT_DILATION = 1
SPINE_SIGMA = 4.0
DENDRITE_LAMBDA = 0.5
DENDRITE_RADIUS = 30
TEMPLATE_MIN_GAIN = 0.05
TEMPLATE_MIN_PX = 1.5


@dataclass
class TrackParams:
    """Tunable knobs of :func:`track_spine`."""

    max_shift: int = MAX_SHIFT
    spine_dilation: int = SPINE_DILATION
    shaft_dilation: int = SHAFT_DILATION
    spine_sigma: float = SPINE_SIGMA
    dendrite_lambda: float = DENDRITE_LAMBDA
    dendrite_radius: int = DENDRITE_RADIUS
    dendrite_sigma: float = HIGHPASS_SIGMA
    min_gain: float = TEMPLATE_MIN_GAIN
    min_px: float = TEMPLATE_MIN_PX
    overlap_tolerance: int = 0
    temporal_penalty: float = 0.0
    track_pre: bool = False
    use_raw_spine: bool = False
    post_fit: str = "none"  # "none" | "linear" | "median"
    criterion: str = "ncc"  # "ncc" | "integral" | "both"
    integral_sigma: float = 0.0  # <= 0: raw image for the contrast criterion
    integral_weight: float = 1.0
    ring_width: int = 2
    static_gain: float = 0.05
    static_min_px: float = 1.5
    static_criterion: str = "integral"  # "integral" | "contain"
    blob_frac: float = 0.5
    blob_smooth: float = 1.0
    blob_window: int = 8
    blob_min_overlap: float = 0.3
    blob_shaft_dilation: int = 3
    blob_cap_above_peak: bool = True
    blob_snap_dendrite: bool = True
    contain_slack: float = 0.02
    contain_stay_slack: float = 0.05


def spine_weight(
    spine_mask: np.ndarray,
    shaft_mask: np.ndarray | None,
    *,
    dilation: int = SPINE_DILATION,
    shaft_dilation: int = SHAFT_DILATION,
) -> np.ndarray:
    """Float weight of the spine template: dilated spine without shaft pixels."""
    region = ndi.binary_dilation(spine_mask, structure=disk(dilation)) if dilation > 0 else spine_mask.copy()
    if shaft_mask is not None and shaft_mask.any():
        shaft = ndi.binary_dilation(shaft_mask, structure=disk(shaft_dilation)) if shaft_dilation > 0 else shaft_mask
        trimmed = region & ~shaft
        if trimmed.sum() >= 0.5 * spine_mask.sum():
            region = trimmed
    return region.astype(np.float64)


def forbidden_shifts(
    spine_mask: np.ndarray,
    shaft_mask: np.ndarray | None,
    other_spines: np.ndarray | None,
    max_shift: int,
    *,
    tolerance: int = 0,
) -> np.ndarray:
    """Boolean grid of shifts that push the ROI onto the shaft or another spine.

    A shift is forbidden when the translated mask overlaps the shaft or other
    spines by more pixels than the untranslated auto mask already does (plus
    ``tolerance``). Shifts that leave the image are forbidden as well.
    """
    size = 2 * max_shift + 1
    bad = np.zeros((size, size), dtype=bool)
    obstacles = []
    if shaft_mask is not None and shaft_mask.any():
        obstacles.append(shaft_mask)
    if other_spines is not None and other_spines.any():
        obstacles.append(other_spines)
    area = int(spine_mask.sum())
    base = [int((spine_mask & ob).sum()) for ob in obstacles]
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            moved = translate_mask(spine_mask, dy, dx)
            if int(moved.sum()) < area:
                bad[dy + max_shift, dx + max_shift] = True
                continue
            for ob, b in zip(obstacles, base):
                if int((moved & ob).sum()) > b + tolerance:
                    bad[dy + max_shift, dx + max_shift] = True
                    break
    return bad


def mask_integral_grid(image: np.ndarray, mask: np.ndarray, max_shift: int) -> np.ndarray:
    """Mean of ``image`` inside the mask translated by every candidate shift.

    This is the criterion an annotator applies by eye: put the ROI where it
    collects the spine signal. ``image`` should be background-free (highpass)
    so that haze and the shaft do not dominate.
    """
    size = 2 * max_shift + 1
    grid = np.full((size, size), np.nan, dtype=np.float64)
    img = np.asarray(image, dtype=np.float64)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            moved = translate_mask(mask, dy, dx)
            if moved.any():
                grid[dy + max_shift, dx + max_shift] = float(img[moved].mean())
    return grid


def mask_contrast_grid(image: np.ndarray, mask: np.ndarray, max_shift: int, *, ring_width: int = 2) -> np.ndarray:
    """Mean inside the translated mask minus the mean in a ring around it.

    A mask-shaped matched filter: the ring cancels smooth background and the
    glow of a nearby shaft, so the maximum sits where the mask encloses a
    compact blob (the spine head) rather than where the field is brightest.
    """
    size = 2 * max_shift + 1
    grid = np.full((size, size), np.nan, dtype=np.float64)
    img = np.asarray(image, dtype=np.float64)
    ring = ndi.binary_dilation(mask, structure=disk(ring_width)) & ~mask
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            moved = translate_mask(mask, dy, dx)
            moved_ring = translate_mask(ring, dy, dx)
            if moved.any() and moved_ring.any():
                grid[dy + max_shift, dx + max_shift] = float(img[moved].mean() - img[moved_ring].mean())
    return grid


def relative_integral_grid(
    image: np.ndarray,
    mask: np.ndarray,
    max_shift: int,
    *,
    ring_width: int = 2,
) -> np.ndarray:
    """Contrast grid rescaled to [-1, 1] with zero shift at 0.

    The rescaling by the grid range makes the gain gate a fraction of the
    available contrast change, independent of the absolute brightness.
    """
    grid = mask_contrast_grid(image, mask, max_shift, ring_width=ring_width)
    zero = grid[max_shift, max_shift]
    if not np.isfinite(zero):
        zero = float(np.nanmedian(grid))
    span = float(np.nanmax(grid) - np.nanmin(grid))
    if not np.isfinite(span) or span <= 1e-9:
        span = 1.0
    return (grid - zero) / span


def locate_static(
    pre_mean: np.ndarray,
    spine_mask: np.ndarray,
    shaft_mask: np.ndarray | None,
    other_spines: np.ndarray | None,
    *,
    params: "TrackParams | None" = None,
) -> Pick:
    """Pre-frame offset of the auto mask that centers it on the spine signal.

    Uses the mask-integral criterion on the highpass pre-mean image with the
    shaft/other-spine exclusion and the same gain and magnitude gates as the
    tracking stage.
    """
    p = params or TrackParams()
    bad = forbidden_shifts(spine_mask, shaft_mask, other_spines, p.max_shift, tolerance=p.overlap_tolerance)
    if p.static_criterion == "contain":
        return _contain_pick(pre_mean, spine_mask, shaft_mask, other_spines, bad, p)
    image = _integral_image(pre_mean, p.integral_sigma)
    grid = relative_integral_grid(image, spine_mask, p.max_shift, ring_width=p.ring_width)
    grid = np.where(bad, np.nan, grid)
    return _pick(grid, p.max_shift, min_gain=p.static_gain, min_px=p.static_min_px, previous=None, temporal_penalty=0.0)


def _contain_pick(
    image: np.ndarray,
    roi: np.ndarray,
    shaft_mask: np.ndarray | None,
    other_spines: np.ndarray | None,
    bad: np.ndarray,
    p: "TrackParams",
) -> Pick:
    """Minimal shift that keeps the spine blob of ``image`` inside the ROI."""
    blob = spine_blob(
        image,
        roi,
        shaft_mask,
        other_spines,
        smooth=p.blob_smooth,
        frac=p.blob_frac,
        window=p.blob_window,
        min_overlap=p.blob_min_overlap,
        shaft_dilation=p.blob_shaft_dilation,
        cap_above_peak=p.blob_cap_above_peak,
        snap_dendrite=p.blob_snap_dendrite,
    )
    grid = coverage_grid(blob, roi, p.max_shift)
    return minimal_containing_shift(grid, bad, p.max_shift, slack=p.contain_slack, stay_slack=p.contain_stay_slack)


def dendrite_mask(
    image: np.ndarray,
    shaft_mask: np.ndarray | None,
    window_mask: np.ndarray,
    *,
    search_dilation: int = 4,
    frac: float = 0.6,
    grow: int = 2,
    bg_percentile: float = 20.0,
    protect: np.ndarray | None = None,
) -> np.ndarray:
    """Bright dendrite pixels near the RESPAN shaft fit, snapped to the image.

    The shaft fit comes from the reference stack and is often a few pixels
    off in later frames, so the fit is only used to say where to look: bright
    components (above ``frac`` of the way from background to the dendrite
    peak) that touch the fit dilated by ``search_dilation`` are the dendrite,
    grown by ``grow`` pixels. Pixels inside ``protect`` (the ROI) are never
    part of the dendrite so a spine that touches the dendrite keeps its head.
    """
    if shaft_mask is None or not shaft_mask.any():
        return np.zeros_like(window_mask, dtype=bool)
    img = np.asarray(image, dtype=np.float64)
    near = ndi.binary_dilation(shaft_mask, structure=disk(search_dilation))
    region = window_mask | near
    background = float(np.percentile(img[region], bg_percentile))
    peak = float(np.percentile(img[near], 95))
    if peak <= background:
        return ndi.binary_dilation(shaft_mask, structure=disk(grow))
    bright = img > background + frac * (peak - background)
    if protect is not None:
        bright &= ~protect
    labels, n_labels = ndi.label(bright)
    out = np.zeros_like(window_mask, dtype=bool)
    if n_labels:
        touching = np.unique(labels[near & bright])
        touching = touching[touching > 0]
        out = np.isin(labels, touching)
    out |= shaft_mask
    if grow > 0:
        out = ndi.binary_dilation(out, structure=disk(grow))
    if protect is not None:
        out &= ~protect
    return out


def spine_blob(
    image: np.ndarray,
    roi: np.ndarray,
    shaft_mask: np.ndarray | None,
    other_spines: np.ndarray | None,
    *,
    smooth: float = 1.0,
    frac: float = 0.5,
    bg_percentile: float = 20.0,
    window: int = 8,
    min_overlap: float = 0.3,
    shaft_dilation: int = 3,
    cap_above_peak: bool = True,
    snap_dendrite: bool = True,
) -> np.ndarray:
    """Binary blob of the spine head near ``roi`` in one image.

    The image is lightly smoothed, thresholded at ``frac`` of the way from
    the local background (``bg_percentile`` inside a window around the ROI)
    to the brightest ROI pixel, and the connected components that overlap the
    ROI are kept. Shaft pixels (dilated by ``shaft_dilation`` because the
    RESPAN shaft fit is thinner than the dendrite) and other RESPAN spines are
    removed first, and with ``cap_above_peak`` pixels brighter than the ROI
    peak are removed too, so the blob does not leak along the dendrite or
    into a neighbour.
    """
    img = ndi.gaussian_filter(np.asarray(image, dtype=np.float64), smooth) if smooth > 0 else np.asarray(image, dtype=np.float64)
    window_mask = ndi.binary_dilation(roi, structure=disk(window))
    exclude = np.zeros_like(roi, dtype=bool)
    if shaft_mask is not None and shaft_mask.any():
        exclude |= ndi.binary_dilation(shaft_mask, structure=disk(shaft_dilation)) if shaft_dilation > 0 else shaft_mask
        if snap_dendrite:
            exclude |= dendrite_mask(img, shaft_mask, window_mask, protect=roi)
    if other_spines is not None and other_spines.any():
        exclude |= other_spines
    region = window_mask & ~exclude
    if region.sum() < 10:
        return np.zeros_like(roi, dtype=bool)
    background = float(np.percentile(img[region], bg_percentile))
    core = roi & ~exclude
    if not core.any():
        core = roi
    peak = float(img[core].max())
    if peak <= background:
        return np.zeros_like(roi, dtype=bool)
    foreground = (img > background + frac * (peak - background)) & region
    if cap_above_peak:
        foreground &= img <= peak
    labels, n_labels = ndi.label(foreground)
    if n_labels == 0:
        return np.zeros_like(roi, dtype=bool)
    blob = np.zeros_like(roi, dtype=bool)
    best_label, best_overlap = 0, 0.0
    for index in range(1, n_labels + 1):
        component = labels == index
        size = int(component.sum())
        overlap = float((component & roi).sum()) / float(size)
        if overlap > best_overlap:
            best_label, best_overlap = index, overlap
        if overlap >= min_overlap:
            blob |= component
    if best_label and not blob.any():
        blob |= labels == best_label
    return blob


def coverage_grid(blob: np.ndarray, roi: np.ndarray, max_shift: int) -> np.ndarray:
    """Fraction of blob pixels inside the translated ROI for every candidate shift."""
    size = 2 * max_shift + 1
    grid = np.full((size, size), np.nan, dtype=np.float64)
    total = float(blob.sum())
    if total <= 0:
        return grid
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            grid[dy + max_shift, dx + max_shift] = float((translate_mask(roi, dy, dx) & blob).sum()) / total
    return grid


def minimal_containing_shift(
    grid: np.ndarray,
    bad: np.ndarray,
    max_shift: int,
    *,
    slack: float = 0.02,
    stay_slack: float = 0.05,
) -> Pick:
    """Smallest shift whose coverage is within ``slack`` of the best allowed coverage.

    If the zero shift is already within ``stay_slack`` of the best, the ROI
    stays: annotators leave the ROI alone while the spine is inside it.
    """
    score = np.where(bad | ~np.isfinite(grid), -np.inf, grid)
    if not np.isfinite(score).any():
        return Pick((0, 0), (0.0, 0.0), float("nan"), float("nan"), True)
    best = float(score.max())
    zero = float(score[max_shift, max_shift])
    if np.isfinite(zero) and zero >= best - stay_slack:
        return Pick((0, 0), (0.0, 0.0), zero, zero, True)
    yy, xx = np.mgrid[-max_shift : max_shift + 1, -max_shift : max_shift + 1]
    candidates = score >= best - slack
    distance = np.where(candidates, np.hypot(yy, xx), np.inf)
    flat = int(np.argmin(distance))
    py, px = np.unravel_index(flat, distance.shape)
    shift = (int(yy[py, px]), int(xx[py, px]))
    return Pick(shift, (float(shift[0]), float(shift[1])), float(score[py, px]), zero, False)


def containment_veto(
    image: np.ndarray,
    roi: np.ndarray,
    shaft_mask: np.ndarray | None,
    other_spines: np.ndarray | None,
    shift: tuple[int, int],
    *,
    max_drop: float = 0.10,
    params: "TrackParams | None" = None,
) -> bool:
    """True when ``shift`` would push the spine blob out of the ROI.

    Used as a sanity check on a dendrite-based static offset: the offset is
    only accepted if the blob coverage after the move is within ``max_drop``
    of the coverage without the move. The spine image is not used to choose
    the shift, only to reject one that is clearly wrong.
    """
    if shift == (0, 0):
        return False
    p = params or TrackParams()
    blob = spine_blob(
        image,
        roi,
        shaft_mask,
        other_spines,
        smooth=p.blob_smooth,
        frac=p.blob_frac,
        window=p.blob_window,
        min_overlap=p.blob_min_overlap,
        shaft_dilation=p.blob_shaft_dilation,
        cap_above_peak=p.blob_cap_above_peak,
        snap_dendrite=p.blob_snap_dendrite,
    )
    total = float(blob.sum())
    if total <= 0:
        return False
    before = float((roi & blob).sum()) / total
    after = float((translate_mask(roi, *shift) & blob).sum()) / total
    return after < before - max_drop


def _integral_image(image: np.ndarray, sigma: float) -> np.ndarray:
    """Image used by the contrast criterion: raw when sigma <= 0, else highpass."""
    if sigma <= 0:
        return np.asarray(image, dtype=np.float64)
    return highpass(image, sigma)


def _prepare(image: np.ndarray, sigma: float, raw: bool) -> np.ndarray:
    if raw:
        return np.asarray(image, dtype=np.float64)
    return highpass(image, sigma)


@dataclass
class Pick:
    """Result of one grid search."""

    shift: tuple[int, int]
    refined: tuple[float, float]
    peak: float
    zero: float
    gated: bool


def _pick(
    grid: np.ndarray,
    max_shift: int,
    *,
    min_gain: float,
    min_px: float,
    previous: tuple[int, int] | None,
    temporal_penalty: float,
) -> Pick:
    """Best allowed shift of a combined score grid with gain and magnitude gates.

    ``refined`` is the subpixel peak before any gate so callers can fit a
    trajectory through the raw estimates; ``shift`` is the gated integer.
    """
    score = np.array(grid, dtype=np.float64, copy=True)
    if previous is not None and temporal_penalty > 0:
        yy, xx = np.mgrid[-max_shift : max_shift + 1, -max_shift : max_shift + 1]
        score -= temporal_penalty * np.hypot(yy - previous[0], xx - previous[1])
    zero = score[max_shift, max_shift]
    zero = float(zero) if np.isfinite(zero) else float("-inf")
    if not np.isfinite(score).any():
        return Pick((0, 0), (0.0, 0.0), zero, zero, True)
    flat = np.nanargmax(np.where(np.isfinite(score), score, -np.inf))
    py, px = np.unravel_index(int(flat), score.shape)
    peak = float(score[py, px])
    shift = (int(py) - max_shift, int(px) - max_shift)
    fy, fx = subpixel_peak(np.where(np.isfinite(score), score, np.nan), (int(py), int(px)))
    refined = (fy - max_shift, fx - max_shift)
    if shift == (0, 0):
        return Pick(shift, refined, peak, zero, False)
    if np.isfinite(zero) and peak < zero + min_gain:
        return Pick((0, 0), refined, zero, zero, True)
    if min_px > 0 and float(np.hypot(*refined)) < min_px:
        return Pick((0, 0), refined, zero, zero, True)
    return Pick(shift, refined, peak, zero, False)


def fit_post_trajectory(
    times: np.ndarray,
    estimates: np.ndarray,
    *,
    mode: str,
    min_px: float,
    max_shift: int,
) -> np.ndarray:
    """Integer shifts from a smooth fit through per-frame subpixel estimates.

    ``mode`` is ``"linear"`` (least squares line in time, one outlier pass) or
    ``"median"`` (one constant). The GUI interpolates manual ROIs linearly
    between keyframes, so the manual post trajectory is close to a line; a
    fit through noisy per-frame estimates removes most of the jitter.
    Fitted positions closer than ``min_px`` to zero stay at zero.
    """
    est = np.asarray(estimates, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    n = len(t)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)
    fitted = np.zeros_like(est)
    for k in range(2):
        y = est[:, k]
        if mode == "median" or n < 3 or np.ptp(t) <= 0:
            fitted[:, k] = np.median(y)
            continue
        coef = np.polyfit(t, y, 1)
        resid = y - np.polyval(coef, t)
        keep = np.abs(resid) <= max(1.5, 2.0 * np.std(resid))
        if keep.sum() >= 3 and keep.sum() < n:
            coef = np.polyfit(t[keep], y[keep], 1)
        fitted[:, k] = np.polyval(coef, t)
    out = np.round(np.clip(fitted, -max_shift, max_shift)).astype(np.int64)
    if min_px > 0:
        small = np.hypot(fitted[:, 0], fitted[:, 1]) < min_px
        out[small] = 0
    return out


def track_spine(
    stack: np.ndarray,
    phases: list[str],
    spine_mask: np.ndarray,
    shaft_mask: np.ndarray | None,
    other_spines: np.ndarray | None,
    *,
    static: tuple[int, int] = (0, 0),
    params: TrackParams | None = None,
    elapsed_sec: np.ndarray | None = None,
) -> ShiftEstimate:
    """Per-frame integer shift of the auto spine mask that follows the spine head.

    ``static`` is the pre-frame offset of the auto mask (from a reference
    match or zero). The pre-mean image at that position is the template; each
    non-pre frame (one uncaging run = one estimate) is matched inside the
    spine weight plus ``dendrite_lambda`` times the local dendrite NCC.
    With ``params.post_fit`` set, the post frames are replaced by a line or a
    constant fitted through the per-frame subpixel estimates (``elapsed_sec``
    gives the time axis; frame index is used when it is None).
    """
    p = params or TrackParams()
    n_frames = len(stack)
    names = [str(x).strip().lower() for x in phases]
    pre = [t for t, name in enumerate(names) if name == "pre"]
    if not pre:
        pre = [0]
    pre_mean = np.mean(stack[pre], axis=0)
    base_mask = translate_mask(spine_mask, *static)
    sw = spine_weight(base_mask, shaft_mask, dilation=p.spine_dilation, shaft_dilation=p.shaft_dilation)
    dw = dendrite_weight(base_mask, window_radius=p.dendrite_radius, exclude_dilation=p.spine_dilation + 1)
    bad = forbidden_shifts(base_mask, shaft_mask, other_spines, p.max_shift, tolerance=p.overlap_tolerance)
    template_spine = _prepare(pre_mean, p.spine_sigma, p.use_raw_spine)
    template_dend = highpass(pre_mean, p.dendrite_sigma)

    shifts = np.zeros((n_frames, 2), dtype=np.int64)
    scores = np.full(n_frames, np.nan)
    zeros = np.full(n_frames, np.nan)
    refined = np.zeros((n_frames, 2), dtype=np.float64)
    previous: tuple[int, int] | None = None
    for group in frame_groups(names, n_frames):
        if (not p.track_pre and names[group[0]] == "pre") or p.criterion == "none":
            for t in group:
                shifts[t] = static
            continue
        image = stack[group[0]] if len(group) == 1 else np.mean(stack[group], axis=0)
        if p.criterion == "contain":
            pick = _contain_pick(image, base_mask, shaft_mask, other_spines, bad, p)
            previous = pick.shift
            for t in group:
                shifts[t] = (static[0] + pick.shift[0], static[1] + pick.shift[1])
                scores[t] = pick.peak
                zeros[t] = pick.zero
                refined[t] = pick.refined
            continue
        if p.criterion == "integral":
            grid = p.integral_weight * relative_integral_grid(
                _integral_image(image, p.integral_sigma), base_mask, p.max_shift, ring_width=p.ring_width
            )
        else:
            grid = weighted_ncc_grid(template_spine, _prepare(image, p.spine_sigma, p.use_raw_spine), sw, p.max_shift)
            if p.criterion == "both":
                integral = relative_integral_grid(
                    _integral_image(image, p.integral_sigma), base_mask, p.max_shift, ring_width=p.ring_width
                )
                grid = grid + p.integral_weight * np.where(np.isfinite(integral), integral, 0.0)
        if p.dendrite_lambda > 0:
            dend = weighted_ncc_grid(template_dend, highpass(image, p.dendrite_sigma), dw, p.max_shift)
            grid = grid + p.dendrite_lambda * np.where(np.isfinite(dend), dend, 0.0)
        grid = np.where(bad, np.nan, grid)
        pick = _pick(
            grid,
            p.max_shift,
            min_gain=p.min_gain,
            min_px=p.min_px,
            previous=previous,
            temporal_penalty=p.temporal_penalty,
        )
        previous = pick.shift
        for t in group:
            shifts[t] = (static[0] + pick.shift[0], static[1] + pick.shift[1])
            scores[t] = pick.peak
            zeros[t] = pick.zero
            refined[t] = pick.refined
    if p.post_fit != "none":
        post = [t for t, name in enumerate(names) if name == "post"]
        if post:
            times = np.asarray(elapsed_sec, dtype=np.float64)[post] if elapsed_sec is not None else np.asarray(post, dtype=np.float64)
            fitted = fit_post_trajectory(times, refined[post], mode=p.post_fit, min_px=p.min_px, max_shift=p.max_shift)
            for row, t in enumerate(post):
                allowed = not bad[fitted[row, 0] + p.max_shift, fitted[row, 1] + p.max_shift]
                local = tuple(int(v) for v in fitted[row]) if allowed else shifts_local(shifts[t], static)
                shifts[t] = (static[0] + local[0], static[1] + local[1])
    return ShiftEstimate(shifts, scores, zeros)


def shifts_local(total: np.ndarray, static: tuple[int, int]) -> tuple[int, int]:
    """Local part of a total shift."""
    return int(total[0]) - static[0], int(total[1]) - static[1]
