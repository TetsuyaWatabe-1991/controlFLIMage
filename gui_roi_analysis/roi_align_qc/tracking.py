"""Shift tracking used to diagnose alignment and to follow a spine patch.

All object-shift vectors are (dy, dx) in pixels. Positive dy means the
object moved toward larger row indices relative to frame 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift as ndimage_shift
from skimage.feature import match_template
from skimage.registration import phase_cross_correlation


@dataclass
class TrackResult:
    """Per-frame object displacement relative to frame 0, plus match scores."""

    shift_yx: np.ndarray
    score: np.ndarray


def crop_centered(image: np.ndarray, center_y: float, center_x: float, half: int) -> np.ndarray:
    """Return a square crop of width ``2 * half + 1``, zero-padded at the border."""
    half = int(half)
    size = 2 * half + 1
    out = np.zeros((size, size), dtype=np.float32)
    height, width = image.shape
    cy = int(round(center_y))
    cx = int(round(center_x))
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    src_y0, src_y1 = max(0, y0), min(height, y1)
    src_x0, src_x1 = max(0, x0), min(width, x1)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    out[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def _pcc_object_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    normalization: str | None,
) -> tuple[float, float, float]:
    """Return (dy, dx, error) for one phase_cross_correlation call.

    The skimage shift registers ``moving`` onto ``reference``. Object
    displacement is the opposite sign: content that moved toward +y is
    brought back by a negative registration shift.
    """
    shift, error, _ = phase_cross_correlation(
        np.asarray(reference, dtype=np.float32),
        np.asarray(moving, dtype=np.float32),
        upsample_factor=4,
        normalization=normalization,
    )
    return float(-shift[0]), float(-shift[1]), float(error)


def content_shift_yx(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    """Object displacement of ``moving`` relative to ``reference``.

    Uses spatial cross-correlation (``normalization=None``). The default
    phase-only normalization in scikit-image 0.24 returns a 0 px shift on a
    small crop even when the bright object has clearly moved.
    """
    dy, dx, _ = _pcc_object_shift(reference, moving, normalization=None)
    return dy, dx


def phase_only_shift_yx(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    """Object displacement from scikit-image's default phase-only correlation.

    ``error`` near 1 means the peak was not trustworthy. On spine-sized crops
    this often comes back as (0, 0) while spatial correlation still sees the move.
    """
    return _pcc_object_shift(reference, moving, normalization="phase")


def translate_image(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Move image content by (dy, dx). Positive dy moves content down."""
    return ndimage_shift(
        np.asarray(image, dtype=np.float32),
        shift=(float(dy), float(dx)),
        order=1,
        mode="constant",
        cval=0.0,
    )


def translate_mask(mask: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Integer-pixel translation of a boolean mask. Vacated pixels stay False."""
    dy_i = int(round(dy))
    dx_i = int(round(dx))
    height, width = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    y_src0, y_src1 = max(0, -dy_i), min(height, height - dy_i)
    y_dst0, y_dst1 = max(0, dy_i), min(height, height + dy_i)
    x_src0, x_src1 = max(0, -dx_i), min(width, width - dx_i)
    x_dst0, x_dst1 = max(0, dx_i), min(width, width + dx_i)
    if y_src1 <= y_src0 or x_src1 <= x_src0:
        return out
    out[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
    return out


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return (y, x) centroid. Empty masks return (nan, nan)."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return float("nan"), float("nan")
    return float(ys.mean()), float(xs.mean())


def centroid_trajectory(mask_stack: np.ndarray) -> np.ndarray:
    """Return (n_frames, 2) centroids minus the frame-0 centroid."""
    n_frames = mask_stack.shape[0]
    cents = np.full((n_frames, 2), np.nan, dtype=np.float64)
    for i in range(n_frames):
        cents[i] = mask_centroid(mask_stack[i] > 0)
    origin = cents[0].copy()
    return cents - origin


def trajectory_span(shift_yx: np.ndarray) -> float:
    """Max distance from frame 0 over a (n_frames, 2) trajectory."""
    if shift_yx.size == 0:
        return float("nan")
    dist = np.hypot(shift_yx[:, 0], shift_yx[:, 1])
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def mean_trajectory_error(left: np.ndarray, right: np.ndarray) -> float:
    """Mean Euclidean error between two (n_frames, 2) trajectories."""
    n_frames = min(len(left), len(right))
    if n_frames == 0:
        return float("nan")
    err = np.hypot(left[:n_frames, 0] - right[:n_frames, 0], left[:n_frames, 1] - right[:n_frames, 1])
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def track_patch(
    stack: np.ndarray,
    center_y: float,
    center_x: float,
    *,
    half: int = 12,
    search: int = 8,
    max_step: float | None = 3.0,
    min_gain: float = 0.02,
) -> TrackResult:
    """Follow a frame-0 image patch with normalized cross-correlation.

    The search window is recentered on the last accepted position, so a slow
    drift can exceed ``search`` as long as each frame stays within it.
    A candidate step is rejected when it is longer than ``max_step`` or when
    its score does not beat staying still by ``min_gain``. Rejected steps are
    how a neighboring bright spine steals the track in one frame.
    """
    n_frames = stack.shape[0]
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    scores = np.ones(n_frames, dtype=np.float64)
    template = crop_centered(stack[0], center_y, center_x, half)
    if float(np.std(template)) < 1e-6:
        scores[:] = 0.0
        return TrackResult(shifts, scores)

    acc_y = 0.0
    acc_x = 0.0
    for t in range(1, n_frames):
        search_img = crop_centered(stack[t], center_y + acc_y, center_x + acc_x, half + search)
        result = match_template(search_img, template)
        if not np.isfinite(result).any():
            shifts[t] = (acc_y, acc_x)
            scores[t] = 0.0
            continue
        peak = np.unravel_index(int(np.nanargmax(result)), result.shape)
        dy = float(peak[0] - search)
        dx = float(peak[1] - search)
        peak_score = float(result[peak])
        stay_score = float(result[search, search]) if np.isfinite(result[search, search]) else -np.inf
        step = float(np.hypot(dy, dx))
        accept = peak_score >= stay_score + min_gain
        if max_step is not None and step > max_step:
            accept = False
        if accept:
            acc_y += dy
            acc_x += dx
            scores[t] = peak_score
        else:
            scores[t] = stay_score
        shifts[t] = (acc_y, acc_x)
    return TrackResult(shifts, scores)


def max_frame_step(shift_yx: np.ndarray | None) -> float:
    """Largest frame-to-frame jump in a (n_frames, 2) trajectory."""
    if shift_yx is None or len(shift_yx) < 2:
        return float("nan")
    delta = np.diff(shift_yx, axis=0)
    dist = np.hypot(delta[:, 0], delta[:, 1])
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def adjacent_window_shifts(
    stack: np.ndarray,
    center_y: float,
    center_x: float,
    *,
    half: int = 30,
    follow: bool,
    normalization: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative adjacent-frame correlation inside a local window.

    ``follow=False`` keeps the window on the frame-0 center. That is the
    behavior of roi_adjacent: once the spine leaves the window, later frames
    correlate whatever texture remains.
    ``follow=True`` recenters the window on the accumulated shift before the
    next pair, so the same feature stays inside the crop.

    ``normalization='phase'`` reproduces scikit-image's default, which is what
    ``FLIMageAlignment`` calls. ``None`` is ordinary cross-correlation.

    Returns (shift_yx, per_frame_error). Frame 0 error is 0.
    """
    n_frames = stack.shape[0]
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    errors = np.zeros(n_frames, dtype=np.float64)
    acc_y = 0.0
    acc_x = 0.0
    window_y = float(center_y)
    window_x = float(center_x)
    prev = crop_centered(stack[0], window_y, window_x, half)
    for t in range(1, n_frames):
        cur = crop_centered(stack[t], window_y, window_x, half)
        dy, dx, err = _pcc_object_shift(prev, cur, normalization=normalization)
        errors[t] = err
        acc_y += dy
        acc_x += dx
        shifts[t] = (acc_y, acc_x)
        if follow:
            window_y = float(center_y) + acc_y
            window_x = float(center_x) + acc_x
            prev = crop_centered(stack[t], window_y, window_x, half)
        else:
            prev = cur
    return shifts, errors


def full_fov_shifts(stack: np.ndarray) -> np.ndarray:
    """Object displacement of each frame versus frame 0, using the full field."""
    n_frames = stack.shape[0]
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    reference = np.asarray(stack[0], dtype=np.float32)
    for t in range(1, n_frames):
        shifts[t] = content_shift_yx(reference, stack[t])
    return shifts


def pairwise_content_shifts(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Per-frame content displacement from ``before`` to ``after``."""
    n_frames = min(before.shape[0], after.shape[0])
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    for t in range(n_frames):
        shifts[t] = content_shift_yx(before[t], after[t])
    origin = shifts[0].copy()
    return shifts - origin


def best_sign_error(candidate: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """Return (mean error, sign) where sign is +1 or -1 applied to candidate."""
    err_pos = mean_trajectory_error(candidate, target)
    err_neg = mean_trajectory_error(-candidate, target)
    if not np.isfinite(err_neg) or (np.isfinite(err_pos) and err_pos <= err_neg):
        return err_pos, 1
    return err_neg, -1


def mask_sums(stack: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Sum of image pixels inside each frame's mask."""
    n_frames = min(stack.shape[0], masks.shape[0])
    out = np.full(n_frames, np.nan, dtype=np.float64)
    for i in range(n_frames):
        sel = masks[i] > 0
        if np.any(sel):
            out[i] = float(np.sum(stack[i][sel]))
    return out


def delta_over_pre(values: np.ndarray, pre_idx: np.ndarray, post_idx: np.ndarray) -> float:
    """Mean(post) / mean(pre) - 1. NaN when either side is empty or pre is 0."""
    pre = values[pre_idx]
    post = values[post_idx]
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    if pre.size == 0 or post.size == 0:
        return float("nan")
    pre_mean = float(np.mean(pre))
    if abs(pre_mean) < 1e-12:
        return float("nan")
    return float(np.mean(post) / pre_mean - 1.0)


def series_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation on frames where both values are finite."""
    n_frames = min(len(left), len(right))
    a = np.asarray(left[:n_frames], dtype=np.float64)
    b = np.asarray(right[:n_frames], dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def masks_from_shift(base_mask: np.ndarray, shift_yx: np.ndarray) -> np.ndarray:
    """Translate one 2D mask by each row of ``shift_yx``."""
    stack = np.zeros((len(shift_yx),) + base_mask.shape, dtype=bool)
    for i, (dy, dx) in enumerate(shift_yx):
        stack[i] = translate_mask(base_mask, float(dy), float(dx))
    return stack


def classify_alignment(
    *,
    manual_span: float,
    track_vs_manual_err: float,
    tiff_vs_stored_err: float,
    spine_span: float,
    stored_span: float,
    stored_vs_spine_err: float,
    fixed_vs_spine_err: float,
    follow_adj_vs_spine_err: float,
    fullfov_vs_spine_err: float,
    stored_vs_fullfov_err: float,
    cancel_residual: float,
    same_dir_residual: float,
    manual_residual: float,
) -> str:
    """Pick one cause label from trajectory errors. Thresholds are in pixels."""
    if manual_span < 1.5 and track_vs_manual_err < 2.0:
        return "stable"
    if np.isfinite(tiff_vs_stored_err) and tiff_vs_stored_err > 2.0:
        return "recorded_shift_disagrees_with_tiff"
    if track_vs_manual_err > 3.0:
        return "human_edit_not_rigid_drift"
    if (
        np.isfinite(fixed_vs_spine_err)
        and np.isfinite(follow_adj_vs_spine_err)
        and fixed_vs_spine_err > follow_adj_vs_spine_err + 2.0
        and fixed_vs_spine_err > 3.0
    ):
        return "fixed_window_adjacent_lost_spine"
    if (
        np.isfinite(stored_vs_fullfov_err)
        and np.isfinite(stored_vs_spine_err)
        and stored_vs_fullfov_err + 1.0 < stored_vs_spine_err
        and fullfov_vs_spine_err > 3.0
    ):
        return "alignment_followed_field_not_spine"
    if (
        np.isfinite(same_dir_residual)
        and np.isfinite(cancel_residual)
        and same_dir_residual + 1.0 < cancel_residual
        and manual_residual > 2.0
    ):
        return "shift_moves_with_spine_instead_of_against_it"
    if stored_vs_spine_err > 3.0 and stored_span + 1.5 < spine_span:
        return "spine_motion_larger_than_applied_shift"
    if stored_vs_spine_err > 3.0:
        return "applied_shift_missed_spine_motion"
    if manual_residual > 2.0 and cancel_residual < 2.0:
        return "shift_should_cancel_but_image_still_drifts"
    return "residual_drift_shift_partially_matches"
