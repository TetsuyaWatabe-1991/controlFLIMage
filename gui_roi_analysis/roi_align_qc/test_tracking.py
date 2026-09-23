"""Synthetic checks for object-shift sign and for the fixed-window failure mode."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking import (  # noqa: E402
    adjacent_window_shifts,
    classify_alignment,
    content_shift_yx,
    delta_over_pre,
    mask_sums,
    masks_from_shift,
    track_patch,
    trajectory_span,
    translate_image,
    translate_mask,
)


def _blob_image(size: int, y: float, x: float, sigma: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
    noise = np.random.default_rng(0).normal(0.0, 0.01, (size, size))
    return (blob + noise).astype(np.float32)


def test_content_shift_matches_ndimage() -> None:
    base = _blob_image(64, 28, 30)
    moved = translate_image(base, 4.0, -3.0)
    dy, dx = content_shift_yx(base, moved)
    assert abs(dy - 4.0) < 0.6, dy
    assert abs(dx - (-3.0)) < 0.6, dx


def test_patch_track_follows_slow_drift() -> None:
    frames = [_blob_image(80, 30 + 3 * t, 40 - 2 * t) for t in range(6)]
    stack = np.stack(frames)
    tracked = track_patch(stack, 30, 40, half=8, search=6, max_step=6)
    for t in range(6):
        assert abs(tracked.shift_yx[t, 0] - 3 * t) < 0.75, tracked.shift_yx[t]
        assert abs(tracked.shift_yx[t, 1] - (-2 * t)) < 0.75, tracked.shift_yx[t]


def test_fixed_window_loses_blob_that_following_keeps() -> None:
    """A spine that walks out of a fixed crop is no longer the thing being registered."""
    frames = [_blob_image(120, 20 + 6 * t, 24) for t in range(6)]
    stack = np.stack(frames)
    fixed, _ = adjacent_window_shifts(stack, 20, 24, half=8, follow=False)
    followed, _ = adjacent_window_shifts(stack, 20, 24, half=8, follow=True)
    true_end = np.array([30.0, 0.0])
    fixed_err = float(np.hypot(*(fixed[-1] - true_end)))
    follow_err = float(np.hypot(*(followed[-1] - true_end)))
    assert follow_err < 2.5, follow_err
    assert fixed_err > follow_err + 8.0, (fixed_err, follow_err, fixed[-1], followed[-1])


def test_phase_only_misses_shift_on_small_crop() -> None:
    """Default phase correlation reports 0 px on a spine-sized crop."""
    from tracking import crop_centered, phase_only_shift_yx

    frame0 = _blob_image(96, 40, 40)
    frame1 = _blob_image(96, 46, 40)
    crop0 = crop_centered(frame0, 40, 40, 8)
    crop1 = crop_centered(frame1, 40, 40, 8)
    phase_dy, phase_dx, phase_err = phase_only_shift_yx(crop0, crop1)
    spatial_dy, spatial_dx = content_shift_yx(crop0, crop1)
    assert abs(spatial_dy - 6.0) < 1.0, spatial_dy
    assert abs(spatial_dx) < 1.0
    assert abs(phase_dy) < 1.0 and abs(phase_dx) < 1.0
    assert phase_err > 0.5


def test_tracked_mask_keeps_blob_sum() -> None:
    frames = []
    shifts = []
    for t in range(4):
        frames.append(_blob_image(48, 16 + 3 * t, 20))
        shifts.append((3 * t, 0))
    stack = np.stack(frames)
    base = translate_mask(np.zeros((48, 48), dtype=bool), 0, 0)
    yy, xx = np.mgrid[0:48, 0:48]
    base = (yy - 16) ** 2 + (xx - 20) ** 2 <= 9
    static_masks = np.repeat(base[None, ...], 4, axis=0)
    moving_masks = masks_from_shift(base, np.array(shifts, dtype=np.float64))
    static_sums = mask_sums(stack, static_masks)
    moving_sums = mask_sums(stack, moving_masks)
    assert moving_sums[-1] > static_sums[-1] * 2.0
    assert abs(moving_sums[-1] / moving_sums[0] - 1.0) < 0.25


def test_track_rejects_jump_to_a_second_blob() -> None:
    frame0 = _blob_image(90, 30, 40)
    frame1 = _blob_image(90, 60, 40)
    stack = np.stack([frame0, frame1])
    held = track_patch(stack, 30, 40, half=6, search=40, max_step=3)
    free = track_patch(stack, 30, 40, half=6, search=40, max_step=None, min_gain=0.0)
    assert abs(free.shift_yx[1, 0] - 30.0) < 2.0, free.shift_yx[1]
    assert abs(held.shift_yx[1, 0]) <= 3.0
    assert abs(held.shift_yx[1, 1]) <= 3.0


def test_delta_over_pre() -> None:
    values = np.array([2.0, 2.0, 4.0, 4.0])
    delta = delta_over_pre(values, np.array([0, 1]), np.array([2, 3]))
    assert abs(delta - 1.0) < 1e-6


def test_classify_names_fixed_window_failure() -> None:
    label = classify_alignment(
        manual_span=6.0,
        track_vs_manual_err=0.8,
        tiff_vs_stored_err=0.4,
        spine_span=8.0,
        stored_span=8.0,
        stored_vs_spine_err=6.0,
        fixed_vs_spine_err=7.0,
        follow_adj_vs_spine_err=1.0,
        fullfov_vs_spine_err=2.0,
        stored_vs_fullfov_err=5.0,
        cancel_residual=6.0,
        same_dir_residual=6.0,
        manual_residual=6.0,
    )
    assert label == "fixed_window_adjacent_lost_spine"


def test_classify_stable() -> None:
    label = classify_alignment(
        manual_span=0.4,
        track_vs_manual_err=0.3,
        tiff_vs_stored_err=0.2,
        spine_span=1.0,
        stored_span=1.0,
        stored_vs_spine_err=0.5,
        fixed_vs_spine_err=0.5,
        follow_adj_vs_spine_err=0.4,
        fullfov_vs_spine_err=0.6,
        stored_vs_fullfov_err=0.4,
        cancel_residual=0.4,
        same_dir_residual=2.0,
        manual_residual=0.4,
    )
    assert label == "stable"
    assert trajectory_span(np.array([[0.0, 0.0], [3.0, 4.0]])) == 5.0


def main() -> None:
    tests = [
        test_content_shift_matches_ndimage,
        test_patch_track_follows_slow_drift,
        test_track_rejects_jump_to_a_second_blob,
        test_fixed_window_loses_blob_that_following_keeps,
        test_phase_only_misses_shift_on_small_crop,
        test_tracked_mask_keeps_blob_sum,
        test_delta_over_pre,
        test_classify_names_fixed_window_failure,
        test_classify_stable,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
