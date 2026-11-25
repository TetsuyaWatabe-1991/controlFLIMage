"""Spine-template / containment tracking on synthetic images."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manual_shift_dataset import translate_mask  # noqa: E402
from spine_template_tracking import (  # noqa: E402
    TrackParams,
    containment_veto,
    coverage_grid,
    fit_post_trajectory,
    forbidden_shifts,
    locate_static,
    minimal_containing_shift,
    spine_blob,
    spine_weight,
    track_spine,
)
from test_local_dendrite_shift import _scene, _shift_image  # noqa: E402


def _scene_with_shaft() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, mask = _scene()
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    shaft_y = 70 + 8 * np.sin(xx[0] / 17.0)
    shaft = np.abs(yy - shaft_y[None, :]) <= 2.0
    return image, mask, shaft


def _move_spine(image: np.ndarray, mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Move only the spine head (not the dendrite) by (dy, dx)."""
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    spine = 2.5 * np.exp(-((yy - 52) ** 2 + (xx - 60) ** 2) / (2 * 3.0**2))
    moved = 2.5 * np.exp(-((yy - 52 - dy) ** 2 + (xx - 60 - dx) ** 2) / (2 * 3.0**2))
    return image - spine + moved


def test_spine_weight_excludes_shaft() -> None:
    _, mask, shaft = _scene_with_shaft()
    weight = spine_weight(mask, shaft, dilation=2, shaft_dilation=1)
    assert weight.sum() >= mask.sum()
    assert not (weight.astype(bool) & shaft).any()


def test_forbidden_shifts_block_shaft() -> None:
    _, mask, _ = _scene_with_shaft()
    shaft = np.zeros_like(mask)
    shaft[60:64, :] = True  # 4 px below the mask (rows 48..56)
    bad = forbidden_shifts(mask, shaft, None, 6)
    assert not bad[6, 6]
    assert not bad[6 + 3, 6]
    # Moving the mask 5 px down lands on the shaft.
    assert bad[6 + 5, 6]


def test_template_tracking_follows_spine_only_motion() -> None:
    image, mask, shaft = _scene_with_shaft()
    stack = np.stack([image, image, _move_spine(image, mask, 2, -2), _move_spine(image, mask, 3, -3)])
    phases = ["pre", "pre", "post", "post"]
    params = TrackParams(dendrite_lambda=0.0, min_px=0.0, min_gain=0.0)
    estimate = track_spine(stack, phases, mask, shaft, None, params=params)
    assert estimate.shift[:2].tolist() == [[0, 0], [0, 0]]
    assert estimate.shift[2].tolist() == [2, -2], estimate.shift[2]
    assert estimate.shift[3].tolist() == [3, -3], estimate.shift[3]


def test_template_tracking_ignores_brightening() -> None:
    image, mask, shaft = _scene_with_shaft()
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    spine = np.exp(-((yy - 52) ** 2 + (xx - 60) ** 2) / (2 * 3.0**2))
    bright = image + 5.0 * spine  # F/F0 ~ 3 with no motion
    estimate = track_spine(np.stack([image, bright]), ["pre", "post"], mask, shaft, None, params=TrackParams(min_px=0.0))
    assert estimate.shift[1].tolist() == [0, 0], estimate.shift[1]


def test_containment_moves_only_when_needed() -> None:
    image, mask, shaft = _scene_with_shaft()
    params = TrackParams(static_criterion="contain", criterion="contain")
    blob = spine_blob(image, mask, shaft, None)
    assert blob.any() and not (blob & shaft).any()
    grid = coverage_grid(blob, mask, 6)
    pick = minimal_containing_shift(grid, forbidden_shifts(mask, shaft, None, 6), 6)
    assert pick.shift == (0, 0)
    # Spine 3 px outside the ROI: the ROI follows, but not further than needed.
    moved = _move_spine(image, mask, 0, 4)
    static = locate_static(moved, mask, shaft, None, params=params)
    assert 2 <= static.shift[1] <= 4 and static.shift[0] == 0, static.shift
    assert not containment_veto(image, mask, shaft, None, (0, 0))
    assert containment_veto(image, mask, shaft, None, (0, 6))


def test_fit_post_trajectory_denoises() -> None:
    times = np.arange(6, dtype=float)
    estimates = np.column_stack([np.array([0.0, 1.0, 0.0, 1.0, 2.0, 1.0]), np.linspace(0, -3, 6)])
    fitted = fit_post_trajectory(times, estimates, mode="linear", min_px=0.0, max_shift=6)
    assert fitted[-1].tolist() == [1, -3], fitted[-1]
    constant = fit_post_trajectory(times, estimates, mode="median", min_px=0.0, max_shift=6)
    assert (constant == constant[0]).all()
    gated = fit_post_trajectory(times, 0.4 * np.ones((6, 2)), mode="median", min_px=1.5, max_shift=6)
    assert not gated.any()


def test_post_fit_in_tracking_is_linear() -> None:
    image, mask, shaft = _scene_with_shaft()
    frames = [image, image] + [_move_spine(image, mask, 0, k) for k in (1, 1, 2, 3, 3)]
    phases = ["pre", "pre", "post", "post", "post", "post", "post"]
    params = TrackParams(dendrite_lambda=0.0, min_px=0.0, min_gain=0.0, post_fit="linear")
    estimate = track_spine(np.stack(frames), phases, mask, shaft, None, params=params)
    dx = estimate.shift[2:, 1]
    assert (np.diff(dx) >= 0).all(), dx
    assert dx[-1] >= 2, dx
    assert translate_mask(mask, *estimate.shift[-1]).sum() == mask.sum()


def main() -> None:
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()
