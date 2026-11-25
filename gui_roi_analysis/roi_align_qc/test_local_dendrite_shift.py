"""Dendrite-based ROI shift estimators on synthetic spine/dendrite images."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_dendrite_shift import (  # noqa: E402
    best_shift,
    dendrite_weight,
    estimate_shifts,
    estimate_two_stage,
    frame_groups,
    masks_from_shifts,
    smooth_within_phase,
    weighted_ncc_grid,
)
from manual_shift_dataset import mask_translation, translate_mask  # noqa: E402


def _scene(size: int = 128, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dendrite with texture plus one spine. Returns (image, spine_mask)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    image = np.zeros((size, size), dtype=np.float32)
    # Curved shaft with varicosities so the match is not ambiguous along it.
    shaft_y = 70 + 8 * np.sin(xx[0] / 17.0)
    for x in range(size):
        image[:, x] += 3.0 * np.exp(-((yy[:, x] - shaft_y[x]) ** 2) / (2 * 2.5**2))
    for cx in (20, 47, 71, 99, 118):
        image += 2.0 * np.exp(-((yy - shaft_y[cx]) ** 2 + (xx - cx) ** 2) / (2 * 3.0**2))
    # Spine head above the shaft.
    spine = np.exp(-((yy - 52) ** 2 + (xx - 60) ** 2) / (2 * 3.0**2))
    image += 2.5 * spine
    image += 0.05 * rng.standard_normal(image.shape).astype(np.float32)
    mask = spine > np.exp(-0.5 * (4.0 / 3.0) ** 2)
    return image, mask


def _shift_image(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(image)
    src = translate_mask(np.ones(image.shape, bool), dy, dx)
    ys, xs = np.nonzero(src)
    out[ys, xs] = image[ys - dy, xs - dx]
    return out


def test_mask_translation_roundtrip() -> None:
    _, mask = _scene()
    moved = translate_mask(mask, 3, -2)
    shift, pure = mask_translation(mask, moved)
    assert shift == (3, -2) and pure


def test_weighted_ncc_recovers_shift() -> None:
    image, mask = _scene()
    weight = dendrite_weight(mask, window_radius=0, exclude_dilation=3)
    moved = _shift_image(image, 2, -3)
    grid = weighted_ncc_grid(image, moved, weight, 6)
    shift, peak, zero = best_shift(grid, 6, min_gain=0.02)
    assert shift == (2, -3), shift
    assert peak > zero


def test_estimate_shifts_follows_field() -> None:
    image, mask = _scene()
    stack = np.stack([image, _shift_image(image, 1, 0), _shift_image(image, -2, 3)])
    estimate = estimate_shifts(image, stack, mask, window_radius=0)
    assert estimate.shift.tolist() == [[0, 0], [1, 0], [-2, 3]], estimate.shift.tolist()


def test_two_stage_static_plus_drift() -> None:
    image, mask = _scene()
    reference = image
    static = (2, 1)
    pre = _shift_image(image, *static)
    post1 = _shift_image(image, static[0] + 1, static[1] - 2)
    post2 = _shift_image(image, static[0] + 2, static[1] - 3)
    stack = np.stack([pre, pre, post1, post2])
    phases = ["pre", "pre", "post", "post"]
    estimate = estimate_two_stage(reference, stack, mask, phases, static_radius=0, local_radius=30)
    assert estimate.shift.tolist() == [[2, 1], [2, 1], [3, -1], [4, -2]], estimate.shift.tolist()
    # Gates: a 1-px static offset is dropped with static_min_px=1.5.
    small = _shift_image(image, 1, 0)
    gated = estimate_two_stage(reference, np.stack([small, small]), mask, ["pre", "pre"], static_min_px=1.5)
    assert gated.shift.tolist() == [[0, 0], [0, 0]]
    ungated = estimate_two_stage(reference, np.stack([small, small]), mask, ["pre", "pre"], static_min_px=0.0)
    assert ungated.shift.tolist() == [[1, 0], [1, 0]]
    # use_static=False trusts the auto position on pre and only tracks drift.
    drift = estimate_two_stage(reference, stack, mask, phases, use_static=False, local_radius=0)
    assert drift.shift.tolist() == [[0, 0], [0, 0], [1, -2], [2, -3]], drift.shift.tolist()


def test_uncaging_run_shares_one_estimate() -> None:
    phases = ["pre", "unc", "unc", "unc", "post", "post"]
    groups = frame_groups(phases, len(phases))
    assert groups == [[0], [1, 2, 3], [4], [5]]
    shifts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [2, 2]])
    smoothed = smooth_within_phase(shifts, phases)
    assert smoothed[1].tolist() == smoothed[2].tolist() == smoothed[3].tolist()


def test_masks_from_shifts_keeps_area() -> None:
    _, mask = _scene()
    masks = masks_from_shifts(mask, np.array([[0, 0], [2, -1], [-3, 4]]))
    assert masks.shape[0] == 3
    assert all(int(m.sum()) == int(mask.sum()) for m in masks)


def main() -> None:
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()
