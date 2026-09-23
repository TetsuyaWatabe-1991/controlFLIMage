"""The tiled alignment view must undo a known shift and keep the manual ROI."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_align_frames import (  # noqa: E402
    BANNER_H,
    COLS,
    PANEL_SPECS,
    PANEL_W,
    ROWS,
    frames_to_keep,
    map_manual_mask,
    render_alignment_grid,
    _plot_xy,
    plot_limits,
    plotted_frame_indices,
    render_quant_grid,
    roi_ff0_traces,
    roi_mean,
    to_ff0,
)
from tracking import mask_centroid, translate_image, translate_mask  # noqa: E402


def test_manual_alignment_returns_the_spine_and_the_roi() -> None:
    mask0 = np.zeros((64, 64), dtype=bool)
    mask0[20:28, 30:38] = True
    raw = mask0.astype(np.float32)
    applied = np.array([1.0, -2.0])
    manual_traj = np.array([4.0, -3.0])
    target = manual_traj - applied
    view_shift = -target
    drifted = translate_image(raw, float(target[0]), float(target[1]))
    aligned = translate_image(drifted, float(view_shift[0]), float(view_shift[1]))
    back_y, back_x = mask_centroid(aligned > 0.5)
    origin_y, origin_x = mask_centroid(mask0)
    assert abs(back_y - origin_y) < 0.6
    assert abs(back_x - origin_x) < 0.6

    manual_on_after = translate_mask(mask0, float(manual_traj[0]), float(manual_traj[1]))
    mapped = map_manual_mask(manual_on_after, view_shift, applied)
    assert np.array_equal(mapped, mask0)

    # The saved alignment already contains `applied`, so the mask is not moved again.
    assert np.array_equal(map_manual_mask(manual_on_after, applied, applied), manual_on_after)


def test_only_first_and_last_uncaging_frames_are_kept() -> None:
    phases = ["pre", "pre", "uncaging", "unc", "uncaging", "post", "post"]
    assert frames_to_keep(phases) == [0, 1, 2, 4, 5, 6]
    assert frames_to_keep(["pre", "uncaging", "post"]) == [0, 1, 2]
    assert frames_to_keep(["pre", "post"]) == [0, 1]


def test_ff0_uses_the_pre_mean() -> None:
    scaled = to_ff0(np.array([2.0, 2.0, 4.0]), np.array([0, 1]))
    assert abs(scaled[2] - 2.0) < 1e-6


def test_following_shift_keeps_roi_intensity() -> None:
    blob = np.zeros((48, 48), dtype=np.float32)
    blob[8:14, 16:22] = 2.0
    mask = blob > 0
    before = np.stack([translate_image(blob, 6.0 * t, 0.0) for t in range(4)])
    after = before.copy()
    manual = np.stack([translate_mask(mask, 6.0 * t, 0.0) for t in range(4)])
    shifts = {key: np.zeros((4, 2), dtype=np.float64) for key, _label in PANEL_SPECS}
    shifts["manual"] = np.array([[0.0, 0.0], [-6.0, 0.0], [-12.0, 0.0], [-18.0, 0.0]])
    phases = ["pre", "post", "post", "post"]
    manual_ff0, method_ff0 = roi_ff0_traces(before, after, manual, shifts, phases)
    assert abs(manual_ff0[-1] - 1.0) < 0.15
    assert method_ff0["raw"][-1] < 0.3
    assert method_ff0["manual"][-1] > 0.8
    assert roi_mean(blob, mask) > 1.5


def test_uncaging_values_do_not_set_the_plot_scale() -> None:
    phases = ["pre", "uncaging", "uncaging", "post"]
    manual = np.array([1.0, 80.0, 90.0, 1.4])
    traces = {key: manual.copy() for key, _label in PANEL_SPECS}
    ymin, ymax = plot_limits(manual, traces, phases)
    assert ymax < 5.0
    assert ymin < 1.0


def test_set_keys_keep_each_spine_once() -> None:
    import pandas as pd

    from export_align_frames import _set_keys

    frame = pd.DataFrame(
        {
            "group": ["a", "a", "a", "b"],
            "nth_set_label": [0, 0, -1, 2],
        }
    )
    assert _set_keys(frame) == [("a", 0), ("b", 2)]


def test_pre_and_post_use_the_full_x_axis() -> None:
    phases = ["pre", "uncaging", "uncaging", "post"]
    shown = plotted_frame_indices(phases)
    assert shown == [0, 3]
    left = _plot_xy(0, 1.0, len(shown), 0.0, 2.0, 40, 10, 200, 100)
    right = _plot_xy(len(shown) - 1, 1.0, len(shown), 0.0, 2.0, 40, 10, 200, 100)
    assert left is not None and left[0] == 40
    assert right is not None and right[0] == 200


def test_quant_strip_matches_image_height() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:16, 10:18] = 1.0
    mask = image > 0
    keys = [key for key, _label in PANEL_SPECS]
    images = {key: image for key in keys}
    masks = {key: mask for key in keys}
    shifts = {key: np.zeros((1, 2)) for key in keys}
    canvas = render_alignment_grid(0, images, mask, masks, shifts, 12.0, 14.0, 0.0, 1.0, "synthetic frame 0")
    manual = np.ones(4)
    traces = {key: np.linspace(1.0, 1.5, 4) for key in keys}
    phases = ["pre", "uncaging", "uncaging", "post"]
    panel_h = (canvas.shape[0] - BANNER_H) // ROWS
    plots = render_quant_grid(manual, traces, phases, 2, panel_h, PANEL_W)
    assert plots.shape[0] == canvas.shape[0]
    assert plots.shape[1] == COLS * PANEL_W


def test_grid_has_one_tile_per_method() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:16, 10:18] = 1.0
    mask = image > 0
    keys = [
        "raw",
        "current",
        "manual",
        "full_spatial",
        "full_phase",
        "highpass_spatial",
        "edge_spatial",
        "tile_median",
        "farneback",
        "pyramidal_lucas_kanade",
        "iterative_lucas_kanade",
        "tvl1",
        "orb_ransac",
    ]
    images = {key: image for key in keys}
    masks = {key: mask for key in keys}
    shifts = {key: np.zeros((1, 2)) for key in keys}
    canvas = render_alignment_grid(
        0,
        images,
        mask,
        masks,
        shifts,
        12.0,
        14.0,
        0.0,
        1.0,
        "synthetic frame 0",
    )
    assert canvas.ndim == 3 and canvas.shape[2] == 3
    assert canvas.shape[1] % COLS == 0
    assert (canvas.shape[0] - 28) % ROWS == 0


def main() -> None:
    test_manual_alignment_returns_the_spine_and_the_roi()
    test_only_first_and_last_uncaging_frames_are_kept()
    test_ff0_uses_the_pre_mean()
    test_uncaging_values_do_not_set_the_plot_scale()
    test_set_keys_keep_each_spine_once()
    test_pre_and_post_use_the_full_x_axis()
    test_following_shift_keeps_roi_intensity()
    test_quant_strip_matches_image_height()
    test_grid_has_one_tile_per_method()
    print("PASS test_export_align_frames")


if __name__ == "__main__":
    main()
