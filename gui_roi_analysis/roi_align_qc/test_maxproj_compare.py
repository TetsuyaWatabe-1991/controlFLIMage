"""Max-projection tiles are two rows, and the RGB overlay assigns the three frames."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_align_frames import SetArrays  # noqa: E402
from export_highpass_summary import SCALE  # noqa: E402
from export_maxproj_compare import (  # noqa: E402
    METHOD_ORDER,
    max_projection,
    render_maxproj_compare,
    rgb_overlay,
)


def test_rgb_overlay_assigns_blue_green_red() -> None:
    pre = np.zeros((16, 16), dtype=np.float32)
    post_first = np.zeros_like(pre)
    post_last = np.zeros_like(pre)
    pre[4, 4] = 10.0
    post_first[4, 8] = 10.0
    post_last[8, 4] = 10.0
    color = rgb_overlay(pre, post_first, post_last)
    assert color[4, 4, 0] > color[4, 4, 1]
    assert color[4, 8, 1] > color[4, 8, 0]
    assert color[8, 4, 2] > color[8, 4, 0]


def test_max_projection_skips_a_frame_that_is_not_listed() -> None:
    dim = np.zeros((8, 8), dtype=np.float32)
    bright = np.zeros_like(dim)
    bright[2, 2] = 9.0
    stack = np.stack([dim, bright, dim])
    projected = max_projection(stack, np.zeros((3, 2)), [0, 2])
    assert projected[2, 2] == 0.0


def test_compare_image_is_two_rows_of_four_methods() -> None:
    frame = np.zeros((24, 24), dtype=np.float32)
    frame[6:10, 8:12] = 3.0
    before = np.stack([frame, frame, frame])
    manual = np.stack([(frame > 0) for _ in range(3)])
    arrays = SetArrays(
        session="s",
        group="g",
        set_label=0,
        before=before,
        after=before.copy(),
        applied=np.zeros((3, 2)),
        target=np.zeros((3, 2)),
        manual=manual,
        phase=["pre", "uncaging", "post"],
    )
    shifts = {name: np.zeros((3, 2)) for name in METHOD_ORDER}
    canvas = render_maxproj_compare(arrays, shifts)
    assert canvas is not None
    assert canvas.shape[1] == 24 * SCALE * 4
    assert canvas.shape[0] > 24 * SCALE * 2


def main() -> None:
    test_rgb_overlay_assigns_blue_green_red()
    test_max_projection_skips_a_frame_that_is_not_listed()
    test_compare_image_is_two_rows_of_four_methods()
    print("PASS test_maxproj_compare")


if __name__ == "__main__":
    main()
