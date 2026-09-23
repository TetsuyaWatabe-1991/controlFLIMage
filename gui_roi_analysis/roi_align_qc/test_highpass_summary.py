"""The four-frame summary keeps pre and post and drops uncaging."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_align_frames import SetArrays  # noqa: E402
from export_highpass_summary import SCALE, render_summary, summary_frame_indices  # noqa: E402


def test_summary_frames_exclude_uncaging() -> None:
    phases = ["pre", "pre", "uncaging", "uncaging", "post", "post"]
    assert summary_frame_indices(phases) == (0, 4, 5, [0, 1, 4, 5])
    assert summary_frame_indices(["pre", "uncaging"]) is None


def test_summary_image_has_four_panels() -> None:
    frame = np.zeros((32, 32), dtype=np.float32)
    frame[8:14, 10:16] = 4.0
    before = np.stack([frame, frame, frame])
    manual = np.stack([(frame > 0), (frame > 0), (frame > 0)])
    arrays = SetArrays(
        session="s",
        group="g",
        set_label=1,
        before=before,
        after=before.copy(),
        applied=np.zeros((3, 2)),
        target=np.zeros((3, 2)),
        manual=manual,
        phase=["pre", "uncaging", "post"],
    )
    canvas = render_summary(arrays, np.zeros((3, 2)))
    assert canvas is not None
    assert canvas.shape[1] == 32 * SCALE * 4


def main() -> None:
    test_summary_frames_exclude_uncaging()
    test_summary_image_has_four_panels()
    print("PASS test_highpass_summary")


if __name__ == "__main__":
    main()
