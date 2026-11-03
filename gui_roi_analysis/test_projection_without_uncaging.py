# -*- coding: utf-8 -*-
"""Headless test: the ROI GUI overview projection leaves out the uncaging frames.

Usage:
    python test_projection_without_uncaging.py
"""

import numpy as np

from gui_integration_tiff_only import projection_without_uncaging


def _stack():
    # 3 pre, 4 uncaging (very bright spot), 2 post; (t, y, x)
    s = np.zeros((9, 10, 10), np.uint16)
    s[:3, 2, 2] = 50          # pre
    s[3:7, 5, 5] = 1000       # uncaging artefact
    s[7:, 8, 8] = 60          # post
    return s


def test_phases_from_frame_info():
    phases = ["pre"] * 3 + ["unc"] * 4 + ["post"] * 2
    img, n = projection_without_uncaging(_stack(), phases)
    assert n == 5 and img[5, 5] == 0 and img[2, 2] == 50 and img[8, 8] == 60


def test_counts_without_frame_info_and_4d():
    s4 = _stack()[:, None].repeat(3, axis=1)          # (t, z, y, x)
    img, n = projection_without_uncaging(s4, None, n_pre=3, n_unc=4)
    assert img.shape == (10, 10) and n == 5 and img[5, 5] == 0


def test_all_uncaging_falls_back_to_all_frames():
    img, n = projection_without_uncaging(_stack(), ["unc"] * 9)
    assert n == 9 and img[5, 5] == 1000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all tests passed")
