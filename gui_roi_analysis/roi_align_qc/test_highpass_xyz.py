"""Highpass XY/YZ registration recovers a known 3D shift."""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import shift as ndimage_shift

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from highpass_xyz import apply_registration_zyx, highpass_registration_zyx  # noqa: E402


def test_highpass_recovers_zyx_shift() -> None:
    reference = np.zeros((16, 48, 48), dtype=np.float32)
    reference[6:10, 18:28, 20:30] = 8.0
    reference[4:12, 10:14, 8:40] = 5.0
    true_object = np.array([2.0, 3.0, -4.0])
    moving = ndimage_shift(reference, true_object, order=1, mode="constant", cval=0.0)
    registration = highpass_registration_zyx(reference, moving)
    assert np.max(np.abs(registration + true_object)) < 0.75
    restored = apply_registration_zyx(moving, registration)
    assert abs(float(np.argmax(restored.max(axis=(1, 2)))) - 7) <= 1


def main() -> None:
    test_highpass_recovers_zyx_shift()
    print("PASS test_highpass_xyz")


if __name__ == "__main__":
    main()
