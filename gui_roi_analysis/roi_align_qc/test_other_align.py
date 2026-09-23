"""Non-correlation aligners should recover a rigid dendrite shift."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from other_align import OTHER_ALIGNERS  # noqa: E402
from tracking import translate_image  # noqa: E402


def _dendrite(size: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=np.float32)
    image[12:84, 40:48] = 1.0
    image[44:52, 16:80] = 0.85
    image[20:32, 48:64] = 0.6
    image += 0.15 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    return image


def test_other_aligners_recover_rigid_shift() -> None:
    base = _dendrite()
    moved = translate_image(base, 4.0, -3.0)
    stack = np.stack([base, moved])
    for name, aligner in OTHER_ALIGNERS.items():
        shift = aligner(stack)
        assert abs(shift[1, 0] - 4.0) < 1.0, (name, shift[1])
        assert abs(shift[1, 1] - (-3.0)) < 1.0, (name, shift[1])


def main() -> None:
    test_other_aligners_recover_rigid_shift()
    print("PASS test_other_aligners_recover_rigid_shift")


if __name__ == "__main__":
    main()
