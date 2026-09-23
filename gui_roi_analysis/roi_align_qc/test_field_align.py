"""Field aligners should follow a rigid move of a dendrite-like image."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from field_align import FIELD_ALIGNERS  # noqa: E402
from tracking import translate_image  # noqa: E402


def _dendrite(size: int = 96) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float32)
    # A shaft and a branch that do not span the whole frame, so both
    # directions of a rigid shift are visible in more than one tile.
    image[12:84, 40:48] = 1.0
    image[44:52, 16:80] = 0.85
    image[20:32, 48:64] = 0.6
    return image


def test_field_aligners_recover_rigid_shift() -> None:
    base = _dendrite()
    moved = translate_image(base, 4.0, -3.0)
    stack = np.stack([base, moved])
    for name, aligner in FIELD_ALIGNERS.items():
        shift = aligner(stack)
        assert abs(shift[1, 0] - 4.0) < 0.75, (name, shift[1])
        assert abs(shift[1, 1] - (-3.0)) < 0.75, (name, shift[1])


def main() -> None:
    test_field_aligners_recover_rigid_shift()
    print("PASS test_field_aligners_recover_rigid_shift")


if __name__ == "__main__":
    main()
