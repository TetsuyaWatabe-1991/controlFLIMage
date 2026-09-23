"""The unhealthy export is one square max projection."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from export_unhealthy_pre1 import SIDE, render_pre_max, stretch_u8  # noqa: E402


class UnhealthyPreExportTest(unittest.TestCase):
    def test_stretch_uses_the_bright_pixels(self) -> None:
        image = np.zeros((4, 4), dtype=float)
        image[0, 0] = 10
        stretched = stretch_u8(image)
        self.assertEqual(stretched.dtype, np.uint8)
        self.assertEqual(int(stretched.max()), 255)

    def test_render_is_the_pre_square_with_the_outline(self) -> None:
        image = np.ones((8, 8), dtype=float)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        painted = render_pre_max(image, mask)
        self.assertEqual(painted.shape, (SIDE, SIDE, 3))
        self.assertTrue(np.any(painted[:, :, 0] != painted[:, :, 1]))


if __name__ == "__main__":
    unittest.main()
