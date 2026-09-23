"""A lone bleb is dead. A long shaft, with or without a small bead, is not."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from obvious_dead import (  # noqa: E402
    BLEB_SIGMA_UM,
    SHAFT_MAX_UM,
    TUNED_XY_UM,
    dead_scores,
    obviously_dead,
    xy_um_from_state,
)


def _disk(radius: int) -> np.ndarray:
    image = np.zeros((96, 96), dtype=np.float32)
    yy, xx = np.ogrid[-48:48, -48:48]
    image[(yy ** 2 + xx ** 2) <= radius ** 2] = 1
    return image


class ObviousDeadTest(unittest.TestCase):
    def test_a_large_disk_is_dead_and_a_shaft_is_not(self) -> None:
        bleb = _disk(8)
        shaft = np.zeros((96, 128), dtype=np.float32)
        shaft[46:49, :] = 1
        bleb_peak, bleb_shaft = dead_scores(bleb, TUNED_XY_UM)
        shaft_peak, shaft_len = dead_scores(shaft, TUNED_XY_UM)
        self.assertGreater(bleb_peak, shaft_peak)
        self.assertLess(bleb_shaft, shaft_len)
        self.assertTrue(obviously_dead(bleb, TUNED_XY_UM))
        self.assertFalse(obviously_dead(shaft, TUNED_XY_UM))

    def test_a_bead_on_a_long_shaft_is_not_dead(self) -> None:
        image = np.zeros((96, 128), dtype=np.float32)
        image[46:49, :] = 1
        yy, xx = np.ogrid[-48:48, -64:64]
        image[(yy ** 2 + xx ** 2) <= 3 ** 2] = 1
        self.assertFalse(obviously_dead(image, TUNED_XY_UM))

    def test_cutoffs_are_micrometers_at_the_tuned_pixel_size(self) -> None:
        state = {
            "State.Acq.FOV_default": [273, 271],
            "State.Acq.zoom": 14,
            "State.Acq.pixelsPerLine": 128,
            "State.Acq.linesPerFrame": 128,
        }
        self.assertAlmostEqual(xy_um_from_state(state), TUNED_XY_UM)
        self.assertAlmostEqual(SHAFT_MAX_UM, 4.25)
        self.assertAlmostEqual(SHAFT_MAX_UM / TUNED_XY_UM, 28.0)
        self.assertAlmostEqual(BLEB_SIGMA_UM[0] / TUNED_XY_UM, 3.0)
        self.assertAlmostEqual(BLEB_SIGMA_UM[-1] / TUNED_XY_UM, 11.0)
        # A coarser zoom uses fewer pixels for the same physical shaft.
        zoom_15 = {
            "State.Acq.FOV_default": [273, 271],
            "State.Acq.zoom": 15,
            "State.Acq.pixelsPerLine": 128,
            "State.Acq.linesPerFrame": 128,
        }
        self.assertAlmostEqual(SHAFT_MAX_UM / xy_um_from_state(zoom_15), 30.0)

    def test_the_micrometer_shaft_cutoff_is_what_the_call_uses(self) -> None:
        shaft = np.zeros((96, 128), dtype=np.float32)
        shaft[46:49, :] = 1
        _peak, shaft_um = dead_scores(shaft, TUNED_XY_UM)
        self.assertGreater(shaft_um, SHAFT_MAX_UM)
        self.assertFalse(obviously_dead(shaft, TUNED_XY_UM, bleb_min=0.0, shaft_max_um=shaft_um - 1e-3))
        self.assertTrue(obviously_dead(shaft, TUNED_XY_UM, bleb_min=0.0, shaft_max_um=shaft_um))
        self.assertFalse(obviously_dead(_disk(8), TUNED_XY_UM, bleb_min=10.0))


if __name__ == "__main__":
    unittest.main()
