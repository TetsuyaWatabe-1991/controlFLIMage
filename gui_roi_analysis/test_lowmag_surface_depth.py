"""Unit tests for uncaging depth below the fitted surface."""

import os
import sys
import unittest

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from lowmag_img_save_with_GCdata_tiff_roi_surface_depth import (
    depth_um_below_surface,
    surface_slice_at_display_um,
)


class SurfaceDepthTests(unittest.TestCase):
    def test_depth_rounds_to_integer_um(self) -> None:
        # Surface at slice 10 of a stack whose bottom is 1000 um, step 2 um -> 1020 um.
        # Uncaging at slice 4 of a stack whose bottom is 1008 um, step 1 um -> 1012 um.
        # Depth below surface is 8 um.
        depth = depth_um_below_surface(
            surface_slice=10,
            lowmag_motor_z_um=1000,
            lowmag_slice_step_um=2,
            uncaging_slice=4,
            highmag_motor_z_um=1008,
            highmag_slice_step_um=1,
        )
        self.assertEqual(depth, 8)

    def test_fractional_micrometer_rounds_to_nearest_integer(self) -> None:
        depth = depth_um_below_surface(
            surface_slice=0.4,
            lowmag_motor_z_um=0,
            lowmag_slice_step_um=2,
            uncaging_slice=0,
            highmag_motor_z_um=0,
            highmag_slice_step_um=1,
        )
        self.assertEqual(depth, 1)

    def test_point_above_surface_is_negative(self) -> None:
        depth = depth_um_below_surface(
            surface_slice=0,
            lowmag_motor_z_um=1000,
            lowmag_slice_step_um=1,
            uncaging_slice=5,
            highmag_motor_z_um=1000,
            highmag_slice_step_um=1,
        )
        self.assertEqual(depth, -5)

    def test_display_um_maps_to_pixel_and_rejects_outside(self) -> None:
        surface = np.arange(4, dtype=float).reshape(2, 2)
        self.assertEqual(surface_slice_at_display_um(surface, 1.0, 1.0, 4.0, 4.0), 0.0)
        self.assertTrue(np.isnan(surface_slice_at_display_um(surface, -1.0, 1.0, 4.0, 4.0)))


if __name__ == "__main__":
    unittest.main()
