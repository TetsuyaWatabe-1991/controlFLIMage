"""Pairing rules for reduced-resolution lowmag and for_align grabs."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from gui_roi_analysis.roi_align_qc.export_lowmag_for_align_xyz import (
    list_pairs,
    to_common_shape,
)


class PairListingTest(unittest.TestCase):
    def test_pairs_reduced_names_to_the_live_reference(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for name in (
                "cnt_2_pos1_001.flim",
                "cnt_2_pos1_002.flim",
                "cnt_2_pos1__highmag_3_002.flim",
                "cnt_2_pos1__highmag_3_014.flim",
                "for_align_cnt_2_pos1__highmag_3_003.flim",
                "for_align_cnt_2_pos1__highmag_3_014.flim",
                "AP5_11_pos1_001.flim",
            ):
                open(os.path.join(folder, name), "wb").close()
            pairs = {(item.kind, item.reference_name, item.moving_name) for item in list_pairs(folder)}
        self.assertEqual(
            pairs,
            {
                ("lowmag", "cnt_2_pos1_001.flim", "cnt_2_pos1_002.flim"),
                (
                    "for_align",
                    "cnt_2_pos1__highmag_3_002.flim",
                    "for_align_cnt_2_pos1__highmag_3_003.flim",
                ),
                (
                    "for_align",
                    "cnt_2_pos1__highmag_3_002.flim",
                    "for_align_cnt_2_pos1__highmag_3_014.flim",
                ),
            },
        )

    def test_common_shape_uses_the_smaller_grid(self) -> None:
        reference = np.zeros((15, 128, 128), dtype=np.float32)
        moving = np.ones((15, 64, 64), dtype=np.float32)
        left, right = to_common_shape(reference, moving)
        self.assertEqual(left.shape, (15, 64, 64))
        self.assertEqual(right.shape, (15, 64, 64))


if __name__ == "__main__":
    unittest.main()
