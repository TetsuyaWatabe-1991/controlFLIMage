"""Assignment panel is one max projection, lined up with the sheet's first pre."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from qc_assignment_source import (  # noqa: E402
    PRE_SIDE,
    SELECTED_BGR,
    SHEET_BANNER_H,
    SHEET_CAPTION_H,
    match_uncaging_record,
    render_assignment_panel,
)
from qc_label_gui import matched_pre_view  # noqa: E402


class AssignmentSourceTest(unittest.TestCase):
    def test_record_just_before_the_uncaging_file_is_chosen(self) -> None:
        records = [
            SimpleNamespace(flim_path=r"G:\data\highmag_1_005.flim", spine_stem="id020"),
            SimpleNamespace(flim_path=r"G:\data\highmag_1_014.flim", spine_stem="id012"),
        ]
        chosen = match_uncaging_record(r"G:\data\highmag_1_015.flim", records)
        self.assertEqual(chosen.spine_stem, "id012")

    def test_panel_is_one_square_the_same_size_as_the_first_pre(self) -> None:
        volume = np.zeros((3, 12, 12), dtype=np.float32)
        volume[1, 2:8, 2:8] = 10
        mask = np.zeros((12, 12), dtype=bool)
        mask[3:7, 3:7] = True
        image = render_assignment_panel(volume, mask, [], head_z=1, spine_name="id012")
        self.assertEqual(image.shape, (SHEET_CAPTION_H + PRE_SIDE, PRE_SIDE, 3))
        self.assertTrue(np.any(np.all(image == np.array(SELECTED_BGR), axis=2)))

    def test_assignment_square_lines_up_with_the_pre_square(self) -> None:
        scale, origin_x, origin_y = matched_pre_view(0.5, 10.0, 400.0, PRE_SIDE)
        self.assertEqual(scale, 0.5)
        self.assertEqual(origin_x, 400.0 - PRE_SIDE * 0.5)
        self.assertEqual(origin_y, 10.0 + SHEET_BANNER_H * 0.5)


if __name__ == "__main__":
    unittest.main()
