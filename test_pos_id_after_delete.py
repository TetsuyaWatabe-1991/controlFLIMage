# -*- coding: utf-8 -*-
"""Headless checks for which position is shown after Delete in the lowmag clicker."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multidim_tiff_viewer import pos_id_after_delete  # noqa: E402


def test_delete_middle_stays_on_the_position_that_slides_into_the_slot() -> None:
    # Positions 1,2,3,4. Delete 2. Old 3 is the new 2.
    assert pos_id_after_delete(2, 3) == 2


def test_delete_last_returns_to_position_1() -> None:
    assert pos_id_after_delete(4, 3) == 1
    assert pos_id_after_delete(1, 0) == 0


def test_delete_first_lands_on_the_new_first() -> None:
    assert pos_id_after_delete(1, 3) == 1


if __name__ == "__main__":
    test_delete_middle_stays_on_the_position_that_slides_into_the_slot()
    test_delete_last_returns_to_position_1()
    test_delete_first_lands_on_the_new_first()
    print("PASS test_pos_id_after_delete")
