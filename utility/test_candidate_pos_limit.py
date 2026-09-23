"""Tests for capping candidate positions without padding."""

from __future__ import annotations

import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_CONTROLFLIMAGE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _CONTROLFLIMAGE not in sys.path:
    sys.path.insert(0, _CONTROLFLIMAGE)

from utility.dendritic_shaft_detection import (
    limit_candidate_points,
    plot_skeleton_with_points,
)


class CandidateLimitTest(unittest.TestCase):
    def test_fewer_than_the_cap_is_kept_as_is(self) -> None:
        points = np.array([[1, 0, 0], [5, 2, 3]])
        kept = limit_candidate_points(points, 8)
        self.assertEqual(len(kept), 2)
        self.assertEqual(int(kept[0, 0]), 5)
        self.assertEqual(int(kept[1, 0]), 1)

    def test_cap_drops_the_lower_z_points(self) -> None:
        points = np.array([[i, 0, 0] for i in range(10)])
        kept = limit_candidate_points(points, 3)
        self.assertEqual([int(z) for z in kept[:, 0]], [9, 8, 7])

    def test_empty_skeleton_stays_empty(self) -> None:
        kept = limit_candidate_points(np.array([]), 8)
        self.assertEqual(len(kept), 0)
        self.assertEqual(kept.shape, (0, 3))

    def test_saved_figure_does_not_open_a_window(self) -> None:
        shown = {"n": 0}
        original_show = plt.show
        plt.show = lambda *args, **kwargs: shown.__setitem__("n", shown["n"] + 1)
        try:
            skeleton = np.zeros((4, 8, 8), dtype=bool)
            points = np.array([[1, 2, 3]])
            plot_skeleton_with_points(skeleton, points, showplot=False, saveplot=False)
        finally:
            plt.show = original_show
        self.assertEqual(shown["n"], 0)


if __name__ == "__main__":
    unittest.main()
