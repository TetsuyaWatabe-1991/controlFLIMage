"""Reject-rule sensitivity counts caught rejects. Specificity counts kept keeps."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from score_qc_criteria import (  # noqa: E402
    apply_threshold,
    best_threshold,
    confusion,
    distance_to_mask,
    drift_span_yx,
    roi_contrast,
)


class ScoreQcCriteriaTest(unittest.TestCase):
    def test_sensitivity_is_caught_rejects_and_specificity_is_kept_keeps(self) -> None:
        stats = confusion(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]))
        self.assertEqual(stats["sensitivity"], 0.5)
        self.assertEqual(stats["specificity"], 0.5)
        self.assertEqual((stats["tp"], stats["fn"], stats["fp"], stats["tn"]), (1, 1, 1, 1))

    def test_missing_values_do_not_reject(self) -> None:
        pred = apply_threshold(np.array([np.nan, 5.0, 1.0]), 4.0, higher_rejects=True)
        self.assertEqual(pred.tolist(), [False, True, False])

    def test_best_threshold_separates_a_shift(self) -> None:
        values = np.array([1.0, 1.0, 8.0, 9.0])
        truth = np.array([False, False, True, True])
        threshold, stats = best_threshold(values, truth, higher_rejects=True)
        self.assertGreaterEqual(threshold, 1.0)
        self.assertEqual(stats["sensitivity"], 1.0)
        self.assertEqual(stats["specificity"], 1.0)

    def test_drift_is_relative_to_the_first_frame(self) -> None:
        self.assertAlmostEqual(drift_span_yx(np.array([2.0, 2.0, 5.0]), np.array([-1.0, 0.0, -1.0])), 3.0)

    def test_inside_the_roi_is_zero_and_contrast_uses_the_ring(self) -> None:
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[6:10, 6:10] = 1
        self.assertEqual(distance_to_mask(7, 7, mask), 0.0)
        self.assertGreater(distance_to_mask(0, 0, mask), 5.0)
        image = np.ones((16, 16), dtype=np.float32)
        image[6:10, 6:10] = 4
        self.assertAlmostEqual(roi_contrast(image, mask), 4.0)


if __name__ == "__main__":
    unittest.main()
