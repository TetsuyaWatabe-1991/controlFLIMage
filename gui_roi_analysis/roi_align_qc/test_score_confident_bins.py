"""Confident reject spends a keep budget. Confident include keeps rejects out."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from score_confident_bins import (  # noqa: E402
    absolute_reject_cutoffs,
    apply_reject_cutoffs,
    assign_bins,
    bin_counts,
    fit_include_limits,
    fit_reject_limits,
    local_corr,
)


class ConfidentBinsTest(unittest.TestCase):
    def test_local_window_correlation_is_one_for_the_same_image(self) -> None:
        image = np.arange(64, dtype=float).reshape(8, 8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        self.assertAlmostEqual(local_corr(image, image, mask, pad=1), 1.0)

    def test_reject_budget_leaves_specificity_at_least_98_percent(self) -> None:
        rows = []
        for index in range(100):
            rows.append({"category": 9, "assign_shift_px": float(index), "unc_dist_px": 0.0})
        for index in range(20):
            rows.append({"category": 1, "assign_shift_px": 200.0 + index, "unc_dist_px": 0.0})
        frame = pd.DataFrame(rows)
        for name in (
            "drift_info_px",
            "low_assign_corr",
            "low_contrast",
            "low_spine_drop",
            "low_z_match",
            "low_z_neighbor",
            "bead_fraction",
            "bright_cv",
        ):
            frame[name] = 0.0
        frame["low_assign_corr"] = 0.0
        limits = fit_reject_limits(frame, max_keeps=2)
        bins = assign_bins(frame, limits, fit_include_limits(frame, max_rejects=0))
        counts = bin_counts(frame, bins)
        self.assertLessEqual(counts["reject_keeps"], 2)
        self.assertGreaterEqual(counts["reject_specificity"], 0.98)
        self.assertGreater(counts["reject_rejects"], 0)

    def test_a_cutoff_from_one_day_can_flag_a_later_keep(self) -> None:
        train = pd.DataFrame({"category": [9, 9, 1], "assign_shift_px": [1.0, 2.0, 9.0]})
        test = pd.DataFrame({"category": [9], "assign_shift_px": [5.0]})
        for name in (
            "unc_dist_px",
            "drift_info_px",
            "low_assign_corr",
            "low_contrast",
            "low_spine_drop",
            "low_z_match",
            "low_z_neighbor",
            "bead_fraction",
            "bright_cv",
        ):
            train[name] = 0.0
            test[name] = 0.0
        cutoffs = absolute_reject_cutoffs(train, {name: 0 for name in train.columns if name != "category"})
        self.assertTrue(bool(apply_reject_cutoffs(test, cutoffs)[0]))


if __name__ == "__main__":
    unittest.main()
