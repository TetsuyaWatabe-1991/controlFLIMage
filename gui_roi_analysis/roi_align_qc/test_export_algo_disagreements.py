"""Changed calls and human-label errors land in separate folders."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from export_algo_disagreements import (  # noqa: E402
    algorithm_rejects,
    annotate_sheet,
    disagreement_folders,
    rule_lines,
)


class ExportAlgoDisagreementsTest(unittest.TestCase):
    def test_a_large_uncaging_distance_rejects(self) -> None:
        self.assertFalse(algorithm_rejects(1.0, 0.8, 0.0))
        self.assertTrue(algorithm_rejects(1.0, 0.8, 10.0))
        fired = [flag for flag, _text in rule_lines(1.0, 0.2, 0.0)]
        self.assertEqual(fired, [False, True, False])

    def test_folders_split_prior_flips_from_human_errors(self) -> None:
        self.assertEqual(
            disagreement_folders("reject", 1, algo_reject=False),
            ["prior_reject_now_include", "false_negative"],
        )
        self.assertEqual(
            disagreement_folders("keep", 9, algo_reject=True),
            ["prior_include_now_reject", "false_positive"],
        )
        self.assertEqual(disagreement_folders("keep", 2, algo_reject=True), ["prior_include_now_reject"])
        self.assertEqual(disagreement_folders("reject", 9, algo_reject=False), ["prior_reject_now_include"])

    def test_banner_sits_above_the_sheet(self) -> None:
        sheet = np.zeros((20, 80, 3), dtype=np.uint8)
        painted = annotate_sheet(sheet, "FALSE NEGATIVE", "human 1", rule_lines(1.0, 0.9, 0.0))
        self.assertGreater(painted.shape[0], sheet.shape[0])
        self.assertEqual(painted.shape[1], 80)
        self.assertTrue(np.any(painted[:20] != 0))


if __name__ == "__main__":
    unittest.main()
