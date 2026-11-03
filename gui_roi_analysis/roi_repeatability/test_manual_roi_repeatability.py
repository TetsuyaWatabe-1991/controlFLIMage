# -*- coding: utf-8 -*-
"""Headless tests for manual_roi_repeatability.py (synthetic data, no FLIM / GUI).

Usage:
    python test_manual_roi_repeatability.py      # no pytest needed
    python -m pytest test_manual_roi_repeatability.py -q
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import manual_roi_repeatability as m  # noqa: E402

GROUPS = ["P_1_pos1__highmag_1_", "P_1_pos1__highmag_2_"]
BASES = [f"{g}_{s:.1f}_after_align_full" for g in GROUPS for s in (0.0, 1.0)]


def _fake_csv(scale: dict[str, float]) -> pd.DataFrame:
    """33 uncaging frames; post frames 30 min after the 2nd uncaging frame."""
    rows = []
    for g in GROUPS:
        for s in (0.0, 1.0):
            f = scale.get(f"{g}_{s:.1f}", 1.0)
            for i in range(3):
                rows.append(dict(group=g, set_label=s, phase="pre", elapsed_time_sec=i * 60.0, nAveFrame=8,
                                 Spine_Ch2_intensity=10.0, DendriticShaft_Ch2_intensity=50.0,
                                 Spine_Ch2_lifetime=np.nan))
            for i in range(33):
                rows.append(dict(group=g, set_label=s, phase="unc", elapsed_time_sec=600.0 + i, nAveFrame=8,
                                 Spine_Ch2_intensity=15.0, DendriticShaft_Ch2_intensity=50.0,
                                 Spine_Ch2_lifetime=2.0 if i % 8 == 0 else np.nan))
            for t in (10, 30, 32):
                rows.append(dict(group=g, set_label=s, phase="post", elapsed_time_sec=601.0 + t * 60, nAveFrame=8,
                                 Spine_Ch2_intensity=10.0 * f if t >= 25 else 99.0,
                                 DendriticShaft_Ch2_intensity=50.0, Spine_Ch2_lifetime=np.nan))
    return pd.DataFrame(rows)


def _mask(shift: int = 0, n: int = 4) -> np.ndarray:
    a = np.zeros((n, 16, 16), np.uint8)
    a[:, 5 + shift:9 + shift, 5:9] = 1
    return a


def test_set_summary_matches_ltp_definition():
    s = m.set_summary(_fake_csv({f"{GROUPS[0]}_0.0": 1.5}))
    r = s[(s.group == GROUPS[0]) & (s.set_label == 0.0)].iloc[0]
    assert math.isclose(r.spine_dFF0_post, 0.5, abs_tol=1e-9)  # 10-min post frame (99) is outside 25-35 min
    assert math.isclose(r.spine_dFF0_unc, 0.5, abs_tol=1e-9)
    assert math.isclose(r.shaft_dFF0_post, 0.0, abs_tol=1e-9)
    assert "spine_unc_lifetime" not in s.columns  # intensity only
    assert not r.pre_unstable
    assert len(s) == 4


def test_icc_perfect_and_noisy():
    x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], float)
    assert math.isclose(m.icc_2_1(x), 1.0, abs_tol=1e-9)
    rng = np.random.default_rng(0)
    noisy = x[:, :1] + rng.normal(0, 5, (4, 3))
    assert m.icc_2_1(noisy) < 0.9
    assert np.isnan(m.icc_2_1(x[:1]))


def test_mask_pair_stats():
    iou, d = m.mask_pair_stats(_mask() > 0, _mask() > 0)
    assert math.isclose(iou, 1.0, abs_tol=1e-9) and math.isclose(d, 0.0, abs_tol=1e-9)
    iou, d = m.mask_pair_stats(_mask() > 0, _mask(2) > 0)
    assert math.isclose(iou, 8 / 24, abs_tol=1e-9) and math.isclose(d, 2.0, abs_tol=1e-9)


def test_set_base():
    assert m.set_base(BASES, GROUPS[1], 1.0) == BASES[3]
    assert m.set_base(BASES, "other_", 0.0) == ""


def _make_experiment(tmp_path: Path) -> m.Experiment:
    exp = m.Experiment(tmp_path, "P")
    exp.work_tif.mkdir(parents=True)
    for b in BASES:
        tifffile.imwrite(exp.work_tif / f"{b}_Spine_roi_mask.tif", _mask())
        tifffile.imwrite(exp.work_tif / f"{b}_DendriticShaft_roi_mask.tif", _mask(5))
    _fake_csv({}).to_csv(exp.work_csv, index=False)
    pd.DataFrame({"group": GROUPS, "nth_set_label": [0.0, 0.0], "reject": [0, 0]}).to_pickle(exp.work_pkl)
    m.save_state(exp, {"rounds": {}, "active_round": None, "bases": BASES})
    m.snapshot_round(exp, 0, BASES)
    return exp


def _edit_round(exp: m.Experiment, n: int, shift: int, scale: float, reject: str | None) -> None:
    m.start_round(exp, n, open_viewer=False)
    # reset must have removed the previous round's edits
    assert np.array_equal(tifffile.imread(exp.work_tif / f"{BASES[0]}_Spine_roi_mask.tif"), _mask())
    assert not list(exp.work_tif.glob("*_rejected.flag"))
    tifffile.imwrite(exp.work_tif / f"{BASES[0]}_Spine_roi_mask.tif", _mask(shift))
    _fake_csv({f"{GROUPS[0]}_0.0": scale}).to_csv(exp.work_csv, index=False)
    if reject:
        (exp.work_tif / f"{reject}_rejected.flag").write_text("x")
    m.collect_round(exp, n)


def test_rounds_reset_collect_and_compare(tmp_path):
    exp = _make_experiment(tmp_path)
    _edit_round(exp, 1, 1, 1.30, None)
    _edit_round(exp, 2, 2, 1.40, BASES[3])
    _edit_round(exp, 3, 1, 1.30, None)
    try:
        m.start_round(exp, 1, open_viewer=False)  # already collected
        raise AssertionError("re-starting a collected round must fail")
    except RuntimeError:
        pass

    assert exp.collected_rounds() == [0, 1, 2, 3]
    st = m.load_state(exp)
    assert st["rounds"]["2"]["n_rejected"] == 1 and st["rounds"]["2"]["n_spine_edited"] == 1
    # round0 itself is never modified
    assert np.array_equal(tifffile.imread(exp.round_dir(0) / "tif" / f"{BASES[0]}_Spine_roi_mask.tif"), _mask())

    m.compare(exp)
    out = exp.compare_dir
    for f in ["per_set_metrics_long.csv", "metric_agreement_summary.csv", "spine_mask_agreement.csv",
              "reject_table.csv", "report.txt", "spine_dFF0_post_per_set.png", "spine_dFF0_post_pairwise.png",
              "spine_roi_overlap.png", "spine_dFF0_post_beeswarm.png", "spine_dFF0_post_beeswarm.pdf"]:
        assert (out / f).exists(), f
    long = pd.read_csv(out / "per_set_metrics_long.csv")
    assert len(long) == 4 * 4 and long.rejected.sum() == 1
    summ = pd.read_csv(out / "metric_agreement_summary.csv").set_index("metric")
    r = summ.loc["spine_dFF0_post"]
    assert r.n_sets_common_accepted == 3  # BASES[3] rejected in R2
    # only set 0: R1/R3 0.30, R2 0.40 -> pairwise |diff| 0.1, 0, 0.1 on 1 of 3 sets
    assert math.isclose(r.mean_abs_diff_manual_vs_manual, (0.1 + 0.0 + 0.1) / 3 / 3, abs_tol=1e-9)
    assert math.isclose(r.max_abs_diff_manual_vs_manual, 0.1, abs_tol=1e-9)
    masks = pd.read_csv(out / "spine_mask_agreement.csv")
    assert set(masks.kind) == {"auto_vs_manual", "manual_vs_manual"}
    assert masks[(masks.set == BASES[0]) & (masks.pair == "R1-R3")].identical.iloc[0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        if "tmp_path" in t.__code__.co_varnames[: t.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as td:
                t(Path(td))
        else:
            t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
