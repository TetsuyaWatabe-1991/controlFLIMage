# -*- coding: utf-8 -*-
"""Headless checks for LTP post-frame time-window selection."""

from __future__ import annotations

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from flim_summarize_func import format_respan_path_assignments, select_ltp_post_frames  # noqa: E402


def test_includes_point_just_outside_35_min() -> None:
    # 2101 s = 35.02 min was excluded by exclusive < 35*60, then fallback
    # averaged 80 min frames into delta_FF0 (~1.34 instead of ~0.35).
    post = pd.DataFrame(
        {
            "aligned_time_sec": [80.6, 1358.5, 2101.0, 4759.4, 5075.2],
            "y": [0.7, 0.2, 0.35, 2.18, 1.51],
        }
    )
    selected, mode = select_ltp_post_frames(post, window_min=(25, 35), pad_sec=60)
    assert mode == "window"
    assert list(selected["aligned_time_sec"]) == [2101.0]


def test_includes_point_just_before_25_min() -> None:
    post = pd.DataFrame({"aligned_time_sec": [1480.2, 2215.7, 4876.1]})
    selected, mode = select_ltp_post_frames(post, window_min=(25, 35), pad_sec=60)
    assert mode == "window"
    assert list(selected["aligned_time_sec"]) == [1480.2]


def test_nearest_if_nothing_near_window() -> None:
    post = pd.DataFrame({"aligned_time_sec": [80.0, 400.0, 5000.0]})
    selected, mode = select_ltp_post_frames(post, window_min=(25, 35), pad_sec=60)
    assert mode == "nearest"
    assert list(selected["aligned_time_sec"]) == [400.0]


def test_empty() -> None:
    selected, mode = select_ltp_post_frames(pd.DataFrame({"aligned_time_sec": []}))
    assert mode == "none"
    assert len(selected) == 0


def test_format_respan_path_assignments_is_copy_paste_python() -> None:
    text = format_respan_path_assignments(
        r"G:/ImagingData/Tetsuya/20260506/RabCalf_tdTGC6s/auto1\combined_df_respan.pkl",
        r"G:/ImagingData/Tetsuya/20260506/RabCalf_tdTGC6s/auto1\combined_df_respan_intensity_lifetime_all_frames.csv",
    )
    expected = (
        'df_save_path_1 = r"G:/ImagingData/Tetsuya/20260506/RabCalf_tdTGC6s/auto1\\combined_df_respan.pkl"\n'
        'out_csv_path = r"G:/ImagingData/Tetsuya/20260506/RabCalf_tdTGC6s/auto1\\combined_df_respan_intensity_lifetime_all_frames.csv"'
    )
    assert text == expected
    namespace: dict[str, str] = {}
    exec(text, namespace)
    assert namespace["df_save_path_1"].endswith("combined_df_respan.pkl")
    assert namespace["out_csv_path"].endswith("combined_df_respan_intensity_lifetime_all_frames.csv")


def main() -> int:
    tests = [
        test_includes_point_just_outside_35_min,
        test_includes_point_just_before_25_min,
        test_nearest_if_nothing_near_window,
        test_empty,
        test_format_respan_path_assignments_is_copy_paste_python,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL: {fn.__name__}: {exc}")
    print(f"Done: {len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
