# -*- coding: utf-8 -*-
"""Headless checks: uncaging_frame_num includes 80/144 for RESPAN ROI detection."""

from __future__ import annotations

import inspect
import os
import sys

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTROL = os.path.normpath(os.path.join(_GUI_DIR, ".."))
_ANALYSIS = os.path.join(_CONTROL, "AnalysisForFLIMage")
for _p in (_GUI_DIR, _CONTROL, _ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REQUIRED = (33, 36, 55, 80, 144)


def _default_uncaging_frame_num(func) -> list:
    param = inspect.signature(func).parameters["uncaging_frame_num"]
    default = param.default
    assert default is not inspect.Parameter.empty, f"{func.__name__} missing default"
    return list(default)


def test_get_annotation_default() -> None:
    from get_annotation_unc_multiple import get_uncaging_pos_multiple

    nums = _default_uncaging_frame_num(get_uncaging_pos_multiple)
    for n in REQUIRED:
        assert n in nums, f"get_uncaging_pos_multiple default missing {n}: {nums}"


def test_gui_integration_default() -> None:
    from gui_integration import first_processing_for_flim_files

    nums = _default_uncaging_frame_num(first_processing_for_flim_files)
    for n in REQUIRED:
        assert n in nums, f"first_processing_for_flim_files default missing {n}: {nums}"


def test_respan_runner_none_branch() -> None:
    """Mirror the None-default assignment inside run_tiff_uncaging_roi_respan."""
    uncaging_frame_num = None
    if uncaging_frame_num is None:
        uncaging_frame_num = [33, 34, 35, 36, 55, 80, 144]
    for n in REQUIRED:
        assert n in uncaging_frame_num


def test_classify_lengths_as_uncaging() -> None:
    uncaging_frame_num = [33, 34, 35, 36, 55, 80, 144]
    titration_frame_num = [32]
    first_n_images = 26  # typical highmag z-stack length

    for n_images in (33, 36, 55, 80, 144):
        assert n_images != first_n_images
        assert n_images not in titration_frame_num
        assert n_images in uncaging_frame_num


def test_valid_set_and_stack_path_helpers() -> None:
    import pandas as pd
    from gui_roi_respan_seg_masks import (
        _has_full_size_stack_paths,
        _has_valid_roi_sets,
    )

    assert _has_valid_roi_sets(pd.DataFrame({"nth_set_label": [-1, -1]})) is False
    assert _has_valid_roi_sets(pd.DataFrame({"nth_set_label": [-1, 0]})) is True
    assert _has_full_size_stack_paths(pd.DataFrame({"nth_set_label": [0]})) is False
    assert (
        _has_full_size_stack_paths(
            pd.DataFrame(
                {
                    "after_align_full_save_path": [r"C:\tmp\stack.tif"],
                    "n_pre_frames": [3],
                }
            )
        )
        is True
    )


def main() -> int:
    tests = [
        test_get_annotation_default,
        test_gui_integration_default,
        test_respan_runner_none_branch,
        test_classify_lengths_as_uncaging,
        test_valid_set_and_stack_path_helpers,
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
