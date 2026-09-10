# -*- coding: utf-8 -*-
"""Headless checks: majority-shape skip, OOB index skip, error-log write."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTROL = os.path.normpath(os.path.join(_GUI_DIR, ".."))
for _p in (_GUI_DIR, _CONTROL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SHAPE_15 = (15, 1, 2, 128, 128, 64)
SHAPE_45 = (45, 1, 2, 128, 128, 64)


def test_majority_shape_keeps_15_skips_45() -> None:
    from gui_roi_fast_simple import select_majority_shape_files

    majority_files = [
        rf"C:\data\1_pos1__highmag_5_{i:03d}.flim" for i in (1, 2, 3, 4, 6, 7)
    ]
    outlier = r"C:\data\1_pos1__highmag_5_005.flim"
    filelist = majority_files[:4] + [outlier] + majority_files[4:]
    shape_by_path = {path: SHAPE_15 for path in majority_files}
    shape_by_path[outlier] = SHAPE_45

    kept, skipped = select_majority_shape_files(
        filelist, shape_by_path=shape_by_path
    )
    assert kept == majority_files, kept
    assert len(skipped) == 1, skipped
    skipped_path, reason = skipped[0]
    assert skipped_path == outlier
    assert "45" in reason and "15" in reason, reason


def test_unreadable_file_is_skipped() -> None:
    from gui_roi_fast_simple import select_majority_shape_files

    ok_a = r"C:\data\a.flim"
    ok_b = r"C:\data\b.flim"
    bad = r"C:\data\missing.flim"
    kept, skipped = select_majority_shape_files(
        [ok_a, bad, ok_b],
        shape_by_path={ok_a: SHAPE_15, ok_b: SHAPE_15},
    )
    assert kept == [ok_a, ok_b]
    assert skipped[0][0] == bad
    assert "could not read" in skipped[0][1]


def test_oob_index_is_skipped_not_raised() -> None:
    from gui_roi_fast_simple import aligned_array_index_or_none, safe_aligned_zproj

    aligned = np.zeros((1, 4, 8, 8), dtype=np.float32)
    aligned[0, :, 2, 2] = 1.0
    path_ok = r"C:\data\keep.flim"
    path_oob = r"C:\data\oob.flim"
    path_missing = r"C:\data\missing.flim"
    mapping = {path_ok: 0, path_oob: 1}

    assert aligned_array_index_or_none(path_ok, mapping, aligned.shape[0]) == 0
    assert aligned_array_index_or_none(path_oob, mapping, aligned.shape[0]) is None
    assert aligned_array_index_or_none(path_missing, mapping, aligned.shape[0]) is None

    zproj = safe_aligned_zproj(aligned, 0, 0, 4)
    assert zproj is not None
    assert zproj.shape == (8, 8)
    assert safe_aligned_zproj(aligned, 1, 0, 4) is None
    assert safe_aligned_zproj(aligned, None, 0, 4) is None


def test_error_log_empty_not_written() -> None:
    from gui_roi_fast_simple import save_roi_analysis_error_log

    with tempfile.TemporaryDirectory() as tmp:
        path = save_roi_analysis_error_log(tmp, [])
        assert path is None
        assert os.listdir(tmp) == []


def test_error_log_written_with_errors_n() -> None:
    from gui_roi_fast_simple import (
        format_roi_analysis_errors_block,
        record_roi_error,
        save_roi_analysis_error_log,
    )

    error_log: list[str] = []
    record_roi_error(
        error_log,
        group="1_pos1__highmag_5",
        set_label="set1",
        file="1_pos1__highmag_5_005.flim",
        reason="raw shape (45,) != majority (15,)",
    )
    block = format_roi_analysis_errors_block(error_log)
    assert block.startswith("ERRORS (1)\n")
    assert "1_pos1__highmag_5_005.flim" in block

    with tempfile.TemporaryDirectory() as tmp:
        path = save_roi_analysis_error_log(tmp, error_log, timestamp="20260817_120000")
        assert path is not None
        assert os.path.basename(path) == "roi_analysis_errors_20260817_120000.txt"
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        assert text == block


def main() -> int:
    tests = [
        test_majority_shape_keeps_15_skips_45,
        test_unreadable_file_is_skipped,
        test_oob_index_is_skipped_not_raised,
        test_error_log_empty_not_written,
        test_error_log_written_with_errors_n,
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
