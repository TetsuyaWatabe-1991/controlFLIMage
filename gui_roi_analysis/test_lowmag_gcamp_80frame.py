# -*- coding: utf-8 -*-
"""Headless checks: 80-frame GCaMP windows and nAve-then-pre intensity norm."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
if _GUI_DIR not in sys.path:
    sys.path.insert(0, _GUI_DIR)


def test_80frame_uses_metadata_and_averages_windows() -> None:
    from lowmag_img_save_with_GCdata_tiff_roi import _extract_gcamp_pre_unc_sum

    image = np.zeros((80, 1, 1, 4, 4, 2), dtype=np.float32)
    image[:8] = 1.0
    image[8:10] = 3.0
    statedict = {
        "State.Uncaging.FramesBeforeUncage": 8,
        "State.Uncaging.Uncage_FrameInterval": 2,
    }
    gc_pre, gc_unc = _extract_gcamp_pre_unc_sum(image, statedict=statedict)
    assert gc_pre is not None and gc_unc is not None
    assert gc_pre.shape == (4, 4)
    # Sum over 2 lifetime bins, then divide by the window length.
    assert np.allclose(gc_pre, 2.0)
    assert np.allclose(gc_unc, 6.0)


def test_80frame_without_metadata_is_skipped() -> None:
    from lowmag_img_save_with_GCdata_tiff_roi import _extract_gcamp_pre_unc_sum

    image = np.zeros((80, 1, 1, 2, 2, 1), dtype=np.float32)
    gc_pre, gc_unc = _extract_gcamp_pre_unc_sum(image, statedict=None)
    assert gc_pre is None and gc_unc is None


def test_intensity_divides_by_nave_before_pre_baseline() -> None:
    from lowmag_img_save_with_GCdata_tiff_roi import _compute_norm_series_from_intensity_csv

    # Pre raw sum is 40 over 4 averages -> 10 per frame.
    # Uncaging raw sum is 5 over 1 average -> 5 per frame, so F/F0-1 = -0.5.
    # Skipping nAve would make uncaging 5/40 - 1 = -0.875.
    rows = []
    for i, phase, spine, nave in (
        (0, "pre", 40.0, 4),
        (1, "pre", 40.0, 4),
        (2, "unc", 5.0, 1),
    ):
        rows.append(
            {
                "group": "g",
                "set_label": 1.0,
                "phase": phase,
                "elapsed_time_sec": float(i),
                "nAveFrame": nave,
                "Spine_Ch2_intensity": spine,
                "Background_Ch2_intensity": 0.0,
            }
        )
    out = _compute_norm_series_from_intensity_csv(pd.DataFrame(rows), "g", 1.0, 2)
    assert out is not None
    pre = out.iloc[:2]["norm_intensity"].to_numpy()
    unc = float(out.iloc[2]["norm_intensity"])
    assert np.allclose(pre, 0.0)
    assert np.isclose(unc, -0.5)


def test_full_frame_marker_uses_center_not_corrected() -> None:
    from lowmag_img_save_with_GCdata_tiff_roi import _uncaging_marker_um_from_row

    row = pd.Series(
        {
            "center_x": 40.0,
            "center_y": 50.0,
            "corrected_uncaging_x": 70.0,
            "corrected_uncaging_y": 80.0,
            "small_x_from": 10.0,
            "small_y_from": 10.0,
        }
    )
    unc_x_um, unc_y_um, _, _ = _uncaging_marker_um_from_row(
        row, (128, 128), highmag_side_length_um=128.0 / 15.0, highmag_pixel=128,
    )
    um_per_px = (128.0 / 15.0) / 128.0
    assert np.isclose(unc_x_um, 40.0 * um_per_px)
    assert np.isclose(unc_y_um, 50.0 * um_per_px)


def test_uncaging_rate_labels() -> None:
    from lowmag_img_save_with_GCdata_tiff_roi import uncaging_rate_hz_label

    base = {"State.Acq.msPerLine": 2.0, "State.Acq.linesPerFrame": 128}
    assert uncaging_rate_hz_label({**base, "State.Uncaging.Uncage_FrameInterval": 2}) == "2 Hz"
    assert uncaging_rate_hz_label({**base, "State.Uncaging.Uncage_FrameInterval": 4}) == "1 Hz"
    assert uncaging_rate_hz_label({**base, "State.Uncaging.Uncage_FrameInterval": 8}) == "0.5 Hz"
    assert uncaging_rate_hz_label({"State.Uncaging.pulseSetInterval_forFrame": 500}) == "2 Hz"


def main() -> int:
    tests = [
        test_80frame_uses_metadata_and_averages_windows,
        test_80frame_without_metadata_is_skipped,
        test_intensity_divides_by_nave_before_pre_baseline,
        test_full_frame_marker_uses_center_not_corrected,
        test_uncaging_rate_labels,
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
