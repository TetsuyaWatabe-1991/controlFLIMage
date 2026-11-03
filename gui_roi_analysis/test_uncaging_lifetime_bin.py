# -*- coding: utf-8 -*-
"""Headless checks for uncaging lifetime frame binning (no FLIM files)."""

from __future__ import annotations

import os
import sys

import numpy as np

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTROL = os.path.normpath(os.path.join(_GUI_DIR, ".."))
for _p in (_GUI_DIR, _CONTROL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gui_roi_fast_simple import (  # noqa: E402
    apply_binned_uncaging_lifetime,
    binned_uncaging_lifetime_records,
    masked_decay_histogram,
    uncaging_lifetime_bin_ranges,
)


class _SumFitter:
    """Stand-in fitter: lifetime equals the histogram sum, so the bin total is visible."""

    def fit_double_exponential(self, x, y, ps_per_unit, sync_rate):
        return {"lifetime": float(np.sum(y))}


def test_bin_ranges_8_and_16() -> None:
    by8 = uncaging_lifetime_bin_ranges(264, 8)
    assert len(by8) == 33
    assert all(end - start == 8 for start, end in by8)
    assert by8[0] == (0, 8) and by8[-1] == (256, 264)

    by16 = uncaging_lifetime_bin_ranges(264, 16)
    assert len(by16) == 17
    assert all(end - start == 16 for start, end in by16[:-1])
    assert by16[-1] == (256, 264)


def test_histogram_drops_dim_and_outside_pixels() -> None:
    frame = np.zeros((3, 3, 2), dtype=np.float64)
    frame[0, 0] = [15, 0]  # at threshold: excluded
    frame[1, 1] = [20, 1]  # inside ROI
    frame[2, 2] = [100, 0]  # bright but outside the mask
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 1
    hist = masked_decay_histogram(frame, mask, photon_threshold=15)
    assert np.allclose(hist, [20, 1])


def test_binned_lifetime_sums_frames_and_keeps_short_tail() -> None:
    n_pre, n_unc, n_time = 2, 10, 3
    image = np.zeros((n_unc, 1, 1, 4, 4, n_time), dtype=np.float64)
    image[:, 0, 0, 1, 1, 0] = 20
    image[:, 0, 0, 1, 1, 1] = 1
    image[:, 0, 0, 0, 0, 0] = 100  # outside the ROI
    mask = np.zeros((n_pre + n_unc, 4, 4), dtype=np.uint8)
    mask[:, 1, 1] = 1
    records = binned_uncaging_lifetime_records(
        image,
        {"Spine": mask},
        n_pre,
        bin_frames=8,
        photon_threshold=15,
        total_photon_threshold=100,
        fitter=_SumFitter(),
        sync_rate=80e6,
    )
    assert [r["slice"] for r in records] == [0, 8]
    assert records[0]["unc_lifetime_bin_n"] == 8
    assert records[1]["unc_lifetime_bin_n"] == 2
    assert records[0]["Spine_Ch1_total_photon"] == 8 * 21
    assert records[0]["Spine_Ch1_lifetime"] == 8 * 21
    assert records[1]["Spine_Ch1_total_photon"] == 2 * 21
    assert np.isnan(records[1]["Spine_Ch1_lifetime"])

    rows = []
    for t in range(n_unc):
        rows.append({
            "phase": "unc",
            "slice": t,
            "Spine_Ch1_lifetime": np.nan,
            "Spine_Ch1_total_photon": np.nan,
        })
    rows.append({"phase": "pre", "slice": 0, "Spine_Ch1_lifetime": np.nan})
    apply_binned_uncaging_lifetime(rows, records)
    assert rows[0]["Spine_Ch1_lifetime"] == 8 * 21
    assert all(np.isnan(rows[t]["Spine_Ch1_lifetime"]) for t in range(1, 8))
    assert rows[3]["unc_lifetime_bin_start"] == 0 and rows[3]["unc_lifetime_bin_n"] == 8
    assert np.isnan(rows[8]["Spine_Ch1_lifetime"])
    assert rows[9]["unc_lifetime_bin_start"] == 8 and rows[9]["unc_lifetime_bin_frames"] == 8
    assert "unc_lifetime_bin_start" not in rows[-1]


def test_bin_frames_must_be_positive() -> None:
    try:
        uncaging_lifetime_bin_ranges(10, 0)
    except ValueError:
        return
    raise AssertionError("bin size 0 should be rejected")


def test_skip_lifetime_overrides_uncaging_bin() -> None:
    import contextlib
    import io

    import pandas as pd
    from gui_roi_fast_simple import effective_uncaging_lifetime_bin, quantify_intensity_from_flim

    assert effective_uncaging_lifetime_bin(True, 8) is None
    assert effective_uncaging_lifetime_bin(True, None) is None
    assert effective_uncaging_lifetime_bin(False, 8) == 8
    assert effective_uncaging_lifetime_bin(False, None) is None
    # the entry point applies it before anything else
    buf = io.StringIO()
    empty = pd.DataFrame(columns=["filepath_without_number", "group", "nth_set_label"])
    with contextlib.redirect_stdout(buf):
        quantify_intensity_from_flim(empty, 2, 1, "unused.csv", skip_lifetime_analysis=True,
                                     uncaging_lifetime_bin_frames=8)
    out = buf.getvalue()
    assert "binning (8 frames) ignored" in out and "Uncaging lifetime binned" not in out, out


if __name__ == "__main__":
    test_bin_ranges_8_and_16()
    test_histogram_drops_dim_and_outside_pixels()
    test_binned_lifetime_sums_frames_and_keeps_short_tail()
    test_bin_frames_must_be_positive()
    test_skip_lifetime_overrides_uncaging_bin()
    print("test_uncaging_lifetime_bin: all passed")
