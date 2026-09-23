"""highpass is an Align_4d_array method. Global defaults stay traditional."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as ndimage_shift

_QC = os.path.dirname(os.path.abspath(__file__))
_CONTROL = str(Path(__file__).resolve().parents[2])
_ASI = str(Path(__file__).resolve().parents[3] / "ongoing" / "ASIcontroller")
for _path in (_QC, _CONTROL, _ASI):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from FLIMageAlignment import (  # noqa: E402
    DEFAULT_ALIGN_METHOD,
    MOTOR_ALIGN_METHOD,
    POST_ACQUISITION_ALIGN_METHOD,
    Align_4d_array,
    highpass_registration_yx,
    highpass_registration_zyx,
)
from respan_spine_quant import (  # noqa: E402
    IncrementalAlignmentCache,
    _align_uncaging_frames_to_reference,
)


def test_defaults_stay_traditional() -> None:
    assert DEFAULT_ALIGN_METHOD == "traditional"
    assert MOTOR_ALIGN_METHOD == "traditional"
    assert POST_ACQUISITION_ALIGN_METHOD == "traditional"


def test_align_4d_highpass_recovers_shift() -> None:
    reference = np.zeros((16, 48, 48), dtype=np.float32)
    reference[6:10, 18:28, 20:30] = 8.0
    reference[4:12, 10:14, 8:40] = 5.0
    true_object = np.array([2.0, 3.0, -4.0])
    moving = ndimage_shift(reference, true_object, order=1, mode="constant", cval=0.0)
    shifts, _aligned = Align_4d_array(
        np.stack([reference, moving]),
        method="highpass",
        apply_shifts=False,
    )
    direct = highpass_registration_zyx(reference, moving)
    assert np.max(np.abs(shifts[1] - direct)) < 1e-6
    assert np.max(np.abs(shifts[1] + true_object)) < 0.75


def test_uncaging_highpass_recovers_yx() -> None:
    reference = np.zeros((48, 48), dtype=np.float32)
    reference[18:28, 20:30] = 8.0
    moving = ndimage_shift(reference, (3.0, -4.0), order=1, mode="constant", cval=0.0)
    _aligned, drift = _align_uncaging_frames_to_reference(
        reference, [moving], method="highpass"
    )
    direct = highpass_registration_yx(reference, moving)
    assert abs(drift[0] - float(direct[0])) < 1e-6
    assert abs(drift[1] - float(direct[1])) < 1e-6
    assert abs(drift[0] + 3.0) < 0.75
    assert abs(drift[1] - 4.0) < 0.75


def test_quant_cache_method_is_explicit() -> None:
    with tempfile.TemporaryDirectory() as folder:
        default_cache = IncrementalAlignmentCache("ref.flim", ch=2, cache_dir=folder)
        assert default_cache.align_method == "traditional"
        highpass_dir = os.path.join(folder, "hp")
        highpass_cache = IncrementalAlignmentCache(
            "ref.flim", ch=2, cache_dir=highpass_dir, align_method="highpass"
        )
        assert highpass_cache.align_method == "highpass"


def main() -> None:
    test_defaults_stay_traditional()
    test_align_4d_highpass_recovers_shift()
    test_uncaging_highpass_recovers_yx()
    test_quant_cache_method_is_explicit()
    print("PASS test_highpass_align_method")


if __name__ == "__main__":
    main()
