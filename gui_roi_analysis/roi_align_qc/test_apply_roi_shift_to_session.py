"""Self-tests for apply_roi_shift_to_session (no data or hardware required).

Run:
    python test_apply_roi_shift_to_session.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apply_roi_shift_to_session import mask_path, write_masks  # noqa: E402
from manual_shift_dataset import SpineSet, translate_mask  # noqa: E402


def _synthetic_set(tmp_dir: str, n_frames: int = 5, empty_last: bool = False) -> SpineSet:
    """Build a tiny SpineSet whose after_path lives in ``tmp_dir``."""
    after_path = os.path.join(tmp_dir, "Tln_1_pos1__highmag_1__0.0_after_align_full.tif")
    stack = np.random.default_rng(0).random((n_frames, 32, 32)).astype(np.float32)
    tifffile.imwrite(after_path, stack)
    auto = np.zeros((32, 32), dtype=bool)
    auto[10:14, 12:17] = True
    manual = np.repeat(auto[None], n_frames, axis=0).copy()
    if empty_last:
        manual[-1] = False
    phases = ["pre"] * 2 + ["unc"] + ["post"] * (n_frames - 3)
    return SpineSet(
        session="s",
        group="g",
        set_label=0,
        spine_stem="Tln_1_pos1__highmag_1_id001",
        before=stack,
        after=stack,
        stored_shift=np.zeros((n_frames, 2)),
        manual_spine=manual,
        manual_shaft=None,
        auto_spine=auto,
        auto_shaft=None,
        auto_spine_core=None,
        phases=phases,
        elapsed_sec=np.arange(n_frames, dtype=float),
        flim_paths=[""] * n_frames,
        z_from=np.zeros(n_frames),
        z_to=np.zeros(n_frames),
        uncaging_yx=None,
        reference_flim_path=None,
        other_spines=np.zeros_like(auto),
        after_path=after_path,
    )


def test_write_masks_translates_and_backs_up() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spine_set = _synthetic_set(tmp)
        out_path = mask_path(spine_set)
        previous = np.repeat(spine_set.auto_spine[None], spine_set.n_frames, axis=0).astype(np.uint8)
        tifffile.imwrite(out_path, previous)
        side = out_path.replace(".tif", "_source.json")
        with open(side, "w", encoding="utf-8") as fh:
            json.dump({"source": "respan_tracked_test"}, fh)

        shifts = np.array([[0, 0], [0, 0], [1, -2], [2, -3], [2, -3]])
        info = write_masks(spine_set, shifts, "unit", backup=True)
        assert info["previous_source"] == "respan_tracked_test"

        written = tifffile.imread(out_path)
        assert written.shape == (5, 32, 32) and written.dtype == np.uint8
        for t, (dy, dx) in enumerate(shifts):
            assert np.array_equal(written[t] > 0, translate_mask(spine_set.auto_spine, dy, dx)), t
        # Same shape in every frame (pixel count preserved; no clipping here)
        assert len(set(int((written[t] > 0).sum()) for t in range(5))) == 1

        backup = out_path.replace("_Spine_roi_mask.tif", "_Spine_roi_mask_before_unit.tif")
        assert os.path.exists(backup) and np.array_equal(tifffile.imread(backup), previous)
        assert os.path.exists(side.replace("_source.json", "_source_before_unit.json"))

        meta = json.loads(open(side, encoding="utf-8").read())
        assert meta["source"] == "local_dendrite_shift:unit"
        assert meta["shifts_dy_dx"] == shifts.tolist()
        assert meta["sha1"] == hashlib.sha1(np.ascontiguousarray(written > 0).tobytes()).hexdigest()

        # Second write must not overwrite the original backup
        write_masks(spine_set, np.zeros_like(shifts), "unit", backup=True)
        assert np.array_equal(tifffile.imread(backup), previous)


def test_spine_set_accepts_empty_manual_frame() -> None:
    """Centroid-based correction must stay finite when a previous mask frame is empty."""
    with tempfile.TemporaryDirectory() as tmp:
        spine_set = _synthetic_set(tmp, empty_last=True)
        assert spine_set.correction.shape == (5, 2)
        assert np.all(np.isfinite(spine_set.correction))
        assert not spine_set.pure_shift[-1]


def test_write_masks_clips_at_border() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spine_set = _synthetic_set(tmp, n_frames=3)
        shifts = np.array([[0, 0], [-11, 0], [0, 16]])
        write_masks(spine_set, shifts, "unit", backup=False)
        written = tifffile.imread(mask_path(spine_set)) > 0
        assert written[0].sum() == spine_set.auto_spine.sum()
        assert written[1].sum() == 3 * 5  # one row clipped off the top
        assert written[2].sum() == 4 * 4  # one column clipped off the right


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
