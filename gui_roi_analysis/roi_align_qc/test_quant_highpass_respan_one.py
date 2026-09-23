"""ROI construction for the one-set highpass RESPAN re-quantification."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant_highpass_respan_one import (  # noqa: E402
    CENTER_THICKNESS,
    N_Z,
    SHAFT_DILATION_PX,
    SPINE_DILATION_PX,
    TRACE_ORDER,
    dilate_mask,
    maxproj_zyx,
    outline_mask,
    pick_overlapping_component,
    render_comparison_sheet,
    representative_frame_indices,
    spine_and_dendrite_mips,
)


def test_outline_removes_dilated_dendrite() -> None:
    spine = np.zeros((24, 24), dtype=bool)
    dendrite = np.zeros_like(spine)
    spine[8:12, 8:12] = True
    dendrite[12:20, :] = True
    outline = outline_mask(spine, dendrite)
    assert not np.any(outline & dilate_mask(dendrite, SHAFT_DILATION_PX))
    assert np.any(outline & ~dilate_mask(spine, 0))
    assert SPINE_DILATION_PX == 4


def test_maxproj_fills_only_the_central_slab() -> None:
    image = np.full((8, 8), 9.2, dtype=np.float32)
    volume = maxproj_zyx(image)
    assert volume.shape == (N_Z, 8, 8)
    start = (N_Z - CENTER_THICKNESS) // 2
    assert np.all(volume[start : start + CENTER_THICKNESS] == 9)
    assert np.all(volume[:start] == 0)
    assert np.all(volume[start + CENTER_THICKNESS :] == 0)


def test_pick_uses_overlap_then_nearest_centroid() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:5, 2:5] = 1
    labels[14:18, 14:18] = 2
    manual = np.zeros((20, 20), dtype=bool)
    manual[14:17, 14:17] = True
    chosen, overlap, n_components = pick_overlapping_component(labels, manual)
    assert n_components == 2
    assert overlap > 0
    assert chosen[15, 15]
    assert not chosen[3, 3]

    missed = np.zeros_like(manual)
    missed[0:2, 0:2] = True
    nearest, overlap_zero, _n = pick_overlapping_component(labels, missed)
    assert overlap_zero == 0
    assert nearest[3, 3]
    assert not nearest[15, 15]


def test_filtered_instances_are_preferred_over_class_labels() -> None:
    class_labels = np.zeros((2, 6, 6), dtype=np.uint8)
    class_labels[:, 1:3, 1:3] = 1
    class_labels[:, 4:6, 4:6] = 2
    filtered = np.zeros((2, 6, 6), dtype=np.uint8)
    filtered[:, 1:3, 1:3] = 7
    spine_labels, dendrite = spine_and_dendrite_mips(class_labels, filtered)
    assert spine_labels[1, 1] == 7
    assert dendrite[5, 5]
    assert not dendrite[1, 1]


def test_cnt_keys_and_label_names() -> None:
    import pandas as pd

    from quant_highpass_respan_cnt_batch import cnt_set_keys, find_label_tiff

    frame = pd.DataFrame(
        {
            "group": ["cnt_a_", "AP5_a_", "cnt_a_", "cnt_b_"],
            "nth_set_label": [0, 0, 0, 1],
        }
    )
    assert cnt_set_keys(frame) == [("cnt_a_", 0), ("cnt_b_", 1)]
    missing = find_label_tiff(Path("."), "no_such_file.tif")
    assert missing is None


def test_representative_frames_cover_pre_and_post() -> None:
    phases = ["pre", "pre", "uncaging", "post", "post", "post"]
    marks = representative_frame_indices(phases)
    assert [index for _label, index in marks] == [0, 1, 3, 4, 5]
    assert [label for label, _index in marks] == [
        "pre first",
        "pre last",
        "post first",
        "post middle",
        "post last",
    ]


def test_sheet_has_one_row_per_roi() -> None:
    frame = np.zeros((16, 16), dtype=np.float32)
    frame[6:10, 6:10] = 5.0
    stack = np.stack([frame for _ in range(6)])
    manual = np.stack([(frame > 0) for _ in range(6)])
    phases = ["pre", "pre", "uncaging", "post", "post", "post"]
    minutes = np.arange(6, dtype=np.float64)
    traces = {name: np.linspace(1.0, 1.4, 6) for name in TRACE_ORDER}
    sheet = render_comparison_sheet(
        stack,
        stack,
        manual,
        np.zeros((6, 2)),
        phases,
        minutes,
        traces,
        frame > 0,
        frame > 0,
        scale=2,
    )
    assert sheet.shape[0] > 16 * 2 * 4
    assert sheet.shape[1] > 16 * 2 * 5


def main() -> None:
    test_outline_removes_dilated_dendrite()
    test_maxproj_fills_only_the_central_slab()
    test_pick_uses_overlap_then_nearest_centroid()
    test_filtered_instances_are_preferred_over_class_labels()
    test_cnt_keys_and_label_names()
    test_representative_frames_cover_pre_and_post()
    test_sheet_has_one_row_per_roi()
    print("PASS test_quant_highpass_respan_one")


if __name__ == "__main__":
    main()
