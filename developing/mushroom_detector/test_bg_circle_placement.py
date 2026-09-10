# -*- coding: utf-8 -*-
"""Headless checks: loosened background-circle placement."""

from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTROL = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_THIS_DIR, _CONTROL):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _far_dark_patch_scene():
    """Spine near (40, 40); dark BG patch around (40, 95), ~55 px away."""
    from respan_mushroom_core import build_circle_mask_2d

    height, width = 128, 128
    spine = build_circle_mask_2d((height, width), 40.0, 40.0, 5.0)
    shaft = build_circle_mask_2d((height, width), 40.0, 40.0, 4.0)
    low_int = np.zeros((height, width), dtype=bool)
    low_int[30:51, 80:111] = True
    return low_int, spine, shaft


def test_constants_match_plan() -> None:
    from respan_mushroom_core import (
        BG_APPROX_OFFSET_SCALES,
        BG_CIRCLE_MIN_INSIDE_FRACTION,
        BG_EDGE_EXCLUDE_PERCENT,
        BG_EXCLUSION_RADIUS_SCALE,
        BG_INTENSITY_PERCENTILE,
        BG_SEARCH_RADIUS_SCALE,
    )

    assert BG_INTENSITY_PERCENTILE == 70.0
    assert BG_SEARCH_RADIUS_SCALE == 4.0
    assert BG_EDGE_EXCLUDE_PERCENT == 15.0
    assert BG_EXCLUSION_RADIUS_SCALE == 2.0
    assert BG_CIRCLE_MIN_INSIDE_FRACTION == 0.70
    assert BG_APPROX_OFFSET_SCALES == (1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.0)


def test_old_approximate_search_misses_far_patch() -> None:
    from respan_mushroom_core import (
        BG_EDGE_EXCLUDE_PERCENT,
        BG_EXCLUSION_RADIUS_SCALE,
        MIN_BG_CIRCLE_RADIUS_PX,
        _find_bg_center_approximate,
        build_bg_exclusion_mask_2d,
        build_image_interior_mask_2d,
        mask_equivalent_circle_radius_px,
    )
    import respan_mushroom_core as core

    low_int, spine, shaft = _far_dark_patch_scene()
    image_shape = low_int.shape
    placement = low_int & build_image_interior_mask_2d(
        image_shape, BG_EDGE_EXCLUDE_PERCENT
    )
    exclusion = build_bg_exclusion_mask_2d(
        spine, shaft, radius_scale=BG_EXCLUSION_RADIUS_SCALE
    )
    base_radius = max(
        mask_equivalent_circle_radius_px(spine),
        mask_equivalent_circle_radius_px(shaft),
        MIN_BG_CIRCLE_RADIUS_PX,
    )
    old_scales = core.BG_APPROX_OFFSET_SCALES
    core.BG_APPROX_OFFSET_SCALES = (1.0, 1.5, 2.0, 2.5)
    try:
        center, _ = _find_bg_center_approximate(
            40.0,
            40.0,
            base_radius,
            placement,
            exclusion,
            image_shape,
            search_radius_scale=2.0,
            mask_radius_scale=1.0,
        )
    finally:
        core.BG_APPROX_OFFSET_SCALES = old_scales
    assert center is None, f"old search should miss far patch, got {center}"


def test_wider_search_accepts_far_dark_patch() -> None:
    from respan_mushroom_core import compute_background_circle_roi

    low_int, spine, shaft = _far_dark_patch_scene()
    bg_mask, center, radius = compute_background_circle_roi(
        low_int, spine, shaft, shaft, approximate=True
    )
    assert bg_mask.any(), "wider search should place a non-empty BG circle"
    assert center is not None
    cy, cx = center
    assert 30 <= cy <= 50, cy
    assert 80 <= cx <= 110, cx
    assert radius > 0


def test_centroid_fallback_when_offsets_miss() -> None:
    from respan_mushroom_core import (
        MIN_BG_CIRCLE_RADIUS_PX,
        _find_bg_center_largest_component_centroid,
        build_circle_mask_2d,
        build_image_interior_mask_2d,
        BG_EDGE_EXCLUDE_PERCENT,
        BG_EXCLUSION_RADIUS_SCALE,
        build_bg_exclusion_mask_2d,
    )

    height, width = 128, 128
    spine = build_circle_mask_2d((height, width), 40.0, 40.0, 5.0)
    shaft = build_circle_mask_2d((height, width), 40.0, 40.0, 4.0)
    placement = np.zeros((height, width), dtype=bool)
    placement[70:90, 90:110] = True
    placement &= build_image_interior_mask_2d(
        (height, width), BG_EDGE_EXCLUDE_PERCENT
    )
    exclusion = build_bg_exclusion_mask_2d(
        spine, shaft, radius_scale=BG_EXCLUSION_RADIUS_SCALE
    )
    center, radius = _find_bg_center_largest_component_centroid(
        placement, exclusion, MIN_BG_CIRCLE_RADIUS_PX
    )
    assert center is not None
    cy, cx = center
    assert 70 <= cy <= 90, cy
    assert 90 <= cx <= 110, cx
    assert radius == MIN_BG_CIRCLE_RADIUS_PX


def test_seventy_percent_containment_allows_partial_overlap() -> None:
    from respan_mushroom_core import (
        circle_fully_inside_mask,
        circle_mostly_inside_mask,
        build_circle_mask_2d,
    )

    allowed = np.zeros((64, 64), dtype=bool)
    allowed[16:48, 16:48] = True
    # Center slightly inside the allowed region so most, but not all, of the circle is inside.
    cy, cx, radius = 18.0, 32.0, 6.0
    circle = build_circle_mask_2d((64, 64), cy, cx, radius)
    frac = float(allowed[circle].sum()) / float(circle.sum())
    assert 0.70 <= frac < 1.0, frac
    assert not circle_fully_inside_mask(cy, cx, radius, allowed)
    assert circle_mostly_inside_mask(cy, cx, radius, allowed)


def main() -> int:
    tests = [
        test_constants_match_plan,
        test_old_approximate_search_misses_far_patch,
        test_wider_search_accepts_far_dark_patch,
        test_centroid_fallback_when_offsets_miss,
        test_seventy_percent_containment_allows_partial_overlap,
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
