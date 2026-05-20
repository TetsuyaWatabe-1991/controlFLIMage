# -*- coding: utf-8 -*-
"""ROI parameter grids for MIP-only comparison sweeps."""
from __future__ import annotations

import copy
import itertools
from typing import Any

FIXED_MIN_ROI = 1
FIXED_MAX_ROI = 200

# Match deepd3_mushroom_spine_assign_save.py when updating baseline
BASELINE_ROI_PARAMS: dict[str, Any] = {
    "roi_peakThreshold": 0.95,
    "roi_areaThreshold": 0.5,
    "roi_seedDelta": 0.1,
    "roi_distanceToSeed": 10,
    "min_roi_size": FIXED_MIN_ROI,
    "max_roi_size": FIXED_MAX_ROI,
    "min_planes": 1,
}

BASELINE_SPINE_FILTERS: dict[str, float] = {
    "upper_spine_pixel_percentile": 98.0,
    "lower_spine_pixel_percentile": 1.0,
    "upper_spine_intensity_percentile": 98.0,
    "lower_spine_intensity_percentile": 1.0,
}


def combo_id(roi: dict[str, Any], *, roi_mode: str = "thresholded") -> str:
    a = int(round(float(roi["roi_areaThreshold"]) * 100))
    base = f"a{a:03d}_min{FIXED_MIN_ROI}_max{FIXED_MAX_ROI}"
    if roi_mode == "floodfill":
        p = int(round(float(roi["roi_peakThreshold"]) * 100))
        return f"{base}_p{p:03d}_floodfill"
    return base


def _with_baseline(**overrides: Any) -> dict[str, Any]:
    c = copy.deepcopy(BASELINE_ROI_PARAMS)
    c["min_roi_size"] = FIXED_MIN_ROI
    c["max_roi_size"] = FIXED_MAX_ROI
    c.update(overrides)
    return c


def _dedupe_combos(combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in combos:
        key = combo_id(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def grid_quick() -> list[dict[str, Any]]:
    """Baseline + area tweaks (min=1, max=200; thresholded mode)."""
    return [
        {"roi": _with_baseline(), "roi_mode": "thresholded"},
        {"roi": _with_baseline(roi_areaThreshold=0.35), "roi_mode": "thresholded"},
        {"roi": _with_baseline(roi_areaThreshold=0.45), "roi_mode": "thresholded"},
        {"roi": _with_baseline(roi_areaThreshold=0.55), "roi_mode": "thresholded"},
    ]


def grid_area() -> list[dict[str, Any]]:
    areas = [round(0.10 + 0.03 * i, 2) for i in range(21)]
    return [{"roi": _with_baseline(roi_areaThreshold=a), "roi_mode": "thresholded"} for a in areas]


def grid_large() -> list[dict[str, Any]]:
    """
    Main sweep: roi_areaThreshold only (thresholded mode).
    peakThreshold is ignored in thresholded mode — see grid_floodfill.
    """
    areas = [round(0.08 + 0.01 * i, 2) for i in range(63)]  # 0.08 .. 0.70 step 0.01
    return [
        {"roi": _with_baseline(roi_areaThreshold=a), "roi_mode": "thresholded"}
        for a in areas
    ]


def grid_compare_a015() -> list[dict[str, Any]]:
    """Side-by-side: thresholded vs floodfill at roi_areaThreshold=0.15."""
    roi = _with_baseline(roi_areaThreshold=0.15)
    return [
        {
            "roi": roi,
            "roi_mode": "thresholded",
            "combo_id": "a015_thresh",
        },
        {
            "roi": roi,
            "roi_mode": "floodfill",
            "combo_id": "a015_p095_floodfill",
        },
    ]


def grid_floodfill_a020_peak() -> list[dict[str, Any]]:
    """Floodfill: area=0.2 fixed, peak 0.2..0.9 step 0.1 (8 combos)."""
    peaks = [round(0.2 + 0.1 * i, 2) for i in range(8)]
    combos: list[dict[str, Any]] = []
    for p in peaks:
        roi = _with_baseline(roi_areaThreshold=0.2, roi_peakThreshold=p)
        combos.append(
            {
                "roi": roi,
                "roi_mode": "floodfill",
                "combo_id": combo_id(roi, roi_mode="floodfill"),
            }
        )
    return combos


def grid_floodfill30() -> list[dict[str, Any]]:
    """
    ~30 floodfill combos: area x peak (main knobs).
    Centered around a=0.15 p=0.95 that looked reasonable on 256_4x low-mag.
    """
    areas = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    peaks = [0.75, 0.80, 0.85, 0.90, 0.95]
    combos: list[dict[str, Any]] = []
    for a, p in itertools.product(areas, peaks):
        roi = _with_baseline(roi_areaThreshold=a, roi_peakThreshold=p)
        combos.append(
            {
                "roi": roi,
                "roi_mode": "floodfill",
                "combo_id": combo_id(roi, roi_mode="floodfill"),
            }
        )
    return combos


def grid_floodfill() -> list[dict[str, Any]]:
    """Wider floodfill grid (area 0.15–0.70 x peak)."""
    areas = [round(0.15 + 0.05 * i, 2) for i in range(12)]
    peaks = [0.70, 0.80, 0.90, 0.95]
    combos: list[dict[str, Any]] = []
    for a, p in itertools.product(areas, peaks):
        roi = _with_baseline(roi_areaThreshold=a, roi_peakThreshold=p)
        combos.append(
            {
                "roi": roi,
                "roi_mode": "floodfill",
                "combo_id": combo_id(roi, roi_mode="floodfill"),
            }
        )
    return combos


def grid_mega() -> list[dict[str, Any]]:
    """
    area x spine upper percentiles (thresholded); min=1 max=200 fixed.
    """
    areas = [round(0.10 + 0.02 * i, 2) for i in range(31)]
    upper_px = [50.0, 60.0, 70.0, 80.0, 90.0, 98.0]
    upper_int = [50.0, 70.0, 90.0, 98.0]
    combos: list[dict[str, Any]] = []
    for a, upx, uint in itertools.product(areas, upper_px, upper_int):
        roi = _with_baseline(roi_areaThreshold=a)
        cid = f"{combo_id(roi)}_px{int(upx)}_in{int(uint)}"
        combos.append(
            {
                "roi": roi,
                "roi_mode": "thresholded",
                "combo_id": cid,
                "spine_filters": {
                    "upper_spine_pixel_percentile": upx,
                    "lower_spine_pixel_percentile": 1.0,
                    "upper_spine_intensity_percentile": uint,
                    "lower_spine_intensity_percentile": 1.0,
                },
            }
        )
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in combos:
        key = c["combo_id"]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def normalize_combo_entry(
    entry: dict[str, Any] | dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], str, str]:
    """Return (roi_params, spine_filters, combo_id, roi_mode)."""
    if isinstance(entry, dict) and "roi" in entry:
        roi = entry["roi"]
        mode = entry.get("roi_mode", "thresholded")
        filters = entry.get("spine_filters", BASELINE_SPINE_FILTERS)
        cid = entry.get("combo_id", combo_id(roi, roi_mode=mode))
        return roi, filters, cid, mode
    roi = entry  # type: ignore[assignment]
    return (
        roi,
        copy.deepcopy(BASELINE_SPINE_FILTERS),
        combo_id(roi),
        "thresholded",
    )


# Legacy names map to fixed-min/max grids
def grid_max_roi() -> list[dict[str, Any]]:
    return grid_area()


def grid_area_x_maxroi() -> list[dict[str, Any]]:
    return grid_large()


GRID_BUILDERS = {
    "quick": grid_quick,
    "area": grid_area,
    "max_roi": grid_max_roi,
    "area_x_maxroi": grid_area_x_maxroi,
    "large": grid_large,
    "compare_a015": grid_compare_a015,
    "floodfill_a020_peak": grid_floodfill_a020_peak,
    "floodfill30": grid_floodfill30,
    "floodfill": grid_floodfill,
    "mega": grid_mega,
}


def get_roi_grid(name: str) -> list[dict[str, Any]]:
    if name not in GRID_BUILDERS:
        raise ValueError(f"Unknown grid {name!r}. Choose from: {list(GRID_BUILDERS)}")
    return GRID_BUILDERS[name]()
