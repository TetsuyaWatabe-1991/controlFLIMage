# -*- coding: utf-8 -*-
"""Rule-based auto rating (1-4) from RESPAN mushroom detection features."""

from __future__ import annotations

from typing import Any

from mushroom_bandpass import (
    HEAD_AREA_TIER_BAND,
    HEAD_TO_DEND_TIER_BAND,
    HEAD_VOL_TIER_BAND,
    SEG_AREA_TIER_BAND,
    band_tier_score,
)

AUTO_RATING_LABELS = {
    1: "absolutely_reject",
    2: "usually_not_selected",
    3: "acceptable",
    4: "appropriate",
}

AUTO_RATING_SHORT_LABELS = {
    1: "1 REJECT",
    2: "2 POOR",
    3: "3 OK",
    4: "4 GOOD",
}

AUTO_RATING_COLORS = {
    1: "#ff3333",
    2: "#ffaa00",
    3: "#66cc44",
    4: "#00cc66",
}

def predict_auto_rating(
    *,
    head_vol_um3: float,
    head_area_um2: float,
    respan_head_euclidean_dist_to_dend: float,
    shaft_to_head_um: float,
    seg_area_um2: float | None = None,
) -> tuple[int, str, str]:
    """
    Predict a 1-4 quality score from detection geometry.

    Returns:
        (rating, rating_label, short_reason)
    """
    seg_area = seg_area_um2 if seg_area_um2 is not None else head_area_um2 * 0.75
    parts: list[tuple[str, float]] = [
        ("head_vol", band_tier_score(head_vol_um3, HEAD_VOL_TIER_BAND)),
        (
            "dist_dend",
            band_tier_score(
                respan_head_euclidean_dist_to_dend,
                HEAD_TO_DEND_TIER_BAND,
                prefer_lower=True,
            ),
        ),
        ("head_area", band_tier_score(head_area_um2, HEAD_AREA_TIER_BAND)),
        ("seg_area", band_tier_score(seg_area, SEG_AREA_TIER_BAND)),
    ]
    weighted = (
        2.5 * parts[0][1]
        + 2.0 * parts[1][1]
        + 1.5 * parts[2][1]
        + 1.5 * parts[3][1]
    )
    if weighted >= 5.0:
        rating = 4
    elif weighted >= 2.0:
        rating = 3
    elif weighted >= 0.0:
        rating = 2
    else:
        rating = 1

    reason_bits = []
    for name, sub in parts:
        if sub >= 1.0:
            reason_bits.append(f"{name}+")
        elif sub <= -0.5:
            reason_bits.append(f"{name}-")
    reason = ",".join(reason_bits) if reason_bits else "mixed"
    return rating, AUTO_RATING_LABELS[rating], reason


def predict_auto_rating_from_feature_row(row: dict[str, Any]) -> tuple[int, str, str]:
    """Convenience wrapper for feature dicts / CSV rows."""
    seg_area = row.get("seg_area_um2")
    return predict_auto_rating(
        head_vol_um3=float(row["head_vol_um3"]),
        head_area_um2=float(row["head_area_um2"]),
        respan_head_euclidean_dist_to_dend=float(row["respan_head_euclidean_dist_to_dend"]),
        shaft_to_head_um=float(row["shaft_to_head_um"]),
        seg_area_um2=float(seg_area) if seg_area not in (None, "", "nan") else None,
    )
