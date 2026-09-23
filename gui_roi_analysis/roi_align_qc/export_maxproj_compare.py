"""Max projections for raw, current, full spatial, and highpass, plus an RGB overlay.

Two rows. The top row is the max of every pre and post frame under each
alignment. Uncaging is excluded. The bottom row is the highpass view of the
first pre (blue), first post (green), and last post (red) in one image.

Usage:
    python export_maxproj_compare.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from export_align_frames import (  # noqa: E402
    DEFAULT_OUT_DIR,
    GREEN,
    SESSIONS,
    SetArrays,
    _paint_two,
    _safe,
    _set_keys,
    _to_gray,
    load_set,
    mask_edge,
)
from export_highpass_summary import SCALE, summary_frame_indices  # noqa: E402
from field_align import full_spatial, highpass_spatial  # noqa: E402
from tracking import translate_image  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(DEFAULT_OUT_DIR), "maxproj_methods")
HEADER_H = 28
BANNER_H = 28
METHOD_ORDER = ("raw", "current", "full spatial", "highpass")


def method_view_shifts(before: np.ndarray, applied: np.ndarray) -> dict[str, np.ndarray]:
    """Content shift applied to the raw frame for each max-projection method."""
    return {
        "raw": np.zeros_like(applied),
        "current": np.asarray(applied, dtype=np.float64),
        "full spatial": -full_spatial(before),
        "highpass": -highpass_spatial(before),
    }


def max_projection(before: np.ndarray, shifts: np.ndarray, frames: list[int]) -> np.ndarray:
    """Max of the aligned pre and post frames."""
    aligned = [
        translate_image(before[i], float(shifts[i, 0]), float(shifts[i, 1]))
        for i in frames
    ]
    return np.max(np.stack(aligned), axis=0)


def rgb_overlay(pre: np.ndarray, post_first: np.ndarray, post_last: np.ndarray) -> np.ndarray:
    """BGR image: blue = first pre, green = first post, red = last post."""
    blue = _to_gray(pre, *_limits(pre))
    green = _to_gray(post_first, *_limits(post_first))
    red = _to_gray(post_last, *_limits(post_last))
    return np.dstack([blue, green, red])


def _limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1, 99.5])
    return float(low), float(high)


def _gray_panel(image: np.ndarray, frame0_mask: np.ndarray, label: str) -> np.ndarray:
    vmin, vmax = _limits(image)
    gray = _to_gray(image, vmin, vmax)
    big = cv2.resize(gray, (gray.shape[1] * SCALE, gray.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST)
    green = cv2.resize(
        frame0_mask.astype(np.uint8),
        (frame0_mask.shape[1] * SCALE, frame0_mask.shape[0] * SCALE),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    painted = _paint_two(big, green, np.zeros_like(green))
    header = np.zeros((HEADER_H, painted.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, painted])


def _rgb_panel(color: np.ndarray, frame0_mask: np.ndarray) -> np.ndarray:
    big = cv2.resize(color, (color.shape[1] * SCALE, color.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST)
    green = cv2.resize(
        frame0_mask.astype(np.uint8),
        (frame0_mask.shape[1] * SCALE, frame0_mask.shape[0] * SCALE),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    big[mask_edge(green)] = (255, 255, 255)
    header = np.zeros((HEADER_H, big.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "highpass RGB", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, big])


def render_maxproj_compare(arrays: SetArrays, shifts: dict[str, np.ndarray]) -> np.ndarray | None:
    """Two-row tile: four max projections, then the highpass RGB overlay."""
    chosen = summary_frame_indices(arrays.phase)
    if chosen is None:
        return None
    pre_first, post_first, post_last, max_frames = chosen
    frame0 = arrays.manual[0]
    panels = []
    for name in METHOD_ORDER:
        image = max_projection(arrays.before, shifts[name], max_frames)
        panels.append(_gray_panel(image, frame0, name))
    top = np.hstack(panels)
    pre = translate_image(arrays.before[pre_first], float(shifts["highpass"][pre_first, 0]), float(shifts["highpass"][pre_first, 1]))
    post_a = translate_image(arrays.before[post_first], float(shifts["highpass"][post_first, 0]), float(shifts["highpass"][post_first, 1]))
    post_b = translate_image(arrays.before[post_last], float(shifts["highpass"][post_last, 0]), float(shifts["highpass"][post_last, 1]))
    rgb = _rgb_panel(rgb_overlay(pre, post_a, post_b), frame0)
    bottom = np.zeros((rgb.shape[0], top.shape[1], 3), dtype=np.uint8)
    bottom[:, : rgb.shape[1]] = rgb
    cv2.putText(bottom, "blue: pre first", (rgb.shape[1] + 16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 1, cv2.LINE_AA)
    cv2.putText(bottom, "green: post first", (rgb.shape[1] + 16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(bottom, "red: post last", (rgb.shape[1] + 16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(bottom, "white: frame-0 ROI", (rgb.shape[1] + 16, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    banner = np.zeros((BANNER_H, top.shape[1], 3), dtype=np.uint8)
    title = f"{arrays.session}  {arrays.group}  set {arrays.set_label}"
    cv2.putText(banner, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, "green outline: frame-0 manual ROI", (700, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    return np.vstack([banner, top, bottom])


def export_maxproj(out_dir: str = OUT_DIR) -> list[str]:
    """Write one two-row PNG per set. Returns the stems written."""
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    lines = [
        "top row: max projection of pre and post, uncaging excluded",
        "methods left to right: raw, current, full spatial, highpass",
        "bottom: highpass frames overlaid, blue=pre first, green=post first, red=post last",
        "green outline on the max projections and white outline on the RGB image: frame-0 manual ROI",
        "",
    ]
    for session, df_path in SESSIONS.items():
        print(f"session {session}", flush=True)
        combined = pd.read_pickle(df_path)
        for group, set_label in _set_keys(combined):
            arrays = load_set(combined, session, group, set_label)
            stem = f"{_safe(session)}_{_safe(group)}_set{set_label}"
            if arrays is None:
                lines.append(f"SKIP {stem}")
                continue
            shifts = method_view_shifts(arrays.before, arrays.applied)
            canvas = render_maxproj_compare(arrays, shifts)
            if canvas is None:
                print(f"  skip {group} set {set_label}", flush=True)
                lines.append(f"SKIP {stem} no pre or post")
                continue
            cv2.imwrite(os.path.join(out_dir, stem + ".png"), canvas)
            written.append(stem)
            print(f"  {group} set {set_label}", flush=True)
    with open(os.path.join(out_dir, "index.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return written


def main() -> None:
    written = export_maxproj(OUT_DIR)
    print(f"sets {len(written)} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
