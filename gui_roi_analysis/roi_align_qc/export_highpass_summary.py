"""Four-frame highpass summary with the manual ROI overlaid.

Each set becomes one PNG: first pre, first post, last post, and the max
projection of every pre and post frame. Uncaging frames are left out of the
projection. The green outline is the frame-0 manual ROI. On the three single
frames, orange is that frame's manual ROI mapped into the highpass view.

Usage:
    python export_highpass_summary.py
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
    ORANGE,
    SESSIONS,
    SetArrays,
    _paint_two,
    _safe,
    _set_keys,
    _to_gray,
    load_set,
    map_manual_mask,
)
from field_align import highpass_spatial  # noqa: E402
from tracking import mask_centroid, translate_image  # noqa: E402

SUMMARY_DIR = os.path.join(os.path.dirname(DEFAULT_OUT_DIR), "highpass_four_frames")
SCALE = 4
HEADER_H = 28
BANNER_H = 48


def summary_frame_indices(phases: list[str]) -> tuple[int, int, int, list[int]] | None:
    """First pre, first post, last post, and every pre/post index.

    Returns None when either phase is missing. Uncaging is not included.
    """
    pre = [i for i, phase in enumerate(phases) if str(phase).strip().lower() == "pre"]
    post = [i for i, phase in enumerate(phases) if str(phase).strip().lower() == "post"]
    if not pre or not post:
        return None
    return pre[0], post[0], post[-1], pre + post


def highpass_view_shifts(before: np.ndarray) -> np.ndarray:
    """Content shift that undoes the highpass estimate. Shape (T, 2)."""
    return -highpass_spatial(before)


def _aligned(frame: np.ndarray, shift_yx: np.ndarray) -> np.ndarray:
    return translate_image(frame, float(shift_yx[0]), float(shift_yx[1]))


def render_summary(arrays: SetArrays, shifts: np.ndarray) -> np.ndarray | None:
    """One horizontal strip of the four highpass views."""
    chosen = summary_frame_indices(arrays.phase)
    if chosen is None:
        return None
    pre_first, post_first, post_last, max_frames = chosen

    views: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    singles = (
        ("pre first", pre_first),
        ("post first", post_first),
        ("post last", post_last),
    )
    for label, frame in singles:
        view_shift = shifts[frame]
        views[label] = _aligned(arrays.before[frame], view_shift)
        masks[label] = map_manual_mask(arrays.manual[frame], view_shift, arrays.applied[frame])
    aligned_stack = np.stack([_aligned(arrays.before[i], shifts[i]) for i in max_frames])
    views["max pre+post"] = np.max(aligned_stack, axis=0)
    masks["max pre+post"] = np.zeros_like(arrays.manual[0])

    frame0 = arrays.manual[0]
    panels = []
    for label in ("pre first", "post first", "post last", "max pre+post"):
        image = views[label]
        finite = image[np.isfinite(image)]
        if finite.size:
            vmin, vmax = float(np.percentile(finite, 1)), float(np.percentile(finite, 99.5))
        else:
            vmin, vmax = 0.0, 1.0
        gray = _to_gray(image, vmin, vmax)
        green = frame0
        orange = masks[label]
        big = cv2.resize(gray, (gray.shape[1] * SCALE, gray.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST)
        green_big = cv2.resize(green.astype(np.uint8), (green.shape[1] * SCALE, green.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST) > 0
        orange_big = cv2.resize(orange.astype(np.uint8), (orange.shape[1] * SCALE, orange.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST) > 0
        painted = _paint_two(big, green_big, orange_big)
        header = np.zeros((HEADER_H, painted.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.vstack([header, painted]))
    strip = np.hstack(panels)
    banner = np.zeros((BANNER_H, strip.shape[1], 3), dtype=np.uint8)
    title = f"{arrays.session}  {arrays.group}  set {arrays.set_label}   highpass"
    cv2.putText(banner, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, "green: frame-0 ROI", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
    cv2.putText(banner, "orange: this-frame manual ROI", (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)
    cv2.putText(banner, "yellow: overlap", (560, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, strip])


def export_summaries(out_dir: str = SUMMARY_DIR) -> list[str]:
    """Write one PNG per set that has both pre and post. Returns stems written."""
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    lines = [
        "highpass alignment of the raw projection",
        "panels: first pre, first post, last post, max of all pre and post",
        "uncaging frames are excluded from the max projection",
        "green: frame-0 manual ROI, fixed",
        "orange: this frame's manual ROI, mapped into the highpass view",
        "the max panel has only the green ROI",
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
            shifts = highpass_view_shifts(arrays.before)
            canvas = render_summary(arrays, shifts)
            if canvas is None:
                print(f"  skip {group} set {set_label}", flush=True)
                lines.append(f"SKIP {stem} no pre or post")
                continue
            path = os.path.join(out_dir, stem + ".png")
            cv2.imwrite(path, canvas)
            written.append(stem)
            cy, cx = mask_centroid(arrays.manual[0])
            lines.append(f"{stem}  roi_y={cy:.1f}  roi_x={cx:.1f}")
            print(f"  {group} set {set_label}", flush=True)
    with open(os.path.join(out_dir, "index.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return written


def main() -> None:
    written = export_summaries(SUMMARY_DIR)
    print(f"sets {len(written)} -> {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
