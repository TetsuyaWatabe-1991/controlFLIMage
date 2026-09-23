"""Save one tiled PNG per frame comparing alignment methods on large-drift sets.

Each tile is the raw projection after that method's shift. The green outline
is the frame-0 manual spine ROI, held fixed. The orange outline is the manual
ROI of the displayed frame, mapped into the same view. Both are references:
the manual ROI can itself be off the spine.

Usage:
    python export_align_frames.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from compare_field_aligners import _before_path, _load_stack  # noqa: E402
from field_align import FIELD_ALIGNERS  # noqa: E402
from other_align import OTHER_ALIGNERS  # noqa: E402
from tracking import (  # noqa: E402
    best_sign_error,
    centroid_trajectory,
    mask_centroid,
    pairwise_content_shifts,
    translate_image,
    translate_mask,
)

# Ten sets with the largest human ROI motion or a runaway stored shift.
# Chosen from alignment_diagnosis.csv so the pictures show real drift.
SELECTED: list[tuple[str, str, int]] = [
    ("auto1_20260909", "cnt_3_pos1__highmag_4_", 3),
    ("auto1_20260909", "cnt_4_pos1__highmag_4_", 1),
    ("auto1_20260909", "AP5_14_pos1__highmag_6_", 0),
    ("auto1_20260623", "3_pos1__highmag_2_", 2),
    ("auto1_20260909", "AP5_14_pos1__highmag_1_", 1),
    ("auto1_20260909", "cnt_3_pos1__highmag_5_", 1),
    ("auto1_20260909", "cnt_2_pos1__highmag_2_", 3),
    ("auto1_20260909", "AP5_12_pos1__highmag_4_", 0),
    ("auto1_20260909", "AP5_14_pos1__highmag_3_", 1),
    ("auto1_20260909", "AP5_13_pos1__highmag_7_", 2),
]

SESSIONS: dict[str, str] = {
    "auto1_20260701": r"G:\ImagingData\Tetsuya\20260701\auto1\combined_df_respan.pkl",
    "auto1_20260623": r"G:\ImagingData\Tetsuya\20260623\auto1\combined_df_respan.pkl",
    "auto1_20260909": "//RY-LAB-WS04/ImagingData/Tetsuya/20260909/auto1/combined_df_respan.pkl",
}

DEFAULT_OUT_DIR = r"G:\ImagingData\Tetsuya\20260701\auto1\roi_align_qc\align_compare_frames"

# (key, label). raw / current / manual are built from the stored shift.
# The rest are aligners run on the before-alignment stack.
PANEL_SPECS: list[tuple[str, str]] = [
    ("raw", "raw"),
    ("current", "current"),
    ("manual", "manual ROI"),
    ("full_spatial", "full spatial"),
    ("full_phase", "full phase"),
    ("highpass_spatial", "highpass"),
    ("edge_spatial", "edge"),
    ("tile_median", "tile median"),
    ("farneback", "Farneback"),
    ("pyramidal_lucas_kanade", "pyr LK"),
    ("iterative_lucas_kanade", "iter LK"),
    ("tvl1", "TV-L1"),
    ("orb_ransac", "ORB"),
]

_ALIGNERS = {**FIELD_ALIGNERS, **OTHER_ALIGNERS}

COLS = 5
ROWS = 3
FULL_SCALE = 2
ZOOM_HALF = 22
ZOOM_SCALE = 5
PANEL_W = 256
HEADER_H = 36
BANNER_H = 28
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)


@dataclass
class SetArrays:
    """Raw stack, stored correction, and the manual masks for one spine set."""

    session: str
    group: str
    set_label: int
    before: np.ndarray
    after: np.ndarray
    applied: np.ndarray
    target: np.ndarray
    manual: np.ndarray
    phase: list[str]


def mask_edge(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner boundary of a boolean mask."""
    region = np.asarray(mask, dtype=bool)
    up = np.zeros_like(region)
    down = np.zeros_like(region)
    left = np.zeros_like(region)
    right = np.zeros_like(region)
    up[1:] = region[:-1]
    down[:-1] = region[1:]
    left[:, 1:] = region[:, :-1]
    right[:, :-1] = region[:, 1:]
    return region & ~(up & down & left & right)


def map_manual_mask(mask: np.ndarray, view_shift: np.ndarray, applied: np.ndarray) -> np.ndarray:
    """Place an after-image manual mask into a view built from the raw frame.

    ``view_shift`` is the content translation applied to the raw frame.
    ``applied`` is the content translation already present in the saved
    aligned TIFF, where the manual mask was drawn.
    """
    delta_y = float(view_shift[0] - applied[0])
    delta_x = float(view_shift[1] - applied[1])
    return translate_mask(mask, delta_y, delta_x)


def view_shifts_for_set(before: np.ndarray, applied: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Content translation applied to the raw stack, per method, shape (T, 2)."""
    n_frames = len(before)
    shifts: dict[str, np.ndarray] = {
        "raw": np.zeros((n_frames, 2), dtype=np.float64),
        "current": np.asarray(applied, dtype=np.float64),
        "manual": -np.asarray(target, dtype=np.float64),
    }
    for name, aligner in _ALIGNERS.items():
        # Aligners return object displacement. Negate it to bring content back.
        shifts[name] = -aligner(before)
    return shifts


def _to_gray(image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scaled = (np.asarray(image, dtype=np.float32) - vmin) / max(vmax - vmin, 1e-6)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _paint_two(gray: np.ndarray, green_mask: np.ndarray, orange_mask: np.ndarray) -> np.ndarray:
    """Green is the fixed ROI, orange is this frame, yellow is where they overlap."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    green_edge = mask_edge(green_mask)
    orange_edge = mask_edge(orange_mask)
    rgb[orange_edge & ~green_edge] = ORANGE
    rgb[green_edge & ~orange_edge] = GREEN
    rgb[green_edge & orange_edge] = (0, 255, 255)
    return rgb


def _zoom(image: np.ndarray, mask_green: np.ndarray, mask_orange: np.ndarray, cy: float, cx: float) -> np.ndarray:
    half = ZOOM_HALF
    size = 2 * half + 1
    gray = np.zeros((size, size), dtype=np.uint8)
    green = np.zeros((size, size), dtype=bool)
    orange = np.zeros((size, size), dtype=bool)
    height, width = image.shape
    y0 = int(round(cy)) - half
    x0 = int(round(cx)) - half
    src_y0, src_y1 = max(0, y0), min(height, y0 + size)
    src_x0, src_x1 = max(0, x0), min(width, x0 + size)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    gray[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = image[src_y0:src_y1, src_x0:src_x1]
    green[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = mask_green[src_y0:src_y1, src_x0:src_x1]
    orange[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = mask_orange[src_y0:src_y1, src_x0:src_x1]
    big = cv2.resize(gray, (size * ZOOM_SCALE, size * ZOOM_SCALE), interpolation=cv2.INTER_NEAREST)
    green_big = cv2.resize(green.astype(np.uint8), (size * ZOOM_SCALE, size * ZOOM_SCALE), interpolation=cv2.INTER_NEAREST) > 0
    orange_big = cv2.resize(orange.astype(np.uint8), (size * ZOOM_SCALE, size * ZOOM_SCALE), interpolation=cv2.INTER_NEAREST) > 0
    painted = _paint_two(big, green_big, orange_big)
    pad = PANEL_W - painted.shape[1]
    left = max(pad // 2, 0)
    canvas = np.zeros((painted.shape[0], PANEL_W, 3), dtype=np.uint8)
    canvas[:, left : left + painted.shape[1]] = painted[:, : PANEL_W - left]
    return canvas


def render_panel(
    image: np.ndarray,
    frame0_mask: np.ndarray,
    frame_mask: np.ndarray,
    center_y: float,
    center_x: float,
    vmin: float,
    vmax: float,
    label: str,
    shift_yx: np.ndarray,
) -> np.ndarray:
    """One method: title, full field, and a zoom around the frame-0 spine."""
    gray = _to_gray(image, vmin, vmax)
    full = cv2.resize(gray, (PANEL_W, gray.shape[0] * FULL_SCALE), interpolation=cv2.INTER_NEAREST)
    green = cv2.resize(frame0_mask.astype(np.uint8), (PANEL_W, frame0_mask.shape[0] * FULL_SCALE), interpolation=cv2.INTER_NEAREST) > 0
    orange = cv2.resize(frame_mask.astype(np.uint8), (PANEL_W, frame_mask.shape[0] * FULL_SCALE), interpolation=cv2.INTER_NEAREST) > 0
    top = _paint_two(full, green, orange)
    zoom = _zoom(gray, frame0_mask, frame_mask, center_y, center_x)
    header = np.zeros((HEADER_H, PANEL_W, 3), dtype=np.uint8)
    cv2.putText(header, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        header,
        f"dy {shift_yx[0]:+.1f}  dx {shift_yx[1]:+.1f}",
        (6, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 220, 180),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, top, zoom])


def render_alignment_grid(
    frame: int,
    images: dict[str, np.ndarray],
    frame0_mask: np.ndarray,
    frame_masks: dict[str, np.ndarray],
    shifts: dict[str, np.ndarray],
    center_y: float,
    center_x: float,
    vmin: float,
    vmax: float,
    title: str,
) -> np.ndarray:
    """Tile every method for one frame. Green is fixed, orange is this frame."""
    panels = []
    for key, label in PANEL_SPECS:
        panels.append(
            render_panel(
                images[key],
                frame0_mask,
                frame_masks[key],
                center_y,
                center_x,
                vmin,
                vmax,
                label,
                shifts[key][frame],
            )
        )
    panel_h, panel_w = panels[0].shape[:2]
    blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    cv2.putText(blank, "green: frame-0 manual ROI", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1, cv2.LINE_AA)
    cv2.putText(blank, "orange: this-frame manual ROI", (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ORANGE, 1, cv2.LINE_AA)
    cv2.putText(blank, "yellow: those two overlap", (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(blank, "lower image is the spine zoom", (8, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    while len(panels) < COLS * ROWS:
        panels.append(blank.copy())
    rows = []
    for r in range(ROWS):
        rows.append(np.hstack(panels[r * COLS : (r + 1) * COLS]))
    grid = np.vstack(rows)
    banner = np.zeros((BANNER_H, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, grid])


def load_set(combined: pd.DataFrame, session: str, group: str, set_label: int) -> SetArrays | None:
    """Load the raw stack and the shifts needed to redraw one spine set."""
    labels = pd.to_numeric(combined["nth_set_label"], errors="coerce")
    hit = combined[(combined["group"].astype(str) == group) & (labels == int(set_label))]
    if len(hit) == 0:
        return None
    set_df = hit
    after_path = set_df["after_align_full_save_path"].iloc[0] if "after_align_full_save_path" in set_df else None
    if not isinstance(after_path, str) or not os.path.exists(after_path):
        after_path = set_df["after_align_save_path"].iloc[0]
    if not isinstance(after_path, str) or not os.path.exists(after_path):
        return None
    before_path = _before_path(after_path, set_df)
    if before_path is None:
        return None
    before = _load_stack(before_path)
    after = _load_stack(after_path)
    if before is None or after is None:
        return None
    tiff_dir = os.path.dirname(after_path)
    base = os.path.splitext(os.path.basename(after_path))[0]
    manual_path = os.path.join(tiff_dir, f"{base}_Spine_roi_mask.tif")
    info_path = os.path.join(tiff_dir, f"{base}_frame_info.csv")
    if not os.path.exists(manual_path) or not os.path.exists(info_path):
        return None
    manual = tifffile_read_mask(manual_path)
    info = pd.read_csv(info_path)
    n_frames = min(len(before), len(after), len(manual), len(info))
    before = before[:n_frames]
    after = after[:n_frames]
    manual = np.asarray(manual[:n_frames] > 0)
    info = info.iloc[:n_frames].reset_index(drop=True)
    manual_traj = centroid_trajectory(manual)
    stored = np.column_stack(
        [
            pd.to_numeric(info["shift_y"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(info["shift_x"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    stored = stored - stored[0]
    tiff_shift = pairwise_content_shifts(before, after)
    _err, stored_sign = best_sign_error(stored, tiff_shift)
    applied = stored_sign * stored
    target = manual_traj - applied
    phase = info["phase"].astype(str).tolist() if "phase" in info.columns else [""] * n_frames
    return SetArrays(session, group, int(set_label), before, after, applied, target, manual, phase)


def tifffile_read_mask(path: str) -> np.ndarray:
    import tifffile

    manual = tifffile.imread(path)
    if manual.ndim == 2:
        manual = manual[np.newaxis, ...]
    return manual


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def roi_mean(image: np.ndarray, mask: np.ndarray) -> float:
    """Mean pixel value inside a mask. Empty masks return NaN."""
    selected = np.asarray(mask) > 0
    if not np.any(selected):
        return float("nan")
    return float(np.mean(np.asarray(image)[selected]))


def to_ff0(values: np.ndarray, pre_index: np.ndarray) -> np.ndarray:
    """Divide a trace by its own pre-period mean. Pre of 0 leaves the trace unchanged."""
    baseline = float(np.nanmean(values[pre_index])) if len(pre_index) else float("nan")
    if not np.isfinite(baseline) or abs(baseline) < 1e-8:
        return np.asarray(values, dtype=np.float64).copy()
    return np.asarray(values, dtype=np.float64) / baseline


def _pre_index(phases: list[str]) -> np.ndarray:
    index = np.array([i for i, phase in enumerate(phases) if str(phase).strip().lower() == "pre"], dtype=int)
    if index.size == 0:
        return np.array([0], dtype=int)
    return index


def roi_ff0_traces(
    before: np.ndarray,
    after: np.ndarray,
    manual: np.ndarray,
    shifts: dict[str, np.ndarray],
    phases: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """F/F0 of mean ROI intensity for the manual mask and for each alignment.

    Manual uses the per-frame hand-drawn mask on the saved aligned image.
    Each alignment uses the frame-0 mask on that method's registered image,
    which is the region inside the green outline.
    """
    pre_index = _pre_index(phases)
    n_frames = len(before)
    frame0 = manual[0]
    manual_mean = np.array([roi_mean(after[t], manual[t]) for t in range(n_frames)], dtype=np.float64)
    method_ff0: dict[str, np.ndarray] = {}
    for key, _label in PANEL_SPECS:
        means = np.empty(n_frames, dtype=np.float64)
        for t in range(n_frames):
            view_shift = shifts[key][t]
            view = translate_image(before[t], float(view_shift[0]), float(view_shift[1]))
            means[t] = roi_mean(view, frame0)
        method_ff0[key] = to_ff0(means, pre_index)
    return to_ff0(manual_mean, pre_index), method_ff0


def _plot_xy(
    index: int,
    value: float,
    n_frames: int,
    ymin: float,
    ymax: float,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[int, int] | None:
    if not np.isfinite(value):
        return None
    x = x0 if n_frames <= 1 else x0 + int(round(index / (n_frames - 1) * (x1 - x0)))
    frac = (float(value) - ymin) / max(ymax - ymin, 1e-6)
    y = y1 - int(round(np.clip(frac, 0.0, 1.0) * (y1 - y0)))
    return x, y


def plotted_frame_indices(phases: list[str]) -> list[int]:
    """Frames drawn on the F/F0 plot. Uncaging samples are omitted."""
    return [i for i, phase in enumerate(phases) if str(phase).strip().lower() not in {"unc", "uncaging"}]


def plot_limits(
    manual_ff0: np.ndarray,
    method_ff0: dict[str, np.ndarray],
    phases: list[str],
) -> tuple[float, float]:
    """Y limits from pre and post only, so uncaging spikes do not set the scale."""
    index = np.array(plotted_frame_indices(phases), dtype=int)
    if index.size == 0:
        index = np.arange(len(manual_ff0))
    stacked = np.concatenate([manual_ff0[index]] + [method_ff0[key][index] for key, _label in PANEL_SPECS])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return 0.0, 2.0
    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    if ymax - ymin < 0.05:
        return ymin - 0.1, ymax + 0.1
    pad = 0.08 * (ymax - ymin)
    return ymin - pad, ymax + pad


def render_quant_panel(
    label: str,
    manual_ff0: np.ndarray,
    method_ff0: np.ndarray,
    phases: list[str],
    frame: int,
    ymin: float,
    ymax: float,
    panel_h: int,
    panel_w: int,
) -> np.ndarray:
    """One F/F0 plot. Manual is black. This alignment is red. Both are points joined by lines.

    Uncaging frames are left off the x axis, so pre and post use the full width.
    """
    panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
    cv2.putText(panel, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "F/F0", (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1, cv2.LINE_AA)
    x0, x1 = 40, panel_w - 8
    y0, y1 = HEADER_H + 8, panel_h - 26
    cv2.rectangle(panel, (x0, y0), (x1, y1), (80, 80, 80), thickness=1)
    shown = plotted_frame_indices(phases)
    n_shown = len(shown)
    one = _plot_xy(0, 1.0, max(n_shown, 1), ymin, ymax, x0, y0, x1, y1)
    if one is not None and y0 <= one[1] <= y1:
        cv2.line(panel, (x0, one[1]), (x1, one[1]), (200, 200, 200), 1, cv2.LINE_AA)
    if frame in shown:
        here = _plot_xy(shown.index(frame), ymin, n_shown, ymin, ymax, x0, y0, x1, y1)
        if here is not None:
            cv2.line(panel, (here[0], y0), (here[0], y1), (170, 170, 170), 1, cv2.LINE_AA)

    def _shown_points(series: np.ndarray) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        for rank, index in enumerate(shown):
            point = _plot_xy(rank, float(series[index]), n_shown, ymin, ymax, x0, y0, x1, y1)
            if point is not None:
                points.append(point)
        return points

    manual_pts = _shown_points(manual_ff0)
    for p0, p1 in zip(manual_pts, manual_pts[1:]):
        _dashed_line(panel, p0, p1, (0, 0, 0))
    for point in manual_pts:
        cv2.circle(panel, point, 3, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    method_pts = _shown_points(method_ff0)
    if len(method_pts) >= 2:
        cv2.polylines(panel, [np.array(method_pts, dtype=np.int32)], False, (0, 0, 255), 1, cv2.LINE_AA)
    for point in method_pts:
        cv2.circle(panel, point, 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(panel, f"{ymax:.2f}", (2, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{ymin:.2f}", (2, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
    if shown:
        cv2.putText(panel, str(shown[0]), (x0, panel_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            panel,
            str(shown[-1]),
            (max(x1 - 24, x0), panel_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return panel


def _dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    dash: int = 5,
    gap: int = 3,
) -> None:
    """Draw a dashed segment. Used for the manual trace."""
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if length < 1.0:
        return
    step = dash + gap
    covered = 0
    while covered < length:
        run = min(dash, length - covered)
        t0 = covered / length
        t1 = (covered + run) / length
        p0 = (int(round(start[0] + t0 * (end[0] - start[0]))), int(round(start[1] + t0 * (end[1] - start[1]))))
        p1 = (int(round(start[0] + t1 * (end[0] - start[0]))), int(round(start[1] + t1 * (end[1] - start[1]))))
        cv2.line(image, p0, p1, color, 1, cv2.LINE_AA)
        covered += step


def render_quant_grid(
    manual_ff0: np.ndarray,
    method_ff0: dict[str, np.ndarray],
    phases: list[str],
    frame: int,
    panel_h: int,
    panel_w: int,
) -> np.ndarray:
    """Plot tiles in the same order as the image tiles, including the banner row."""
    ymin, ymax = plot_limits(manual_ff0, method_ff0, phases)
    panels = [
        render_quant_panel(label, manual_ff0, method_ff0[key], phases, frame, ymin, ymax, panel_h, panel_w)
        for key, label in PANEL_SPECS
    ]
    blank = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
    cv2.putText(blank, "black dashed: manual ROI", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(blank, "red: this alignment", (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(blank, "gray line: this frame", (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(blank, "x axis: pre and post only", (8, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
    while len(panels) < COLS * ROWS:
        panels.append(blank.copy())
    rows = [np.hstack(panels[r * COLS : (r + 1) * COLS]) for r in range(ROWS)]
    grid = np.vstack(rows)
    banner = np.full((BANNER_H, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        banner,
        "F/F0   black dashed = manual   red = this alignment",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([banner, grid])


def frames_to_keep(phases: list[str]) -> list[int]:
    """Keep every pre/post frame, and only the first and last uncaging frame."""
    unc = [i for i, phase in enumerate(phases) if str(phase).strip().lower() in {"unc", "uncaging"}]
    drop = set(unc[1:-1])
    return [i for i in range(len(phases)) if i not in drop]


def export_set(arrays: SetArrays, out_dir: str) -> int:
    """Write one PNG per kept frame. Middle uncaging frames are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    shifts = view_shifts_for_set(arrays.before, arrays.applied, arrays.target)
    manual_ff0, method_ff0 = roi_ff0_traces(arrays.before, arrays.after, arrays.manual, shifts, arrays.phase)
    cy, cx = mask_centroid(arrays.manual[0])
    n_written = 0
    stem = f"{_safe(arrays.session)}_{_safe(arrays.group)}_set{arrays.set_label}"
    for frame in frames_to_keep(arrays.phase):
        images: dict[str, np.ndarray] = {}
        frame_masks: dict[str, np.ndarray] = {}
        for key, _label in PANEL_SPECS:
            view_shift = shifts[key][frame]
            images[key] = translate_image(arrays.before[frame], float(view_shift[0]), float(view_shift[1]))
            frame_masks[key] = map_manual_mask(arrays.manual[frame], view_shift, arrays.applied[frame])
        phase = arrays.phase[frame]
        title = f"{arrays.session}  {arrays.group}  set {arrays.set_label}  frame {frame}  {phase}"
        finite = images["raw"][np.isfinite(images["raw"])]
        if finite.size:
            vmin, vmax = (float(np.percentile(finite, 1)), float(np.percentile(finite, 99.5)))
        else:
            vmin, vmax = 0.0, 1.0
        canvas = render_alignment_grid(
            frame,
            images,
            arrays.manual[0],
            frame_masks,
            shifts,
            cy,
            cx,
            vmin,
            vmax,
            title,
        )
        panel_h = (canvas.shape[0] - BANNER_H) // ROWS
        plots = render_quant_grid(manual_ff0, method_ff0, arrays.phase, frame, panel_h, PANEL_W)
        canvas = np.hstack([canvas, plots])
        path = os.path.join(out_dir, f"{stem}_f{frame:03d}.png")
        cv2.imwrite(path, canvas)
        n_written += 1
    return n_written


def export_selected(out_dir: str, selected: list[tuple[str, str, int]] | None = None) -> list[str]:
    """Export the chosen sets into one folder. Returns the PNG paths' stems."""
    if selected is None:
        selected = SELECTED
    os.makedirs(out_dir, exist_ok=True)
    by_session: dict[str, list[tuple[str, int]]] = {}
    for session, group, set_label in selected:
        by_session.setdefault(session, []).append((group, set_label))
    written: list[str] = []
    lines = [
        "green outline: frame-0 manual ROI, fixed in every panel",
        "orange outline: this frame's manual ROI, mapped into that alignment",
        "yellow: the two outlines land on the same pixels",
        "dy dx: content shift applied to the raw frame (positive dy moves content down)",
        "uncaging frames: only the first and last of that phase are saved",
        "right-hand plots: F/F0 of mean ROI intensity. black dashed = manual ROI, red = that alignment",
        "plot x axis is pre and post only, stretched across the full width. markers are joined by lines",
        "",
    ]
    for session, picks in by_session.items():
        df_path = SESSIONS[session]
        combined = pd.read_pickle(df_path)
        for group, set_label in picks:
            print(f"loading {session} {group} set {set_label}", flush=True)
            arrays = load_set(combined, session, group, set_label)
            if arrays is None:
                print(f"  skip {group} set {set_label}", flush=True)
                lines.append(f"SKIP {session} {group} set {set_label}")
                continue
            n_frames = export_set(arrays, out_dir)
            stem = f"{_safe(session)}_{_safe(group)}_set{set_label}"
            written.append(stem)
            lines.append(
                f"{stem}  frames={n_frames}  "
                f"manual_span={_span(arrays.target + arrays.applied):.2f}  "
                f"applied_span={_span(arrays.applied):.2f}  "
                f"raw_spine_span={_span(arrays.target):.2f}"
            )
            print(f"  wrote {n_frames} frames", flush=True)
    with open(os.path.join(out_dir, "index.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return written


def _span(shift_yx: np.ndarray) -> float:
    dist = np.hypot(shift_yx[:, 0], shift_yx[:, 1])
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _set_keys(combined: pd.DataFrame) -> list[tuple[str, int]]:
    """Unique (group, set label) pairs with a non-negative set label."""
    labels = pd.to_numeric(combined["nth_set_label"], errors="coerce")
    keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for group, set_label in zip(combined["group"].astype(str), labels):
        if not np.isfinite(set_label) or int(set_label) < 0:
            continue
        key = (str(group), int(set_label))
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _mark_finished_stems(out_dir: str) -> None:
    """Treat an existing frame-0 PNG as a finished set from the first export."""
    for name in os.listdir(out_dir):
        if not name.endswith("_f000.png"):
            continue
        stem = name[: -len("_f000.png")]
        marker = os.path.join(out_dir, stem + ".complete")
        if not os.path.exists(marker):
            open(marker, "w", encoding="utf-8").close()


def export_remaining(out_dir: str = DEFAULT_OUT_DIR) -> list[str]:
    """Export every set in the three sessions that does not already have tiles.

    All alignment panels stay in the grid, including the ones that looked poor.
    """
    os.makedirs(out_dir, exist_ok=True)
    _mark_finished_stems(out_dir)
    index_path = os.path.join(out_dir, "index.txt")
    written: list[str] = []
    with open(index_path, "a", encoding="utf-8") as index:
        index.write("\nremaining sets\n")
        for session, df_path in SESSIONS.items():
            print(f"session {session}", flush=True)
            combined = pd.read_pickle(df_path)
            for group, set_label in _set_keys(combined):
                stem = f"{_safe(session)}_{_safe(group)}_set{set_label}"
                marker = os.path.join(out_dir, stem + ".complete")
                if os.path.exists(marker):
                    continue
                print(f"loading {session} {group} set {set_label}", flush=True)
                arrays = load_set(combined, session, group, set_label)
                if arrays is None:
                    print(f"  skip {group} set {set_label}", flush=True)
                    index.write(f"SKIP {stem}\n")
                    continue
                n_frames = export_set(arrays, out_dir)
                open(marker, "w", encoding="utf-8").close()
                written.append(stem)
                index.write(
                    f"{stem}  frames={n_frames}  "
                    f"manual_span={_span(arrays.target + arrays.applied):.2f}  "
                    f"applied_span={_span(arrays.applied):.2f}  "
                    f"raw_spine_span={_span(arrays.target):.2f}\n"
                )
                index.flush()
                print(f"  wrote {n_frames} frames", flush=True)
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Tile alignment methods and F/F0 plots")
    parser.add_argument("--all", action="store_true", help="Export every remaining set in the three sessions")
    args = parser.parse_args()
    if args.all:
        written = export_remaining(DEFAULT_OUT_DIR)
    else:
        written = export_selected(DEFAULT_OUT_DIR)
    print(f"sets {len(written)} -> {DEFAULT_OUT_DIR}")


if __name__ == "__main__":
    main()
