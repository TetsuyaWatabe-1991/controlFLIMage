# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:44:15 2026

@author: WatabeT
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.append(r"..\..")
import numpy as np
import pandas as pd
from typing import Any


def build_group_header_combined(group_header_dict):
    """Merge dict entries that share the same display name (each_header_name)."""
    combined = defaultdict(list)
    for each_header, each_header_name in group_header_dict.items():
        combined[each_header_name].append(each_header)
    return dict(combined)


def df_matches_group_headers(df, header_list):
    mask = pd.Series(False, index=df.index)
    for each_header in header_list:
        mask |= df["group"].str.contains(each_header, na=False)
    return df[mask]


def group_header_file_tag(header_list):
    return "_".join(header_list)

def add_uncaging_label_between_ylabel_and_axis(
    ax: Any,
    power_mw: float,
    fig: Any,
    gap_points: float = 4.0,
) -> None:
    """Place uncaging power label between y-axis label and axis, with fixed gap.

    This function computes positions in display coordinates based on the
    rendered y-axis label and axes bounding boxes, then converts the
    x-position back into axes fraction coordinates.
    """
    if power_mw is None:
        return

    renderer = fig.canvas.get_renderer()
    ylabel_text = ax.yaxis.get_label()
    if not ylabel_text.get_text():
        return

    label_bbox = ylabel_text.get_window_extent(renderer=renderer)
    axes_bbox = ax.get_window_extent(renderer=renderer)

    # Right edge of ylabel in display coords, add small gap to the right
    x_display = label_bbox.x1 + gap_points
    # Vertical center of axes in display coords
    y_display = axes_bbox.y0 + 0.5 * axes_bbox.height

    # Convert display coordinates back to axes fraction for x
    inv = ax.transAxes.inverted()
    x_axes, _ = inv.transform((x_display, y_display))

    ax.text(
        x_axes,
        0.5,
        f"{power_mw} mW",
        transform=ax.transAxes,
        ha="left",
        va="center",
    )



def reshape_axes_to_2d(axes: Any, n_rows: int, n_cols: int) -> np.ndarray:
    """Reshape matplotlib `axes` into a stable (n_rows, n_cols) array.

    `plt.subplots(n_rows, n_cols)` returns different shapes depending on whether
    `n_rows` or `n_cols` equals 1. This helper prevents indexing errors.
    """

    axes_arr = np.array(axes, dtype=object)

    # n_rows == 1 and n_cols == 1: single Axes object (0-dim array).
    if axes_arr.ndim == 0:
        return np.array([[axes]], dtype=object)

    # One of (n_rows, n_cols) equals 1: plt returns a 1-d array.
    if axes_arr.ndim == 1:
        if n_rows == 1 and n_cols > 1:
            return axes_arr.reshape(1, n_cols)
        if n_cols == 1 and n_rows > 1:
            return axes_arr.reshape(n_rows, 1)
        return axes_arr.reshape(n_rows, n_cols)

    # Both n_rows and n_cols > 1: already 2-d.
    if axes_arr.ndim == 2:
        if axes_arr.shape != (n_rows, n_cols):
            return axes_arr.reshape(n_rows, n_cols)
        return axes_arr

    raise ValueError(f"Unexpected axes array shape: {axes_arr.shape}")


def gcamp_windows_json_path(df_save_path: str) -> str:
    """Sidecar written next to a combined_df pickle by the ROI analysis script."""
    stem, _ext = os.path.splitext(df_save_path)
    return stem + "_gcamp_windows.json"


def save_gcamp_windows(df_save_path: str, windows: dict) -> str:
    """Write per-length GCaMP windows beside the combined_df pickle.

    Each value is a GCaMPWindow or a dict with baseline, response, and time_zero.
    Ranges are 0-based and stop-exclusive. An empty dict clears a previous file.
    """
    payload: dict[str, dict] = {}
    for n_frames, window in windows.items():
        if hasattr(window, "baseline"):
            record = {
                "baseline": [int(window.baseline[0]), int(window.baseline[1])],
                "response": [int(window.response[0]), int(window.response[1])],
                "time_zero": None if window.time_zero is None else int(window.time_zero),
            }
        else:
            record = {
                "baseline": [int(window["baseline"][0]), int(window["baseline"][1])],
                "response": [int(window["response"][0]), int(window["response"][1])],
                "time_zero": None if window.get("time_zero") is None else int(window["time_zero"]),
            }
        payload[str(int(n_frames))] = record
    path = gcamp_windows_json_path(df_save_path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def load_gcamp_windows(df_save_path: str) -> dict[int, dict]:
    """Load the GCaMP window sidecar. Missing file returns an empty dict."""
    path = gcamp_windows_json_path(df_save_path)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    loaded: dict[int, dict] = {}
    for key, record in payload.items():
        loaded[int(key)] = {
            "baseline": (int(record["baseline"][0]), int(record["baseline"][1])),
            "response": (int(record["response"][0]), int(record["response"][1])),
            "time_zero": None if record.get("time_zero") is None else int(record["time_zero"]),
        }
    return loaded


def format_respan_path_assignments(df_save_path: str, out_csv_path: str) -> str:
    """Return copy-paste Python assignments for LTP analysis scripts.

    ROI analysis prints this block; paste it as-is into the experiment script.
    """
    return (
        f'df_save_path_1 = r"{df_save_path}"\n'
        f'out_csv_path = r"{out_csv_path}"'
    )


def select_ltp_post_frames(
    post_df: pd.DataFrame,
    *,
    time_col: str = "aligned_time_sec",
    window_min: list[float] | tuple[float, float] = (25.0, 35.0),
    pad_sec: float = 60.0,
) -> tuple[pd.DataFrame, str]:
    """Select post frames for LTP quantification around ``window_min``.

    Uses an inclusive window plus ``pad_sec`` so a point a few seconds outside
    25-35 min is still used. If nothing falls in that window, return the single
    frame closest to the window center. Do not average all later times (that
    mixes ~80 min points into the 30 min LTP metric).
    """
    if post_df is None or len(post_df) == 0:
        empty = post_df if post_df is not None else pd.DataFrame()
        return empty, "none"
    t0 = float(window_min[0]) * 60.0
    t1 = float(window_min[1]) * 60.0
    t = post_df[time_col].astype(float)
    in_win = post_df[(t >= (t0 - pad_sec)) & (t <= (t1 + pad_sec))]
    if len(in_win) > 0:
        return in_win, "window"
    center = 0.5 * (t0 + t1)
    nearest_idx = (t - center).abs().idxmin()
    return post_df.loc[[nearest_idx]], "nearest"

