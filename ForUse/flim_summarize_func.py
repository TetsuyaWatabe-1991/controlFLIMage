# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:44:15 2026

@author: WatabeT
"""
import sys
sys.path.append(r"..\..")
import numpy as np
from typing import Any

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

