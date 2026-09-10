"""
Utilities for uncaging ROI keyframe selection and position interpolation.

When uncaging_roi_keyframe_count < n_unc, ROI editing is limited to representative
uncaging stack frames; intermediate frames receive translated ROIs (shape unchanged).
"""

from __future__ import annotations

import copy
from typing import Any


def select_uncaging_keyframe_stack_indices(
    n_pre: int,
    n_unc: int,
    keyframe_count: int | None,
) -> list[int]:
    """
    Return stack frame indices (0-based) for uncaging ROI editing.

    Args:
        n_pre: Number of pre frames in the stack.
        n_unc: Number of uncaging frames in the stack.
        keyframe_count: Number of representative uncaging frames to edit.
            None or >= n_unc -> all uncaging frames.

    Returns:
        Sorted list of stack indices in the uncaging phase.
    """
    if n_unc <= 0:
        return []

    all_unc = list(range(n_pre, n_pre + n_unc))
    if keyframe_count is None or keyframe_count <= 0 or keyframe_count >= n_unc:
        return all_unc

    if keyframe_count == 1:
        local_indices = [0]
    elif keyframe_count == 2:
        local_indices = [0, n_unc - 1]
    else:
        local_indices = [
            int(round(i * (n_unc - 1) / (keyframe_count - 1)))
            for i in range(keyframe_count)
        ]

    stack_indices = sorted({n_pre + li for li in local_indices})
    return stack_indices


def build_navigation_stack_indices(
    n_pre: int,
    n_unc: int,
    n_post: int,
    keyframe_count: int | None,
) -> list[int]:
    """
    Return stack frame indices exposed in the ROI GUI seek bar.

    In keyframe mode, uncaging contributes only representative frames (not every
    uncaging stack frame). Pre and post segments are unchanged.
    """
    pre_indices = list(range(n_pre))
    post_start = n_pre + n_unc
    post_indices = list(range(post_start, post_start + n_post))

    if is_uncaging_keyframe_mode(keyframe_count, n_unc):
        unc_indices = select_uncaging_keyframe_stack_indices(
            n_pre, n_unc, keyframe_count
        )
    else:
        unc_indices = list(range(n_pre, n_pre + n_unc))

    return pre_indices + unc_indices + post_indices


def stack_index_to_navigation_index(
    stack_idx: int,
    navigation_stack_indices: list[int],
) -> int:
    """Map a stack frame index to its navigation slider position."""
    try:
        return navigation_stack_indices.index(stack_idx)
    except ValueError:
        if not navigation_stack_indices:
            return 0
        if stack_idx <= navigation_stack_indices[0]:
            return 0
        return len(navigation_stack_indices) - 1


def is_uncaging_keyframe_mode(
    keyframe_count: int | None,
    n_unc: int,
) -> bool:
    """Return True when keyframe-limited uncaging ROI editing is active."""
    return (
        keyframe_count is not None
        and n_unc > 0
        and 0 < keyframe_count < n_unc
    )


def is_uncaging_stack_frame(frame_idx: int, n_pre: int, n_unc: int) -> bool:
    """Return True if frame_idx falls in the uncaging segment of the stack."""
    if n_unc <= 0:
        return False
    return n_pre <= frame_idx < n_pre + n_unc


def roi_centroid(params: dict[str, Any], roi_shape: str) -> tuple[float, float]:
    """Compute ROI centroid (x, y) from roi_parameters."""
    if not params:
        return 0.0, 0.0

    if roi_shape == "rectangle":
        x = float(params.get("x", 0))
        y = float(params.get("y", 0))
        w = float(params.get("width", 0))
        h = float(params.get("height", 0))
        return x + w / 2.0, y + h / 2.0

    if roi_shape == "ellipse":
        return (
            float(params.get("center_x", 0)),
            float(params.get("center_y", 0)),
        )

    if roi_shape == "polygon" and "points" in params:
        pts = params["points"]
        if not pts:
            return 0.0, 0.0
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    return 0.0, 0.0


def translate_roi_params(
    params: dict[str, Any],
    roi_shape: str,
    dx: float,
    dy: float,
) -> dict[str, Any]:
    """Return a copy of params shifted by (dx, dy); size/shape unchanged."""
    out = copy.deepcopy(params)

    if roi_shape == "rectangle":
        out["x"] = float(out.get("x", 0)) + dx
        out["y"] = float(out.get("y", 0)) + dy
    elif roi_shape == "ellipse":
        out["center_x"] = float(out.get("center_x", 0)) + dx
        out["center_y"] = float(out.get("center_y", 0)) + dy
    elif roi_shape == "polygon" and "points" in out:
        out["points"] = [
            (float(px) + dx, float(py) + dy)
            for px, py in out["points"]
        ]

    return out


def interpolate_roi_params_for_frame(
    frame_idx: int,
    keyframe_indices: list[int],
    frame_roi_parameters: dict[int, dict[str, Any]],
    roi_shape: str,
) -> dict[str, Any] | None:
    """
    Interpolate ROI position for frame_idx from keyframe definitions.

    Uses linear centroid interpolation between bracketing keyframes and translates
    the left keyframe's shape to the interpolated centroid.
    """
    if not keyframe_indices:
        return None

    if frame_idx in frame_roi_parameters:
        return copy.deepcopy(frame_roi_parameters[frame_idx])

    defined_keyframes = sorted(
        idx for idx in keyframe_indices if idx in frame_roi_parameters
    )
    if not defined_keyframes:
        return None

    sorted_keys = defined_keyframes
    if frame_idx <= sorted_keys[0]:
        if sorted_keys[0] in frame_roi_parameters:
            return copy.deepcopy(frame_roi_parameters[sorted_keys[0]])
        return None
    if frame_idx >= sorted_keys[-1]:
        if sorted_keys[-1] in frame_roi_parameters:
            return copy.deepcopy(frame_roi_parameters[sorted_keys[-1]])
        return None

    left_idx = sorted_keys[0]
    right_idx = sorted_keys[-1]
    for i in range(len(sorted_keys) - 1):
        if sorted_keys[i] <= frame_idx <= sorted_keys[i + 1]:
            left_idx = sorted_keys[i]
            right_idx = sorted_keys[i + 1]
            break

    if left_idx not in frame_roi_parameters or right_idx not in frame_roi_parameters:
        return None

    params_left = frame_roi_parameters[left_idx]
    params_right = frame_roi_parameters[right_idx]
    cx_left, cy_left = roi_centroid(params_left, roi_shape)
    cx_right, cy_right = roi_centroid(params_right, roi_shape)

    if right_idx == left_idx:
        t = 0.0
    else:
        t = (frame_idx - left_idx) / float(right_idx - left_idx)

    cx = (1.0 - t) * cx_left + t * cx_right
    cy = (1.0 - t) * cy_left + t * cy_right
    dx = cx - cx_left
    dy = cy - cy_left
    return translate_roi_params(params_left, roi_shape, dx, dy)


def expand_uncaging_roi_keyframes(
    frame_roi_parameters: dict[int, dict[str, Any]],
    n_pre: int,
    n_unc: int,
    keyframe_count: int | None,
    roi_shape: str,
) -> dict[int, dict[str, Any]]:
    """
    Expand keyframe ROI definitions to all uncaging stack frames via interpolation.

    Non-uncaging entries in frame_roi_parameters are preserved unchanged.
    """
    if not is_uncaging_keyframe_mode(keyframe_count, n_unc):
        return frame_roi_parameters

    keyframe_indices = select_uncaging_keyframe_stack_indices(
        n_pre, n_unc, keyframe_count
    )
    out = copy.deepcopy(frame_roi_parameters)

    for frame_idx in range(n_pre, n_pre + n_unc):
        if frame_idx in keyframe_indices:
            continue
        interpolated = interpolate_roi_params_for_frame(
            frame_idx, keyframe_indices, out, roi_shape
        )
        if interpolated is not None:
            out[frame_idx] = interpolated

    return out
