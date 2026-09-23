"""Compare XYZ alignment on one highmag time series from 20260909.

The imaging script calls the same 3D phase correlation for the lowmag motor
move and for the first highmag for_align grab. Quantification then runs an
adjacent-frame phase correlation on the spine crop. This figure runs both,
plus a highpass version of each, on cnt_2_pos1__highmag_3_ set 0.

Green is the first pre volume. Magenta is the volume being aligned.
XY is the max over Z. YZ is the max over X, with Z stretched for display.

Usage:
    python export_highpass_xyz_overlay.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTROL = str(Path(__file__).resolve().parents[2])
for _path in (_THIS_DIR, _CONTROL):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from FLIMageAlignment import Align_4d_array, flim_files_to_nparray  # noqa: E402
from export_align_frames import DEFAULT_OUT_DIR  # noqa: E402
from highpass_xyz import (  # noqa: E402
    apply_registration_zyx,
    cumulative_adjacent,
    highpass_registration_zyx,
)

SESSION_DIR = "//RY-LAB-WS04/ImagingData/Tetsuya/20260909/auto1"
FRAME_INFO = (
    SESSION_DIR + "/tif/cnt_2_pos1__highmag_3__0.0_after_align_full_frame_info.csv"
)
CHANNEL_INDEX = 1  # ch_1or2 = 2 in the imaging script
CROP_Z = (5, 10)
CROP_Y = (32, 93)
CROP_X = (16, 77)
OUT_DIR = Path(os.path.dirname(DEFAULT_OUT_DIR)) / "highpass_xyz_align"
XY_SCALE = 3


def traditional_registration_zyx(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """3D phase-correlation shift used by the motor align (dz, dy, dx)."""
    shifts, _aligned = Align_4d_array(
        np.stack([reference, moving]),
        method="traditional",
        apply_shifts=False,
    )
    return np.asarray(shifts[1], dtype=np.float64)


def _crop(volume: np.ndarray) -> np.ndarray:
    z0, z1 = CROP_Z
    y0, y1 = CROP_Y
    x0, x1 = CROP_X
    return volume[z0:z1, y0:y1, x0:x1]


def load_series() -> tuple[list[np.ndarray], list[str], list[str]]:
    """Unique FLIM stacks in acquisition order, with phase labels."""
    info = pd.read_csv(FRAME_INFO)
    seen: list[str] = []
    phases: list[str] = []
    for name, phase in zip(info["filename"].astype(str), info["phase"].astype(str)):
        if name not in seen:
            seen.append(name)
            phases.append(phase)
    volumes = []
    kept_names: list[str] = []
    kept_phases: list[str] = []
    reference_shape: tuple[int, int, int] | None = None
    for name, phase in zip(seen, phases):
        path = os.path.join(SESSION_DIR, name)
        stack, _iminfo, _seconds = flim_files_to_nparray([path], ch=CHANNEL_INDEX)
        volume = np.asarray(stack[0], dtype=np.float32)
        if reference_shape is None:
            reference_shape = tuple(volume.shape)
        if tuple(volume.shape) != reference_shape:
            print(f"  skip {name} shape {volume.shape} (not a {reference_shape} Z stack)", flush=True)
            continue
        volumes.append(volume)
        kept_names.append(name)
        kept_phases.append(phase)
    return volumes, kept_names, kept_phases


def display_indices(phases: list[str]) -> list[tuple[str, int]]:
    """Pre last, post first, post middle, and post last, against frame 0."""
    pre = [i for i, phase in enumerate(phases) if phase.strip().lower() == "pre"]
    post = [i for i, phase in enumerate(phases) if phase.strip().lower() == "post"]
    if not pre or not post:
        raise RuntimeError(f"Need pre and post volumes, got {phases}")
    middle = post[(len(post) - 1) // 2]
    return [
        ("pre last", pre[-1]),
        ("post first", post[0]),
        ("post middle", middle),
        ("post last", post[-1]),
    ]


def _to_u8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    gray = np.clip((image.astype(np.float64) - low) / max(high - low, 1e-6), 0, 1)
    return (gray * 255).astype(np.uint8)


def _overlay(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """BGR image. Green is the reference. Magenta is the moving image."""
    finite = np.concatenate([reference.ravel(), moving.ravel()])
    finite = finite[np.isfinite(finite)]
    low, high = (np.percentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0))
    ref_u8 = _to_u8(reference, float(low), float(high))
    mov_u8 = _to_u8(moving, float(low), float(high))
    canvas = np.zeros(reference.shape + (3,), dtype=np.uint8)
    canvas[..., 0] = mov_u8
    canvas[..., 1] = ref_u8
    canvas[..., 2] = mov_u8
    return canvas


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def _projection_pair(reference: np.ndarray, moving: np.ndarray, width: int) -> np.ndarray:
    """XY on top, stretched YZ below, both width pixels wide."""
    xy = _overlay(np.max(reference, axis=0), np.max(moving, axis=0))
    xy = _resize(xy, width, width)
    yz_ref = np.repeat(np.max(reference, axis=2), XY_SCALE, axis=0)
    yz_mov = np.repeat(np.max(moving, axis=2), XY_SCALE, axis=0)
    yz = _overlay(yz_ref, yz_mov)
    yz = _resize(yz, width, width // 2)
    return np.vstack([xy, yz])


def _shift_text(shift_zyx: np.ndarray) -> str:
    dz, dy, dx = (float(v) for v in shift_zyx)
    return f"dz {dz:+.1f}  dy {dy:+.1f}  dx {dx:+.1f}"


def render_sheet(
    volumes: list[np.ndarray],
    names: list[str],
    phases: list[str],
    columns: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    """One block per displayed frame. columns are (label, shift_zyx per volume)."""
    marks = display_indices(phases)
    width = volumes[0].shape[-1] * XY_SCALE
    label_w = 220
    header_h = 32
    value_h = 28
    rows = []
    banner = (
        "green = first pre    magenta = this volume    "
        "top = XY max    bottom = YZ max, Z stretched"
    )
    for label, index in marks:
        header = np.zeros((header_h, label_w + width * len(columns), 3), dtype=np.uint8)
        cv2.putText(
            header,
            f"{label}   {phases[index]}   {names[index]}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cells = []
        for column, (_name, shifts) in enumerate(columns):
            moved = apply_registration_zyx(volumes[index], shifts[index])
            picture = _projection_pair(volumes[0], moved, width)
            bar = np.zeros((value_h, width, 3), dtype=np.uint8)
            cv2.putText(
                bar,
                _shift_text(shifts[index]),
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            title = np.zeros((header_h, width, 3), dtype=np.uint8)
            cv2.putText(
                title,
                columns[column][0],
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(np.vstack([title, picture, bar]))
        body = np.hstack(cells)
        gutter = np.zeros((body.shape[0], label_w, 3), dtype=np.uint8)
        cv2.putText(gutter, label, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(gutter, "XY", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(gutter, "YZ", (8, width + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([header, np.hstack([gutter, body])]))
    sheet = np.vstack(rows)
    top = np.zeros((36, sheet.shape[1], 3), dtype=np.uint8)
    cv2.putText(top, banner, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([top, sheet])


def build_columns(volumes: list[np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """Shifts for before, motor phase, highpass, adjacent phase, and highpass adjacent."""
    count = len(volumes)
    before = np.zeros((count, 3), dtype=np.float64)
    motor = np.zeros((count, 3), dtype=np.float64)
    highpass = np.zeros((count, 3), dtype=np.float64)
    for index in range(1, count):
        motor[index] = traditional_registration_zyx(volumes[0], volumes[index])
        highpass[index] = highpass_registration_zyx(volumes[0], volumes[index])
        print(f"  pair {index}: motor {motor[index]} highpass {highpass[index]}", flush=True)
    crops = [_crop(volume) for volume in volumes]
    adjacent = cumulative_adjacent(crops, traditional_registration_zyx)
    highpass_adj = cumulative_adjacent(crops, highpass_registration_zyx)
    return [
        ("before", before),
        ("motor 3D phase", motor),
        ("highpass XYZ", highpass),
        ("adjacent 3D phase", adjacent),
        ("highpass adjacent", highpass_adj),
    ]


def main() -> None:
    print("loading FLIM stacks", flush=True)
    volumes, names, phases = load_series()
    print(f"volumes {len(volumes)} shape {volumes[0].shape}", flush=True)
    columns = build_columns(volumes)
    sheet = render_sheet(volumes, names, phases, columns)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "auto1_20260909_cnt_2_pos1__highmag_3__set0.png"
    cv2.imwrite(str(path), sheet)
    lines = [
        "green = first pre, magenta = the volume being aligned",
        "motor 3D phase: full-volume phase correlation vs first pre.",
        "The imaging script uses this for both the lowmag motor move and the first highmag for_align.",
        "highpass XYZ: dy,dx from the highpass XY projection; dz from the highpass YZ projection.",
        "adjacent: phase correlation of each spine crop against the previous raw crop, summed back to the first pre.",
        "highpass adjacent: the same chain using highpass projections of that crop.",
        f"crop zyx {CROP_Z} {CROP_Y} {CROP_X}",
        "Uncaging file 017 is 33 single-plane frames, not a 15-slice stack, so it is left out of this XYZ comparison.",
        "",
    ]
    for name, shifts in columns:
        lines.append(name)
        for index, (filename, phase) in enumerate(zip(names, phases)):
            lines.append(f"  {index} {phase} {filename} {_shift_text(shifts[index])}")
    (OUT_DIR / "shifts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
