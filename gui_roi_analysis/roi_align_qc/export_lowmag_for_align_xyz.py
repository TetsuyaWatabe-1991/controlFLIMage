"""XYZ overlay of the low-resolution grabs used to motor-correct cnt imaging.

Lowmag: each reduced-resolution lowmag (128) against the original lowmag (256).
for_align: each reduced highmag grab (64) against the first full highmag (128).

Green is the reference. Magenta is the grab. Columns are before, the live
motor method (3D phase correlation), and highpass XYZ. Both volumes are
resized to the shared smaller shape before the shift is measured, matching
align_two_flimfile_different_resolution.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.transform import resize

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from FLIMageAlignment import flim_files_to_nparray  # noqa: E402
from gui_roi_analysis.roi_align_qc.export_highpass_xyz_overlay import (  # noqa: E402
    _overlay,
    _resize,
    _shift_text,
    traditional_registration_zyx,
)
from gui_roi_analysis.roi_align_qc.highpass_xyz import (  # noqa: E402
    apply_registration_zyx,
    highpass_registration_zyx,
)

SESSION = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260909/auto1"
OUT_DIR = (
    r"G:\ImagingData\Tetsuya\20260701\auto1\roi_align_qc"
    r"\highpass_xyz_align\lowmag_for_align"
)
CH = 1
PANEL_WIDTH = 256

_FOR_ALIGN = re.compile(
    r"^for_align_(cnt_\d+_pos1__highmag_\d+)_(\d+)\.flim$"
)
_LOWMAG = re.compile(r"^(cnt_\d+_pos1)_(\d+)\.flim$")


@dataclass(frozen=True)
class Pair:
    """One reference volume and one reduced-resolution grab."""

    kind: str
    reference_name: str
    moving_name: str

    @property
    def stem(self) -> str:
        return os.path.splitext(self.moving_name)[0]


def list_pairs(folder: str) -> list[Pair]:
    """Pair each reduced cnt grab with the reference the live script uses."""
    names = [name for name in os.listdir(folder) if name.endswith(".flim")]
    pairs: list[Pair] = []

    low_by_stem: dict[str, list[tuple[int, str]]] = {}
    for name in names:
        match = _LOWMAG.match(name)
        if match is None:
            continue
        low_by_stem.setdefault(match.group(1), []).append((int(match.group(2)), name))
    for _stem, numbered in sorted(low_by_stem.items()):
        numbered.sort()
        reference = numbered[0][1]
        for _index, name in numbered[1:]:
            pairs.append(Pair("lowmag", reference, name))

    highmag_first: dict[str, str] = {}
    for name in sorted(names):
        match = re.match(r"^(cnt_\d+_pos1__highmag_\d+)_(\d+)\.flim$", name)
        if match is None:
            continue
        highmag_first.setdefault(match.group(1), name)
    for name in sorted(names):
        match = _FOR_ALIGN.match(name)
        if match is None:
            continue
        reference = highmag_first.get(match.group(1))
        if reference is None:
            continue
        pairs.append(Pair("for_align", reference, name))
    return pairs


def _load(path: str) -> np.ndarray:
    stack, _info, _ = flim_files_to_nparray([path], ch=CH)
    volume = np.asarray(stack[0], dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"{os.path.basename(path)} shape {volume.shape}")
    return volume


def to_common_shape(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize both volumes to the shared minimum ZYX, as the live aligner does."""
    shape = tuple(min(a, b) for a, b in zip(reference.shape, moving.shape))

    def _one(volume: np.ndarray) -> np.ndarray:
        if volume.shape == shape:
            return volume.astype(np.float32, copy=False)
        return resize(
            volume, shape, preserve_range=True, anti_aliasing=True
        ).astype(np.float32)

    return _one(reference), _one(moving)


def _projection_pair(reference: np.ndarray, moving: np.ndarray, width: int) -> np.ndarray:
    xy = _resize(_overlay(np.max(reference, axis=0), np.max(moving, axis=0)), width, width)
    z_repeat = max(1, width // max(reference.shape[0] * 4, 1))
    yz = _overlay(
        np.repeat(np.max(reference, axis=2), z_repeat, axis=0),
        np.repeat(np.max(moving, axis=2), z_repeat, axis=0),
    )
    yz = _resize(yz, width, max(width // 2, 1))
    return np.vstack([xy, yz])


def render_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    motor_shift: np.ndarray,
    highpass_shift: np.ndarray,
    title: str,
) -> np.ndarray:
    """Three columns: before, motor 3D phase, highpass XYZ."""
    views = [
        ("before", moving),
        ("motor 3D phase", apply_registration_zyx(moving, motor_shift)),
        ("highpass XYZ", apply_registration_zyx(moving, highpass_shift)),
    ]
    columns = [_projection_pair(reference, image, PANEL_WIDTH) for _label, image in views]
    body = np.hstack(columns)
    banner_h = 56
    banner = np.full((banner_h, body.shape[1], 3), 255, np.uint8)
    cv2.putText(banner, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for index, ((label, _image), shift) in enumerate(zip(views, [np.zeros(3), motor_shift, highpass_shift])):
        x = index * PANEL_WIDTH + 8
        cv2.putText(banner, label, (x, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        text = "no shift" if index == 0 else _shift_text(shift)
        cv2.putText(banner, text, (x, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([banner, body])


def export_pairs(folder: str, out_dir: str, pairs: list[Pair] | None = None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    pairs = list_pairs(folder) if pairs is None else pairs
    cache: dict[str, np.ndarray] = {}
    lines = ["kind\tmoving\treference\tshape\tmotor_dz\tmotor_dy\tmotor_dx\thp_dz\thp_dy\thp_dx"]
    written = 0
    for pair in pairs:
        out_path = os.path.join(out_dir, f"{pair.stem}.png")
        if os.path.exists(out_path):
            written += 1
            continue
        try:
            if pair.reference_name not in cache:
                cache[pair.reference_name] = _load(os.path.join(folder, pair.reference_name))
            moving_raw = _load(os.path.join(folder, pair.moving_name))
            reference, moving = to_common_shape(cache[pair.reference_name], moving_raw)
        except (OSError, ValueError) as exc:
            lines.append(f"FAIL\t{pair.moving_name}\t{exc}")
            print("FAIL", pair.moving_name, exc, flush=True)
            continue
        reference_raw = cache[pair.reference_name]
        if pair.kind == "lowmag" and moving_raw.shape[-1] >= reference_raw.shape[-1]:
            lines.append(f"SKIP\t{pair.moving_name}\tnot reduced\t{tuple(moving_raw.shape)}")
            print("SKIP", pair.moving_name, moving_raw.shape, flush=True)
            continue
        motor = traditional_registration_zyx(reference, moving)
        highpass = highpass_registration_zyx(reference, moving)
        title = f"{pair.kind}  {pair.moving_name}  vs  {pair.reference_name}"
        canvas = render_pair(reference, moving, motor, highpass, title)
        cv2.imwrite(out_path, canvas)
        shape = "x".join(str(v) for v in reference.shape)
        lines.append(
            f"{pair.kind}\t{pair.moving_name}\t{pair.reference_name}\t{shape}\t"
            f"{motor[0]:.3f}\t{motor[1]:.3f}\t{motor[2]:.3f}\t"
            f"{highpass[0]:.3f}\t{highpass[1]:.3f}\t{highpass[2]:.3f}"
        )
        written += 1
        print(f"{written}/{len(pairs)} {pair.moving_name} motor {motor} highpass {highpass}", flush=True)
    with open(os.path.join(out_dir, "shifts.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return written


def main() -> None:
    count = export_pairs(SESSION, OUT_DIR)
    print(f"wrote {count} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
