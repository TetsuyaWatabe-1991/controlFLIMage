"""Original RESPAN assignment image for one quantified set.

The left side of the review window is one max projection, the same pixel size
as the large first-pre panel on the QC sheet, with that spine ROI and the
other spines in the field.
"""

from __future__ import annotations

import configparser
import os
import sys

import cv2
import numpy as np

_CONTROL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ASI = os.path.normpath(os.path.join(_CONTROL, "..", "ongoing", "ASIcontroller"))
for _path in (_CONTROL, _ASI):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from FLIMageAlignment import flim_files_to_nparray  # noqa: E402
from respan_uncaging_log import parse_uncaging_records  # noqa: E402

SESSION_ROOTS = {
    "20260701": r"G:/ImagingData/Tetsuya/20260701/auto1",
    "20260623": r"G:/ImagingData/Tetsuya/20260623/auto1",
    "20260909": r"//RY-LAB-WS04/ImagingData/Tetsuya/20260909/auto1",
}
SESSION_PKLS = {
    session: os.path.join(root, "combined_df_respan.pkl") for session, root in SESSION_ROOTS.items()
}

SELECTED_BGR = (0, 165, 255)
OTHER_BGR = (255, 220, 80)
# Matches the already-exported QC sheets: banner, caption bar, and the large pre square.
SHEET_BANNER_H = 36
SHEET_CAPTION_H = 18
PRE_SIDE = 512


def normalize_set_label(value) -> str:
    """Index and pickle both store set labels as one decimal, such as 1.0."""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value).strip()


def flim_sequence_number(path: str) -> int | None:
    stem = os.path.splitext(os.path.basename(path))[0]
    suffix = stem.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def highmag_folder_name(group: str) -> str:
    name = str(group).strip()
    if name.endswith("_"):
        name = name[:-1]
    return name


def match_uncaging_record(unc_flim_path: str, records: list) -> object | None:
    """Pick the log entry whose FLIM is the same acquisition or just before it."""
    unc_number = flim_sequence_number(unc_flim_path)
    if unc_number is None:
        return None
    best = None
    best_delta = 99
    for record in records:
        number = flim_sequence_number(record.flim_path)
        if number is None:
            continue
        delta = unc_number - number
        if 0 <= delta <= 3 and delta < best_delta:
            best = record
            best_delta = delta
    return best


def read_ini_zyx(path: str) -> tuple[int, int, int] | None:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    try:
        section = parser["uncaging_settings"]
        return int(float(section["spine_z"])), int(float(section["spine_y"])), int(float(section["spine_x"]))
    except (KeyError, ValueError):
        return None


def first_assignment_flim(auto_root: str, highmag_name: str) -> str | None:
    matches: list[str] = []
    folder = os.path.join(auto_root, highmag_name)
    for directory in (folder, auto_root):
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not name.startswith(highmag_name + "_") or not name.endswith(".flim"):
                continue
            if "for_align" in name:
                continue
            matches.append(os.path.join(directory, name))
    matches = sorted(set(matches), key=lambda path: flim_sequence_number(path) or 10**9)
    return matches[0] if matches else None


def _outline_path(folder: str, stem: str) -> str | None:
    path = os.path.join(folder, "seg_masks", f"{stem}_spine_outline_mask.tif")
    if os.path.isfile(path):
        return path
    return None


def load_uncaging_file_by_set(pkl_path: str) -> dict[tuple[str, str], str]:
    """Map (group, set label) to the uncaging FLIM path."""
    import pandas as pd

    frame = pd.read_pickle(pkl_path)
    found: dict[tuple[str, str], str] = {}
    phase = frame["phase"].astype(str).str.lower()
    unc = frame[phase.isin(["unc", "uncaging"])]
    for _, row in unc.iterrows():
        path = row.get("file_path")
        if not isinstance(path, str) or not path:
            continue
        found[(str(row["group"]), normalize_set_label(row["nth_set_label"]))] = path
    return found


def render_assignment_panel(
    volume_zyx: np.ndarray,
    selected_mask: np.ndarray | None,
    other_masks: list[np.ndarray],
    head_z: int | None,
    spine_name: str,
) -> np.ndarray:
    """One max projection at the same square size as the sheet's first pre."""
    volume = np.asarray(volume_zyx, dtype=np.float32)
    if volume.ndim == 2:
        volume = volume[None, :, :]
    image = volume.max(axis=0)
    finite = image[np.isfinite(image)]
    low, high = (np.percentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0))
    label = spine_name
    if head_z is not None:
        label = f"{spine_name}  head Z {int(head_z)}"
    return _draw_panel(image, low, high, selected_mask, other_masks, label, True)


def _draw_panel(
    image: np.ndarray,
    low: float,
    high: float,
    selected_mask: np.ndarray | None,
    other_masks: list[np.ndarray],
    label: str,
    emphasize: bool,
) -> np.ndarray:
    gray = np.clip((image.astype(np.float64) - low) / max(high - low, 1e-6), 0, 1)
    canvas = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(canvas, (PRE_SIDE, PRE_SIDE), interpolation=cv2.INTER_NEAREST)
    for mask in other_masks:
        _draw_outline(canvas, mask, OTHER_BGR, 1)
    if selected_mask is not None:
        _draw_outline(canvas, selected_mask, SELECTED_BGR, 2 if emphasize else 1)
    bar = np.full((SHEET_CAPTION_H, PRE_SIDE, 3), 255, np.uint8)
    cv2.putText(bar, label[:64], (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def _draw_outline(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    if binary.shape[:2] != canvas.shape[:2]:
        binary = cv2.resize(binary, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, thickness)


class AssignmentLookup:
    """Cache pickle rows and FLIM volumes while the review window is open."""

    def __init__(self) -> None:
        self._unc_files: dict[str, dict[tuple[str, str], str]] = {}
        self._volumes: dict[str, np.ndarray] = {}

    def column_for(self, session: str, group: str, set_label: str) -> tuple[np.ndarray | None, str]:
        """Return the assignment max projection and a short status line."""
        root = SESSION_ROOTS.get(session)
        pkl = SESSION_PKLS.get(session)
        if root is None or pkl is None or not os.path.isfile(pkl):
            return None, "assignment session unknown"
        folder_name = highmag_folder_name(group)
        folder = os.path.join(root, folder_name)
        if not os.path.isdir(folder):
            return None, "assignment folder missing"
        if session not in self._unc_files:
            self._unc_files[session] = load_uncaging_file_by_set(pkl)
        unc_path = self._unc_files[session].get((str(group), normalize_set_label(set_label)), "")
        records = parse_uncaging_records(folder)
        record = match_uncaging_record(unc_path, records) if unc_path else None
        selected = None
        others: list[np.ndarray] = []
        head_z = None
        spine_name = "unmatched spine"
        selected_stem = record.spine_stem if record is not None else ""
        if record is not None:
            spine_name = record.spine_stem
            zyx = read_ini_zyx(record.ini_path)
            if zyx is not None:
                head_z = zyx[0]
            selected_path = _outline_path(folder, record.spine_stem)
            if selected_path:
                import tifffile

                selected = tifffile.imread(selected_path) > 0
        seg_dir = os.path.join(folder, "seg_masks")
        if os.path.isdir(seg_dir):
            import tifffile

            for name in sorted(os.listdir(seg_dir)):
                if not name.endswith("_spine_outline_mask.tif"):
                    continue
                if selected_stem and name.startswith(selected_stem + "_"):
                    continue
                others.append(tifffile.imread(os.path.join(seg_dir, name)) > 0)
        flim_path = first_assignment_flim(root, folder_name)
        if flim_path is None:
            return None, "assignment FLIM missing"
        if flim_path not in self._volumes:
            stack, _info, _times = flim_files_to_nparray([flim_path], ch=1)
            volume = np.asarray(stack[0])
            self._volumes[flim_path] = volume if volume.ndim == 3 else volume[None, ...]
        image = render_assignment_panel(self._volumes[flim_path], selected, others, head_z, spine_name)
        return image, spine_name
