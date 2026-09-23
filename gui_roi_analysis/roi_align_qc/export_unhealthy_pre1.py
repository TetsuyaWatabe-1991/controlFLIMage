"""Save the first pre max projection for sheets labeled unhealthy.

Nothing else from the QC sheet is included. The spine ROI is drawn so the
field can be matched to the review sheet.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from qc_assignment_source import SESSION_PKLS, normalize_set_label  # noqa: E402
from score_qc_criteria import FEATURE_CSV, _mask_at, _pre_post, _projection  # noqa: E402

OUT_DIR = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260701/auto1/roi_align_qc/unhealthy_pre1"
SIDE = 512
SPINE_BGR = (0, 165, 255)
UNHEALTHY_CATEGORY = 8


def stretch_u8(image: np.ndarray) -> np.ndarray:
    """Map a finite image to 8-bit using the 1st and 99.5th percentiles."""
    values = np.asarray(image, dtype=float)
    finite = values[np.isfinite(values)]
    low, high = (np.percentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0))
    gray = np.clip((values - low) / max(float(high - low), 1e-6), 0, 1)
    gray[~np.isfinite(values)] = 0
    return (gray * 255).astype(np.uint8)


def render_pre_max(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """One square max projection with the spine outline."""
    gray = stretch_u8(image)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = cv2.resize(color, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
    if mask is not None:
        small = np.asarray(mask > 0, dtype=np.uint8)
        small = cv2.resize(small, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color, contours, -1, SPINE_BGR, 2)
    return color


def first_pre(info: pd.DataFrame, stack: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shown = _pre_post(info)
    pre = shown[shown["phase"].astype(str).str.lower().eq("pre")]
    if pre.empty:
        raise ValueError("no pre frame")
    row = pre.iloc[0]
    index = int(row["frame"]) if "frame" in row.index and pd.notna(row["frame"]) else int(row.name)
    return _projection(stack, index), _mask_at(mask, index)


def export_unhealthy(feature_csv: str = FEATURE_CSV, out_dir: str = OUT_DIR) -> list[str]:
    features = pd.read_csv(feature_csv)
    chosen = features[features["category"] == UNHEALTHY_CATEGORY]
    pooled = {session: pd.read_pickle(path) for session, path in SESSION_PKLS.items() if os.path.isfile(path)}
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for record in chosen.itertuples(index=False):
        table = pooled[str(record.session)]
        set_label = normalize_set_label(record.set_label)
        matched = table[
            (table["group"].astype(str) == str(record.group))
            & (table["nth_set_label"].map(normalize_set_label) == set_label)
        ]
        after = next(path for path in matched["after_align_full_save_path"] if isinstance(path, str) and path)
        base = os.path.splitext(after)[0]
        info = pd.read_csv(base + "_frame_info.csv")
        stack = np.asarray(tifffile.imread(after))
        mask = np.asarray(tifffile.imread(base + "_Spine_roi_mask.tif"))
        image, roi = first_pre(info, stack, mask)
        name = os.path.basename(str(record.image_rel))
        destination = os.path.join(out_dir, name)
        if not cv2.imwrite(destination, render_pre_max(image, roi)):
            raise OSError(destination)
        written.append(name)
    with open(os.path.join(out_dir, "index.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(written) + "\n")
    return written


def main() -> None:
    names = export_unhealthy()
    print(f"{len(names)} images")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
