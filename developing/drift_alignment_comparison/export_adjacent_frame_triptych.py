# -*- coding: utf-8 -*-
"""
Export per-frame triptych PNGs: Raw | Baseline fourier | ROI adjacent cumulative (E).

Outputs (per FLIM series):
  adjacent_triptych/zproj/frame_XXX.png
  adjacent_triptych/roi_trim/frame_XXX.png

Each PNG is 3 panels in one row. Short side of the saved image is SHORT_SIDE_PX.

Run:
  C:\\Users\\yasudalab\\AppData\\Local\\anaconda3\\python.exe export_adjacent_frame_triptych.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _SCRIPT_DIR.parents[1]
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from FLIMageAlignment import (  # noqa: E402
    flim_files_to_nparray,
    get_flimfile_list,
)
from align_methods import (  # noqa: E402
    RoiCropSpec,
    _align_series,
    align_traditional,
    crop_zyx,
    estimate_shift_3d,
)
from uncaging_utils import uncaging_info_from_series  # noqa: E402

FLIM_PATHS = [
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_1_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_2_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_3_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_4_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_5_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_6_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_7_002.flim",
]
CHANNEL_1OR2 = 2
SMALL_REGION_SIZE = 60
SMALL_Z_PLUS_MINUS = 2
SHORT_SIDE_PX = 500
PANEL_LABELS = ("00_raw", "A_baseline_fourier", "E_roi_adjacent")
VMAX_PERCENTILE = 99.5


def align_roi_adjacent(stack: np.ndarray, roi_spec: RoiCropSpec):
    """E: ROI-cropped adjacent-frame cumulative alignment."""

    def estimator(ref, query):
        return estimate_shift_3d(ref, query, upsample_factor=10)

    return _align_series(
        stack,
        estimator,
        reference="adjacent",
        shift_mode="constant",
        roi_spec=roi_spec,
    )


def _resize_short_side(png_bytes: bytes, short_side_px: int) -> Image.Image:
    """Resize image so min(width, height) == short_side_px."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    scale = short_side_px / min(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def save_triptych_png(
    path: Path,
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    titles: tuple[str, str, str],
    vmax: float,
    *,
    short_side_px: int = SHORT_SIDE_PX,
) -> None:
    """Save one row of three grayscale panels; output short side = short_side_px."""
    n_cols = 3
    dpi = 100
    fig_w_in = (short_side_px * n_cols) / dpi
    fig_h_in = short_side_px / dpi

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w_in, fig_h_in + 0.6))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_axis_off()

    plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.06)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)

    out_img = _resize_short_side(buf.getvalue(), short_side_px)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(path)


def process_one_flim(flim_path: str) -> None:
    ch = CHANNEL_1OR2 - 1
    stem = Path(flim_path).stem
    out_root = _SCRIPT_DIR / "output" / stem / "adjacent_triptych"
    zproj_dir = out_root / "zproj"
    roi_dir = out_root / "roi_trim"

    filelist = get_flimfile_list(flim_path)
    _, roi_spec = uncaging_info_from_series(
        filelist,
        ch,
        half_z=SMALL_Z_PLUS_MINUS,
        half_y=SMALL_REGION_SIZE // 2,
        half_x=SMALL_REGION_SIZE // 2,
    )
    if roi_spec is None:
        print(f"Skip {stem}: no uncaging ROI")
        return

    stack, _, _ = flim_files_to_nparray(filelist, ch=ch)
    _, aligned_baseline = align_traditional(stack)
    _, aligned_adjacent = align_roi_adjacent(stack, roi_spec)
    _, roi_bounds = crop_zyx(stack[0], roi_spec)
    z0, z1, y0, y1, x0, x1 = roi_bounds

    zproj_vmax = float(np.percentile(np.max(stack[0], axis=0), VMAX_PERCENTILE))
    roi_vmax = float(
        np.percentile(np.max(stack[0, z0:z1, y0:y1, x0:x1], axis=0), VMAX_PERCENTILE)
    )

    print(f"\n{stem}: {stack.shape[0]} frames -> {out_root}")

    for t in range(stack.shape[0]):
        tag = f"frame_{t:03d}"
        raw_vol = stack[t]
        base_vol = aligned_baseline[t]
        adj_vol = aligned_adjacent[t]

        zproj_triplet = (
            np.max(raw_vol, axis=0),
            np.max(base_vol, axis=0),
            np.max(adj_vol, axis=0),
        )
        roi_triplet = (
            np.max(raw_vol[z0:z1, y0:y1, x0:x1], axis=0),
            np.max(base_vol[z0:z1, y0:y1, x0:x1], axis=0),
            np.max(adj_vol[z0:z1, y0:y1, x0:x1], axis=0),
        )

        titles = tuple(f"{lab}\n{tag}" for lab in PANEL_LABELS)
        save_triptych_png(
            zproj_dir / f"{tag}.png",
            zproj_triplet,
            titles,
            zproj_vmax,
        )
        save_triptych_png(
            roi_dir / f"{tag}.png",
            roi_triplet,
            titles,
            roi_vmax,
        )

    print(f"  saved {stack.shape[0]} x 2 PNGs (short side {SHORT_SIDE_PX}px)")


def main() -> None:
    for flim_path in FLIM_PATHS:
        process_one_flim(flim_path)


if __name__ == "__main__":
    main()
