# -*- coding: utf-8 -*-
"""
Simple RESPAN class-overlay MIP (full Z): red=spine, green=shaft.

Uses existing Segmentation_Labels + annotation stack (or .flim). No ratings,
no local-Z windows, no per-spine panels.

Example:
  python make_respan_simple_class_overlay_mip.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _SCRIPT_DIR.parents[1]
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

from FLIMageAlignment import get_xyz_pixel_um  # noqa: E402
from FLIMageFileReader2 import FileReader  # noqa: E402

DEFAULT_FLIM_FOLDER = r"G:\ImagingData\Tetsuya\20260608\mushroom_multi_dend"
DEFAULT_OUT_FOLDER = r"G:\ImagingData\Tetsuya\20260608\mushroom_multi_dend_respan_simple"
DEFAULT_FLIM_GLOB = "*__highmag_*_002.flim"
DEFAULT_CHANNEL = 2
DEFAULT_OVERLAY_ALPHA = 0.55

SPINE_LABEL_VALUE = 1
DENDRITE_LABEL_VALUE = 2
COLOR_SPINE = np.array([1.0, 0.2, 0.2], dtype=np.float32)
COLOR_SHAFT = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def base_name_from_flim_path(flim_path: str | Path) -> str:
    return os.path.basename(str(flim_path)[:-9])


def normalize_to_float(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    low, high = np.percentile(arr, (1.0, 99.5))
    if high <= low:
        high = low + 1.0
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def load_flim_zyx(flim_path: Path, channel: int) -> np.ndarray:
    iminfo = FileReader()
    iminfo.read_imageFile(str(flim_path), True)
    zyx = np.array(iminfo.image)[:, :, channel - 1, :, :, :].sum(axis=(1, 4))
    return np.asarray(zyx, dtype=np.float32)


def annotation_tiff_path(flim_path: Path, channel: int) -> Path:
    stem = f"{flim_path.stem}_ch{channel}_zyx"
    return flim_path.parent / "deepd3_annotation_stacks" / f"{stem}.tif"


def load_raw_zyx(flim_path: Path, channel: int) -> np.ndarray:
    tiff_path = annotation_tiff_path(flim_path, channel)
    if tiff_path.is_file():
        return np.asarray(tf.imread(tiff_path), dtype=np.float32)
    return load_flim_zyx(flim_path, channel)


def find_label_path(flim_path: Path, channel: int) -> Path:
    """Locate Segmentation_Labels TIFF for this FLIM."""
    parent = flim_path.parent
    tiff_name = f"{flim_path.stem}_ch{channel}_zyx.tif"
    candidates = [
        parent / "respan_runs" / flim_path.stem / "Validation_Data" / "Segmentation_Labels" / tiff_name,
        parent / "respan_runs" / f"{flim_path.stem}_ch{channel}_zyx" / "Validation_Data" / "Segmentation_Labels" / tiff_name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    # Fallback: search under respan_runs for this TIFF name
    runs = parent / "respan_runs"
    if runs.is_dir():
        matches = list(runs.glob(f"**/Segmentation_Labels/{tiff_name}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"RESPAN Segmentation_Labels not found for {flim_path.name} "
        f"(looked for {tiff_name})"
    )


def build_class_overlay_mip(
    raw_zyx: np.ndarray,
    class_labels_zyx: np.ndarray,
    *,
    overlay_alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> np.ndarray:
    if class_labels_zyx.shape != raw_zyx.shape:
        raise ValueError(
            f"Shape mismatch raw={raw_zyx.shape} labels={class_labels_zyx.shape}"
        )
    gray = normalize_to_float(raw_zyx.max(axis=0))
    rgb = np.stack([gray, gray, gray], axis=-1)
    # Draw shaft first, then spine on top (same as DeepD3 simple overlay)
    shaft_mip = np.any(class_labels_zyx == DENDRITE_LABEL_VALUE, axis=0)
    spine_mip = np.any(class_labels_zyx == SPINE_LABEL_VALUE, axis=0)
    rgb[shaft_mip] = (1.0 - overlay_alpha) * rgb[shaft_mip] + overlay_alpha * COLOR_SHAFT
    rgb[spine_mip] = (1.0 - overlay_alpha) * rgb[spine_mip] + overlay_alpha * COLOR_SPINE
    return np.clip(rgb, 0.0, 1.0)


def process_one_flim(
    flim_path: Path,
    out_folder: Path,
    *,
    channel: int,
    overlay_alpha: float,
) -> Path:
    base_name = base_name_from_flim_path(flim_path)
    print(f"\nFLIM: {flim_path}")
    raw = load_raw_zyx(flim_path, channel)
    label_path = find_label_path(flim_path, channel)
    labels = np.asarray(tf.imread(label_path))
    print(f"  raw: {raw.shape}, labels: {labels.shape} from {label_path}")

    overlay = build_class_overlay_mip(raw, labels, overlay_alpha=overlay_alpha)
    out_path = out_folder / f"{base_name}_respan_class_overlay_mip.png"
    plt.imsave(out_path, overlay)
    plt.imsave(
        out_folder / f"{base_name}_raw_mip.png",
        normalize_to_float(raw.max(axis=0)),
        cmap="gray",
    )
    print(f"  saved: {out_path}")
    return out_path


def list_flim_files(folder: Path, pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(str(folder / pattern))]
    return sorted(paths, key=lambda p: p.name.lower())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple RESPAN full-Z class-overlay MIP (red spine / green shaft)."
    )
    parser.add_argument("--folder", default=DEFAULT_FLIM_FOLDER)
    parser.add_argument("--out", default=DEFAULT_OUT_FOLDER)
    parser.add_argument("--glob", dest="filename_pattern", default=DEFAULT_FLIM_GLOB)
    parser.add_argument("--channel", type=int, default=DEFAULT_CHANNEL, choices=[1, 2])
    parser.add_argument("--overlay-alpha", type=float, default=DEFAULT_OVERLAY_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    folder = Path(args.folder)
    out_folder = Path(args.out)
    out_folder.mkdir(parents=True, exist_ok=True)

    flim_paths = list_flim_files(folder, args.filename_pattern)
    if not flim_paths:
        print(f"No files matching {args.filename_pattern} in {folder}")
        return

    print(f"Folder: {folder}")
    print(f"Out:    {out_folder}")
    print(f"Found {len(flim_paths)} FLIM file(s)")

    ok = 0
    for idx, flim_path in enumerate(flim_paths, start=1):
        print(f"[{idx}/{len(flim_paths)}]")
        try:
            process_one_flim(
                flim_path,
                out_folder,
                channel=args.channel,
                overlay_alpha=args.overlay_alpha,
            )
            ok += 1
        except Exception as exc:
            print(f"  FAILED: {exc}")

    print(f"\nDone. ok={ok}/{len(flim_paths)} -> {out_folder}")


if __name__ == "__main__":
    main()
