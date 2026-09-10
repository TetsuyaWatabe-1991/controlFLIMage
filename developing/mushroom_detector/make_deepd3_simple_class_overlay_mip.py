# -*- coding: utf-8 -*-
"""
Simple DeepD3 class-overlay MIP for fair comparison with RESPAN.

For each high-mag .flim: raw Z-MIP with spine (red) and shaft (green) hatches only.
No per-spine panels, ratings, ROI geometry, local-Z windows, or fancy colors.

Prediction source (in order):
  1) Existing {stem}_S_spine.tif / {stem}_S_shaft.tif next to the .flim
  2) Fresh DeepD3 predictWholeImage (no thin-branch fusion, no local Z-MIP)

Example:
  C:\\Users\\yasudalab\\Documents\\Tetsuya_GIT\\deepd3\\Scripts\\python.exe ^
    make_deepd3_simple_class_overlay_mip.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from io import BytesIO
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
DEFAULT_OUT_FOLDER = r"G:\ImagingData\Tetsuya\20260608\mushroom_multi_dend_deepd3_simple"
DEFAULT_FLIM_GLOB = "*__highmag_*_002.flim"
DEFAULT_CHANNEL = 2
DEFAULT_SPINE_THRESHOLD = 0.2
DEFAULT_SHAFT_THRESHOLD = 0.2
DEFAULT_OVERLAY_ALPHA = 0.55
DEFAULT_MODEL = (
    r"C:\Users\yasudalab\Documents\Tetsuya_GIT\ongoing\deepd3\DeepD3_8F.h5"
)

COLOR_SPINE = np.array([1.0, 0.2, 0.2], dtype=np.float32)
COLOR_SHAFT = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def base_name_from_flim_path(flim_path: str | Path) -> str:
    return os.path.basename(str(flim_path)[:-9])


def savefolder_from_flim_path(flim_path: str | Path) -> Path:
    legacy = Path(str(flim_path)[:-9])
    if legacy.is_dir():
        return legacy
    stem = Path(flim_path).with_suffix("")
    return stem if stem.is_dir() else legacy


def normalize_to_float(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    low, high = np.percentile(arr, (1.0, 99.5))
    if high <= low:
        high = low + 1.0
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def load_flim_zyx(flim_path: Path, channel: int) -> tuple[np.ndarray, float, float]:
    iminfo = FileReader()
    iminfo.read_imageFile(str(flim_path), True)
    zyx = np.array(iminfo.image)[:, :, channel - 1, :, :, :].sum(axis=(1, 4))
    zyx = np.asarray(zyx, dtype=np.float32)
    xy_um, _, z_um = get_xyz_pixel_um(iminfo)
    return zyx, float(xy_um), float(z_um)


def prediction_tiff_paths(savefolder: Path, base_name: str) -> tuple[Path, Path]:
    return (
        savefolder / f"{base_name}_S_spine.tif",
        savefolder / f"{base_name}_S_shaft.tif",
    )


def load_existing_predictions(
    savefolder: Path, base_name: str
) -> tuple[np.ndarray, np.ndarray] | None:
    spine_path, shaft_path = prediction_tiff_paths(savefolder, base_name)
    if not (spine_path.is_file() and shaft_path.is_file()):
        return None
    spine = np.asarray(tf.imread(spine_path), dtype=np.float32)
    shaft = np.asarray(tf.imread(shaft_path), dtype=np.float32)
    return spine, shaft


def run_deepd3_predict(
    zyx: np.ndarray,
    xy_um: float,
    z_um: float,
    model_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    from deepd3.core.analysis import Stack

    buf = BytesIO()
    tf.imwrite(buf, np.asarray(zyx, dtype=np.float32))
    stack = Stack(buf, dimensions=dict(xy=xy_um, z=z_um))
    print(f"  DeepD3 predict: {model_path}")
    stack.predictWholeImage(model_path)
    shaft = np.asarray(stack.prediction[..., 0], dtype=np.float32)
    spine = np.asarray(stack.prediction[..., 1], dtype=np.float32)
    return spine, shaft


def build_simple_class_overlay_mip(
    raw_zyx: np.ndarray,
    spine_pred_zyx: np.ndarray,
    shaft_pred_zyx: np.ndarray,
    *,
    spine_threshold: float,
    shaft_threshold: float,
    overlay_alpha: float,
) -> np.ndarray:
    """Full-Z MIP: gray raw + green shaft + red spine (spine drawn last)."""
    if spine_pred_zyx.shape != raw_zyx.shape or shaft_pred_zyx.shape != raw_zyx.shape:
        raise ValueError(
            f"Shape mismatch raw={raw_zyx.shape} spine={spine_pred_zyx.shape} "
            f"shaft={shaft_pred_zyx.shape}"
        )
    gray = normalize_to_float(raw_zyx.max(axis=0))
    rgb = np.stack([gray, gray, gray], axis=-1)
    shaft_mip = np.any(shaft_pred_zyx >= shaft_threshold, axis=0)
    spine_mip = np.any(spine_pred_zyx >= spine_threshold, axis=0)
    rgb[shaft_mip] = (1.0 - overlay_alpha) * rgb[shaft_mip] + overlay_alpha * COLOR_SHAFT
    rgb[spine_mip] = (1.0 - overlay_alpha) * rgb[spine_mip] + overlay_alpha * COLOR_SPINE
    return np.clip(rgb, 0.0, 1.0)


def process_one_flim(
    flim_path: Path,
    out_folder: Path,
    *,
    channel: int,
    spine_threshold: float,
    shaft_threshold: float,
    overlay_alpha: float,
    model_path: str,
    force_repredict: bool,
    save_prediction_tiffs: bool,
) -> Path:
    base_name = base_name_from_flim_path(flim_path)
    legacy_save = savefolder_from_flim_path(flim_path)
    print(f"\nFLIM: {flim_path}")
    print(f"  base_name: {base_name}")

    zyx, xy_um, z_um = load_flim_zyx(flim_path, channel)
    print(f"  raw ZYX: {zyx.shape}, xy={xy_um:.4f} um, z={z_um:.4f} um")

    preds = None if force_repredict else load_existing_predictions(legacy_save, base_name)
    if preds is None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"DeepD3 model not found: {model_path}")
        spine, shaft = run_deepd3_predict(zyx, xy_um, z_um, model_path)
        src = "fresh DeepD3 predict"
        if save_prediction_tiffs:
            out_pred = out_folder / base_name
            out_pred.mkdir(parents=True, exist_ok=True)
            tf.imwrite(out_pred / f"{base_name}_S_spine.tif", spine)
            tf.imwrite(out_pred / f"{base_name}_S_shaft.tif", shaft)
    else:
        spine, shaft = preds
        src = f"existing prediction TIFFs in {legacy_save}"
    print(f"  prediction source: {src}")

    overlay = build_simple_class_overlay_mip(
        zyx,
        spine,
        shaft,
        spine_threshold=spine_threshold,
        shaft_threshold=shaft_threshold,
        overlay_alpha=overlay_alpha,
    )
    out_path = out_folder / f"{base_name}_deepd3_class_overlay_mip.png"
    plt.imsave(out_path, overlay)
    raw_path = out_folder / f"{base_name}_raw_mip.png"
    plt.imsave(raw_path, normalize_to_float(zyx.max(axis=0)), cmap="gray")
    print(f"  saved: {out_path}")
    return out_path


def list_flim_files(folder: Path, pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(str(folder / pattern))]
    return sorted(paths, key=lambda p: p.name.lower())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple DeepD3 spine/shaft class-overlay MIP (red/green)."
    )
    parser.add_argument("--folder", default=DEFAULT_FLIM_FOLDER)
    parser.add_argument("--out", default=DEFAULT_OUT_FOLDER)
    parser.add_argument("--glob", dest="filename_pattern", default=DEFAULT_FLIM_GLOB)
    parser.add_argument("--channel", type=int, default=DEFAULT_CHANNEL, choices=[1, 2])
    parser.add_argument("--spine-threshold", type=float, default=DEFAULT_SPINE_THRESHOLD)
    parser.add_argument("--shaft-threshold", type=float, default=DEFAULT_SHAFT_THRESHOLD)
    parser.add_argument("--overlay-alpha", type=float, default=DEFAULT_OVERLAY_ALPHA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--force-repredict",
        action="store_true",
        help="Ignore existing S_spine/S_shaft TIFFs and re-run DeepD3.",
    )
    parser.add_argument(
        "--save-prediction-tiffs",
        action="store_true",
        help="When re-predicting, also write S_spine/S_shaft under --out.",
    )
    parser.add_argument(
        "--sweep-thresholds",
        default="",
        help=(
            "Comma-separated thresholds for spine=shaft sweep "
            "(e.g. 0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5). "
            "Writes per-threshold PNGs plus a montage per field."
        ),
    )
    return parser.parse_args()


def _parse_threshold_list(text: str) -> list[float]:
    vals: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("Empty --sweep-thresholds list")
    return vals


def process_one_flim_threshold_sweep(
    flim_path: Path,
    out_folder: Path,
    *,
    channel: int,
    thresholds: list[float],
    overlay_alpha: float,
    model_path: str,
    force_repredict: bool,
) -> Path:
    """Save overlay at each threshold and a side-by-side montage."""
    base_name = base_name_from_flim_path(flim_path)
    legacy_save = savefolder_from_flim_path(flim_path)
    print(f"\nFLIM: {flim_path}")
    zyx, xy_um, z_um = load_flim_zyx(flim_path, channel)
    preds = None if force_repredict else load_existing_predictions(legacy_save, base_name)
    if preds is None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"DeepD3 model not found: {model_path}")
        spine, shaft = run_deepd3_predict(zyx, xy_um, z_um, model_path)
    else:
        spine, shaft = preds

    overlays: list[tuple[float, np.ndarray]] = []
    for thr in thresholds:
        rgb = build_simple_class_overlay_mip(
            zyx,
            spine,
            shaft,
            spine_threshold=thr,
            shaft_threshold=thr,
            overlay_alpha=overlay_alpha,
        )
        thr_tag = f"t{thr:.2f}".replace(".", "p")
        plt.imsave(out_folder / f"{base_name}_deepd3_overlay_{thr_tag}.png", rgb)
        overlays.append((thr, rgb))

    n = len(overlays)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes_list = np.atleast_1d(axes).ravel()
    for ax, (thr, rgb) in zip(axes_list, overlays):
        ax.imshow(rgb, origin="upper")
        ax.set_title(f"thr={thr:.2f}", fontsize=10)
        ax.axis("off")
    for ax in axes_list[n:]:
        ax.axis("off")
    fig.suptitle(
        f"{base_name} spine/shaft thr sweep (red=spine, green=shaft)",
        fontsize=11,
    )
    fig.tight_layout()
    montage = out_folder / f"{base_name}_thresh_montage.png"
    fig.savefig(montage, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  montage: {montage}")
    return montage


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
    if args.sweep_thresholds.strip():
        thresholds = _parse_threshold_list(args.sweep_thresholds)
        print(f"Sweep thresholds: {thresholds}, alpha={args.overlay_alpha}")
        for idx, flim_path in enumerate(flim_paths, start=1):
            print(f"\n[{idx}/{len(flim_paths)}]")
            try:
                process_one_flim_threshold_sweep(
                    flim_path,
                    out_folder,
                    channel=args.channel,
                    thresholds=thresholds,
                    overlay_alpha=args.overlay_alpha,
                    model_path=args.model,
                    force_repredict=args.force_repredict,
                )
                ok += 1
            except Exception as exc:
                print(f"  FAILED: {exc}")
    else:
        print(
            f"Thresholds: spine>={args.spine_threshold}, shaft>={args.shaft_threshold}, "
            f"alpha={args.overlay_alpha}"
        )
        for idx, flim_path in enumerate(flim_paths, start=1):
            print(f"\n[{idx}/{len(flim_paths)}]")
            try:
                process_one_flim(
                    flim_path,
                    out_folder,
                    channel=args.channel,
                    spine_threshold=args.spine_threshold,
                    shaft_threshold=args.shaft_threshold,
                    overlay_alpha=args.overlay_alpha,
                    model_path=args.model,
                    force_repredict=args.force_repredict,
                    save_prediction_tiffs=args.save_prediction_tiffs,
                )
                ok += 1
            except Exception as exc:
                print(f"  FAILED: {exc}")

    print(f"\nDone. ok={ok}/{len(flim_paths)} -> {out_folder}")


if __name__ == "__main__":
    main()
