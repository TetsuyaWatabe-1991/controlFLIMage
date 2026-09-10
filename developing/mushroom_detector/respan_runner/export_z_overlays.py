"""Export per-Z-slice overlay images from a RESPAN run folder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf


LABEL_COLORS = {
    1: np.array([1.0, 0.2, 0.2], dtype=np.float32),  # spine: red
    2: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # dendrite: green
}
OVERLAY_ALPHA = 0.55


def _normalize_to_float(image: np.ndarray) -> np.ndarray:
    """Map image to [0, 1] using robust percentiles."""
    arr = image.astype(np.float32)
    low, high = np.percentile(arr, (1.0, 99.5))
    if high <= low:
        high = low + 1.0
    arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return arr


def _build_overlay_slice(
    raw_slice: np.ndarray,
    label_slice: np.ndarray,
    overlay_alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """Return RGB overlay for one Z slice."""
    gray = _normalize_to_float(raw_slice)
    rgb = np.stack([gray, gray, gray], axis=-1)

    for label_value, color in LABEL_COLORS.items():
        mask = label_slice == label_value
        if not np.any(mask):
            continue
        rgb[mask] = (1.0 - overlay_alpha) * rgb[mask] + overlay_alpha * color

    return np.clip(rgb, 0.0, 1.0)


def _load_spine_annotations(csv_path: Path | None) -> dict[int, list[tuple[int, float, float, str]]]:
    """Group spine centroids by nearest Z index."""
    if csv_path is None or not csv_path.exists():
        return {}

    grouped: dict[int, list[tuple[int, float, float, str]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            spine_id = int(float(row["spine_id"]))
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            spine_type = row.get("spine_type", "")
            z_idx = int(round(z))
            grouped.setdefault(z_idx, []).append((spine_id, x, y, spine_type))
    return grouped


def _resolve_paths(
    run_dir: Path,
    raw_tiff: Path | None,
    label_tiff: Path | None,
    spines_csv: Path | None,
) -> tuple[Path, Path, Path | None]:
    if raw_tiff is None:
        candidates = sorted(run_dir.glob("**/*.tif"))
        raw_candidates = [
            p for p in candidates
            if "Segmentation_Labels" not in str(p)
            and "nnUnet_input" not in str(p)
            and "Validation_MIPs" not in str(p)
            and "Z_slice_overlays" not in str(p)
            and "MIP_" not in p.name
        ]
        if not raw_candidates:
            raise FileNotFoundError(f"No raw TIFF found under {run_dir}")
        raw_tiff = raw_candidates[0]

    stem = raw_tiff.stem
    if label_tiff is None:
        label_tiff = run_dir / "Validation_Data" / "Segmentation_Labels" / f"{stem}.tif"
    if spines_csv is None:
        spines_csv = run_dir / "Tables" / f"{stem}_detected_spines.csv"
        if not spines_csv.exists():
            spines_csv = None

    if not raw_tiff.exists():
        raise FileNotFoundError(f"Raw TIFF not found: {raw_tiff}")
    if not label_tiff.exists():
        raise FileNotFoundError(f"Label TIFF not found: {label_tiff}")

    return raw_tiff, label_tiff, spines_csv


def export_z_overlays(
    run_dir: Path,
    raw_tiff: Path | None = None,
    label_tiff: Path | None = None,
    spines_csv: Path | None = None,
    output_dir: Path | None = None,
    dpi: int = 200,
    annotate_spines: bool = True,
) -> Path:
    """Write one PNG per Z slice with segmentation overlaid on the raw image."""
    raw_tiff, label_tiff, spines_csv = _resolve_paths(
        run_dir, raw_tiff, label_tiff, spines_csv
    )

    raw = tf.imread(raw_tiff)
    labels = tf.imread(label_tiff)
    if raw.ndim != 3 or labels.ndim != 3:
        raise ValueError(f"Expected ZYX stacks, got raw={raw.shape}, labels={labels.shape}")
    if raw.shape != labels.shape:
        raise ValueError(f"Shape mismatch: raw={raw.shape}, labels={labels.shape}")

    if output_dir is None:
        output_dir = run_dir / "Validation_Data" / "Z_slice_overlays" / raw_tiff.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    spine_by_z = _load_spine_annotations(spines_csv) if annotate_spines else {}
    stem = raw_tiff.stem

    for z in range(raw.shape[0]):
        overlay = _build_overlay_slice(raw[z], labels[z])
        spine_px = int((labels[z] == 1).sum())
        dendrite_px = int((labels[z] == 2).sum())
        spine_count = len(spine_by_z.get(z, []))

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.imshow(overlay, origin="upper")
        ax.set_title(
            f"{stem} | Z={z} | spine px={spine_px} | dendrite px={dendrite_px} | spines={spine_count}",
            fontsize=10,
        )
        ax.axis("off")

        for spine_id, x, y, spine_type in spine_by_z.get(z, []):
            ax.plot(x, y, marker="o", markersize=5, markerfacecolor="yellow",
                    markeredgecolor="black", markeredgewidth=0.5)
            label = str(spine_id)
            if spine_type:
                label = f"{spine_id}"
            ax.text(
                x + 1.5,
                y - 1.5,
                label,
                color="yellow",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.0, "edgecolor": "none"},
            )

        out_path = output_dir / f"{stem}_z{z:02d}_overlay.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    montage_path = output_dir / f"{stem}_all_z_montage.png"
    n_z = raw.shape[0]
    fig, axes = plt.subplots(1, n_z, figsize=(2.2 * n_z, 2.5), dpi=dpi)
    if n_z == 1:
        axes = [axes]
    for z, ax in enumerate(axes):
        overlay = _build_overlay_slice(raw[z], labels[z])
        ax.imshow(overlay, origin="upper")
        ax.set_title(f"Z{z}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{stem} segmentation overlay (red=spine, green=dendrite)", fontsize=10)
    fig.savefig(montage_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved {n_z} Z-slice overlays to: {output_dir}")
    print(f"Montage: {montage_path}")
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-Z overlay PNGs from RESPAN segmentation output."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="RESPAN run output folder.",
    )
    parser.add_argument("--raw-tiff", type=Path, default=None, help="Original ZYX TIFF.")
    parser.add_argument("--label-tiff", type=Path, default=None, help="Segmentation label TIFF.")
    parser.add_argument("--spines-csv", type=Path, default=None, help="Detected spines CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder for PNGs.")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    parser.add_argument(
        "--no-spine-labels",
        action="store_true",
        help="Do not annotate detected spine IDs on overlays.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    export_z_overlays(
        run_dir=args.run_dir,
        raw_tiff=args.raw_tiff,
        label_tiff=args.label_tiff,
        spines_csv=args.spines_csv,
        output_dir=args.output_dir,
        dpi=args.dpi,
        annotate_spines=not args.no_spine_labels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
