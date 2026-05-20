# -*- coding: utf-8 -*-
"""
Sweep shaft (dendrite) parameters without spine assignment.

Example (from repo root, DeepD3 env):
  python controlFLIMage/shaft_detection_study/run_shaft_param_sweep.py ^
    --flim "\\\\server\\path\\sample_256_4x_001.flim" ^
    --grid quick

Outputs under shaft_detection_study/results/<flim_stem>/:
  - sweep_summary.csv
  - montage_fused_mip.png
  - compare_<combo_id>.png  (raw | shaft raw | shaft fused)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

# Windows console: avoid UnicodeEncodeError from tqdm/keras progress bars
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ.setdefault("PYTHONUTF8", "1")
from datetime import datetime
from pathlib import Path

import numpy as np

_STUDY_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _STUDY_DIR.parent
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

from shaft_detection_study.shaft_inference import (
    load_flim_stack,
    raw_z_mip,
    run_shaft_combo,
)
from shaft_detection_study.shaft_metrics import compare_raw_vs_fused, summarize_shaft_map
from shaft_detection_study.shaft_param_grid import (
    combo_id,
    get_parameter_grid,
    shaft_parameter_help,
)
from shaft_detection_study.shaft_viz import save_montage, save_triple_mip_png

DEFAULT_MODEL = r"C:\Users\WatabeT\Documents\Git\DeepD3\deepd3\DeepD3_32F_94nm.h5"
DEFAULT_RESULTS = _STUDY_DIR / "results"


def resolve_flim_paths(
    flim: str | None,
    folder: str | None,
    pattern: str,
) -> list[str]:
    if flim:
        if not os.path.isfile(flim):
            raise FileNotFoundError(flim)
        return [flim]
    if folder:
        paths = sorted(glob.glob(os.path.join(folder, pattern)))
        if not paths:
            raise FileNotFoundError(
                f"No files for folder={folder!r} pattern={pattern!r}"
            )
        return paths
    raise ValueError("Provide --flim or --folder")


def flatten_combo_for_csv(combo: dict) -> dict:
    return {
        "stack_preprocess": combo.get("stack_preprocess", "none"),
        "enhance_thin_branches": bool(combo.get("enhance_thin_branches", False)),
        "image_fusion_percentile": combo.get("image_fusion_percentile", ""),
        "image_fusion_weight": combo.get("image_fusion_weight", ""),
        "dendrite_closing_iterations": combo.get("dendrite_closing_iterations", ""),
    }


def run_sweep_for_flim(
    flim_path: str,
    combos: list[dict],
    model_path: str,
    out_root: Path,
    *,
    raw_vmax_percentile: float = 95.0,
    contour_level: float = 0.2,
    ch_1or2: int | None = None,
) -> Path:
    stem = Path(flim_path).stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {flim_path} ===")
    print(f"Output: {out_dir}")
    zyx, xy_um, z_um = load_flim_stack(flim_path, ch_1or2=ch_1or2)
    mip = raw_z_mip(zyx)
    print(f"  stack shape ZYX={zyx.shape}  xy={xy_um:.3f} um  z={z_um:.3f} um")

    pred_cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    montage_imgs: list[np.ndarray] = []
    montage_labels: list[str] = []

    for i, combo in enumerate(combos):
        cid = combo_id(combo)
        print(f"  [{i + 1}/{len(combos)}] {cid}")

        result = run_shaft_combo(
            zyx,
            xy_um,
            z_um,
            model_path,
            combo,
            dendrite_pred_raw_cache=pred_cache,
        )
        dend_raw = result["dendrite_pred_raw"]
        dend_fused = result["dendrite_pred_fused"]

        row = {
            "flim": flim_path,
            "combo_id": cid,
            "xy_pixel_um": xy_um,
            "z_pixel_um": z_um,
            **flatten_combo_for_csv(combo),
        }
        row.update(summarize_shaft_map(dend_raw, tag="raw"))
        row.update(
            {f"fused_{k}": v for k, v in summarize_shaft_map(dend_fused, tag="fused").items()}
        )
        row.update(compare_raw_vs_fused(dend_raw, dend_fused))
        rows.append(row)

        png_path = out_dir / f"compare_{cid}.png"
        save_triple_mip_png(
            mip,
            dend_raw,
            dend_fused,
            str(png_path),
            title_suffix=cid,
            raw_vmax_percentile=raw_vmax_percentile,
            contour_level=contour_level,
        )

        montage_imgs.append(np.max(dend_fused, axis=0))
        montage_labels.append(cid)

    csv_path = out_dir / "sweep_summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV: {csv_path}")

    montage_path = out_dir / "montage_fused_mip.png"
    save_montage(montage_imgs, montage_labels, str(montage_path), ncol=4)
    print(f"  montage: {montage_path}")

    meta_path = out_dir / "run_meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"flim={flim_path}\n")
        f.write(f"model={model_path}\n")
        f.write(f"n_combos={len(combos)}\n")
        f.write(f"timestamp={datetime.now().isoformat()}\n")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep shaft (dendrite) parameters; spine ROIs are skipped.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=shaft_parameter_help(),
    )
    parser.add_argument("--flim", type=str, default=None, help="Single .flim path")
    parser.add_argument("--folder", type=str, default=None, help="Folder with .flim files")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*256_4x_001.flim",
        help="Glob under --folder",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="quick",
        choices=["quick", "preprocess", "fusion", "full"],
        help="Parameter grid preset",
    )
    parser.add_argument(
        "--stack-preprocess",
        type=str,
        default="tophat_clahe",
        help="For --grid fusion only",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_RESULTS),
        help="Root output directory",
    )
    parser.add_argument("--raw-vmax-p", type=float, default=95.0)
    parser.add_argument("--contour", type=float, default=0.2)
    parser.add_argument("--ch", type=int, default=None, help="FLIM channel 1 or 2")
    parser.add_argument("--help-params", action="store_true", help="Print parameter guide")
    args = parser.parse_args()

    if args.help_params:
        print(shaft_parameter_help())
        return

    combos = get_parameter_grid(
        args.grid,
        stack_preprocess=args.stack_preprocess,
    )
    print(f"Grid={args.grid!r}  combinations={len(combos)}")

    flim_paths = resolve_flim_paths(args.flim, args.folder, args.pattern)
    out_root = Path(args.out_dir)

    for flim_path in flim_paths:
        run_sweep_for_flim(
            flim_path,
            combos,
            args.model,
            out_root,
            raw_vmax_percentile=args.raw_vmax_p,
            contour_level=args.contour,
            ch_1or2=args.ch,
        )

    print(f"\nDone. Results under: {out_root}")


if __name__ == "__main__":
    main()
