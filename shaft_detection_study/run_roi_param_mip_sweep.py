# -*- coding: utf-8 -*-
"""
Sweep DEEPD3_ROI_PARAMS and save {base_name}_mip_{combo}.png only (no per-spine ini).

DeepD3 inference runs once; each combo rebuilds 3D ROIs on the same prediction.

Example:
  python controlFLIMage/shaft_detection_study/run_roi_param_mip_sweep.py ^
    --flim "\\\\server\\path\\AP5_pos6_256_4x_001.flim" ^
    --grid quick

Outputs (next to .flim savefolder):
  roi_mip_sweep/{base_name}_mip_{combo_id}.png
  roi_mip_sweep/sweep_summary.csv
  roi_mip_sweep/montage_mip.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

_STUDY_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _STUDY_DIR.parent
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ.setdefault("PYTHONUTF8", "1")

from deepd3_spine_head_detector import resolve_savefolder
from shaft_detection_study.roi_mip_viz import (
    save_mip_overview_png,
    save_montage_thumbnails,
    save_two_panel_compare,
)
from shaft_detection_study.roi_param_grid import (
    BASELINE_SPINE_FILTERS,
    get_roi_grid,
    normalize_combo_entry,
)
from shaft_detection_study.roi_sweep_core import (
    build_roi_maps,
    run_deepd3_prediction,
    summarize_rois,
)

# Sync with deepd3_mushroom_spine_assign_save.py input settings
DEFAULT_MODEL = r"C:\Users\WatabeT\Documents\Git\DeepD3\deepd3\DeepD3_8F.h5"
MODEL_COMPARE_PATHS = [
    r"C:\Users\WatabeT\Documents\Git\DeepD3\deepd3\DeepD3_32F.h5",
    r"C:\Users\WatabeT\Documents\Git\DeepD3\deepd3\DeepD3_8F.h5",
    r"C:\Users\WatabeT\Documents\Git\DeepD3\deepd3\DeepD3_32F_94nm.h5",
]
USE_LOCAL_Z_MIP = False
LOCAL_Z_MIP_RADIUS = 0
STACK_PREPROCESS = "tophat_clahe"
ENHANCE_THIN_BRANCHES = False
DEFAULT_SUBDIR = "roi_mip_sweep_min1max200"


def model_tag_from_path(model_path: str) -> str:
    stem = Path(model_path).stem
    if stem == "DeepD3_32F_94nm":
        return "32F94nm"
    if stem.startswith("DeepD3_"):
        return stem.replace("DeepD3_", "")
    return stem


def base_name_from_flim(flim_path: str) -> str:
    return os.path.basename(flim_path[:-9])


def output_dir_for_flim(flim_path: str, subdir: str = DEFAULT_SUBDIR) -> str:
    out = os.path.join(resolve_savefolder(flim_path), subdir)
    os.makedirs(out, exist_ok=True)
    return out


def resolve_flim_paths(flim: str | None, folder: str | None, pattern: str) -> list[str]:
    if flim:
        if not os.path.isfile(flim):
            raise FileNotFoundError(flim)
        return [flim]
    if folder:
        paths = sorted(glob.glob(os.path.join(folder, pattern)))
        if not paths:
            raise FileNotFoundError(f"No files: {folder!r} / {pattern!r}")
        return paths
    raise ValueError("Provide --flim or --folder")


def combo_title(roi: dict, summary: dict, roi_mode: str = "thresholded") -> str:
    if roi_mode == "floodfill":
        return (
            f"a={roi['roi_areaThreshold']:.2f} p={roi['roi_peakThreshold']:.2f} floodfill "
            f"min=1 max=200 | R={summary['n_roi']} c={summary['n_cand']}"
        )
    return (
        f"a={roi['roi_areaThreshold']:.2f} thresh min=1 max=200 "
        f"| R={summary['n_roi']} c={summary['n_cand']}"
    )


def run_sweep(
    flim_path: str,
    model_path: str,
    grid_name: str,
    *,
    subdir: str = DEFAULT_SUBDIR,
    spine_filters: dict | None = None,
    model_tag: str | None = None,
    prefix_model_tag_in_filename: bool = False,
) -> str:
    spine_filters = spine_filters or BASELINE_SPINE_FILTERS
    base_name = base_name_from_flim(flim_path)
    tag = model_tag or model_tag_from_path(model_path)
    out_dir = output_dir_for_flim(flim_path, subdir=subdir)
    raw_combos = get_roi_grid(grid_name)
    combos = [normalize_combo_entry(c) for c in raw_combos]

    print(f"\n=== {flim_path} ===")
    print(f"Model: {model_path} ({tag})")
    print(f"Output: {out_dir}")
    print(f"Grid={grid_name!r}  combos={len(combos)}")
    print("Running DeepD3 once...")
    cache = run_deepd3_prediction(
        flim_path,
        model_path,
        use_local_z_mip=USE_LOCAL_Z_MIP,
        local_z_mip_radius=LOCAL_Z_MIP_RADIUS,
        stack_preprocess=STACK_PREPROCESS,
        enhance_thin_branches=ENHANCE_THIN_BRANCHES,
    )

    rows: list[dict] = []
    mip_paths: list[str] = []
    mip_labels: list[str] = []

    for i, (roi, combo_filters, cid, roi_mode) in enumerate(combos):
        print(f"  [{i + 1}/{len(combos)}] {cid} ({roi_mode})")
        r, _ = build_roi_maps(cache, roi, roi_mode=roi_mode)
        summary = summarize_rois(cache, r, combo_filters)
        title = combo_title(roi, summary, roi_mode=roi_mode)
        if prefix_model_tag_in_filename:
            mip_name = f"{base_name}_mip_{tag}_{cid}.png"
        else:
            mip_name = f"{base_name}_mip_{cid}.png"
        mip_path = os.path.join(out_dir, mip_name)
        save_mip_overview_png(
            cache.S,
            r,
            summary["prop_dict"],
            summary["cand_spines"],
            mip_path,
            title=title,
        )
        print(f"    -> {mip_name}  (ROI {summary['n_roi']}, cand {summary['n_cand']})")
        mip_paths.append(mip_path)
        mip_labels.append(cid)
        rows.append(
            {
                "flim": flim_path,
                "base_name": base_name,
                "model": model_path,
                "model_tag": tag,
                "combo_id": cid,
                "roi_mode": roi_mode,
                "mip_png": mip_path,
                "n_roi": summary["n_roi"],
                "n_cand": summary["n_cand"],
                **{k: roi[k] for k in roi},
                **{f"filter_{k}": v for k, v in combo_filters.items()},
            }
        )

    csv_path = os.path.join(out_dir, "sweep_summary.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV: {csv_path}")

    montage_path = os.path.join(out_dir, "montage_mip.png")
    montage_max = min(len(mip_paths), 120)
    montage_ncol = 6 if len(mip_paths) <= 36 else 8
    save_montage_thumbnails(
        mip_paths[:montage_max],
        mip_labels[:montage_max],
        montage_path,
        ncol=montage_ncol,
    )
    print(f"  montage: {montage_path} (first {montage_max} panels)")

    if len(mip_paths) == 2:
        pair_path = os.path.join(out_dir, f"{base_name}_compare_thresh_vs_floodfill_a015.png")
        save_two_panel_compare(
            mip_paths[0],
            mip_paths[1],
            mip_labels[0],
            mip_labels[1],
            pair_path,
        )
        print(f"  pair compare: {pair_path}")

    meta = os.path.join(out_dir, "run_meta.txt")
    with open(meta, "w", encoding="utf-8") as f:
        f.write(f"flim={flim_path}\n")
        f.write(f"model={model_path}\n")
        f.write(f"model_tag={tag}\n")
        f.write(f"grid={grid_name}\n")
        f.write(f"use_local_z_mip={USE_LOCAL_Z_MIP} r={LOCAL_Z_MIP_RADIUS}\n")
        f.write(f"stack_preprocess={STACK_PREPROCESS}\n")
        f.write(f"timestamp={datetime.now().isoformat()}\n")

    return out_dir


def resolve_model_paths(single_model: str, models: list[str] | None, model_compare: bool) -> list[str]:
    if models:
        paths = models
    elif model_compare:
        paths = list(MODEL_COMPARE_PATHS)
    else:
        paths = [single_model]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI param sweep -> _mip.png only")
    parser.add_argument("--flim", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*256_4x_001.flim")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Run sweep for each model path (e.g. with --grid floodfill_a020_peak)",
    )
    parser.add_argument(
        "--model-compare",
        action="store_true",
        help="Use built-in 32F / 8F / 32F_94nm model list",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="large",
        choices=[
            "quick",
            "area",
            "max_roi",
            "area_x_maxroi",
            "large",
            "compare_a015",
            "floodfill_a020_peak",
            "floodfill30",
            "floodfill",
            "mega",
        ],
        help="floodfill_a020_peak=area0.2 peak0.2-0.9; floodfill30=area x peak",
    )
    parser.add_argument("--subdir", type=str, default=DEFAULT_SUBDIR)
    args = parser.parse_args()

    model_paths = resolve_model_paths(args.model, args.models, args.model_compare)
    multi_model = len(model_paths) > 1

    for flim_path in resolve_flim_paths(args.flim, args.folder, args.pattern):
        all_rows: list[dict] = []
        for model_path in model_paths:
            tag = model_tag_from_path(model_path)
            subdir = args.subdir
            if multi_model:
                subdir = os.path.join(args.subdir, tag)
            out_dir = run_sweep(
                flim_path,
                model_path,
                args.grid,
                subdir=subdir,
                model_tag=tag,
                prefix_model_tag_in_filename=multi_model,
            )
            summary_csv = os.path.join(out_dir, "sweep_summary.csv")
            if os.path.isfile(summary_csv):
                with open(summary_csv, newline="", encoding="utf-8") as f:
                    all_rows.extend(list(csv.DictReader(f)))

        if multi_model and all_rows:
            base_name = base_name_from_flim(flim_path)
            combined_dir = output_dir_for_flim(flim_path, subdir=args.subdir)
            combined_csv = os.path.join(combined_dir, "sweep_summary_all_models.csv")
            with open(combined_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
            montage_ncol = 8
            by_model: dict[str, list[tuple[str, str]]] = {}
            for row in all_rows:
                by_model.setdefault(row["model_tag"], []).append(
                    (row["mip_png"], row["combo_id"])
                )
            for mtag, items in by_model.items():
                paths = [p for p, _ in items]
                labels = [lab for _, lab in items]
                montage_path = os.path.join(
                    combined_dir, f"montage_mip_{mtag}.png"
                )
                save_montage_thumbnails(
                    paths, labels, montage_path, ncol=montage_ncol
                )
            print(f"  combined CSV: {combined_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
