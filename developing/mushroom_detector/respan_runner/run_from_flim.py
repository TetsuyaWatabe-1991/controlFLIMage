"""Run RESPAN and Z-slice overlays starting from a single .flim file."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from respan_runner.bootstrap import ensure_respan_syspath
from respan_runner.export_z_overlays import export_z_overlays
from respan_runner.paths import nnunet_predict_bat, respan_model_dir
from respan_runner.run_batch_stacks import (
    _configure_settings,
    _load_resolution,
    _resolve_nnunet_fold,
    _run_subfolder_name,
    _write_analysis_settings,
)

ensure_respan_syspath()

from RESPAN.Environment import imgan, main as respan_main, sr  # noqa: E402
from export_highmag_flim_stacks_for_annotation import (  # noqa: E402
    load_flim_zyx_uint16,
    save_annotation_stack,
)

MODEL_DIR = respan_model_dir()
NNUNET_PREDICT_BAT = nnunet_predict_bat()


def _build_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("respan_from_flim")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    log_path = run_dir / f"RESPAN_Log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _export_stack_from_flim(
    flim_path: Path,
    export_dir: Path,
    channel: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    stem = flim_path.stem
    out_tif = export_dir / f"{stem}_ch{channel}_zyx.tif"
    out_json = export_dir / f"{stem}_ch{channel}_zyx.json"

    if out_tif.exists() and out_json.exists() and not overwrite:
        return out_tif, out_json

    export_dir.mkdir(parents=True, exist_ok=True)
    zyx, meta = load_flim_zyx_uint16(flim_path, ch_1or2=channel)
    save_annotation_stack(zyx, meta, out_tif, out_json=out_json, save_mip=True)
    return out_tif, out_json


def _setup_run_folder(
    tiff_path: Path,
    json_path: Path,
    run_parent: Path,
    rerun: bool,
) -> Path:
    run_dir = run_parent / _run_subfolder_name(tiff_path)
    if run_dir.exists() and rerun:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_parent.mkdir(parents=True, exist_ok=True)

    target_tif = run_dir / tiff_path.name
    if not target_tif.exists():
        try:
            os.link(tiff_path, target_tif)
        except OSError:
            shutil.copy2(tiff_path, target_tif)

    input_xy, input_z, model_xy, model_z = _load_resolution(json_path)
    _write_analysis_settings(
        run_dir / "Analysis_Settings.yaml",
        input_xy,
        input_z,
        model_xy,
        model_z,
    )
    return run_dir


def run_from_flim(
    flim_path: Path,
    *,
    channel: int = 2,
    run_parent: Path | None = None,
    export_subdir: str = "deepd3_annotation_stacks",
    rerun: bool = False,
    overwrite_export: bool = False,
    skip_analysis: bool = False,
    skip_overlays: bool = False,
    nnunet_fold: str | int = "all",
) -> dict:
    """Export .flim to ZYX TIFF, run RESPAN, and write Z overlays."""
    flim_path = flim_path.resolve()
    if not flim_path.exists():
        raise FileNotFoundError(f"FLIM file not found: {flim_path}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")
    if not NNUNET_PREDICT_BAT.exists():
        raise FileNotFoundError(f"nnUNet launcher not found: {NNUNET_PREDICT_BAT}")

    flim_parent = flim_path.parent
    export_dir = flim_parent / export_subdir
    if run_parent is None:
        run_parent = flim_parent / "respan_runs"

    tiff_path, json_path = _export_stack_from_flim(
        flim_path,
        export_dir,
        channel=channel,
        overwrite=overwrite_export,
    )
    run_dir = _setup_run_folder(tiff_path, json_path, run_parent, rerun=rerun)
    logger = _build_logger(run_dir)

    logger.info("RESPAN from FLIM")
    logger.info(f"  source FLIM: {flim_path}")
    logger.info(f"  exported TIFF: {tiff_path}")
    logger.info(f"  run folder: {run_dir}")

    if not skip_analysis:
        settings, locations = respan_main.initialize_RESPAN(str(run_dir) + os.sep)
        _configure_settings(settings, nnunet_fold=nnunet_fold)
        resolved_fold = _resolve_nnunet_fold(nnunet_fold)
        if str(nnunet_fold) != resolved_fold:
            logger.info(
                f"  nnUNet fold {nnunet_fold} unavailable; using fold_{resolved_fold} checkpoint"
            )
        logger.info(f"  nnUNet fold: {settings.nnunet_fold}")

        log = sr.restore_and_segment(settings, locations, logger)
        if log != 0 or settings.Track:
            raise RuntimeError(f"RESPAN segmentation failed with code {log}")
        imgan.analyze_spines(settings, locations, log, logger)
        logger.info("RESPAN analysis complete.")

    overlay_dir = None
    if not skip_overlays:
        overlay_dir = export_z_overlays(
            run_dir=run_dir,
            raw_tiff=tiff_path,
            label_tiff=run_dir / "Validation_Data" / "Segmentation_Labels" / tiff_path.name,
            spines_csv=run_dir / "Tables" / f"{tiff_path.stem}_detected_spines.csv",
        )
        logger.info(f"Z-slice overlays: {overlay_dir}")

    summary_csv = run_dir / "Tables" / f"{tiff_path.stem}_detected_spines.csv"
    spine_count = 0
    if summary_csv.exists():
        with summary_csv.open(encoding="utf-8") as handle:
            spine_count = max(sum(1 for _ in handle) - 1, 0)

    return {
        "flim_path": str(flim_path),
        "tiff_path": str(tiff_path),
        "json_path": str(json_path),
        "run_dir": str(run_dir),
        "overlay_dir": str(overlay_dir) if overlay_dir else None,
        "spine_count": spine_count,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RESPAN pipeline from one .flim file.")
    parser.add_argument("--flim", type=Path, required=True, help="Input .flim path.")
    parser.add_argument("--channel", type=int, default=2, choices=[1, 2], help="FLIM channel.")
    parser.add_argument(
        "--run-parent",
        type=Path,
        default=None,
        help="Parent folder for RESPAN run outputs (default: <flim_dir>/respan_runs).",
    )
    parser.add_argument(
        "--export-subdir",
        default="deepd3_annotation_stacks",
        help="Subfolder under flim directory for exported TIFF/JSON.",
    )
    parser.add_argument("--rerun", action="store_true", help="Delete existing run folder first.")
    parser.add_argument("--overwrite-export", action="store_true", help="Re-export TIFF from FLIM.")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument(
        "--nnunet-fold",
        default="all",
        help="nnUNet fold: 'all' (fold_all checkpoint) or 0-4 if present in model dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_from_flim(
        args.flim,
        channel=args.channel,
        run_parent=args.run_parent,
        export_subdir=args.export_subdir,
        rerun=args.rerun,
        overwrite_export=args.overwrite_export,
        skip_analysis=args.skip_analysis,
        skip_overlays=args.skip_overlays,
        nnunet_fold=args.nnunet_fold,
    )
    print("\n=== Done ===")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    raise SystemExit(main())
