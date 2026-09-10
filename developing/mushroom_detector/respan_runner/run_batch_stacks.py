"""Run RESPAN and Z-slice overlays for every TIFF stack in a folder."""

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
from respan_runner.paths import (
    clean_launcher_path,
    conda_root,
    nnunet_predict_bat,
    nnunet_trainer_dir,
    respan_model_dir,
    respan_nnunet_python,
    respan_repo_root,
)

ensure_respan_syspath()

from RESPAN.Environment import imgan, main as respan_main, sr  # noqa: E402

MODEL_DIR = respan_model_dir()
NNUNET_TRAINER_DIR = nnunet_trainer_dir()
CONDA_ROOT = conda_root()
NNUNET_PY = respan_nnunet_python()
NNUNET_PREDICT_BAT = nnunet_predict_bat()
CLEAN_LAUNCHER = clean_launcher_path()


def _resolve_nnunet_fold(requested: str | int) -> str:
    """Map fold id to an available checkpoint (this model ships fold_all only)."""
    requested_str = str(requested)
    if requested_str == "all":
        return "all"
    fold_ckpt = NNUNET_TRAINER_DIR / f"fold_{requested_str}" / "checkpoint_final.pth"
    if fold_ckpt.is_file():
        return requested_str
    fold_all_ckpt = NNUNET_TRAINER_DIR / "fold_all" / "checkpoint_final.pth"
    if fold_all_ckpt.is_file():
        return "all"
    return requested_str


def _run_subfolder_name(tiff_path: Path) -> str:
    from respan_runner.paths import run_subfolder_name_from_tiff

    return run_subfolder_name_from_tiff(tiff_path)


def _load_resolution(json_path: Path) -> tuple[float, float, float, float]:
    with json_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    input_xy = (float(meta["x_pixel_um"]) + float(meta["y_pixel_um"])) / 2.0
    input_z = float(meta["z_pixel_um"])
    model_xy = 0.102
    model_z = 1.0
    return input_xy, input_z, model_xy, model_z


def _write_analysis_settings(
    yaml_path: Path,
    input_xy: float,
    input_z: float,
    model_xy: float,
    model_z: float,
) -> None:
    template = respan_repo_root() / "Templates" / "Analysis_Settings.yaml"
    text = template.read_text(encoding="utf-8")
    text = text.replace("input_resXY: 0.102", f"input_resXY: {input_xy}")
    text = text.replace("input_resZ: 1", f"input_resZ: {input_z}")
    text = text.replace("model_resXY: 0.102", f"model_resXY: {model_xy}")
    text = text.replace("model_resZ: 1", f"model_resZ: {model_z}")
    text = text.replace("restore: True", "restore: False")
    yaml_path.write_text(text, encoding="utf-8")


def _configure_settings(settings, *, nnunet_fold: str | int = "all") -> None:
    settings.neuron_seg_model_path = str(MODEL_DIR)
    settings.nnunet_fold = _resolve_nnunet_fold(nnunet_fold)
    settings.nnunet_fold_requested = str(nnunet_fold)
    settings.nnUnet_type = "3d_fullres"
    settings.nnUnet_conda_path = str(CONDA_ROOT)
    settings.nnUnet_env_path = str(CONDA_ROOT / "envs")
    settings.nnUnet_env = "respan_nnunet"
    settings.internal_py_path = str(NNUNET_PY)
    settings.clean_launcher = str(CLEAN_LAUNCHER)
    settings.nnunet_predict_bat = str(NNUNET_PREDICT_BAT)
    settings.basepath = str(respan_repo_root() / "Scripts")

    settings.save_intermediate_data = True
    settings.save_val_data = True
    settings.dask_enabled = False
    settings.image_restore = False
    settings.axial_restore = False
    settings.neuron_channel = 1
    settings.analysis_method = "Whole Neuron"
    settings.Vaa3d = False
    settings.HistMatch = False
    settings.Track = False
    settings.reg_method = "Elastic"
    settings.neck_generation = True
    settings.recover_filopodia = False
    settings.keep_partial_spines = True
    settings.drop_spurious_necks = True
    settings.second_pass = False
    settings.patch_for_nnunet = False
    settings.model_type = 3
    settings.use_vox_measurements = False
    settings.resave_omezarr = False
    settings.additional_logging = True
    settings.additional_logging_dev = False
    settings.checkmem = True

    min_dendrite_vol_um = 1.0
    spine_vol_um = (0.03, 15.0)
    spine_dist_um = 4.0
    settings.min_dendrite_vol = round(
        min_dendrite_vol_um
        / settings.input_resXY
        / settings.input_resXY
        / settings.input_resZ,
        0,
    )
    settings.neuron_spine_size = [
        round(x / (settings.input_resXY * settings.input_resXY * settings.input_resZ), 0)
        for x in spine_vol_um
    ]
    settings.neuron_spine_dist = round(spine_dist_um / settings.input_resXY, 2)


def _build_logger(run_dir: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
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


def _setup_run_folder(
    tiff_path: Path,
    json_path: Path,
    run_parent: Path,
    rerun: bool,
) -> Path:
    run_dir = run_parent / _run_subfolder_name(tiff_path)
    if run_dir.exists():
        if rerun:
            shutil.rmtree(run_dir)
        else:
            return run_dir

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


def _analysis_complete(run_dir: Path, tiff_path: Path) -> bool:
    label_path = run_dir / "Validation_Data" / "Segmentation_Labels" / tiff_path.name
    table_glob = list((run_dir / "Tables").glob("*_detected_spines.csv"))
    return label_path.exists() and bool(table_glob)


def _overlays_complete(run_dir: Path, tiff_path: Path) -> bool:
    overlay_dir = (
        run_dir / "Validation_Data" / "Z_slice_overlays" / tiff_path.stem
    )
    montage = overlay_dir / f"{tiff_path.stem}_all_z_montage.png"
    return montage.exists()


def run_one_stack(
    tiff_path: Path,
    run_parent: Path,
    rerun: bool,
    skip_analysis: bool,
    skip_overlays: bool,
) -> tuple[str, bool, str]:
    json_path = tiff_path.with_suffix(".json")
    if not json_path.exists():
        return tiff_path.name, False, "missing JSON sidecar"

    run_dir = _setup_run_folder(tiff_path, json_path, run_parent, rerun=rerun)
    logger = _build_logger(run_dir, f"respan_{tiff_path.stem}")

    try:
        if not skip_analysis:
            if rerun or not _analysis_complete(run_dir, tiff_path):
                input_xy, input_z, model_xy, model_z = _load_resolution(json_path)
                logger.info("RESPAN batch run")
                logger.info(f"  input TIFF: {tiff_path}")
                logger.info(f"  run folder: {run_dir}")
                logger.info(f"  resolution: XY={input_xy:.6f} um, Z={input_z:.3f} um")

                settings, locations = respan_main.initialize_RESPAN(str(run_dir) + os.sep)
                _configure_settings(settings)

                log = sr.restore_and_segment(settings, locations, logger)
                if log != 0 or settings.Track:
                    return tiff_path.name, False, f"segmentation failed (code={log})"
                imgan.analyze_spines(settings, locations, log, logger)
                logger.info("RESPAN analysis complete.")
            else:
                logger.info(f"Skipping analysis (already complete): {tiff_path.name}")

        if not skip_overlays:
            if rerun or not _overlays_complete(run_dir, tiff_path):
                export_z_overlays(
                    run_dir=run_dir,
                    raw_tiff=tiff_path,
                    label_tiff=run_dir / "Validation_Data" / "Segmentation_Labels" / tiff_path.name,
                    spines_csv=run_dir / "Tables" / f"{tiff_path.stem}_detected_spines.csv",
                )
                logger.info("Z-slice overlays exported.")
            else:
                logger.info(f"Skipping overlays (already complete): {tiff_path.name}")

        return tiff_path.name, True, "ok"
    except Exception as exc:
        logger.exception(f"Failed on {tiff_path.name}")
        return tiff_path.name, False, str(exc)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch RESPAN analysis with Z overlays.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-parent", type=Path, required=True)
    parser.add_argument("--rerun", action="store_true", help="Delete and rerun completed stacks.")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-overlays", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")
    if not NNUNET_PREDICT_BAT.exists():
        raise FileNotFoundError(f"nnUNet launcher not found: {NNUNET_PREDICT_BAT}")

    tiff_files = sorted(args.input_dir.glob("*.tif"))
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files in {args.input_dir}")

    print(f"Found {len(tiff_files)} stacks in {args.input_dir}")
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for index, tiff_path in enumerate(tiff_files, start=1):
        print(f"\n[{index}/{len(tiff_files)}] {tiff_path.name}")
        name, ok, message = run_one_stack(
            tiff_path=tiff_path,
            run_parent=args.run_parent,
            rerun=args.rerun,
            skip_analysis=args.skip_analysis,
            skip_overlays=args.skip_overlays,
        )
        if ok:
            successes.append(name)
            print(f"  OK: {message}")
        else:
            failures.append((name, message))
            print(f"  FAIL: {message}")

    print("\n=== Batch summary ===")
    print(f"Success: {len(successes)}/{len(tiff_files)}")
    if failures:
        print("Failures:")
        for name, message in failures:
            print(f"  - {name}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    raise SystemExit(main())
