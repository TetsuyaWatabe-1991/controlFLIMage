"""Segment every TIFF in one folder with a single RESPAN process.

Invoke with the respan_gpu Python. The model is loaded once and then applied
to every stack in the folder.

Usage:
    python run_respan_folder.py --input-dir DIR --run-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_MUSHROOM = Path(__file__).resolve().parents[2] / "developing" / "mushroom_detector"
if str(_MUSHROOM) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM))

from respan_runner.run_batch_stacks import (  # noqa: E402
    _build_logger,
    _configure_settings,
    _load_resolution,
    _write_analysis_settings,
    imgan,
    respan_main,
    sr,
)


def _link_or_copy(source: Path, dest: Path) -> None:
    if dest.exists():
        dest.unlink()
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def prepare_run_dir(input_dir: Path, run_dir: Path) -> list[str]:
    """Copy stacks into one RESPAN run folder and write Analysis_Settings.yaml."""
    meta_path = input_dir / "batch_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    names: list[str] = []
    for tiff_path in sorted(input_dir.glob("*.tif")):
        _link_or_copy(tiff_path, run_dir / tiff_path.name)
        names.append(tiff_path.name)
    sidecar = run_dir / "_resolution.json"
    sidecar.write_text(
        json.dumps(
            {
                "x_pixel_um": meta["x_pixel_um"],
                "y_pixel_um": meta["y_pixel_um"],
                "z_pixel_um": meta["z_pixel_um"],
            }
        ),
        encoding="utf-8",
    )
    input_xy, input_z, model_xy, model_z = _load_resolution(sidecar)
    sidecar.unlink()
    _write_analysis_settings(run_dir / "Analysis_Settings.yaml", input_xy, input_z, model_xy, model_z)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RESPAN once on a folder of TIFFs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    names = prepare_run_dir(args.input_dir, args.run_dir)
    print(f"stacks {len(names)}", flush=True)
    logger = _build_logger(args.run_dir, "respan_cnt_batch")
    settings, locations = respan_main.initialize_RESPAN(str(args.run_dir) + os.sep)
    _configure_settings(settings)
    log = sr.restore_and_segment(settings, locations, logger)
    if log != 0 or settings.Track:
        print(f"segmentation failed code={log}", flush=True)
        return 1
    imgan.analyze_spines(settings, locations, log, logger)
    print("RESPAN folder complete", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
