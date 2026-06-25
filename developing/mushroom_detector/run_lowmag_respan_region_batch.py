# -*- coding: utf-8 -*-
"""Batch: RESPAN spine detect + spaced region pick + panel PNGs for low-mag *001.flim."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pick_respan_spaced_regions import pick_spaced_regions_from_respan  # noqa: E402
from respan_mushroom_core import (  # noqa: E402
    base_name_from_flim_path,
    detect_spines_from_flim_respan,
    respan_outputs_ready,
    savefolder_from_flim_path,
)

DEFAULT_FOLDER = r"G:\ImagingData\Tetsuya\20260530\copied_auto1"


def list_flim_files(folder: str, pattern: str) -> list[str]:
    paths = glob.glob(os.path.join(folder, pattern))
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch low-mag RESPAN detection + 12 spaced regions + panels."
    )
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    parser.add_argument("--glob", dest="filename_pattern", default="*001.flim")
    parser.add_argument("--channel", type=int, default=2, choices=[1, 2])
    parser.add_argument("--max-pos", type=int, default=12)
    parser.add_argument("--min-spacing-um", type=float, default=15.0)
    parser.add_argument("--cluster-eps-um", type=float, default=6.0)
    parser.add_argument("--min-spines-per-region", type=int, default=2)
    parser.add_argument("--rerun-respan", action="store_true")
    parser.add_argument("--skip-existing-respan", action="store_true")
    parser.add_argument("--force-regions", action="store_true")
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument(
        "--nnunet-fold",
        default="all",
        help="nnUNet fold: 'all' (fold_all) or 0-4 if checkpoint exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    flim_paths = list_flim_files(args.folder, args.filename_pattern)
    if not flim_paths:
        print(f"No files matching {args.filename_pattern} in {args.folder}")
        return

    log_path = Path(args.folder) / "lowmag_respan_region_batch_log.txt"
    print(f"Folder: {args.folder}")
    print(f"Found {len(flim_paths)} FLIM file(s)")
    t_batch = time.perf_counter()
    ok = 0
    failed = 0

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== batch start {datetime.now().isoformat()} ===\n")
        log.write(f"folder={args.folder} n={len(flim_paths)}\n")

        for idx, flim_path in enumerate(flim_paths, start=1):
            print(f"\n[{idx}/{len(flim_paths)}] {flim_path}")
            log.write(f"\n[{idx}/{len(flim_paths)}] {flim_path}\n")
            t0 = time.perf_counter()
            try:
                need_respan = args.rerun_respan or not respan_outputs_ready(Path(flim_path), args.channel)
                savefolder = savefolder_from_flim_path(flim_path)
                base_name = base_name_from_flim_path(flim_path)
                feature_csv = Path(savefolder) / f"{base_name}_respan_spine_features.csv"

                if need_respan or not (args.skip_existing_respan and feature_csv.is_file()):
                    print("  RESPAN spine detection...")
                    detect_spines_from_flim_respan(
                        flim_path,
                        channel=args.channel,
                        rerun_respan=args.rerun_respan,
                        nnunet_fold=args.nnunet_fold,
                        low_mag_mode=True,
                    )
                else:
                    print("  RESPAN spine detection: skip existing")

                print("  spaced regions + panels...")
                pick_spaced_regions_from_respan(
                    flim_path,
                    max_pos_cand_num=args.max_pos,
                    min_spacing_um=args.min_spacing_um,
                    cluster_eps_um=args.cluster_eps_um,
                    min_spines_per_region=args.min_spines_per_region,
                    skip_if_defined=not args.force_regions,
                    export_panels=not args.no_panels,
                )
                elapsed = time.perf_counter() - t0
                print(f"  OK in {elapsed:.1f} s")
                log.write(f"  OK {elapsed:.1f}s\n")
                ok += 1
            except Exception as exc:
                failed += 1
                print(f"  FAILED: {exc}")
                log.write(f"  FAILED: {exc}\n")
                log.write(traceback.format_exc() + "\n")

        total = time.perf_counter() - t_batch
        summary = f"batch done: ok={ok} failed={failed} total={total:.1f}s"
        print(f"\n=========== {summary} ============")
        log.write(summary + "\n")
        print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
