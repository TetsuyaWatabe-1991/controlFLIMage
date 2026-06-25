# -*- coding: utf-8 -*-
"""RESPAN mushroom detection batch for manual 1-4 rating (training export)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from detect_mushroom_respan_folder import list_target_flim_files  # noqa: E402
from respan_mushroom_core import (  # noqa: E402
    DEDUPE_MUSHROOM_XY_SEP_UM,
    MIN_SHAFT_TO_HEAD_UM,
    base_name_from_flim_path,
    collect_respan_feature_rows_from_folder,
    detect_mushroom_from_flim_respan,
    save_mushroom_assign_summary_csv,
    savefolder_from_flim_path,
)

DEFAULT_FOLDER = r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy"
DEFAULT_FLIM_GLOB = "*highmag*002.flim"
EXCLUDE_NAME_PREFIXES = ("for_align", "for_aling")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RESPAN mushroom batch for rating export (highmag 002 FLIMs)."
    )
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    parser.add_argument("--glob", dest="flim_glob", default=DEFAULT_FLIM_GLOB)
    parser.add_argument("--channel", type=int, default=2, choices=[1, 2])
    parser.add_argument("--min-shaft-to-head-um", type=float, default=MIN_SHAFT_TO_HEAD_UM)
    parser.add_argument("--dedupe-xy-sep-um", type=float, default=DEDUPE_MUSHROOM_XY_SEP_UM)
    parser.add_argument("--rerun-respan", action="store_true")
    parser.add_argument("--force-detect", action="store_true", help="Re-run even if features CSV exists.")
    parser.add_argument("--no-z-triplets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    flim_paths = list_target_flim_files(
        args.folder,
        args.flim_glob,
        exclude_name_prefixes=EXCLUDE_NAME_PREFIXES,
    )
    if not flim_paths:
        print(f"No matching FLIM in {args.folder} ({args.flim_glob})")
        return

    print(f"Folder: {args.folder}")
    print(f"Glob: {args.flim_glob}")
    print(f"Found {len(flim_paths)} FLIM file(s)")
    ok = 0
    failed = 0
    n_spines = 0

    for idx, flim_path in enumerate(flim_paths, start=1):
        base_name = base_name_from_flim_path(flim_path)
        savefolder = savefolder_from_flim_path(flim_path)
        csv_path = Path(savefolder) / f"{base_name}_respan_mushroom_features.csv"
        if csv_path.is_file() and not args.force_detect:
            print(f"[{idx}/{len(flim_paths)}] skip existing: {Path(flim_path).name}")
            ok += 1
            continue

        print(f"\n[{idx}/{len(flim_paths)}] {flim_path}")
        try:
            rows = detect_mushroom_from_flim_respan(
                flim_path,
                channel=args.channel,
                min_shaft_to_head_um=args.min_shaft_to_head_um,
                dedupe_mushroom_xy_sep_um=args.dedupe_xy_sep_um,
                rerun_respan=args.rerun_respan,
                save_z_triplets=not args.no_z_triplets,
            )
            n_spines += len(rows)
            ok += 1
            print(f"  -> {len(rows)} mushroom spine(s)")
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}")

    summary_rows = collect_respan_feature_rows_from_folder(args.folder)
    summary_path = save_mushroom_assign_summary_csv(args.folder, summary_rows)
    print(f"\n=========== done: ok={ok} failed={failed} mushrooms={len(summary_rows)} ============")
    if summary_path:
        print(f"Summary: {summary_path}")
    print(
        f"Rate spines:\n"
        f'  python rate_mushroom_spines_gui.py "{args.folder}"'
    )


if __name__ == "__main__":
    main()
