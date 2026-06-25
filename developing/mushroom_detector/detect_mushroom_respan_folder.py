# -*- coding: utf-8 -*-
"""Run RESPAN-based mushroom detection on all .flim files in a folder."""

from __future__ import annotations

import argparse
import glob
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from respan_mushroom_core import (  # noqa: E402
    DEDUPE_MUSHROOM_XY_SEP_UM,
    MIN_SHAFT_TO_HEAD_UM,
    base_name_from_flim_path,
    collect_respan_feature_rows_from_folder,
    detect_mushroom_from_flim_respan,
    save_mushroom_assign_summary_csv,
    savefolder_from_flim_path,
)

FLIM_FOLDER = r"G:\ImagingData\Tetsuya\20260608\mushroom_multi_dend - Copy"
FLIM_GLOB = "*.flim"
EXCLUDE_NAME_PREFIXES = ("for_align", "for_aling")


def _should_exclude_flim(filename: str, exclude_name_prefixes: tuple[str, ...]) -> bool:
    lower_name = os.path.basename(filename).lower()
    return any(lower_name.startswith(prefix.lower()) for prefix in exclude_name_prefixes)


def list_target_flim_files(
    folder: str,
    flim_glob: str = "*.flim",
    *,
    exclude_name_prefixes: tuple[str, ...] = EXCLUDE_NAME_PREFIXES,
) -> list[str]:
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")
    paths = glob.glob(os.path.join(folder, flim_glob))
    paths = [p for p in paths if not _should_exclude_flim(p, exclude_name_prefixes)]
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RESPAN mushroom detection for a folder.")
    parser.add_argument("--folder", default=FLIM_FOLDER)
    parser.add_argument("--glob", dest="flim_glob", default=FLIM_GLOB)
    parser.add_argument("--channel", type=int, default=2, choices=[1, 2])
    parser.add_argument("--min-shaft-to-head-um", type=float, default=MIN_SHAFT_TO_HEAD_UM)
    parser.add_argument("--dedupe-xy-sep-um", type=float, default=DEDUPE_MUSHROOM_XY_SEP_UM)
    parser.add_argument("--rerun-respan", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--no-ini", action="store_true")
    parser.add_argument("--no-seg-tiff", action="store_true")
    parser.add_argument("--no-z-triplets", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    flim_paths = list_target_flim_files(args.folder, args.flim_glob)
    if not flim_paths:
        print(f"No .flim files found in {args.folder}")
        return

    print(f"Folder: {args.folder}")
    print(f"Found {len(flim_paths)} .flim file(s)")
    all_rows: list[dict] = []

    for idx, flim_path in enumerate(flim_paths, start=1):
        base_name = base_name_from_flim_path(flim_path)
        savefolder = savefolder_from_flim_path(flim_path)
        csv_path = os.path.join(savefolder, f"{base_name}_respan_mushroom_features.csv")
        if args.skip_existing and os.path.isfile(csv_path):
            print(f"[{idx}/{len(flim_paths)}] skip existing: {os.path.basename(flim_path)}")
            import pandas as pd

            all_rows.extend(pd.read_csv(csv_path).to_dict(orient="records"))
            continue

        print(f"\n[{idx}/{len(flim_paths)}] {flim_path}")
        try:
            rows = detect_mushroom_from_flim_respan(
                flim_path,
                channel=args.channel,
                min_shaft_to_head_um=args.min_shaft_to_head_um,
                dedupe_mushroom_xy_sep_um=args.dedupe_xy_sep_um,
                rerun_respan=args.rerun_respan,
                save_per_spine_png=not args.no_png,
                save_per_spine_ini=not args.no_ini,
                save_per_spine_seg_tiff=not args.no_seg_tiff,
                save_z_triplets=not args.no_z_triplets,
            )
            all_rows.extend(rows)
            print(f"  -> {len(rows)} mushroom spine(s)")
        except Exception as exc:
            print(f"  FAILED: {exc}")

    summary_rows = collect_respan_feature_rows_from_folder(args.folder)
    summary_path = save_mushroom_assign_summary_csv(args.folder, summary_rows)
    print(f"\n=========== batch done: {len(all_rows)} mushroom spine(s) this run ============")
    if summary_rows:
        print(f"Folder summary: {len(summary_rows)} mushroom spine(s)")
    if summary_path:
        print(f"Rate spines: python rate_mushroom_spines_gui.py \"{args.folder}\"")


if __name__ == "__main__":
    main()
