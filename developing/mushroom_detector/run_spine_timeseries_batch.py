# -*- coding: utf-8 -*-
"""Batch ROI detection, alignment, quantification, and plotting for highmag 002 FLIMs."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from respan_mushroom_core import (  # noqa: E402
    DEDUPE_MUSHROOM_XY_SEP_UM,
    MIN_SHAFT_TO_HEAD_UM,
    base_name_from_flim_path,
    detect_mushroom_from_flim_respan,
    savefolder_from_flim_path,
)
from run_spine_timeseries_respan import run_spine_timeseries  # noqa: E402

DEFAULT_ROOT = r"G:\ImagingData\Tetsuya\20260608"
DEFAULT_FLIM_GLOB = "*highmag*002.flim"
EXCLUDE_NAME_PREFIXES = ("for_align", "for_aling")


def _should_exclude_flim(filename: str, exclude_name_prefixes: tuple[str, ...]) -> bool:
    lower_name = os.path.basename(filename).lower()
    return any(lower_name.startswith(prefix.lower()) for prefix in exclude_name_prefixes)


def list_reference_flim_files(
    root: str,
    flim_glob: str = DEFAULT_FLIM_GLOB,
    *,
    exclude_name_prefixes: tuple[str, ...] = EXCLUDE_NAME_PREFIXES,
    exclude_dir_substrings: tuple[str, ...] = (),
) -> list[str]:
    pattern = os.path.join(root, "**", flim_glob)
    paths = glob.glob(pattern, recursive=True)
    filtered: list[str] = []
    for path in paths:
        if _should_exclude_flim(path, exclude_name_prefixes):
            continue
        norm = path.replace("\\", "/").lower()
        if any(sub.lower() in norm for sub in exclude_dir_substrings):
            continue
        filtered.append(path)
    return sorted(filtered, key=lambda p: p.lower())


def _has_spine_outline_masks(savefolder: Path, base_name: str) -> bool:
    seg_dir = savefolder / "seg_masks"
    if not seg_dir.is_dir():
        return False
    return any(seg_dir.glob(f"{base_name}_*_spine_outline_mask.tif"))


def _detection_complete(flim_path: str) -> bool:
    savefolder = Path(savefolder_from_flim_path(flim_path))
    base_name = base_name_from_flim_path(flim_path)
    feature_csv = savefolder / f"{base_name}_respan_mushroom_features.csv"
    return feature_csv.is_file() and _has_spine_outline_masks(savefolder, base_name)


def _timeseries_complete(flim_path: str, output_subdir: str) -> bool:
    savefolder = Path(savefolder_from_flim_path(flim_path))
    ts_csv = savefolder / output_subdir / "spine_timeseries.csv"
    if not ts_csv.is_file() or ts_csv.stat().st_size == 0:
        return False
    overlay_local = savefolder / output_subdir / "zproj_overlays_local"
    if not overlay_local.is_dir():
        return False
    return any(overlay_local.rglob("quant_timeseries_norm_p10.png"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch mushroom ROI + spine time-series pipeline for highmag 002 FLIMs."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--glob", dest="flim_glob", default=DEFAULT_FLIM_GLOB)
    parser.add_argument("--channel", type=int, default=2, choices=[1, 2])
    parser.add_argument("--min-shaft-to-head-um", type=float, default=MIN_SHAFT_TO_HEAD_UM)
    parser.add_argument("--dedupe-xy-sep-um", type=float, default=DEDUPE_MUSHROOM_XY_SEP_UM)
    parser.add_argument("--z-half-window", type=int, default=2)
    parser.add_argument("--small-region-size", type=int, default=60)
    parser.add_argument("--no-local-align", action="store_true")
    parser.add_argument("--output-subdir", default="spine_timeseries")
    parser.add_argument("--skip-existing-detection", action="store_true")
    parser.add_argument("--skip-existing-timeseries", action="store_true")
    parser.add_argument(
        "--force-detect",
        action="store_true",
        help="Regenerate mushroom ROI/masks (e.g. after dilation change); does not rerun RESPAN.",
    )
    parser.add_argument(
        "--rerun-respan",
        action="store_true",
        help="Also rerun full RESPAN segmentation (slow; use with --force-detect).",
    )
    parser.add_argument(
        "--exclude-dir-substrings",
        nargs="*",
        default=(),
        help="Skip paths whose normalized path contains any of these substrings.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files (0 = all).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    flim_paths = list_reference_flim_files(
        args.root,
        args.flim_glob,
        exclude_dir_substrings=tuple(args.exclude_dir_substrings),
    )
    if args.limit > 0:
        flim_paths = flim_paths[: args.limit]

    print(f"Root: {args.root}")
    print(f"Found {len(flim_paths)} reference FLIM(s)")
    if args.dry_run:
        for path in flim_paths:
            print(path)
        return

    log_path = Path(args.root) / "spine_timeseries_batch_log.txt"
    ok = 0
    failed = 0
    skipped = 0

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== batch start {datetime.now().isoformat()} ===\n")
        log.write(f"root={args.root} n={len(flim_paths)}\n")

        for idx, flim_path in enumerate(flim_paths, start=1):
            base_name = base_name_from_flim_path(flim_path)
            print(f"\n[{idx}/{len(flim_paths)}] {flim_path}")
            log.write(f"\n[{idx}/{len(flim_paths)}] {flim_path}\n")

            try:
                need_detect = args.force_detect or not _detection_complete(flim_path)
                if need_detect:
                    if args.skip_existing_detection and _detection_complete(flim_path):
                        print("  detection: skip existing")
                    else:
                        print("  detection: running RESPAN mushroom ROI...")
                        rows = detect_mushroom_from_flim_respan(
                            flim_path,
                            channel=args.channel,
                            min_shaft_to_head_um=args.min_shaft_to_head_um,
                            dedupe_mushroom_xy_sep_um=args.dedupe_xy_sep_um,
                            rerun_respan=args.rerun_respan,
                        )
                        print(f"  detection: {len(rows)} spine(s)")
                        if not rows:
                            msg = "  detection: no mushroom spines; skipping time series"
                            print(msg)
                            log.write(msg + "\n")
                            skipped += 1
                            continue
                else:
                    print("  detection: already complete")

                if args.skip_existing_timeseries and _timeseries_complete(flim_path, args.output_subdir):
                    print("  time series: skip existing")
                    log.write("  time series: skip existing\n")
                    skipped += 1
                    continue

                print("  time series: align + quantify + plot...")
                run_spine_timeseries(
                    flim_path,
                    channel=args.channel,
                    z_half_window=args.z_half_window,
                    small_region_size=args.small_region_size,
                    local_align=not args.no_local_align,
                    output_subdir=args.output_subdir,
                )
                print("  time series: done")
                log.write("  OK\n")
                ok += 1
            except Exception as exc:
                failed += 1
                err = f"  FAILED: {exc}"
                print(err)
                log.write(err + "\n")
                log.write(traceback.format_exc() + "\n")

        summary = f"batch done: ok={ok} skipped={skipped} failed={failed}"
        print(f"\n=========== {summary} ============")
        log.write(summary + "\n")
        print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
