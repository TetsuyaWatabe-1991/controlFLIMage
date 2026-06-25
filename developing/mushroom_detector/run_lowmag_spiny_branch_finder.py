# -*- coding: utf-8 -*-
"""
Run texture-based spiny-branch ROI picker on low-mag .flim files.

Requires OpenCV (cv2) via utility.dendritic_shaft_detection, e.g. respan_nnunet env:
  C:\\Users\\yasudalab\\AppData\\Local\\miniconda3\\envs\\respan_nnunet\\python.exe run_lowmag_spiny_branch_finder.py --flim ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_CONTROLFLIMAGE = Path(__file__).resolve().parents[2]
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

from lowmag_mushroom_branch_finder import (  # noqa: E402
    find_and_save_spiny_branches,
    run_lowmag_mushroom_branch_finder,
)

DEFAULT_FLIM = r"G:\ImagingData\Tetsuya\20260608\pos1_001.flim"
DEFAULT_FOLDER = r"G:\ImagingData\Tetsuya\20260608"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick spaced spiny dendrite branches on low-mag FLIM (texture-based)."
    )
    parser.add_argument("--flim", default=None, help="Single .flim path.")
    parser.add_argument(
        "--folder",
        default=DEFAULT_FOLDER,
        help="Folder for batch mode (with --glob).",
    )
    parser.add_argument("--glob", dest="filename_pattern", default="*_001.flim")
    parser.add_argument("--max-pos", type=int, default=8, help="Max branch picks.")
    parser.add_argument("--min-spacing-um", type=float, default=20.0)
    parser.add_argument("--local-radius-um", type=float, default=10.0)
    parser.add_argument("--dendrite-percentile", type=float, default=94.0)
    parser.add_argument("--shaft-opening-um", type=float, default=1.2)
    parser.add_argument("--tube-radius-um", type=float, default=3.5)
    parser.add_argument("--max-shaft-thickness-um", type=float, default=3.0)
    parser.add_argument("--min-branch-score", type=float, default=0.0)
    parser.add_argument("--min-skeleton-length-pix", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()

    if args.flim:
        flim_path = args.flim
        if not os.path.isfile(flim_path):
            raise FileNotFoundError(f"FLIM not found: {flim_path}")
        find_and_save_spiny_branches(
            flim_path,
            max_pos_cand_num=args.max_pos,
            min_spacing_um=args.min_spacing_um,
            local_radius_um=args.local_radius_um,
            dendrite_percentile=args.dendrite_percentile,
            shaft_opening_um=args.shaft_opening_um,
            tube_radius_um=args.tube_radius_um,
            max_shaft_thickness_um=args.max_shaft_thickness_um,
            min_branch_score=args.min_branch_score,
            min_skeleton_length_pix=args.min_skeleton_length_pix,
            skip_if_defined=not args.force,
        )
    else:
        run_lowmag_mushroom_branch_finder(
            args.folder,
            filename_pattern=args.filename_pattern,
            max_pos_cand_num=args.max_pos,
            min_spacing_um=args.min_spacing_um,
            local_radius_um=args.local_radius_um,
            dendrite_percentile=args.dendrite_percentile,
            shaft_opening_um=args.shaft_opening_um,
            tube_radius_um=args.tube_radius_um,
            max_shaft_thickness_um=args.max_shaft_thickness_um,
            min_branch_score=args.min_branch_score,
            min_seconds_since_modification=0,
            skip_if_defined=not args.force,
        )

    elapsed = time.perf_counter() - t0
    print(f"\n=========== done in {elapsed:.2f} s ============")


if __name__ == "__main__":
    main()
