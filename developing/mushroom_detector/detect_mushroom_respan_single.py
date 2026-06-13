# -*- coding: utf-8 -*-
"""
Detect mushroom spines from a single .flim file using RESPAN segmentation.

Filters by head-to-dendrite distance (mushroom morphology) and dedupes by XY
separation so kept spines are roughly 1-2 um apart.

Run with respan_gpu env, e.g.:
  C:\\Users\\yasudalab\\AppData\\Local\\miniconda3\\envs\\respan_gpu\\python.exe detect_mushroom_respan_single.py
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from respan_mushroom_core import (  # noqa: E402
    DEDUPE_MUSHROOM_XY_SEP_UM,
    MIN_SHAFT_TO_HEAD_UM,
    detect_mushroom_from_flim_respan,
)

# --- edit here ---
FLIM_PATH = (
    r"G:\ImagingData\Tetsuya\20260608\mushroom_1dend - Copy\pos1__highmag_1_002.flim"
)

MIN_SHAFT_TO_HEAD_UM_OVERRIDE = MIN_SHAFT_TO_HEAD_UM
DEDUPE_XY_SEP_UM_OVERRIDE = DEDUPE_MUSHROOM_XY_SEP_UM
SAVE_PER_SPINE_PNG = True
SAVE_PER_SPINE_INI = True
SAVE_PER_SPINE_SEG_TIFF = True
RERUN_RESPAN = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RESPAN-based mushroom spine detection.")
    parser.add_argument("--flim", default=FLIM_PATH, help="Input .flim path.")
    parser.add_argument("--channel", type=int, default=2, choices=[1, 2])
    parser.add_argument(
        "--min-shaft-to-head-um",
        type=float,
        default=MIN_SHAFT_TO_HEAD_UM_OVERRIDE,
        help="Minimum head-to-dendrite XY distance (um).",
    )
    parser.add_argument(
        "--dedupe-xy-sep-um",
        type=float,
        default=DEDUPE_XY_SEP_UM_OVERRIDE,
        help="Minimum XY spacing between kept mushrooms (um).",
    )
    parser.add_argument("--rerun-respan", action="store_true", default=RERUN_RESPAN)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--no-ini", action="store_true")
    parser.add_argument("--no-seg-tiff", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = detect_mushroom_from_flim_respan(
        args.flim,
        channel=args.channel,
        min_shaft_to_head_um=args.min_shaft_to_head_um,
        dedupe_mushroom_xy_sep_um=args.dedupe_xy_sep_um,
        rerun_respan=args.rerun_respan,
        save_per_spine_png=not args.no_png,
        save_per_spine_ini=SAVE_PER_SPINE_INI and not args.no_ini,
        save_per_spine_seg_tiff=SAVE_PER_SPINE_SEG_TIFF and not args.no_seg_tiff,
    )
    print(f"\ndone — {len(rows)} mushroom spine(s) saved")
    for row in rows:
        print(
            f"  RESPAN spine {row['respan_spine_id']}: "
            f"shaft-to-head {row['shaft_to_head_um']:.3f} um, "
            f"uncaging ({row['uncaging_x_pix']:.1f}, {row['uncaging_y_pix']:.1f}), "
            f"ini={os.path.basename(row['ini_path']) if row.get('ini_path') else 'n/a'}"
        )
    print("=========== done ============")


if __name__ == "__main__":
    main()
