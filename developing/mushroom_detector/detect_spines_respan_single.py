# -*- coding: utf-8 -*-
"""Detect all RESPAN spines from a single .flim file (no mushroom morphology filter)."""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from respan_mushroom_core import (  # noqa: E402
    EDGE_EXCLUDE_PERCENT,
    detect_spines_from_flim_respan,
)

DEFAULT_FLIM = r"G:\ImagingData\Tetsuya\20260608\pos1_001.flim"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RESPAN-based spine detection (all spines, no mushroom filter)."
    )
    parser.add_argument("--flim", default=DEFAULT_FLIM, help="Input .flim path.")
    parser.add_argument("--channel", type=int, default=2, choices=[1, 2])
    parser.add_argument("--min-head-vol-um3", type=float, default=0.0)
    parser.add_argument(
        "--edge-exclude-percent",
        type=float,
        default=EDGE_EXCLUDE_PERCENT,
        help="Drop spines whose head lies in the outer margin (percent).",
    )
    parser.add_argument(
        "--dedupe-xy-sep-um",
        type=float,
        default=0.0,
        help="Optional minimum XY spacing between kept spines (0 = off).",
    )
    parser.add_argument("--rerun-respan", action="store_true")
    parser.add_argument(
        "--nnunet-fold",
        default="all",
        help="nnUNet fold: 'all' (fold_all) or 0-4 if checkpoint exists.",
    )
    parser.add_argument(
        "--full-per-spine-roi",
        action="store_true",
        help="Save per-spine ini/png/seg masks with shaft/BG geometry (slow).",
    )
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument("--no-class-overlay-pngs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    low_mag_mode = not args.full_per_spine_roi
    rows = detect_spines_from_flim_respan(
        args.flim,
        channel=args.channel,
        min_head_vol_um3=args.min_head_vol_um3,
        edge_exclude_percent=args.edge_exclude_percent,
        dedupe_xy_sep_um=args.dedupe_xy_sep_um,
        rerun_respan=args.rerun_respan,
        nnunet_fold=args.nnunet_fold,
        low_mag_mode=low_mag_mode,
        save_overview_pngs=not args.no_overview,
        save_class_overlay_pngs=not args.no_class_overlay_pngs,
        save_per_spine_png=args.full_per_spine_roi,
        save_per_spine_ini=args.full_per_spine_roi,
        save_per_spine_seg_tiff=args.full_per_spine_roi,
    )
    print(f"\ndone — {len(rows)} spine(s) in feature CSV")
    if args.full_per_spine_roi:
        for row in rows:
            print(
                f"  RESPAN spine {row['respan_spine_id']}: "
                f"shaft-to-head {row['shaft_to_head_um']:.3f} um, "
                f"ini={os.path.basename(row['ini_path']) if row.get('ini_path') else 'n/a'}"
            )
    else:
        print("  outputs: class_overlay_z_slices/, class_overlay_zproj/")
    print("=========== done ============")


if __name__ == "__main__":
    main()
