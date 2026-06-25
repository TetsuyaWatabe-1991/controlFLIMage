# -*- coding: utf-8 -*-
"""Build folder-level mushroom_spine_assign_summary.csv from RESPAN feature CSVs."""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from respan_mushroom_core import (  # noqa: E402
    collect_respan_feature_rows_from_folder,
    save_mushroom_assign_summary_csv,
)

DEFAULT_FOLDER = r"G:\ImagingData\Tetsuya\20260608\mushroom_multi_dend - Copy"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-FLIM RESPAN feature CSVs into one summary CSV."
    )
    parser.add_argument("folder", nargs="?", default=DEFAULT_FOLDER)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = collect_respan_feature_rows_from_folder(args.folder)
    if not rows:
        print(f"No *_respan_mushroom_features.csv found under {args.folder}")
        return
    save_mushroom_assign_summary_csv(args.folder, rows)
    print(f"Rate spines: python rate_mushroom_spines_gui.py \"{args.folder}\"")


if __name__ == "__main__":
    main()
