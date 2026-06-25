# -*- coding: utf-8 -*-
"""
Batch mushroom spine detection across multiple experiment folders.

Targets *highmag*_002.flim files, excluding for_align / for_aling prefixes.
"""
from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTROLFLIMAGE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
for _path in (_SCRIPT_DIR, _CONTROLFLIMAGE_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from detect_mushroom_folder import (  # noqa: E402
    DEEPD3_MODEL,
    detect_mushroom_in_folder,
    list_target_flim_files,
)

# --- edit here ---
TRAINING_FOLDERS = [
    r"G:\ImagingData\Tetsuya\20260530\auto1",
    r"G:\ImagingData\Tetsuya\20260528\auto1",
    r"G:\ImagingData\Tetsuya\20260526\auto1",
    r"G:\ImagingData\Tetsuya\20260510\auto1",
    r"G:\ImagingData\Tetsuya\20260429\auto1",
]

FLIM_GLOB = "*highmag*_002.flim"
EXCLUDE_NAME_PREFIXES = ("for_align", "for_aling")


def detect_training_batch(
    folders: list[str] | None = None,
    *,
    flim_glob: str = FLIM_GLOB,
    exclude_name_prefixes: tuple[str, ...] = EXCLUDE_NAME_PREFIXES,
) -> list[dict]:
    """Run mushroom detection on all target .flim files in each folder."""
    folders = folders or TRAINING_FOLDERS
    all_summary_rows: list[dict] = []

    for folder in folders:
        targets = list_target_flim_files(
            folder,
            flim_glob,
            exclude_name_prefixes=exclude_name_prefixes,
        )
        print(f"\n{'=' * 72}")
        print(f"Folder: {folder}")
        print(f"Target files: {len(targets)}")
        if not targets:
            print("  (skip — no matching .flim)")
            continue

        summary_rows = detect_mushroom_in_folder(
            folder,
            flim_glob,
            exclude_name_prefixes=exclude_name_prefixes,
            skip_existing=True,
        )
        all_summary_rows.extend(summary_rows)

    return all_summary_rows


def main() -> None:
    summary_rows = detect_training_batch()
    mushroom_rows = [row for row in summary_rows if row.get("deepd3_label") is not None]
    flim_paths = sorted({row["flim_path"] for row in mushroom_rows if row.get("flim_path")})

    print(f"\n{'=' * 72}")
    print(f"Batch done — {len(flim_paths)} file(s), {len(mushroom_rows)} mushroom spine(s) total")
    for flim_path in flim_paths:
        n = sum(1 for row in mushroom_rows if row["flim_path"] == flim_path)
        print(f"  {flim_path}: {n} mushroom(s)")


if __name__ == "__main__":
    main()
