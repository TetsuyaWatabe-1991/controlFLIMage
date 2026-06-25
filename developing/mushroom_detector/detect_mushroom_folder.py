# -*- coding: utf-8 -*-
"""
Run mushroom spine detection on all .flim files in a folder.

Uses controlFLIMage/deepd3_mushroom_spine_assign_save.py.
"""
import os
import sys

_CONTROLFLIMAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _CONTROLFLIMAGE_DIR not in sys.path:
    sys.path.insert(0, _CONTROLFLIMAGE_DIR)

from deepd3_mushroom_spine_assign_save import (  # noqa: E402
    DEEPD3_MODEL_PATH,
    MIN_SHAFT_TO_HEAD_UM,
    base_name_from_flim_path,
    loop_mushroom_spine_assign_save,
    savefolder_from_flim_path,
)

# --- edit here ---
FLIM_FOLDER = r"G:\ImagingData\Tetsuya\20260608\mushroom_multi_dend"
FLIM_GLOB = "*.flim"
EXCLUDE_NAME_PREFIXES = ("for_align", "for_aling")

DEEPD3_MODEL = (
    r"C:\Users\yasudalab\Documents\Tetsuya_GIT\ongoing\deepd3\DeepD3_8F.h5"
)
MIN_SHAFT_TO_HEAD_UM_OVERRIDE = MIN_SHAFT_TO_HEAD_UM
SAVE_PER_SPINE_PNG = True
SAVE_PER_SPINE_INI = False


def _should_exclude_flim(filename: str, exclude_name_prefixes: tuple[str, ...]) -> bool:
    lower_name = os.path.basename(filename).lower()
    return any(lower_name.startswith(prefix.lower()) for prefix in exclude_name_prefixes)


def list_target_flim_files(
    folder: str,
    flim_glob: str = "*.flim",
    *,
    exclude_name_prefixes: tuple[str, ...] = EXCLUDE_NAME_PREFIXES,
) -> list[str]:
    """List matching .flim files in folder after name-prefix exclusions."""
    import glob

    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")

    paths = glob.glob(os.path.join(folder, flim_glob))
    paths = [
        path for path in paths
        if not _should_exclude_flim(path, exclude_name_prefixes)
    ]
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


def mushroom_features_csv_path(flim_path: str) -> str:
    savefolder = savefolder_from_flim_path(flim_path)
    base_name = base_name_from_flim_path(flim_path)
    return os.path.join(savefolder, f"{base_name}_mushroom_features.csv")


def detect_mushroom_in_folder(
    folder: str,
    flim_glob: str = "*.flim",
    *,
    exclude_name_prefixes: tuple[str, ...] = EXCLUDE_NAME_PREFIXES,
    min_shaft_to_head_um: float = MIN_SHAFT_TO_HEAD_UM,
    deepd3_model_path: str | None = None,
    save_per_spine_png: bool = SAVE_PER_SPINE_PNG,
    save_per_spine_ini: bool = SAVE_PER_SPINE_INI,
    skip_existing: bool = False,
):
    """Run mushroom detection on all matching .flim files in folder."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")

    target_files = list_target_flim_files(
        folder,
        flim_glob,
        exclude_name_prefixes=exclude_name_prefixes,
    )
    if not target_files:
        print("No matching .flim files after filters.")
        return []

    model_path = deepd3_model_path or DEEPD3_MODEL_PATH
    if not os.path.isfile(model_path):
        if os.path.isfile(DEEPD3_MODEL):
            print(f"DeepD3 model not found at {model_path!r}; using {DEEPD3_MODEL!r}")
            model_path = DEEPD3_MODEL
        else:
            raise FileNotFoundError(f"DeepD3 model not found: {model_path!r}")

    print("Folder:", folder)
    print("Glob:", flim_glob)
    print("Exclude prefixes:", exclude_name_prefixes)
    print(f"Matched files: {len(target_files)}")
    print(f"Mushroom threshold: shaft-to-head XY >= {min_shaft_to_head_um} um")

    all_rows: list[dict] = []
    skipped = 0
    for flim_path in target_files:
        if skip_existing and os.path.isfile(mushroom_features_csv_path(flim_path)):
            print(f"Skip (already detected): {flim_path}")
            skipped += 1
            continue
        filename = os.path.basename(flim_path)
        rows = loop_mushroom_spine_assign_save(
            highmag_folder=folder,
            highmag_filename=filename,
            exclude_ini_saved=False,
            from_latest_file=False,
            min_shaft_to_head_um=min_shaft_to_head_um,
            min_seconds_since_modification=0,
            deepd3_model_path=model_path,
            save_per_spine_ini=save_per_spine_ini,
            save_per_spine_png=save_per_spine_png,
        )
        all_rows.extend(rows)
    if skip_existing and skipped:
        print(f"Skipped {skipped} file(s) with existing features CSV.")
    return all_rows


def main():
    summary_rows = detect_mushroom_in_folder(
        FLIM_FOLDER,
        FLIM_GLOB,
        min_shaft_to_head_um=MIN_SHAFT_TO_HEAD_UM_OVERRIDE,
    )
    mushroom_rows = [row for row in summary_rows if row.get("deepd3_label") is not None]
    flim_paths = sorted({row["flim_path"] for row in mushroom_rows if row.get("flim_path")})
    print(f"\ndone — {len(flim_paths)} file(s), {len(mushroom_rows)} mushroom spine(s) total")
    for flim_path in flim_paths:
        n = sum(1 for row in mushroom_rows if row["flim_path"] == flim_path)
        print(f"  {os.path.basename(flim_path)}: {n} mushroom(s)")


if __name__ == "__main__":
    main()
