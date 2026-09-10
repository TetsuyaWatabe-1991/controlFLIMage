# -*- coding: utf-8 -*-
"""Remap absolute paths in combined_df when a pickle is opened on another computer."""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

sys.path.append(os.path.dirname(__file__))

from simple_dialog import ask_open_path_gui, ask_save_folder_gui, ask_yes_no_gui

_FILE_SUFFIXES = (".flim", ".tif", ".tiff", ".png", ".csv", ".pkl", ".ini", ".txt")
_NUMBERED_FLIM_RE = re.compile(r"^(.*)_\d{3}\.[^.]+$", re.IGNORECASE)
_MAX_DIALOG_TRIES = 5


def ensure_combined_df_paths_exist(
    combined_df: pd.DataFrame,
    *,
    df_save_path: str | None = None,
    anchors: Sequence[str] | None = None,
    save_if_changed: bool = True,
) -> tuple[pd.DataFrame | None, bool]:
    """
    If pickle path strings do not exist on this computer, remap prefixes.

    Tries pkl folder and provided anchors first; asks for a .flim or folder if needed.

    Returns (df, changed). If the user cancels, returns (None, False).
    """
    if combined_df is None or combined_df.empty:
        return combined_df, False

    missing = _missing_file_paths(combined_df)
    if not missing:
        return combined_df, False

    print("=" * 60)
    print("Some files stored in the pickle were not found on this computer.")
    print(f"  missing (showing up to 5 of {len(missing)}):")
    for p in missing[:5]:
        print(f"    {p}")
    print("=" * 60)

    mapping = _infer_prefix_map_from_anchors(combined_df, missing, df_save_path, anchors)
    if mapping is None:
        mapping = _infer_prefix_map_from_dialog(combined_df, missing, df_save_path)

    if mapping is None:
        print("Path remap cancelled. Cannot continue with missing files.")
        return None, False

    old_prefix, new_prefix = mapping
    remapped = _apply_prefix_map(combined_df, old_prefix, new_prefix)
    still_missing = _missing_file_paths(remapped)
    if still_missing:
        print("Path remap did not resolve all files. Example still missing:")
        for p in still_missing[:5]:
            print(f"    {p}")
        print("Try selecting a .flim that belongs to this experiment.")
        mapping = _infer_prefix_map_from_dialog(combined_df, missing, df_save_path)
        if mapping is None:
            print("Path remap cancelled. Cannot continue with missing files.")
            return None, False
        old_prefix, new_prefix = mapping
        remapped = _apply_prefix_map(combined_df, old_prefix, new_prefix)
        still_missing = _missing_file_paths(remapped)
        if still_missing:
            print("Path remap failed. Remaining missing files:")
            for p in still_missing[:8]:
                print(f"    {p}")
            return None, False

    print(f"Remapped pickle paths:\n  {old_prefix}\n  -> {new_prefix}")
    if save_if_changed and df_save_path:
        remapped.to_pickle(df_save_path)
        remapped.to_csv(df_save_path.replace(".pkl", ".csv"))
        print(f"Saved remapped combined_df: {df_save_path}")
    return remapped, True


def _path_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        name = str(col).lower()
        if "path" not in name and "filepath" not in name:
            continue
        if df[col].dtype == object or str(df[col].dtype) == "string":
            cols.append(col)
    return cols


def _is_path_str(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    try:
        if pd.isna(value):
            return False
    except (ValueError, TypeError):
        pass
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(text) and text.lower() not in ("nan", "none")


def _looks_like_file(path: str) -> bool:
    lower = path.lower().replace("/", "\\")
    return any(lower.endswith(suf) for suf in _FILE_SUFFIXES)


def _file_like_paths(df: pd.DataFrame) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    preferred = [c for c in ("file_path",) if c in df.columns]
    cols = preferred + [c for c in _path_columns(df) if c not in preferred]
    for col in cols:
        for value in df[col].tolist():
            if not _is_path_str(value):
                continue
            path = str(value)
            if not _looks_like_file(path):
                continue
            key = os.path.normcase(path.replace("/", "\\"))
            if key in seen:
                continue
            seen.add(key)
            values.append(path)
    return values


def _missing_file_paths(df: pd.DataFrame) -> list[str]:
    """Return stored paths that should exist, preferring the file_path column."""
    if "file_path" in df.columns:
        missing: list[str] = []
        seen: set[str] = set()
        for value in df["file_path"].tolist():
            if not _is_path_str(value):
                continue
            path = str(value)
            key = os.path.normcase(path.replace("/", "\\"))
            if key in seen:
                continue
            seen.add(key)
            if not os.path.exists(path):
                missing.append(path)
        return missing
    return [p for p in _file_like_paths(df) if not os.path.exists(p)]


def _path_parts(path: str) -> tuple[str, ...]:
    text = str(path).strip().replace("/", "\\")
    if not text:
        return ()
    return Path(text).parts


def _join_parts(parts: tuple[str, ...]) -> str:
    if not parts:
        return ""
    return str(Path(*parts))


def infer_prefix_map(old_path: str, new_path: str) -> tuple[str, str] | None:
    """Infer (old_prefix, new_prefix) from longest case-insensitive common suffix."""
    mapping = _prefix_map_from_common_suffix(old_path, new_path)
    if mapping is not None:
        return mapping
    old_dir = os.path.dirname(old_path)
    new_dir = os.path.dirname(new_path)
    if old_dir and new_dir and old_dir != old_path:
        return _prefix_map_from_common_suffix(old_dir, new_dir)
    return None


def _prefix_map_from_common_suffix(old_path: str, new_path: str) -> tuple[str, str] | None:
    old_parts = _path_parts(old_path)
    new_parts = _path_parts(new_path)
    if not old_parts or not new_parts:
        return None
    n_match = 0
    limit = min(len(old_parts), len(new_parts))
    while n_match < limit:
        if old_parts[-(n_match + 1)].lower() != new_parts[-(n_match + 1)].lower():
            break
        n_match += 1
    if n_match == 0:
        return None
    if n_match == len(old_parts) and n_match == len(new_parts):
        return None
    old_prefix = _join_parts(old_parts[:-n_match]) if n_match < len(old_parts) else ""
    new_prefix = _join_parts(new_parts[:-n_match]) if n_match < len(new_parts) else ""
    if not old_prefix or not new_prefix:
        return None
    if os.path.normcase(old_prefix) == os.path.normcase(new_prefix):
        return None
    return old_prefix, new_prefix


def _remap_one(value: object, old_prefix: str, new_prefix: str) -> object:
    if not _is_path_str(value):
        return value
    path = str(value)
    old_parts = _path_parts(old_prefix)
    new_parts = _path_parts(new_prefix)
    path_parts = _path_parts(path)
    n = len(old_parts)
    if n == 0 or len(path_parts) < n:
        return value
    if [p.lower() for p in path_parts[:n]] != [p.lower() for p in old_parts]:
        return value
    return _join_parts(new_parts + path_parts[n:])


def _apply_prefix_map(
    df: pd.DataFrame, old_prefix: str, new_prefix: str
) -> pd.DataFrame:
    out = df.copy()
    for col in _path_columns(out):
        out[col] = out[col].map(lambda v, o=old_prefix, n=new_prefix: _remap_one(v, o, n))
    return out


def _group_key(path: str) -> str:
    name = os.path.basename(str(path))
    match = _NUMBERED_FLIM_RE.match(name)
    if match:
        return match.group(1).lower()
    return ""


def _stored_path_matching_anchor(df: pd.DataFrame, anchor_file: str) -> str | None:
    basename = os.path.basename(anchor_file).lower()
    stored = _file_like_paths(df)
    for path in stored:
        if os.path.basename(path).lower() == basename:
            return path
    group = _group_key(anchor_file)
    if group:
        for path in stored:
            if _group_key(path) == group:
                return path
    return None


def _find_basename_under_folder(folder: str, basename: str) -> str | None:
    direct = os.path.join(folder, basename)
    if os.path.isfile(direct):
        return direct
    one_level = glob.glob(os.path.join(folder, "*", basename))
    if one_level:
        one_level.sort(key=len)
        return one_level[0]
    recursive = glob.glob(os.path.join(folder, "**", basename), recursive=True)
    if recursive:
        recursive.sort(key=len)
        return recursive[0]
    return None


def _candidate_new_path_from_folder(folder: str, missing: Sequence[str]) -> str | None:
    for old_path in missing:
        found = _find_basename_under_folder(folder, os.path.basename(old_path))
        if found:
            return found
    flims = [
        p
        for p in _file_like_paths_from_list(missing)
        if p.lower().endswith(".flim")
    ]
    if not flims:
        flims = list(missing)
    for old_path in flims:
        group = _group_key(old_path)
        if not group:
            continue
        pattern = os.path.join(folder, f"{group}_*.flim")
        hits = glob.glob(pattern)
        if not hits:
            hits = glob.glob(os.path.join(folder, "**", f"{group}_*.flim"), recursive=True)
        if hits:
            hits.sort(key=len)
            return hits[0]
    return None


def _file_like_paths_from_list(paths: Sequence[str]) -> list[str]:
    return [p for p in paths if _looks_like_file(p)]


def _try_mapping(
    old_path: str, new_path: str, df: pd.DataFrame
) -> tuple[str, str] | None:
    if not old_path or not new_path or not os.path.exists(new_path):
        return None
    mapping = infer_prefix_map(old_path, new_path)
    if mapping is None:
        return None
    old_prefix, new_prefix = mapping
    trial = _apply_prefix_map(df, old_prefix, new_prefix)
    if _missing_file_paths(trial):
        return None
    return mapping


def _iter_anchor_dirs(df_save_path: str | None, anchors: Sequence[str] | None) -> list[str]:
    dirs: list[str] = []
    seen: set[str] = set()

    def _add(path: str | None) -> None:
        if not path:
            return
        candidate = path
        if os.path.isfile(candidate):
            candidate = os.path.dirname(candidate)
        if not os.path.isdir(candidate):
            return
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            return
        seen.add(key)
        dirs.append(candidate)

    _add(df_save_path)
    if anchors:
        for anchor in anchors:
            _add(anchor)
    return dirs


def _infer_prefix_map_from_anchors(
    df: pd.DataFrame,
    missing: Sequence[str],
    df_save_path: str | None,
    anchors: Sequence[str] | None,
) -> tuple[str, str] | None:
    file_anchors: list[str] = []
    if anchors:
        file_anchors.extend([a for a in anchors if a and os.path.isfile(a)])

    for anchor_file in file_anchors:
        stored = _stored_path_matching_anchor(df, anchor_file)
        if stored is None:
            continue
        mapping = _try_mapping(stored, anchor_file, df)
        if mapping:
            print(f"Auto path remap from selected file:\n  {anchor_file}")
            return mapping

    for folder in _iter_anchor_dirs(df_save_path, anchors):
        new_path = _candidate_new_path_from_folder(folder, missing)
        if not new_path:
            continue
        stored = _stored_path_matching_anchor(df, new_path) or missing[0]
        mapping = _try_mapping(stored, new_path, df)
        if mapping:
            print(f"Auto path remap from folder:\n  {folder}")
            return mapping
    return None


def _infer_prefix_map_from_dialog(
    df: pd.DataFrame,
    missing: Sequence[str],
    df_save_path: str | None,
) -> tuple[str, str] | None:
    initial = ""
    if df_save_path:
        initial = os.path.dirname(df_save_path)

    for _ in range(_MAX_DIALOG_TRIES):
        pick_flim = ask_yes_no_gui(
            "Some files in the pickle were not found on this computer. "
            "Select a .flim file from this experiment? "
            "(No = select the experiment folder)"
        )
        new_path = None
        stored = None
        if pick_flim:
            picked = ask_open_path_gui(filetypes=[("FLIM files", "*.flim")])
            if picked and os.path.isfile(picked):
                new_path = picked
                stored = _stored_path_matching_anchor(df, picked)
                if stored is None:
                    print(
                        "Selected .flim name was not found in the pickle. "
                        "Trying folder-level match."
                    )
                    stored = missing[0] if missing else None
        else:
            folder = ask_save_folder_gui(default_folder=initial)
            if folder and os.path.isdir(folder):
                new_path = _candidate_new_path_from_folder(folder, missing)
                if new_path is None:
                    print(f"No matching .flim found under: {folder}")
                    continue
                stored = _stored_path_matching_anchor(df, new_path) or (
                    missing[0] if missing else None
                )

        if not new_path or not stored:
            print("No file selected for path remap.")
            if not ask_yes_no_gui("Try selecting a path again?"):
                return None
            continue

        mapping = infer_prefix_map(stored, new_path)
        if mapping is None:
            print("Could not infer a path prefix from the selected location.")
            if not ask_yes_no_gui("Try selecting a path again?"):
                return None
            continue

        old_prefix, new_prefix = mapping
        trial = _apply_prefix_map(df, old_prefix, new_prefix)
        still_missing = _missing_file_paths(trial)
        if still_missing:
            print("Selected location did not make the pickle paths valid. Still missing:")
            for p in still_missing[:5]:
                print(f"    {p}")
            if not ask_yes_no_gui("Try selecting a different .flim or folder?"):
                return None
            continue
        return mapping
    return None
