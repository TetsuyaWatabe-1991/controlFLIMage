# -*- coding: utf-8 -*-
"""
Respan highmag ROI workflow: pre-defined Spine / Shaft / Background masks from seg_masks.

Uses the same alignment policy as tpem_low_high_spine_multi_merged_titrate_uncaging_pow_respan.py:
  - Global pre/post: roi_adjacent (FLIMageAlignment POST_ACQUISITION_ALIGN_METHOD)
  - Local crop: adjacent-frame roi_adjacent stored as local_shift_y/x (not added
    onto shift_y/x). FLIM raw masks use integer before/after TIFF shifts.

Does not modify gui_roi_fast_simple.py or analayze_all_flim_roi_gui2.py.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import tifffile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from FLIMageAlignment import (  # noqa: E402
    POST_ACQUISITION_ALIGN_METHOD,
    Align_4d_array,
    flim_files_to_nparray,
)
from gui_integration import first_processing_for_flim_files  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
from file_selection_gui_tiff_only import launch_file_selection_gui_tiff_only  # noqa: E402
from gui_roi_fast_simple import (  # noqa: E402
    BACKGROUND_MODE_MIP_P20,
    ROI_MASK_RAW_SUFFIX,
    ROI_TYPES,
    print_roi_analysis_errors,
    quantify_intensity_from_flim,
    rebuild_tiff_full_size_for_roi,
    record_roi_error,
    save_drift_corrected_roi_masks,
    save_roi_analysis_error_log,
)
from simple_dialog import ask_open_path_gui, ask_yes_no_gui  # noqa: E402
from combined_df_path_remap import ensure_combined_df_paths_exist  # noqa: E402

DEFAULT_COMBINED_DF_NAME = "combined_df_respan.pkl"

# Explicit alignment policy (matches live respan titration script).
GLOBAL_ALIGN_METHOD = POST_ACQUISITION_ALIGN_METHOD  # "roi_adjacent"
LOCAL_ALIGN_MODE = "adjacent"  # respan_spine_quant.LocalAlignMode.ADJACENT

SEG_MASK_SUBDIR = "seg_masks"
# ROIs drawn / pre-filled in this workflow. Background is not an ROI here: it is the
# 20th percentile of the quantified image (BACKGROUND_MODE_MIP_P20).
RESPAN_ROI_TYPES = ["Spine", "DendriticShaft"]
SEG_MASK_FILES = {
    "Spine": "{stem}_spine_outline_mask.tif",
    "DendriticShaft": "{stem}_shaft_fit_radius_mask.tif",
    "Background": "{stem}_bg_mask.tif",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASI_CONTROLLER = _REPO_ROOT / "ongoing" / "ASIcontroller"
if str(_ASI_CONTROLLER) not in sys.path:
    sys.path.insert(0, str(_ASI_CONTROLLER))

from respan_uncaging_log import parse_uncaging_records  # noqa: E402


def load_and_align_data_explicit(
    filelist: list[str],
    ch: int,
    *,
    align_method: str = GLOBAL_ALIGN_METHOD,
    fast_mode: bool = False,
    intensity_cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Load FLIM files and align with an explicit FLIMageAlignment method."""
    if fast_mode:
        from flim_fast_io import flim_files_to_nparray_fast

        tiff_multi, iminfo, relative_sec_list = flim_files_to_nparray_fast(
            filelist, ch=ch, intensity_cache=intensity_cache
        )
        shifts, aligned = Align_4d_array(
            tiff_multi,
            iminfo=iminfo,
            method=align_method,
            upsample_factor=1,
            apply_shifts=False,
        )
        return aligned, shifts, relative_sec_list
    tiff_multi, iminfo, relative_sec_list = flim_files_to_nparray(
        filelist, ch=ch, normalize_by_averageNum=True
    )
    shifts, aligned = Align_4d_array(
        tiff_multi, iminfo=iminfo, method=align_method
    )
    return aligned, shifts, relative_sec_list


@contextmanager
def _patch_load_and_align(
    align_method: str = GLOBAL_ALIGN_METHOD,
    *,
    fast_mode: bool = False,
    intensity_cache: dict[str, np.ndarray] | None = None,
) -> Iterator[None]:
    """Temporarily override gui_integration.load_and_align_data (no permanent edits)."""
    import gui_integration as gi

    original = gi.load_and_align_data

    def _wrapped(filelist, ch):
        return load_and_align_data_explicit(
            filelist,
            ch,
            align_method=align_method,
            fast_mode=fast_mode,
            intensity_cache=intensity_cache,
        )

    gi.load_and_align_data = _wrapped
    original_grfs = None
    if fast_mode:
        import gui_roi_fast_simple as grfs

        original_grfs = grfs.load_and_align_data
        grfs.load_and_align_data = _wrapped
    try:
        yield
    finally:
        gi.load_and_align_data = original
        if original_grfs is not None:
            import gui_roi_fast_simple as grfs

            grfs.load_and_align_data = original_grfs


def highmag_savefolder_from_filepath_without_number(filepath_without_number: str) -> str:
    """e.g. .../auto3/pos3__highmag_1_ -> .../auto3/pos3__highmag_1"""
    folder = os.path.dirname(filepath_without_number)
    stem = os.path.basename(filepath_without_number).rstrip("_")
    return os.path.join(folder, stem)


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def match_uncaging_record_for_set(
    each_set_df: pd.DataFrame,
    records: list,
) -> object | None:
    """Match one titration set to an uncaged_spines.txt entry via last pre FLIM path."""
    pre_df = each_set_df[each_set_df["phase"] == "pre"].sort_values("nth_omit_induction")
    if len(pre_df) == 0:
        return None
    last_pre = str(pre_df.iloc[-1]["file_path"])
    target_base = os.path.basename(last_pre).lower()
    basename_hit = None
    try:
        target = _norm_path(last_pre)
    except (OSError, ValueError):
        target = None
    for rec in records:
        rec_path = str(rec.flim_path)
        if target is not None:
            try:
                if _norm_path(rec_path) == target:
                    return rec
            except (OSError, ValueError):
                pass
        if os.path.basename(rec_path).lower() == target_base:
            if basename_hit is None:
                basename_hit = rec
    return basename_hit


def seg_mask_paths(highmag_folder: str, spine_stem: str) -> dict[str, Path]:
    """Return existing seg_mask paths keyed by ROI_TYPES name."""
    seg_dir = Path(highmag_folder) / SEG_MASK_SUBDIR
    out: dict[str, Path] = {}
    for roi_type, pattern in SEG_MASK_FILES.items():
        path = seg_dir / pattern.format(stem=spine_stem)
        if path.is_file():
            out[roi_type] = path
    return out


def _load_mask_2d(path: Path) -> np.ndarray:
    mask = tifffile.imread(str(path))
    return np.asarray(mask > 0, dtype=np.uint8)


def _crop_2d(frame: np.ndarray, center_yx: tuple[float, float], half: int) -> np.ndarray:
    h, w = frame.shape
    cy, cx = int(round(center_yx[0])), int(round(center_yx[1]))
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    return np.asarray(frame[y0:y1, x0:x1], dtype=np.float32)


def adjacent_local_shifts_yx(
    frames: list[np.ndarray],
    center_yx: tuple[float, float],
    *,
    half_size: int = 60,
    align_method: str = GLOBAL_ALIGN_METHOD,
    fast_mode: bool = False,
) -> list[tuple[float, float]]:
    """
    Adjacent-frame local roi_adjacent shifts on a spine-centered crop.
    Returns cumulative (shift_y, shift_x) per frame (first frame = 0).
    """
    if not frames:
        return []
    cumulative = np.zeros(3, dtype=np.float64)
    out: list[tuple[float, float]] = [(0.0, 0.0)]
    prev_crop = _crop_2d(frames[0], center_yx, half_size)
    roi_cy, roi_cx = prev_crop.shape[0] // 2, prev_crop.shape[1] // 2
    align_kwargs = {}
    if fast_mode:
        align_kwargs = {"upsample_factor": 1, "apply_shifts": False}

    for i in range(1, len(frames)):
        crop = _crop_2d(frames[i], center_yx, half_size)
        pair_4d = np.stack([prev_crop, crop], axis=0)[:, np.newaxis, :, :]
        shifts, _ = Align_4d_array(
            pair_4d,
            method=align_method,
            roi_center_zyx=(0, roi_cy, roi_cx),
            **align_kwargs,
        )
        cumulative += np.asarray(shifts[-1], dtype=np.float64)
        out.append((float(cumulative[1]), float(cumulative[2])))
        prev_crop = crop
    return out


def augment_frame_info_local_adjacent(
    combined_df: pd.DataFrame,
    *,
    ch_1or2: int = 2,
    z_plus_minus: int = 2,
    local_half_size: int = 60,
    local_align_mode: str = LOCAL_ALIGN_MODE,
    fast_mode: bool = False,
    intensity_cache: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Write adjacent local shifts into local_shift_y / local_shift_x on each
    *_frame_info.csv. Does not add them onto shift_y / shift_x (those stay
    the TIFF-build / global values used for FLIM raw-mask mapping).

    If an older file already mixed local into shift_y/x (local_align_mode set
    but no local_shift columns), subtract the newly computed local to restore
    the TIFF shifts.

    Skipped when local_align_mode is not 'adjacent'.
    """
    if local_align_mode != "adjacent":
        return combined_df

    import gui_roi_fast_simple as grfs

    for filepath_wo in combined_df["filepath_without_number"].unique():
        filegroup = combined_df[combined_df["filepath_without_number"] == filepath_wo]
        highmag_folder = highmag_savefolder_from_filepath_without_number(filepath_wo)
        records = parse_uncaging_records(highmag_folder)

        for group in filegroup["group"].unique():
            group_df = filegroup[filegroup["group"] == group]
            for set_label in group_df["nth_set_label"].unique():
                if set_label == -1:
                    continue
                set_df = group_df[group_df["nth_set_label"] == set_label]
                tiff_path = set_df["after_align_full_save_path"].iloc[0]
                if pd.isna(tiff_path) or not os.path.exists(tiff_path):
                    continue

                record = match_uncaging_record_for_set(set_df, records)
                if record is None:
                    print(
                        f"  Set {group}_{set_label}: no uncaged_spines match, "
                        "skip local adjacent augmentation"
                    )
                    continue

                center_yx = (float(record.uncaging_y_pix), float(record.uncaging_x_pix))
                n_pre = int(set_df["n_pre_frames"].iloc[0])
                n_unc = int(set_df["n_unc_frames"].iloc[0])
                n_post = int(set_df["n_post_frames"].iloc[0])
                corrected_z = int(set_df["corrected_uncaging_z"].iloc[0])
                z_from = max(0, corrected_z - z_plus_minus)
                z_to = corrected_z + z_plus_minus + 1

                pre_frames = []
                for _, row in set_df[set_df["phase"] == "pre"].sort_values(
                    "nth_omit_induction"
                ).iterrows():
                    try:
                        pre_frames.append(
                            grfs._load_flim_zproj_full(
                                str(row["file_path"]),
                                ch_1or2,
                                z_from,
                                z_to,
                                fast_mode=fast_mode,
                                intensity_cache=intensity_cache,
                            )
                        )
                    except Exception:
                        pass

                post_frames = []
                for _, row in set_df[set_df["phase"] == "post"].sort_values(
                    "nth_omit_induction"
                ).iterrows():
                    try:
                        post_frames.append(
                            grfs._load_flim_zproj_full(
                                str(row["file_path"]),
                                ch_1or2,
                                z_from,
                                z_to,
                                fast_mode=fast_mode,
                                intensity_cache=intensity_cache,
                            )
                        )
                    except Exception:
                        pass

                pre_local = adjacent_local_shifts_yx(
                    pre_frames,
                    center_yx,
                    half_size=local_half_size,
                    fast_mode=fast_mode,
                )
                post_local: list[tuple[float, float]] = []
                if post_frames:
                    bridge = pre_frames[-1] if pre_frames else post_frames[0]
                    post_chain = [bridge] + post_frames
                    post_local_full = adjacent_local_shifts_yx(
                        post_chain,
                        center_yx,
                        half_size=local_half_size,
                        fast_mode=fast_mode,
                    )
                    post_local = post_local_full[1:]

                tiff_dir = os.path.dirname(tiff_path)
                base = os.path.splitext(os.path.basename(tiff_path))[0]
                frame_info_path = os.path.join(tiff_dir, f"{base}_frame_info.csv")
                if not os.path.exists(frame_info_path):
                    continue

                frame_info = pd.read_csv(frame_info_path)
                poisoned = (
                    "local_align_mode" in frame_info.columns
                    and "local_shift_y" not in frame_info.columns
                )
                pre_i = 0
                post_i = 0
                for idx, row in frame_info.iterrows():
                    phase = str(row.get("phase", "")).lower()
                    loc_y, loc_x = 0.0, 0.0
                    if phase == "pre" and pre_i < len(pre_local):
                        loc_y, loc_x = float(pre_local[pre_i][0]), float(pre_local[pre_i][1])
                        pre_i += 1
                    elif phase == "post" and post_i < len(post_local):
                        loc_y, loc_x = float(post_local[post_i][0]), float(post_local[post_i][1])
                        post_i += 1
                    if poisoned:
                        sy = float(row.get("shift_y", 0) or 0) - loc_y
                        sx = float(row.get("shift_x", 0) or 0) - loc_x
                        frame_info.at[idx, "shift_y"] = sy
                        frame_info.at[idx, "shift_x"] = sx
                    frame_info.at[idx, "local_shift_y"] = loc_y
                    frame_info.at[idx, "local_shift_x"] = loc_x
                    frame_info.at[idx, "local_align_mode"] = local_align_mode

                frame_info.to_csv(frame_info_path, index=False)
                print(
                    f"  Set {group}_{set_label}: frame_info local adjacent "
                    f"({record.spine_stem}; local not added to shift_y/x)"
                )

    return combined_df


def create_roi_masks_from_seg_masks(
    combined_df: pd.DataFrame,
    *,
    require_all_three: bool = True,
    skip_if_roi_mask_exists: bool = True,
) -> None:
    """
    Write Type-A ROI masks (*_roi_mask.tif) from seg_masks for every set.
    Spine / DendriticShaft come from imaging-time seg_masks (RESPAN_ROI_TYPES).
    The Background seg mask is not used (background = image 20th percentile).

    When skip_if_roi_mask_exists is True, existing *_roi_mask.tif files are left
    unchanged so manually saved ROIs survive workflow re-runs.
    """
    required_cols = [
        "filepath_without_number",
        "group",
        "nth_set_label",
        "phase",
        "after_align_save_path",
        "n_pre_frames",
        "n_unc_frames",
        "n_post_frames",
    ]
    missing = [c for c in required_cols if c not in combined_df.columns]
    if missing:
        print(f"create_roi_masks_from_seg_masks: missing columns {missing}")
        return

    print("Creating ROI masks from respan seg_masks (Spine, DendriticShaft)...")
    for filepath_wo in combined_df["filepath_without_number"].unique():
        filegroup = combined_df[combined_df["filepath_without_number"] == filepath_wo]
        highmag_folder = highmag_savefolder_from_filepath_without_number(filepath_wo)
        records = parse_uncaging_records(highmag_folder)
        if not records:
            print(f"  {highmag_folder}: no uncaged_spines.txt, skip")
            continue

        for group in filegroup["group"].unique():
            group_df = filegroup[filegroup["group"] == group]
            for set_label in group_df["nth_set_label"].unique():
                if set_label == -1:
                    continue
                set_df = group_df[group_df["nth_set_label"] == set_label]
                tiff_path = set_df["after_align_save_path"].iloc[0]
                if pd.isna(tiff_path) or not os.path.exists(tiff_path):
                    print(f"  Set {group}_{set_label}: TIFF missing, skip")
                    continue

                record = match_uncaging_record_for_set(set_df, records)
                if record is None:
                    print(f"  Set {group}_{set_label}: no uncaging log match, skip")
                    continue

                mask_paths = {
                    k: v for k, v in seg_mask_paths(highmag_folder, record.spine_stem).items()
                    if k in RESPAN_ROI_TYPES
                }
                if require_all_three and len(mask_paths) < len(RESPAN_ROI_TYPES):
                    missing_types = set(RESPAN_ROI_TYPES) - set(mask_paths)
                    print(
                        f"  Set {group}_{set_label}: incomplete seg_masks "
                        f"for {record.spine_stem}, missing {missing_types}, skip"
                    )
                    continue

                n_total = (
                    int(set_df["n_pre_frames"].iloc[0])
                    + int(set_df["n_unc_frames"].iloc[0])
                    + int(set_df["n_post_frames"].iloc[0])
                )
                if n_total <= 0:
                    continue

                tiff_dir = os.path.dirname(tiff_path)
                base_name = os.path.splitext(os.path.basename(tiff_path))[0]

                for roi_type in RESPAN_ROI_TYPES:
                    if roi_type not in mask_paths:
                        continue
                    mask_2d = _load_mask_2d(mask_paths[roi_type])
                    stack = np.stack([mask_2d] * n_total, axis=0)
                    out_path = os.path.join(
                        tiff_dir, f"{base_name}_{roi_type}_roi_mask.tif"
                    )
                    if skip_if_roi_mask_exists and os.path.exists(out_path):
                        print(
                            f"    {base_name}: {roi_type} skip "
                            f"(existing ROI mask)"
                        )
                        continue
                    tifffile.imwrite(
                        out_path, stack.astype(np.uint8), photometric="minisblack"
                    )
                    print(
                        f"    {base_name}: {roi_type} <- "
                        f"{mask_paths[roi_type].name}"
                    )

    print("create_roi_masks_from_seg_masks: done.")


def uncaging_xy_in_tiff(set_df: pd.DataFrame, tiff_path: str) -> tuple[float, float] | None:
    """Uncaging position on the uncaging frames of the GUI TIFF (after_align_full pixels).

    The header position (center_x/y = State.Uncaging.Position) is where the laser was
    on the raw uncaging image; FLIMage draws its cross there. The GUI TIFF shows the
    raw uncaging frames moved by unc_drift (frame_info "uncaging" rows), so the marker
    is moved by the same shift and stays on the same tissue as in FLIMage. It is not
    corrected towards the spine: if the acquisition aimed off the spine, it shows so.
    """
    unc = set_df[set_df["phase"] == "unc"]
    if not len(unc) or not {"center_x", "center_y"}.issubset(unc.columns):
        return None
    cx, cy = float(unc.center_x.iloc[0]), float(unc.center_y.iloc[0])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None
    sy = sx = None
    fi_path = os.path.splitext(str(tiff_path))[0] + "_frame_info.csv"
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        u = fi[fi["phase"].astype(str).str.lower().str.startswith("unc")]
        if len(u) and pd.notna(u["shift_y"].iloc[0]) and pd.notna(u["shift_x"].iloc[0]):
            sy, sx = float(u["shift_y"].iloc[0]), float(u["shift_x"].iloc[0])
    if sy is None:
        sy = float(pd.to_numeric(unc.get("unc_drift_y", 0), errors="coerce").fillna(0).iloc[0])
        sx = float(pd.to_numeric(unc.get("unc_drift_x", 0), errors="coerce").fillna(0).iloc[0])
    return cx + sx, cy + sy


def set_uncaging_display_columns(combined_df: pd.DataFrame) -> pd.DataFrame:
    """uncaging_display_x/y (ROI GUI marker) = uncaging_xy_in_tiff for every set with a TIFF.

    Sets without frame_info / header position keep corrected_uncaging_x/y.
    """
    if "after_align_full_save_path" not in combined_df.columns:
        return combined_df
    full_mask = combined_df["after_align_full_save_path"].notna()
    if "corrected_uncaging_x" in combined_df.columns:
        combined_df.loc[full_mask, "uncaging_display_x"] = combined_df.loc[full_mask, "corrected_uncaging_x"]
        combined_df.loc[full_mask, "uncaging_display_y"] = combined_df.loc[full_mask, "corrected_uncaging_y"]
    key_cols = ["filepath_without_number", "group", "nth_set_label"]
    for _, each_set_df in combined_df[full_mask].groupby(key_cols, sort=False):
        xy = uncaging_xy_in_tiff(each_set_df, each_set_df["after_align_full_save_path"].iloc[0])
        if xy is not None:
            combined_df.loc[each_set_df.index, "uncaging_display_x"] = xy[0]
            combined_df.loc[each_set_df.index, "uncaging_display_y"] = xy[1]
    return combined_df


def _prepare_combined_df_for_roi_gui(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Point after_align_save_path at full-size stacks and set the uncaging marker
    (uncaging_display_x/y = uncaging position on the uncaging frames, set_uncaging_display_columns)."""
    if "after_align_full_save_path" not in combined_df.columns:
        return combined_df
    combined_df = combined_df.copy()
    combined_df["after_align_save_path"] = combined_df[
        "after_align_full_save_path"
    ].fillna(combined_df.get("after_align_save_path"))
    return set_uncaging_display_columns(combined_df)


TIME_WINDOW_MIN = (-40.0, 50.0)
_ORIG_SET_COL = "nth_set_label_before_time_window"
_ORIG_PHASE_COL = "phase_before_time_window"


def apply_time_window(
    combined_df: pd.DataFrame, time_window_min: tuple[float, float] | None = TIME_WINDOW_MIN
) -> tuple[pd.DataFrame, bool]:
    """Remove frames outside a time window (minutes from uncaging) from their set.

    A pre/post frame with relative_time_min outside [lo, hi] gets nth_set_label = -1 and
    phase = "None", so it is not in the TIFF, the ROI GUI, the viewer or the
    quantification. A set left without pre or post frames is removed as a whole.
    The labels before the window are kept in *_before_time_window columns and restored
    first, so a changed window (or None = no window) is applied to the original sets.

    Returns:
        (combined_df, changed): changed is True when set membership differs from the
        input (the full-size TIFF stacks must then be rebuilt).
    """
    df = combined_df.copy()
    if "relative_time_min" not in df.columns or "nth_set_label" not in df.columns:
        return df, False
    before = df[["nth_set_label", "phase"]].copy()
    if _ORIG_SET_COL in df.columns:
        keep = df[_ORIG_SET_COL].notna()
        df.loc[keep, "nth_set_label"] = df.loc[keep, _ORIG_SET_COL]
        df.loc[keep, "phase"] = df.loc[keep, _ORIG_PHASE_COL]
    else:
        df[_ORIG_SET_COL] = df["nth_set_label"]
        df[_ORIG_PHASE_COL] = df["phase"]
    df["excluded_time_window"] = False
    if time_window_min is not None:
        lo, hi = time_window_min
        t = pd.to_numeric(df["relative_time_min"], errors="coerce")
        in_set = (df["nth_set_label"] >= 0) & df["phase"].isin(["pre", "post"])
        out = in_set & t.notna() & ((t < lo) | (t > hi))
        df.loc[out, ["nth_set_label", "phase", "excluded_time_window"]] = [-1, "None", True]
        for (g, s), sdf in df[df["nth_set_label"] >= 0].groupby(["group", "nth_set_label"]):
            if not (sdf["phase"] == "pre").any() or not (sdf["phase"] == "post").any():
                print(f"time window {lo:g} to {hi:g} min: {g} set {s:g} has no pre or post frame left; set removed")
                df.loc[sdf.index, ["nth_set_label", "phase", "excluded_time_window"]] = [-1, "None", True]
        n = int(df["excluded_time_window"].sum())
        print(f"time window {lo:g} to {hi:g} min from uncaging: {n} frames removed from their sets")
    changed = not (before["nth_set_label"].astype(float).equals(df["nth_set_label"].astype(float))
                   and before["phase"].astype(str).equals(df["phase"].astype(str)))
    return df, changed


def _frame_keys(frame_info: pd.DataFrame) -> list[tuple[str, int]]:
    """(file name, n-th frame of that file) per TIFF frame (an uncaging FLIM has many)."""
    seen: dict[str, int] = {}
    keys = []
    for name in frame_info["filename"].astype(str).str.lower():
        keys.append((name, seen.get(name, 0)))
        seen[name] = seen.get(name, 0) + 1
    return keys


def snapshot_frame_info(combined_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """frame_info.csv of every existing full-size stack (before it is rebuilt)."""
    out = {}
    if "after_align_full_save_path" not in combined_df.columns:
        return out
    for tiff in combined_df["after_align_full_save_path"].dropna().astype(str).unique():
        fi = os.path.splitext(tiff)[0] + "_frame_info.csv"
        if os.path.exists(fi):
            out[tiff] = pd.read_csv(fi)
    return out


def _applied_shifts(frame_info: pd.DataFrame) -> list[tuple[int, int]]:
    """Integer (y, x) shift applied to each TIFF frame (frame_info shift_y/x)."""
    if "shift_y" not in frame_info.columns or "shift_x" not in frame_info.columns:
        return [(0, 0)] * len(frame_info)
    sy = pd.to_numeric(frame_info["shift_y"], errors="coerce").fillna(0).round().astype(int)
    sx = pd.to_numeric(frame_info["shift_x"], errors="coerce").fillna(0).round().astype(int)
    return list(zip(sy.tolist(), sx.tolist()))


def _translate(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Move a 2D array by whole pixels (zero fill)."""
    out = np.zeros_like(img)
    h, w = img.shape
    if abs(dy) >= h or abs(dx) >= w:
        return out
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        img[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def remap_roi_mask_stacks(old_frame_info: dict[str, pd.DataFrame]) -> int:
    """Carry existing <base>_*_roi_mask*.tif over to the rebuilt stack.

    Frames are matched by file name (a rebuild may remove frames, time window). ROI
    masks in TIFF coordinates (<base>_<roi>_roi_mask.tif, also GUI edits) are also moved
    by the change of the applied integer shift of that frame (new alignment, e.g. to
    Pre 1), so they stay on the same tissue. *_raw masks (raw FLIM coordinates) are only
    re-ordered. Frames not in the old stack get an empty mask. Returns the number of
    mask files rewritten.
    """
    import glob

    n_files = 0
    for tiff, old in old_frame_info.items():
        base = os.path.splitext(tiff)[0]
        new_fi = base + "_frame_info.csv"
        if not os.path.exists(new_fi):
            continue
        new = pd.read_csv(new_fi)
        old_keys, new_keys = _frame_keys(old), _frame_keys(new)
        old_sh, new_sh = _applied_shifts(old), _applied_shifts(new)
        old_idx = {k: i for i, k in enumerate(old_keys)}
        idx = [old_idx.get(k, -1) for k in new_keys]
        moves = [(new_sh[j][0] - old_sh[i][0], new_sh[j][1] - old_sh[i][1]) if i >= 0 else (0, 0)
                 for j, i in enumerate(idx)]
        if old_keys == new_keys and not any(moves):
            continue
        for path in sorted(glob.glob(glob.escape(base) + "_*_roi_mask*.tif")):
            stack = tifffile.imread(path)
            if stack.ndim != 3 or stack.shape[0] != len(old_keys):
                print(f"  mask not remapped (frame count {stack.shape[0]} != {len(old_keys)}): {path}")
                continue
            tiff_coords = not path.endswith("_raw.tif")
            out = np.zeros((len(new_keys),) + stack.shape[1:], dtype=stack.dtype)
            for j, i in enumerate(idx):
                if i >= 0:
                    out[j] = _translate(stack[i], *moves[j]) if tiff_coords else stack[i]
            tifffile.imwrite(path, out)
            n_files += 1
        n_moved = sum(1 for m in moves if any(m))
        print(f"  ROI masks remapped {len(old_keys)} -> {len(new_keys)} frames, "
              f"{n_moved} frames moved: {os.path.basename(base)}")
    return n_files


def _has_valid_roi_sets(combined_df: pd.DataFrame) -> bool:
    """True if at least one set is labeled for ROI (nth_set_label >= 0)."""
    if combined_df is None or combined_df.empty:
        return False
    if "nth_set_label" not in combined_df.columns:
        return False
    return bool(
        (
            (combined_df["nth_set_label"] >= 0)
            & combined_df["nth_set_label"].notna()
        ).any()
    )


def _has_full_size_stack_paths(combined_df: pd.DataFrame) -> bool:
    """True if rebuild already stored full-size TIFF paths and frame counts."""
    if "after_align_full_save_path" not in combined_df.columns:
        return False
    if "n_pre_frames" not in combined_df.columns:
        return False
    return bool(combined_df["after_align_full_save_path"].notna().any())


def _print_no_valid_set_diagnostics(combined_df: pd.DataFrame) -> None:
    """Explain why the ROI GUI would be empty (usually unknown uncaging n_images)."""
    print("ERROR: no valid ROI sets (nth_set_label >= 0). Skipping empty GUI.")
    if "nth_set_label" in combined_df.columns:
        print(f"  nth_set_label unique: {combined_df['nth_set_label'].unique()}")
    if "phase" in combined_df.columns:
        print(
            f"  phase counts: {combined_df['phase'].value_counts(dropna=False).to_dict()}"
        )
    if "unknown_frame" in combined_df.columns:
        unk = combined_df[combined_df["unknown_frame"] == True]
        if len(unk) > 0:
            print(
                "  Files classified as unknown (n_images not in uncaging_frame_num):"
            )
            for path in unk["file_path"].tolist():
                print(f"    {os.path.basename(str(path))}")
            print(
                "  Add those files' n_images to uncaging_frame_num and re-run "
                "first_processing (do not reuse this pickle)."
            )
    if "uncaging_frame" in combined_df.columns:
        n_unc = int((combined_df["uncaging_frame"] == True).sum())
        print(f"  uncaging_frame True count: {n_unc}")


def _launch_roi_review_gui(
    combined_df: pd.DataFrame,
    df_save_path: str,
    *,
    uncaging_roi_keyframe_count: int | None = None,
) -> pd.DataFrame:
    """Open TIFF ROI GUI so the user can review or edit pre-filled seg_mask ROIs."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    # Keep a Python reference; dropping it lets Qt destroy the window immediately.
    file_selection_gui = launch_file_selection_gui_tiff_only(
        combined_df,
        df_save_path,
        additional_columns=["dt"],
        save_auto=False,
        uncaging_roi_keyframe_count=uncaging_roi_keyframe_count,
        roi_types=RESPAN_ROI_TYPES,
    )
    app.exec_()
    print("ROI review/edit (full-size) finished.")
    _ = file_selection_gui
    if os.path.exists(df_save_path):
        return pd.read_pickle(df_save_path)
    return combined_df


def run_tiff_uncaging_roi_respan(
    ch_1or2: int = 2,
    z_plus_minus: int = 2,
    pre_length: int = 3,
    photon_threshold: int = 15,
    total_photon_threshold: int = 1000,
    *,
    global_align_method: str = GLOBAL_ALIGN_METHOD,
    local_align_mode: str = LOCAL_ALIGN_MODE,
    local_crop_half_size: int = 60,
    predefined_df_path: str | None = None,
    uncaging_frame_num: list[int] | None = None,
    titration_frame_num: list[int] | None = None,
    flim_path: str | None = None,
    skip_roi_gui: bool = False,
    uncaging_roi_keyframe_count: int | None = None,
    overwrite_seg_roi_masks: bool = False,
    skip_lifetime_analysis: bool = False,
    fast_mode: bool = False,
    spine_roi_source: str = "seg",
    respan_track_queue_root: str | None = None,
    time_window_min: tuple[float, float] | None = TIME_WINDOW_MIN,
    align_reference: str = "pre1",
) -> tuple[str, str] | tuple[None, None]:
    """
    Full ROI quantification for respan highmag data using pre-built seg_masks.

    Alignment (explicit):
      global_align_method: roi_adjacent for pre/post FLIM chain
      local_align_mode: adjacent writes local_shift_y/x on frame_info (not added
      to shift_y/x). Raw FLIM masks use integer before/after TIFF shifts.

    fast_mode:
      Default False (same as before). True enables faster GUI prep:
      skip repeated FLIM decode / second align, header-only shape peek,
      intensity-only decode + disk cache, no first_processing PNG/small TIFF,
      upsample_factor=1 without applying subpixel shifts to the full stack,
      reuse intensity arrays for local adjacent.

    ROI flow:
      1) Pre-fill Spine / DendriticShaft from seg_masks (no Background ROI;
         background = 20th percentile of the quantified image)
      2) ROI GUI for review and edits (unless skip_roi_gui=True)
      3) Drift-corrected masks and FLIM quantification

    spine_roi_source:
      "seg" (default): Spine ROI from seg_masks (step 1 above).
      "respan_tracked": after step 1, replace the Spine ROI by the uncaged spine
      tracked with per-frame RESPAN (respan_tracked_spine_roi.py; needs
      ongoing/ASIcontroller/respan_track_session.py to have run). Masks edited in
      the GUI are never overwritten; sets without tracking keep the seg ROI.

    align_reference:
      "pre1" (default): every pre/post frame of a set is registered to the set's Pre 1
      (pre1_alignment.py); the TIFF, z windows and masks follow it. "002": the global
      registration to the group's 002 (former). When it changes the shifts of a loaded
      combined_df, the stacks are rebuilt and the ROI masks moved with the frames.

    time_window_min:
      (lo, hi) minutes from uncaging (default -40 to +50). Pre/post frames outside
      are removed from their set before the TIFF stacks are built, so they are not
      reviewed or quantified (apply_time_window). None keeps every frame. When the
      window changes the sets of a loaded combined_df, the stacks are rebuilt.

    Set skip_lifetime_analysis=True to quantify intensity only (lifetime/total_photon as NaN).
    """
    if uncaging_frame_num is None:
        uncaging_frame_num = [33, 34, 35, 36, 55, 80, 144]
    if titration_frame_num is None:
        titration_frame_num = []

    intensity_cache: dict[str, np.ndarray] = {}
    if fast_mode:
        print(
            "fast_mode=True: intensity-only decode, disk cache, skip PNG/small TIFF, "
            "skip second full align, upsample_factor=1, reuse arrays for local shifts"
        )

    summary = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"workflow: respan seg_masks",
        f"ch_1or2: {ch_1or2}",
        f"z_plus_minus: {z_plus_minus}",
        f"pre_length: {pre_length}",
        f"global_align_method: {global_align_method}",
        f"local_align_mode: {local_align_mode}",
        f"local_crop_half_size: {local_crop_half_size}",
        f"photon_threshold: {photon_threshold}",
        f"total_photon_threshold: {total_photon_threshold}",
        f"skip_roi_gui: {skip_roi_gui}",
        f"uncaging_roi_keyframe_count: {uncaging_roi_keyframe_count}",
        f"overwrite_seg_roi_masks: {overwrite_seg_roi_masks}",
        f"skip_lifetime_analysis: {skip_lifetime_analysis}",
        f"fast_mode: {fast_mode}",
        f"spine_roi_source: {spine_roi_source}",
        f"time_window_min: {time_window_min}",
        f"align_reference: {align_reference}",
        "=" * 60,
    ]

    use_predefined_df = bool(predefined_df_path and os.path.exists(predefined_df_path))
    one_of_filepath_list: list[str] = []

    if use_predefined_df:
        df_save_path = predefined_df_path
        print("=" * 60)
        print("Using predefined combined_df (dialog-free mode)")
        print(f"  combined_df: {df_save_path}")
        print("=" * 60)
    else:
        if flim_path:
            one_of_filepath_list = [flim_path]
        else:
            picked = ask_open_path_gui(filetypes=[("FLIM files", "*.flim")])
            if not picked:
                print("No file selected.")
                return None, None
            one_of_filepath_list = [picked]
        df_save_path = os.path.join(
            os.path.dirname(one_of_filepath_list[0]), DEFAULT_COMBINED_DF_NAME
        )

    summary.append(f"predefined_df_path: {predefined_df_path}")
    summary.append(f"df_save_path: {df_save_path}")

    combined_df: pd.DataFrame | None = None
    loaded_existing_combined_df = False

    if use_predefined_df:
        try:
            combined_df = pd.read_pickle(df_save_path)
            loaded_existing_combined_df = True
            print(f"Loaded predefined df: {df_save_path}")
        except Exception as e:
            print(f"Failed to load predefined_df_path: {e}")
            raise
    else:
        if os.path.exists(df_save_path):
            print(f"Found existing pickle: {df_save_path}")
            use_found_pkl = ask_yes_no_gui(
                f"Found {DEFAULT_COMBINED_DF_NAME} in this folder. Use this file?"
            )
            if use_found_pkl:
                combined_df = pd.read_pickle(df_save_path)
                loaded_existing_combined_df = True
                print(f"Loaded: {df_save_path}")
            elif ask_yes_no_gui(
                "Select a different pickle? (No = rebuild from FLIM files)"
            ):
                picked_pkl = ask_open_path_gui(filetypes=[("Pickle files", "*.pkl")])
                if picked_pkl and os.path.exists(picked_pkl):
                    df_save_path = picked_pkl
                    combined_df = pd.read_pickle(df_save_path)
                    loaded_existing_combined_df = True
                    print(f"Loaded: {df_save_path}")
                else:
                    print("No pickle selected; will run first_processing.")
            else:
                print(
                    f"Not using {DEFAULT_COMBINED_DF_NAME}; "
                    "will run first_processing."
                )

    first_processing_errors: list[str] = []
    if combined_df is None:
        if not one_of_filepath_list:
            print("No FLIM path for first_processing. Exiting.")
            return None, None
        fp_kwargs: dict = {
            "pre_length": pre_length,
            "uncaging_frame_num": uncaging_frame_num,
        }
        if titration_frame_num is not None:
            fp_kwargs["titration_frame_num"] = titration_frame_num

        with _patch_load_and_align(
            global_align_method,
            fast_mode=fast_mode,
            intensity_cache=intensity_cache if fast_mode else None,
        ):
            combined_df = pd.DataFrame()
            for one_path in one_of_filepath_list:
                print(f"\nfirst_processing (global={global_align_method}): {one_path}\n")
                temp_df, error_dict = first_processing_for_flim_files(
                    one_path,
                    z_plus_minus,
                    ch_1or2,
                    save_plot_TF=not fast_mode,
                    save_tif_TF=not fast_mode,
                    return_error_dict=True,
                    **fp_kwargs,
                )
                for group_key, reason in error_dict.items():
                    record_roi_error(
                        first_processing_errors,
                        group=group_key,
                        set_label="",
                        file="",
                        reason=str(reason),
                    )
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        combined_df.to_pickle(df_save_path)
        combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))
        print(f"Saved: {df_save_path}")

    if combined_df is None or combined_df.empty:
        print("No data.")
        return None, None

    combined_df, _ = ensure_combined_df_paths_exist(
        combined_df,
        df_save_path=df_save_path,
        anchors=one_of_filepath_list,
    )
    if combined_df is None:
        print("Path remap cancelled.")
        return None, None

    if loaded_existing_combined_df and not _has_valid_roi_sets(combined_df):
        _print_no_valid_set_diagnostics(combined_df)
        raise RuntimeError(
            "Loaded combined_df has no valid ROI sets (nth_set_label >= 0). "
            "Do not reuse this pickle; re-run first_processing after fixing "
            "uncaging_frame_num."
        )

    combined_df, sets_changed_by_window = apply_time_window(combined_df, time_window_min)
    if not _has_valid_roi_sets(combined_df):
        print("No set left inside the time window", time_window_min)
        return None, None
    from pre1_alignment import realign_sets_to_pre1

    combined_df, shifts_changed = realign_sets_to_pre1(combined_df, ch_1or2, reference=align_reference)

    error_log: list[str] = list(first_processing_errors)
    session_dir = os.path.dirname(df_save_path)
    out_csv = df_save_path.replace(".pkl", "_intensity_lifetime_all_frames.csv")

    try:
        skip_full_size_build = False
        skip_tiff_if_exists = False
        old_frame_info: dict[str, pd.DataFrame] = {}
        if loaded_existing_combined_df and (sets_changed_by_window or shifts_changed):
            print(
                "Time window or alignment reference changed the loaded combined_df: "
                "rebuilding all full-size stacks (existing ROI masks follow the frames)."
            )
            old_frame_info = snapshot_frame_info(combined_df)
        elif use_predefined_df and loaded_existing_combined_df:
            skip_tiff_if_exists = True
            print(
                "Predefined df mode: running rebuild with skip_tiff_if_exists=True "
                "(refresh frame_info.csv, skip TIFF write if exists)"
            )
        elif loaded_existing_combined_df:
            if _has_full_size_stack_paths(combined_df):
                skip_full_size_build = ask_yes_no_gui(
                    "Skip 'Building full-size stacks for ROI definition' and use existing stack paths?"
                )
            else:
                print(
                    "Existing combined_df has no full-size TIFF stacks "
                    "(missing after_align_full_save_path / n_pre_frames). "
                    "Will build them instead of skipping."
                )

        if skip_full_size_build:
            print("Skipped building full-size stacks for ROI definition.")
        else:
            print("\n" + "=" * 60)
            print(
                f"Building full-size stacks (global_align={global_align_method}, "
                f"local_align={local_align_mode}"
                + (", fast_mode" if fast_mode else "")
                + ")"
            )
            print("=" * 60)
            with _patch_load_and_align(
                global_align_method,
                fast_mode=fast_mode,
                intensity_cache=intensity_cache if fast_mode else None,
            ):
                combined_df = rebuild_tiff_full_size_for_roi(
                    combined_df,
                    ch_1or2,
                    z_plus_minus,
                    skip_tiff_if_exists=skip_tiff_if_exists,
                    error_log=error_log,
                    fast_mode=fast_mode,
                    intensity_cache=intensity_cache if fast_mode else None,
                )
            if old_frame_info:
                remap_roi_mask_stacks(old_frame_info)
            combined_df = augment_frame_info_local_adjacent(
                combined_df,
                ch_1or2=ch_1or2,
                z_plus_minus=z_plus_minus,
                local_half_size=local_crop_half_size,
                local_align_mode=local_align_mode,
                fast_mode=fast_mode,
                intensity_cache=intensity_cache if fast_mode else None,
            )
        combined_df.to_pickle(df_save_path)
        combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))

        combined_df = _prepare_combined_df_for_roi_gui(combined_df)
        combined_df.to_pickle(df_save_path)
        combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))

        if not _has_valid_roi_sets(combined_df):
            _print_no_valid_set_diagnostics(combined_df)
            raise RuntimeError(
                "No valid ROI sets (nth_set_label >= 0). "
                "Uncaging files were not detected; TIFF stacks were not built."
            )

        create_roi_masks_from_seg_masks(
            combined_df,
            skip_if_roi_mask_exists=not overwrite_seg_roi_masks,
        )
        if spine_roi_source == "respan_tracked":
            from respan_tracked_spine_roi import apply_tracked_spine_rois

            track_summary = apply_tracked_spine_rois(
                combined_df, queue_root=respan_track_queue_root
            )
            if len(track_summary):
                track_summary.to_csv(
                    df_save_path.replace(".pkl", "_respan_tracked_spine_roi.csv"),
                    index=False,
                )
        elif spine_roi_source != "seg":
            raise ValueError(f"unknown spine_roi_source: {spine_roi_source!r}")

        if skip_roi_gui:
            print("skip_roi_gui=True: skip ROI review GUI.")
        else:
            print("\nLaunching ROI GUI (Spine / Shaft pre-filled; review and edit as needed)...")
            combined_df = _launch_roi_review_gui(
                combined_df,
                df_save_path,
                uncaging_roi_keyframe_count=uncaging_roi_keyframe_count,
            )

        print("\nSaving drift-corrected ROI masks (Type B)...")
        save_drift_corrected_roi_masks(combined_df, roi_types=RESPAN_ROI_TYPES)
        combined_df.to_pickle(df_save_path)
        combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))

        if "reject" not in combined_df.columns:
            combined_df["reject"] = 0
        else:
            combined_df["reject"] = (
                (combined_df["reject"] == True) | (combined_df["reject"] == 1)
            ).astype(int)

        if skip_lifetime_analysis:
            print("Quantifying intensity from FLIM (lifetime skipped)...")
        else:
            print("Quantifying intensity and lifetime from FLIM...")
        quantify_intensity_from_flim(
            combined_df,
            ch_1or2,
            z_plus_minus,
            out_csv,
            photon_threshold=photon_threshold,
            total_photon_threshold=total_photon_threshold,
            skip_lifetime_analysis=skip_lifetime_analysis,
            background_mode=BACKGROUND_MODE_MIP_P20,
        )

        summary.append(f"df_save_path: {df_save_path}")
        summary.append(f"out_csv_path: {out_csv}")
        summary.append("finished at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        summary_text = "\n".join(summary) + "\n"
        print(summary_text)

        summary_path = os.path.join(session_dir, "summary_str_respan.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(summary_text)

        return df_save_path, out_csv
    except Exception as e:
        record_roi_error(
            error_log,
            group="",
            set_label="",
            file="",
            reason=f"workflow failed: {type(e).__name__}: {e}",
        )
        raise
    finally:
        print_roi_analysis_errors(error_log)
        saved_errors = save_roi_analysis_error_log(session_dir, error_log)
        if saved_errors:
            print(f"Wrote error log: {saved_errors}")
