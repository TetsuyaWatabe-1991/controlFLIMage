# -*- coding: utf-8 -*-
"""
Respan highmag ROI workflow: pre-defined Spine / Shaft / Background masks from seg_masks.

Uses the same alignment policy as tpem_low_high_spine_multi_merged_titrate_uncaging_pow_respan.py:
  - Global pre/post: roi_adjacent (FLIMageAlignment POST_ACQUISITION_ALIGN_METHOD)
  - Local crop: adjacent-frame roi_adjacent (LocalAlignMode.ADJACENT)

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
    ROI_MASK_RAW_SUFFIX,
    ROI_TYPES,
    quantify_intensity_from_flim,
    rebuild_tiff_full_size_for_roi,
    save_drift_corrected_roi_masks,
)
from simple_dialog import ask_open_path_gui, ask_yes_no_gui  # noqa: E402

DEFAULT_COMBINED_DF_NAME = "combined_df_respan.pkl"

# Explicit alignment policy (matches live respan titration script).
GLOBAL_ALIGN_METHOD = POST_ACQUISITION_ALIGN_METHOD  # "roi_adjacent"
LOCAL_ALIGN_MODE = "adjacent"  # respan_spine_quant.LocalAlignMode.ADJACENT

SEG_MASK_SUBDIR = "seg_masks"
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
) -> tuple[np.ndarray, np.ndarray, list]:
    """Load FLIM files and align with an explicit FLIMageAlignment method."""
    tiff_multi, iminfo, relative_sec_list = flim_files_to_nparray(
        filelist, ch=ch, normalize_by_averageNum=True
    )
    shifts, aligned = Align_4d_array(
        tiff_multi, iminfo=iminfo, method=align_method
    )
    return aligned, shifts, relative_sec_list


@contextmanager
def _patch_load_and_align(align_method: str = GLOBAL_ALIGN_METHOD) -> Iterator[None]:
    """Temporarily override gui_integration.load_and_align_data (no permanent edits)."""
    import gui_integration as gi

    original = gi.load_and_align_data

    def _wrapped(filelist, ch):
        return load_and_align_data_explicit(filelist, ch, align_method=align_method)

    gi.load_and_align_data = _wrapped
    try:
        yield
    finally:
        gi.load_and_align_data = original


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
    target = _norm_path(last_pre)
    for rec in records:
        if _norm_path(rec.flim_path) == target:
            return rec
    return None


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

    for i in range(1, len(frames)):
        crop = _crop_2d(frames[i], center_yx, half_size)
        pair_4d = np.stack([prev_crop, crop], axis=0)[:, np.newaxis, :, :]
        shifts, _ = Align_4d_array(
            pair_4d,
            method=align_method,
            roi_center_zyx=(0, roi_cy, roi_cx),
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
) -> pd.DataFrame:
    """
    Add adjacent local shifts (Y, X) on top of global shifts in each *_frame_info.csv.
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
                                str(row["file_path"]), ch_1or2, z_from, z_to
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
                                str(row["file_path"]), ch_1or2, z_from, z_to
                            )
                        )
                    except Exception:
                        pass

                pre_local = adjacent_local_shifts_yx(
                    pre_frames, center_yx, half_size=local_half_size
                )
                post_local: list[tuple[float, float]] = []
                if post_frames:
                    bridge = pre_frames[-1] if pre_frames else post_frames[0]
                    post_chain = [bridge] + post_frames
                    post_local_full = adjacent_local_shifts_yx(
                        post_chain, center_yx, half_size=local_half_size
                    )
                    post_local = post_local_full[1:]

                tiff_dir = os.path.dirname(tiff_path)
                base = os.path.splitext(os.path.basename(tiff_path))[0]
                frame_info_path = os.path.join(tiff_dir, f"{base}_frame_info.csv")
                if not os.path.exists(frame_info_path):
                    continue

                frame_info = pd.read_csv(frame_info_path)
                pre_i = 0
                post_i = 0
                for idx, row in frame_info.iterrows():
                    phase = str(row.get("phase", "")).lower()
                    sy = float(row.get("shift_y", 0) or 0)
                    sx = float(row.get("shift_x", 0) or 0)
                    if phase == "pre" and pre_i < len(pre_local):
                        sy += pre_local[pre_i][0]
                        sx += pre_local[pre_i][1]
                        pre_i += 1
                    elif phase == "post" and post_i < len(post_local):
                        sy += post_local[post_i][0]
                        sx += post_local[post_i][1]
                        post_i += 1
                    frame_info.at[idx, "shift_y"] = sy
                    frame_info.at[idx, "shift_x"] = sx
                    frame_info.at[idx, "local_align_mode"] = local_align_mode

                frame_info.to_csv(frame_info_path, index=False)
                print(
                    f"  Set {group}_{set_label}: frame_info local adjacent "
                    f"({record.spine_stem})"
                )

    return combined_df


def create_roi_masks_from_seg_masks(
    combined_df: pd.DataFrame,
    *,
    require_all_three: bool = True,
) -> None:
    """
    Write Type-A ROI masks (*_roi_mask.tif) from seg_masks for every set.
    Spine / DendriticShaft / Background come from imaging-time seg_masks.
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

    print("Creating ROI masks from respan seg_masks (Spine, DendriticShaft, Background)...")
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

                mask_paths = seg_mask_paths(highmag_folder, record.spine_stem)
                if require_all_three and len(mask_paths) < len(SEG_MASK_FILES):
                    missing_types = set(SEG_MASK_FILES) - set(mask_paths)
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

                for roi_type in ROI_TYPES:
                    if roi_type not in mask_paths:
                        continue
                    mask_2d = _load_mask_2d(mask_paths[roi_type])
                    stack = np.stack([mask_2d] * n_total, axis=0)
                    out_path = os.path.join(
                        tiff_dir, f"{base_name}_{roi_type}_roi_mask.tif"
                    )
                    tifffile.imwrite(
                        out_path, stack.astype(np.uint8), photometric="minisblack"
                    )
                    print(
                        f"    {base_name}: {roi_type} <- "
                        f"{mask_paths[roi_type].name}"
                    )

    print("create_roi_masks_from_seg_masks: done.")


def _prepare_combined_df_for_roi_gui(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Point after_align_save_path at full-size stacks and set uncaging display coords."""
    if "after_align_full_save_path" not in combined_df.columns:
        return combined_df
    combined_df = combined_df.copy()
    combined_df["after_align_save_path"] = combined_df[
        "after_align_full_save_path"
    ].fillna(combined_df.get("after_align_save_path"))
    full_mask = combined_df["after_align_full_save_path"].notna()
    if "corrected_uncaging_x" in combined_df.columns:
        combined_df.loc[full_mask, "uncaging_display_x"] = combined_df.loc[
            full_mask, "corrected_uncaging_x"
        ]
        combined_df.loc[full_mask, "uncaging_display_y"] = combined_df.loc[
            full_mask, "corrected_uncaging_y"
        ]
    if all(
        c in combined_df.columns
        for c in ["center_x", "center_y", "unc_drift_x", "unc_drift_y"]
    ):
        key_cols = ["filepath_without_number", "group", "nth_set_label"]
        for _, each_set_df in combined_df[full_mask].groupby(key_cols, sort=False):
            unc_rows = each_set_df[each_set_df["phase"] == "unc"]
            if len(unc_rows) == 0:
                continue
            unc_row = unc_rows.iloc[0]
            display_x = float(unc_row.get("center_x", 0) or 0) + float(
                unc_row.get("unc_drift_x", 0) or 0
            )
            display_y = float(unc_row.get("center_y", 0) or 0) + float(
                unc_row.get("unc_drift_y", 0) or 0
            )
            combined_df.loc[each_set_df.index, "uncaging_display_x"] = display_x
            combined_df.loc[each_set_df.index, "uncaging_display_y"] = display_y
    return combined_df


def _launch_roi_review_gui(combined_df: pd.DataFrame, df_save_path: str) -> pd.DataFrame:
    """Open TIFF ROI GUI so the user can review or edit pre-filled seg_mask ROIs."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    launch_file_selection_gui_tiff_only(
        combined_df,
        df_save_path,
        additional_columns=["dt"],
        save_auto=False,
    )
    app.exec_()
    print("ROI review/edit (full-size) finished.")
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
) -> tuple[str, str] | tuple[None, None]:
    """
    Full ROI quantification for respan highmag data using pre-built seg_masks.

    Alignment (explicit):
      global_align_method: roi_adjacent for pre/post FLIM chain
      local_align_mode: adjacent for spine-centered local refinement (frame_info)

    ROI flow:
      1) Pre-fill Spine / DendriticShaft / Background from seg_masks
      2) ROI GUI for review and edits (unless skip_roi_gui=True)
      3) Drift-corrected masks and FLIM quantification
    """
    if uncaging_frame_num is None:
        uncaging_frame_num = [33, 34, 35, 55]
    if titration_frame_num is None:
        titration_frame_num = []

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
        yn_already_have = ask_yes_no_gui(
            f"Do you already have {DEFAULT_COMBINED_DF_NAME}?"
        )
        if yn_already_have:
            picked_pkl = ask_open_path_gui(filetypes=[("Pickle files", "*.pkl")])
            if picked_pkl and os.path.exists(picked_pkl):
                df_save_path = picked_pkl
                combined_df = pd.read_pickle(df_save_path)
                loaded_existing_combined_df = True
                print(f"Loaded: {df_save_path}")

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

        with _patch_load_and_align(global_align_method):
            combined_df = pd.DataFrame()
            for one_path in one_of_filepath_list:
                print(f"\nfirst_processing (global={global_align_method}): {one_path}\n")
                temp_df = first_processing_for_flim_files(
                    one_path,
                    z_plus_minus,
                    ch_1or2,
                    save_plot_TF=True,
                    save_tif_TF=True,
                    return_error_dict=False,
                    **fp_kwargs,
                )
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        combined_df.to_pickle(df_save_path)
        combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))
        print(f"Saved: {df_save_path}")

    if combined_df is None or combined_df.empty:
        print("No data.")
        return None, None

    skip_full_size_build = False
    skip_tiff_if_exists = False
    if use_predefined_df and loaded_existing_combined_df:
        skip_tiff_if_exists = True
        print(
            "Predefined df mode: running rebuild with skip_tiff_if_exists=True "
            "(refresh frame_info.csv, skip TIFF write if exists)"
        )
    elif loaded_existing_combined_df:
        skip_full_size_build = ask_yes_no_gui(
            "Skip 'Building full-size stacks for ROI definition' and use existing stack paths?"
        )

    if skip_full_size_build:
        print("Skipped building full-size stacks for ROI definition.")
    else:
        print("\n" + "=" * 60)
        print(
            f"Building full-size stacks (global_align={global_align_method}, "
            f"local_align={local_align_mode})"
        )
        print("=" * 60)
        with _patch_load_and_align(global_align_method):
            combined_df = rebuild_tiff_full_size_for_roi(
                combined_df,
                ch_1or2,
                z_plus_minus,
                skip_tiff_if_exists=skip_tiff_if_exists,
            )
        combined_df = augment_frame_info_local_adjacent(
            combined_df,
            ch_1or2=ch_1or2,
            z_plus_minus=z_plus_minus,
            local_half_size=local_crop_half_size,
            local_align_mode=local_align_mode,
        )
    combined_df.to_pickle(df_save_path)
    combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))

    combined_df = _prepare_combined_df_for_roi_gui(combined_df)
    combined_df.to_pickle(df_save_path)
    combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))

    create_roi_masks_from_seg_masks(combined_df)

    if skip_roi_gui:
        print("skip_roi_gui=True: skip ROI review GUI.")
    else:
        print("\nLaunching ROI GUI (seg_masks pre-filled; review and edit as needed)...")
        combined_df = _launch_roi_review_gui(combined_df, df_save_path)

    print("\nSaving drift-corrected ROI masks (Type B)...")
    save_drift_corrected_roi_masks(combined_df)
    combined_df.to_pickle(df_save_path)
    combined_df.to_csv(df_save_path.replace(".pkl", ".csv"))

    if "reject" not in combined_df.columns:
        combined_df["reject"] = 0
    else:
        combined_df["reject"] = (
            (combined_df["reject"] == True) | (combined_df["reject"] == 1)
        ).astype(int)

    out_csv = df_save_path.replace(".pkl", "_intensity_lifetime_all_frames.csv")
    print("Quantifying intensity and lifetime from FLIM...")
    quantify_intensity_from_flim(
        combined_df,
        ch_1or2,
        z_plus_minus,
        out_csv,
        photon_threshold=photon_threshold,
        total_photon_threshold=total_photon_threshold,
    )

    summary.append(f"df_save_path: {df_save_path}")
    summary.append(f"out_csv_path: {out_csv}")
    summary.append("finished at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    summary_text = "\n".join(summary) + "\n"
    print(summary_text)

    summary_path = os.path.join(os.path.dirname(df_save_path), "summary_str_respan.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary_text)

    return df_save_path, out_csv
