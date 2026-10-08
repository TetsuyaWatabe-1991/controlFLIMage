"""Align every frame of a set directly to the set's Pre 1 (instead of the group's 002).

The global alignment registers each FLIM file of a group to 002 on its own. A frame far
from 002 in time (or with a changed field) can lock onto a wrong peak, and then frames of
one set are tens of pixels apart although the tissue barely moved. Here each pre/post
frame of a set is registered to the first pre frame (Pre 1) of the same set by 3D phase
correlation (FLIMageAlignment.Align_4d_array, method "highpass" as the global alignment).

The result is written back to shift_z/shift_y/shift_x as
    shift = shift of Pre 1 (global, unchanged) + shift relative to Pre 1,
so every consumer that uses differences within a set (TIFF frames relative to Pre 1,
per-frame z windows relative to the last pre, raw ROI masks, uncaging drift) gets the
Pre 1 registration. The global values are kept in shift_*_002 and restored first, so the
step can be repeated or turned off (reference "002").
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

REFERENCE_PRE1 = "pre1"
REFERENCE_002 = "002"
SHIFT_COLS = ("shift_z", "shift_y", "shift_x")
ORIG_SUFFIX = "_002"
DEFAULT_METHOD = "highpass"  # same as the global alignment; more precise than "traditional" here


def _volume(flim_path: str, ch_1or2: int) -> np.ndarray:
    """(Z, Y, X) intensity of one channel (fast_mode intensity cache)."""
    from flim_fast_io import load_flim_intensity

    return np.asarray(load_flim_intensity(flim_path, use_cache=True)[0][:, 0, ch_1or2 - 1], dtype=np.float64)


def shift_to_reference(ref: np.ndarray, mov: np.ndarray, method: str = DEFAULT_METHOD) -> tuple[float, float, float]:
    """(z, y, x) shift of mov relative to ref, same sign as the global alignment."""
    from FLIMageAlignment import Align_4d_array

    shifts, _ = Align_4d_array(np.stack([ref, mov]), method=method, apply_shifts=False)
    s = shifts[-1]
    return float(s[0]), float(s[1]), float(s[2])


def restore_global_shifts(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Put the global (002) shifts back where they were saved."""
    df = combined_df.copy()
    for c in SHIFT_COLS:
        o = c + ORIG_SUFFIX
        if o in df.columns:
            keep = df[o].notna()
            df.loc[keep, c] = df.loc[keep, o]
    if "align_reference" in df.columns:
        df["align_reference"] = REFERENCE_002
    return df


def realign_sets_to_pre1(
    combined_df: pd.DataFrame, ch_1or2: int, reference: str = REFERENCE_PRE1, log=print,
    method: str = DEFAULT_METHOD,
) -> tuple[pd.DataFrame, bool]:
    """Register pre/post frames of every set to its Pre 1 (see module docstring).

    Args:
        combined_df: session table with nth_set_label, phase, file_path, shift_z/y/x.
        ch_1or2: channel used for the registration.
        reference: "pre1" (this step) or "002" (global shifts, former behaviour).

    Returns:
        (combined_df, changed): changed is True when any shift differs from the input.
    """
    if reference not in (REFERENCE_PRE1, REFERENCE_002):
        raise ValueError(f"reference must be {REFERENCE_PRE1!r} or {REFERENCE_002!r}, got {reference!r}")
    if not all(c in combined_df.columns for c in SHIFT_COLS):
        return combined_df, False
    before = combined_df[list(SHIFT_COLS)].astype(float).copy()
    df = combined_df.copy()
    for c in SHIFT_COLS:
        if c + ORIG_SUFFIX not in df.columns:
            df[c + ORIG_SUFFIX] = df[c]
    df = restore_global_shifts(df)
    if reference == REFERENCE_PRE1:
        n_sets = n_frames = 0
        rows = df[(df["nth_set_label"] >= 0) & df["phase"].isin(["pre", "post"])]
        for (g, s), sdf in rows.groupby(["group", "nth_set_label"]):
            sdf = sdf.sort_values("nth_omit_induction")
            pre = sdf[sdf["phase"] == "pre"]
            if not len(pre):
                continue
            p1 = pre.iloc[0]
            try:
                v1 = _volume(str(p1["file_path"]), ch_1or2)
            except Exception as exc:
                log(f"pre1 alignment: {g} set {s:g}: Pre 1 not readable ({exc!r}); global shifts kept")
                continue
            base = [float(p1[c]) for c in SHIFT_COLS]
            for idx, row in sdf.iterrows():
                if idx == p1.name:
                    continue
                try:
                    v = _volume(str(row["file_path"]), ch_1or2)
                except Exception as exc:
                    log(f"pre1 alignment: {os.path.basename(str(row['file_path']))} not readable ({exc!r})")
                    continue
                if v.shape != v1.shape:
                    log(f"pre1 alignment: {os.path.basename(str(row['file_path']))} shape {v.shape} != "
                        f"Pre 1 {v1.shape}; global shift kept")
                    continue
                rel = shift_to_reference(v1, v, method)
                for c, b, r in zip(SHIFT_COLS, base, rel):
                    df.loc[idx, c] = b + r
                n_frames += 1
            n_sets += 1
        df["align_reference"] = REFERENCE_PRE1
        log(f"pre1 alignment: {n_frames} frames of {n_sets} sets registered to their Pre 1")
    after = df[list(SHIFT_COLS)].astype(float)
    changed = not np.allclose(before.to_numpy(), after.to_numpy(), equal_nan=True)
    return df, changed
