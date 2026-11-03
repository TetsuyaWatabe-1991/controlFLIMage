# -*- coding: utf-8 -*-
"""
Repeatability of manual ROI correction (intra-rater variability).

The same sets are corrected by hand several times, each round starting from the same
RESPAN auto ROIs, and the rounds are compared (ROI overlap, reject decisions, and the
quantified spine volume change).

Everything runs on an isolated copy of the selected sets, so the production files of
the session (<session>/tif/*, combined_df_respan.pkl, *_all_frames.csv) are never
modified:

    <session>/manual_roi_repeatability_<prefix>/
        work/                 copy the viewer edits (pkl, all-frames CSV, tif/)
        round0_auto/          RESPAN auto ROIs + their quantification (reference)
        round1/, round2/ ...  snapshot after each manual round
        compare/              comparison tables and figures
        state.json            round bookkeeping (start/end time, viewer time)

Sets whose production Spine ROI was already edited by hand get their auto ROIs
re-created (seg_masks for Shaft, RESPAN tracking for Spine) in the copy.

Usage:
    python manual_roi_repeatability.py setup   [--session DIR] [--prefix B6_0908]
    python manual_roi_repeatability.py start   --round 1   # reset to auto ROIs, open viewer
    python manual_roi_repeatability.py resume  --round 1   # continue the same round
    python manual_roi_repeatability.py collect --round 1   # snapshot (done after viewer closes)
    python manual_roi_repeatability.py compare
    python manual_roi_repeatability.py status
    python manual_roi_repeatability.py rebuild-auto --label d1   # re-create round0 with current auto ROI code
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

HERE = Path(__file__).resolve().parent
GUI_DIR = HERE.parent
for p in (str(GUI_DIR), str(GUI_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_SESSION = r"G:\ImagingData\Tetsuya\20261012\auto1"
DEFAULT_PREFIX = "B6_0908"
PKL_NAME = "combined_df_respan.pkl"
CSV_NAME = "combined_df_respan_intensity_lifetime_all_frames.csv"
ROI_TYPES = ["Spine", "DendriticShaft"]
PATH_COLUMNS = [
    "after_align_save_path",
    "after_align_full_save_path",
    "before_align_save_path",
    "before_align_full_save_path",
    "save_path_tzyx_aligned",
]
# Files beside each set TIFF that are copied into the work folder.
SET_FILE_SUFFIXES = [
    ".tif",
    "_frame_info.csv",
    "_alignment_r.csv",
]
# Per-set files that make up one round's ROI state.
ROUND_FILE_SUFFIXES = [
    "_Spine_roi_mask.tif",
    "_Spine_roi_mask_raw.tif",
    "_Spine_roi_mask_source.json",
    "_DendriticShaft_roi_mask.tif",
    "_DendriticShaft_roi_mask_raw.tif",
]
ROUND_OPTIONAL_SUFFIXES = ["_rejected.flag", "_gui_intensity.csv"]

# Same settings as analayze_all_flim_roi_respan.py (keep in step), except that only
# intensity is quantified: no uncaging lifetime, so no FLIM decode (intensity cache only).
CH_1OR2 = 2
Z_PLUS_MINUS = 1
PHOTON_THRESHOLD = 15
TOTAL_PHOTON_THRESHOLD = 300
SKIP_LIFETIME_ANALYSIS = True
UNCAGING_LIFETIME_BIN_FRAMES = None
UNCAGING_ROI_KEYFRAME_COUNT = 3
TIME_WINDOW_MIN = (-40, 50)

# Same summary as plot_auto_manual_swarmplot_respan.py / LTP analysis.
LTP_POST_MIN = (25, 35)
UNC_TOTAL_FRAME_FIRST_UNC = {33: 2, 55: 5}
PRE_INSTABILITY_MAX_MIN_RATIO = 1.5

# Categorical colors (validated default palette, fixed order).
COLOR_AUTO = "#52514e"
ROUND_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


# --------------------------------------------------------------------------- paths
@dataclass(frozen=True)
class Experiment:
    """Folder layout of one repeatability experiment."""

    session: Path
    prefix: str

    @property
    def root(self) -> Path:
        return self.session / f"manual_roi_repeatability_{self.prefix}"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def work_tif(self) -> Path:
        return self.work / "tif"

    @property
    def work_pkl(self) -> Path:
        return self.work / PKL_NAME

    @property
    def work_csv(self) -> Path:
        return self.work / CSV_NAME

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def compare_dir(self) -> Path:
        return self.root / "compare"

    def round_dir(self, n: int) -> Path:
        return self.root / ("round0_auto" if n == 0 else f"round{n}")

    def collected_rounds(self) -> list[int]:
        out = []
        for p in self.root.glob("round*"):
            name = p.name.replace("_auto", "")
            if name[5:].isdigit() and (p / CSV_NAME).exists():
                out.append(int(name[5:]))
        return sorted(out)


def load_state(exp: Experiment) -> dict:
    if exp.state_path.exists():
        return json.loads(exp.state_path.read_text(encoding="utf-8"))
    return {"rounds": {}, "active_round": None}


def save_state(exp: Experiment, state: dict) -> None:
    exp.state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")


def now_str() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def mask_sha(mask: np.ndarray) -> str:
    """Same hash as respan_tracked_spine_roi._sha (binary mask)."""
    return hashlib.sha1(np.ascontiguousarray(mask > 0).tobytes()).hexdigest()


def read_mask(path: Path) -> np.ndarray:
    a = tifffile.imread(str(path)) > 0
    return a[None] if a.ndim == 2 else a


# --------------------------------------------------------------------------- set selection
def select_sets(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rows of groups whose name starts with prefix."""
    return df[df["group"].astype(str).str.startswith(prefix)].copy()


def set_bases(df: pd.DataFrame) -> list[str]:
    """TIFF base names (no extension) of every set, in dataframe order."""
    col = "after_align_full_save_path" if "after_align_full_save_path" in df.columns else "after_align_save_path"
    out = []
    for p in df.loc[df["nth_set_label"] != -1, col].dropna().astype(str):
        b = os.path.splitext(os.path.basename(os.path.normpath(p)))[0]
        if b not in out:
            out.append(b)
    return out


def set_base(bases: list[str], group: str, set_label: float) -> str:
    """TIFF base of (group, set_label); group names end with '_' (e.g. 'x__highmag_1_' + '_0.0_...')."""
    head = f"{group}_{float(set_label):.1f}_"
    return next((b for b in bases if b.startswith(head)), "")


def remap_paths(df: pd.DataFrame, new_dir: Path) -> pd.DataFrame:
    """Point set TIFF path columns to new_dir when the file exists there."""
    out = df.copy()
    for col in PATH_COLUMNS:
        if col not in out.columns:
            continue

        def _one(p):
            if not isinstance(p, str):
                return p
            cand = new_dir / os.path.basename(os.path.normpath(p))
            return str(cand) if cand.exists() else p

        out[col] = out[col].map(_one)
    return out


# --------------------------------------------------------------------------- round files
def round_files(tif_dir: Path, bases: list[str]) -> list[Path]:
    files = []
    for b in bases:
        for suf in ROUND_FILE_SUFFIXES + ROUND_OPTIONAL_SUFFIXES:
            p = tif_dir / f"{b}{suf}"
            if p.exists():
                files.append(p)
    return files


def snapshot_round(exp: Experiment, n: int, bases: list[str]) -> Path:
    """Copy the work ROI state (masks, flags, CSV, pkl reject column) into round<n>."""
    dst = exp.round_dir(n)
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "tif").mkdir(parents=True)
    for p in round_files(exp.work_tif, bases):
        shutil.copy2(p, dst / "tif" / p.name)
    shutil.copy2(exp.work_csv, dst / CSV_NAME)
    rej = reject_table(exp.work_tif, bases)
    rej.to_csv(dst / "rejected_sets.csv", index=False)
    return dst


def reject_table(tif_dir: Path, bases: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"set": bases, "rejected": [(tif_dir / f"{b}_rejected.flag").exists() for b in bases]})


def reset_work_to_round0(exp: Experiment, bases: list[str]) -> None:
    """Restore the auto ROIs and auto quantification in the work folder."""
    src = exp.round_dir(0)
    if not (src / CSV_NAME).exists():
        raise RuntimeError(f"round0 snapshot missing: {src} (run setup first)")
    for b in bases:
        for suf in ROUND_FILE_SUFFIXES + ROUND_OPTIONAL_SUFFIXES:
            p = exp.work_tif / f"{b}{suf}"
            if p.exists():
                p.unlink()
            s = src / "tif" / f"{b}{suf}"
            if s.exists():
                shutil.copy2(s, p)
    shutil.copy2(src / CSV_NAME, exp.work_csv)
    if exp.work_pkl.exists():
        df = pd.read_pickle(exp.work_pkl)
        df["reject"] = 0
        df.to_pickle(exp.work_pkl)


# --------------------------------------------------------------------------- setup
def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _regenerate_auto_masks(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Run the workflow's auto ROI creation (seg_masks, then RESPAN tracking) into out_dir."""
    from gui_roi_respan_seg_masks import create_roi_masks_from_seg_masks
    from respan_tracked_spine_roi import apply_tracked_spine_rois

    out_dir.mkdir(parents=True, exist_ok=True)
    for b in set_bases(df):
        for suf in [".tif", "_frame_info.csv"]:
            src = exp_src_dir(df) / f"{b}{suf}"
            if src.exists() and not (out_dir / src.name).exists():
                _link_or_copy(src, out_dir / src.name)
    rdf = remap_paths(df, out_dir)
    create_roi_masks_from_seg_masks(rdf, skip_if_roi_mask_exists=False)
    return apply_tracked_spine_rois(rdf)


def exp_src_dir(df: pd.DataFrame) -> Path:
    p = df["after_align_full_save_path"].dropna().astype(str).iloc[0]
    return Path(os.path.dirname(os.path.normpath(p)))


def _quantify_all(df: pd.DataFrame, out_csv: Path) -> None:
    import gui_roi_fast_simple as g

    q = df.copy()
    q["reject"] = 0
    g.save_drift_corrected_roi_masks(q, roi_types=ROI_TYPES)
    g.quantify_intensity_from_flim(
        q, CH_1OR2, Z_PLUS_MINUS, str(out_csv),
        photon_threshold=PHOTON_THRESHOLD, total_photon_threshold=TOTAL_PHOTON_THRESHOLD,
        skip_lifetime_analysis=SKIP_LIFETIME_ANALYSIS, background_mode=g.BACKGROUND_MODE_MIP_P20,
        uncaging_lifetime_bin_frames=UNCAGING_LIFETIME_BIN_FRAMES,
    )


def setup(exp: Experiment, *, force: bool = False) -> None:
    """Build the work copy and the round0 (auto ROI) reference."""
    if exp.work.exists() and not force:
        raise RuntimeError(f"{exp.work} already exists (use --force to rebuild; collected rounds are kept)")
    prod_pkl = exp.session / PKL_NAME
    df = select_sets(pd.read_pickle(prod_pkl), exp.prefix)
    bases = set_bases(df)
    if not bases:
        raise RuntimeError(f"no sets with prefix {exp.prefix!r} in {prod_pkl}")
    src_dir = exp_src_dir(df)
    print(f"{len(bases)} sets, source {src_dir}")

    if exp.work.exists():
        shutil.rmtree(exp.work)
    exp.work_tif.mkdir(parents=True)
    for b in bases:
        for suf in SET_FILE_SUFFIXES + ROUND_FILE_SUFFIXES:
            s = src_dir / f"{b}{suf}"
            if s.exists():
                shutil.copy2(s, exp.work_tif / s.name)

    # Auto ROIs re-created for every set in a scratch folder; used for the sets edited
    # by hand in production, and as a check that re-creation reproduces the others.
    regen_dir = exp.root / "_regen_auto"
    if regen_dir.exists():
        shutil.rmtree(regen_dir)
    track = _regenerate_auto_masks(df, regen_dir)
    rows = []
    for b in bases:
        row = {"set": b}
        side = src_dir / f"{b}_Spine_roi_mask_source.json"
        prod_spine = src_dir / f"{b}_Spine_roi_mask.tif"
        sha_ok = side.exists() and prod_spine.exists() and \
            json.loads(side.read_text(encoding="utf-8")).get("sha1") == mask_sha(read_mask(prod_spine))
        row["production_spine_edited"] = not sha_ok
        for roi in ROI_TYPES:
            pp, rp = src_dir / f"{b}_{roi}_roi_mask.tif", regen_dir / f"{b}_{roi}_roi_mask.tif"
            row[f"{roi}_regen_equals_production"] = (
                pp.exists() and rp.exists() and np.array_equal(read_mask(pp), read_mask(rp)))
        if row["production_spine_edited"]:
            for suf in ["_Spine_roi_mask.tif", "_Spine_roi_mask_source.json", "_DendriticShaft_roi_mask.tif"]:
                s = regen_dir / f"{b}{suf}"
                if s.exists():
                    shutil.copy2(s, exp.work_tif / s.name)
            row["work_masks_from"] = "re-created"
        else:
            row["work_masks_from"] = "production"
        rows.append(row)
    check = pd.DataFrame(rows)
    exp.root.mkdir(parents=True, exist_ok=True)
    check.to_csv(exp.root / "setup_auto_roi_check.csv", index=False)
    if len(track):
        track.to_csv(exp.root / "setup_respan_tracking.csv", index=False)
    shutil.rmtree(regen_dir, ignore_errors=True)

    wdf = remap_paths(df, exp.work_tif)
    wdf["reject"] = 0
    wdf.to_pickle(exp.work_pkl)
    print("Quantifying all sets with the auto ROIs...")
    _quantify_all(wdf, exp.work_csv)
    snapshot_round(exp, 0, bases)

    state = load_state(exp)
    state.update(session=str(exp.session), prefix=exp.prefix, bases=bases, setup=now_str(), active_round=None)
    save_state(exp, state)
    pristine = check[~check.production_spine_edited]
    print("\n=== setup summary ===")
    print(f"sets: {len(bases)}; production Spine edited by hand (re-created): "
          f"{int(check.production_spine_edited.sum())} {list(check.loc[check.production_spine_edited, 'set'])}")
    for roi in ROI_TYPES:
        print(f"re-creation reproduces production {roi} ROI in unedited sets: "
              f"{int(pristine[f'{roi}_regen_equals_production'].sum())}/{len(pristine)}")
    print(f"round0 (auto) saved: {exp.round_dir(0)}")


def rebuild_auto(exp: Experiment, backup_label: str) -> None:
    """Re-create the round0 auto ROIs of every set with the current auto ROI code.

    The previous round0 is kept as round0_auto_<backup_label>; manual rounds are not touched.
    Use after changing the auto ROI method (e.g. respan_tracked_spine_roi.ROI_DILATE_PX).
    """
    state = load_state(exp)
    bases = state["bases"]
    backup = exp.root / f"round0_auto_{backup_label}"
    if backup.exists():
        raise RuntimeError(f"{backup} already exists; choose another --label")
    df = pd.read_pickle(exp.work_pkl)
    # regenerate from the production TIFF folder paths (frame_info / TIFF are identical copies)
    prod = select_sets(pd.read_pickle(exp.session / PKL_NAME), exp.prefix)
    regen_dir = exp.root / "_regen_auto"
    if regen_dir.exists():
        shutil.rmtree(regen_dir)
    track = _regenerate_auto_masks(prod, regen_dir)
    missing = [b for b in bases if not (regen_dir / f"{b}_Spine_roi_mask.tif").exists()]
    if missing:
        raise RuntimeError(f"auto ROI not re-created for {missing}")
    shutil.copytree(exp.round_dir(0), backup)
    for b in bases:
        for suf in ROUND_FILE_SUFFIXES + ROUND_OPTIONAL_SUFFIXES:
            p = exp.work_tif / f"{b}{suf}"
            if p.exists():
                p.unlink()
        for suf in ["_Spine_roi_mask.tif", "_Spine_roi_mask_source.json", "_DendriticShaft_roi_mask.tif"]:
            s = regen_dir / f"{b}{suf}"
            if s.exists():
                shutil.copy2(s, exp.work_tif / s.name)
    if len(track):
        track.to_csv(exp.root / f"rebuild_respan_tracking_{now_str().replace(':', '')}.csv", index=False)
    shutil.rmtree(regen_dir, ignore_errors=True)
    df["reject"] = 0
    df.to_pickle(exp.work_pkl)
    print("Quantifying all sets with the re-created auto ROIs...")
    _quantify_all(df, exp.work_csv)
    snapshot_round(exp, 0, bases)
    state["active_round"] = None
    state.setdefault("auto_rebuilds", []).append({"time": now_str(), "previous_kept_as": backup.name})
    save_state(exp, state)
    print(f"round0 re-created: {exp.round_dir(0)}  (previous: {backup})")


# --------------------------------------------------------------------------- rounds
def _viewer(exp: Experiment, n: int) -> float:
    """Open the set review viewer on the work copy (set order shuffled per round)."""
    from PyQt5.QtWidgets import QApplication

    from set_review_viewer import ReviewSession, make_viewer

    app = QApplication.instance() or QApplication(sys.argv)
    session = ReviewSession(
        str(exp.work_pkl), str(exp.work_csv), ch_1or2=CH_1OR2, z_plus_minus=Z_PLUS_MINUS,
        photon_threshold=PHOTON_THRESHOLD, total_photon_threshold=TOTAL_PHOTON_THRESHOLD,
        skip_lifetime_analysis=SKIP_LIFETIME_ANALYSIS, uncaging_roi_keyframe_count=UNCAGING_ROI_KEYFRAME_COUNT,
        show_alignment_r=True, time_window_min=TIME_WINDOW_MIN,
        uncaging_lifetime_bin_frames=UNCAGING_LIFETIME_BIN_FRAMES,
    )
    random.Random(1000 + n).shuffle(session.sets)
    print(f"Round {n}: {len(session.sets)} sets (order shuffled). Close the viewer when done.")
    viewer = make_viewer(session)
    viewer.setWindowTitle(f"[Repeatability round {n}] " + viewer.windowTitle())
    t0 = time.time()
    viewer.show()
    app.exec_()
    return time.time() - t0


def start_round(exp: Experiment, n: int, *, force: bool = False, open_viewer: bool = True) -> None:
    if n < 1:
        raise ValueError("round must be >= 1 (round 0 is the auto ROI reference)")
    state = load_state(exp)
    if exp.round_dir(n).exists() and not force:
        raise RuntimeError(f"round {n} already collected: {exp.round_dir(n)} (use --force to redo it)")
    open_n = state.get("active_round")
    if open_n not in (None, n) and not exp.round_dir(open_n).exists() and not force:
        raise RuntimeError(f"round {open_n} is still open and not collected; run collect --round {open_n} first")
    bases = state["bases"]
    reset_work_to_round0(exp, bases)
    state["active_round"] = n
    state["rounds"][str(n)] = {"started": now_str(), "viewer_seconds": 0.0}
    save_state(exp, state)
    print(f"Round {n}: work folder reset to the auto ROIs.")
    if open_viewer:
        _run_and_collect(exp, n)


def resume_round(exp: Experiment, n: int) -> None:
    state = load_state(exp)
    if state.get("active_round") != n:
        raise RuntimeError(f"round {n} is not the open round (open: {state.get('active_round')})")
    _run_and_collect(exp, n)


def _run_and_collect(exp: Experiment, n: int) -> None:
    sec = _viewer(exp, n)
    state = load_state(exp)
    state["rounds"][str(n)]["viewer_seconds"] = state["rounds"][str(n)].get("viewer_seconds", 0.0) + sec
    save_state(exp, state)
    print(f"Viewer open for {sec / 60:.1f} min.")
    collect_round(exp, n)
    print(f"If round {n} is not finished yet, continue with: resume --round {n}")


def collect_round(exp: Experiment, n: int) -> None:
    state = load_state(exp)
    bases = state["bases"]
    dst = snapshot_round(exp, n, bases)
    info = state["rounds"].setdefault(str(n), {})
    info["collected"] = now_str()
    rej = reject_table(exp.work_tif, bases)
    edited = sum(
        1 for b in bases
        if not np.array_equal(read_mask(exp.work_tif / f"{b}_Spine_roi_mask.tif"),
                              read_mask(exp.round_dir(0) / "tif" / f"{b}_Spine_roi_mask.tif"))
    )
    info.update(n_rejected=int(rej.rejected.sum()), n_spine_edited=int(edited))
    save_state(exp, state)
    print(f"Round {n} collected -> {dst}  (Spine edited: {edited}, rejected: {int(rej.rejected.sum())})")


def status(exp: Experiment) -> None:
    state = load_state(exp)
    print(f"experiment: {exp.root}")
    print(f"open round: {state.get('active_round')}")
    print(f"collected rounds: {exp.collected_rounds()}")
    for k, v in sorted(state.get("rounds", {}).items()):
        print(f"  round {k}: {v}")


# --------------------------------------------------------------------------- comparison
def set_summary(q: pd.DataFrame, ch: int = CH_1OR2) -> pd.DataFrame:
    """Per-set spine volume change etc. (same definition as the LTP analysis)."""
    rows = []
    for (grp, sl), s in q.groupby(["group", "set_label"], sort=False):
        s = s.copy()
        unc = s[s.phase == "unc"]
        pre = s[s.phase == "pre"]
        post = s[s.phase == "post"]
        if len(unc):
            k = UNC_TOTAL_FRAME_FIRST_UNC.get(len(unc), 1)
            t0 = unc["elapsed_time_sec"].iloc[k - 1]
        else:
            t0 = np.nan
        t = post["elapsed_time_sec"] - t0
        win = post[(t > LTP_POST_MIN[0] * 60) & (t < LTP_POST_MIN[1] * 60)]
        if not len(win):
            win = post[t > LTP_POST_MIN[0] * 60]
        row = {"group": grp, "set_label": sl, "set": f"{grp}_{sl:g}"}
        for roi, tag in (("Spine", "spine"), ("DendriticShaft", "shaft")):
            col = f"{roi}_Ch{ch}_intensity"
            f0 = pre[col].mean()
            row[f"{tag}_pre_F"] = f0
            row[f"{tag}_dFF0_post"] = win[col].mean() / f0 - 1 if len(win) else np.nan
            row[f"{tag}_dFF0_unc"] = (unc[col] / unc["nAveFrame"]).mean() / (pre[col] / pre["nAveFrame"]).mean() - 1
        pv = (pre[f"Spine_Ch{ch}_intensity"] / pre["nAveFrame"]).dropna()
        row["pre_unstable"] = bool(len(pv) >= 2 and pv.min() > 0 and pv.max() / pv.min() > PRE_INSTABILITY_MAX_MIN_RATIO)
        rows.append(row)
    return pd.DataFrame(rows)


def icc_2_1(x: np.ndarray) -> float:
    """ICC(2,1), two-way random, absolute agreement, single rater. x: n_subjects x k_raters."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 2 or k < 2:
        return np.nan
    gm = x.mean()
    ssr = k * ((x.mean(axis=1) - gm) ** 2).sum()
    ssc = n * ((x.mean(axis=0) - gm) ** 2).sum()
    sse = ((x - gm) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else np.nan


def mask_pair_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean per-frame IoU and mean centroid distance (px) of two Type-A mask stacks."""
    ious, dists = [], []
    for fa, fb in zip(a, b):
        u = (fa | fb).sum()
        if u == 0:
            continue
        ious.append((fa & fb).sum() / u)
        if fa.any() and fb.any():
            ca, cb = np.argwhere(fa).mean(0), np.argwhere(fb).mean(0)
            dists.append(float(np.hypot(*(ca - cb))))
    return (float(np.mean(ious)) if ious else np.nan, float(np.mean(dists)) if dists else np.nan)


def _label(n: int) -> str:
    return "auto" if n == 0 else f"R{n}"


def compare(exp: Experiment) -> None:
    rounds = exp.collected_rounds()
    manual = [r for r in rounds if r > 0]
    if 0 not in rounds or not manual:
        raise RuntimeError(f"need round0 and at least one manual round (have {rounds})")
    state = load_state(exp)
    bases = state["bases"]
    out = exp.compare_dir
    out.mkdir(parents=True, exist_ok=True)

    # per-set metrics, long format
    longs, rejects = [], {}
    for r in rounds:
        s = set_summary(pd.read_csv(exp.round_dir(r) / CSV_NAME))
        s["round"] = _label(r)
        rej = pd.read_csv(exp.round_dir(r) / "rejected_sets.csv")
        rejects[_label(r)] = dict(zip(rej.set, rej.rejected))
        longs.append(s)
    long = pd.concat(longs, ignore_index=True)
    long["base"] = [set_base(bases, g, s) for g, s in zip(long.group, long.set_label)]
    long["rejected"] = [bool(rejects[r].get(b, False)) for r, b in zip(long["round"], long["base"])]
    long.to_csv(out / "per_set_metrics_long.csv", index=False)

    metrics = ["spine_dFF0_post", "spine_dFF0_unc", "shaft_dFF0_post", "spine_pre_F"]
    labels = [_label(r) for r in rounds]
    mlabels = [_label(r) for r in manual]
    # sets used for agreement: accepted in every manual round
    acc = long[long["round"].isin(mlabels)].groupby("set")["rejected"].any()
    common = sorted(acc[~acc].index)

    summ = []
    for m in metrics:
        wide = long.pivot(index="set", columns="round", values=m).reindex(columns=labels)
        if wide.isna().all().all():
            continue
        w = wide.loc[wide.index.intersection(common)]
        row = {"metric": m, "n_sets_common_accepted": len(w)}
        mm = w[mlabels]
        row["manual_mean"] = float(np.nanmean(mm.to_numpy()))
        if len(mlabels) >= 2:
            row["ICC21_manual_rounds"] = icc_2_1(mm.to_numpy())
            row["within_set_SD_manual"] = float(np.sqrt(np.nanmean(mm.var(axis=1, ddof=1))))
            pair = [np.nanmean(np.abs(mm[a] - mm[b])) for a, b in combinations(mlabels, 2)]
            row["mean_abs_diff_manual_vs_manual"] = float(np.mean(pair))
            row["max_abs_diff_manual_vs_manual"] = float(np.nanmax([np.nanmax(np.abs(mm[a] - mm[b]))
                                                                     for a, b in combinations(mlabels, 2)]))
        row["mean_abs_diff_auto_vs_manual"] = float(np.mean([np.nanmean(np.abs(w["auto"] - w[r])) for r in mlabels]))
        row["ICC21_auto_and_manual"] = icc_2_1(w[labels].to_numpy())
        for r in labels:
            v = long[(long["round"] == r) & ~long["rejected"]][m].dropna()
            row[f"{r}_accepted_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{r}_accepted_sem"] = float(v.sem()) if len(v) > 1 else np.nan
            row[f"{r}_n_accepted"] = int(len(v))
        summ.append(row)
    summary = pd.DataFrame(summ)
    summary.to_csv(out / "metric_agreement_summary.csv", index=False)

    # ROI overlap
    mrows = []
    for b in bases:
        masks = {}
        for r in rounds:
            p = exp.round_dir(r) / "tif" / f"{b}_Spine_roi_mask.tif"
            if p.exists():
                masks[_label(r)] = read_mask(p)
        for a, c in combinations(masks, 2):
            if masks[a].shape != masks[c].shape:
                continue
            iou, dist = mask_pair_stats(masks[a], masks[c])
            kind = "auto_vs_manual" if "auto" in (a, c) else "manual_vs_manual"
            mrows.append({"set": b, "pair": f"{a}-{c}", "kind": kind, "spine_iou": iou, "centroid_dist_px": dist,
                          "identical": bool(np.array_equal(masks[a], masks[c]))})
    mask_df = pd.DataFrame(mrows)
    mask_df.to_csv(out / "spine_mask_agreement.csv", index=False)

    rej_table = pd.DataFrame({"set": bases, **{r: [bool(rejects[r].get(b, False)) for b in bases] for r in labels}})
    rej_table.to_csv(out / "reject_table.csv", index=False)

    _plots(long, summary, mask_df, labels, mlabels, common, out)
    _write_report(exp, state, long, summary, mask_df, rej_table, mlabels, common, out)


def _plots(long, summary, mask_df, labels, mlabels, common, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"auto": COLOR_AUTO, **{r: ROUND_COLORS[i % len(ROUND_COLORS)] for i, r in enumerate(mlabels)}}
    m = "spine_dFF0_post"
    wide = long.pivot(index="set", columns="round", values=m).reindex(columns=labels)
    rejw = long.pivot(index="set", columns="round", values="rejected").reindex(columns=labels)
    # rejected points are not plotted; sets without any accepted manual value are dropped
    shown = wide.mask(rejw.fillna(False).astype(bool))
    shown = shown[shown[mlabels].notna().any(axis=1)]
    order = shown[mlabels].mean(axis=1).sort_values().index

    # 1) per-set values across rounds
    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(order) + 2), 4.5))
    x = np.arange(len(order))
    for i, r in enumerate(labels):
        y = shown.loc[order, r].to_numpy()
        off = (i - (len(labels) - 1) / 2) * 0.12
        kw = dict(s=26, color=colors[r], zorder=3, edgecolors="white", linewidths=0.8)
        if r == "auto":
            kw.update(marker="D", s=22)
        ax.scatter(x + off, y, label=r, **kw)
    lo = shown.loc[order, mlabels].min(axis=1).to_numpy()
    hi = shown.loc[order, mlabels].max(axis=1).to_numpy()
    ax.vlines(x, lo, hi, color="#b5b4af", lw=2, zorder=1)
    ax.axhline(0, color="#d6d5d0", lw=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_pos1__highmag", " hm") for s in order], rotation=90, fontsize=7)
    ax.set_ylabel("Spine dF/F0 (post 25-35 min)")
    ax.set_title("Spine volume change per set across rounds (rejected sets not shown)")
    ax.legend(frameon=False, ncol=len(labels), loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "spine_dFF0_post_per_set.png", dpi=150)
    plt.close(fig)

    _beeswarm(wide, labels, common, colors, out)

    # 2) pairwise scatter (common accepted sets)
    pairs = list(combinations(mlabels, 2)) + [("auto", r) for r in mlabels]
    ncol = min(3, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow), squeeze=False)
    w = wide.loc[wide.index.intersection(common)].dropna()
    lim = [np.nanmin(w.to_numpy()), np.nanmax(w.to_numpy())]
    pad = 0.05 * (lim[1] - lim[0] if np.isfinite(lim).all() and lim[1] > lim[0] else 1)
    lim = [lim[0] - pad, lim[1] + pad]
    for ax, (a, b) in zip(axes.flat, pairs):
        ax.plot(lim, lim, color="#d6d5d0", lw=1, zorder=0)
        ax.scatter(w[a], w[b], s=22, color=colors[b], edgecolors="white", linewidths=0.8, zorder=2)
        d = np.nanmean(np.abs(w[a] - w[b]))
        ax.set_title(f"{a} vs {b}   mean|diff| = {d:.3f}", fontsize=9)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in list(axes.flat)[len(pairs):]:
        ax.axis("off")
    fig.suptitle(f"Spine dF/F0 (post), sets accepted in all manual rounds (n={len(w)})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "spine_dFF0_post_pairwise.png", dpi=150)
    plt.close(fig)

    # 3) ROI overlap distributions
    if len(mask_df):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, col, xl in ((axes[0], "spine_iou", "Spine ROI IoU (mean over frames)"),
                            (axes[1], "centroid_dist_px", "Spine ROI centroid distance (px)")):
            for kind, c in (("manual_vs_manual", ROUND_COLORS[0]), ("auto_vs_manual", COLOR_AUTO)):
                v = mask_df.loc[mask_df.kind == kind, col].dropna()
                if len(v):
                    ax.hist(v, bins=20, alpha=0.6, color=c, label=f"{kind.replace('_', ' ')} (n={len(v)})")
            ax.set_xlabel(xl)
            ax.set_ylabel("set pairs")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "spine_roi_overlap.png", dpi=150)
        plt.close(fig)


def _beeswarm(wide: pd.DataFrame, labels: list[str], common: list[str], colors: dict, out: Path) -> None:
    """Publication-style beeswarm of spine dF/F0, auto and each manual round (paired sets, mean +- SEM)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    w = wide.loc[wide.index.intersection(common), labels].dropna()
    long_w = w.reset_index().melt(id_vars="set", var_name="round", value_name="dff")
    names = {r: ("Auto\n(RESPAN)" if r == "auto" else f"Manual\n#{r[1:]}") for r in labels}
    long_w["name"] = long_w["round"].map(names)
    order = [names[r] for r in labels]
    with plt.rc_context({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        sns.swarmplot(data=long_w, x="name", y="dff", order=order, hue="name", hue_order=order,
                      palette={names[r]: colors[r] for r in labels}, size=3.6, edgecolor="white", linewidth=0.4,
                      legend=False, ax=ax, zorder=2)
        for k, r in enumerate(labels):
            v = w[r]
            m, se = v.mean(), v.sem()
            ax.hlines(m, k - 0.28, k + 0.28, color="black", lw=1.6, zorder=4)
            ax.errorbar(k, m, yerr=se, color="black", lw=1.2, capsize=4, capthick=1.2, zorder=4)
        ax.axhline(0, color="#b5b4af", lw=0.8, ls="--", zorder=0)
        ax.set_xlabel("")
        ax.set_ylabel(r"Spine $\Delta F/F_0$ (25-35 min)")
        ax.set_title(f"n = {len(w)} spines", fontsize=9)
        sns.despine(ax=ax)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"spine_dFF0_post_beeswarm.{ext}", dpi=300)
        plt.close(fig)
    w.to_csv(out / "spine_dFF0_post_beeswarm_data.csv")


def _write_report(exp, state, long, summary, mask_df, rej_table, mlabels, common, out: Path) -> None:
    lines = [f"Manual ROI repeatability: {exp.root}", f"generated {now_str()}", ""]
    lines.append("Rounds:")
    for k, v in sorted(state.get("rounds", {}).items()):
        lines.append(f"  R{k}: viewer {v.get('viewer_seconds', 0) / 60:.1f} min, Spine edited "
                     f"{v.get('n_spine_edited')}, rejected {v.get('n_rejected')}")
    lines.append(f"Sets: {len(state['bases'])}; accepted in all manual rounds: {len(common)}")
    rj = rej_table[mlabels]
    lines.append(f"Reject decisions: same in all manual rounds for {int((rj.all(axis=1) | ~rj.any(axis=1)).sum())}"
                 f"/{len(rj)} sets; rejected in some but not all: {int((rj.any(axis=1) & ~rj.all(axis=1)).sum())}")
    if len(mask_df):
        lines.append("")
        lines.append("Spine ROI overlap (mean IoU / centroid distance px / identical fraction):")
        for kind, g in mask_df.groupby("kind"):
            lines.append(f"  {kind}: {g.spine_iou.mean():.3f} / {g.centroid_dist_px.mean():.2f} / "
                         f"{g.identical.mean():.2f}  (n={len(g)})")
    lines.append("")
    lines.append("Quantification agreement (sets accepted in all manual rounds):")
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.float_format", "{:.4f}".format):
        cols = [c for c in summary.columns if not c.endswith(("_sem", "_n_accepted"))]
        lines.append(summary[cols].to_string(index=False))
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Outputs: {out}")


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["setup", "start", "resume", "collect", "compare", "status", "rebuild-auto"])
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--label", default=None, help="rebuild-auto: suffix for the kept previous round0")
    ap.add_argument("--no-viewer", action="store_true", help="start: reset only, do not open the viewer")
    a = ap.parse_args(argv)
    exp = Experiment(Path(a.session), a.prefix)
    if a.command in ("start", "resume", "collect") and a.round is None:
        ap.error("--round is required")
    if a.command == "setup":
        setup(exp, force=a.force)
    elif a.command == "start":
        start_round(exp, a.round, force=a.force, open_viewer=not a.no_viewer)
    elif a.command == "resume":
        resume_round(exp, a.round)
    elif a.command == "collect":
        collect_round(exp, a.round)
    elif a.command == "rebuild-auto":
        if not a.label:
            ap.error("--label is required (e.g. d1)")
        rebuild_auto(exp, a.label)
    elif a.command == "compare":
        compare(exp)
    else:
        status(exp)


if __name__ == "__main__":
    main()
