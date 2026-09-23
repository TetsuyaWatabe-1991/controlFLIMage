"""Diagnose why saved alignments leave a moving spine, and score simple fixes.

Reads existing combined_df pickles, aligned TIFFs, ROI masks, and the
quantification CSV. Does not modify those files. Writes a new ``roi_align_qc``
folder next to the pickle (the folder that also holds the .flim files).

Usage:
    python diagnose_alignment.py --df-path G:/.../combined_df_respan.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import find_boundaries

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ASI = Path(__file__).resolve().parents[3] / "ongoing" / "ASIcontroller"
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if str(_ASI) not in sys.path:
    sys.path.insert(0, str(_ASI))

from respan_uncaging_log import parse_uncaging_records  # noqa: E402
from tracking import (  # noqa: E402
    adjacent_window_shifts,
    best_sign_error,
    centroid_trajectory,
    classify_alignment,
    delta_over_pre,
    full_fov_shifts,
    mask_centroid,
    mask_sums,
    masks_from_shift,
    mean_trajectory_error,
    pairwise_content_shifts,
    series_correlation,
    max_frame_step,
    track_patch,
    trajectory_span,
)

LARGE_MANUAL_SPAN_PX = 3.0
ROI_HALF_PX = 30
PATCH_HALF_PX = 12
PATCH_SEARCH_PX = 8


def _highmag_folder(filepath_without_number: str) -> str:
    folder = os.path.dirname(str(filepath_without_number))
    stem = os.path.basename(str(filepath_without_number)).rstrip("_")
    return os.path.join(folder, stem)


def _match_record(set_df: pd.DataFrame, records: list):
    pre = set_df[set_df["phase"] == "pre"].sort_values("nth_omit_induction")
    if len(pre) == 0 or not records:
        return None
    target = os.path.basename(str(pre.iloc[-1]["file_path"])).lower()
    for rec in records:
        if os.path.basename(str(rec.flim_path)).lower() == target:
            return rec
    return None


def _load_stack(path: str) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    stack = tifffile.imread(path)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        return None
    return np.asarray(stack, dtype=np.float32)


def _before_path(after_path: str, set_df: pd.DataFrame) -> str | None:
    if "before_align_full_save_path" in set_df.columns:
        candidate = set_df["before_align_full_save_path"].iloc[0]
        if isinstance(candidate, str) and os.path.exists(candidate):
            return candidate
    guessed = after_path.replace("after_align", "before_align")
    if guessed != after_path and os.path.exists(guessed):
        return guessed
    return None


def _phase_index(info: pd.DataFrame, names: set[str]) -> np.ndarray:
    phase = info["phase"].astype(str).str.lower()
    return np.flatnonzero(phase.isin(names).to_numpy())


def _post_ltp_index(info: pd.DataFrame) -> np.ndarray:
    """Post frames 25-35 min after the first uncaging row, when times exist."""
    post = _phase_index(info, {"post"})
    if "elapsed_time_sec" not in info.columns or len(post) == 0:
        return post
    elapsed = pd.to_numeric(info["elapsed_time_sec"], errors="coerce").to_numpy(dtype=float)
    unc = _phase_index(info, {"unc", "uncaging"})
    if len(unc) == 0 or not np.isfinite(elapsed[unc[0]]):
        return post
    aligned = elapsed - float(elapsed[unc[0]])
    window = post[(aligned[post] > 25 * 60) & (aligned[post] < 35 * 60)]
    return window if len(window) else post


def _draw_boundary(ax: plt.Axes, mask: np.ndarray, color: str) -> None:
    if mask is None or not np.any(mask):
        return
    edge = find_boundaries(mask.astype(bool), mode="outer")
    ys, xs = np.nonzero(edge)
    ax.scatter(xs, ys, s=2, c=color, linewidths=0)


def _save_example(
    out_path: str,
    after: np.ndarray,
    manual: np.ndarray,
    tracked_masks: np.ndarray,
    manual_traj: np.ndarray,
    track_traj: np.ndarray,
    spine_before: np.ndarray | None,
    applied: np.ndarray | None,
    title: str,
) -> None:
    last = len(after) - 1
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, frame_idx, label in (
        (axes[0, 0], 0, "frame 0"),
        (axes[0, 1], last, "last frame"),
    ):
        image = after[frame_idx]
        vmin, vmax = np.percentile(image, [2, 98])
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        _draw_boundary(ax, manual[frame_idx], "lime")
        _draw_boundary(ax, tracked_masks[frame_idx], "red")
        ax.set_title(f"{label}  lime=manual red=tracked", fontsize=8)
        ax.axis("off")
    frames = np.arange(len(manual_traj))
    for ax, col, name in ((axes[1, 0], 0, "dy"), (axes[1, 1], 1, "dx")):
        ax.plot(frames, manual_traj[:, col], label="manual on aligned", color="limegreen")
        ax.plot(frames, track_traj[:, col], label="patch track on aligned", color="red")
        if spine_before is not None:
            ax.plot(frames, spine_before[: len(frames), col], label="spine on before", color="black", lw=0.8)
        if applied is not None:
            ax.plot(frames, applied[: len(frames), col], label="applied content shift", color="C0", lw=0.8)
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_ylabel(name + " (px)")
        ax.set_xlabel("frame")
        ax.legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _csv_delta(
    quant: pd.DataFrame | None,
    group: str,
    set_label: int,
    ch: int,
) -> float:
    if quant is None:
        return float("nan")
    col = f"Spine_Ch{ch}_intensity"
    if col not in quant.columns:
        return float("nan")
    sub = quant[(quant["group"].astype(str) == str(group)) & (quant["set_label"].astype(int) == int(set_label))]
    if len(sub) == 0 or "nAveFrame" not in sub.columns:
        return float("nan")
    values = sub[col].to_numpy(dtype=float) / sub["nAveFrame"].to_numpy(dtype=float)
    phase = sub["phase"].astype(str).str.lower()
    pre = np.flatnonzero((phase == "pre").to_numpy())
    post = np.flatnonzero((phase == "post").to_numpy())
    return delta_over_pre(values, pre, post)


def diagnose_set(
    set_df: pd.DataFrame,
    *,
    group: str,
    set_label: int,
    session: str,
    quant: pd.DataFrame | None,
    ch: int,
    example_dir: str | None,
) -> dict | None:
    after_path = set_df["after_align_full_save_path"].iloc[0] if "after_align_full_save_path" in set_df else None
    if not isinstance(after_path, str) or not os.path.exists(after_path):
        after_path = set_df["after_align_save_path"].iloc[0]
    if not isinstance(after_path, str) or not os.path.exists(after_path):
        return None
    after = _load_stack(after_path)
    if after is None:
        return None

    tiff_dir = os.path.dirname(after_path)
    base = os.path.splitext(os.path.basename(after_path))[0]
    manual_path = os.path.join(tiff_dir, f"{base}_Spine_roi_mask.tif")
    info_path = os.path.join(tiff_dir, f"{base}_frame_info.csv")
    if not os.path.exists(manual_path) or not os.path.exists(info_path):
        return None
    manual = tifffile.imread(manual_path)
    if manual.ndim == 2:
        manual = np.repeat(manual[None, ...], len(after), axis=0)
    manual = np.asarray(manual > 0)
    info = pd.read_csv(info_path)
    n_frames = min(len(after), len(manual), len(info))
    after = after[:n_frames]
    manual = manual[:n_frames]
    info = info.iloc[:n_frames].reset_index(drop=True)

    manual_traj = centroid_trajectory(manual)
    manual_span = trajectory_span(manual_traj)
    cy0, cx0 = mask_centroid(manual[0])
    if not np.isfinite(cy0):
        return None

    tracked = track_patch(after, cy0, cx0, half=PATCH_HALF_PX, search=PATCH_SEARCH_PX)
    track_err = mean_trajectory_error(tracked.shift_yx, manual_traj)

    pre_idx = _phase_index(info, {"pre"})
    post_idx = _phase_index(info, {"post"})
    if len(pre_idx) == 0:
        pre_idx = np.array([0])
    if len(post_idx) == 0:
        post_idx = np.array([n_frames - 1])
    ltp_idx = _post_ltp_index(info)
    if len(ltp_idx) == 0:
        ltp_idx = post_idx

    manual_sums = mask_sums(after, manual)
    tracked_masks = masks_from_shift(manual[0], tracked.shift_yx)
    tracked_sums = mask_sums(after, tracked_masks)

    seg = None
    initial_offset = float("nan")
    highmag = ""
    if "filepath_without_number" in set_df.columns:
        highmag = _highmag_folder(str(set_df["filepath_without_number"].iloc[0]))
        records = parse_uncaging_records(highmag) if os.path.isdir(highmag) else []
        rec = _match_record(set_df, records)
        if rec is not None:
            seg_path = Path(highmag) / "seg_masks" / f"{rec.spine_stem}_spine_outline_mask.tif"
            if seg_path.is_file():
                seg_img = np.asarray(tifffile.imread(str(seg_path)) > 0)
                if seg_img.shape == after.shape[1:]:
                    seg = seg_img
                    sy, sx = mask_centroid(seg)
                    initial_offset = float(np.hypot(sy - cy0, sx - cx0))

    static_base = seg if seg is not None else manual[0]
    static_masks = np.repeat(static_base[None, ...], n_frames, axis=0)
    static_sums = mask_sums(after, static_masks)
    if seg is not None:
        seg_y, seg_x = mask_centroid(seg)
        seg_track = track_patch(after, seg_y, seg_x, half=PATCH_HALF_PX, search=PATCH_SEARCH_PX)
        seg_tracked_sums = mask_sums(after, masks_from_shift(seg, seg_track.shift_yx))
    else:
        seg_tracked_sums = np.full(n_frames, np.nan)

    before = _load_stack(_before_path(after_path, set_df) or "")
    spine_before = None
    applied = None
    fixed = follow = fullfov = phase_fixed = None
    phase_error_mean = float("nan")
    tiff_vs_stored_err = float("nan")
    stored_sign = 0
    frame0_mae = float("nan")
    if before is not None:
        before = before[:n_frames]
        frame0_mae = float(np.mean(np.abs(before[0] - after[0])))
        spine_before = track_patch(before, cy0, cx0, half=PATCH_HALF_PX, search=PATCH_SEARCH_PX).shift_yx
        fixed, _ = adjacent_window_shifts(before, cy0, cx0, half=ROI_HALF_PX, follow=False)
        follow, _ = adjacent_window_shifts(before, cy0, cx0, half=ROI_HALF_PX, follow=True)
        fullfov = full_fov_shifts(before)
        phase_fixed, phase_err = adjacent_window_shifts(
            before, cy0, cx0, half=ROI_HALF_PX, follow=False, normalization="phase"
        )
        phase_error_mean = float(np.mean(phase_err[1:])) if len(phase_err) > 1 else float("nan")
        tiff_shift = pairwise_content_shifts(before, after)
        if "shift_y" in info.columns and "shift_x" in info.columns:
            stored = np.column_stack(
                [
                    pd.to_numeric(info["shift_y"], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(info["shift_x"], errors="coerce").to_numpy(dtype=float),
                ]
            )
            stored = stored - stored[0]
            tiff_vs_stored_err, stored_sign = best_sign_error(stored, tiff_shift)
            applied = stored_sign * stored
        else:
            applied = tiff_shift

    def _err(est: np.ndarray | None, target: np.ndarray | None) -> float:
        if est is None or target is None:
            return float("nan")
        return mean_trajectory_error(est, target)

    spine_span = trajectory_span(spine_before) if spine_before is not None else float("nan")
    stored_span = trajectory_span(applied) if applied is not None else float("nan")
    cancel_residual = _err(
        None if applied is None or spine_before is None else spine_before + applied,
        np.zeros_like(manual_traj),
    )
    same_dir_residual = _err(
        None if applied is None or spine_before is None else spine_before - applied,
        np.zeros_like(manual_traj),
    )
    # Error of applied content shift against the shift that would cancel each motion estimate.
    stored_vs_spine = _err(None if applied is None or spine_before is None else applied, None if spine_before is None else -spine_before)
    stored_vs_fullfov = _err(None if applied is None or fullfov is None else applied, None if fullfov is None else -fullfov)
    if before is None:
        if manual_span < 1.5 and track_err < 2.0:
            label = "stable"
        elif track_err < 2.0:
            label = "residual_trackable_no_before_stack"
        else:
            label = "human_edit_not_rigid_drift"
    else:
        label = classify_alignment(
            manual_span=manual_span,
            track_vs_manual_err=track_err,
            tiff_vs_stored_err=tiff_vs_stored_err if np.isfinite(tiff_vs_stored_err) else 0.0,
            spine_span=spine_span if np.isfinite(spine_span) else 0.0,
            stored_span=stored_span if np.isfinite(stored_span) else 0.0,
            stored_vs_spine_err=stored_vs_spine if np.isfinite(stored_vs_spine) else 0.0,
            fixed_vs_spine_err=_err(fixed, spine_before),
            follow_adj_vs_spine_err=_err(follow, spine_before),
            fullfov_vs_spine_err=_err(fullfov, spine_before),
            stored_vs_fullfov_err=stored_vs_fullfov if np.isfinite(stored_vs_fullfov) else 99.0,
            cancel_residual=cancel_residual if np.isfinite(cancel_residual) else 99.0,
            same_dir_residual=same_dir_residual if np.isfinite(same_dir_residual) else 99.0,
            manual_residual=manual_span,
        )
    phase_span = trajectory_span(phase_fixed) if phase_fixed is not None else float("nan")
    if (
        label not in ("stable", "human_edit_not_rigid_drift", "recorded_shift_disagrees_with_tiff")
        and np.isfinite(phase_span)
        and phase_span < 1.5
        and np.isfinite(spine_span)
        and spine_span > 3.0
        and np.isfinite(phase_error_mean)
        and phase_error_mean > 0.4
    ):
        label = "phase_correlation_blind_in_roi_crop"
    if label == "stable" and np.isfinite(initial_offset) and initial_offset > 3.0:
        label = "stable_drift_but_initial_mask_offset"
    applied_step = max_frame_step(applied)
    manual_step = max_frame_step(manual_traj)
    phase_step = max_frame_step(phase_fixed)
    if (
        np.isfinite(applied_step)
        and applied_step >= 4.0
        and manual_step <= 2.0
        and manual_span >= LARGE_MANUAL_SPAN_PX
    ):
        label = "applied_shift_jumps_while_roi_is_steady"

    row = {
        "session": session,
        "group": group,
        "set_label": int(set_label),
        "n_frames": n_frames,
        "cause": label,
        "manual_span_px": manual_span,
        "track_vs_manual_err_px": track_err,
        "track_score_median": float(np.median(tracked.score)),
        "initial_seg_offset_px": initial_offset,
        "frame0_before_after_mae": frame0_mae,
        "spine_before_span_px": spine_span,
        "applied_shift_span_px": stored_span,
        "tiff_vs_stored_err_px": tiff_vs_stored_err,
        "stored_sign": stored_sign,
        "fixed_vs_spine_err_px": _err(fixed, spine_before),
        "follow_adj_vs_spine_err_px": _err(follow, spine_before),
        "fullfov_vs_spine_err_px": _err(fullfov, spine_before),
        "phase_fixed_span_px": phase_span,
        "phase_error_mean": phase_error_mean,
        "applied_max_step_px": applied_step,
        "manual_max_step_px": manual_step,
        "phase_max_step_px": phase_step,
        "cancel_residual_px": cancel_residual,
        "same_dir_residual_px": same_dir_residual,
        "delta_ff0_csv": _csv_delta(quant, group, int(set_label), ch),
        "delta_ff0_manual_tiff": delta_over_pre(manual_sums, pre_idx, ltp_idx),
        "delta_ff0_static_tiff": delta_over_pre(static_sums, pre_idx, ltp_idx),
        "delta_ff0_tracked_manual0_tiff": delta_over_pre(tracked_sums, pre_idx, ltp_idx),
        "delta_ff0_tracked_seg_tiff": delta_over_pre(seg_tracked_sums, pre_idx, ltp_idx),
        "corr_tracked_vs_manual_tiff": series_correlation(tracked_sums, manual_sums),
        "corr_static_vs_manual_tiff": series_correlation(static_sums, manual_sums),
        "manual_post_span_px": trajectory_span(manual_traj[post_idx] - manual_traj[post_idx[0]])
        if len(post_idx)
        else float("nan"),
    }
    # Post-only span above subtracts the first post frame, which hides drift that
    # happened by the start of post. Also store max distance during post from frame 0.
    if len(post_idx):
        row["manual_post_from_start_px"] = trajectory_span(manual_traj[post_idx])
    else:
        row["manual_post_from_start_px"] = float("nan")

    if example_dir and manual_span >= LARGE_MANUAL_SPAN_PX:
        safe = f"{group}_set{int(set_label)}".replace(os.sep, "_").replace(":", "")
        _save_example(
            os.path.join(example_dir, f"{safe}.png"),
            after,
            manual,
            tracked_masks,
            manual_traj,
            tracked.shift_yx,
            spine_before,
            applied,
            f"{session} {group} set {int(set_label)}  {label}  span={manual_span:.1f}px",
        )
    return row


def _summarize(table: pd.DataFrame) -> str:
    lines = [f"sets {len(table)}"]
    if len(table) == 0:
        return "\n".join(lines)
    lines.append("cause counts:")
    lines.append(table["cause"].value_counts().to_string())
    large = table[table["manual_span_px"] >= LARGE_MANUAL_SPAN_PX]
    small = table[table["manual_span_px"] < LARGE_MANUAL_SPAN_PX]
    lines.append(f"large manual span (>={LARGE_MANUAL_SPAN_PX} px): {len(large)}")
    if len(large):
        lines.append(large["cause"].value_counts().to_string())
        lines.append(
            "large-set medians: "
            f"manual_span={large['manual_span_px'].median():.2f} "
            f"spine_before={large['spine_before_span_px'].median():.2f} "
            f"applied_shift={large['applied_shift_span_px'].median():.2f} "
            f"track_err={large['track_vs_manual_err_px'].median():.2f} "
            f"phase_span={large['phase_fixed_span_px'].median():.2f} "
            f"phase_error={large['phase_error_mean'].median():.2f} "
            f"applied_step={large['applied_max_step_px'].median():.2f} "
            f"manual_step={large['manual_max_step_px'].median():.2f} "
            f"fixed_err={large['fixed_vs_spine_err_px'].median():.2f} "
            f"follow_err={large['follow_adj_vs_spine_err_px'].median():.2f} "
            f"fullfov_err={large['fullfov_vs_spine_err_px'].median():.2f}"
        )
    lines.append(f"small manual span: {len(small)}")

    def _abs_delta(frame: pd.DataFrame, col: str) -> pd.Series:
        return (frame[col] - frame["delta_ff0_manual_tiff"]).abs()

    for name, frame in (("all", table), ("large", large)):
        if len(frame) == 0:
            continue
        static_err = _abs_delta(frame, "delta_ff0_static_tiff")
        tracked_err = _abs_delta(frame, "delta_ff0_tracked_manual0_tiff")
        seg_err = _abs_delta(frame, "delta_ff0_tracked_seg_tiff")
        lines.append(
            f"{name} |dFF0| vs manual-on-tiff: "
            f"static median={static_err.median():.3f} "
            f"patch-from-manual0 median={tracked_err.median():.3f} "
            f"patch-from-seg median={seg_err.median():.3f} "
            f"frac patch-manual0 better={float((tracked_err < static_err).mean()):.2f} "
            f"corr static={frame['corr_static_vs_manual_tiff'].median():.3f} "
            f"corr patch={frame['corr_tracked_vs_manual_tiff'].median():.3f}"
        )
        if frame["delta_ff0_csv"].notna().any():
            csv_vs_manual = (frame["delta_ff0_csv"] - frame["delta_ff0_manual_tiff"]).abs()
            csv_vs_patch = (frame["delta_ff0_csv"] - frame["delta_ff0_tracked_manual0_tiff"]).abs()
            csv_vs_static = (frame["delta_ff0_csv"] - frame["delta_ff0_static_tiff"]).abs()
            lines.append(
                f"{name} |dFF0| vs FLIM csv: "
                f"tiff-manual median={csv_vs_manual.median():.3f} "
                f"static median={csv_vs_static.median():.3f} "
                f"patch-from-manual0 median={csv_vs_patch.median():.3f}"
            )
    return "\n".join(lines)


def run_session(df_path: str, out_dir: str | None, ch: int) -> pd.DataFrame:
    combined = pd.read_pickle(df_path)
    session = Path(df_path).parent.name + "_" + Path(df_path).parents[1].name
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(df_path), "roi_align_qc")
    os.makedirs(out_dir, exist_ok=True)
    example_dir = os.path.join(out_dir, "large_shift_examples")
    os.makedirs(example_dir, exist_ok=True)

    quant_path = df_path.replace(".pkl", "_intensity_lifetime_all_frames.csv")
    quant = pd.read_csv(quant_path) if os.path.exists(quant_path) else None

    rows: list[dict] = []
    n_sets = 0
    for _filepath_wo, filegroup in combined.groupby("filepath_without_number", sort=False):
        for group, group_df in filegroup.groupby("group", sort=False):
            for set_label, set_df in group_df.groupby("nth_set_label", sort=False):
                if int(set_label) < 0:
                    continue
                n_sets += 1
                row = diagnose_set(
                    set_df,
                    group=str(group),
                    set_label=int(set_label),
                    session=session,
                    quant=quant,
                    ch=ch,
                    example_dir=example_dir,
                )
                if row is not None:
                    rows.append(row)
                print(f"  {group} set {int(set_label)}: {'ok' if row else 'skip'}", flush=True)

    table = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "alignment_diagnosis.csv")
    table.to_csv(csv_path, index=False)
    summary = _summarize(table)
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary + "\n")
    print(summary)
    print(f"wrote {csv_path}")
    print(f"candidates seen {n_sets}, diagnosed {len(table)}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose residual spine drift after alignment")
    parser.add_argument("--df-path", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--ch", type=int, default=2)
    args = parser.parse_args()
    run_session(args.df_path, args.out_dir, args.ch)


if __name__ == "__main__":
    main()
