"""Score automatic ROI shifts against the manual Spine ROI, frame by frame.

For every set the auto (RESPAN) ROI is moved by an integer shift per frame
from one of several estimators:

* ``local_dendrite_shift`` two-stage: static offset of the pre mean against
  the group's reference stack plus a per-frame drift of the local field
  (spine pixels excluded), optionally gated by gain and magnitude;
* ``spine_template_tracking``: NCC of the spine itself (brightness
  invariant), a containment criterion (move only as far as needed to keep
  the spine blob inside the ROI), and a linear post-trajectory fit.

The moved ROI is compared with the manual ROI of the same frame: exact
match of the integer shift, centroid error in pixels, IoU, the post/pre
intensity ratio and the per-frame F/F0 trace on the saved aligned TIFF.
``trace_abs_diff`` (mean |F/F0 predicted - F/F0 manual| over post frames)
is the metric that matters for the quantification.

Outputs go next to the data, under
``<session of the first --df-path>/roi_align_qc/local_dendrite_shift_eval/``
(per_frame.csv, per_set.csv, summary.csv, per_session.csv, summary.png and
optional examples/*.png).

Usage:
    python evaluate_local_dendrite_shift.py --include-qc --example-variant gate_s15_d0 --n-examples 12
    python evaluate_local_dendrite_shift.py --df-path G:/.../combined_df_respan.pkl --cache-dir D:/scratch
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from local_dendrite_shift import (  # noqa: E402
    estimate_shifts,
    estimate_two_stage,
    masks_from_shifts,
    smooth_within_phase,
)
from spine_template_tracking import TrackParams, containment_veto, locate_static, track_spine  # noqa: E402
from manual_shift_dataset import (  # noqa: E402
    QC_SESSION_DF_PATHS,
    SESSION_DF_PATHS,
    SpineSet,
    iter_sets,
    mask_iou,
    reference_projection,
    translate_mask,
)

DEFAULT_OUT_DIR = r"G:\ImagingData\Tetsuya\20260701\auto1\roi_align_qc\local_dendrite_shift_eval"

# name -> kwargs. "mode" selects the estimator: "direct" matches every frame to
# the reference stack, "two_stage" adds a pre-mean offset and a local drift.
VARIANTS: dict[str, dict] = {
    "zero": {},
    "direct_r30": dict(mode="direct", window_radius=30),
    "direct_r30_g05": dict(mode="direct", window_radius=30, min_gain=0.05),
    "direct_full": dict(mode="direct", window_radius=0),
    "direct_full_g05": dict(mode="direct", window_radius=0, min_gain=0.05),
    "direct_full_g10": dict(mode="direct", window_radius=0, min_gain=0.10),
    "ts_full_r20": dict(mode="two_stage", static_radius=0, local_radius=20),
    "ts_full_r30": dict(mode="two_stage", static_radius=0, local_radius=30),
    "ts_full_r30_g05": dict(mode="two_stage", static_radius=0, local_radius=30, static_gain=0.05, local_gain=0.05),
    "ts_full_r30_g10": dict(mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10),
    "ts_full_r30_sg05": dict(mode="two_stage", static_radius=0, local_radius=30, static_gain=0.05, local_gain=0.02),
    "ts_full_full": dict(mode="two_stage", static_radius=0, local_radius=0),
    "ts_r40_r30": dict(mode="two_stage", static_radius=40, local_radius=30),
    "ts_full_r30_s6": dict(mode="two_stage", static_radius=0, local_radius=30, static_sigma=6.0, local_sigma=6.0),
    "ts_full_r30_sub": dict(mode="two_stage", static_radius=0, local_radius=30, subpixel=True),
    "ts_static_only": dict(mode="two_stage", static_radius=0, local_radius=30, local_gain=9.0),
    "ts_full_r30_prefree": dict(mode="two_stage", static_radius=0, local_radius=30, pre_only_static=False),
    # Drift-only: trust the auto ROI on pre, follow the field from the pre mean.
    "drift_full": dict(mode="two_stage", use_static=False, local_radius=0),
    "drift_full_g0": dict(mode="two_stage", use_static=False, local_radius=0, local_gain=0.0),
    "drift_full_g05": dict(mode="two_stage", use_static=False, local_radius=0, local_gain=0.05),
    "drift_full_sub": dict(mode="two_stage", use_static=False, local_radius=0, subpixel=True),
    "drift_full_s2": dict(mode="two_stage", use_static=False, local_radius=0, local_sigma=2.0),
    "drift_full_s5": dict(mode="two_stage", use_static=False, local_radius=0, local_sigma=5.0),
    "drift_r40": dict(mode="two_stage", use_static=False, local_radius=40),
    "drift_r30": dict(mode="two_stage", use_static=False, local_radius=30),
    "drift_r20": dict(mode="two_stage", use_static=False, local_radius=20),
    "drift_full_sm": dict(mode="two_stage", use_static=False, local_radius=0, smooth=True),
    # Magnitude-gated two-stage: move only when the refined peak is far from zero.
    "gate_s25_d15": dict(
        mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10,
        static_min_px=2.5, local_min_px=1.5,
    ),
    "gate_s15_d15": dict(
        mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10,
        static_min_px=1.5, local_min_px=1.5,
    ),
    "gate_s25_d25": dict(
        mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10,
        static_min_px=2.5, local_min_px=2.5,
    ),
    "gate_s20_d15_full": dict(
        mode="two_stage", static_radius=0, local_radius=0, static_gain=0.10, local_gain=0.10,
        static_min_px=2.0, local_min_px=1.5,
    ),
    "gate_drift_d15": dict(mode="two_stage", use_static=False, local_radius=0, local_gain=0.05, local_min_px=1.5),
    "gate_s25_d0": dict(
        mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10, static_min_px=2.5,
    ),
    "gate_s15_d0": dict(
        mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10, static_min_px=1.5,
    ),
    "ts_g10_veto10": dict(
        mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10, contain_veto=0.10,
    ),
    "ts_g10_veto0": dict(
        mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10, contain_veto=0.0,
    ),
    "gate_s15_d0_veto10": dict(
        mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10, static_min_px=1.5,
        contain_veto=0.10,
    ),
    "ts_g10_r0": dict(mode="two_stage", static_radius=0, local_radius=0, static_gain=0.10, local_gain=0.10),
    "ts_g05": dict(mode="two_stage", static_radius=0, local_radius=30, static_gain=0.05, local_gain=0.05),
    "ts_g10_s6": dict(mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10, static_sigma=6.0, local_sigma=6.0),
    "ts_g10_s2": dict(mode="two_stage", static_radius=0, local_radius=30, static_gain=0.10, local_gain=0.10, static_sigma=2.0, local_sigma=2.0),
    # Spine-template tracking (brightness-normalized NCC of the spine itself).
    # "static" selects the pre offset: "none" keeps the auto position,
    # "dendrite" uses the gated reference match (static_min_px 2.5).
    "tpl": dict(mode="template", static="none"),
    "tpl_nodend": dict(mode="template", static="none", dendrite_lambda=0.0),
    "tpl_lam1": dict(mode="template", static="none", dendrite_lambda=1.0),
    "tpl_d0": dict(mode="template", static="none", spine_dilation=0),
    "tpl_d4": dict(mode="template", static="none", spine_dilation=4),
    "tpl_raw": dict(mode="template", static="none", use_raw_spine=True),
    "tpl_s2": dict(mode="template", static="none", spine_sigma=2.0),
    "tpl_s8": dict(mode="template", static="none", spine_sigma=8.0),
    "tpl_px0": dict(mode="template", static="none", min_px=0.0),
    "tpl_px25": dict(mode="template", static="none", min_px=2.5),
    "tpl_g0": dict(mode="template", static="none", min_gain=0.0),
    "tpl_tol99": dict(mode="template", static="none", overlap_tolerance=10_000),
    "tpl_tol3": dict(mode="template", static="none", overlap_tolerance=3),
    "tpl_tp02": dict(mode="template", static="none", temporal_penalty=0.02),
    "tpl_pre": dict(mode="template", static="none", track_pre=True),
    "tpl_sdend": dict(mode="template", static="dendrite"),
    "tpl_sdend_pre": dict(mode="template", static="dendrite", track_pre=True),
    "tpl_sdend_lin": dict(mode="template", static="dendrite", post_fit="linear"),
    "tpl_sdend_lin_px0": dict(mode="template", static="dendrite", post_fit="linear", min_px=0.0),
    "tpl_sdend_lin_px1": dict(mode="template", static="dendrite", post_fit="linear", min_px=1.0),
    "tpl_sdend_med": dict(mode="template", static="dendrite", post_fit="median"),
    "tpl_sdend_lin_d4": dict(mode="template", static="dendrite", post_fit="linear", spine_dilation=4),
    "tpl_sdend_lin_lam1": dict(mode="template", static="dendrite", post_fit="linear", dendrite_lambda=1.0),
    "tpl_sdend_lin_lam02": dict(mode="template", static="dendrite", post_fit="linear", dendrite_lambda=0.2),
    "tpl_sdend_lin_s2": dict(mode="template", static="dendrite", post_fit="linear", spine_sigma=2.0),
    "tpl_lin": dict(mode="template", static="none", post_fit="linear"),
    "tpl_sdend0": dict(mode="template", static="dendrite_ungated"),
    "tpl_sdend0_lin": dict(mode="template", static="dendrite_ungated", post_fit="linear"),
    # Integral (signal-collecting) criterion: static on the pre mean, then tracking.
    "int_static_only": dict(mode="template", static="integral", min_gain=9.0),
    "int_static_only_r1": dict(mode="template", static="integral", min_gain=9.0, ring_width=1),
    "int_static_only_r3": dict(mode="template", static="integral", min_gain=9.0, ring_width=3),
    "int_static_only_hp": dict(mode="template", static="integral", min_gain=9.0, integral_sigma=6.0),
    "int_static_only_g0": dict(mode="template", static="integral", min_gain=9.0, static_gain=0.0),
    "int_static_only_g10": dict(mode="template", static="integral", min_gain=9.0, static_gain=0.10),
    "int_static_only_px25": dict(mode="template", static="integral", min_gain=9.0, static_min_px=2.5),
    "int_ncc_lin": dict(mode="template", static="integral", post_fit="linear"),
    "int_int_lin": dict(mode="template", static="integral", criterion="integral", post_fit="linear"),
    "int_int": dict(mode="template", static="integral", criterion="integral"),
    "int_both_lin": dict(mode="template", static="integral", criterion="both", post_fit="linear"),
    "int_both_lin_w05": dict(mode="template", static="integral", criterion="both", post_fit="linear", integral_weight=0.5),
    "int_both_lin_w2": dict(mode="template", static="integral", criterion="both", post_fit="linear", integral_weight=2.0),
    "int_both": dict(mode="template", static="integral", criterion="both"),
    "int_both_lin_px1": dict(mode="template", static="integral", criterion="both", post_fit="linear", min_px=1.0),
    # Containment: move the ROI only as far as needed to keep the spine blob inside it.
    "con_static": dict(mode="template", static="integral", static_criterion="contain", criterion="none"),
    "con_static_f04": dict(mode="template", static="integral", static_criterion="contain", criterion="none", blob_frac=0.4),
    "con_static_f06": dict(mode="template", static="integral", static_criterion="contain", criterion="none", blob_frac=0.6),
    "con_static_stay10": dict(mode="template", static="integral", static_criterion="contain", criterion="none", contain_stay_slack=0.10),
    "con_static_stay02": dict(mode="template", static="integral", static_criterion="contain", criterion="none", contain_stay_slack=0.02),
    "con_static_sm0": dict(mode="template", static="integral", static_criterion="contain", criterion="none", blob_smooth=0.0),
    "con_con": dict(mode="template", static="integral", static_criterion="contain", criterion="contain"),
    "con_con_lin": dict(mode="template", static="integral", static_criterion="contain", criterion="contain", post_fit="linear", min_px=0.0),
    "con_con_med": dict(mode="template", static="integral", static_criterion="contain", criterion="contain", post_fit="median", min_px=0.0),
    "con_con_stay10": dict(mode="template", static="integral", static_criterion="contain", criterion="contain", contain_stay_slack=0.10),
    "zero_con": dict(mode="template", static="none", criterion="contain"),
    "zero_con_lin": dict(mode="template", static="none", criterion="contain", post_fit="linear", min_px=0.0),
    "sdend_con": dict(mode="template", static="dendrite", criterion="contain"),
    "sdend_con_lin": dict(mode="template", static="dendrite", criterion="contain", post_fit="linear", min_px=0.0),
    "con06_con": dict(mode="template", static="integral", static_criterion="contain", criterion="contain", blob_frac=0.6),
    "con06_tpl_lin": dict(mode="template", static="integral", static_criterion="contain", blob_frac=0.6, post_fit="linear"),
}

# Shortlist used when --variants is not given. "gate_s15_d0" is the
# recommended default (dendrite two-stage, static gated at 1.5 px),
# "gate_s25_d0" the conservative choice (fewer moves the annotator did not
# make), "ts_full_r30_g10" the ungated reference, "tpl_sdend" the spine
# template tracker and "zero_con" the containment tracker.
DEFAULT_VARIANTS = ["zero", "gate_s15_d0", "gate_s25_d0", "ts_full_r30_g10", "tpl_sdend", "zero_con"]


def _cache_path(cache_dir: str | None, spine_set: SpineSet) -> str | None:
    if not cache_dir:
        return None
    safe = spine_set.set_id.replace("|", "__").replace("/", "_").replace("\\", "_")
    return os.path.join(cache_dir, f"{safe}_ref.npy")


def load_reference(spine_set: SpineSet, cache_dir: str | None) -> np.ndarray | None:
    path = _cache_path(cache_dir, spine_set)
    if path and os.path.exists(path):
        return np.load(path)
    reference = reference_projection(spine_set)
    if reference is not None and path:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, reference)
    return reference


def predict_variant(name: str, spine_set: SpineSet, reference: np.ndarray) -> np.ndarray:
    """Integer (T, 2) shift of the auto ROI for one variant."""
    if name == "zero":
        return np.zeros((spine_set.n_frames, 2), dtype=np.int64)
    kwargs = dict(VARIANTS[name])
    smooth = bool(kwargs.pop("smooth", False))
    mode = kwargs.pop("mode", "direct")
    if mode == "template":
        static_mode = kwargs.pop("static", "none")
        static = (0, 0)
        if static_mode in ("dendrite", "dendrite_ungated"):
            est = estimate_two_stage(
                reference,
                spine_set.after,
                spine_set.auto_spine,
                spine_set.phases,
                static_radius=0,
                local_radius=30,
                static_gain=0.10,
                local_gain=9.0,
                static_min_px=2.5 if static_mode == "dendrite" else 0.0,
            )
            pre = spine_set.phase_index("pre")
            index = int(pre[0]) if len(pre) else 0
            static = (int(est.shift[index, 0]), int(est.shift[index, 1]))
        elif static_mode == "integral":
            pre = spine_set.phase_index("pre")
            pre_mean = np.mean(spine_set.after[pre if len(pre) else [0]], axis=0)
            static = locate_static(
                pre_mean,
                spine_set.auto_spine,
                spine_set.auto_shaft,
                spine_set.other_spines,
                params=TrackParams(**kwargs),
            ).shift
        estimate = track_spine(
            spine_set.after,
            spine_set.phases,
            spine_set.auto_spine,
            spine_set.auto_shaft,
            spine_set.other_spines,
            static=static,
            params=TrackParams(**kwargs),
            elapsed_sec=spine_set.elapsed_sec,
        )
    elif mode == "two_stage":
        veto_drop = kwargs.pop("contain_veto", None)
        estimate = estimate_two_stage(reference, spine_set.after, spine_set.auto_spine, spine_set.phases, **kwargs)
        if veto_drop is not None:
            pre = spine_set.phase_index("pre")
            index = int(pre[0]) if len(pre) else 0
            static = (int(estimate.shift[index, 0]), int(estimate.shift[index, 1]))
            pre_mean = np.mean(spine_set.after[pre if len(pre) else [0]], axis=0)
            if containment_veto(
                pre_mean,
                spine_set.auto_spine,
                spine_set.auto_shaft,
                spine_set.other_spines,
                static,
                max_drop=float(veto_drop),
                params=TrackParams(blob_frac=0.6),
            ):
                estimate.shift[:, 0] -= static[0]
                estimate.shift[:, 1] -= static[1]
    else:
        estimate = estimate_shifts(reference, spine_set.after, spine_set.auto_spine, phases=spine_set.phases, **kwargs)
    shifts = estimate.shift
    if smooth:
        shifts = smooth_within_phase(shifts, spine_set.phases)
    return shifts


def ratio_post_pre(stack: np.ndarray, masks: np.ndarray, pre: np.ndarray, post: np.ndarray) -> float:
    """Mean ROI intensity over post divided by mean over pre, minus 1."""
    if len(pre) == 0 or len(post) == 0:
        return float("nan")
    means = np.array([float(stack[t][masks[t]].mean()) if masks[t].any() else np.nan for t in range(len(stack))])
    pre_mean = float(np.nanmean(means[pre]))
    if not np.isfinite(pre_mean) or abs(pre_mean) < 1e-9:
        return float("nan")
    return float(np.nanmean(means[post]) / pre_mean - 1.0)


def normalized_trace(stack: np.ndarray, masks: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """Per-frame mean ROI intensity divided by the pre mean (F/F0 trace)."""
    means = np.array([float(stack[t][masks[t]].mean()) if masks[t].any() else np.nan for t in range(len(stack))])
    pre_mean = float(np.nanmean(means[pre])) if len(pre) else float("nan")
    if not np.isfinite(pre_mean) or abs(pre_mean) < 1e-9:
        return np.full(len(stack), np.nan)
    return means / pre_mean


def score_set(spine_set: SpineSet, reference: np.ndarray, variants: list[str]) -> tuple[list[dict], list[dict]]:
    """Per-frame and per-set rows for every variant."""
    pre = spine_set.phase_index("pre")
    post = spine_set.phase_index("post")
    manual_ratio = ratio_post_pre(spine_set.after, spine_set.manual_spine, pre, post)
    manual_trace = normalized_trace(spine_set.after, spine_set.manual_spine, pre)
    frame_rows: list[dict] = []
    set_rows: list[dict] = []
    for name in variants:
        shifts = predict_variant(name, spine_set, reference)
        masks = masks_from_shifts(spine_set.auto_spine, shifts)
        pred_trace = normalized_trace(spine_set.after, masks, pre)
        trace_abs_diff = float(np.nanmean(np.abs(pred_trace[post] - manual_trace[post]))) if len(post) else float("nan")
        for t in range(spine_set.n_frames):
            manual = spine_set.correction[t]
            frame_rows.append(
                {
                    "variant": name,
                    "session": spine_set.session,
                    "set_id": spine_set.set_id,
                    "frame": t,
                    "phase": str(spine_set.phases[t]).strip().lower(),
                    "manual_dy": int(manual[0]),
                    "manual_dx": int(manual[1]),
                    "pred_dy": int(shifts[t, 0]),
                    "pred_dx": int(shifts[t, 1]),
                    "err_px": float(np.hypot(shifts[t, 0] - manual[0], shifts[t, 1] - manual[1])),
                    "exact": bool(shifts[t, 0] == manual[0] and shifts[t, 1] == manual[1]),
                    "iou": mask_iou(masks[t], spine_set.manual_spine[t]),
                    "pure_shift": bool(spine_set.pure_shift[t]),
                }
            )
        set_rows.append(
            {
                "variant": name,
                "session": spine_set.session,
                "set_id": spine_set.set_id,
                "spine_stem": spine_set.spine_stem,
                "n_frames": spine_set.n_frames,
                "manual_ratio": manual_ratio,
                "pred_ratio": ratio_post_pre(spine_set.after, masks, pre, post),
                "trace_abs_diff": trace_abs_diff,
                "manual_post_move_px": float(
                    np.max(np.hypot(*(spine_set.correction[post] - np.median(spine_set.correction[pre], axis=0)).T))
                )
                if len(pre) and len(post)
                else float("nan"),
            }
        )
    return frame_rows, set_rows


def summarize(frames: pd.DataFrame, sets: pd.DataFrame) -> pd.DataFrame:
    """One row per variant and phase, plus an 'all' phase and the ratio agreement."""
    rows: list[dict] = []
    sets = sets.copy()
    sets["ratio_abs_diff"] = (sets["pred_ratio"] - sets["manual_ratio"]).abs()
    frames = frames.copy()
    frames["manual_px"] = np.hypot(frames["manual_dy"], frames["manual_dx"])
    for variant, group in frames.groupby("variant", sort=False):
        set_rows = sets[sets["variant"] == variant]
        big_sets = set_rows[set_rows["manual_post_move_px"] >= 2.0]
        for phase, sub in [("all", group)] + list(group.groupby("phase")):
            large = sub[sub["manual_px"] >= 2.0]
            small = sub[sub["manual_px"] < 1.5]
            rows.append(
                {
                    "variant": variant,
                    "phase": phase,
                    "n_frames": len(sub),
                    "exact_frac": float(sub["exact"].mean()),
                    "mean_err_px": float(sub["err_px"].mean()),
                    "median_err_px": float(sub["err_px"].median()),
                    "frac_err_le1": float((sub["err_px"] <= 1.0).mean()),
                    "frac_err_ge2": float((sub["err_px"] >= 2.0).mean()),
                    "mean_iou": float(sub["iou"].mean()),
                    # Frames the human moved by >= 2 px: did the estimate get closer?
                    "n_large_manual": len(large),
                    "large_mean_err_px": float(large["err_px"].mean()) if len(large) else float("nan"),
                    "large_frac_err_le1": float((large["err_px"] <= 1.0).mean()) if len(large) else float("nan"),
                    # Frames the human left nearly alone: how often did the estimate wander >= 2 px?
                    "n_small_manual": len(small),
                    "small_frac_err_ge2": float((small["err_px"] >= 2.0).mean()) if len(small) else float("nan"),
                    "ratio_abs_diff_mean": float(set_rows["ratio_abs_diff"].mean()),
                    "ratio_abs_diff_median": float(set_rows["ratio_abs_diff"].median()),
                    "ratio_abs_diff_p90": float(set_rows["ratio_abs_diff"].quantile(0.9)),
                    "ratio_abs_diff_mean_bigmove_sets": float(big_sets["ratio_abs_diff"].mean()) if len(big_sets) else float("nan"),
                    "n_bigmove_sets": len(big_sets),
                    "trace_abs_diff_mean": float(set_rows["trace_abs_diff"].mean()),
                    "trace_abs_diff_median": float(set_rows["trace_abs_diff"].median()),
                }
            )
    return pd.DataFrame(rows)


def _crop(image: np.ndarray, cy: float, cx: float, half: int) -> np.ndarray:
    y0, x0 = int(round(cy)) - half, int(round(cx)) - half
    out = np.zeros((2 * half + 1, 2 * half + 1), dtype=image.dtype)
    ys0, ys1 = max(0, y0), min(image.shape[0], y0 + 2 * half + 1)
    xs0, xs1 = max(0, x0), min(image.shape[1], x0 + 2 * half + 1)
    out[ys0 - y0 : ys1 - y0, xs0 - x0 : xs1 - x0] = image[ys0:ys1, xs0:xs1]
    return out


def render_example(spine_set: SpineSet, shifts: np.ndarray, variant: str, out_path: str, half: int = 14) -> None:
    """Tiles of two pre, one uncaging and all post frames with auto/manual/predicted outlines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi

    pre = spine_set.phase_index("pre")
    cy, cx = np.argwhere(spine_set.auto_spine).mean(axis=0)
    frames = list(pre[:2]) + list(spine_set.phase_index("unc")[:1]) + list(spine_set.phase_index("post"))
    cols = max(1, len(frames))
    fig, axes = plt.subplots(2, cols, figsize=(2.0 * cols, 4.4), squeeze=False)
    pre_mean = spine_set.after[pre if len(pre) else [0]].mean(axis=0)
    vmin, vmax = np.percentile(_crop(pre_mean, cy, cx, half), [1, 99.5])
    for i, t in enumerate(frames):
        image = spine_set.after[t]
        predicted = translate_mask(spine_set.auto_spine, int(shifts[t, 0]), int(shifts[t, 1]))
        for k, im in enumerate((image, image - ndi.gaussian_filter(image, 3))):
            ax = axes[k, i]
            tile = _crop(im, cy, cx, half)
            lo, hi = (vmin, vmax) if k == 0 else np.percentile(tile, [1, 99.5])
            ax.imshow(tile, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.contour(_crop(spine_set.auto_spine.astype(float), cy, cx, half), levels=[0.5], colors="lime", linewidths=0.7)
            ax.contour(_crop(spine_set.manual_spine[t].astype(float), cy, cx, half), levels=[0.5], colors="orange", linewidths=0.9)
            ax.contour(_crop(predicted.astype(float), cy, cx, half), levels=[0.5], colors="magenta", linewidths=0.9, linestyles="dashed")
            if spine_set.auto_shaft is not None:
                ax.contour(_crop(spine_set.auto_shaft.astype(float), cy, cx, half), levels=[0.5], colors="cyan", linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                manual = spine_set.correction[t]
                ax.set_title(
                    f"f{t} {str(spine_set.phases[t])[:3]} manual({manual[0]:+d},{manual[1]:+d}) pred({shifts[t,0]:+d},{shifts[t,1]:+d})",
                    fontsize=6,
                )
    fig.suptitle(
        f"{spine_set.set_id} {spine_set.spine_stem} [{variant}] lime=auto orange=manual magenta=predicted cyan=shaft; row 2 highpass",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_summary_figure(sets: pd.DataFrame, variants: list[str], out_path: str) -> None:
    """Per-session mean |F/F0 trace difference| per variant and a manual-vs-predicted ratio scatter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = sets.copy()
    sets["ratio_abs_diff"] = (sets["pred_ratio"] - sets["manual_ratio"]).abs()
    table = sets.pivot_table(index="session", columns="variant", values="trace_abs_diff", aggfunc="mean")
    table = table[[v for v in variants if v in table.columns]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    table.plot.bar(ax=axes[0], width=0.8)
    axes[0].set_ylabel("mean |F/F0(pred) - F/F0(manual)| over post frames")
    axes[0].set_title("Per-session F/F0 trace disagreement (lower is better)")
    axes[0].legend(fontsize=7)
    axes[0].tick_params(axis="x", labelrotation=20, labelsize=8)
    best = [v for v in variants if v != "zero"]
    for name, color in (("zero", "gray"), (best[0] if best else "zero", "tab:red")):
        sub = sets[sets["variant"] == name]
        axes[1].scatter(sub["manual_ratio"], sub["pred_ratio"], s=10, alpha=0.6, label=name, color=color)
    lim = [-0.6, 3.0]
    axes[1].plot(lim, lim, "k--", linewidth=0.8)
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("manual post/pre - 1")
    axes[1].set_ylabel("predicted post/pre - 1")
    axes[1].set_title("Per-set F/F0 change: manual vs predicted ROI")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run(
    df_paths: list[str],
    out_dir: str,
    cache_dir: str | None,
    variants: list[str],
    limit: int | None,
    *,
    example_variant: str | None = None,
    n_examples: int = 0,
    example_min_move_px: float = 2.0,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    frame_rows: list[dict] = []
    set_rows: list[dict] = []
    started = time.time()
    n_sets = 0
    n_rendered = 0
    example_dir = os.path.join(out_dir, "examples")
    for spine_set in iter_sets(df_paths):
        reference = load_reference(spine_set, cache_dir)
        if reference is None:
            print(f"  {spine_set.set_id}: no reference stack, skip")
            continue
        frames, sets = score_set(spine_set, reference, variants)
        frame_rows.extend(frames)
        set_rows.extend(sets)
        if example_variant and n_rendered < n_examples:
            moved = next((s["manual_post_move_px"] for s in sets if s["variant"] == example_variant), float("nan"))
            if np.isfinite(moved) and moved >= example_min_move_px:
                os.makedirs(example_dir, exist_ok=True)
                safe = spine_set.set_id.replace("|", "_").replace("/", "_").replace("\\", "_")
                render_example(
                    spine_set,
                    predict_variant(example_variant, spine_set, reference),
                    example_variant,
                    os.path.join(example_dir, f"{safe}.png"),
                )
                n_rendered += 1
        n_sets += 1
        if n_sets % 20 == 0:
            print(f"  {n_sets} sets, {time.time() - started:.0f}s", flush=True)
        if limit and n_sets >= limit:
            break
    frames_df = pd.DataFrame(frame_rows)
    sets_df = pd.DataFrame(set_rows)
    frames_df.to_csv(os.path.join(out_dir, "per_frame.csv"), index=False)
    sets_df.to_csv(os.path.join(out_dir, "per_set.csv"), index=False)
    summary = summarize(frames_df, sets_df)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    per_session = (
        frames_df.groupby(["variant", "session"])
        .agg(exact_frac=("exact", "mean"), mean_err_px=("err_px", "mean"), mean_iou=("iou", "mean"))
        .reset_index()
    )
    per_session.to_csv(os.path.join(out_dir, "per_session.csv"), index=False)
    show_cols = [
        "variant",
        "phase",
        "mean_err_px",
        "frac_err_le1",
        "frac_err_ge2",
        "mean_iou",
        "large_mean_err_px",
        "small_frac_err_ge2",
        "ratio_abs_diff_mean",
        "ratio_abs_diff_median",
        "ratio_abs_diff_mean_bigmove_sets",
        "trace_abs_diff_mean",
        "trace_abs_diff_median",
    ]
    with pd.option_context("display.width", 300, "display.max_rows", 500):
        print(summary.loc[summary["phase"].isin(["pre", "post"]), show_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if len(sets_df):
        try:
            render_summary_figure(sets_df, variants, os.path.join(out_dir, "summary.png"))
        except Exception as exc:  # noqa: BLE001
            print(f"summary figure failed: {exc}")
    print(f"sets {n_sets}, examples {n_rendered}, wrote {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dendrite-based ROI shifts against manual ROIs")
    parser.add_argument("--df-path", action="append", default=None, help="combined_df_respan.pkl (repeatable)")
    parser.add_argument("--include-qc", action="store_true", help="Also score the 20260909 QC session")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cache-dir", default=None, help="Scratch folder for reference projections")
    parser.add_argument("--variants", default=None, help="Comma-separated subset of variant names")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many sets (smoke test)")
    parser.add_argument("--example-variant", default=None, help="Variant whose ROIs are drawn in example tiles")
    parser.add_argument("--n-examples", type=int, default=0, help="Number of example tile PNGs (sets with manual post move >= 2 px)")
    args = parser.parse_args()
    df_paths = args.df_path or list(SESSION_DF_PATHS)
    if args.include_qc:
        df_paths = df_paths + QC_SESSION_DF_PATHS
    out_dir = args.out_dir
    if out_dir is None:
        first = Path(df_paths[0])
        out_dir = str(first.parent / "roi_align_qc" / "local_dendrite_shift_eval") if args.df_path else DEFAULT_OUT_DIR
    variants = args.variants.split(",") if args.variants else list(DEFAULT_VARIANTS)
    run(
        df_paths,
        out_dir,
        args.cache_dir,
        variants,
        args.limit,
        example_variant=args.example_variant,
        n_examples=args.n_examples,
    )


if __name__ == "__main__":
    main()
