"""Score simple reject rules against the manual QC labels.

Human reject is any category other than 9. Thresholds are chosen on these
same labels, and again with each session held out. No classifier is trained.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, shift as ndimage_shift

_THIS = os.path.dirname(os.path.abspath(__file__))
_CONTROL = os.path.dirname(os.path.dirname(_THIS))
if _CONTROL not in sys.path:
    sys.path.insert(0, _CONTROL)
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from FLIMageAlignment import HIGHPASS_XY_SIGMA, flim_files_to_nparray, highpass_registration_yx  # noqa: E402
from qc_assignment_source import (  # noqa: E402
    SESSION_PKLS,
    first_assignment_flim,
    highmag_folder_name,
    normalize_set_label,
)

LABELS_CSV = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260701/auto1/roi_align_qc/reject_qc/qc_labels.csv"
FEATURE_CSV = os.path.join(_THIS, "qc_criteria_features.csv")
RING_RADIUS = 4

# Higher values reject, except the two brightness/correlation measures.
RULES = (
    ("assign_shift_px", True),
    ("assign_corr", False),
    ("drift_info_px", True),
    ("contrast_first", False),
    ("spine_drop", False),
    ("unc_dist_px", True),
)


def confusion(reject_true: np.ndarray, reject_pred: np.ndarray) -> dict[str, float]:
    """Sensitivity is the reject catch rate. Specificity is the keep catch rate."""
    truth = np.asarray(reject_true, dtype=bool)
    pred = np.asarray(reject_pred, dtype=bool)
    tp = int(np.sum(truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))
    fp = int(np.sum(~truth & pred))
    return {
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "n": int(truth.size),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }


def apply_threshold(values: np.ndarray, threshold: float, higher_rejects: bool) -> np.ndarray:
    """Missing measurements do not reject."""
    values = np.asarray(values, dtype=float)
    pred = np.zeros(values.shape, dtype=bool)
    ok = np.isfinite(values)
    if higher_rejects:
        pred[ok] = values[ok] >= threshold
    else:
        pred[ok] = values[ok] <= threshold
    return pred


def best_threshold(
    values: np.ndarray,
    reject_true: np.ndarray,
    higher_rejects: bool,
) -> tuple[float, dict[str, float]]:
    """Threshold with the largest sensitivity + specificity - 1, on finite rows."""
    values = np.asarray(values, dtype=float)
    truth = np.asarray(reject_true, dtype=bool)
    ok = np.isfinite(values)
    if not np.any(ok):
        return float("nan"), confusion(truth, np.zeros(truth.shape, dtype=bool))
    best_t = float(values[ok][0])
    best_j = -np.inf
    best = confusion(truth, np.zeros(truth.shape, dtype=bool))
    for threshold in np.unique(values[ok]):
        stats = confusion(truth, apply_threshold(values, float(threshold), higher_rejects))
        youden = stats["sensitivity"] + stats["specificity"] - 1.0
        if youden > best_j:
            best_j = youden
            best_t = float(threshold)
            best = stats
    return best_t, best


def drift_span_yx(y_shift: np.ndarray, x_shift: np.ndarray) -> float:
    """Largest XY move relative to the first finite sample, in the same units."""
    y_shift = np.asarray(y_shift, dtype=float)
    x_shift = np.asarray(x_shift, dtype=float)
    ok = np.isfinite(y_shift) & np.isfinite(x_shift)
    if not np.any(ok):
        return float("nan")
    y_shift = y_shift[ok]
    x_shift = x_shift[ok]
    return float(np.max(np.hypot(y_shift - y_shift[0], x_shift - x_shift[0])))


def distance_to_mask(x_coord: float, y_coord: float, mask: np.ndarray) -> float:
    """Pixels from a point to the nearest positive mask pixel. Inside is 0."""
    binary = np.asarray(mask > 0, dtype=np.uint8)
    if binary.sum() == 0 or not np.isfinite(x_coord) or not np.isfinite(y_coord):
        return float("nan")
    dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
    height, width = dist.shape
    xi = int(round(float(x_coord)))
    yi = int(round(float(y_coord)))
    if 0 <= yi < height and 0 <= xi < width:
        return float(dist[yi, xi])
    ys, xs = np.nonzero(binary)
    return float(np.min(np.hypot(ys - float(y_coord), xs - float(x_coord))))


def roi_contrast(image: np.ndarray, mask: np.ndarray, radius: int = RING_RADIUS) -> float:
    """Median inside the ROI divided by the median of a ring just outside it."""
    roi = np.asarray(mask > 0)
    if int(roi.sum()) < 5:
        return float("nan")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(roi.astype(np.uint8), kernel) > 0
    ring = dilated & ~roi
    if int(ring.sum()) < 5:
        return float("nan")
    image = np.asarray(image, dtype=float)
    ring_level = float(np.median(image[ring]))
    if ring_level <= 0:
        return float("nan")
    return float(np.median(image[roi]) / ring_level)


def highpass_corr(reference: np.ndarray, moving: np.ndarray, dy: float, dx: float) -> float:
    """Pearson correlation after shifting ``moving`` onto ``reference``."""
    ref_hp = np.asarray(reference, dtype=np.float32) - gaussian_filter(np.asarray(reference, dtype=np.float32), HIGHPASS_XY_SIGMA)
    mov_hp = np.asarray(moving, dtype=np.float32) - gaussian_filter(np.asarray(moving, dtype=np.float32), HIGHPASS_XY_SIGMA)
    shifted = ndimage_shift(mov_hp, shift=(float(dy), float(dx)), order=1, mode="constant", cval=np.nan)
    valid = np.isfinite(shifted)
    if int(valid.sum()) < 100:
        return float("nan")
    left = ref_hp[valid]
    right = shifted[valid]
    if float(left.std()) < 1e-6 or float(right.std()) < 1e-6:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def combine_rules(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    """Reject when any one rule fires."""
    pred = np.zeros(len(frame), dtype=bool)
    for name, higher in RULES:
        pred |= apply_threshold(frame[name].to_numpy(), thresholds[name], higher)
    return pred


def fit_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["human_reject"].to_numpy(dtype=bool)
    found = {}
    for name, higher in RULES:
        threshold, _stats = best_threshold(frame[name].to_numpy(), truth, higher)
        found[name] = threshold
    return found


def leave_one_session_out(frame: pd.DataFrame) -> dict[str, float]:
    """Fit thresholds on the other sessions and score the held-out session."""
    parts = []
    for session in sorted(frame["session"].unique()):
        train = frame[frame["session"] != session]
        test = frame[frame["session"] == session]
        pred = combine_rules(test, fit_thresholds(train))
        stats = confusion(test["human_reject"].to_numpy(dtype=bool), pred)
        stats["session"] = session
        parts.append(stats)
    tp = sum(item["tp"] for item in parts)
    fn = sum(item["fn"] for item in parts)
    tn = sum(item["tn"] for item in parts)
    fp = sum(item["fp"] for item in parts)
    pooled = confusion(
        np.array([True] * (tp + fn) + [False] * (tn + fp)),
        np.array([True] * tp + [False] * fn + [False] * tn + [True] * fp),
    )
    pooled["folds"] = parts
    return pooled


def _projection(stack: np.ndarray, index: int) -> np.ndarray:
    frame = np.asarray(stack[index])
    if frame.ndim == 3:
        return frame.max(axis=0)
    return frame


def _mask_at(mask: np.ndarray, index: int) -> np.ndarray:
    if mask.ndim == 2:
        return mask
    if index < mask.shape[0]:
        return mask[index]
    return mask[0]


def _pre_post(info: pd.DataFrame) -> pd.DataFrame:
    phase = info["phase"].astype(str).str.lower()
    return info[phase.isin(["pre", "post"])]


def _assignment_max(path: str, cache: dict[str, np.ndarray]) -> np.ndarray | None:
    if path in cache:
        return cache[path]
    with contextlib.redirect_stdout(io.StringIO()):
        stack, _info, _times = flim_files_to_nparray([path], ch=1)
    volume = np.asarray(stack[0])
    while volume.ndim > 2:
        volume = volume.max(axis=0)
    cache[path] = np.asarray(volume, dtype=np.float32)
    return cache[path]


def features_for_row(label: pd.Series, pooled: dict[str, pd.DataFrame], flim_cache: dict[str, np.ndarray]) -> dict:
    """One labeled sheet. Image features are NaN when a file is missing."""
    session = str(label["session"])
    group = str(label["group"])
    set_label = normalize_set_label(label["set_label"])
    out = {
        "image_rel": label["image_rel"],
        "session": session,
        "group": group,
        "set_label": set_label,
        "category": int(label["category"]),
        "human_reject": int(label["category"]) != 9,
    }
    for name, _higher in RULES:
        out[name] = np.nan
    out["drift_df_px"] = np.nan
    out["drift_local_px"] = np.nan
    frame = pooled.get(session)
    if frame is None:
        return out
    rows = frame[(frame["group"].astype(str) == group) & (frame["nth_set_label"].map(normalize_set_label) == set_label)]
    if rows.empty:
        return out
    phase = rows["phase"].astype(str).str.lower()
    timed = rows[phase.isin(["pre", "post"])]
    if not timed.empty:
        out["drift_df_px"] = drift_span_yx(timed["shift_y"].to_numpy(), timed["shift_x"].to_numpy())
        out["drift_local_px"] = drift_span_yx(timed["small_shift_y"].to_numpy(), timed["small_shift_x"].to_numpy())
    after = next((path for path in rows["after_align_full_save_path"] if isinstance(path, str) and path), "")
    if not after or not os.path.isfile(after):
        return out
    base = os.path.splitext(after)[0]
    info_path = base + "_frame_info.csv"
    mask_path = base + "_Spine_roi_mask.tif"
    if not os.path.isfile(info_path) or not os.path.isfile(mask_path):
        return out
    info = pd.read_csv(info_path)
    stack = np.asarray(__import__("tifffile").imread(after))
    mask = np.asarray(__import__("tifffile").imread(mask_path))
    shown = _pre_post(info)
    if shown.empty:
        return out
    out["drift_info_px"] = drift_span_yx(shown["shift_y"].to_numpy(), shown["shift_x"].to_numpy())
    first = shown.iloc[0]
    first_index = int(first["frame"]) if "frame" in shown.columns else int(shown.index[0])
    first_image = _projection(stack, first_index)
    first_mask = _mask_at(mask, first_index)
    out["contrast_first"] = roi_contrast(first_image, first_mask)
    later = []
    for _, item in shown.iloc[1:].iterrows():
        index = int(item["frame"]) if "frame" in shown.columns else int(item.name)
        later.append(roi_contrast(_projection(stack, index), _mask_at(mask, index)))
    later = np.asarray(later, dtype=float)
    if np.isfinite(later).any() and np.isfinite(out["contrast_first"]) and out["contrast_first"] > 0:
        out["spine_drop"] = float(np.nanmin(later) / out["contrast_first"])
    unc_x = pd.to_numeric(rows["uncaging_display_x"], errors="coerce")
    unc_y = pd.to_numeric(rows["uncaging_display_y"], errors="coerce")
    if not unc_x.notna().any():
        unc_x = pd.to_numeric(rows["corrected_uncaging_x"], errors="coerce")
        unc_y = pd.to_numeric(rows["corrected_uncaging_y"], errors="coerce")
    if unc_x.notna().any():
        out["unc_dist_px"] = distance_to_mask(float(unc_x.dropna().iloc[0]), float(unc_y.dropna().iloc[0]), first_mask)
    root = os.path.dirname(SESSION_PKLS[session])
    flim_path = first_assignment_flim(root, highmag_folder_name(group))
    if flim_path and os.path.isfile(flim_path):
        assign = _assignment_max(flim_path, flim_cache)
        if assign.shape != first_image.shape:
            assign = cv2.resize(assign, (first_image.shape[1], first_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        dy, dx = highpass_registration_yx(first_image, assign)
        out["assign_shift_px"] = float(np.hypot(dy, dx))
        out["assign_corr"] = highpass_corr(first_image, assign, dy, dx)
    return out


def collect_features(labels_csv: str = LABELS_CSV) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    pooled = {session: pd.read_pickle(path) for session, path in SESSION_PKLS.items() if os.path.isfile(path)}
    cache: dict[str, np.ndarray] = {}
    rows = []
    for number, (_idx, label) in enumerate(labels.iterrows(), start=1):
        rows.append(features_for_row(label, pooled, cache))
        if number % 20 == 0 or number == len(labels):
            print(f"features {number}/{len(labels)}", flush=True)
    return pd.DataFrame(rows)


def format_report(frame: pd.DataFrame) -> str:
    """Text table of each rule and the combined or-rule."""
    truth = frame["human_reject"].to_numpy(dtype=bool)
    lines = [
        f"n={len(frame)}  human_reject={int(truth.sum())}  human_keep={int((~truth).sum())}",
        "",
        "each rule, threshold fit on all labels",
        f"{'rule':<18} {'thr':>8} {'sens':>6} {'spec':>6} {'tp':>4} {'fn':>4} {'fp':>4} {'tn':>4}",
    ]
    thresholds = {}
    for name, higher in RULES:
        threshold, stats = best_threshold(frame[name].to_numpy(), truth, higher)
        thresholds[name] = threshold
        lines.append(
            f"{name:<18} {threshold:8.3f} {stats['sensitivity']:6.3f} {stats['specificity']:6.3f} "
            f"{stats['tp']:4.0f} {stats['fn']:4.0f} {stats['fp']:4.0f} {stats['tn']:4.0f}"
        )
    combined = confusion(truth, combine_rules(frame, thresholds))
    lines.append("")
    lines.append(
        "combined OR, same labels  "
        f"sens={combined['sensitivity']:.3f} spec={combined['specificity']:.3f} "
        f"tp={combined['tp']} fn={combined['fn']} fp={combined['fp']} tn={combined['tn']}"
    )
    caught = frame.loc[combine_rules(frame, thresholds), "category"]
    missed = frame.loc[truth & ~combine_rules(frame, thresholds), "category"]
    lines.append("caught by category " + _counts(caught))
    lines.append("missed rejects by category " + _counts(missed))
    held = leave_one_session_out(frame)
    lines.append("")
    lines.append(
        "combined OR, leave-one-session-out  "
        f"sens={held['sensitivity']:.3f} spec={held['specificity']:.3f} "
        f"tp={held['tp']} fn={held['fn']} fp={held['fp']} tn={held['tn']}"
    )
    for fold in held["folds"]:
        lines.append(
            f"  hold out {fold['session']:<10} sens={fold['sensitivity']:.3f} spec={fold['specificity']:.3f} "
            f"tp={fold['tp']} fn={fold['fn']} fp={fold['fp']} tn={fold['tn']}"
        )
    lines.append("")
    lines.append("median by human category")
    medians = frame.groupby("category")[[name for name, _higher in RULES]].median()
    lines.append(medians.to_string(float_format=lambda value: f"{value:7.3f}"))
    return "\n".join(lines)


def _counts(categories: pd.Series) -> str:
    if categories.empty:
        return "(none)"
    counts = categories.value_counts().sort_index()
    return " ".join(f"{int(cat)}:{int(n)}" for cat, n in counts.items())


def main() -> None:
    frame = collect_features()
    frame.to_csv(FEATURE_CSV, index=False)
    print(format_report(frame))
    print(FEATURE_CSV)


if __name__ == "__main__":
    main()
