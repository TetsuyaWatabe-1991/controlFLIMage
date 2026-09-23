"""Split sheets into confident reject, confident include, and human review.

A confident reject is allowed to contain at most 2 percent of human keeps
(specificity at least 98 percent). A confident include is allowed to contain
at most 2 percent of human rejects. Everything else stays for review.
Category 4 may stay in review; it is not required to be caught.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from export_algo_disagreements import SHEET_ROOT, annotate_sheet  # noqa: E402
from qc_assignment_source import SESSION_PKLS, normalize_set_label  # noqa: E402
from qc_label_store import CATEGORIES  # noqa: E402
from score_qc_criteria import FEATURE_CSV, _mask_at, _projection  # noqa: E402

OUT_DIR = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260701/auto1/roi_align_qc/algo_reclass/confident"
EXTRA_CSV = os.path.join(_THIS, "qc_confident_features.csv")
KEEP_CATEGORY = 9
# 3/195 = 1.5 percent of keeps. 2/97 = 2.1 percent of rejects, so the include cap is 1.
MAX_KEEPS_IN_REJECT = 3
MAX_REJECTS_IN_INCLUDE = 1
WINDOW_PAD = 16

# Higher values are worse. Correlation and contrast are flipped before use.
WORSE_HIGH = (
    "assign_shift_px",
    "unc_dist_px",
    "drift_info_px",
    "low_assign_corr",
    "low_contrast",
    "low_spine_drop",
    "low_z_match",
    "low_z_neighbor",
    "bead_fraction",
    "bright_cv",
)


def local_corr(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray, pad: int = WINDOW_PAD) -> float:
    """Pearson correlation of two images in a window around the ROI."""
    image_a = np.asarray(image_a, dtype=float)
    image_b = np.asarray(image_b, dtype=float)
    if image_b.shape != image_a.shape:
        image_b = cv2.resize(image_b.astype(np.float32), (image_a.shape[1], image_a.shape[0]), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if ys.size == 0:
        return float("nan")
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, image_a.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, image_a.shape[1])
    left = image_a[y0:y1, x0:x1].ravel()
    right = image_b[y0:y1, x0:x1].ravel()
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 20:
        return float("nan")
    left = left[valid]
    right = right[valid]
    if float(left.std()) < 1e-6 or float(right.std()) < 1e-6:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def bead_fraction(image: np.ndarray) -> float:
    """Share of bright pixels that sit in small round blobs."""
    values = np.asarray(image, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 50:
        return float("nan")
    bright = (values >= np.percentile(finite, 92)).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(bright, 8)
    round_area = 0
    total = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        total += area
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        aspect = min(width, height) / max(width, height, 1)
        if aspect >= 0.65 and area <= 40:
            round_area += area
    if total == 0:
        return float("nan")
    return round_area / total


def bright_cv(image: np.ndarray) -> float:
    """Coefficient of variation of the brightest pixels."""
    values = np.asarray(image, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 50:
        return float("nan")
    bright = finite[finite >= np.percentile(finite, 80)]
    mean = float(bright.mean())
    if mean <= 0:
        return float("nan")
    return float(bright.std() / mean)


def _frame_index(row: pd.Series) -> int:
    if "frame" in row.index and pd.notna(row["frame"]):
        return int(row["frame"])
    return int(row.name)


def image_features_for_set(info: pd.DataFrame, stack: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Z agreement near the ROI, and a first-frame beading score."""
    phase = info["phase"].astype(str).str.lower()
    pre = info[phase.eq("pre")]
    unc = info[phase.isin(["unc", "uncaging"])]
    out = {"z_match_corr": np.nan, "z_neighbor_corr": np.nan, "bead_fraction": np.nan, "bright_cv": np.nan}
    if pre.empty:
        return out
    first = pre.iloc[0]
    first_image = _projection(stack, _frame_index(first))
    first_mask = _mask_at(mask, _frame_index(first))
    out["bead_fraction"] = bead_fraction(first_image)
    out["bright_cv"] = bright_cv(first_image)
    unc_images = [_projection(stack, _frame_index(row)) for _, row in unc.iterrows()]
    if unc_images:
        matched = [local_corr(first_image, image, first_mask) for image in unc_images]
        out["z_match_corr"] = float(np.nanmedian(np.asarray(matched, dtype=float)))
    if len(unc_images) >= 2:
        neighbors = [
            local_corr(unc_images[index], unc_images[index + 1], first_mask)
            for index in range(len(unc_images) - 1)
        ]
        out["z_neighbor_corr"] = float(np.nanmedian(np.asarray(neighbors, dtype=float)))
    return out


def add_worse_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns where a larger number is a worse sheet."""
    out = frame.copy()
    out["low_assign_corr"] = -out["assign_corr"]
    out["low_contrast"] = -out["contrast_first"]
    out["low_spine_drop"] = -out["spine_drop"]
    out["low_z_match"] = -out["z_match_corr"]
    out["low_z_neighbor"] = -out["z_neighbor_corr"]
    return out


def _keep_limit(values: np.ndarray, keep: np.ndarray, taken: int) -> float:
    """Cutoff that newly includes ``taken`` keeps, from the worst keep downward."""
    keep_values = np.sort(values[keep & np.isfinite(values)])[::-1]
    if taken <= 0 or taken > keep_values.size:
        return float("inf")
    return float(keep_values[taken - 1])


def reject_mask_for_limits(frame: pd.DataFrame, limits: dict[str, int]) -> np.ndarray:
    """Reject when any feature reaches the worst ``limits[name]`` keeps."""
    keep = frame["category"].to_numpy() == KEEP_CATEGORY
    pred = np.zeros(len(frame), dtype=bool)
    for name in WORSE_HIGH:
        taken = int(limits.get(name, 0))
        if taken <= 0:
            cutoff = _keep_limit(frame[name].to_numpy(dtype=float), keep, 0)
            values = frame[name].to_numpy(dtype=float)
            finite_keeps = values[keep & np.isfinite(values)]
            if finite_keeps.size == 0:
                continue
            pred |= np.isfinite(values) & (values > float(np.max(finite_keeps)))
            continue
        cutoff = _keep_limit(frame[name].to_numpy(dtype=float), keep, taken)
        values = frame[name].to_numpy(dtype=float)
        pred |= np.isfinite(values) & (values >= cutoff)
    return pred


def absolute_reject_cutoffs(frame: pd.DataFrame, limits: dict[str, int]) -> dict[str, tuple[float, bool]]:
    """Freeze cutoffs on this frame so they can be applied to another day.

    The second value is True when the test is strict greater-than, which is
    how a zero keep-budget excludes every keep used to set the cutoff.
    """
    keep = frame["category"].to_numpy() == KEEP_CATEGORY
    cutoffs = {}
    for name, taken in limits.items():
        values = frame[name].to_numpy(dtype=float)
        finite = np.sort(values[keep & np.isfinite(values)])[::-1]
        if finite.size == 0:
            continue
        if int(taken) <= 0:
            cutoffs[name] = (float(finite[0]), True)
        else:
            cutoffs[name] = (float(finite[min(int(taken), finite.size) - 1]), False)
    return cutoffs


def apply_reject_cutoffs(frame: pd.DataFrame, cutoffs: dict[str, tuple[float, bool]]) -> np.ndarray:
    """Apply cutoffs learned on another set of sheets."""
    pred = np.zeros(len(frame), dtype=bool)
    for name, (cutoff, strict) in cutoffs.items():
        values = frame[name].to_numpy(dtype=float)
        if strict:
            pred |= np.isfinite(values) & (values > cutoff)
        else:
            pred |= np.isfinite(values) & (values >= cutoff)
    return pred


def fit_reject_limits(frame: pd.DataFrame, max_keeps: int = MAX_KEEPS_IN_REJECT) -> dict[str, int]:
    """Spend a keep budget on the features that catch the most rejects."""
    limits = {name: 0 for name in WORSE_HIGH}
    keep = frame["category"].to_numpy() == KEEP_CATEGORY
    reject = ~keep
    current = reject_mask_for_limits(frame, limits)
    while int((current & keep).sum()) < max_keeps:
        best_name = ""
        best_gain = 0
        best_mask = current
        for name in WORSE_HIGH:
            trial = dict(limits)
            trial[name] = limits[name] + 1
            mask = reject_mask_for_limits(frame, trial)
            if int((mask & keep).sum()) > max_keeps:
                continue
            gain = int((mask & reject).sum()) - int((current & reject).sum())
            if gain > best_gain:
                best_gain = gain
                best_name = name
                best_mask = mask
        if best_gain <= 0:
            break
        limits[best_name] += 1
        current = best_mask
    return limits


def include_mask_for_limits(frame: pd.DataFrame, limits: dict[str, float]) -> np.ndarray:
    """Include only sheets that are normal on every measured feature."""
    pred = np.ones(len(frame), dtype=bool)
    for name, cutoff in limits.items():
        values = frame[name].to_numpy(dtype=float)
        pred &= np.isfinite(values) & (values <= cutoff)
    return pred


def fit_include_limits(frame: pd.DataFrame, max_rejects: int = MAX_REJECTS_IN_INCLUDE) -> dict[str, float]:
    """Loosen each cutoff while at most ``max_rejects`` human rejects slip in."""
    keep = frame["category"].to_numpy() == KEEP_CATEGORY
    reject = ~keep
    levels = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
    cutoffs = {}
    for name in WORSE_HIGH:
        keep_values = frame.loc[keep, name].to_numpy(dtype=float)
        keep_values = keep_values[np.isfinite(keep_values)]
        cutoffs[name] = float(np.quantile(keep_values, levels[0])) if keep_values.size else float("inf")
    current = include_mask_for_limits(frame, cutoffs)
    improved = True
    while improved:
        improved = False
        best_name = ""
        best_cutoff = 0.0
        best_keeps = int((current & keep).sum())
        best_mask = current
        for name in WORSE_HIGH:
            keep_values = frame.loc[keep, name].to_numpy(dtype=float)
            keep_values = keep_values[np.isfinite(keep_values)]
            if keep_values.size == 0:
                continue
            for level in levels:
                cutoff = float(np.quantile(keep_values, level))
                if cutoff <= cutoffs[name] + 1e-9:
                    continue
                trial = dict(cutoffs)
                trial[name] = cutoff
                mask = include_mask_for_limits(frame, trial)
                if int((mask & reject).sum()) > max_rejects:
                    continue
                keeps_in = int((mask & keep).sum())
                if keeps_in > best_keeps:
                    best_keeps = keeps_in
                    best_name = name
                    best_cutoff = cutoff
                    best_mask = mask
        if best_name:
            cutoffs[best_name] = best_cutoff
            current = best_mask
            improved = True
    return cutoffs


def assign_bins(frame: pd.DataFrame, reject_limits: dict[str, int], include_limits: dict[str, float]) -> np.ndarray:
    """Confident calls win. A sheet that meets both tests stays in review."""
    rejected = reject_mask_for_limits(frame, reject_limits)
    included = include_mask_for_limits(frame, include_limits)
    bins = np.full(len(frame), "review", dtype=object)
    bins[included & ~rejected] = "include"
    bins[rejected & ~included] = "reject"
    return bins


def bin_counts(frame: pd.DataFrame, bins: np.ndarray) -> dict[str, int]:
    keep = frame["category"].to_numpy() == KEEP_CATEGORY
    out = {}
    for name in ("reject", "include", "review"):
        chosen = bins == name
        out[f"{name}_n"] = int(chosen.sum())
        out[f"{name}_keeps"] = int((chosen & keep).sum())
        out[f"{name}_rejects"] = int((chosen & ~keep).sum())
    keeps = int(keep.sum())
    out["reject_specificity"] = (keeps - out["reject_keeps"]) / keeps if keeps else float("nan")
    return out


def collect_extra(features: pd.DataFrame) -> pd.DataFrame:
    """Add Z-window correlations and the first-pre beading score."""
    pooled = {session: pd.read_pickle(path) for session, path in SESSION_PKLS.items() if os.path.isfile(path)}
    rows = []
    for number, record in enumerate(features.itertuples(index=False), start=1):
        measured = {"z_match_corr": np.nan, "z_neighbor_corr": np.nan, "bead_fraction": np.nan, "bright_cv": np.nan}
        table = pooled.get(str(record.session))
        if table is not None:
            set_label = normalize_set_label(record.set_label)
            matched = table[
                (table["group"].astype(str) == str(record.group))
                & (table["nth_set_label"].map(normalize_set_label) == set_label)
            ]
            after = next((path for path in matched["after_align_full_save_path"] if isinstance(path, str) and path), "")
            info_path = os.path.splitext(after)[0] + "_frame_info.csv" if after else ""
            mask_path = os.path.splitext(after)[0] + "_Spine_roi_mask.tif" if after else ""
            if after and os.path.isfile(after) and os.path.isfile(info_path) and os.path.isfile(mask_path):
                import tifffile

                info = pd.read_csv(info_path)
                stack = np.asarray(tifffile.imread(after))
                mask = np.asarray(tifffile.imread(mask_path))
                measured = image_features_for_set(info, stack, mask)
        rows.append(measured)
        if number % 40 == 0 or number == len(features):
            print(f"extra {number}/{len(features)}", flush=True)
    extra = pd.DataFrame(rows)
    return add_worse_columns(pd.concat([features.reset_index(drop=True), extra], axis=1))


def format_bins(frame: pd.DataFrame, bins: np.ndarray) -> str:
    counts = bin_counts(frame, bins)
    lines = [
        f"confident reject {counts['reject_n']}  (keeps inside {counts['reject_keeps']}, rejects caught {counts['reject_rejects']})",
        f"reject-call specificity {counts['reject_specificity']:.3f}",
        f"confident include {counts['include_n']}  (keeps inside {counts['include_keeps']}, rejects inside {counts['include_rejects']})",
        f"review {counts['review_n']}",
        "category by bin",
    ]
    shown = frame.assign(bin=bins)
    lines.append(pd.crosstab(shown["category"], shown["bin"]).to_string())
    return "\n".join(lines)


def leave_one_session_out(frame: pd.DataFrame) -> str:
    """Fit the keep budget on the other dates and score the held-out date."""
    parts = []
    for session in sorted(frame["session"].astype(str).unique()):
        train = frame[frame["session"].astype(str) != session]
        test = frame[frame["session"].astype(str) == session]
        rejected = apply_reject_cutoffs(test, absolute_reject_cutoffs(train, fit_reject_limits(train)))
        included = include_mask_for_limits(test, fit_include_limits(train))
        bins = np.full(len(test), "review", dtype=object)
        bins[included & ~rejected] = "include"
        bins[rejected & ~included] = "reject"
        counts = bin_counts(test, bins)
        counts["session"] = session
        parts.append(counts)
    keeps_flagged = sum(item["reject_keeps"] for item in parts)
    keeps = sum(item["reject_keeps"] + (item["include_keeps"] + (item["review_n"] - item["review_rejects"])) for item in parts)
    # review keeps = review_n - review_rejects, include keeps, reject keeps.
    keep_total = sum(item["reject_keeps"] + item["include_keeps"] + (item["review_n"] - item["review_rejects"]) for item in parts)
    lines = [f"leave-one-session-out  keeps auto-rejected {keeps_flagged}/{keep_total}"]
    for item in parts:
        lines.append(
            f"  {item['session']}  reject {item['reject_n']} (keeps {item['reject_keeps']})  "
            f"include {item['include_n']} (rejects inside {item['include_rejects']})  "
            f"spec {item['reject_specificity']:.3f}"
        )
    return "\n".join(lines)


def _num(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "missing"
    return f"{float(value):.{digits}f}"


def export_bins(frame: pd.DataFrame, bins: np.ndarray, out_dir: str = OUT_DIR) -> None:
    titles = {
        "reject": "CONFIDENT REJECT",
        "include": "CONFIDENT INCLUDE",
        "review": "NEEDS REVIEW",
    }
    for name in titles:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    for record, bin_name in zip(frame.itertuples(index=False), bins):
        source = os.path.join(SHEET_ROOT, str(record.image_rel).replace("/", os.sep))
        sheet = cv2.imread(source, cv2.IMREAD_COLOR)
        if sheet is None:
            continue
        human = f"human {int(record.category)}  {CATEGORIES.get(int(record.category), '')}"
        rules = [
            (False, f"z match {_num(record.z_match_corr)}   z neighbor {_num(record.z_neighbor_corr)}"),
            (False, f"assign shift {_num(record.assign_shift_px, 1)} px   corr {_num(record.assign_corr)}"),
            (False, f"uncaging {_num(record.unc_dist_px, 1)} px   drift {_num(record.drift_info_px, 1)} px"),
        ]
        painted = annotate_sheet(sheet, titles[str(bin_name)], human, rules)
        destination = os.path.join(out_dir, str(bin_name), os.path.basename(str(record.image_rel)))
        cv2.imwrite(destination, painted)


def stage1_dead_call(frame: pd.DataFrame) -> str:
    """First-frame beading only. A keep is auto-skipped only past the keep maximum."""
    keep = frame["category"].to_numpy() == KEEP_CATEGORY
    dead = frame["category"].to_numpy() == 8
    called = np.zeros(len(frame), dtype=bool)
    for name in ("bead_fraction", "bright_cv"):
        values = frame[name].to_numpy(dtype=float)
        finite = values[keep & np.isfinite(values)]
        if finite.size == 0:
            continue
        called |= np.isfinite(values) & (values > float(np.max(finite)))
    keeps_hit = int((called & keep).sum())
    dead_hit = int((called & dead).sum())
    spec = (int(keep.sum()) - keeps_hit) / int(keep.sum())
    return (
        f"stage1 first-frame beading  dead caught {dead_hit}/{int(dead.sum())}  "
        f"keeps skipped {keeps_hit}  specificity {spec:.3f}"
    )


def main() -> None:
    if os.path.isfile(EXTRA_CSV):
        frame = pd.read_csv(EXTRA_CSV)
    else:
        frame = collect_extra(pd.read_csv(FEATURE_CSV))
        frame.to_csv(EXTRA_CSV, index=False)
    medians = frame.groupby("category")[["z_match_corr", "z_neighbor_corr", "bead_fraction", "bright_cv"]].median()
    print(medians.to_string(float_format=lambda value: f"{value:7.3f}"))
    reject_limits = fit_reject_limits(frame)
    include_limits = fit_include_limits(frame)
    bins = assign_bins(frame, reject_limits, include_limits)
    print(stage1_dead_call(frame))
    print(format_bins(frame, bins))
    keep = frame["category"].to_numpy() == KEEP_CATEGORY
    print("worse than every keep")
    for name in WORSE_HIGH:
        values = frame[name].to_numpy(dtype=float)
        finite = values[keep & np.isfinite(values)]
        if finite.size == 0:
            continue
        hit = np.isfinite(values) & (values > float(np.max(finite)))
        if not hit.any():
            continue
        cats = frame.loc[hit, "category"].value_counts().sort_index()
        print(f"  {name}  {int(hit.sum())}  " + " ".join(f"{int(cat)}:{int(n)}" for cat, n in cats.items()))
    print("reject keep-budget", {name: taken for name, taken in reject_limits.items() if taken})
    print(leave_one_session_out(frame))
    export_bins(frame, bins)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
