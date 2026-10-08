"""One-set review panel: spine crops with ROIs + normalized spine volume.

Left: crops of the GUI TIFF (after_align_full) around the spine for every pre and
post frame and the first / last uncaging frame, with the Spine (red) and
DendriticShaft (blue) Type-A ROI contours. Pre/post crops share one display range so
volume changes are visible; uncaging crops (single plane, different averaging) use
their own range.
Right: quantification from the FLIM files (all-frames CSV), intensity / nAveFrame,
normalized so that the mean of pre = 1 (Spine and DendriticShaft; pre/post only).

The TIFF, the masks and the CSV are the ones the ROI GUI and quantify_intensity_from_flim
use, so what is shown is what is measured.

Usage (one set to PNG):
    python set_review_panel.py --pkl combined_df_respan.pkl --csv <all_frames.csv>
        --group 4_pos1__highmag_1_ --set 0 --out review.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import tifffile

ROI_COLORS = {"Spine": "red", "DendriticShaft": "deepskyblue"}
CROP_HALF_PX = 24
PRE_POST_VMAX_PERCENTILE = 99.5


@dataclass
class ReviewFrame:
    label: str
    phase: str
    image: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    warning: str = ""  # e.g. the uncaging plane lies outside this frame's z-stack
    note: str = ""     # small text in the lower-left corner (alignment correlation)


@dataclass
class SetReview:
    title: str
    frames: list[ReviewFrame]
    quant: pd.DataFrame  # columns: t_min, phase, Spine_norm, DendriticShaft_norm
    crop_box: tuple[int, int, int, int]  # y0, y1, x0, x1 in the GUI TIFF frame
    notes: list[str] = field(default_factory=list)
    rejected: bool = False
    # Uncaging position (FLIM header) in GUI TIFF (after_align_full) pixels, drawn as a
    # yellow cross on the uncaging frames.
    uncaging_xy: tuple[float, float] | None = None


def _crop_box(center_yx: tuple[float, float], shape: tuple[int, int], half: int) -> tuple[int, int, int, int]:
    h, w = shape
    cy, cx = int(round(center_yx[0])), int(round(center_yx[1]))
    y0 = int(np.clip(cy - half, 0, max(h - 2 * half, 0)))
    x0 = int(np.clip(cx - half, 0, max(w - 2 * half, 0)))
    return y0, min(h, y0 + 2 * half), x0, min(w, x0 + 2 * half)


def normalized_quant(set_quant: pd.DataFrame, ch: int = 2) -> pd.DataFrame:
    """Pre/post rows: intensity / nAveFrame, divided by the mean of pre (pre mean = 1)."""
    q = set_quant.copy()
    n_ave = pd.to_numeric(q.get("nAveFrame", 1), errors="coerce").replace(0, np.nan).fillna(1)
    unc = q[q.phase == "unc"]
    t0 = float(unc.elapsed_time_sec.min()) if len(unc) else 0.0
    q["t_min"] = (pd.to_numeric(q.elapsed_time_sec, errors="coerce") - t0) / 60.0
    out = q[q.phase.isin(["pre", "post"])][["t_min", "phase", "file_path"]].copy()
    for roi in ROI_COLORS:
        col = f"{roi}_Ch{ch}_intensity"
        if col not in q.columns:
            continue
        v = q[col] / n_ave
        pre_mean = v[q.phase == "pre"].mean()
        out[f"{roi}_norm"] = (v / pre_mean).loc[out.index] if pre_mean and np.isfinite(pre_mean) else np.nan
    return out.sort_values("t_min")


def load_set_review(tiff_path: str, set_quant: pd.DataFrame, *, title: str = "", ch: int = 2,
                    corr: pd.DataFrame | None = None,
                    crop_half_px: int = CROP_HALF_PX) -> SetReview:
    """Collect crops, ROI contours and normalized quantification for one set."""
    base = os.path.splitext(tiff_path)[0]
    stack = tifffile.imread(tiff_path).astype(np.float32)
    fi = pd.read_csv(base + "_frame_info.csv")
    masks = {}
    for roi in ROI_COLORS:
        p = f"{base}_{roi}_roi_mask.tif"
        if os.path.exists(p):
            m = tifffile.imread(p) > 0
            masks[roi] = m if m.ndim == 3 else np.repeat(m[None], stack.shape[0], 0)
    notes = [f"no {r} ROI" for r in ROI_COLORS if r not in masks]
    phases = fi.phase.astype(str).str.lower().tolist()
    pre = [i for i, p in enumerate(phases) if p == "pre"]
    post = [i for i, p in enumerate(phases) if p == "post"]
    unc = [i for i, p in enumerate(phases) if p.startswith("unc")]
    picks = [(i, f"Pre {k + 1}") for k, i in enumerate(pre)]
    if unc:
        picks += [(unc[0], "Unc first"), (unc[-1], "Unc last")]
    q = normalized_quant(set_quant, ch)
    t_by_file = {os.path.basename(str(f)).lower(): t for f, t in zip(q.file_path, q.t_min)}
    picks += [(i, f"Post {k + 1}") for k, i in enumerate(post)]

    ref = pre[-1] if pre else 0
    if "Spine" in masks and masks["Spine"][ref].any():
        ys, xs = np.nonzero(masks["Spine"][ref])
        center = (ys.mean(), xs.mean())
    else:
        center = (stack.shape[1] / 2, stack.shape[2] / 2)
        notes.append("spine ROI empty: crop at image centre")
    y0, y1, x0, x1 = _crop_box(center, stack.shape[1:], crop_half_px)
    def z_warning(i: int) -> str:
        if not {"z_center", "n_z"}.issubset(fi.columns):
            return ""
        c, nz = fi.z_center.iloc[i], fi.n_z.iloc[i]
        if pd.isna(c) or pd.isna(nz):
            return ""
        if c < 0 or c > nz - 1:
            return f"z {int(c)} outside stack 0-{int(nz) - 1}"
        return ""

    corr_by = {} if corr is None else {str(r.filename).lower(): r for r in corr.itertuples()}

    def corr_note(name: str) -> str:
        r = corr_by.get(name)
        if r is None:
            return ""
        def fmt(v):
            return "-" if pd.isna(v) else f"{float(v):.3f}"
        return f"H {fmt(r.r_high)}  L {fmt(r.r_low)}"

    frames = []
    for i, label in picks:
        name = str(fi.filename.iloc[i]).lower()
        if phases[i] in ("pre", "post") and name in t_by_file and np.isfinite(t_by_file[name]):
            t = int(round(float(t_by_file[name])))
            label = f"{label}\n{t:+d} min" if t else f"{label}\n0 min"
        frames.append(ReviewFrame(label=label, phase=phases[i], image=stack[i, y0:y1, x0:x1],
                                  masks={r: m[i, y0:y1, x0:x1] for r, m in masks.items()},
                                  warning=z_warning(i) if phases[i] in ("pre", "post") else "",
                                  note=corr_note(name)))
    n_out = sum(1 for f in frames if f.warning)
    if n_out:
        notes.append(f"{n_out} frame(s): uncaging plane outside the z-stack")
    return SetReview(title=title or os.path.basename(base), frames=frames, quant=q,
                     crop_box=(y0, y1, x0, x1), notes=notes)


def draw_set_review(fig, review: SetReview, *, n_cols: int = 6) -> list:
    """Draw the panel into a matplotlib figure (cleared first).

    Returns one (axes, ReviewFrame) pair per crop (for click handling).
    """
    crop_axes = []
    fig.clf()
    n = len(review.frames)
    n_rows = int(np.ceil(n / n_cols))
    gs = fig.add_gridspec(n_rows, n_cols + 4, wspace=0.08, hspace=0.25)
    pp = [f.image for f in review.frames if f.phase in ("pre", "post")]
    vmax_pp = float(np.percentile(np.stack(pp), PRE_POST_VMAX_PERCENTILE)) if pp else None
    for k, f in enumerate(review.frames):
        ax = fig.add_subplot(gs[k // n_cols, k % n_cols])
        vmax = vmax_pp if f.phase in ("pre", "post") else float(np.percentile(f.image, PRE_POST_VMAX_PERCENTILE))
        ax.imshow(f.image, cmap="gray", vmin=0, vmax=max(vmax or 1.0, 1e-6), interpolation="nearest")
        for roi, m in f.masks.items():
            if m.any():
                ax.contour(m.astype(float), [0.5], colors=ROI_COLORS[roi], linewidths=0.9)
        color = "red" if f.warning else ("darkorange" if f.phase.startswith("unc") else "black")
        ax.set_title(f.label + (f"\n{f.warning}" if f.warning else ""), fontsize=8, color=color)
        if f.warning:
            for sp in ax.spines.values():
                sp.set_edgecolor("red")
                sp.set_linewidth(2)
        if f.phase.startswith("unc") and review.uncaging_xy is not None:
            ux = review.uncaging_xy[0] - review.crop_box[2]
            uy = review.uncaging_xy[1] - review.crop_box[0]
            ax.plot(ux, uy, "+", color="yellow", ms=12, mew=2)
        ax.set_xlim(-0.5, f.image.shape[1] - 0.5)
        ax.set_ylim(f.image.shape[0] - 0.5, -0.5)
        if f.note:
            # just below the image, outside the axes
            ax.text(0.0, -0.03, f.note, transform=ax.transAxes, ha="left", va="top", fontsize=6,
                    color="black", family="monospace", clip_on=False)
        ax.set_xticks([])
        ax.set_yticks([])
        crop_axes.append((ax, f))
    axq = fig.add_subplot(gs[:, n_cols + 1:])
    q = review.quant
    for roi, color in ROI_COLORS.items():
        col = f"{roi}_norm"
        if col in q.columns and q[col].notna().any():
            axq.plot(q.t_min, q[col], "o-", color=color, ms=4, lw=1.2,
                     label="Spine" if roi == "Spine" else "Shaft")
    axq.axhline(1.0, color="gray", ls="--", lw=0.8)
    axq.axvline(0.0, color="darkorange", lw=0.8, alpha=0.7)
    axq.set_xlabel("Time from uncaging (min)")
    axq.set_ylabel("Intensity / nAve, pre mean = 1")
    if axq.get_legend_handles_labels()[0]:
        axq.legend(frameon=False, fontsize=8)
    else:
        axq.text(0.5, 0.5, "no quantification", ha="center", va="center", transform=axq.transAxes, color="gray")
    axq.spines[["top", "right"]].set_visible(False)
    if review.rejected:
        axq.text(0.5, 0.92, "REJECTED", ha="center", va="top", transform=axq.transAxes, color="red",
                 fontsize=22, fontweight="bold", alpha=0.6)
        for sp in axq.spines.values():
            sp.set_edgecolor("red")
    notes = (["REJECTED"] if review.rejected else []) + review.notes
    title = review.title + (f"   [{'; '.join(notes)}]" if notes else "")
    fig.suptitle(title, fontsize=10, color="red" if notes else "black")
    return crop_axes


def set_review_from_paths(pkl: str, csv: str, group: str, set_label: float, ch: int = 2) -> SetReview:
    df = pd.read_pickle(pkl)
    sdf = df[(df.group.astype(str) == str(group)) & (df.nth_set_label.astype(float) == float(set_label))]
    if not len(sdf):
        raise KeyError(f"set {group} {set_label} not in {pkl}")
    col = "after_align_full_save_path" if "after_align_full_save_path" in sdf.columns else "after_align_save_path"
    tiff = str(sdf[col].dropna().iloc[0])
    q = pd.read_csv(csv)
    q = q[(q.group.astype(str) == str(group)) & (q.set_label.astype(float) == float(set_label))]
    return load_set_review(tiff, q, title=f"{group} set {set_label:g}", ch=ch)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Review panel for one set (PNG).")
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--set", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    review = set_review_from_paths(a.pkl, a.csv, a.group, a.set)
    fig = plt.figure(figsize=(16, 2.6 * int(np.ceil(len(review.frames) / 6)) + 0.6))
    draw_set_review(fig, review)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
