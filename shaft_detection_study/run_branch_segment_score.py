# -*- coding: utf-8 -*-
"""
Score 15 um sliding-window segments (5 um overlap) per qualifying branch.

Filters: width 0.6–1.0 um, length >= 20 um. max ~30 branches (spaced in 3D).

Example:
  python controlFLIMage/shaft_detection_study/run_branch_segment_score.py ^
    --flim "\\\\server\\path\\AP5_pos6_256_4x_001.flim"
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

def _is_interactive_env() -> bool:
    """Return True when running inside IPython / VS Code Interactive Window / Jupyter."""
    try:
        from IPython import get_ipython  # type: ignore
        shell = get_ipython()
        return shell is not None
    except ImportError:
        return False

if not _is_interactive_env():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_STUDY_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _STUDY_DIR.parent
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ.setdefault("PYTHONUTF8", "1")

from deepd3_spine_head_detector import resolve_savefolder
from shaft_detection_study.branch_zoom_presets import (
    format_preset_summary,
    resolve_scoring_params,
)
from shaft_detection_study.dendrite_branch_score_core import score_all_branch_segments

DEFAULT_FLIM = (
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260515\lowmags\AP5_pos6_256_4x_001.flim"
)
DEFAULT_OUT_SUBDIR = "branch_segment_score"

SEGMENT_CSV_FIELDS = [
    "branch_id",
    "segment_index",
    "arc_start_um",
    "arc_end_um",
    "path_length_um",
    "branch_mean_width_um",
    "z_pix",
    "y_pix",
    "x_pix",
    "segment_score",
    "spine_signal",
    "smoothness",
    "width_cv",
    "width_range_ratio",
    "mean_width_um",
    "min_width_um",
    "max_width_um",
    "protrusion_density",
    "puncta_count",
    "roughness",
    "skeleton_length_pix",
]


def save_segment_csv(rows: list[dict], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=SEGMENT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SEGMENT_CSV_FIELDS})


def _draw_rank_markers(ax, rows_sorted: list[dict], radius_px: float) -> None:
    """Numbered circle markers on MIP. Labels stay inside marker so no collisions."""
    H, W = ax.get_images()[0].get_array().shape[:2]
    cmap = plt.cm.viridis
    for rank, row in enumerate(rows_sorted, start=1):
        t = 1.0 - (rank - 1) / max(len(rows_sorted) - 1, 1)
        face = cmap(0.15 + 0.7 * t)
        ax.scatter(
            [row["x_pix"]],
            [row["y_pix"]],
            s=radius_px ** 2,
            facecolor=face,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.text(
            row["x_pix"],
            row["y_pix"],
            str(rank),
            color="black" if t > 0.55 else "white",
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
        )


def save_overview_png(result: dict, savepath: str | None = None):
    zyx = result["zyx"]
    rows = result["segment_rows"]
    summaries = result["branch_summaries"]
    mip = zyx.max(axis=0)
    vmax = float(np.percentile(mip, 99.5)) if mip.max() > 0 else 1.0

    if not rows:
        fig, ax = plt.subplots(facecolor="black")
        ax.imshow(mip, cmap="gray", vmax=vmax)
        ax.set_title("No qualifying segments", color="white")
        ax.axis("off")
        if savepath:
            fig.savefig(savepath, dpi=160, bbox_inches="tight", facecolor="black")
        if not _is_interactive_env():
            plt.close(fig)
        return fig

    # ── Branch colour map (each branch gets a distinct colour) ──────────────
    branch_ids_ordered = [s["branch_id"] for s in summaries]  # sorted by mean score desc
    n_br = len(branch_ids_ordered)
    br_cmap = plt.cm.tab20
    br_color = {bid: br_cmap(i / max(n_br - 1, 1)) for i, bid in enumerate(branch_ids_ordered)}

    all_scores = np.array([r["segment_score"] for r in rows])
    smin, smax = float(all_scores.min()), float(all_scores.max())

    # ── Figure: [MIP | score heatmap per branch | component bar chart] ──────
    n_chart_rows = len(rows)
    fig_h = max(6.0, 0.38 * n_chart_rows + 2.0)
    fig = plt.figure(figsize=(16, fig_h), facecolor="black")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.1, 1.1], wspace=0.42)
    ax_mip = fig.add_subplot(gs[0, 0])
    ax_score = fig.add_subplot(gs[0, 1])
    ax_comp = fig.add_subplot(gs[0, 2])

    for ax in (ax_mip, ax_score, ax_comp):
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("0.4")

    # ── Left: MIP + markers ──────────────────────────────────────────────────
    ax_mip.imshow(mip, cmap="gray", vmax=vmax)
    for row in rows:
        bid = row["branch_id"]
        fc = br_color[bid]
        sc = row["segment_score"]
        t = (sc - smin) / (smax - smin + 1e-9)
        ax_mip.scatter(
            [row["x_pix"]], [row["y_pix"]],
            s=200,
            facecolor=(*fc[:3], 0.85),
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax_mip.text(
            row["x_pix"], row["y_pix"],
            f"b{bid}\ns{row['segment_index']}",
            color="white", fontsize=5.5, fontweight="bold",
            ha="center", va="center", zorder=4, linespacing=1.1,
        )
    ax_mip.set_title(
        f"15 µm sliding segments (5 µm overlap), n={len(rows)}",
        color="white", fontsize=10,
    )
    ax_mip.axis("off")

    # ── Middle: horizontal score bars, grouped by branch ────────────────────
    # rows sorted branch-first (mean score desc), then arc position asc
    rows_grouped = sorted(
        rows,
        key=lambda r: (
            -next(s["mean_segment_score"] for s in summaries if s["branch_id"] == r["branch_id"]),
            r["arc_start_um"],
        ),
    )
    y_labels = []
    y_pos = []
    cur_y = 0.0
    prev_bid = None
    gap = 0.5
    bar_h = 0.72
    for row in rows_grouped:
        bid = row["branch_id"]
        if prev_bid is not None and bid != prev_bid:
            cur_y -= gap
        y_labels.append(f"b{bid} s{row['segment_index']}\n{row['arc_start_um']:.0f}–{row['arc_end_um']:.0f}µm")
        y_pos.append(cur_y)
        cur_y -= 1.0
        prev_bid = bid

    scores_plot = [r["segment_score"] for r in rows_grouped]
    for yi, sc, row in zip(y_pos, scores_plot, rows_grouped):
        fc = br_color[row["branch_id"]]
        t = (sc - smin) / (smax - smin + 1e-9)
        ax_score.barh(yi, sc, height=bar_h,
                      color=(*fc[:3], 0.85), edgecolor="white", linewidth=0.35)
        ax_score.text(sc * 1.02, yi, f"{sc:.0f}",
                      color="white", fontsize=7, va="center", ha="left")

    ax_score.set_yticks(y_pos, y_labels, fontsize=7)
    ax_score.set_xlabel("segment_score (smooth × spine)", color="white", fontsize=8)
    ax_score.set_title("Score by segment", color="white", fontsize=10)
    ax_score.set_xlim(0, max(scores_plot) * 1.40)
    ax_score.grid(axis="x", color="0.22", linewidth=0.4)

    # ── Right: components (smoothness + spine_signal normalised) ────────────
    smoothness_vals = [r["smoothness"] for r in rows_grouped]
    spine_vals = [r["spine_signal"] for r in rows_grouped]
    spine_norm_vals = np.array(spine_vals) / max(max(spine_vals), 1e-9)
    puncta_vals = [int(r["puncta_count"]) for r in rows_grouped]
    width_vals = [r["mean_width_um"] for r in rows_grouped]
    bh2 = bar_h * 0.46

    for yi, sm, sp_n, pn, w, row in zip(
        y_pos, smoothness_vals, spine_norm_vals, puncta_vals, width_vals, rows_grouped
    ):
        fc = br_color[row["branch_id"]]
        ax_comp.barh(yi + bh2, sm, height=bh2 * 1.8,
                     color="#5dd39e", edgecolor="none", alpha=0.9)
        ax_comp.barh(yi - bh2, sp_n, height=bh2 * 1.8,
                     color="#f4a261", edgecolor="none", alpha=0.9)
        ax_comp.text(sm + 0.02, yi + bh2, f"{sm:.2f}",
                     color="#5dd39e", fontsize=6, va="center")
        ax_comp.text(sp_n + 0.02, yi - bh2,
                     f"pun={pn} w={w:.2f}µm",
                     color="#f4a261", fontsize=6, va="center")

    ax_comp.set_yticks(y_pos, y_labels, fontsize=7)
    ax_comp.set_xlim(0, 1.55)
    ax_comp.set_xlabel("relative magnitude", color="white", fontsize=8)
    ax_comp.set_title("Components", color="white", fontsize=10)
    from matplotlib.patches import Patch
    ax_comp.legend(
        handles=[
            Patch(facecolor="#5dd39e", label="smoothness (0–1)"),
            Patch(facecolor="#f4a261", label="spine_signal (norm.)"),
        ],
        facecolor="0.12", edgecolor="0.4", labelcolor="white",
        fontsize=7, loc="lower right",
    )
    ax_comp.grid(axis="x", color="0.22", linewidth=0.4)

    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight", facecolor="black")
    if not _is_interactive_env():
        plt.close(fig)
    return fig


def _segment_center_um(row: dict, x_um: float, y_um: float, z_um: float) -> tuple[float, float, float]:
    return (
        float(row["x_pix"]) * x_um,
        float(row["y_pix"]) * y_um,
        float(row["z_pix"]) * z_um,
    )


def segments_roi_overlap(
    row_a: dict,
    row_b: dict,
    x_um: float,
    y_um: float,
    z_um: float,
    *,
    min_xy_sep_um: float = 20.0,
    min_z_sep_um: float = 10.0,
) -> bool:
    """
  True if two segment ROIs are considered overlapping.

    Non-overlap (independent) when XY center separation >= min_xy_sep_um OR
    |dz| >= min_z_sep_um. Overlap is the negation: close in XY AND close in Z.
    """
    ax, ay, az = _segment_center_um(row_a, x_um, y_um, z_um)
    bx, by, bz = _segment_center_um(row_b, x_um, y_um, z_um)
    dxy = float(np.hypot(ax - bx, ay - by))
    dz = abs(az - bz)
    return dxy < min_xy_sep_um and dz < min_z_sep_um


def select_top_segments_non_overlapping(
    rows: list[dict],
    top_n: int,
    x_um: float,
    y_um: float,
    z_um: float,
    *,
    min_xy_sep_um: float = 20.0,
    min_z_sep_um: float = 10.0,
) -> list[dict]:
    """Greedy top-by-score picks, skipping candidates that overlap any already chosen."""
    ordered = sorted(rows, key=lambda r: -r["segment_score"])
    selected: list[dict] = []
    for row in ordered:
        if len(selected) >= top_n:
            break
        if any(
            segments_roi_overlap(
                row, prev, x_um, y_um, z_um,
                min_xy_sep_um=min_xy_sep_um,
                min_z_sep_um=min_z_sep_um,
            )
            for prev in selected
        ):
            continue
        selected.append(row)
    return selected


def save_top_roi_tiles(
    result: dict,
    savepath: str | None = None,
    top_n: int = 8,
    roi_um: float = 15.0,
    min_roi_xy_sep_um: float = 20.0,
    min_roi_z_sep_um: float = 10.0,
):
    """
    For up to top_n segments (by score, non-overlapping), output:
      - Top row: full Z-MIP with numbered ROI boxes
      - Bottom rows: Z±2 local MIP cropped to roi_um × roi_um per segment

    Overlap rule: exclude if XY center distance < min_roi_xy_sep_um AND
    |Z| separation < min_roi_z_sep_um vs any already selected segment.
    """
    from matplotlib.patches import Rectangle

    zyx = result["zyx"]
    rows = result["segment_rows"]
    x_um: float = result["x_um"]
    y_um: float = result.get("y_um", x_um)
    z_um: float = result["z_um"]
    if not rows:
        return None

    top = select_top_segments_non_overlapping(
        rows,
        top_n,
        x_um,
        y_um,
        z_um,
        min_xy_sep_um=min_roi_xy_sep_um,
        min_z_sep_um=min_roi_z_sep_um,
    )
    n = len(top)
    Z, Y, X = zyx.shape

    # pixel half-size for 15 µm ROI (xy resolution)
    half_px = int(round(roi_um / 2.0 / x_um))
    # z half-slices (±2 slices = ±2*z_um µm)
    z_half = 2

    # Full Z-MIP for overview panel
    full_mip = zyx.max(axis=0).astype(np.float32)
    vmax_full = float(np.percentile(full_mip, 99.5)) if full_mip.max() > 0 else 1.0

    n_cols = min(n, 4)
    n_tile_rows = (n + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 3.5, 3.5 + n_tile_rows * 3.6), facecolor="black")
    # gridspec: 1 overview row + n_tile_rows of tiles
    gs = fig.add_gridspec(
        1 + n_tile_rows, n_cols,
        height_ratios=[2.0] + [1.0] * n_tile_rows,
        hspace=0.25, wspace=0.10,
    )

    # ── Overview row (spans all columns) ────────────────────────────────────
    ax_ov = fig.add_subplot(gs[0, :])
    ax_ov.set_facecolor("black")
    ax_ov.imshow(full_mip, cmap="gray", vmax=vmax_full, origin="upper")
    ax_ov.axis("off")
    ax_ov.set_title(
        f"Top {n} segments (score-ranked, non-overlap: XY≥{min_roi_xy_sep_um:.0f} or Z≥{min_roi_z_sep_um:.0f} µm)",
        color="white", fontsize=10, pad=4,
    )

    scores = np.array([r["segment_score"] for r in top])
    smin, smax = scores.min(), scores.max()
    rank_cmap = plt.cm.plasma

    for rank, row in enumerate(top, start=1):
        t = 1.0 - (rank - 1) / max(n - 1, 1)
        col = rank_cmap(0.1 + 0.8 * t)
        cx, cy = row["x_pix"], row["y_pix"]
        x0, y0 = cx - half_px, cy - half_px
        rect = Rectangle(
            (x0 - 0.5, y0 - 0.5), half_px * 2, half_px * 2,
            linewidth=1.2, edgecolor=col, facecolor="none",
        )
        ax_ov.add_patch(rect)
        ax_ov.text(
            x0 + half_px, y0 - 4, str(rank),
            color=col, fontsize=8, fontweight="bold",
            ha="center", va="bottom",
        )

    # ── Tile rows ────────────────────────────────────────────────────────────
    for rank, row in enumerate(top, start=1):
        tile_row = (rank - 1) // n_cols
        tile_col = (rank - 1) % n_cols
        ax_t = fig.add_subplot(gs[1 + tile_row, tile_col])
        ax_t.set_facecolor("black")
        ax_t.axis("off")

        cz = int(row["z_pix"])
        cy = int(row["y_pix"])
        cx = int(row["x_pix"])

        # Z±2 local MIP
        z0 = max(0, cz - z_half)
        z1 = min(Z, cz + z_half + 1)
        local_mip = zyx[z0:z1, :, :].max(axis=0).astype(np.float32)
        vmax_t = float(np.percentile(local_mip, 99.5)) if local_mip.max() > 0 else 1.0

        # XY crop
        x0 = max(0, cx - half_px)
        x1 = min(X, cx + half_px)
        y0 = max(0, cy - half_px)
        y1 = min(Y, cy + half_px)
        crop = local_mip[y0:y1, x0:x1]

        ax_t.imshow(crop, cmap="gray", vmax=vmax_t, origin="upper")

        t = 1.0 - (rank - 1) / max(n - 1, 1)
        col = rank_cmap(0.1 + 0.8 * t)
        # ROI border
        for spine in ax_t.spines.values():
            spine.set_visible(True)
            spine.set_color(col)
            spine.set_linewidth(2.0)

        # Info text
        z_range_um = z_half * z_um
        title = (
            f"#{rank}  b{row['branch_id']} seg{row['segment_index']}\n"
            f"score={row['segment_score']:.0f}  w={row['mean_width_um']:.2f}µm\n"
            f"Z={cz} (±{z_range_um:.1f}µm)  pun={int(row['puncta_count'])}"
        )
        ax_t.set_title(title, color=col, fontsize=7.5, pad=3, linespacing=1.2)

    if savepath:
        fig.savefig(savepath, dpi=180, bbox_inches="tight", facecolor="black")
    if not _is_interactive_env():
        plt.close(fig)
    return fig


def _merge_zoom_scoring_kwargs(
    flim_path: str,
    *,
    use_zoom_presets: bool,
    min_branch_width_um: float | None,
    max_branch_width_um: float | None,
    min_branch_length_um: float | None,
    segment_length_um: float | None,
    overlap_um: float | None,
    dendrite_percentile: float | None,
) -> tuple[dict[str, float], dict]:
    """Apply zoom presets from .flim; explicit args override preset values."""
    overrides: dict[str, float] = {}
    if min_branch_width_um is not None:
        overrides["min_branch_width_um"] = min_branch_width_um
    if max_branch_width_um is not None:
        overrides["max_branch_width_um"] = max_branch_width_um
    if min_branch_length_um is not None:
        overrides["min_branch_length_um"] = min_branch_length_um
    if segment_length_um is not None:
        overrides["segment_length_um"] = segment_length_um
    if overlap_um is not None:
        overrides["overlap_um"] = overlap_um
    if dendrite_percentile is not None:
        overrides["dendrite_percentile"] = dendrite_percentile

    if use_zoom_presets:
        resolved = resolve_scoring_params(
            flim_path, use_zoom_presets=True, overrides=overrides or None
        )
    else:
        resolved = resolve_scoring_params(flim_path, use_zoom_presets=False, overrides=overrides)
        defaults = {
            "min_branch_width_um": 0.6,
            "max_branch_width_um": 1.0,
            "min_branch_length_um": 20.0,
            "segment_length_um": 15.0,
            "overlap_um": 5.0,
            "dendrite_percentile": 94.0,
        }
        for k, v in defaults.items():
            resolved.setdefault(k, v)

    scoring = {
        "min_branch_width_um": float(resolved["min_branch_width_um"]),
        "max_branch_width_um": float(resolved["max_branch_width_um"]),
        "min_branch_length_um": float(resolved["min_branch_length_um"]),
        "segment_length_um": float(resolved["segment_length_um"]),
        "overlap_um": float(resolved["overlap_um"]),
        "dendrite_percentile": float(resolved["dendrite_percentile"]),
    }
    return scoring, resolved


def run_one(
    flim_path: str,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
    min_branch_width_um: float | None = None,
    max_branch_width_um: float | None = None,
    min_branch_length_um: float | None = None,
    segment_length_um: float | None = None,
    overlap_um: float | None = None,
    max_branches: int = 30,
    min_spacing_um: float = 20.0,
    dendrite_percentile: float | None = None,
    use_zoom_presets: bool = True,
    top_n: int = 8,
    roi_um: float = 15.0,
    min_roi_xy_sep_um: float = 20.0,
    min_roi_z_sep_um: float = 10.0,
    profile: bool = False,
) -> tuple[str, dict]:
    scoring, preset_meta = _merge_zoom_scoring_kwargs(
        flim_path,
        use_zoom_presets=use_zoom_presets,
        min_branch_width_um=min_branch_width_um,
        max_branch_width_um=max_branch_width_um,
        min_branch_length_um=min_branch_length_um,
        segment_length_um=segment_length_um,
        overlap_um=overlap_um,
        dendrite_percentile=dendrite_percentile,
    )
    min_branch_width_um = scoring["min_branch_width_um"]
    max_branch_width_um = scoring["max_branch_width_um"]
    min_branch_length_um = scoring["min_branch_length_um"]
    segment_length_um = scoring["segment_length_um"]
    overlap_um = scoring["overlap_um"]
    dendrite_percentile = scoring["dendrite_percentile"]

    out_dir = os.path.join(resolve_savefolder(flim_path), out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {flim_path} ===", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print(f"  {format_preset_summary(preset_meta)}", flush=True)
    print(
        f"  branch filter: {min_branch_width_um} < mean width <= {max_branch_width_um} um, "
        f"path >= {min_branch_length_um} um"
    )
    stride = segment_length_um - overlap_um
    print(f"  segment: {segment_length_um} um, overlap: {overlap_um} um, stride: {stride:.1f} um")
    print(f"  max branches: {max_branches}  min spacing: {min_spacing_um} um")
    print(f"  dendrite_percentile: {dendrite_percentile}")
    print(
        f"  ROI tiles: top {top_n} segments, {roi_um} um square, "
        f"non-overlap XY<{min_roi_xy_sep_um} & Z<{min_roi_z_sep_um} um"
    )

    result = score_all_branch_segments(
        flim_path,
        min_branch_width_um=min_branch_width_um,
        max_branch_width_um=max_branch_width_um,
        min_branch_length_um=min_branch_length_um,
        segment_length_um=segment_length_um,
        overlap_um=overlap_um,
        max_branches=max_branches,
        min_spacing_um=min_spacing_um,
        dendrite_percentile=dendrite_percentile,
        profile=profile,
    )

    if profile and result.get("timings_sec"):
        print("  --- timing breakdown (cumulative sec) ---", flush=True)
        prev = 0.0
        for name, t in result["timings_sec"].items():
            delta = t - prev
            print(f"    {name}: {t:.2f}s  (+{delta:.2f}s)", flush=True)
            prev = t

    rows = result["segment_rows"]
    summaries = result["branch_summaries"]
    print(f"  skeleton branches (after {min_branch_length_um} um prune): {result['n_branches']}")
    print(
        f"  qualifying: {result['n_qualifying']}  analyzed: {result['n_analyzed']}  "
        f"segments scored: {len(rows)}"
    )

    csv_path = os.path.join(out_dir, "segment_scores.csv")
    save_segment_csv(rows, csv_path)
    png_path = os.path.join(out_dir, "segment_score_overview.png")
    save_overview_png(result, png_path)

    roi_png_path = os.path.join(out_dir, f"top{top_n}_roi_tiles.png")
    save_top_roi_tiles(
        result,
        roi_png_path,
        top_n=top_n,
        roi_um=roi_um,
        min_roi_xy_sep_um=min_roi_xy_sep_um,
        min_roi_z_sep_um=min_roi_z_sep_um,
    )

    meta = os.path.join(out_dir, "run_meta.txt")
    with open(meta, "w", encoding="utf-8") as f:
        f.write(f"flim={flim_path}\n")
        f.write(f"timestamp={datetime.now().isoformat()}\n")
        f.write(f"n_segments={len(rows)}\n")
        f.write(f"n_branches={len(summaries)}\n")
        f.write(f"top_n={top_n}\n")
        f.write(f"roi_um={roi_um}\n")
        f.write(f"{format_preset_summary(preset_meta)}\n")
        for k in (
            "min_branch_width_um",
            "max_branch_width_um",
            "dendrite_percentile",
            "min_branch_length_um",
            "segment_length_um",
            "overlap_um",
        ):
            f.write(f"{k}={scoring[k]}\n")

    print(f"  CSV: {csv_path}")
    print(f"  PNG: {png_path}")
    print(f"  ROI: {roi_png_path}")
    if rows:
        best = max(rows, key=lambda r: r["segment_score"])
        print(
            f"  best b{best['branch_id']} seg{best['segment_index']} "
            f"({best['arc_start_um']:.0f}-{best['arc_end_um']:.0f} um): "
            f"score={best['segment_score']:.3f} smooth={best['smoothness']:.3f} "
            f"spine={best['spine_signal']:.3f}"
        )
    return out_dir, result


def run_interactive(
    flim_path: str = DEFAULT_FLIM,
    min_branch_width_um: float | None = None,
    max_branch_width_um: float | None = None,
    min_branch_length_um: float | None = None,
    segment_length_um: float | None = None,
    overlap_um: float | None = None,
    max_branches: int = 30,
    min_spacing_um: float = 20.0,
    top_n: int = 8,
    roi_um: float = 15.0,
    min_roi_xy_sep_um: float = 20.0,
    min_roi_z_sep_um: float = 10.0,
    save: bool = True,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
    dendrite_percentile: float | None = None,
    use_zoom_presets: bool = True,
):
    """
    Interactive-window entry point.
    Runs scoring, shows both figures inline, and optionally saves to disk.

    Parameters
    ----------
    top_n : int
        Number of highest-scoring segments to show in ROI tile figure (default 8).
    roi_um : float
        Side length of square ROI crop in µm (default 15).
    min_roi_xy_sep_um, min_roi_z_sep_um : float
        Two ROIs overlap when XY center distance < min_roi_xy_sep_um AND
        |Z| separation < min_roi_z_sep_um (default 20 and 10 µm).

    Returns
    -------
    result : dict
        Full scoring result (segment_rows, branch_summaries, zyx, …).
    fig_overview : matplotlib.figure.Figure
    fig_roi : matplotlib.figure.Figure
    """
    if save:
        out_dir, result = run_one(
            flim_path,
            out_subdir=out_subdir,
            min_branch_width_um=min_branch_width_um,
            max_branch_width_um=max_branch_width_um,
            min_branch_length_um=min_branch_length_um,
            segment_length_um=segment_length_um,
            overlap_um=overlap_um,
            max_branches=max_branches,
            min_spacing_um=min_spacing_um,
            dendrite_percentile=dendrite_percentile,
            use_zoom_presets=use_zoom_presets,
            top_n=top_n,
            roi_um=roi_um,
            min_roi_xy_sep_um=min_roi_xy_sep_um,
            min_roi_z_sep_um=min_roi_z_sep_um,
        )
        fig_overview = save_overview_png(result, savepath=None)
        fig_roi = save_top_roi_tiles(
            result,
            savepath=None,
            top_n=top_n,
            roi_um=roi_um,
            min_roi_xy_sep_um=min_roi_xy_sep_um,
            min_roi_z_sep_um=min_roi_z_sep_um,
        )
    else:
        scoring, preset_meta = _merge_zoom_scoring_kwargs(
            flim_path,
            use_zoom_presets=use_zoom_presets,
            min_branch_width_um=min_branch_width_um,
            max_branch_width_um=max_branch_width_um,
            min_branch_length_um=min_branch_length_um,
            segment_length_um=segment_length_um,
            overlap_um=overlap_um,
            dendrite_percentile=dendrite_percentile,
        )
        print(f"  {format_preset_summary(preset_meta)}", flush=True)
        result = score_all_branch_segments(
            flim_path,
            max_branches=max_branches,
            min_spacing_um=min_spacing_um,
            **scoring,
        )
        fig_overview = save_overview_png(result, savepath=None)
        fig_roi = save_top_roi_tiles(
            result,
            savepath=None,
            top_n=top_n,
            roi_um=roi_um,
            min_roi_xy_sep_um=min_roi_xy_sep_um,
            min_roi_z_sep_um=min_roi_z_sep_um,
        )

    plt.show()
    return result, fig_overview, fig_roi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="15 um segment scoring on 0.6-1.0 um, >=20 um branches"
    )
    parser.add_argument("--flim", type=str, default=DEFAULT_FLIM)
    parser.add_argument("--subdir", type=str, default=DEFAULT_OUT_SUBDIR)
    parser.add_argument(
        "--no-auto-zoom",
        action="store_true",
        help="Disable zoom-based presets from .flim State.Acq.zoom",
    )
    parser.add_argument(
        "--min-branch-width-um",
        type=float,
        default=None,
        help="Override min branch width (default: from zoom preset)",
    )
    parser.add_argument("--max-branch-width-um", type=float, default=None)
    parser.add_argument("--min-branch-length-um", type=float, default=None)
    parser.add_argument("--segment-length-um", type=float, default=None)
    parser.add_argument("--dendrite-percentile", type=float, default=None)
    parser.add_argument(
        "--overlap-um",
        type=float,
        default=5.0,
        help="Segment overlap in µm (stride = segment-length - overlap, default 5)",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=30,
        help="Max branches to score (representative segment each)",
    )
    parser.add_argument(
        "--min-spacing-um",
        type=float,
        default=20.0,
        help="Min 3D distance between selected branch rep. points",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of top-scoring segments in ROI tile PNG (default 8)",
    )
    parser.add_argument(
        "--roi-um",
        type=float,
        default=15.0,
        help="ROI square side length in µm for tile crops (default 15)",
    )
    parser.add_argument(
        "--min-roi-xy-sep-um",
        type=float,
        default=20.0,
        help="ROI overlap if XY center distance below this (default 20)",
    )
    parser.add_argument(
        "--min-roi-z-sep-um",
        type=float,
        default=10.0,
        help="ROI overlap if |Z| separation below this (default 10)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-step timing breakdown",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.flim):
        raise FileNotFoundError(args.flim)
    run_one(
        args.flim,
        out_subdir=args.subdir,
        min_branch_width_um=args.min_branch_width_um,
        max_branch_width_um=args.max_branch_width_um,
        min_branch_length_um=args.min_branch_length_um,
        segment_length_um=args.segment_length_um,
        overlap_um=args.overlap_um,
        dendrite_percentile=args.dendrite_percentile,
        max_branches=args.max_branches,
        min_spacing_um=args.min_spacing_um,
        use_zoom_presets=not args.no_auto_zoom,
        top_n=args.top_n,
        roi_um=args.roi_um,
        min_roi_xy_sep_um=args.min_roi_xy_sep_um,
        min_roi_z_sep_um=args.min_roi_z_sep_um,
        profile=args.profile,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
