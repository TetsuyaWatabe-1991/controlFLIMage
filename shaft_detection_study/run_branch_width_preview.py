# -*- coding: utf-8 -*-
"""
Fast preview: raw | skeleton before prune | after prune | overlay | soma | legend.

Default width bins: 0.5–1.0 um in 0.1 um steps (<0.5, >=1.0 outside).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
from lowmag_mushroom_branch_finder import (
    build_dendrite_and_protrusion_maps,
    label_skeleton_branches,
)
from shaft_detection_study.dendrite_branch_score_core import (
    load_flim_zyx,
    shaft_width_profile_um,
)

DEFAULT_FLIM = (
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260515\lowmags\AP5_pos6_256_4x_001.flim"
)

BIN_START_UM = 0.5
BIN_STEP_UM = 0.1
N_BINS = 5  # 0.5–0.6 .. 0.9–1.0 um
DEFAULT_MIN_BRANCH_LENGTH_UM = 20.0


def make_width_bin_palette(
    bin_start: float = BIN_START_UM,
    bin_step: float = BIN_STEP_UM,
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[str]]:
    """below color, bin colors, above color, bin labels."""
    below = np.array([0.12, 0.12, 0.45], dtype=np.float32)
    above = np.array([0.55, 0.08, 0.08], dtype=np.float32)
    cmap = plt.cm.tab10(np.linspace(0.05, 0.95, n_bins))[:, :3].astype(np.float32)
    labels: list[str] = [f"<{bin_start:.1f}"]
    for i in range(n_bins):
        lo = bin_start + i * bin_step
        hi = lo + bin_step
        labels.append(f"{lo:.1f}–{hi:.1f}")
    labels.append(f"≥{bin_start + n_bins * bin_step:.1f}")
    return below, list(cmap), above, labels


def color_for_mean_width(
    mean_width_um: float,
    below: np.ndarray,
    bin_colors: list[np.ndarray],
    above: np.ndarray,
    *,
    bin_start: float = BIN_START_UM,
    bin_step: float = BIN_STEP_UM,
) -> tuple[np.ndarray, str, int]:
    n_bins = len(bin_colors)
    if mean_width_um < bin_start:
        return below, f"<{bin_start:.1f}", -1
    idx = int((mean_width_um - bin_start) / bin_step)
    if idx >= n_bins:
        return above, f"≥{bin_start + n_bins * bin_step:.1f}", n_bins
    lo = bin_start + idx * bin_step
    hi = lo + bin_step
    return bin_colors[idx], f"{lo:.1f}–{hi:.1f}", idx


def build_colored_branch_mip(
    labeled_skeleton: np.ndarray,
    dendrite_mask: np.ndarray,
    xy_um: float,
    z_um: float,
    *,
    bin_start: float = BIN_START_UM,
    bin_step: float = BIN_STEP_UM,
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, list[dict], list[str]]:
    below, bin_colors, above, bin_labels = make_width_bin_palette(
        bin_start, bin_step, n_bins
    )
    h, w = labeled_skeleton.shape[1], labeled_skeleton.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    stats: list[dict] = []
    bin_counts = {lab: 0 for lab in bin_labels}

    n_branch = int(labeled_skeleton.max())
    for branch_id in range(1, n_branch + 1):
        branch_pts = np.array(np.where(labeled_skeleton == branch_id)).T
        if len(branch_pts) < 3:
            continue
        widths = shaft_width_profile_um(branch_pts, dendrite_mask, xy_um)
        mean_w = float(np.mean(widths))
        color, lab, bidx = color_for_mean_width(
            mean_w, below, bin_colors, above, bin_start=bin_start, bin_step=bin_step
        )
        bin_counts[lab] = bin_counts.get(lab, 0) + 1

        for z, y, x in branch_pts:
            rgb[int(y), int(x)] = color

        stats.append(
            {
                "branch_id": branch_id,
                "mean_width_um": mean_w,
                "width_bin": lab,
                "bin_index": bidx,
                "min_width_um": float(np.min(widths)),
                "max_width_um": float(np.max(widths)),
            }
        )

    return rgb, stats, bin_labels


def save_preview_png(
    zyx: np.ndarray,
    branch_rgb_before: np.ndarray,
    branch_rgb_after: np.ndarray,
    soma_mask: np.ndarray,
    savepath: str,
    *,
    min_branch_length_um: float,
    bin_step_um: float,
    below: np.ndarray,
    bin_colors: list[np.ndarray],
    above: np.ndarray,
    bin_labels: list[str],
    bin_counts: dict[str, int],
    n_before: int,
    n_after: int,
) -> None:
    mip = zyx.max(axis=0)
    vmax = float(np.percentile(mip, 99.5)) if mip.max() > 0 else 1.0
    gray_kw = dict(cmap="gray", vmin=0.0, vmax=vmax)

    sk_before_rgba = _rgb_to_rgba(branch_rgb_before, alpha=0.95)
    sk_after_rgba = _rgb_to_rgba(branch_rgb_after, alpha=0.95)
    soma_mip = soma_mask.max(axis=0).astype(bool)

    fig = plt.figure(figsize=(22.5, 4.8), facecolor="black")
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.34], wspace=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_ov = fig.add_subplot(gs[0, 3])
    ax3 = fig.add_subplot(gs[0, 4])
    ax_leg = fig.add_subplot(gs[0, 5])

    for ax in (ax0, ax1, ax2, ax_ov, ax3, ax_leg):
        ax.set_facecolor("black")

    ax0.imshow(mip, **gray_kw)
    ax0.set_title("Raw Z-MIP", color="white", fontsize=10)
    ax0.axis("off")

    ax1.imshow(np.zeros((branch_rgb_before.shape[0], branch_rgb_before.shape[1], 3)))
    ax1.imshow(sk_before_rgba)
    ax1.set_title(
        f"Before prune (n={n_before}), {bin_step_um:.1f} um bins",
        color="white",
        fontsize=9,
    )
    ax1.axis("off")

    ax2.imshow(np.zeros((branch_rgb_after.shape[0], branch_rgb_after.shape[1], 3)))
    ax2.imshow(sk_after_rgba)
    ax2.set_title(
        f"After prune >= {min_branch_length_um:.0f} um (n={n_after})",
        color="white",
        fontsize=9,
    )
    ax2.axis("off")

    ax_ov.imshow(mip, **gray_kw, zorder=0)
    ax_ov.imshow(sk_after_rgba, zorder=1)
    ax_ov.set_title(
        f"Overlay: kept >= {min_branch_length_um:.0f} um",
        color="white",
        fontsize=9,
    )
    ax_ov.axis("off")

    soma_rgba = np.zeros((*soma_mip.shape, 4), dtype=np.float32)
    soma_rgba[soma_mip, 0] = 1.0
    soma_rgba[soma_mip, 3] = 0.55
    ax3.imshow(mip, **gray_kw, zorder=0)
    ax3.imshow(soma_rgba, zorder=1)
    ax3.set_title("Soma mask Z-MIP (excluded)", color="white", fontsize=10)
    ax3.axis("off")

    legend_patches = [
        Patch(
            facecolor=below,
            edgecolor="0.7",
            label=f"{bin_labels[0]} (n={bin_counts.get(bin_labels[0], 0)})",
        )
    ]
    for i, c in enumerate(bin_colors):
        lab = bin_labels[i + 1]
        legend_patches.append(
            Patch(
                facecolor=c,
                edgecolor="0.7",
                label=f"{lab} um (n={bin_counts.get(lab, 0)})",
            )
        )
    legend_patches.append(
        Patch(
            facecolor=above,
            edgecolor="0.7",
            label=f"{bin_labels[-1]} (n={bin_counts.get(bin_labels[-1], 0)})",
        )
    )
    leg = ax_leg.legend(
        handles=legend_patches,
        loc="center left",
        fontsize=7,
        frameon=True,
        facecolor="0.15",
        edgecolor="0.5",
        labelcolor="white",
        title="Mean shaft width (um)",
        title_fontsize=8,
    )
    leg.get_title().set_color("white")
    ax_leg.axis("off")

    fig.savefig(savepath, dpi=160, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _rgb_to_rgba(rgb: np.ndarray, *, alpha: float) -> np.ndarray:
    mask = np.any(rgb > 0, axis=2)
    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
    rgba[..., :3] = rgb
    rgba[..., 3] = mask.astype(np.float32) * alpha
    return rgba


def _print_bin_counts(bin_labels: list[str], stats: list[dict], header: str) -> None:
    counts = {lab: 0 for lab in bin_labels}
    for s in stats:
        counts[s["width_bin"]] = counts.get(s["width_bin"], 0) + 1
    print(f"  {header}:")
    for lab in bin_labels:
        if counts.get(lab, 0):
            print(f"    {lab}: {counts[lab]}")


def run_preview(
    flim_path: str,
    out_subdir: str = "branch_width_preview",
    bin_start: float = BIN_START_UM,
    bin_step: float = BIN_STEP_UM,
    n_bins: int = N_BINS,
    min_branch_length_um: float = DEFAULT_MIN_BRANCH_LENGTH_UM,
) -> str:
    out_dir = os.path.join(resolve_savefolder(flim_path), out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {flim_path} ===")
    print(f"Output: {out_dir}")
    print(f"  width bins: {bin_start}–{bin_start + n_bins * bin_step} um, step {bin_step}")

    zyx, x_um, _, z_um = load_flim_zyx(flim_path)
    min_branch_pix = max(8, int(round(min_branch_length_um / x_um)))
    print(
        f"  min branch length at skeletonize: {min_branch_length_um} um "
        f"({min_branch_pix} voxels, xy={x_um:.3f} um/pix)"
    )

    (
        _,
        dendrite_mask,
        _,
        skeleton_after,
        soma_mask,
        skeleton_pruned_off,
        skeleton_before,
    ) = build_dendrite_and_protrusion_maps(
        zyx, x_um, min_skeleton_length_um=min_branch_length_um
    )

    labeled_before, n_before = label_skeleton_branches(skeleton_before)
    labeled_after, n_after = label_skeleton_branches(skeleton_after)
    n_removed_vox = int(skeleton_pruned_off.sum()) if skeleton_pruned_off is not None else 0
    print(f"  branches before prune: {n_before}  after prune: {n_after}")
    print(f"  pruned skeleton voxels: {n_removed_vox}")

    below, bin_colors, above, bin_labels = make_width_bin_palette(
        bin_start, bin_step, n_bins
    )

    branch_rgb_before, stats_before, _ = build_colored_branch_mip(
        labeled_before,
        dendrite_mask,
        x_um,
        z_um,
        bin_start=bin_start,
        bin_step=bin_step,
        n_bins=n_bins,
    )
    branch_rgb_after, stats_after, _ = build_colored_branch_mip(
        labeled_after,
        dendrite_mask,
        x_um,
        z_um,
        bin_start=bin_start,
        bin_step=bin_step,
        n_bins=n_bins,
    )

    bin_counts: dict[str, int] = {lab: 0 for lab in bin_labels}
    for s in stats_after:
        bin_counts[s["width_bin"]] = bin_counts.get(s["width_bin"], 0) + 1

    _print_bin_counts(bin_labels, stats_before, "bins before prune")
    _print_bin_counts(bin_labels, stats_after, "bins after prune")

    png_path = os.path.join(out_dir, "branch_width_bins.png")
    save_preview_png(
        zyx,
        branch_rgb_before,
        branch_rgb_after,
        soma_mask,
        png_path,
        min_branch_length_um=min_branch_length_um,
        bin_step_um=bin_step,
        below=below,
        bin_colors=bin_colors,
        above=above,
        bin_labels=bin_labels,
        bin_counts=bin_counts,
        n_before=n_before,
        n_after=n_after,
    )
    print(f"  saved: {png_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Branch width bin color preview")
    parser.add_argument("--flim", type=str, default=DEFAULT_FLIM)
    parser.add_argument("--subdir", type=str, default="branch_width_preview")
    parser.add_argument("--bin-start", type=float, default=BIN_START_UM)
    parser.add_argument("--bin-step", type=float, default=BIN_STEP_UM)
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    parser.add_argument(
        "--min-branch-length-um",
        type=float,
        default=DEFAULT_MIN_BRANCH_LENGTH_UM,
        help="Min skeleton component length at skeletonize (um, ~voxels/pixel size)",
    )
    args = parser.parse_args()

    run_preview(
        args.flim,
        out_subdir=args.subdir,
        bin_start=args.bin_start,
        bin_step=args.bin_step,
        n_bins=args.n_bins,
        min_branch_length_um=args.min_branch_length_um,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
