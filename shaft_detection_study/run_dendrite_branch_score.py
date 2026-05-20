# -*- coding: utf-8 -*-
"""
Score low-mag dendrite skeleton branches (smooth shaft + spiny = high score).

Example:
  python controlFLIMage/shaft_detection_study/run_dendrite_branch_score.py ^
    --flim "\\\\server\\path\\AP5_pos6_256_4x_001.flim"
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

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
from lowmag_mushroom_branch_finder import physical_distance_um
from shaft_detection_study.dendrite_branch_score_core import (
    DEFAULT_MAX_BRANCH_WIDTH_UM,
    DEFAULT_MIN_BRANCH_WIDTH_UM,
    score_all_branches,
)

DEFAULT_FLIM = (
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260515\lowmags\AP5_pos6_256_4x_001.flim"
)
DEFAULT_OUT_SUBDIR = "dendrite_branch_score"


def select_spaced(candidates, max_num, min_spacing_um, xy_um, z_um):
    ordered = sorted(candidates, key=lambda c: c["branch_score"], reverse=True)
    selected = []
    for cand in ordered:
        if len(selected) >= max_num:
            break
        if all(
            physical_distance_um(cand["zyx"], s["zyx"], xy_um, z_um) >= min_spacing_um
            for s in selected
        ):
            selected.append(cand)
    return selected


def save_overview_png(result: dict, selected: list, savepath: str) -> None:
    zyx = result["zyx"]
    sk = result["skeleton_3d"]
    prot = result["protrusion"]
    puncta = result["puncta_zyx"]
    mip = zyx.max(axis=0)
    sk_mip = sk.max(axis=0)
    prot_mip = prot.max(axis=0)
    vmax = np.percentile(mip, 99.5) if mip.max() > 0 else 1.0
    pmax = np.percentile(prot_mip[prot_mip > 0], 95) if (prot_mip > 0).any() else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax = axes[0]
    ax.imshow(mip, cmap="gray", vmax=vmax)
    ax.imshow(np.ma.masked_where(prot_mip <= 0, prot_mip), cmap="magma", alpha=0.45, vmax=pmax)
    ax.imshow(np.ma.masked_where(~sk_mip, sk_mip), cmap="Greens", alpha=0.2)
    if len(puncta) > 0:
        ax.scatter(puncta[:, 2], puncta[:, 1], s=5, c="cyan", alpha=0.5)
    for cand in result["candidates"][:20]:
        x, y = cand["zyx"][2], cand["zyx"][1]
        ax.scatter(x, y, s=20, c="white", alpha=0.35)
        ax.text(
            x + 2,
            y,
            f"{cand['branch_score']:.2f}",
            color="white",
            fontsize=5,
            alpha=0.7,
        )
    for cand in selected:
        x, y = cand["zyx"][2], cand["zyx"][1]
        ax.scatter(x, y, s=120, facecolors="none", edgecolors="yellow", linewidths=2)
        ax.text(
            x + 3,
            y + 3,
            f"b{cand['branch_id']}",
            color="yellow",
            fontsize=8,
        )
    ax.set_title("Z-MIP: score labels; yellow = selected")
    ax.axis("off")

    axes[1].imshow(prot_mip, cmap="hot", vmax=pmax)
    axes[1].set_title("Spine protrusion (top-hat Z-MIP)")
    axes[1].axis("off")

    top = result["candidates"][:12]
    labels = [f"b{c['branch_id']}" for c in top]
    scores = [c["branch_score"] for c in top]
    smooth = [c["smoothness"] for c in top]
    xpos = np.arange(len(top))
    axes[2].bar(xpos - 0.2, scores, width=0.4, label="branch_score")
    axes[2].bar(xpos + 0.2, smooth, width=0.4, label="smoothness")
    axes[2].set_xticks(xpos, labels, rotation=45, fontsize=7)
    axes[2].legend(fontsize=7)
    axes[2].set_title("Top branches")
    fig.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scores_csv(candidates: list, selected: list, csv_path: str) -> None:
    fields = [
        "branch_id",
        "z_pix",
        "y_pix",
        "x_pix",
        "branch_score",
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
        "selected",
    ]
    selected_set = {tuple(c["zyx"]) for c in selected}
    with open(csv_path, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        for cand in candidates:
            z, y, x = cand["zyx"]
            writer.writerow(
                {
                    "branch_id": cand["branch_id"],
                    "z_pix": int(z),
                    "y_pix": int(y),
                    "x_pix": int(x),
                    "branch_score": cand["branch_score"],
                    "spine_signal": cand["spine_signal"],
                    "smoothness": cand["smoothness"],
                    "width_cv": cand["width_cv"],
                    "width_range_ratio": cand["width_range_ratio"],
                    "mean_width_um": cand["mean_width_um"],
                    "min_width_um": cand["min_width_um"],
                    "max_width_um": cand["max_width_um"],
                    "protrusion_density": cand["protrusion_density"],
                    "puncta_count": int(cand["puncta_count"]),
                    "roughness": cand["roughness"],
                    "selected": int(tuple(cand["zyx"]) in selected_set),
                }
            )


def run_one(
    flim_path: str,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
    max_candidates: int = 8,
    min_spacing_um: float = 20.0,
    min_branch_width_um: float = DEFAULT_MIN_BRANCH_WIDTH_UM,
    max_branch_width_um: float = DEFAULT_MAX_BRANCH_WIDTH_UM,
) -> str:
    out_dir = os.path.join(resolve_savefolder(flim_path), out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {flim_path} ===")
    print(f"Output: {out_dir}")
    print(
        f"  branch width filter: {min_branch_width_um} < mean <= {max_branch_width_um} um"
    )
    result = score_all_branches(
        flim_path,
        min_branch_width_um=min_branch_width_um,
        max_branch_width_um=max_branch_width_um,
    )
    xy_um, z_um = result["x_um"], result["z_um"]
    candidates = result["candidates"]
    print(f"  skeleton branches: {result['n_branches']}")
    print(f"  scored branches: {len(candidates)}")

    selected = select_spaced(candidates, max_candidates, min_spacing_um, xy_um, z_um)
    print(f"  selected (spaced): {len(selected)}")

    csv_path = os.path.join(out_dir, "branch_scores.csv")
    save_scores_csv(candidates, selected, csv_path)
    overview_path = os.path.join(out_dir, "branch_score_overview.png")
    save_overview_png(result, selected, overview_path)

    meta = os.path.join(out_dir, "run_meta.txt")
    with open(meta, "w", encoding="utf-8") as f:
        f.write(f"flim={flim_path}\n")
        f.write(f"timestamp={datetime.now().isoformat()}\n")
        f.write(f"n_scored={len(candidates)}\n")
        f.write(f"n_selected={len(selected)}\n")

    print(f"  CSV: {csv_path}")
    print(f"  PNG: {overview_path}")
    if candidates:
        best = candidates[0]
        print(
            f"  best branch {best['branch_id']}: score={best['branch_score']:.3f} "
            f"smooth={best['smoothness']:.3f} spine={best['spine_signal']:.3f} "
            f"width_cv={best['width_cv']:.3f}"
        )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score dendrite branches: smooth shaft + spiny protrusions"
    )
    parser.add_argument("--flim", type=str, default=DEFAULT_FLIM)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*256_4x_001.flim")
    parser.add_argument("--subdir", type=str, default=DEFAULT_OUT_SUBDIR)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--min-spacing-um", type=float, default=20.0)
    parser.add_argument(
        "--min-branch-width-um",
        type=float,
        default=DEFAULT_MIN_BRANCH_WIDTH_UM,
        help="Exclude branch if mean shaft width <= this (default 0.3)",
    )
    parser.add_argument(
        "--max-branch-width-um",
        type=float,
        default=DEFAULT_MAX_BRANCH_WIDTH_UM,
        help="Exclude branch if mean shaft width > this (default 1.1)",
    )
    args = parser.parse_args()

    if args.flim:
        paths = [args.flim]
    elif args.folder:
        paths = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    else:
        paths = [DEFAULT_FLIM]

    for flim_path in paths:
        if not os.path.isfile(flim_path):
            raise FileNotFoundError(flim_path)
        run_one(
            flim_path,
            out_subdir=args.subdir,
            max_candidates=args.max_candidates,
            min_spacing_um=args.min_spacing_um,
            min_branch_width_um=args.min_branch_width_um,
            max_branch_width_um=args.max_branch_width_um,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
