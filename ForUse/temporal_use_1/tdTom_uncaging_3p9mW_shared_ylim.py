"""Digitize markers from the screenshot and replot both panels on a shared y-axis.

Y is calibrated from the original tick marks (not the bottom spine). Mean +/- SD
are recomputed from the digitized points and will not match the screenshot exactly.
"""
from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

controlFLIMage_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(controlFLIMage_DIR)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
from custom_plot import plt

IMG_PATH = r"C:\Users\WatabeT\.cursor\projects\c-Users-WatabeT-Documents-Git\assets\c__Users_WatabeT_AppData_Roaming_Cursor_User_workspaceStorage_5bb29d15065c5cb138322ce9e50becbc_images_image-372f59f1-b6b2-462d-abbf-e1cd395ab45d.png"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(OUT_DIR, "tdTom_uncaging_3p9mW_shared_ylim.png")
DEBUG_PATH = os.path.join(OUT_DIR, "tdTom_uncaging_3p9mW_digitize_debug.png")
CSV_PATH = os.path.join(OUT_DIR, "tdTom_uncaging_3p9mW_digitized.csv")

# Data-region crop on the letterbox-trimmed image, plus tick rows for y mapping.
LEFT = dict(x0=150, x1=220, y0=40, y1=330, tick_top_row=54, tick_bot_row=297, y_top=0.8, y_bot=0.0)
RIGHT = dict(x0=450, x1=590, y0=40, y1=340, tick_top_row=63, tick_bot_row=305, y_top=4.0, y_bot=0.0)


def load_trimmed(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"))
    row_mean = arr.mean(axis=(1, 2))
    content = np.where(row_mean > 20)[0]
    return arr[content[0] : content[-1] + 1]


def blue_score(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    score = b - 0.55 * (r + g)
    score[(b < 90) | (b <= r + 8) | (b <= g)] = 0
    return np.clip(score, 0, None)


def detect_markers(rgb: np.ndarray, min_distance: int, threshold_frac: float) -> np.ndarray:
    score = gaussian_filter(blue_score(rgb), sigma=0.9)
    peak = maximum_filter(score, size=min_distance)
    thresh = threshold_frac * float(score.max() if score.max() > 0 else 1.0)
    mask = (score == peak) & (score >= thresh)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=float)

    # Refine to local center of mass so overlapping blobs sit on the marker, not the peak pixel.
    centers = []
    rad = max(2, min_distance // 2)
    h, w = score.shape
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - rad), min(h, y + rad + 1)
        x0, x1 = max(0, x - rad), min(w, x + rad + 1)
        patch = score[y0:y1, x0:x1]
        if patch.sum() <= 0:
            continue
        yy, xx = np.indices(patch.shape)
        cy = y0 + float((yy * patch).sum() / patch.sum())
        cx = x0 + float((xx * patch).sum() / patch.sum())
        centers.append((cy, cx))
    return np.asarray(centers, dtype=float)


def merge_closest(points: np.ndarray, n_target: int) -> np.ndarray:
    """Iteratively replace the closest pair with their midpoint until n_target remains."""
    pts = [p.copy() for p in points]
    while len(pts) > n_target:
        best = (1e9, 0, 1)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = float(np.hypot(pts[i][0] - pts[j][0], 0.35 * (pts[i][1] - pts[j][1])))
                if d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        mid = 0.5 * (pts[i] + pts[j])
        pts = [p for k, p in enumerate(pts) if k not in (i, j)] + [mid]
    return np.asarray(pts)


def row_to_y(row: np.ndarray, spec: dict) -> np.ndarray:
    frac = (row - spec["tick_top_row"]) / (spec["tick_bot_row"] - spec["tick_top_row"])
    return spec["y_top"] + frac * (spec["y_bot"] - spec["y_top"])


def digitize_panel(
    rgb: np.ndarray,
    spec: dict,
    min_distance: int,
    threshold_frac: float,
    n_target: int | None = None,
) -> np.ndarray:
    crop = rgb[spec["y0"] : spec["y1"], spec["x0"] : spec["x1"]]
    local = detect_markers(crop, min_distance=min_distance, threshold_frac=threshold_frac)
    if n_target is not None:
        local = merge_closest(local, n_target)
    if len(local) == 0:
        return np.zeros((0, 2), dtype=float)
    rows = local[:, 0] + spec["y0"]
    cols = local[:, 1] + spec["x0"]
    y = row_to_y(rows, spec)
    return np.column_stack([cols, rows, y])


def plot_panel(ax, y: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(y))
    sd = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    sns.swarmplot(y=y, ax=ax, color="#1f77b4", size=6, zorder=3)
    ax.plot([-0.28, 0.28], [mean, mean], color="#d62728", lw=2.0, zorder=2, solid_capstyle="butt")
    ax.set_xlim(-0.55, 0.55)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_title("tdTom, uncaging 3.9 mW", fontsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=15)
    return mean, sd


def main() -> None:
    rgb = load_trimmed(IMG_PATH)
    left = digitize_panel(rgb, LEFT, min_distance=7, threshold_frac=0.16, n_target=6)
    right = digitize_panel(rgb, RIGHT, min_distance=7, threshold_frac=0.10)

    y_left = left[:, 2]
    y_right = right[:, 2]
    print(f"left  n={len(y_left):2d}  mean={y_left.mean():.3f}  sd={y_left.std(ddof=1):.3f}")
    print("left  y", np.round(np.sort(y_left), 3))
    print(f"right n={len(y_right):2d}  mean={y_right.mean():.3f}  sd={y_right.std(ddof=1):.3f}")
    print("right y", np.round(np.sort(y_right), 3))

    df = pd.concat(
        [
            pd.DataFrame({"panel": "left", "y": y_left}),
            pd.DataFrame({"panel": "right", "y": y_right}),
        ],
        ignore_index=True,
    )
    df.to_csv(CSV_PATH, index=False)
    print("saved", CSV_PATH)

    fig_dbg, ax_dbg = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    ax_dbg.imshow(rgb)
    ax_dbg.scatter(left[:, 0], left[:, 1], s=28, facecolors="none", edgecolors="lime", lw=0.9)
    ax_dbg.scatter(right[:, 0], right[:, 1], s=28, facecolors="none", edgecolors="lime", lw=0.9)
    ax_dbg.set_axis_off()
    fig_dbg.tight_layout()
    fig_dbg.savefig(DEBUG_PATH, dpi=160, bbox_inches="tight")
    print("saved", DEBUG_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0), sharey=True, dpi=220)
    plot_panel(axes[0], y_left)
    plot_panel(axes[1], y_right)
    axes[0].set_ylabel(r"$\Delta$spine volume (a.u.)", fontsize=15)
    axes[0].set_ylim(-0.25, 4.4)
    axes[0].set_yticks([0, 1, 2, 3, 4])
    fig.tight_layout()
    fig.savefig(SAVE_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    print("saved", SAVE_PATH)


if __name__ == "__main__":
    main()
