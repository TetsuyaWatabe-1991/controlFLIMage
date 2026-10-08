"""Digitize the 3-panel screenshot and replot on a shared y-axis."""
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

IMG_PATH = r"C:\Users\WatabeT\.cursor\projects\c-Users-WatabeT-Documents-Git\assets\c__Users_WatabeT_AppData_Roaming_Cursor_User_workspaceStorage_5bb29d15065c5cb138322ce9e50becbc_images_image-839c1b73-1cea-475d-bf8d-5548b2df2c0b.png"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(OUT_DIR, "spine_volume_3panel_shared_ylim.png")
DEBUG_PATH = os.path.join(OUT_DIR, "spine_volume_3panel_digitize_debug.png")
CSV_PATH = os.path.join(OUT_DIR, "spine_volume_3panel_digitized.csv")

PANELS = {
    "left": dict(x0=145, x1=215, y0=15, y1=250, tick_top_row=25, tick_bot_row=216, y_top=1.25, y_bot=-0.25),
    "mid": dict(x0=490, x1=580, y0=15, y1=245, tick_top_row=20, tick_bot_row=233, y_top=3.0, y_bot=-0.5),
    "right": dict(x0=850, x1=920, y0=15, y1=230, tick_top_row=53, tick_bot_row=217, y_top=1.5, y_bot=0.0),
}


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
    sns.swarmplot(y=y, ax=ax, color="#1f77b4", size=5.5, zorder=3)
    ax.plot([-0.28, 0.28], [mean, mean], color="#d62728", lw=2.0, zorder=2, solid_capstyle="butt")
    ax.set_xlim(-0.55, 0.55)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=15)
    return mean, sd


def summarize(name: str, y: np.ndarray) -> None:
    sd = float(y.std(ddof=1)) if len(y) > 1 else 0.0
    print(f"{name:5s} n={len(y):2d}  mean={y.mean():.3f}  sd={sd:.3f}")
    print("     y", np.round(np.sort(y), 3))


def main() -> None:
    rgb = np.asarray(Image.open(IMG_PATH).convert("RGB"))
    left = digitize_panel(rgb, PANELS["left"], min_distance=6, threshold_frac=0.12)
    mid = digitize_panel(rgb, PANELS["mid"], min_distance=6, threshold_frac=0.10)
    right = digitize_panel(rgb, PANELS["right"], min_distance=8, threshold_frac=0.12, n_target=7)

    ys = {"left": left[:, 2], "mid": mid[:, 2], "right": right[:, 2]}
    for name, y in ys.items():
        summarize(name, y)

    df = pd.concat(
        [pd.DataFrame({"panel": name, "y": y}) for name, y in ys.items()],
        ignore_index=True,
    )
    df.to_csv(CSV_PATH, index=False)
    print("saved", CSV_PATH)

    fig_dbg, ax_dbg = plt.subplots(figsize=(10.5, 3.2), dpi=160)
    ax_dbg.imshow(rgb)
    ax_dbg.scatter(left[:, 0], left[:, 1], s=22, facecolors="none", edgecolors="lime", lw=0.8)
    ax_dbg.scatter(mid[:, 0], mid[:, 1], s=22, facecolors="none", edgecolors="lime", lw=0.8)
    ax_dbg.scatter(right[:, 0], right[:, 1], s=22, facecolors="none", edgecolors="lime", lw=0.8)
    ax_dbg.set_axis_off()
    fig_dbg.tight_layout()
    fig_dbg.savefig(DEBUG_PATH, dpi=160, bbox_inches="tight")
    print("saved", DEBUG_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.0), sharey=True, dpi=220)
    for ax, name in zip(axes, ("left", "mid", "right")):
        plot_panel(ax, ys[name])
    axes[0].set_ylabel(r"$\Delta$spine volume (a.u.)", fontsize=15)
    axes[0].set_ylim(-0.55, 3.15)
    axes[0].set_yticks([0, 1, 2, 3])
    fig.tight_layout()
    fig.savefig(SAVE_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    print("saved", SAVE_PATH)


if __name__ == "__main__":
    main()
