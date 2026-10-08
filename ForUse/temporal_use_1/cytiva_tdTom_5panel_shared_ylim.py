"""Digitize the 5-panel screenshot and replot on a shared y-axis."""
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

IMG_PATH = r"C:\Users\WatabeT\.cursor\projects\c-Users-WatabeT-Documents-Git\assets\c__Users_WatabeT_AppData_Roaming_Cursor_User_workspaceStorage_5bb29d15065c5cb138322ce9e50becbc_images_image-370ac749-ae23-4697-b5f7-8a8c15d14e89.png"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(OUT_DIR, "cytiva_tdTom_5panel_shared_ylim.png")
COMBINED_PATH = os.path.join(OUT_DIR, "cytiva_tdTom_combined_shared_ylim.png")
DEBUG_PATH = os.path.join(OUT_DIR, "cytiva_tdTom_5panel_digitize_debug.png")
CSV_PATH = os.path.join(OUT_DIR, "cytiva_tdTom_5panel_digitized.csv")

# tick_top/bot map to y_top/y_bot on the full screenshot.
PANELS = {
    "tdTom_3p9": dict(
        title="tdTom, uncaging 3.9 mW",
        x0=145, x1=200, y0=680, y1=810,
        tick_top_row=536, tick_bot_row=767, y_top=0.8, y_bot=0.0,
        min_distance=7, threshold_frac=0.16, n_target=6,
        grid=(1, 0),
    ),
    "cytiva_2p7": dict(
        title="Cytiva, uncaging 2.7 mW",
        x0=500, x1=570, y0=260, y1=470,
        tick_top_row=218, tick_bot_row=451, y_top=1.25, y_bot=-0.25,
        min_distance=6, threshold_frac=0.12, n_target=None,
        grid=(0, 1),
    ),
    "tdTom_2p7": dict(
        title="tdTom, uncaging 2.7 mW",
        x0=840, x1=900, y0=270, y1=450,
        tick_top_row=204, tick_bot_row=444, y_top=0.6, y_bot=-0.6,
        min_distance=8, threshold_frac=0.12, n_target=3,
        grid=(0, 2),
    ),
    "cytiva_4p0": dict(
        title="Cytiva, uncaging 4.0 mW",
        x0=510, x1=570, y0=530, y1=820,
        tick_top_row=566, tick_bot_row=799, y_top=1.25, y_bot=-0.25,
        min_distance=7, threshold_frac=0.12, n_target=7,
        grid=(1, 1),
    ),
    "tdTom_4p0": dict(
        title="tdTom, uncaging 4.0 mW",
        x0=840, x1=900, y0=520, y1=730,
        tick_top_row=544, tick_bot_row=818, y_top=0.6, y_bot=-0.6,
        min_distance=8, threshold_frac=0.12, n_target=4,
        grid=(1, 2),
    ),
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


def digitize_panel(rgb: np.ndarray, spec: dict) -> np.ndarray:
    crop = rgb[spec["y0"] : spec["y1"], spec["x0"] : spec["x1"]]
    local = detect_markers(crop, spec["min_distance"], spec["threshold_frac"])
    if spec["n_target"] is not None:
        local = merge_closest(local, spec["n_target"])
    if len(local) == 0:
        return np.zeros((0, 2), dtype=float)
    rows = local[:, 0] + spec["y0"]
    cols = local[:, 1] + spec["x0"]
    y = row_to_y(rows, spec)
    return np.column_stack([cols, rows, y])


def plot_panel(ax, y: np.ndarray, title: str, show_mean_text: bool = False) -> tuple[float, float]:
    mean = float(np.mean(y))
    sd = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    sns.swarmplot(y=y, ax=ax, color="#1f77b4", size=6, zorder=3)
    ax.plot([-0.28, 0.28], [mean, mean], color="#d62728", lw=2.0, zorder=2, solid_capstyle="butt")
    if show_mean_text:
        ax.text(0.42, mean, f"{mean:.2f} ± {sd:.2f}", va="center", ha="left", fontsize=15)
        ax.set_xlim(-0.55, 1.55)
    else:
        ax.set_xlim(-0.55, 0.55)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_title(title, fontsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=15)
    return mean, sd


def main() -> None:
    rgb = np.asarray(Image.open(IMG_PATH).convert("RGB"))
    digitized = {}
    rows_out = []
    for name, spec in PANELS.items():
        pts = digitize_panel(rgb, spec)
        y = pts[:, 2]
        sd = float(y.std(ddof=1)) if len(y) > 1 else 0.0
        print(f"{name:12s} n={len(y):2d}  mean={y.mean():.3f}  sd={sd:.3f}")
        print("             y", np.round(np.sort(y), 3))
        digitized[name] = pts
        rows_out.append(pd.DataFrame({"panel": name, "y": y}))

    pd.concat(rows_out, ignore_index=True).to_csv(CSV_PATH, index=False)
    print("saved", CSV_PATH)

    fig_dbg, ax_dbg = plt.subplots(figsize=(8.5, 7.2), dpi=140)
    ax_dbg.imshow(rgb)
    for pts in digitized.values():
        ax_dbg.scatter(pts[:, 0], pts[:, 1], s=22, facecolors="none", edgecolors="lime", lw=0.8)
    ax_dbg.set_axis_off()
    fig_dbg.tight_layout()
    fig_dbg.savefig(DEBUG_PATH, dpi=140, bbox_inches="tight")
    print("saved", DEBUG_PATH)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.4), sharey=True, dpi=220)
    axes[0, 0].axis("off")
    for name, spec in PANELS.items():
        r, c = spec["grid"]
        plot_panel(axes[r, c], digitized[name][:, 2], spec["title"])
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_ylabel(r"$\Delta$spine volume (a.u.)", fontsize=15)
        ax.tick_params(labelleft=True, labelsize=15)
    for ax in axes.ravel():
        ax.set_ylim(-0.7, 1.4)
        ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
    fig.tight_layout()
    fig.savefig(SAVE_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    print("saved", SAVE_PATH)

    combined = [
        ("tdTom, uncaging 3.9 mW", digitized["tdTom_3p9"][:, 2]),
        (
            "Cytiva, uncaging 2.7 / 4.0 mW",
            np.concatenate([digitized["cytiva_2p7"][:, 2], digitized["cytiva_4p0"][:, 2]]),
        ),
        (
            "tdTom, uncaging 2.7 / 4.0 mW",
            np.concatenate([digitized["tdTom_2p7"][:, 2], digitized["tdTom_4p0"][:, 2]]),
        ),
    ]
    fig2, axes2 = plt.subplots(1, 3, figsize=(11.0, 4.2), sharey=True, dpi=220)
    for ax, (title, y) in zip(axes2, combined):
        mean, sd = plot_panel(ax, y, title, show_mean_text=True)
        print(f"combined {title}: n={len(y)}  mean={mean:.3f}  sd={sd:.3f}")
    axes2[0].set_ylabel(r"$\Delta$spine volume (a.u.)", fontsize=15)
    for ax in axes2:
        ax.set_ylim(-0.7, 1.4)
        ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
    fig2.tight_layout()
    fig2.savefig(COMBINED_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    print("saved", COMBINED_PATH)


if __name__ == "__main__":
    main()
