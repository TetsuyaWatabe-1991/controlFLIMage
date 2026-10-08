"""Publication-style scatter of GCaMP F/F0 vs spine volume change.

Each point is one experiment day from
C:\\Users\\WatabeT\\Documents\\LTPanalysis_GCaMP_volMean.xlsx (Sheet2).
X and Y error bars are SD.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba

plt.rcParams["font.family"] = "Arial"

XLSX_PATH = Path(r"C:\Users\WatabeT\Documents\LTPanalysis_GCaMP_volMean.xlsx")
OUT_WHITE = Path(r"C:\Users\WatabeT\Documents\LTPanalysis_GCaMP_volMean.png")
OUT_TRANSPARENT = Path(r"C:\Users\WatabeT\Documents\LTPanalysis_GCaMP_volMean_transparent.png")

COLOR = "black"
SD_ALPHA = 0.35
FIGSIZE = (7.2, 3.6)
XLIM = (0, 20)
YLIM = (-0.4, 1.4)


def to_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("-", pd.NA), errors="coerce")


def load_sheet2(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Sheet2")
    df["deltaVOL"] = to_numeric_clean(df["deltaVOl"])
    df["deltaVolSD"] = to_numeric_clean(df["deltaVolSD"])
    df["ff0_spine"] = to_numeric_clean(df["F/F0 spine"])
    df["spineSD"] = to_numeric_clean(df["spineSD"])
    df["ff0_shaft"] = to_numeric_clean(df["F/F0 shaft"])
    df["shaftSD"] = to_numeric_clean(df["shaftSD"])
    df["vol"] = df["deltaVOL"] / 100.0
    df["volSD"] = df["deltaVolSD"] / 100.0
    return df


def plot_xy_sd(ax, x, y, xerr, yerr) -> None:
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        linestyle="none",
        capsize=0,
        elinewidth=1.0,
        markersize=6,
        color=COLOR,
        ecolor=to_rgba(COLOR, SD_ALPHA),
        markerfacecolor=COLOR,
        markeredgecolor=COLOR,
        zorder=3,
    )


def style_ax(ax, xlabel: str, ylabel: str | None) -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="both", labelsize=10, top=False, right=False)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks(range(0, 21, 5))
    ax.set_yticks([i / 10 for i in range(-4, 15, 2)])
    ax.grid(True, alpha=0.3, color="0.7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="0.6", linewidth=0.8, zorder=1)


def make_fig(df: pd.DataFrame, transparent: bool):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    if transparent:
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0)
        for ax in axes:
            ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor("white")
        for ax in axes:
            ax.set_facecolor("white")

    spine = df.dropna(subset=["ff0_spine", "spineSD", "vol", "volSD"])
    shaft = df.dropna(subset=["ff0_shaft", "shaftSD", "vol", "volSD"])

    plot_xy_sd(axes[0], spine["ff0_spine"], spine["vol"], spine["spineSD"], spine["volSD"])
    plot_xy_sd(axes[1], shaft["ff0_shaft"], shaft["vol"], shaft["shaftSD"], shaft["volSD"])

    style_ax(axes[0], "Spine F/F0", "Normalized \u0394spine volume (a.u.)")
    style_ax(axes[1], "Dendrite F/F0", None)

    fig.tight_layout()
    return fig, len(spine), len(shaft)


def main() -> None:
    df = load_sheet2(XLSX_PATH)
    fig, n_spine, n_shaft = make_fig(df, transparent=False)
    fig.savefig(OUT_WHITE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, _, _ = make_fig(df, transparent=True)
    fig.savefig(
        OUT_TRANSPARENT,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)

    print(f"spine points: {n_spine}")
    print(f"dendrite points: {n_shaft}")
    print(OUT_WHITE)
    print(OUT_TRANSPARENT)


if __name__ == "__main__":
    main()
