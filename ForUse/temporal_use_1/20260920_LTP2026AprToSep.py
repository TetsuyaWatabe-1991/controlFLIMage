"""Plot LTP summary (2026 Apr-Sep) from Desktop LTPanalysis.csv.

Rules:
- X: date, Y: mean, error bars: SD (same hue, alpha 0.35, no caps)
- Exclude reject==1 unless Mg/cAMP/APV/Other/ballistic is 1
- Colors: control black, Mg blue, cAMP vermillion, APV green, Other purple
- ballistic==1: black star, even if rejected
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba

plt.rcParams["font.family"] = "Arial"

CSV_PATH = Path(r"C:\Users\WatabeT\Desktop\LTPanalysis.csv")
OUT_WHITE = Path(r"C:\Users\WatabeT\Desktop\LTPanalysis.png")
OUT_TRANSPARENT = Path(r"C:\Users\WatabeT\Desktop\LTPanalysis_transparent.png")

DATE_YEAR = 2026
YLIM = (-20, 140)
YTICKS = range(-20, 141, 20)
FIGSIZE = (7.2, 3.6)
SD_ALPHA = 0.35

COLOR_CTRL = "black"
COLOR_MG = "#0072B2"
COLOR_CAMP = "#D55E00"
COLOR_APV = "#009E73"
COLOR_OTHER = "#CC79A7"

FLAG_COLS = ["reject", "Mg", "cAMP", "APV", "Other", "ballistic"]


def load_ltp_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in FLAG_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["date_parsed"] = pd.to_datetime(df["date"] + f"-{DATE_YEAR}", format="%d-%b-%Y")
    return df.sort_values("date_parsed")


def split_groups(df: pd.DataFrame) -> list[tuple[pd.DataFrame, str, str, float]]:
    ballistic = df[df["ballistic"] == 1]
    mg = df[(df["Mg"] == 1) & (df["ballistic"] != 1)]
    camp = df[(df["cAMP"] == 1) & (df["Mg"] != 1) & (df["ballistic"] != 1)]
    apv = df[
        (df["APV"] == 1)
        & (df["Mg"] != 1)
        & (df["cAMP"] != 1)
        & (df["ballistic"] != 1)
    ]
    other = df[
        (df["Other"] == 1)
        & (df["Mg"] != 1)
        & (df["cAMP"] != 1)
        & (df["APV"] != 1)
        & (df["ballistic"] != 1)
    ]
    ctrl = df[
        (df["reject"] != 1)
        & (df["Mg"] != 1)
        & (df["cAMP"] != 1)
        & (df["APV"] != 1)
        & (df["Other"] != 1)
        & (df["ballistic"] != 1)
    ]
    return [
        (ctrl, COLOR_CTRL, "o", 6),
        (ballistic, COLOR_CTRL, "*", 10),
        (mg, COLOR_MG, "o", 6),
        (camp, COLOR_CAMP, "o", 6),
        (apv, COLOR_APV, "o", 6),
        (other, COLOR_OTHER, "o", 6),
    ]


def plot_group(ax, data: pd.DataFrame, color: str, marker: str, markersize: float) -> None:
    if data.empty:
        return
    ax.errorbar(
        data["date_parsed"],
        data["mean"],
        yerr=data["sd"],
        fmt=marker,
        linestyle="none",
        capsize=0,
        elinewidth=1.0,
        markersize=markersize,
        color=color,
        ecolor=to_rgba(color, SD_ALPHA),
        markerfacecolor=color,
        markeredgecolor=color,
    )


def make_fig(groups, transparent: bool):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if transparent:
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

    for data, color, marker, markersize in groups:
        plot_group(ax, data, color, marker, markersize)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Normalized \u0394volume (a.u.)", fontsize=10)
    ax.tick_params(axis="both", labelsize=10, top=False, right=False)
    ax.set_ylim(*YLIM)
    ax.set_yticks(list(YTICKS))
    ax.grid(True, alpha=0.3, color="0.7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def main() -> None:
    df = load_ltp_table(CSV_PATH)
    groups = split_groups(df)

    fig = make_fig(groups, transparent=False)
    fig.savefig(OUT_WHITE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig = make_fig(groups, transparent=True)
    fig.savefig(
        OUT_TRANSPARENT,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)

    labels = ["control", "ballistic", "Mg", "cAMP", "APV", "Other"]
    for (data, _, _, _), name in zip(groups, labels):
        print(f"{name}: {len(data)}")
    print(OUT_WHITE)
    print(OUT_TRANSPARENT)


if __name__ == "__main__":
    main()
