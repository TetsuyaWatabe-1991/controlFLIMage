# %%
"""Plot 20260914 pos1 TimeCourse CSVs aligned to uncaging.

nAveFrame is 8 during uncaging and 3 otherwise. Intensity is divided by
nAveFrame, then dF/F0 = F / F_pre - 1, with F_pre = mean of pre-uncaging.
"""
from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

controlFLIMage_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(controlFLIMage_DIR)
sys.path.append(os.path.join(controlFLIMage_DIR, "AnalysisForFLIMage"))

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
from custom_plot import plt
from read_flimagecsv import arrange_for_multipos3, csv_to_df, detect_uncaging

CSV_PATHS = [
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260914\auto1\Analysis\copied\pos1__highmag_7__TimeCourse.csv",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260914\auto1\Analysis\copied\pos1__highmag_8__TimeCourse.csv",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260914\auto1\Analysis\copied\pos1__highmag_2__TimeCourse.csv",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260914\auto1\Analysis\copied\pos1__highmag_3__TimeCourse.csv",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260914\auto1\Analysis\copied\pos1__highmag_4__TimeCourse.csv",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260914\auto1\Analysis\copied\pos1__highmag_6__TimeCourse.csv",
]

SAVE_DIR = os.path.dirname(CSV_PATHS[0])
INTENSITY_COL = "meanIntensity-ROI"
NAVE_UNCAGING = 8
NAVE_OTHER = 3
CHANNELS = (1, 2)
LTP_WINDOW_MIN = (25.0, 35.0)


def spine_label(csv_path: str) -> str:
    """Short label from TimeCourse filename, e.g. highmag_7."""
    stem = os.path.basename(csv_path).replace("__TimeCourse.csv", "")
    if "highmag_" in stem:
        return "highmag_" + stem.split("highmag_")[-1].strip("_")
    return stem


def assign_nave_and_dff0(df: pd.DataFrame, intensity_col: str = INTENSITY_COL) -> pd.DataFrame:
    """Divide intensity by nAveFrame, then set pre-uncaging mean to 0 via F/F0 - 1."""
    out = df.copy()
    out["nAveFrame"] = np.where(out["during_uncaging"] == 1, NAVE_UNCAGING, NAVE_OTHER)
    out["intensity_div_nAve"] = out[intensity_col] / out["nAveFrame"]
    out["dFF0"] = np.nan
    out["F0"] = np.nan
    out["frame_from_unc"] = np.nan

    grouped = out.groupby(["FilePath", "ROInum", "ch"], sort=False)
    for _, idx in grouped.groups.items():
        each = out.loc[idx]
        pre = each.loc[each["time_sec_norm"] < 0, "intensity_div_nAve"]
        f0 = float(pre.mean()) if len(pre) else np.nan
        if not np.isfinite(f0) or f0 == 0:
            continue
        out.loc[idx, "F0"] = f0
        out.loc[idx, "dFF0"] = each["intensity_div_nAve"] / f0 - 1.0
        unc_nth = each.loc[each["first_uncaging"] == 1, "NthFrame"]
        if len(unc_nth):
            out.loc[idx, "frame_from_unc"] = each["NthFrame"] - int(unc_nth.iloc[0])
    return out


def decorate_ax(ax, ylabel: str, xlabel: str, uncaging_end_x: float) -> None:
    """Uncaging marker, y=0 baseline, hide top/right spines."""
    ax.axhline(0.0, color="gray", ls="--", lw=0.8, zorder=0)
    ax.axvline(0.0, color="k", lw=0.8, zorder=1)
    ylim = ax.get_ylim()
    y_bar = ylim[0] + 0.92 * (ylim[1] - ylim[0])
    ax.plot([0.0, uncaging_end_x], [y_bar, y_bar], "k-", lw=1.4, zorder=4)
    ax.text(0.0, y_bar, "  Uncaging", ha="left", va="bottom", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def mean_sem_by_aligned_frame(plot_df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Mean +/- SEM across spines, matched by frame_from_unc (not raw clock time)."""
    g = (
        plot_df.dropna(subset=["dFF0", "frame_from_unc"])
        .groupby("frame_from_unc", as_index=False)
        .agg(
            x=(x_col, "mean"),
            dFF0_mean=("dFF0", "mean"),
            dFF0_sem=("dFF0", lambda s: float(s.std(ddof=1) / np.sqrt(s.count())) if s.count() > 1 else 0.0),
            n=("dFF0", "count"),
        )
        .sort_values("x")
    )
    return g


def plot_channel_panel(ax, plot_df: pd.DataFrame, x_col: str, show_mean: bool = True) -> None:
    """Individual traces plus optional mean +/- SEM aligned by uncaging frame."""
    for label, g in plot_df.groupby("spine_label"):
        gg = g.sort_values(x_col)
        ax.plot(
            gg[x_col].to_numpy(dtype=float),
            gg["dFF0"].to_numpy(dtype=float),
            lw=0.9,
            alpha=0.7,
            label=label,
        )
    if show_mean:
        agg = mean_sem_by_aligned_frame(plot_df, x_col)
        ax.errorbar(
            agg["x"],
            agg["dFF0_mean"],
            yerr=agg["dFF0_sem"],
            fmt="-",
            color="k",
            ecolor="k",
            elinewidth=1.0,
            capsize=1.5,
            linewidth=1.8,
            zorder=3,
        )


def save_fig(fig, name: str) -> str:
    """Save PNG next to the TimeCourse CSVs."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# %% load and normalize
all_df = pd.DataFrame()
for csv_path in CSV_PATHS:
    print(csv_path)
    resultdf = csv_to_df(
        csv_path,
        ch_list=list(CHANNELS),
        prefix_list=[
            "meanIntensity-ROI",
            "meanIntensity_bg-ROI",
            "sumIntensity-ROI",
            "sumIntensity_bg-ROI",
            "nPixels-ROI",
        ],
    )
    if resultdf is None or len(resultdf) < 2:
        print("  skipped: failed to import")
        continue
    resultdf = detect_uncaging(resultdf, time_threshold=5)
    resultdf = arrange_for_multipos3(resultdf, time_min_range=[-30, 80])
    resultdf["spine_label"] = spine_label(csv_path)
    all_df = pd.concat([all_df, resultdf], ignore_index=True)

all_df = assign_nave_and_dff0(all_df, INTENSITY_COL)

print("\nPer-file Uncaging / nAve summary")
for (label, ch), g in all_df.groupby(["spine_label", "ch"]):
    n_pre = int((g["time_sec_norm"] < 0).sum())
    n_unc = int((g["during_uncaging"] == 1).sum())
    n_post = int(((g["time_sec_norm"] >= 0) & (g["during_uncaging"] == 0)).sum())
    t0 = float(g.loc[g["first_uncaging"] == 1, "time_sec"].iloc[0])
    f0 = float(g["F0"].iloc[0])
    print(
        f"  {label} ch{int(ch)}: t0={t0:.2f}s  n_pre={n_pre} n_unc={n_unc} n_post={n_post}  "
        f"F0={f0:.4g}  nAve={sorted(g['nAveFrame'].unique().tolist())}"
    )

processed_csv = os.path.join(SAVE_DIR, "pos1_highmag_timecourse_aligned_dFF0.csv")
all_df.to_csv(processed_csv, index=False)
print(f"Saved: {processed_csv}")

# %% plot: full timecourse (min) and uncaging zoom (s)
unc_end_min = float(
    all_df.loc[all_df["during_uncaging"] == 1, "time_min_norm"].max()
)
unc_end_sec = float(
    all_df.loc[all_df["during_uncaging"] == 1, "time_sec_norm"].max()
)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=False)
for ax, ch in zip(axes, CHANNELS):
    sub = all_df[(all_df["ch"] == ch) & all_df["dFF0"].notna()].copy()
    plot_channel_panel(ax, sub, "time_min_norm")
    decorate_ax(ax, r"$\Delta$F/F$_0$", "Time (min)", unc_end_min)
    ax.set_title(f"Ch{ch}  n={sub['spine_label'].nunique()}")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.suptitle("pos1 highmag TimeCourse  (nAve 8 during Uncaging, 3 otherwise)", fontsize=10, y=1.03)
fig.tight_layout()
save_fig(fig, "pos1_highmag_timecourse_dFF0_full_min.png")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=False)
for ax, ch in zip(axes, CHANNELS):
    sub = all_df[(all_df["ch"] == ch) & all_df["dFF0"].notna()].copy()
    zoom = sub[(sub["time_sec_norm"] >= -40) & (sub["time_sec_norm"] <= 90)]
    plot_channel_panel(ax, zoom, "time_sec_norm")
    decorate_ax(ax, r"$\Delta$F/F$_0$", "Time from Uncaging (s)", unc_end_sec)
    ax.set_xlim(-40, 90)
    ax.set_title(f"Ch{ch}  n={sub['spine_label'].nunique()}")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.suptitle("Uncaging window  (nAve-corrected, F/F0 - 1)", fontsize=10, y=1.03)
fig.tight_layout()
save_fig(fig, "pos1_highmag_timecourse_dFF0_uncaging_zoom_sec.png")

# Individual spines, one row per file, Ch1/Ch2
labels = sorted(all_df["spine_label"].unique(), key=lambda s: int(s.split("_")[-1]))
fig, axes = plt.subplots(len(labels), 2, figsize=(9.5, 2.15 * len(labels)), sharex=True)
if len(labels) == 1:
    axes = np.array([axes])
for row_i, label in enumerate(labels):
    for col_i, ch in enumerate(CHANNELS):
        ax = axes[row_i, col_i]
        sub = all_df[
            (all_df["spine_label"] == label)
            & (all_df["ch"] == ch)
            & all_df["dFF0"].notna()
        ].sort_values("time_min_norm")
        ax.plot(sub["time_min_norm"], sub["dFF0"], color="C0" if ch == 1 else "C3", lw=1.1)
        decorate_ax(ax, r"$\Delta$F/F$_0$" if col_i == 0 else "", "Time (min)" if row_i == len(labels) - 1 else "", unc_end_min)
        if row_i == 0:
            ax.set_title(f"Ch{ch}")
        if col_i == 0:
            ax.set_ylabel(f"{label}\n" + r"$\Delta$F/F$_0$")
fig.tight_layout()
save_fig(fig, "pos1_highmag_timecourse_dFF0_each_spine.png")

# %% swarmplot: Ch2 dFF0, one point per spine in the 25-35 min window
def ltp_point_per_spine(df: pd.DataFrame, ch: int = 2) -> pd.DataFrame:
    """One Ch2 dFF0 value per spine from the 25-35 min post window.

    If a spine has more than one frame in the window, use the mean
    (same as LTP summary in 20260909_AP5vsCont.py).
    """
    t0, t1 = LTP_WINDOW_MIN
    rows = []
    ch_df = df[(df["ch"] == ch) & df["dFF0"].notna()].copy()
    for label, g in ch_df.groupby("spine_label"):
        win = g[(g["time_min_norm"] >= t0) & (g["time_min_norm"] <= t1)].sort_values(
            "time_min_norm"
        )
        if win.empty:
            print(f"  {label} ch{ch}: no point in {t0}-{t1} min")
            continue
        times = win["time_min_norm"].to_numpy(dtype=float)
        vals = win["dFF0"].to_numpy(dtype=float)
        print(
            f"  {label} ch{ch}: n={len(win)}  t={np.round(times, 2).tolist()} min  "
            f"dFF0={np.round(vals, 4).tolist()}  used={float(np.mean(vals)):.4f}"
        )
        rows.append(
            {
                "spine_label": label,
                "ch": ch,
                "n_in_window": int(len(win)),
                "time_min_norm": float(np.mean(times)),
                "dFF0": float(np.mean(vals)),
                "group": f"Ch{ch} {int(t0)}-{int(t1)} min",
            }
        )
    return pd.DataFrame(rows)


print(f"\nCh2 LTP window {LTP_WINDOW_MIN[0]:g}-{LTP_WINDOW_MIN[1]:g} min")
ltp_df = ltp_point_per_spine(all_df, ch=2)
ltp_csv = os.path.join(SAVE_DIR, "pos1_highmag_ch2_dFF0_25_35min.csv")
ltp_df.to_csv(ltp_csv, index=False)
print(f"Saved: {ltp_csv}")

fig, ax = plt.subplots(figsize=(2.2, 3.2))
p = sns.swarmplot(
    y="dFF0",
    data=ltp_df,
    color="#C44E52",
    size=7,
    ax=ax,
)
sns.boxplot(
    showmeans=True,
    meanline=True,
    meanprops={"color": "r", "ls": "-", "lw": 1},
    medianprops={"visible": False},
    whiskerprops={"visible": False},
    zorder=10,
    y="dFF0",
    data=ltp_df,
    showfliers=False,
    showbox=False,
    showcaps=False,
    ax=p,
)
mean = float(ltp_df["dFF0"].mean())
std = float(ltp_df["dFF0"].std(ddof=1))
ax.text(0.2, mean, f"{mean:.2f} ± {std:.2f}", ha="left", va="bottom", fontsize=8)
ymin, ymax = float(ltp_df["dFF0"].min()), float(ltp_df["dFF0"].max())
pad = (ymax - ymin) * 0.15 if ymax > ymin else 0.1
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_ylabel(r"Ch2 $\Delta$F/F$_0$ (25–35 min)")
ax.set_title(f"pos1 highmag  n={len(ltp_df)}")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([])
fig.tight_layout()
save_fig(fig, "pos1_highmag_ch2_dFF0_25_35min_swarmplot.png")

print("done")
