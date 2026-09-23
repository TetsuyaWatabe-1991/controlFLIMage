# %%
"""Plot 20260914 pos1 TimeCourse CSVs aligned to uncaging.

nAveFrame is 8 during uncaging and 3 otherwise. Intensity is divided by
nAveFrame, then dF/F0 = F / F_pre - 1, with F_pre = mean of pre-uncaging.

Ch1 GCaMP F/F0 during uncaging matches 20260916_LTP.py: F0 is the mean of
uncaging-phase frames before the first pulse, F/F0 = F / F0 (not minus 1),
and the representative value is the first-pulse frame.
"""
from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

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
GCAMP_CH = 1
# n_unc frames -> 0-based first-pulse index (same as 20260916_LTP.py)
UNC_TOTAL_FRAME_FIRST_UNC_DICT = {
    33: 2,
    55: 5,
    80: 8,
    144: 8,
}


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


def first_pulse_index(n_unc: int) -> int | None:
    """0-based first-pulse frame within the uncaging cluster."""
    if n_unc in UNC_TOTAL_FRAME_FIRST_UNC_DICT:
        return UNC_TOTAL_FRAME_FIRST_UNC_DICT[n_unc]
    if (n_unc - 1) in UNC_TOTAL_FRAME_FIRST_UNC_DICT:
        return UNC_TOTAL_FRAME_FIRST_UNC_DICT[n_unc - 1]
    if n_unc > 8:
        return 8
    if n_unc >= 2:
        return 1
    return None


def assign_gcamp_ff0(df: pd.DataFrame, ch: int = GCAMP_CH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ch1 F/F0 during uncaging, matching 20260916_LTP.py transient intensity.

    F0 = mean of uncaging-phase frames with slice < first_pulse.
    F/F0 = intensity / F0. Representative value = F/F0 at first_pulse.
    """
    out = df.copy()
    out["gcamp_ff0"] = np.nan
    out["gcamp_F0"] = np.nan
    out["time_sec_from_pulse"] = np.nan
    out["frame_from_pulse"] = np.nan
    rows = []

    ch_df = out[out["ch"] == ch]
    for label, g in ch_df.groupby("spine_label", sort=False):
        unc = g.loc[g["during_uncaging"] == 1].sort_values("NthFrame")
        n_unc = int(len(unc))
        pulse_i = first_pulse_index(n_unc)
        if pulse_i is None or pulse_i >= n_unc:
            pre = g.loc[g["time_sec_norm"] < 0, "intensity_div_nAve"]
            f0 = float(pre.mean()) if len(pre) else np.nan
            pulse_i = 0 if n_unc else None
            method = "pre_timecourse_F0"
        else:
            f0 = float(unc.iloc[:pulse_i]["intensity_div_nAve"].mean())
            method = f"unc_pre_pulse n={pulse_i}"
        if not np.isfinite(f0) or f0 == 0 or pulse_i is None or n_unc == 0:
            print(f"  {label} ch{ch}: skip GCaMP F/F0  n_unc={n_unc}")
            continue
        ff0 = unc["intensity_div_nAve"] / f0
        out.loc[unc.index, "gcamp_F0"] = f0
        out.loc[unc.index, "gcamp_ff0"] = ff0.to_numpy(dtype=float)
        t_pulse = float(unc.iloc[pulse_i]["time_sec"])
        out.loc[unc.index, "time_sec_from_pulse"] = unc["time_sec"] - t_pulse
        out.loc[unc.index, "frame_from_pulse"] = np.arange(n_unc, dtype=float) - pulse_i
        rep = float(ff0.iloc[pulse_i])
        t_rep = float(unc.iloc[pulse_i]["time_sec_norm"])
        print(
            f"  {label} ch{ch}: n_unc={n_unc}  pulse_i={pulse_i}  method={method}  "
            f"F0={f0:.4g}  F/F0={rep:.3f}  t_norm={t_rep:.2f}s"
        )
        rows.append(
            {
                "spine_label": label,
                "ch": ch,
                "n_unc": n_unc,
                "pulse_i": int(pulse_i),
                "gcamp_F0": f0,
                "gcamp_ff0": rep,
                "time_sec_norm_pulse": t_rep,
                "method": method,
            }
        )
    return out, pd.DataFrame(rows)


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

print("\nCh1 GCaMP F/F0 during Uncaging (F0 = pre-pulse frames in uncaging phase)")
all_df, gcamp_df = assign_gcamp_ff0(all_df, ch=GCAMP_CH)
gcamp_csv = os.path.join(SAVE_DIR, "pos1_highmag_ch1_gcamp_ff0_uncaging.csv")
gcamp_df.to_csv(gcamp_csv, index=False)
print(f"Saved: {gcamp_csv}")

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

# %% GCaMP F/F0 during uncaging and vs spine volume (25-35 min)
print(f"\nCh1 GCaMP F/F0  vs  Ch2 dFF0 {LTP_WINDOW_MIN[0]:g}-{LTP_WINDOW_MIN[1]:g} min")
if gcamp_df.empty or ltp_df.empty:
    print("  no paired GCaMP F/F0 and Ch2 LTP points")
    scatter_df = pd.DataFrame()
else:
    scatter_df = gcamp_df.merge(
        ltp_df[["spine_label", "dFF0", "n_in_window", "time_min_norm"]].rename(
            columns={
                "dFF0": "ch2_dFF0_ltp",
                "n_in_window": "ch2_n_in_window",
                "time_min_norm": "ch2_time_min_norm",
            }
        ),
        on="spine_label",
        how="inner",
    )
    scatter_csv = os.path.join(SAVE_DIR, "pos1_highmag_gcamp_ff0_vs_ch2_dFF0_25_35min.csv")
    scatter_df.to_csv(scatter_csv, index=False)
    print(f"Saved: {scatter_csv}")
    for _, row in scatter_df.iterrows():
        print(
            f"  {row['spine_label']}: GCaMP F/F0={row['gcamp_ff0']:.3f}  "
            f"Ch2 dFF0={row['ch2_dFF0_ltp']:.3f}"
        )

# GCaMP F/F0 zoom during uncaging (t=0 = first pulse)
gcamp_zoom = all_df[
    (all_df["ch"] == GCAMP_CH) & all_df["gcamp_ff0"].notna()
].copy()
if not gcamp_zoom.empty:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for label, g in gcamp_zoom.groupby("spine_label"):
        gg = g.sort_values("time_sec_from_pulse")
        ax.plot(
            gg["time_sec_from_pulse"].to_numpy(dtype=float),
            gg["gcamp_ff0"].to_numpy(dtype=float),
            lw=0.9,
            alpha=0.7,
            label=label,
        )
    agg = (
        gcamp_zoom.dropna(subset=["gcamp_ff0", "frame_from_pulse"])
        .groupby("frame_from_pulse", as_index=False)
        .agg(
            x=("time_sec_from_pulse", "mean"),
            gcamp_mean=("gcamp_ff0", "mean"),
            gcamp_sem=(
                "gcamp_ff0",
                lambda s: float(s.std(ddof=1) / np.sqrt(s.count())) if s.count() > 1 else 0.0,
            ),
        )
        .sort_values("x")
    )
    ax.errorbar(
        agg["x"],
        agg["gcamp_mean"],
        yerr=agg["gcamp_sem"],
        fmt="-",
        color="k",
        ecolor="k",
        elinewidth=1.0,
        capsize=1.5,
        linewidth=1.8,
        zorder=3,
    )
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, zorder=0)
    ax.axvline(0.0, color="k", lw=0.8, zorder=1)
    ax.set_ylabel(r"GCaMP F/F$_0$")
    ax.set_xlabel("Time from first pulse (s)")
    ax.set_title(f"Ch1 Uncaging  n={gcamp_zoom['spine_label'].nunique()}")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "pos1_highmag_ch1_gcamp_ff0_uncaging_zoom_sec.png")

if not gcamp_df.empty:
    fig, ax = plt.subplots(figsize=(2.2, 3.2))
    p = sns.swarmplot(
        y="gcamp_ff0",
        data=gcamp_df,
        color="#4C72B0",
        size=7,
        ax=ax,
    )
    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={"color": "k", "ls": "-", "lw": 1},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=10,
        y="gcamp_ff0",
        data=gcamp_df,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=p,
    )
    mean = float(gcamp_df["gcamp_ff0"].mean())
    std = float(gcamp_df["gcamp_ff0"].std(ddof=1)) if len(gcamp_df) > 1 else 0.0
    ax.text(0.2, mean, f"{mean:.2f} ± {std:.2f}", ha="left", va="bottom", fontsize=8)
    ymin, ymax = float(gcamp_df["gcamp_ff0"].min()), float(gcamp_df["gcamp_ff0"].max())
    pad = (ymax - ymin) * 0.15 if ymax > ymin else 0.1
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel(r"GCaMP Spine F/F$_0$")
    ax.set_title(f"pos1 highmag  n={len(gcamp_df)}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    fig.tight_layout()
    save_fig(fig, "pos1_highmag_ch1_gcamp_ff0_swarmplot.png")

if not scatter_df.empty:
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    sns.scatterplot(
        x="gcamp_ff0",
        y="ch2_dFF0_ltp",
        data=scatter_df,
        color="k",
        s=36,
        ax=ax,
    )
    for _, row in scatter_df.iterrows():
        ax.annotate(
            str(row["spine_label"]).replace("highmag_", ""),
            (row["gcamp_ff0"], row["ch2_dFF0_ltp"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7,
        )
    x = scatter_df["gcamp_ff0"].to_numpy(dtype=float)
    y = scatter_df["ch2_dFF0_ltp"].to_numpy(dtype=float)
    if len(scatter_df) >= 3 and np.nanstd(x) > 0:
        lin = scipy_stats.linregress(x, y)
        x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
        ax.plot(x_fit, lin.intercept + lin.slope * x_fit, color="gray", lw=1.0, zorder=0)
        ax.text(
            0.05,
            0.95,
            f"r={lin.rvalue:.2f}  p={lin.pvalue:.3g}  n={len(scatter_df)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    xpad = (x.max() - x.min()) * 0.12 if x.max() > x.min() else 0.5
    ypad = (y.max() - y.min()) * 0.12 if y.max() > y.min() else 0.1
    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)
    ax.axhline(0.0, color="gray", ls="--", lw=0.6, zorder=0)
    ax.set_xlabel(r"GCaMP Spine F/F$_0$")
    ax.set_ylabel(r"$\Delta$spine volume (a.u.) [25–35 min]")
    ax.set_title(f"pos1 highmag  n={len(scatter_df)}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "pos1_highmag_ch2_dFF0_25_35min_vs_gcamp_ff0_scatter.png")

print("done")
