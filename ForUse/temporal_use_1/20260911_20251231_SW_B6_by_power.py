# %%
"""Re-analyze 20251231 LTP_point_df.pkl grouped by strain (SW vs B6) and uncaging power.

The original analysis is Archive/20251231LTP_vs_laserpow.py.
This script uses the same pickle inputs, but assigns strain from the imaging
position with exact matching (not substring contains, which mixed 1_ and 11_).

Lab notebook 20251231: Pos1-5 = Swiss Webster (SW), Pos6-12 = B6.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import ttest_ind

controlFLIMage_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(controlFLIMage_DIR)
from custom_plot import plt
from FLIMageFileReader2 import FileReader

# %% paths and grouping
combined_df_pkl_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20251231\auto1\combined_df.pkl"
LTP_point_df_pkl_path = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20251231\auto1\LTP_point_df.pkl"

# Daily report 20251231: Pos1-5 SW, Pos6-12 B6.
# Use exact position numbers. Do not use "1_" substring matching (that also hits 11_).
# If SW/B6 look swapped in the plots, set this to True.
SWAP_SW_AND_B6 = False
STRAIN_BY_POSITION = {
    1: "SW",
    2: "SW",
    3: "SW",
    4: "SW",
    5: "SW",
    6: "B6",
    7: "B6",
    8: "B6",
    9: "B6",
    10: "B6",
    11: "B6",
    12: "B6",
}
if SWAP_SW_AND_B6:
    STRAIN_BY_POSITION = {pos: ("B6" if strain == "SW" else "SW") for pos, strain in STRAIN_BY_POSITION.items()}
STRAIN_ORDER = ["SW", "B6"]

# Same mapping as Archive/20251231LTP_vs_laserpow.py (percent -> mW at sample).
uncpow_dict = {
    29: 2.8,
    42: 4.0,
}

reject_threshold_too_large = 4.0
reject_threshold_too_small = -2.0

ylabel_dict = {
    "norm_intensity": r"Normalized $\Delta$volume (a.u.)",
    "GCaMP_Spine_F_F0": r"GCaMP Spine F/F0",
    "GCaMP_DendriticShaft_F_F0": r"GCaMP Dendritic Shaft F/F0",
}

# %% helpers
def extract_position_prefix(label: Any) -> str:
    """Return the folder prefix before `_highmag`, e.g. `10_pos1`."""
    base = os.path.basename(str(label))
    if "_highmag" in base:
        return base[: base.find("_highmag")].rstrip("_")
    return base.rstrip("_")


def extract_position(label: Any) -> str:
    """Return the dish/imaging position number as a string, e.g. `10`.

    20251231 labels look like `10_pos1_highmag_...`, not `10_highmag_...`.
    Strain mapping uses only the leading integer.
    """
    prefix = extract_position_prefix(label)
    match = re.match(r"^(\d+)", prefix)
    if match:
        return match.group(1)
    return prefix


def position_to_int(position: Any) -> int | float:
    match = re.match(r"^(\d+)", str(position).strip())
    if match:
        return int(match.group(1))
    return np.nan


def to_float_scalar(value: Any) -> float:
    if isinstance(value, pd.Series):
        if value.empty:
            return np.nan
        return to_float_scalar(value.iloc[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def assign_strain(position: Any) -> str:
    pos_int = position_to_int(position)
    if pd.isna(pos_int):
        return "unknown"
    return STRAIN_BY_POSITION.get(int(pos_int), "unknown")


def p_to_stars(p: float) -> str:
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_significance_bracket(ax, x0: float, x1: float, y: float, h: float, text: str) -> None:
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=0.9, c="black", clip_on=False)
    ax.text((x0 + x1) / 2.0, y + h * 0.12, text, ha="center", va="bottom", fontsize=9, color="black")


def hide_spines(ax=None) -> None:
    ax = ax if ax is not None else plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def add_mean_line(ax, x: str, y: str, data: pd.DataFrame, order=None, hue=None, hue_order=None) -> None:
    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={"color": "r", "ls": "-", "lw": 1},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=10,
        x=x,
        y=y,
        data=data,
        order=order,
        hue=hue,
        hue_order=hue_order,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax,
        dodge=hue is not None,
    )


def get_uncaging_power_from_row_df(unc_df: pd.DataFrame) -> float:
    """Prefer statedict already stored in combined_df; otherwise read the .flim file."""
    if "statedict" in unc_df.columns:
        statedict = unc_df["statedict"].iloc[0]
        if isinstance(statedict, dict) and "State.Uncaging.Power" in statedict:
            return float(statedict["State.Uncaging.Power"])

    if "file_path" not in unc_df.columns:
        raise ValueError("Cannot find statedict or file_path to read uncaging power.")

    uncaging_path = unc_df["file_path"].values[0]
    uncaging_iminfo = FileReader()
    uncaging_iminfo.read_imageFile(uncaging_path, False)
    return float(uncaging_iminfo.statedict["State.Uncaging.Power"])


def select_uncaging_rows(eachlabel_combined_df: pd.DataFrame) -> pd.DataFrame:
    if "uncaging_frame" in eachlabel_combined_df.columns:
        return eachlabel_combined_df[eachlabel_combined_df["uncaging_frame"] == True]
    if "phase" in eachlabel_combined_df.columns:
        return eachlabel_combined_df[eachlabel_combined_df["phase"] == "unc"]
    raise ValueError("combined_df has neither uncaging_frame nor phase.")


def annotate_mean_text(ax, x_vals, y_vals, dx: float = -0.3) -> None:
    for x_pos, y_val in zip(x_vals, y_vals):
        if pd.isna(y_val):
            continue
        ax.text(x_pos + dx, y_val, f"{y_val:.2f}", ha="center", va="bottom", fontsize=8, color="black")


# %% load
combined_df = pd.read_pickle(combined_df_pkl_path)
LTP_point_df = pd.read_pickle(LTP_point_df_pkl_path)

print("n spines:", LTP_point_df.shape[0])

if "label" not in LTP_point_df.columns:
    raise ValueError("LTP_point_df.pkl does not have a 'label' column. This script expects the original 20251231 pickle.")

# %% annotate uncaging power, position, strain
for each_label in LTP_point_df["label"].unique():
    eachlabel_combined_df = combined_df[combined_df["label"] == each_label]
    if eachlabel_combined_df.empty and "group" in combined_df.columns:
        # Fallback if labels were rebuilt; match by basename position.
        pos = extract_position(each_label)
        eachlabel_combined_df = combined_df[
            combined_df["group"].astype(str).str.fullmatch(rf"{re.escape(pos)}_?", na=False)
        ]

    uncaging_df = select_uncaging_rows(eachlabel_combined_df)
    if len(uncaging_df) == 0:
        print(f"No uncaging frame for {each_label}")
        continue
    if "uncaging_frame" in combined_df.columns and len(uncaging_df) != 1:
        print(each_label, len(uncaging_df))
        raise AssertionError("Expected exactly one uncaging_frame=True row per label.")

    uncaging_pow = get_uncaging_power_from_row_df(uncaging_df)
    position_prefix = extract_position_prefix(each_label)
    position = extract_position(each_label)
    strain = assign_strain(position)

    mask = LTP_point_df["label"] == each_label
    LTP_point_df.loc[mask, "uncaging_pow"] = uncaging_pow
    LTP_point_df.loc[mask, "position_prefix"] = position_prefix
    LTP_point_df.loc[mask, "position"] = position
    LTP_point_df.loc[mask, "strain"] = strain

LTP_point_df["uncaging_pow_mw"] = LTP_point_df["uncaging_pow"].map(uncpow_dict)
unmapped_pow = sorted(LTP_point_df.loc[LTP_point_df["uncaging_pow_mw"].isna(), "uncaging_pow"].dropna().unique())
if len(unmapped_pow) > 0:
    print("WARNING: uncaging percent values not in uncpow_dict:", unmapped_pow)

LTP_point_df["position_int"] = LTP_point_df["position"].map(position_to_int)
LTP_point_df["strain"] = LTP_point_df["strain"].astype(str)
LTP_point_df["strain_power"] = (
    LTP_point_df["strain"].astype(str)
    + " "
    + LTP_point_df["uncaging_pow_mw"].map(lambda x: f"{x:g} mW" if pd.notna(x) else "NA")
)

for col in ["GCaMP_Spine_F_F0", "GCaMP_DendriticShaft_F_F0", "norm_intensity"]:
    if col in LTP_point_df.columns:
        LTP_point_df[col] = LTP_point_df[col].map(to_float_scalar)

save_folder = os.path.join(os.path.dirname(LTP_point_df_pkl_path), "summary_SW_B6")
os.makedirs(save_folder, exist_ok=True)

print("\nUncaging power unique:", LTP_point_df["uncaging_pow"].unique())
print("Uncaging mW unique:", LTP_point_df["uncaging_pow_mw"].unique())
print("Position prefixes unique:", sorted(LTP_point_df["position_prefix"].dropna().unique()))
print("Positions unique:", sorted(LTP_point_df["position"].dropna().unique(), key=lambda x: position_to_int(x)))
print("\nAssignment table (all rows):")
print(
    LTP_point_df.groupby(["strain", "position", "uncaging_pow_mw"], observed=True)
    .size()
    .reset_index(name="n")
    .to_string(index=False)
)

unknown_n = int((LTP_point_df["strain"] == "unknown").sum())
if unknown_n:
    print(f"WARNING: {unknown_n} rows have strain=unknown. Check STRAIN_BY_POSITION.")

# %% filter
LTP_point_df_cut_extreme = LTP_point_df[
    (LTP_point_df["norm_intensity"] < reject_threshold_too_large)
    & (LTP_point_df["norm_intensity"] > reject_threshold_too_small)
    & (LTP_point_df["strain"].isin(STRAIN_ORDER))
    & (LTP_point_df["uncaging_pow_mw"].notna())
].copy()

LTP_point_df_cut_extreme["strain"] = pd.Categorical(
    LTP_point_df_cut_extreme["strain"].astype(str),
    categories=STRAIN_ORDER,
    ordered=True,
)
power_order = sorted(LTP_point_df_cut_extreme["uncaging_pow_mw"].dropna().unique())
strain_power_order = [f"{strain} {pow_mw:g} mW" for pow_mw in power_order for strain in STRAIN_ORDER]
LTP_point_df_cut_extreme["strain_power"] = pd.Categorical(
    LTP_point_df_cut_extreme["strain_power"].astype(str),
    categories=[x for x in strain_power_order if x in set(LTP_point_df_cut_extreme["strain_power"].astype(str))],
    ordered=True,
)

print("\nAfter extreme-volume filter:")
print(
    LTP_point_df_cut_extreme.groupby(["strain", "uncaging_pow_mw"], observed=True)
    .size()
    .reset_index(name="n")
    .to_string(index=False)
)

if LTP_point_df_cut_extreme.empty or len(power_order) == 0:
    raise RuntimeError(
        "No rows left for plotting. Check strain assignment "
        "(position should be 1, 6, 10, ... not 10_pos1_)."
    )

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Arial"


def plot_df_for(y_col: str) -> pd.DataFrame:
    df = LTP_point_df_cut_extreme.dropna(subset=[y_col, "strain", "uncaging_pow_mw"]).copy()
    df["strain"] = pd.Categorical(df["strain"].astype(str), categories=STRAIN_ORDER, ordered=True)
    return df


# %% swarmplot: x=power, hue=strain
for y_col, ylab in ylabel_dict.items():
    plot_df = plot_df_for(y_col)
    if plot_df.empty:
        continue

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    sns.swarmplot(
        x="uncaging_pow_mw",
        y=y_col,
        hue="strain",
        data=plot_df,
        order=power_order,
        hue_order=STRAIN_ORDER,
        dodge=True,
        size=5,
        ax=ax,
    )
    add_mean_line(
        ax,
        x="uncaging_pow_mw",
        y=y_col,
        data=plot_df,
        order=power_order,
        hue="strain",
        hue_order=STRAIN_ORDER,
    )
    ax.set_ylabel(ylab)
    ax.set_xlabel("Uncaging Power (mW)")
    hide_spines(ax)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    savepath = os.path.join(save_folder, f"{y_col}_vs_power_hue_strain.png")
    fig.savefig(savepath, dpi=150, bbox_inches="tight", transparent=True)
    print("saved", savepath)
    plt.show()


# %% swarmplot: combined strain x power on one axis
for y_col, ylab in ylabel_dict.items():
    plot_df = plot_df_for(y_col)
    if plot_df.empty:
        continue
    present_order = [x for x in strain_power_order if (plot_df["strain_power"].astype(str) == x).any()]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    p = sns.swarmplot(
        x="strain_power",
        y=y_col,
        data=plot_df,
        order=present_order,
        size=5,
        ax=ax,
    )
    add_mean_line(ax, x="strain_power", y=y_col, data=plot_df, order=present_order)
    means = [plot_df.loc[plot_df["strain_power"].astype(str) == lab, y_col].mean() for lab in present_order]
    annotate_mean_text(p, p.get_xticks(), means)
    ax.set_ylabel(ylab)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    hide_spines(ax)
    fig.tight_layout()
    savepath = os.path.join(save_folder, f"{y_col}_vs_strain_power.png")
    fig.savefig(savepath, dpi=150, bbox_inches="tight", transparent=True)
    print("saved", savepath)
    plt.show()


# %% SW vs B6 at each power, with stats
def summarize_and_test(plot_df: pd.DataFrame, y_col: str, power_mw: float) -> dict[str, Any]:
    row: dict[str, Any] = {"uncaging_pow_mw": power_mw, "y": y_col}
    group_values = {}
    for strain in STRAIN_ORDER:
        vals = plot_df.loc[plot_df["strain"] == strain, y_col].dropna().to_numpy(dtype=float)
        group_values[strain] = vals
        row[f"n_{strain}"] = int(len(vals))
        row[f"mean_{strain}"] = float(np.mean(vals)) if len(vals) else np.nan
        row[f"std_{strain}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        row[f"sem_{strain}"] = float(scipy_stats.sem(vals)) if len(vals) > 1 else np.nan

    v0, v1 = group_values["SW"], group_values["B6"]
    if len(v0) >= 2 and len(v1) >= 2:
        welch = ttest_ind(v0, v1, equal_var=False, nan_policy="omit")
        mw = scipy_stats.mannwhitneyu(v0, v1, alternative="two-sided")
        row["welch_t"] = float(welch.statistic)
        row["welch_p"] = float(welch.pvalue)
        row["mannwhitney_u"] = float(mw.statistic)
        row["mannwhitney_p"] = float(mw.pvalue)
        row["stars_welch"] = p_to_stars(float(welch.pvalue))
    else:
        row["welch_t"] = np.nan
        row["welch_p"] = np.nan
        row["mannwhitney_u"] = np.nan
        row["mannwhitney_p"] = np.nan
        row["stars_welch"] = "n.a."
    return row


stats_rows = []
for y_col, ylab in ylabel_dict.items():
    plot_df_all = plot_df_for(y_col)
    n_powers = len(power_order)
    if n_powers == 0 or plot_df_all.empty:
        print(f"Skip stats plot for {y_col}: no data")
        continue
    fig, axes = plt.subplots(1, n_powers, figsize=(2.8 * n_powers, 3.4), sharey=True)
    if n_powers == 1:
        axes = np.array([axes])

    y_min = float(plot_df_all[y_col].min())
    y_max = float(plot_df_all[y_col].max())
    y_span = max(y_max - y_min, 0.2)

    for ax, power_mw in zip(axes, power_order):
        power_df = plot_df_all[plot_df_all["uncaging_pow_mw"] == power_mw]
        stats_row = summarize_and_test(power_df, y_col, power_mw)
        stats_rows.append(stats_row)

        rng = np.random.default_rng(0)
        means = []
        sems = []
        ns = []
        for i, strain in enumerate(STRAIN_ORDER):
            vals = power_df.loc[power_df["strain"] == strain, y_col].dropna().to_numpy(dtype=float)
            ns.append(len(vals))
            mean = float(np.mean(vals)) if len(vals) else np.nan
            sem = float(scipy_stats.sem(vals)) if len(vals) > 1 else 0.0
            means.append(mean)
            sems.append(sem)
            if len(vals):
                x = np.full(len(vals), i, dtype=float) + rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(x, vals, s=22, c="0.35", edgecolors="black", linewidths=0.4, zorder=2, alpha=0.9)
            ax.errorbar(
                i,
                mean,
                yerr=sem,
                fmt="o",
                color="black",
                ecolor="black",
                elinewidth=1.1,
                capsize=3.5,
                capthick=1.1,
                markersize=5.5,
                zorder=4,
            )
            if pd.notna(mean):
                ax.text(i + 0.18, mean, f"{mean:.2f}", ha="left", va="bottom", fontsize=8)

        if stats_row["stars_welch"] not in ("n.a.",):
            bracket_y = y_max + 0.12 * y_span
            add_significance_bracket(ax, 0, 1, bracket_y, 0.05 * y_span, stats_row["stars_welch"])

        ax.axhline(0.0, color="0.6", lw=0.7, ls="--", zorder=1)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(y_min - 0.12 * y_span, y_max + 0.32 * y_span)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(STRAIN_ORDER, ns)], fontsize=8)
        ax.set_title(f"{power_mw:g} mW", fontsize=10)
        hide_spines(ax)
        if ax is axes[0]:
            ax.set_ylabel(ylab)
        else:
            ax.set_ylabel("")

        if pd.notna(stats_row["welch_p"]):
            ax.text(
                0.02,
                0.98,
                f"Welch p={stats_row['welch_p']:.3g}\nMW p={stats_row['mannwhitney_p']:.3g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="0.2",
            )

    fig.tight_layout()
    savepath = os.path.join(save_folder, f"{y_col}_SW_vs_B6_by_power_stats.png")
    fig.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved", savepath)
    plt.show()

stats_df = pd.DataFrame(stats_rows)
stats_csv = os.path.join(save_folder, "SW_vs_B6_by_power_stats.csv")
stats_df.to_csv(stats_csv, index=False)
print("saved", stats_csv)
print(stats_df.to_string(index=False))


# %% scatter: LTP vs GCaMP, one panel per strain x power
scatter_specs = [
    ("GCaMP_Spine_F_F0", r"GCaMP Spine F/F0", "LTP_vs_GCaMP_Spine"),
    ("GCaMP_DendriticShaft_F_F0", r"GCaMP shaft F/F0", "LTP_vs_GCaMP_Shaft"),
]
for x_col, xlab, tag in scatter_specs:
    n_rows = len(STRAIN_ORDER)
    n_cols = len(power_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.1 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    plot_df = plot_df_for("norm_intensity").dropna(subset=[x_col])
    for row_i, strain in enumerate(STRAIN_ORDER):
        for col_i, power_mw in enumerate(power_order):
            ax = axes[row_i, col_i]
            sub = plot_df[(plot_df["strain"] == strain) & (plot_df["uncaging_pow_mw"] == power_mw)]
            ax.scatter(sub[x_col], sub["norm_intensity"], color="black", s=12)
            ax.set_xlim(0, 20)
            ax.set_ylim(-0.5, 3.7)
            hide_spines(ax)
            if row_i == n_rows - 1:
                ax.set_xlabel(xlab)
            if col_i == 0:
                ax.set_ylabel(r"$\Delta$volume")
            if row_i == 0:
                ax.set_title(f"{power_mw:g} mW")
            if col_i == 0:
                ax.annotate(
                    strain,
                    xy=(0, 0.5),
                    xycoords="axes fraction",
                    xytext=(-40, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                )
    fig.tight_layout()
    savepath = os.path.join(save_folder, f"{tag}_by_strain_power.png")
    fig.savefig(savepath, dpi=150, bbox_inches="tight", transparent=True)
    print("saved", savepath)
    plt.show()


# %% QC: each position within strain, split by power
for y_col, ylab in ylabel_dict.items():
    plot_df = plot_df_for(y_col)
    n_cols = len(power_order)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 3.2), sharey=True)
    if n_cols == 1:
        axes = np.array([axes])
    for ax, power_mw in zip(axes, power_order):
        sub = plot_df[plot_df["uncaging_pow_mw"] == power_mw].copy()
        sub["position"] = sub["position"].astype(str)
        pos_order = sorted(sub["position"].unique(), key=position_to_int)
        sns.swarmplot(
            x="position",
            y=y_col,
            hue="strain",
            data=sub,
            order=pos_order,
            hue_order=STRAIN_ORDER,
            size=5,
            ax=ax,
        )
        add_mean_line(ax, x="position", y=y_col, data=sub, order=pos_order, hue="strain", hue_order=STRAIN_ORDER)
        ax.set_title(f"{power_mw:g} mW")
        ax.set_xlabel("Position")
        hide_spines(ax)
        if ax is axes[0]:
            ax.set_ylabel(ylab)
            ax.legend(frameon=False, title="")
        else:
            ax.set_ylabel("")
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    fig.tight_layout()
    savepath = os.path.join(save_folder, f"{y_col}_vs_position_by_power.png")
    fig.savefig(savepath, dpi=150, bbox_inches="tight", transparent=True)
    print("saved", savepath)
    plt.show()


# %% export
out_csv = os.path.join(save_folder, "LTP_point_df_SW_B6_by_power.csv")
LTP_point_df.to_csv(out_csv, index=False)
print("saved", out_csv)
print("plots were saved to:")
print(save_folder)
# %%
