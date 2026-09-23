# -*- coding: utf-8 -*-
"""Shared grouped LTP analysis (spine volume / GCaMP) used by experiment scripts.

Experiment scripts should only set paths and a filename-prefix -> condition map,
then call ``run_ltp_group_analysis(cfg)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import glob
import json
import os
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import ttest_ind
import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]

_FORUSE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTROLFLIMAGE_DIR = os.path.dirname(_FORUSE_DIR)
if _CONTROLFLIMAGE_DIR not in sys.path:
    sys.path.insert(0, _CONTROLFLIMAGE_DIR)
if _FORUSE_DIR not in sys.path:
    sys.path.insert(0, _FORUSE_DIR)

from custom_plot import plt  # noqa: E402
from flim_summarize_func import reshape_axes_to_2d  # noqa: E402


UNKNOWN_CONDITION = "unknown"


@dataclass
class LTPGroupAnalysisConfig:
    """Experiment-specific settings; analysis/plotting lives in this module."""

    df_save_path: str
    out_csv_path: str
    acquisition_start_datetime_str: str
    condition_prefix_map: Mapping[str, str]
    condition_order: Sequence[str]
    condition_styles: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    ch_1or2: int = 2
    ltp_window_min: Sequence[float] = (25.0, 35.0)
    pre_instability_max_min_ratio_threshold: float = 2.0
    # 25-35 min Δspine volume (F/F0 - 1). None = no cutoff on that side.
    spine_volume_delta_ff0_min: float | None = None
    spine_volume_delta_ff0_max: float | None = None
    aligned_time_xlim_max_sec: float = 2100.0
    binned_time_xlim_min: float = -10.0
    binned_time_xlim_max: float = 35.0
    bin_time_threshold_sec: float = 5.0
    bin_percent_median: float = 0.99
    powermeter_folder: str = (
        r"//RY-LAB-WS04/Users/yasudalab/Documents/Tetsuya_Imaging/powermeter"
    )
    from_thorlab_to_coherent_factor: float = 1.0 / 3.0
    roi_name_list: Sequence[str] = ("Spine", "DendriticShaft")
    unc_total_frame_first_unc_dict: Mapping[int, int] = field(
        default_factory=lambda: {33: 2, 55: 5, 80: 8, 144: 8}
    )
    mean_sem_indiv_color: str = "0.72"
    mean_sem_mean_color: str = "r"


def condition_from_label(
    label,
    prefix_map: Mapping[str, str],
    unknown: str = UNKNOWN_CONDITION,
) -> str:
    """Assign a condition from a path or group name using filename prefixes.

    Matching is case-insensitive. Longer prefixes are tried first so that
    overlapping names do not collide.
    """
    text = str(label).replace("\\", "/")
    candidates = [part for part in text.split("/") if part]
    rules = sorted(prefix_map.items(), key=lambda kv: len(kv[0]), reverse=True)
    for name in reversed(candidates):
        lower = name.lower()
        for prefix, condition in rules:
            if lower.startswith(str(prefix).lower()):
                return str(condition)
    return unknown


def assign_condition(
    file_path=None,
    group=None,
    prefix_map: Mapping[str, str] | None = None,
    unknown: str = UNKNOWN_CONDITION,
) -> str:
    """Resolve condition from file_path first, then group label."""
    if not prefix_map:
        return unknown
    for value in (file_path, group):
        if value is None:
            continue
        cond = condition_from_label(value, prefix_map, unknown=unknown)
        if cond != unknown:
            return cond
    return unknown


def build_condition_group_specs(
    df: pd.DataFrame,
    condition_order: Sequence[str],
    unknown: str = UNKNOWN_CONDITION,
) -> list[tuple[str, str]]:
    """Return [(label, condition), ...] in the requested order, then extras."""
    present = {str(x) for x in df["condition"].dropna().unique()}
    specs = [(c, c) for c in condition_order if c in present]
    extras = sorted(c for c in present if c not in set(condition_order))
    specs += [(c, c) for c in extras]
    return specs


def clamp_aligned_time_xlim(ax, xmax_sec: float) -> None:
    """Use min(auto xmax, xmax_sec) so outlier times do not dominate the X axis."""
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, min(float(hi), float(xmax_sec)))


def assign_binned_min_everymin(
    df: pd.DataFrame,
    time_col: str = "aligned_time_sec",
    time_threshold: float = 5.0,
    bin_percent_median: float = 0.99,
) -> pd.DataFrame:
    """Floor-divide aligned time by typical pre/post interval.

    Same method as read_flimagecsv.everymin_normalize and
    example_gui_usage_reduce_asking.py (binned_sec / binned_min).
    """
    out = df.copy()
    out["delta_sec"] = np.nan
    for _, each_df in out.groupby("group_set_id"):
        prepost = each_df[each_df["phase"].isin(["pre", "post"])].sort_values(time_col)
        if len(prepost) < 2:
            continue
        t = prepost[time_col].astype(float).to_numpy()
        out.loc[prepost.index[:-1], "delta_sec"] = np.diff(t)

    valid = out["delta_sec"] > time_threshold
    if valid.any():
        bin_sec = float(out.loc[valid, "delta_sec"].median()) * bin_percent_median
        bin_sec = 60.0 * (bin_sec // 60.0)
    else:
        bin_sec = 0.0
    if bin_sec <= 0:
        bin_sec = 60.0
        print("Warning: could not estimate bin_sec from delta_sec; using 60 s")
    else:
        print(f"Time bin width: {bin_sec:g} sec ({bin_sec / 60.0:g} min)")

    out["binned_sec"] = bin_sec * (out[time_col] // bin_sec)
    out["binned_min"] = out["binned_sec"] / 60.0
    return out


def plot_df_for_binned_ltp(plot_df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Drop the t=0 bin (uncaging cluster), matching example_gui_usage."""
    if plot_df.empty or "binned_min" not in plot_df.columns:
        return plot_df.iloc[0:0].copy()
    out = plot_df.dropna(subset=[y_col, "binned_min"])
    return out[out["binned_min"] != 0].copy()


def plot_spine_volume_mean_sem(
    ax,
    plot_df: pd.DataFrame,
    y_col: str,
    *,
    mean_color: str,
    label=None,
    show_individuals: bool = True,
    indiv_color: str = "0.72",
) -> int:
    """Faint individual traces + seaborn mean+/-SEM on binned_min."""
    plot_df = plot_df_for_binned_ltp(plot_df, y_col)
    n = int(plot_df["group_set_id"].nunique()) if not plot_df.empty else 0
    if plot_df.empty:
        return 0
    plot_df = (
        plot_df.groupby(["group_set_id", "binned_min"], as_index=False)[y_col].mean()
    )
    if show_individuals:
        for _, g in plot_df.groupby("group_set_id"):
            gg = g.sort_values("binned_min")
            ax.plot(
                gg["binned_min"].to_numpy(dtype=float),
                gg[y_col].to_numpy(dtype=float),
                color=indiv_color,
                lw=0.6,
                alpha=0.35,
                zorder=1,
            )
    mean_label = label if label is not None else f"Mean +/- SEM (n={n})"
    sns.lineplot(
        x="binned_min",
        y=y_col,
        data=plot_df,
        errorbar="se",
        linewidth=5,
        color=mean_color,
        ax=ax,
        label=mean_label,
        zorder=3,
    )
    return n


def decorate_spine_volume_timecourse_ax(
    ax,
    ylim,
    *,
    ltp_window_min: Sequence[float] = (25.0, 35.0),
    xlim_min: float = -10.0,
    xlim_max: float = 35.0,
) -> None:
    """Uncaging bar, LTP window, y=0. X is binned_min."""
    ax.set_ylim(ylim)
    ax.set_xlim(xlim_min, xlim_max)
    ylim_now = ax.get_ylim()
    y90 = ylim_now[0] + (ylim_now[1] - ylim_now[0]) * 0.9
    unc_end_min = (2.048 * 29) / 60.0
    ax.plot([0, unc_end_min], [y90, y90], "k-", lw=1.2, zorder=4)
    ax.text(0, y90, "uncaging", ha="left", va="bottom", fontsize=8)
    ax.fill_between(
        np.array(ltp_window_min, dtype=float),
        ylim[0],
        ylim[1],
        color="pink",
        alpha=0.3,
        zorder=0,
        lw=0,
    )
    xlim = ax.get_xlim()
    ax.plot(xlim, [0, 0], "--", color="gray", lw=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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


def exclude_extreme_spine_volume(
    summary_df: pd.DataFrame,
    fulltimeseries_df: pd.DataFrame,
    y_col: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Drop spines whose LTP Δspine volume is outside [vmin, vmax]."""
    if y_col not in summary_df.columns:
        raise KeyError(f"Missing spine-volume column: {y_col}")
    if vmin is None and vmax is None:
        print(f"Spine volume extreme filter: disabled ({y_col})")
        return summary_df, fulltimeseries_df, []

    mask = pd.Series(False, index=summary_df.index)
    if vmin is not None:
        mask |= summary_df[y_col] < vmin
    if vmax is not None:
        mask |= summary_df[y_col] > vmax
    extreme_ids = summary_df.loc[mask, "group_set_id"].tolist()
    bounds = f"[{vmin}, {vmax}]"
    print(f"Excluded sets (spine volume {y_col} outside {bounds}): {extreme_ids}")
    if extreme_ids:
        cols = ["group_set_id", y_col]
        for extra in ("condition", "uncaging_power_coherent_mW"):
            if extra in summary_df.columns:
                cols.append(extra)
        print(summary_df.loc[mask, cols].to_string(index=False))
    summary_df = summary_df[~mask].reset_index(drop=True)
    fulltimeseries_df = fulltimeseries_df[
        ~fulltimeseries_df["group_set_id"].isin(extreme_ids)
    ].reset_index(drop=True)
    return summary_df, fulltimeseries_df, extreme_ids


def round_uncaging_power_mw(value, decimals: int = 1) -> float:
    """Round coherent mW for matching across experiments."""
    return round(float(value), decimals)


def select_summary_spines(
    summary_df: pd.DataFrame,
    *,
    y_col: str,
    condition: str | None = None,
    powers_mw: Sequence[float] | None = None,
    power_decimals: int = 1,
) -> pd.DataFrame:
    """Return one row per spine, optionally filtered by condition and power."""
    if y_col not in summary_df.columns:
        raise KeyError(f"Missing spine-volume column: {y_col}")
    out = summary_df.copy()
    if condition is not None:
        out = out[out["condition"].astype(str) == str(condition)]
    if powers_mw is not None:
        wanted = {round_uncaging_power_mw(p, power_decimals) for p in powers_mw}
        rounded = out["uncaging_power_coherent_mW"].astype(float).map(
            lambda x: round_uncaging_power_mw(x, power_decimals)
        )
        out = out[rounded.isin(wanted)]
    return out.dropna(subset=[y_col]).reset_index(drop=True)


def combine_labeled_spine_tables(
    labeled_tables: Sequence[tuple[str, pd.DataFrame]],
    y_col: str,
) -> pd.DataFrame:
    """Stack per-experiment tables and assign a display group label."""
    frames = []
    for label, df in labeled_tables:
        if df.empty:
            raise ValueError(f"No spines for group {label!r}")
        tmp = df.copy()
        tmp["group_label"] = label
        tmp["delta_spine_volume"] = tmp[y_col]
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


def format_group_tick_label(label: str, n: int | None = None, *, show_n: bool = True) -> str:
    """Wrap a long group name onto two lines, optionally appending n."""
    text = str(label).replace("\\n", "\n")
    if "\n" not in text and " " in text:
        first, rest = text.split(" ", 1)
        text = f"{first}\n{rest}"
    if show_n and n is not None:
        return f"{text}\n(n={int(n)})"
    return text


def grouped_spine_volume_ylim(plot_df: pd.DataFrame, y_col: str) -> tuple[float, float]:
    """Shared y-limits with padding, always including y=0."""
    vals = plot_df[y_col].astype(float).dropna()
    y_min = float(vals.min())
    y_max = float(vals.max())
    span = max(y_max - y_min, 0.2)
    lo = min(y_min - 0.12 * span, 0.0)
    hi = y_max + 0.18 * span
    return lo, hi


def format_mean_annotation(mean: float, std: float | None = None) -> str:
    """Mean, or mean ± SD, for on-figure labels."""
    if std is None:
        return f"{float(mean):.2f}"
    return f"{float(mean):.2f} ± {float(std):.2f}"


def plot_grouped_spine_volume_swarm(
    plot_df: pd.DataFrame,
    *,
    y_col: str,
    group_order: Sequence[str],
    ax,
    ylim: tuple[float, float],
    rng_seed: int = 0,
    point_size: float = 5.5,
    tick_fontsize: float = 8.0,
    ylabel_fontsize: float = 10.0,
    annotate_fontsize: float = 7.0,
    annotate_y: float = 1.8,
    show_n: bool = False,
) -> pd.DataFrame:
    """Seaborn swarmplot + red mean line. Mean±SD sits at a fixed y."""
    del rng_seed  # kept for call-site compatibility; swarm layout is deterministic
    present = [g for g in group_order if (plot_df["group_label"] == g).any()]
    ordered = plot_df[plot_df["group_label"].isin(present)].copy()
    ordered["group_label"] = pd.Categorical(
        ordered["group_label"], categories=present, ordered=True
    )
    sns.swarmplot(
        x="group_label",
        y=y_col,
        data=ordered,
        order=present,
        color="#1f77b4",
        size=point_size,
        ax=ax,
    )
    sns.boxplot(
        x="group_label",
        y=y_col,
        data=ordered,
        order=present,
        showmeans=True,
        meanline=True,
        meanprops={"color": "r", "ls": "-", "lw": 1.2},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=10,
        showfliers=False,
        showbox=False,
        showcaps=False,
        width=0.55,
        ax=ax,
    )
    stats_rows = []
    for i, g in enumerate(present):
        vals = ordered.loc[ordered["group_label"] == g, y_col].dropna().to_numpy(dtype=float)
        n = int(len(vals))
        mean = float(np.mean(vals)) if n else np.nan
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = float(scipy_stats.sem(vals)) if n > 1 else 0.0
        stats_rows.append(
            {"group_label": g, "n": n, "mean": mean, "std": std, "sem": sem}
        )
        if n:
            ax.text(
                i,
                annotate_y,
                format_mean_annotation(mean, std),
                ha="center",
                va="center",
                fontsize=annotate_fontsize,
                color="black",
                clip_on=False,
            )
    ax.set_xlim(-0.55, len(present) - 0.45)
    ax.set_ylim(ylim)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(
        [
            format_group_tick_label(row["group_label"], row["n"], show_n=show_n)
            for row in stats_rows
        ],
        fontsize=tick_fontsize,
    )
    ax.set_ylabel(r"$\Delta$spine volume (a.u.)", fontsize=ylabel_fontsize)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=3)
    return pd.DataFrame(stats_rows)


MM_PER_INCH = 25.4
NATURE_FULL_COLUMN_MM = 183.0
NATURE_THREE_QUARTER_COLUMN_MM = 137.0


def mm_to_inch(mm: float) -> float:
    """Convert millimeters to inches for matplotlib figure sizes."""
    return float(mm) / MM_PER_INCH


def save_grouped_spine_volume_swarm_figure(
    plot_df: pd.DataFrame,
    *,
    y_col: str,
    group_order: Sequence[str],
    save_path: str,
    width_mm: float,
    height_mm: float,
    ylim: tuple[float, float],
    dpi: int = 600,
    point_size: float = 5.5,
    tick_fontsize: float = 8.0,
    ylabel_fontsize: float = 10.0,
    annotate_fontsize: float = 7.0,
    annotate_y: float = 1.8,
    show_n: bool = False,
) -> pd.DataFrame:
    """Save PNG and PDF of the grouped swarm. Returns per-group stats."""
    fig, ax = plt.subplots(
        figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)),
        dpi=dpi,
    )
    stats = plot_grouped_spine_volume_swarm(
        plot_df,
        y_col=y_col,
        group_order=group_order,
        ax=ax,
        ylim=ylim,
        point_size=point_size,
        tick_fontsize=tick_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        annotate_fontsize=annotate_fontsize,
        annotate_y=annotate_y,
        show_n=show_n,
    )
    fig.subplots_adjust(bottom=0.18, left=0.16, right=0.98, top=0.97)
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    return stats


def add_significance_bracket(
    ax, x0: float, x1: float, y: float, h: float, text: str
) -> None:
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=0.9, c="black", clip_on=False)
    ax.text(
        (x0 + x1) / 2.0,
        y + h * 0.12,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )


def prepare_ltp_group_tables(cfg: LTPGroupAnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load combined RESPAN tables, group by filename prefix, and filter spines.

    Returns ``summary_df``, ``fulltimeseries_df``, and the summary output folder.
    """
    df_save_path_1 = cfg.df_save_path
    out_csv_path = cfg.out_csv_path
    acquisiton_start_datetime_str = cfg.acquisition_start_datetime_str
    prefix_map = dict(cfg.condition_prefix_map)
    prefix_names = ", ".join(prefix_map.keys())
    CONDITION_PREFIX_ORDER = list(cfg.condition_order)
    CONDITION_MEAN_SEM_STYLE = dict(cfg.condition_styles)
    MEAN_SEM_INDIV_COLOR = cfg.mean_sem_indiv_color
    MEAN_SEM_MEAN_COLOR = cfg.mean_sem_mean_color
    ch_1or2 = cfg.ch_1or2
    LTP_data_point_after_min_between = list(cfg.ltp_window_min)
    pre_instability_max_min_ratio_threshold = cfg.pre_instability_max_min_ratio_threshold
    spine_volume_delta_ff0_min = cfg.spine_volume_delta_ff0_min
    spine_volume_delta_ff0_max = cfg.spine_volume_delta_ff0_max
    ALIGNED_TIME_XLIM_MAX_SEC = cfg.aligned_time_xlim_max_sec
    BINNED_TIME_XLIM_MIN = cfg.binned_time_xlim_min
    BINNED_TIME_XLIM_MAX = cfg.binned_time_xlim_max
    BIN_TIME_THRESHOLD_SEC = cfg.bin_time_threshold_sec
    BIN_PERCENT_MEDIAN = cfg.bin_percent_median
    powermeter_folder = cfg.powermeter_folder
    from_Thorlab_to_coherent_factor = cfg.from_thorlab_to_coherent_factor
    ROI_name_list = list(cfg.roi_name_list)
    unc_total_frame_first_unc_dict = dict(cfg.unc_total_frame_first_unc_dict)

    assert os.path.exists(powermeter_folder), f"powermeter_folder does not exist: {powermeter_folder}"
    assert os.path.exists(df_save_path_1), f"df_save_path does not exist: {df_save_path_1}"
    assert os.path.exists(out_csv_path), f"out_csv_path does not exist: {out_csv_path}"




    combined_df = pd.read_pickle(df_save_path_1)
    fulltimeseries_df = pd.read_csv(out_csv_path)

    # normalize the intensity by dividing the intensity by the number of summed frames

    for each_ROI_name in ROI_name_list:
        for each_ch in ['Ch1', 'Ch2']:
            fulltimeseries_df.loc[:,f"{each_ROI_name}_{each_ch}_intensity_div_by_nAve"] = fulltimeseries_df.loc[:,f"{each_ROI_name}_{each_ch}_intensity"] / fulltimeseries_df.loc[:,"nAveFrame"]


    combined_df["dt"] = pd.to_datetime(combined_df["dt_str"])

    save_folder = os.path.join(os.path.dirname(df_save_path_1), "summary")
    os.makedirs(save_folder, exist_ok=True)



    #%% get uncaging power
    combined_df["uncaging_power"] = np.nan
    fulltimeseries_df["uncaging_power"] = np.nan

    unc_df = combined_df[combined_df['phase'] == 'unc']
    for each_file in unc_df["file_path"].unique():
        statedict = unc_df[unc_df["file_path"] == each_file]["statedict"].values[0]
        uncaging_power = statedict["State.Uncaging.Power"]
        combined_df.loc[combined_df["file_path"] == each_file, "uncaging_power"] = uncaging_power
        fulltimeseries_df.loc[fulltimeseries_df["file_path"] == each_file, "uncaging_power"] = uncaging_power

    print("uncaging power in combined_df:")
    print(combined_df["uncaging_power"].unique())
    print("uncaging power in fulltimeseries_df:")
    print(fulltimeseries_df["uncaging_power"].unique())

    list_of_uncaging_power = list(combined_df["uncaging_power"].unique()[combined_df["uncaging_power"].unique() > 0])

    for each_uncaging_power in list_of_uncaging_power:
        print(each_uncaging_power)

    earliest_acq_time = datetime.datetime.strptime(fulltimeseries_df["acq_time_str"].min(), "%Y-%m-%dT%H:%M:%S.%f")

    powermeter_ini_files = glob.glob(os.path.join(powermeter_folder, "*.json")) +\
                        glob.glob(os.path.join(powermeter_folder, "old/*.json"))

    #get the latest powermeter json file before the earliest acq time
    latest_powermeter_json_file = None

    datetime_only_arr = np.array([int(os.path.basename(each_file).replace(".json", "")) for each_file in powermeter_ini_files])
    latest_powermeter_json_basename = f"{datetime_only_arr[datetime_only_arr < int(earliest_acq_time.strftime("%Y%m%d%H%M"))].max()}.json"
    latest_powermeter_json_file = [each_file for each_file in powermeter_ini_files if os.path.basename(each_file) == latest_powermeter_json_basename]
    assert len(latest_powermeter_json_file) == 1, f"Latest powermeter json file is not found: {latest_powermeter_json_file}"
    latest_powermeter_json_file = latest_powermeter_json_file[0]

    # Build power % -> power mW dict from powermeter calibration JSON (same approach as GCaMP_unc_combined_titration / pyautogui_LaserPower_PM16_121)
    # JSON format: {"Laser1": {percent: mW, ...}, "Laser2": {percent: mW, ...}}; Laser2 = 720 nm uncaging
    with open(latest_powermeter_json_file, "r") as f:
        powermeter_calib = json.load(f)
    x_percent = np.array(list(powermeter_calib["Laser2"].keys())).astype(float)
    y_mW = np.array(list(powermeter_calib["Laser2"].values())).astype(float)
    power_slope, power_intercept = np.polyfit(x_percent, y_mW, 1)
    # Dict: uncaging_power (%) -> mW for each percent used in this experiment
    power_percent_to_mW_dict = {
        int(p): round(float(power_slope * p + power_intercept), 2)
        for p in list_of_uncaging_power
    }

    power_percent_to_mW_dict = {
        int(p): round(float(power_slope * p + power_intercept), 2)
        for p in list_of_uncaging_power
    }

    power_percent_to_coherent_mW_dict = {
        int(p): round(float(power_slope * p + power_intercept) * from_Thorlab_to_coherent_factor, 1)
        for p in list_of_uncaging_power
    }

    for each_file in unc_df["file_path"].unique():
        statedict = unc_df[unc_df["file_path"] == each_file]["statedict"].values[0]
        uncaging_power = statedict["State.Uncaging.Power"]
        fulltimeseries_df.loc[fulltimeseries_df["file_path"] == each_file, "uncaging_power_coherent_mW"] = power_percent_to_coherent_mW_dict[int(uncaging_power)]
        combined_df.loc[combined_df["file_path"] == each_file, "uncaging_power_coherent_mW"] = power_percent_to_coherent_mW_dict[int(uncaging_power)]


    #%% align the time_sec based on the uncaging timing

    fulltimeseries_df.loc[:,"aligned_time_sec"] = -999.99

    summary_df = pd.DataFrame()
    for each_group in fulltimeseries_df["group"].unique():
        for each_set_label in fulltimeseries_df[fulltimeseries_df["group"] == each_group]["set_label"].unique():
            group_set_id = f"{each_group}_{each_set_label}"
            # print(f"Processing {group_set_id}")
            each_df = fulltimeseries_df[(fulltimeseries_df["group"] == each_group) & (fulltimeseries_df["set_label"] == each_set_label)].copy()
            # name the each_df with group_set_id
            fulltimeseries_df.loc[each_df.index, "group_set_id"] = group_set_id
            each_df.loc[:, "group_set_id"] = group_set_id

            # uncaging trigger time is 0 seconds
            length_of_unc_df = len(each_df[each_df["phase"] == "unc"])
            if length_of_unc_df in unc_total_frame_first_unc_dict.keys():
                unc_trigger_time = each_df[each_df["phase"] == "unc"]["elapsed_time_sec"].iloc[unc_total_frame_first_unc_dict[length_of_unc_df] - 1]
                aligned_values = each_df["elapsed_time_sec"].values - unc_trigger_time
            else:
                raise ValueError(f"Length of each df is not in unc_total_frame_first_unc_dict: {length_of_unc_df}")        
            each_df.loc[:, "aligned_time_sec"] = aligned_values
            fulltimeseries_df.loc[each_df.index, "aligned_time_sec"] = aligned_values

            for each_ROI_name in ROI_name_list:
                for each_ch in ['Ch1', 'Ch2']:
                    #normalize the lifetime, by subtracting the mean of the lifetime of frames in pre phase
                    pre_phase_lifetime = each_df[each_df["phase"] == "pre"][f"{each_ROI_name}_{each_ch}_lifetime"].mean()
                    fulltimeseries_df.loc[each_df.index, f"{each_ROI_name}_{each_ch}_lifetime_normalized"] = fulltimeseries_df.loc[each_df.index, f"{each_ROI_name}_{each_ch}_lifetime"] - pre_phase_lifetime
                    #normalize the intensity, by dividing the intensity by the mean of the intensity of frames in pre phase and subtract 1
                    pre_phase_intensity = each_df[each_df["phase"] == "pre"][f"{each_ROI_name}_{each_ch}_intensity_div_by_nAve"].mean()
                    fulltimeseries_df.loc[each_df.index, f"{each_ROI_name}_{each_ch}_intensity_normalized"] = fulltimeseries_df.loc[each_df.index, f"{each_ROI_name}_{each_ch}_intensity_div_by_nAve"] / pre_phase_intensity - 1

            #transient analysis normalization
            each_unc_df = each_df[each_df["phase"] == "unc"]
            first_uncaging_frame = unc_total_frame_first_unc_dict[len(each_unc_df)]
            for each_ROI_name in ROI_name_list:
                for each_ch in ['Ch1', 'Ch2']:
                    for each_signal in ['lifetime', 'intensity']:
                        first_uncaging_frame = unc_total_frame_first_unc_dict[len(each_unc_df)]
                        before_uncaging_during_unc_phase_signal = each_unc_df[each_unc_df["slice"] < first_uncaging_frame][f"{each_ROI_name}_{each_ch}_{each_signal}"].mean()
                        if not before_uncaging_during_unc_phase_signal >0 :
                            transient_normalized_signal = np.nan
                        else:
                            if each_signal == "lifetime":
                                transient_normalized_signal = each_unc_df[f"{each_ROI_name}_{each_ch}_{each_signal}"] - before_uncaging_during_unc_phase_signal

                            elif each_signal == "intensity":
                                transient_normalized_signal = each_unc_df[f"{each_ROI_name}_{each_ch}_{each_signal}"] / before_uncaging_during_unc_phase_signal    
                        fulltimeseries_df.loc[each_unc_df.index, f"transient_{each_ROI_name}_{each_ch}_{each_signal}"] = transient_normalized_signal


            #summary for each group_set_id
            each_summary_dict = {}
            each_summary_dict["group"] = each_group
            each_summary_dict["set_label"] = each_set_label
            each_summary_dict["group_set_id"] = group_set_id
            each_summary_dict["n_unc_frames"] = int(length_of_unc_df)
            fulltimeseries_df.loc[each_df.index, "n_unc_frames"] = int(length_of_unc_df)
            file_path_for_condition = each_df["file_path"].iloc[0] if "file_path" in each_df.columns else None
            each_condition = assign_condition(file_path=file_path_for_condition, group=each_group, prefix_map=prefix_map)
            each_summary_dict["condition"] = each_condition
            fulltimeseries_df.loc[each_df.index, "condition"] = each_condition

            # uncaging_power_coherent_mW = each_df[each_df["phase"] == "unc"]["uncaging_power_coherent_mW"]
            # if len(uncaging_power_coherent_mW) != 1:
            #     raise ValueError(f"Length of uncaging power coherent mW is not 1: {len(uncaging_power_coherent_mW)}")
            # each_summary_dict["uncaging_power_coherent_mW"] = uncaging_power_coherent_mW.iloc[0]

            # transient analysis
            for each_ROI_name in ROI_name_list:
                for each_ch in ['Ch1', 'Ch2']:
                    each_phase = "unc"
                    for each_signal in ['lifetime', 'intensity']:
                        each_transient_signal_df = fulltimeseries_df.loc[each_unc_df.index, f"transient_{each_ROI_name}_{each_ch}_{each_signal}"]
                        first_uncaging_frame = unc_total_frame_first_unc_dict[len(each_transient_signal_df)]
                        representative_transient_signal = each_transient_signal_df.iloc[first_uncaging_frame]
                        each_summary_dict[f"transient_{each_ROI_name}_{each_ch}_{each_signal}"] = representative_transient_signal
                        # if each_transient_signal_df.max() > 0:
                        #     print(each_transient_signal_df)
                        #     assert False

                    each_phase = 'pre'
                    for each_signal in ['lifetime', 'intensity']:
                        each_summary_dict[f"{each_ch}_{each_phase}_{each_signal}"] = each_df[each_df["phase"] == each_phase][f"Spine_{each_ch}_{each_signal}"].mean()
                    each_phase = 'post'
                    for each_signal in ['lifetime', 'intensity']:
                        post_LTP_data_point_df = each_df[(each_df["phase"] == each_phase) 
                                                        & (each_df["aligned_time_sec"] > LTP_data_point_after_min_between[0]*60) 
                                                        & (each_df["aligned_time_sec"] < LTP_data_point_after_min_between[1]*60)
                                                        & (each_df["group_set_id"] == group_set_id)
                                                        ]
                        if len(post_LTP_data_point_df) >0:
                            each_summary_dict[f"{each_ch}_{each_phase}_{each_signal}"] = post_LTP_data_point_df[f"Spine_{each_ch}_{each_signal}"].mean()
                            # print("ok1")
                        else:
                            post_LTP_data_point_df = each_df[(each_df["phase"] == each_phase) & (each_df["aligned_time_sec"] > LTP_data_point_after_min_between[0]*60)]
                            if len(post_LTP_data_point_df) >0:
                                each_summary_dict[f"{each_ch}_{each_phase}_{each_signal}"] = post_LTP_data_point_df[f"Spine_{each_ch}_{each_signal}"].mean()
                                # print("ok2")
                            else:
                                each_summary_dict[f"{each_ch}_{each_phase}_{each_signal}"] = np.nan
                                print(f"No LTP data point found for {group_set_id} {each_ch} {each_phase} {each_signal}")

            each_summary_dict["delta_lifetime_ch1"] = each_summary_dict["Ch1_post_lifetime"] - each_summary_dict["Ch1_pre_lifetime"]
            each_summary_dict["delta_FF0_intensity_ch1"] = each_summary_dict["Ch1_post_intensity"] / each_summary_dict["Ch1_pre_intensity"] - 1
            each_summary_dict["delta_lifetime_ch2"] = each_summary_dict["Ch2_post_lifetime"] - each_summary_dict["Ch2_pre_lifetime"]
            each_summary_dict["delta_FF0_intensity_ch2"] = each_summary_dict["Ch2_post_intensity"] / each_summary_dict["Ch2_pre_intensity"] - 1

            first_post_df = each_df[each_df["phase"] == "post"].sort_values("aligned_time_sec")
            for ch_i, ch_label in ((1, "Ch1"), (2, "Ch2")):
                ch_pre_mean = each_summary_dict[f"{ch_label}_pre_intensity"]
                if len(first_post_df) > 0 and ch_pre_mean > 0:
                    each_summary_dict[f"first_post_FF0_intensity_ch{ch_i}"] = (
                        first_post_df[f"Spine_{ch_label}_intensity"].iloc[0] / ch_pre_mean - 1
                    )
                else:
                    each_summary_dict[f"first_post_FF0_intensity_ch{ch_i}"] = np.nan

            each_summary_dict["acq_time_str"] = each_df["acq_time_str"].unique()[0]

            pre_df_phase = each_df[each_df["phase"] == "pre"]
            for each_ROI_name in ROI_name_list:
                for each_ch in ["Ch1", "Ch2"]:
                    col = f"{each_ROI_name}_{each_ch}_intensity_div_by_nAve"
                    pre_vals = pre_df_phase[col].dropna() if col in pre_df_phase.columns else pd.Series(dtype=float)
                    if len(pre_vals) >= 2 and pre_vals.min() > 0:
                        ratio = pre_vals.max() / pre_vals.min()
                    else:
                        ratio = np.nan
                    each_summary_dict[f"pre_max_min_ratio_{each_ROI_name}_{each_ch}"] = ratio

            summary_df = pd.concat([summary_df, pd.DataFrame([each_summary_dict])], ignore_index=True)

    # %%

    #bin the time
    pre_phase_sec = float(fulltimeseries_df[fulltimeseries_df["phase"] == "pre"]["aligned_time_sec"].mean())
    post_phase_sec = float(fulltimeseries_df[fulltimeseries_df["phase"] == "post"]["aligned_time_sec"].mean())
    pre_index = fulltimeseries_df[fulltimeseries_df["phase"] == "pre"].index
    post_index = fulltimeseries_df[fulltimeseries_df["phase"] == "post"].index
    fulltimeseries_df.loc[pre_index, "binned_time_sec"] = pre_phase_sec
    fulltimeseries_df.loc[post_index, "binned_time_sec"] = post_phase_sec

    #calc bin time during uncaging, and assign uncaging power
    average_bin = 0
    num_bin = 0
    for each_group_set_id in fulltimeseries_df["group_set_id"].unique():
        each_df = fulltimeseries_df[fulltimeseries_df["group_set_id"] == each_group_set_id]
        bin_sec_during_uncaging = float((each_df[each_df["phase"] == "unc"]["aligned_time_sec"].max() - each_df[each_df["phase"] == "unc"]["aligned_time_sec"].min()) / len(each_df[each_df["phase"] == "unc"]))
        average_bin += bin_sec_during_uncaging
        num_bin += 1
    average_bin = average_bin / num_bin

    for each_group_set_id in fulltimeseries_df["group_set_id"].unique():
        each_df = fulltimeseries_df[fulltimeseries_df["group_set_id"] == each_group_set_id]
        unc_df = each_df[each_df["phase"] == "unc"]
        uncaging_power_coherent_mW_list = list(unc_df["uncaging_power_coherent_mW"].unique())
        if len(uncaging_power_coherent_mW_list) != 1:
            raise ValueError(f"Length of uncaging power coherent mW is not 1: {len(uncaging_power_coherent_mW_list)}")
        else:
            uncaging_power_coherent_mW = uncaging_power_coherent_mW_list[0]

        length_of_unc_df = len(unc_df)
        if length_of_unc_df in unc_total_frame_first_unc_dict.keys():
            time_0_nth = unc_total_frame_first_unc_dict[length_of_unc_df] - 1
        else:
            raise ValueError(f"Length of each uncaging df is not in unc_total_frame_first_unc_dict: {length_of_unc_df}")
        time_list = [i*average_bin for i in range(-time_0_nth,len(unc_df)-time_0_nth)]
        fulltimeseries_df.loc[unc_df.sort_values("aligned_time_sec").index, "binned_time_sec"] = time_list

        fulltimeseries_df.loc[each_df.index, "uncaging_power_coherent_mW"] = uncaging_power_coherent_mW
        summary_df.loc[summary_df["group_set_id"] == each_group_set_id, "uncaging_power_coherent_mW"] = uncaging_power_coherent_mW


    ratio_cols = [c for c in summary_df.columns if c.startswith("pre_max_min_ratio_")]
    print("Pre-phase intensity stability (max/min ratio):")
    print(summary_df[["group_set_id"] + ratio_cols].to_string(index=False))
    ratio_cols_to_check = [c for c in ratio_cols if f"Ch{ch_1or2}" in c]
    unstable_mask = summary_df[ratio_cols_to_check].gt(pre_instability_max_min_ratio_threshold).any(axis=1)
    unstable_ids = summary_df.loc[unstable_mask, "group_set_id"].tolist()
    print(f"Excluded sets (pre instability max/min > {pre_instability_max_min_ratio_threshold} on {ratio_cols_to_check}): {unstable_ids}")
    summary_df = summary_df[~unstable_mask].reset_index(drop=True)
    fulltimeseries_df = fulltimeseries_df[~fulltimeseries_df["group_set_id"].isin(unstable_ids)].reset_index(drop=True)
    summary_df, fulltimeseries_df, _extreme_ids = exclude_extreme_spine_volume(
        summary_df,
        fulltimeseries_df,
        y_col=f"delta_FF0_intensity_ch{ch_1or2}",
        vmin=spine_volume_delta_ff0_min,
        vmax=spine_volume_delta_ff0_max,
    )
    fulltimeseries_df = assign_binned_min_everymin(fulltimeseries_df, time_threshold=BIN_TIME_THRESHOLD_SEC, bin_percent_median=BIN_PERCENT_MEDIAN)
    summary_csv = os.path.join(save_folder, "summary_df.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary table: {summary_csv}")
    return summary_df, fulltimeseries_df, save_folder


def run_ltp_group_analysis(cfg: LTPGroupAnalysisConfig) -> str:
    """Prepare tables and save the standard LTP plot suite. Returns save_folder."""
    summary_df, fulltimeseries_df, save_folder = prepare_ltp_group_tables(cfg)
    prefix_map = dict(cfg.condition_prefix_map)
    prefix_names = ", ".join(prefix_map.keys())
    CONDITION_PREFIX_ORDER = list(cfg.condition_order)
    CONDITION_MEAN_SEM_STYLE = dict(cfg.condition_styles)
    MEAN_SEM_INDIV_COLOR = cfg.mean_sem_indiv_color
    MEAN_SEM_MEAN_COLOR = cfg.mean_sem_mean_color
    ch_1or2 = cfg.ch_1or2
    LTP_data_point_after_min_between = list(cfg.ltp_window_min)
    ALIGNED_TIME_XLIM_MAX_SEC = cfg.aligned_time_xlim_max_sec
    BINNED_TIME_XLIM_MIN = cfg.binned_time_xlim_min
    BINNED_TIME_XLIM_MAX = cfg.binned_time_xlim_max
    acquisiton_start_datetime_str = cfg.acquisition_start_datetime_str

    # %% condition groups by filename prefix
    condition_group_specs = build_condition_group_specs(fulltimeseries_df, CONDITION_PREFIX_ORDER)
    print("Plot groups by filename prefix:", condition_group_specs)
    print(fulltimeseries_df.groupby("condition")["group_set_id"].nunique())
    unknown_ids = sorted(
        fulltimeseries_df.loc[fulltimeseries_df["condition"] == "unknown", "group_set_id"].dropna().unique()
    )
    if unknown_ids:
        print(f"Warning: unmatched filename prefix (not {prefix_names}): {unknown_ids}")

    # %% line plot
    #plot each data, with light thin color lines
    swarm_ylim = [summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].min()-0.1, 
                    summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].max()+0.1]
    full_range_ylim = [fulltimeseries_df[f"Spine_Ch{ch_1or2}_intensity_normalized"].min()-0.1, 
                    fulltimeseries_df[f"Spine_Ch{ch_1or2}_intensity_normalized"].max()+0.1]
    plot_info_dict = {
        # "lifetime": {"ylabel": r"$\Delta$lifetime (ns)", "y": "Spine_Ch1_lifetime_normalized", "errorbar": "se"},
        "intensity": {"ylabel": r"$\Delta$spine volume (a.u.)", 
                    "xlabel": "Time (sec)",
                    "x": "aligned_time_sec",
                    "y": f"Spine_Ch{ch_1or2}_intensity_normalized", 
                    "errorbar": "se",
                    # "ylim": [-0.4, 5.9]
                    "ylim": [-0.4, swarm_ylim[1]],
                    "plot_zero_line": True,
                    },
        "intensity_full_range": {"ylabel": r"$\Delta$spine volume (a.u.)", 
                    "xlabel": "Time (sec)",
                    "x": "aligned_time_sec",
                    "y": f"Spine_Ch{ch_1or2}_intensity_normalized", 
                    "errorbar": "se",
                    # "ylim": [-0.4, 5.9]
                    "ylim": full_range_ylim,
                    "plot_zero_line": True,
                    },    
        }

    for each_header_name, each_condition in condition_group_specs:
        eachgroup_df = fulltimeseries_df[fulltimeseries_df["condition"] == each_condition]
        if len(eachgroup_df) == 0:
            continue

        for each_uncaging_power_coherent_mW in fulltimeseries_df["uncaging_power_coherent_mW"].unique():
            each_group_same_unc_pow_df = eachgroup_df[eachgroup_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]

            plot_df = each_group_same_unc_pow_df

            for each_plot_type, each_plot_info in plot_info_dict.items():
                plt.figure(figsize=(5, 3))
                g = sns.lineplot(
                            x = each_plot_info["x"],
                            y = each_plot_info["y"],
                            data = plot_df,
                            hue = "group_set_id",
                            linewidth = 0.5,
                            alpha = 0.5,
                            palette = "tab10",
                            legend = False,
                            marker = "o",
                            markersize = 4,
                            )
                #greek delta lifetime
                plt.ylabel(each_plot_info["ylabel"])
                plt.xlabel(each_plot_info["xlabel"])
                plt.title(each_header_name+ f", uncaging {each_uncaging_power_coherent_mW} mW")
                plt.ylim(each_plot_info["ylim"])
                # #plot mean with SEM

                clamp_aligned_time_xlim(plt.gca(), ALIGNED_TIME_XLIM_MAX_SEC)

                ylim = plt.gca().get_ylim()
                ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 0.9
                plt.plot([0, 2.048*29], [ninty_percent_ylim, ninty_percent_ylim], "k-")
                plt.text(0, ninty_percent_ylim*1.005, "uncaging", ha="left", va="bottom")

                current_xlim = plt.gca().get_xlim()

                if each_plot_info["plot_zero_line"]:
                    plt.plot([current_xlim[0], current_xlim[1]], [0, 0], "--", color="gray", linewidth=0.5)

                plt.fill_between(np.array(LTP_data_point_after_min_between)*60,
                each_plot_info["ylim"][0], each_plot_info["ylim"][1], 
                color="pink", 
                alpha=0.3)

                #delete right and top border
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                savepath = os.path.join(save_folder, f"{each_condition}_{each_plot_type}_{each_uncaging_power_coherent_mW}mW_lineplot_time_series.png")
                plt.savefig(savepath, dpi=150, bbox_inches = "tight")
                plt.show()

    # %% line plot (tiled panels for group_header x uncaging_power_coherent_mW)
    sorted_uncaging_powers = sorted(fulltimeseries_df["uncaging_power_coherent_mW"].dropna().unique())
    valid_group_headers = []
    for each_header_name, each_condition in condition_group_specs:
        if (fulltimeseries_df["condition"] == each_condition).any():
            valid_group_headers.append((each_condition, each_header_name))

    if len(sorted_uncaging_powers) > 0 and len(valid_group_headers) > 0:
        for each_plot_type, each_plot_info in plot_info_dict.items():
            n_rows = len(sorted_uncaging_powers)
            n_cols = len(valid_group_headers)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6 * n_cols, 3 * n_rows),
                sharex=True,
                sharey=True,
            )

            axes = reshape_axes_to_2d(axes, n_rows, n_cols)

            for col_idx, (each_condition, each_header_name) in enumerate(valid_group_headers):
                eachgroup_df = fulltimeseries_df[fulltimeseries_df["condition"] == each_condition]
                for row_idx, each_uncaging_power_coherent_mW in enumerate(sorted_uncaging_powers):
                    ax = axes[row_idx, col_idx]
                    plot_df = eachgroup_df[eachgroup_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
                    if plot_df.empty:
                        ax.axis("off")
                        continue

                    sns.lineplot(
                        x=each_plot_info["x"],
                        y=each_plot_info["y"],
                        data=plot_df,
                        hue="group_set_id",
                        linewidth=0.5,
                        alpha=0.5,
                        palette="tab10",
                        legend=False,
                        marker="o",
                        markersize=4,
                        ax=ax,
                    )

                    ax.set_ylim(each_plot_info["ylim"])

                    if row_idx == n_rows - 1:
                        ax.set_xlabel(each_plot_info["xlabel"])
                    else:
                        ax.set_xlabel("")

                    if col_idx == 0:
                        ax.set_ylabel(each_plot_info["ylabel"])
                    else:
                        ax.set_ylabel("")

                    ylim = ax.get_ylim()
                    ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 0.9
                    ax.plot([0, 2.048 * 29], [ninty_percent_ylim, ninty_percent_ylim], "k-")
                    ax.text(0, ninty_percent_ylim * 1.005, "uncaging", ha="left", va="bottom")

                    ax.fill_between(
                        np.array(LTP_data_point_after_min_between) * 60,
                        each_plot_info["ylim"][0],
                        each_plot_info["ylim"][1],
                        color="pink",
                        alpha=0.3,
                    )

                    if row_idx == 0:
                        ax.set_title(each_header_name)
                    else:
                        ax.set_title("")

                    if col_idx == 0:
                        ax.annotate(
                            f"{each_uncaging_power_coherent_mW} mW",
                            xy=(0, 0.5),
                            xycoords="axes fraction",
                            xytext=(-55, 0),
                            textcoords="offset points",
                            ha="right",
                            va="center",
                        )

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

            # Clamp after all panels (sharex may expand while later axes plot).
            for ax in np.ravel(axes):
                if ax.has_data():
                    clamp_aligned_time_xlim(ax, ALIGNED_TIME_XLIM_MAX_SEC)
                    current_xlim = ax.get_xlim()
                    if each_plot_info.get("plot_zero_line"):
                        ax.plot(
                            [current_xlim[0], current_xlim[1]],
                            [0, 0],
                            "--",
                            color="gray",
                            linewidth=0.5,
                        )

            fig.subplots_adjust(left=0.22, right=0.98, top=0.9, bottom=0.14)
            savepath = os.path.join(save_folder, f"panel_{each_plot_type}_lineplot_time_series.png")
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
            plt.show()

    # %% line plot, mean ± SEM (individual traces faint, mean thick, SEM shaded)
    mean_sem_plot_info = plot_info_dict["intensity"]
    y_col_mean_sem = mean_sem_plot_info["y"]
    ylim_mean_sem = mean_sem_plot_info["ylim"]
    ylabel_mean_sem = mean_sem_plot_info["ylabel"]

    for each_header_name, each_condition in condition_group_specs:
        eachgroup_df = fulltimeseries_df[fulltimeseries_df["condition"] == each_condition]
        if eachgroup_df.empty:
            continue
        for each_uncaging_power_coherent_mW in sorted_uncaging_powers:
            plot_df = eachgroup_df[eachgroup_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
            if plot_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=300)
            n = plot_spine_volume_mean_sem(
                ax,
                plot_df,
                y_col_mean_sem,
                mean_color=MEAN_SEM_MEAN_COLOR,
                indiv_color=MEAN_SEM_INDIV_COLOR,
                label=f"Mean ± SEM (n={plot_df['group_set_id'].nunique()})",
            )
            decorate_spine_volume_timecourse_ax(ax, ylim_mean_sem, ltp_window_min=LTP_data_point_after_min_between, xlim_min=BINNED_TIME_XLIM_MIN, xlim_max=BINNED_TIME_XLIM_MAX)
            ax.set_xlabel("Time (min)", fontsize=10)
            ax.set_ylabel(ylabel_mean_sem, fontsize=10)
            ax.set_title(f"{each_header_name}, {each_uncaging_power_coherent_mW:g} mW", fontsize=10)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            fig.tight_layout()
            savepath = os.path.join(
                save_folder,
                f"{each_condition}_intensity_{each_uncaging_power_coherent_mW}mW_lineplot_mean_sem.png",
            )
            plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved mean±SEM time course: {savepath} (n={n})")
            plt.show()

    if len(sorted_uncaging_powers) > 0 and len(valid_group_headers) > 0:
        n_rows = len(sorted_uncaging_powers)
        n_cols = len(valid_group_headers)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.6 * n_cols, 3.1 * n_rows),
            sharex=True,
            sharey=True,
            dpi=300,
        )
        axes = reshape_axes_to_2d(axes, n_rows, n_cols)
        for col_idx, (each_condition, each_header_name) in enumerate(valid_group_headers):
            eachgroup_df = fulltimeseries_df[fulltimeseries_df["condition"] == each_condition]
            for row_idx, each_uncaging_power_coherent_mW in enumerate(sorted_uncaging_powers):
                ax = axes[row_idx, col_idx]
                plot_df = eachgroup_df[eachgroup_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
                if plot_df.empty:
                    ax.axis("off")
                    continue
                plot_spine_volume_mean_sem(
                    ax,
                    plot_df,
                    y_col_mean_sem,
                    mean_color=MEAN_SEM_MEAN_COLOR,
                    indiv_color=MEAN_SEM_INDIV_COLOR,
                    label=f"Mean ± SEM (n={plot_df['group_set_id'].nunique()})",
                )
                decorate_spine_volume_timecourse_ax(ax, ylim_mean_sem, ltp_window_min=LTP_data_point_after_min_between, xlim_min=BINNED_TIME_XLIM_MIN, xlim_max=BINNED_TIME_XLIM_MAX)
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Time (min)")
                else:
                    ax.set_xlabel("")
                if col_idx == 0:
                    ax.set_ylabel(ylabel_mean_sem)
                else:
                    ax.set_ylabel("")
                if row_idx == 0:
                    ax.set_title(each_header_name)
                else:
                    ax.set_title("")
                if col_idx == 0:
                    ax.annotate(
                        f"{each_uncaging_power_coherent_mW} mW",
                        xy=(0, 0.5),
                        xycoords="axes fraction",
                        xytext=(-55, 0),
                        textcoords="offset points",
                        ha="right",
                        va="center",
                    )
                ax.legend(frameon=False, fontsize=7, loc="upper left")
        fig.subplots_adjust(left=0.22, right=0.98, top=0.9, bottom=0.14)
        savepath = os.path.join(save_folder, "panel_intensity_lineplot_mean_sem.png")
        plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved mean±SEM tiled time course: {savepath}")
        plt.show()

    for each_uncaging_power_coherent_mW in sorted_uncaging_powers:
        fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=300)
        n_plotted = 0
        for each_header_name, each_condition in condition_group_specs:
            style = CONDITION_MEAN_SEM_STYLE.get(
                each_condition,
                {"indiv": MEAN_SEM_INDIV_COLOR, "mean": MEAN_SEM_MEAN_COLOR},
            )
            plot_df = fulltimeseries_df[
                (fulltimeseries_df["condition"] == each_condition)
                & (fulltimeseries_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW)
            ]
            if plot_df.empty:
                continue
            n = plot_spine_volume_mean_sem(
                ax,
                plot_df,
                y_col_mean_sem,
                mean_color=style["mean"],
                indiv_color=style["indiv"],
                label=f"{each_header_name} (n={plot_df['group_set_id'].nunique()})",
            )
            n_plotted += n
        if n_plotted == 0:
            plt.close(fig)
            continue
        decorate_spine_volume_timecourse_ax(ax, ylim_mean_sem, ltp_window_min=LTP_data_point_after_min_between, xlim_min=BINNED_TIME_XLIM_MIN, xlim_max=BINNED_TIME_XLIM_MAX)
        ax.set_xlabel("Time (min)", fontsize=10)
        ax.set_ylabel(ylabel_mean_sem, fontsize=10)
        ax.set_title(f"{each_uncaging_power_coherent_mW:g} mW", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        savepath = os.path.join(
            save_folder,
            f"panel_intensity_lineplot_mean_sem_overlay_{each_uncaging_power_coherent_mW:g}mW.png",
        )
        plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved mean±SEM overlay: {savepath}")
        plt.show()

    # %% swarm plot
    swarm_min = summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].min()
    swarm_max = summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].max()
    ten_percent_ylim = (swarm_max - swarm_min) * 0.1
    swarm_ylim = [swarm_min-ten_percent_ylim, swarm_max+ten_percent_ylim]
    plot_info_dict = {
        # "lifetime": {"ylabel": r"$\Delta$lifetime (ns)",
        #                             "y": "delta_lifetime",
        #                             "ylim" : [-0.19, 0.29]},
                      "intensity": {"ylabel": r"$\Delta$spine volume (a.u.)", 
                                    "y": f"delta_FF0_intensity_ch{ch_1or2}", 
                                    "ylim" : swarm_ylim}
                    }

    for each_header_name, each_condition in build_condition_group_specs(summary_df, CONDITION_PREFIX_ORDER):
        each_header_summary_df = summary_df[summary_df["condition"] == each_condition]
        if len(each_header_summary_df) == 0:
            continue
        for each_uncaging_power_coherent_mW in fulltimeseries_df["uncaging_power_coherent_mW"].unique():
            each_group_same_unc_pow_df = each_header_summary_df[each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
            plot_df = each_group_same_unc_pow_df
            for each_plot_type, each_plot_info in plot_info_dict.items():
                plt.figure(figsize=(2, 3))
                p = sns.swarmplot(y=each_plot_info["y"],
                            data=plot_df,
                            palette = "tab10",
                            )

                sns.boxplot(showmeans=True,
                    meanline=True,
                    meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                    medianprops={'visible': False},
                    whiskerprops={'visible': False},
                    zorder=10,
                    y=each_plot_info["y"],
                    data=plot_df,
                    showfliers=False,
                    showbox=False,
                    showcaps=False,
                    ax=p)

                mean = plot_df[each_plot_info["y"]].mean()
                std = plot_df[each_plot_info["y"]].std()
                plt.text(0.2, mean, f"{mean:.2f} ± {std:.2f}", ha="left", va="bottom")

                plt.ylabel(each_plot_info["ylabel"])
                plt.ylim(each_plot_info["ylim"])
                plt.title(each_header_name+ f", uncaging {each_uncaging_power_coherent_mW} mW")
                #delete right and top border
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                savepath = os.path.join(save_folder, f"{each_condition}_{each_plot_type}_{each_uncaging_power_coherent_mW}mW_plot_swarmplot.png")
                plt.savefig(savepath, dpi=150, bbox_inches = "tight")
                plt.show()

    # %% swarm plot (tiled panels for group_header x uncaging_power_coherent_mW)
    sorted_uncaging_powers = sorted(fulltimeseries_df["uncaging_power_coherent_mW"].dropna().unique())
    valid_group_headers = []
    for each_header_name, each_condition in build_condition_group_specs(summary_df, CONDITION_PREFIX_ORDER):
        if (summary_df["condition"] == each_condition).any():
            valid_group_headers.append((each_condition, each_header_name))

    if len(sorted_uncaging_powers) > 0 and len(valid_group_headers) > 0:
        for each_plot_type, each_plot_info in plot_info_dict.items():
            n_rows = len(sorted_uncaging_powers)
            n_cols = len(valid_group_headers)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(3.2 * n_cols, 3 * n_rows),
                sharex=False,
                sharey=True,
            )

            axes = reshape_axes_to_2d(axes, n_rows, n_cols)

            for col_idx, (each_condition, each_header_name) in enumerate(valid_group_headers):
                each_header_summary_df = summary_df[summary_df["condition"] == each_condition]
                for row_idx, each_uncaging_power_coherent_mW in enumerate(sorted_uncaging_powers):
                    ax = axes[row_idx, col_idx]
                    plot_df = each_header_summary_df[
                        each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW
                    ]
                    if plot_df.empty:
                        ax.axis("off")
                        continue

                    p = sns.swarmplot(
                        y=each_plot_info["y"],
                        data=plot_df,
                        palette="tab10",
                        ax=ax,
                    )

                    sns.boxplot(
                        showmeans=True,
                        meanline=True,
                        meanprops={"color": "r", "ls": "-", "lw": 1},
                        medianprops={"visible": False},
                        whiskerprops={"visible": False},
                        zorder=10,
                        y=each_plot_info["y"],
                        data=plot_df,
                        showfliers=False,
                        showbox=False,
                        showcaps=False,
                        ax=p,
                    )

                    mean = plot_df[each_plot_info["y"]].mean()
                    std = plot_df[each_plot_info["y"]].std()
                    ax.text(0.2, mean, f"{mean:.2f} ± {std:.2f}", ha="left", va="bottom")

                    ax.set_ylim(each_plot_info["ylim"])

                    if row_idx == n_rows - 1:
                        ax.set_xlabel("")
                    else:
                        ax.set_xlabel("")

                    if col_idx == 0:
                        ax.set_ylabel(each_plot_info["ylabel"])
                    else:
                        ax.set_ylabel("")

                    if row_idx == 0:
                        ax.set_title(each_header_name)
                    else:
                        ax.set_title("")

                    if col_idx == 0:
                        ax.annotate(
                            f"{each_uncaging_power_coherent_mW} mW",
                            xy=(0, 0.5),
                            xycoords="axes fraction",
                            xytext=(-55, 0),
                            textcoords="offset points",
                            ha="right",
                            va="center",
                        )

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

            fig.subplots_adjust(left=0.26, right=0.98, top=0.9, bottom=0.14)
            savepath = os.path.join(save_folder, f"panel_{each_plot_type}_swarmplot.png")
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
            plt.show()


    # %% combined paper-style swarmplot (all groups in one figure + mean difference test)
    y_col = f"delta_FF0_intensity_ch{ch_1or2}"
    ylabel = r"$\Delta$spine volume (a.u.)"
    combined_rows = []
    group_order = []
    for each_header_name, each_condition in build_condition_group_specs(summary_df, CONDITION_PREFIX_ORDER):
        each_header_summary_df = summary_df[summary_df["condition"] == each_condition]
        if each_header_summary_df.empty:
            continue
        group_order.append(each_header_name)
        tmp = each_header_summary_df.copy()
        tmp["condition"] = each_header_name
        combined_rows.append(tmp)

    if len(combined_rows) >= 2:
        plot_df = pd.concat(combined_rows, ignore_index=True)
        sorted_powers = sorted(plot_df["uncaging_power_coherent_mW"].dropna().unique())
        for each_power in sorted_powers:
            power_df = plot_df[plot_df["uncaging_power_coherent_mW"] == each_power].copy()
            if power_df.empty:
                continue

            present_groups = [g for g in group_order if (power_df["condition"] == g).any()]
            if len(present_groups) < 2:
                continue

            fig, ax = plt.subplots(figsize=(2.6, 3.4), dpi=300)
            rng = np.random.default_rng(0)
            x_positions = {g: i for i, g in enumerate(present_groups)}
            means = []
            sems = []
            ns = []
            group_values = {}

            for g in present_groups:
                vals = power_df.loc[power_df["condition"] == g, y_col].dropna().to_numpy(dtype=float)
                group_values[g] = vals
                n = len(vals)
                ns.append(n)
                mean = float(np.mean(vals)) if n else np.nan
                sem = float(scipy_stats.sem(vals)) if n > 1 else 0.0
                means.append(mean)
                sems.append(sem)
                x = np.full(n, x_positions[g], dtype=float)
                x = x + rng.uniform(-0.08, 0.08, size=n)
                ax.scatter(
                    x,
                    vals,
                    s=22,
                    c="0.35",
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=2,
                    alpha=0.9,
                )

            for i, g in enumerate(present_groups):
                ax.errorbar(
                    i,
                    means[i],
                    yerr=sems[i],
                    fmt="o",
                    color="black",
                    ecolor="black",
                    elinewidth=1.1,
                    capsize=3.5,
                    capthick=1.1,
                    markersize=5.5,
                    zorder=4,
                )

            # Primary test: Welch's t-test on means (unequal variance). Also report Mann-Whitney.
            g0, g1 = present_groups[0], present_groups[1]
            v0, v1 = group_values[g0], group_values[g1]
            welch = ttest_ind(v0, v1, equal_var=False, nan_policy="omit")
            mw = scipy_stats.mannwhitneyu(v0, v1, alternative="two-sided")
            star = p_to_stars(float(welch.pvalue))

            y_max = float(np.nanmax(power_df[y_col].to_numpy(dtype=float)))
            y_min = float(np.nanmin(power_df[y_col].to_numpy(dtype=float)))
            y_span = max(y_max - y_min, 0.2)
            bracket_y = y_max + 0.12 * y_span
            bracket_h = 0.05 * y_span
            add_significance_bracket(ax, 0, 1, bracket_y, bracket_h, star)

            ax.axhline(0.0, color="0.6", lw=0.7, ls="--", zorder=1)
            ax.set_xlim(-0.55, len(present_groups) - 0.45)
            ax.set_ylim(y_min - 0.12 * y_span, bracket_y + 0.22 * y_span)
            ax.set_xticks(range(len(present_groups)))
            ax.set_xticklabels(
                [f"{g}\n(n={n})" for g, n in zip(present_groups, ns)],
                fontsize=8,
            )
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel("")
            ax.set_title(f"{each_power:g} mW", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", width=0.8, length=3)

            stats_txt = (
                f"Welch t-test: p={welch.pvalue:.3g}\n"
                f"Mann-Whitney: p={mw.pvalue:.3g}\n"
                f"mean±SEM: {means[0]:.2f}±{sems[0]:.2f} vs {means[1]:.2f}±{sems[1]:.2f}"
            )
            ax.text(
                0.02,
                0.98,
                stats_txt,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="0.2",
            )

            fig.tight_layout()
            savepath = os.path.join(
                save_folder,
                f"panel_intensity_swarmplot_combined_stats_{each_power:g}mW.png",
            )
            plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
            stats_csv = os.path.join(
                save_folder,
                f"panel_intensity_swarmplot_combined_stats_{each_power:g}mW.csv",
            )
            pd.DataFrame(
                [
                    {
                        "uncaging_power_coherent_mW": each_power,
                        "group_a": g0,
                        "group_b": g1,
                        "n_a": len(v0),
                        "n_b": len(v1),
                        "mean_a": means[0],
                        "sem_a": sems[0],
                        "mean_b": means[1],
                        "sem_b": sems[1],
                        "welch_t": float(welch.statistic),
                        "welch_p": float(welch.pvalue),
                        "mannwhitney_u": float(mw.statistic),
                        "mannwhitney_p": float(mw.pvalue),
                        "stars_welch": star,
                    }
                ]
            ).to_csv(stats_csv, index=False)
            print(f"Saved combined stats swarmplot: {savepath}")
            print(f"Saved stats table: {stats_csv}")
            plt.show()
    elif len(combined_rows) == 1:
        print("Only one group present; skip combined stats swarmplot.")


    # %% line plot, transient
    #plot each data, with light thin color lines
    plot_info_dict = {
        "GCaMP_transient_Spine_intensity": {"ylabel": r"GCaMP F/F0", 
                                "xlabel": "Time (sec)",
                                "x": "aligned_time_sec",
                                "y": "transient_Spine_Ch1_intensity", 
                                "errorbar": "se",
                                "ylim": [fulltimeseries_df["transient_Spine_Ch1_intensity"].min()-0.1, 
                                         25],
                                },
        "GCaMP_transient_DendriticShaft_intensity": {"ylabel": r"GCaMP F/F0", 
                                "xlabel": "Time (sec)",
                                "x": "aligned_time_sec",
                                "y": "transient_DendriticShaft_Ch1_intensity", 
                                "errorbar": "se",
                                "ylim": [fulltimeseries_df["transient_DendriticShaft_Ch1_intensity"].min()-0.1, 
                                         fulltimeseries_df["transient_DendriticShaft_Ch1_intensity"].max()+0.1]}
        }

    for each_header_name, each_condition in condition_group_specs:
        eachgroup_df = fulltimeseries_df[fulltimeseries_df["condition"] == each_condition]
        if len(eachgroup_df) == 0:
            continue

        for each_uncaging_power_coherent_mW in fulltimeseries_df["uncaging_power_coherent_mW"].unique():
            each_group_same_unc_pow_df = eachgroup_df[eachgroup_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
            plot_df = each_group_same_unc_pow_df[each_group_same_unc_pow_df["phase"] == "unc"]

            print(each_header_name, each_uncaging_power_coherent_mW, len(plot_df))

            for each_plot_type, each_plot_info in plot_info_dict.items():
                plt.figure(figsize=(5, 3))
                g = sns.lineplot(
                            x = each_plot_info["x"],
                            y = each_plot_info["y"],
                            data = plot_df,
                            hue = "group_set_id",
                            linewidth = 0.5,
                            alpha = 0.5,
                            palette = "tab10",
                            legend = False,
                            marker = "o",
                            markersize = 4,
                            )
                #greek delta lifetime
                plt.ylabel(each_plot_info["ylabel"])
                plt.xlabel(each_plot_info["xlabel"])
                plt.title(each_header_name+ f", uncaging {each_uncaging_power_coherent_mW} mW")
                plt.ylim(each_plot_info["ylim"])
                # #plot mean with SEM

                ylim = plt.gca().get_ylim()
                ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 0.9
                plt.plot([0, 2.048*29], [ninty_percent_ylim, ninty_percent_ylim], "k-")
                plt.text(0, ninty_percent_ylim*1.005, "uncaging", ha="left", va="bottom")

                # plt.fill_between(np.array(LTP_data_point_after_min_between)*60,
                # each_plot_info["ylim"][0], each_plot_info["ylim"][1], 
                # color="pink", 
                # alpha=0.3)

                #delete right and top border
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                savepath = os.path.join(save_folder, f"{each_condition}_{each_plot_type}_{each_uncaging_power_coherent_mW}mW_lineplot_time_series.png")
                plt.savefig(savepath, dpi=150, bbox_inches = "tight")
                plt.show()

    # %% line plot, transient (tiled panels for group_header x uncaging_power_coherent_mW)
    sorted_uncaging_powers = sorted(fulltimeseries_df["uncaging_power_coherent_mW"].dropna().unique())
    valid_group_headers = []
    for each_header_name, each_condition in condition_group_specs:
        if (fulltimeseries_df["condition"] == each_condition).any():
            valid_group_headers.append((each_condition, each_header_name))

    if len(sorted_uncaging_powers) > 0 and len(valid_group_headers) > 0:
        for each_plot_type, each_plot_info in plot_info_dict.items():
            n_rows = len(sorted_uncaging_powers)
            n_cols = len(valid_group_headers)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6 * n_cols, 3 * n_rows),
                sharex=True,
                sharey=True,
            )

            axes = reshape_axes_to_2d(axes, n_rows, n_cols)

            for col_idx, (each_condition, each_header_name) in enumerate(valid_group_headers):
                eachgroup_df = fulltimeseries_df[fulltimeseries_df["condition"] == each_condition]
                for row_idx, each_uncaging_power_coherent_mW in enumerate(sorted_uncaging_powers):
                    ax = axes[row_idx, col_idx]
                    each_group_same_unc_pow_df = eachgroup_df[
                        eachgroup_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW
                    ]
                    plot_df = each_group_same_unc_pow_df[each_group_same_unc_pow_df["phase"] == "unc"]
                    if plot_df.empty:
                        ax.axis("off")
                        continue

                    sns.lineplot(
                        x=each_plot_info["x"],
                        y=each_plot_info["y"],
                        data=plot_df,
                        hue="group_set_id",
                        linewidth=0.5,
                        alpha=0.5,
                        palette="tab10",
                        legend=False,
                        marker="o",
                        markersize=4,
                        ax=ax,
                    )

                    # current_xlim = ax.get_xlim()
                    # ax.plot([current_xlim[0], current_xlim[1]], [0, 0], "--", color="gray", linewidth=0.5)
                    # print(current_xlim)

                    ax.set_ylim(each_plot_info["ylim"])

                    if row_idx == n_rows - 1:
                        ax.set_xlabel(each_plot_info["xlabel"])
                    else:
                        ax.set_xlabel("")

                    if col_idx == 0:
                        ax.set_ylabel(each_plot_info["ylabel"])
                    else:
                        ax.set_ylabel("")

                    ylim = ax.get_ylim()
                    ninty_percent_ylim = ylim[0] + (ylim[1] - ylim[0]) * 0.9
                    ax.plot([0, 2.048 * 29], [ninty_percent_ylim, ninty_percent_ylim], "k-")
                    ax.text(0, ninty_percent_ylim * 1.005, "uncaging", ha="left", va="bottom")

                    if row_idx == 0:
                        ax.set_title(each_header_name)
                    else:
                        ax.set_title("")

                    if col_idx == 0:
                        ax.annotate(
                            f"{each_uncaging_power_coherent_mW} mW",
                            xy=(0, 0.5),
                            xycoords="axes fraction",
                            xytext=(-55, 0),
                            textcoords="offset points",
                            ha="right",
                            va="center",
                        )

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

            fig.subplots_adjust(left=0.22, right=0.98, top=0.9, bottom=0.14)
            savepath = os.path.join(save_folder, f"panel_{each_plot_type}_transient_lineplot_time_series.png")
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
            plt.show()


    # %% swarm plot for transient

    plot_info_dict = {
        "GCaMP_transient_Spine_intensity": {"ylabel": r"GCaMP F/F0", 
                                "xlabel": "Time (sec)",
                                "x": "aligned_time_sec",
                                "y": "transient_Spine_Ch1_intensity", 
                                "errorbar": "se",
                                "ylim": [summary_df["transient_Spine_Ch1_intensity"].min() - (summary_df["transient_Spine_Ch1_intensity"].max() - summary_df["transient_Spine_Ch1_intensity"].min()) * 0.1, 
                                         summary_df["transient_Spine_Ch1_intensity"].max() + (summary_df["transient_Spine_Ch1_intensity"].max() - summary_df["transient_Spine_Ch1_intensity"].min()) * 0.1]
                                         },
        "GCaMP_transient_DendriticShaft_intensity": {
            "ylabel": r"GCaMP F/F0", 
            "xlabel": "Time (sec)",
            "x": "aligned_time_sec",
            "y": "transient_DendriticShaft_Ch1_intensity", 
            "errorbar": "se",
            "ylim": [summary_df["transient_DendriticShaft_Ch1_intensity"].min() - (summary_df["transient_DendriticShaft_Ch1_intensity"].max() - summary_df["transient_DendriticShaft_Ch1_intensity"].min()) * 0.1, 
                     summary_df["transient_DendriticShaft_Ch1_intensity"].max() + (summary_df["transient_DendriticShaft_Ch1_intensity"].max() - summary_df["transient_DendriticShaft_Ch1_intensity"].min()) * 0.1]}
                    }

    for each_header_name, each_condition in build_condition_group_specs(summary_df, CONDITION_PREFIX_ORDER):
        each_header_summary_df = summary_df[summary_df["condition"] == each_condition]
        if len(each_header_summary_df) == 0:
            continue
        for each_uncaging_power_coherent_mW in fulltimeseries_df["uncaging_power_coherent_mW"].unique():
            each_group_same_unc_pow_df = each_header_summary_df[each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
            plot_df = each_group_same_unc_pow_df
            for each_plot_type, each_plot_info in plot_info_dict.items():
                plt.figure(figsize=(2, 3))
                p = sns.swarmplot(y=each_plot_info["y"],
                            data=plot_df,
                            palette = "tab10",
                            )

                sns.boxplot(showmeans=True,
                    meanline=True,
                    meanprops={'color': 'r', 'ls': '-', 'lw': 1},
                    medianprops={'visible': False},
                    whiskerprops={'visible': False},
                    zorder=10,
                    y=each_plot_info["y"],
                    data=plot_df,
                    showfliers=False,
                    showbox=False,
                    showcaps=False,
                    ax=p)

                mean = plot_df[each_plot_info["y"]].mean()
                std = plot_df[each_plot_info["y"]].std()
                plt.text(0.2, mean, f"{mean:.2f} ± {std:.2f}", ha="left", va="bottom")

                plt.ylabel(each_plot_info["ylabel"])
                plt.ylim(each_plot_info["ylim"])
                plt.title(each_header_name+ f", uncaging {each_uncaging_power_coherent_mW} mW")
                #delete right and top border
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                savepath = os.path.join(save_folder, f"{each_condition}_{each_plot_type}_{each_uncaging_power_coherent_mW}mW_plot_transient_swarmplot.png")
                plt.savefig(savepath, dpi=150, bbox_inches = "tight")
                plt.show()

    # %% swarm plot for transient (tiled panels for group_header x uncaging_power_coherent_mW)
    sorted_uncaging_powers = sorted(fulltimeseries_df["uncaging_power_coherent_mW"].dropna().unique())
    valid_group_headers = []
    for each_header_name, each_condition in build_condition_group_specs(summary_df, CONDITION_PREFIX_ORDER):
        if (summary_df["condition"] == each_condition).any():
            valid_group_headers.append((each_condition, each_header_name))

    if len(sorted_uncaging_powers) > 0 and len(valid_group_headers) > 0:
        for each_plot_type, each_plot_info in plot_info_dict.items():
            n_rows = len(sorted_uncaging_powers)
            n_cols = len(valid_group_headers)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(3.2 * n_cols, 3 * n_rows),
                sharex=False,
                sharey=True,
            )

            axes = reshape_axes_to_2d(axes, n_rows, n_cols)

            for col_idx, (each_condition, each_header_name) in enumerate(valid_group_headers):
                each_header_summary_df = summary_df[summary_df["condition"] == each_condition]
                for row_idx, each_uncaging_power_coherent_mW in enumerate(sorted_uncaging_powers):
                    ax = axes[row_idx, col_idx]
                    plot_df = each_header_summary_df[
                        each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW
                    ]
                    if plot_df.empty:
                        ax.axis("off")
                        continue

                    p = sns.swarmplot(
                        y=each_plot_info["y"],
                        data=plot_df,
                        palette="tab10",
                        ax=ax,
                    )

                    sns.boxplot(
                        showmeans=True,
                        meanline=True,
                        meanprops={"color": "r", "ls": "-", "lw": 1},
                        medianprops={"visible": False},
                        whiskerprops={"visible": False},
                        zorder=10,
                        y=each_plot_info["y"],
                        data=plot_df,
                        showfliers=False,
                        showbox=False,
                        showcaps=False,
                        ax=p,
                    )

                    mean = plot_df[each_plot_info["y"]].mean()
                    std = plot_df[each_plot_info["y"]].std()
                    ax.text(0.2, mean, f"{mean:.2f} ± {std:.2f}", ha="left", va="bottom")

                    ax.set_ylim(each_plot_info["ylim"])

                    if row_idx == n_rows - 1:
                        ax.set_xlabel("")
                    else:
                        ax.set_xlabel("")

                    if col_idx == 0:
                        ax.set_ylabel(each_plot_info["ylabel"])
                    else:
                        ax.set_ylabel("")

                    if row_idx == 0:
                        ax.set_title(each_header_name)
                    else:
                        ax.set_title("")

                    if col_idx == 0:
                        ax.annotate(
                            f"{each_uncaging_power_coherent_mW} mW",
                            xy=(0, 0.5),
                            xycoords="axes fraction",
                            xytext=(-55, 0),
                            textcoords="offset points",
                            ha="right",
                            va="center",
                        )

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

            fig.subplots_adjust(left=0.26, right=0.98, top=0.9, bottom=0.14)
            savepath = os.path.join(save_folder, f"panel_{each_plot_type}_transient_swarmplot.png")
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
            plt.show()



    # %% scatter plot

    plot_info_dict = {
        "GCaMP_LTP_level_against_vs_DendriticShaft_F_F0": 
                                {"ylabel": r"$\Delta$spine volume (a.u.)", 
                                "xlabel": "GCaMP Dendritic Shaft F/F0",
                                "y": "delta_FF0_intensity_ch2", 
                                "x": "transient_DendriticShaft_Ch1_intensity",
                                "ylim": [summary_df["delta_FF0_intensity_ch2"].min() - (summary_df["delta_FF0_intensity_ch2"].max() - summary_df["delta_FF0_intensity_ch2"].min()) * 0.1, 
                                         summary_df["delta_FF0_intensity_ch2"].max() + (summary_df["delta_FF0_intensity_ch2"].max() - summary_df["delta_FF0_intensity_ch2"].min()) * 0.1],
                                "xlim": [summary_df["transient_DendriticShaft_Ch1_intensity"].min() - (summary_df["transient_DendriticShaft_Ch1_intensity"].max() - summary_df["transient_DendriticShaft_Ch1_intensity"].min()) * 0.1, 
                                         summary_df["transient_DendriticShaft_Ch1_intensity"].max() + (summary_df["transient_DendriticShaft_Ch1_intensity"].max() - summary_df["transient_DendriticShaft_Ch1_intensity"].min()) * 0.1],
                                         },
        "GCaMP_LTP_level_against_vs_Spine_F_F0": 
                                {"ylabel": r"$\Delta$spine volume (a.u.)", 
                                "xlabel": "GCaMP Spine F/F0",
                                "y": "delta_FF0_intensity_ch2", 
                                "x": "transient_Spine_Ch1_intensity",
                                "ylim": [summary_df["delta_FF0_intensity_ch2"].min() - (summary_df["delta_FF0_intensity_ch2"].max() - summary_df["delta_FF0_intensity_ch2"].min()) * 0.1, 
                                         summary_df["delta_FF0_intensity_ch2"].max() + (summary_df["delta_FF0_intensity_ch2"].max() - summary_df["delta_FF0_intensity_ch2"].min()) * 0.1],
                                "xlim": [summary_df["transient_Spine_Ch1_intensity"].min() - (summary_df["transient_Spine_Ch1_intensity"].max() - summary_df["transient_Spine_Ch1_intensity"].min()) * 0.1, 
                                         summary_df["transient_Spine_Ch1_intensity"].max() + (summary_df["transient_Spine_Ch1_intensity"].max() - summary_df["transient_Spine_Ch1_intensity"].min()) * 0.1],
                                         },
        "DendriticShaft_F_F0_vs_Spine_F_F0": 
                                {"ylabel": "GCaMP Spine F/F0", 
                                "xlabel": "GCaMP Dendritic Shaft F/F0",
                                "y": "transient_Spine_Ch1_intensity", 
                                "x": "transient_DendriticShaft_Ch1_intensity",
                                "ylim": [summary_df["transient_Spine_Ch1_intensity"].min() - (summary_df["transient_Spine_Ch1_intensity"].max() - summary_df["transient_Spine_Ch1_intensity"].min()) * 0.1, 
                                         summary_df["transient_Spine_Ch1_intensity"].max() + (summary_df["transient_Spine_Ch1_intensity"].max() - summary_df["transient_Spine_Ch1_intensity"].min()) * 0.1],
                                "xlim": [summary_df["transient_DendriticShaft_Ch1_intensity"].min() - (summary_df["transient_DendriticShaft_Ch1_intensity"].max() - summary_df["transient_DendriticShaft_Ch1_intensity"].min()) * 0.1, 
                                         summary_df["transient_DendriticShaft_Ch1_intensity"].max() + (summary_df["transient_DendriticShaft_Ch1_intensity"].max() - summary_df["transient_DendriticShaft_Ch1_intensity"].min()) * 0.1],
                                         },
        f"delta_FF0_ch{ch_1or2}_vs_first_post_FF0_ch{ch_1or2}":
                                {"ylabel": r"$\Delta$spine volume (a.u.) [25-35 min]",
                                "xlabel": r"$\Delta$spine volume (a.u.) [1st post frame]",
                                "y": f"delta_FF0_intensity_ch{ch_1or2}",
                                "x": f"first_post_FF0_intensity_ch{ch_1or2}",
                                "ylim": [summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].min() - (summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].max() - summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].min()) * 0.1,
                                         summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].max() + (summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].max() - summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].min()) * 0.1],
                                "xlim": [summary_df[f"first_post_FF0_intensity_ch{ch_1or2}"].min() - (summary_df[f"first_post_FF0_intensity_ch{ch_1or2}"].max() - summary_df[f"first_post_FF0_intensity_ch{ch_1or2}"].min()) * 0.1,
                                         summary_df[f"first_post_FF0_intensity_ch{ch_1or2}"].max() + (summary_df[f"first_post_FF0_intensity_ch{ch_1or2}"].max() - summary_df[f"first_post_FF0_intensity_ch{ch_1or2}"].min()) * 0.1],
                                "plot_zero_lines": True,
                                         }
                    }

    for each_header_name, each_condition in build_condition_group_specs(summary_df, CONDITION_PREFIX_ORDER):
        each_header_summary_df = summary_df[summary_df["condition"] == each_condition]
        if len(each_header_summary_df) == 0:
            continue
        for each_uncaging_power_coherent_mW in fulltimeseries_df["uncaging_power_coherent_mW"].unique():
            each_group_same_unc_pow_df = each_header_summary_df[each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
            plot_df = each_group_same_unc_pow_df
            for each_plot_type, each_plot_info in plot_info_dict.items():
                plt.figure(figsize=(3, 3))
                p = sns.scatterplot(x=each_plot_info["x"],
                                    y=each_plot_info["y"],
                                    data=plot_df,
                                    palette = "tab10",
                                    )

                plt.ylabel(each_plot_info["ylabel"])
                plt.xlabel(each_plot_info["xlabel"])
                plt.xlim(each_plot_info["xlim"])
                plt.ylim(each_plot_info["ylim"])
                if each_plot_info.get("plot_zero_lines"):
                    current_xlim = plt.gca().get_xlim()
                    current_ylim = plt.gca().get_ylim()
                    plt.plot([current_xlim[0], current_xlim[1]], [0, 0], "--", color="gray", linewidth=0.5)
                    plt.plot([0, 0], [current_ylim[0], current_ylim[1]], "--", color="gray", linewidth=0.5)
                plt.title(each_header_name+ f", uncaging {each_uncaging_power_coherent_mW} mW")
                #delete right and top border
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                savepath = os.path.join(save_folder, f"{each_condition}_{each_plot_type}_{each_uncaging_power_coherent_mW}mW_plot_scatterplot.png")
                plt.savefig(savepath, dpi=150, bbox_inches = "tight")
                plt.show()

    # %% scatter plot (tiled panels for group_header x uncaging_power_coherent_mW)
    sorted_uncaging_powers = sorted(fulltimeseries_df["uncaging_power_coherent_mW"].dropna().unique())
    valid_group_headers = []
    for each_header_name, each_condition in build_condition_group_specs(summary_df, CONDITION_PREFIX_ORDER):
        if (summary_df["condition"] == each_condition).any():
            valid_group_headers.append((each_condition, each_header_name))

    if len(sorted_uncaging_powers) > 0 and len(valid_group_headers) > 0:
        for each_plot_type, each_plot_info in plot_info_dict.items():
            n_rows = len(sorted_uncaging_powers)
            n_cols = len(valid_group_headers)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4.5 * n_cols, 3 * n_rows),
                sharex=True,
                sharey=True,
            )

            axes = reshape_axes_to_2d(axes, n_rows, n_cols)

            for col_idx, (each_condition, each_header_name) in enumerate(valid_group_headers):
                each_header_summary_df = summary_df[summary_df["condition"] == each_condition]
                for row_idx, each_uncaging_power_coherent_mW in enumerate(sorted_uncaging_powers):
                    ax = axes[row_idx, col_idx]
                    plot_df = each_header_summary_df[
                        each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW
                    ]
                    if plot_df.empty:
                        ax.axis("off")
                        continue

                    sns.scatterplot(
                        x=each_plot_info["x"],
                        y=each_plot_info["y"],
                        data=plot_df,
                        palette="tab10",
                        ax=ax,
                    )

                    ax.set_xlim(each_plot_info["xlim"])
                    ax.set_ylim(each_plot_info["ylim"])

                    if each_plot_info.get("plot_zero_lines"):
                        ax.plot([each_plot_info["xlim"][0], each_plot_info["xlim"][1]], [0, 0], "--", color="gray", linewidth=0.5)
                        ax.plot([0, 0], [each_plot_info["ylim"][0], each_plot_info["ylim"][1]], "--", color="gray", linewidth=0.5)

                    if row_idx == n_rows - 1:
                        ax.set_xlabel(each_plot_info["xlabel"])
                    else:
                        ax.set_xlabel("")

                    if col_idx == 0:
                        ax.set_ylabel(each_plot_info["ylabel"])
                    else:
                        ax.set_ylabel("")

                    if row_idx == 0:
                        ax.set_title(each_header_name)
                    else:
                        ax.set_title("")

                    if col_idx == 0:
                        ax.annotate(
                            f"{each_uncaging_power_coherent_mW} mW",
                            xy=(0, 0.5),
                            xycoords="axes fraction",
                            xytext=(-55, 0),
                            textcoords="offset points",
                            ha="right",
                            va="center",
                        )

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

            fig.subplots_adjust(left=0.24, right=0.98, top=0.9, bottom=0.16)
            savepath = os.path.join(save_folder, f"panel_{each_plot_type}_scatterplot.png")
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
            plt.show()


    # %% plot against acq_time_str
    acquisiton_start_datetime = datetime.datetime.strptime(acquisiton_start_datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
    summary_df["acq_time_datetime"] = pd.to_datetime(summary_df["acq_time_str"])
    summary_df["time_sec_incubation"] = (summary_df["acq_time_datetime"] - acquisiton_start_datetime).dt.total_seconds()
    summary_df["time_hours_incubation"] = summary_df["time_sec_incubation"] / 3600

    plot_info_dict = {
        "GCaMP_LTP_level_against_vs_time_hours_incubation": 
                                {"ylabel": r"$\Delta$spine volume (a.u.)", 
                                "xlabel": "Incubation time (hours)",
                                "y": f"delta_FF0_intensity_ch{ch_1or2}", 
                                "x": "time_hours_incubation",
                                "ylim": [summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].min() - (summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].max() - summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].min()) * 0.1, 
                                         summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].max() + (summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].max() - summary_df[f"delta_FF0_intensity_ch{ch_1or2}"].min()) * 0.1],
                                "xlim": [summary_df["time_hours_incubation"].min() - (summary_df["time_hours_incubation"].max() - summary_df["time_hours_incubation"].min()) * 0.1, 
                                         summary_df["time_hours_incubation"].max() + (summary_df["time_hours_incubation"].max() - summary_df["time_hours_incubation"].min()) * 0.1],
                                         },
        # "Spine_F_F0_vs_time_hours_incubation": 
        #                         {"ylabel": r"GCaMP Spine F/F0", 
        #                         "xlabel": "Incubation time (hours)",
        #                         "y": "transient_Spine_Ch1_intensity", 
        #                         "x": "time_hours_incubation",
        #                         "ylim": [summary_df["transient_Spine_Ch1_intensity"].min() - (summary_df["transient_Spine_Ch1_intensity"].max() - summary_df["transient_Spine_Ch1_intensity"].min()) * 0.1, 
        #                                  summary_df["transient_Spine_Ch1_intensity"].max() + (summary_df["transient_Spine_Ch1_intensity"].max() - summary_df["transient_Spine_Ch1_intensity"].min()) * 0.1],
        #                         "xlim": [summary_df["time_hours_incubation"].min() - (summary_df["time_hours_incubation"].max() - summary_df["time_hours_incubation"].min()) * 0.1, 
        #                                  summary_df["time_hours_incubation"].max() + (summary_df["time_hours_incubation"].max() - summary_df["time_hours_incubation"].min()) * 0.1],
        #                                  },
        # "DendriticShaft_F_F0_vs_time_hours_incubation": 
        #                         {"ylabel": "GCaMP Dendritic Shaft F/F0", 
        #                         "xlabel": "Incubation time (hours)",
        #                         "y": "transient_DendriticShaft_Ch1_intensity", 
        #                         "x": "time_hours_incubation",
        #                         "ylim": [summary_df["transient_DendriticShaft_Ch1_intensity"].min() - (summary_df["transient_DendriticShaft_Ch1_intensity"].max() - summary_df["transient_DendriticShaft_Ch1_intensity"].min()) * 0.1, 
        #                                  summary_df["transient_DendriticShaft_Ch1_intensity"].max() + (summary_df["transient_DendriticShaft_Ch1_intensity"].max() - summary_df["transient_DendriticShaft_Ch1_intensity"].min()) * 0.1],
        #                         "xlim": [summary_df["time_hours_incubation"].min() - (summary_df["time_hours_incubation"].max() - summary_df["time_hours_incubation"].min()) * 0.1, 
        #                                  summary_df["time_hours_incubation"].max() + (summary_df["time_hours_incubation"].max() - summary_df["time_hours_incubation"].min()) * 0.1],
        #                                  }
                    }

    for each_header_name, each_condition in build_condition_group_specs(summary_df, CONDITION_PREFIX_ORDER):
        each_header_summary_df = summary_df[summary_df["condition"] == each_condition]
        if len(each_header_summary_df) == 0:
            continue
        for each_uncaging_power_coherent_mW in fulltimeseries_df["uncaging_power_coherent_mW"].unique():
            each_group_same_unc_pow_df = each_header_summary_df[each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
            plot_df = each_group_same_unc_pow_df
            for each_plot_type, each_plot_info in plot_info_dict.items():
                plt.figure(figsize=(4, 3))
                p = sns.scatterplot(x=each_plot_info["x"],
                                    y=each_plot_info["y"],
                                    data=plot_df,
                                    palette = "tab10",
                                    )

                plt.ylabel(each_plot_info["ylabel"])
                plt.xlabel(each_plot_info["xlabel"])
                plt.xlim(each_plot_info["xlim"])
                plt.ylim(each_plot_info["ylim"])
                plt.title(each_header_name+ f", uncaging {each_uncaging_power_coherent_mW} mW")
                #delete right and top border
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                savepath = os.path.join(save_folder, f"{each_condition}_{each_plot_type}_{each_uncaging_power_coherent_mW}mW_plot_scatterplot.png")
                plt.savefig(savepath, dpi=150, bbox_inches = "tight")
                plt.show()

    # %% plot against acq_time_str (tiled panels for group_header x uncaging_power_coherent_mW)
    sorted_uncaging_powers = sorted(fulltimeseries_df["uncaging_power_coherent_mW"].dropna().unique())
    valid_group_headers = []
    for each_header_name, each_condition in build_condition_group_specs(summary_df, CONDITION_PREFIX_ORDER):
        if (summary_df["condition"] == each_condition).any():
            valid_group_headers.append((each_condition, each_header_name))

    if len(sorted_uncaging_powers) > 0 and len(valid_group_headers) > 0:
        for each_plot_type, each_plot_info in plot_info_dict.items():
            n_rows = len(sorted_uncaging_powers)
            n_cols = len(valid_group_headers)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(5 * n_cols, 3 * n_rows),
                sharex=True,
                sharey=True,
            )

            axes = reshape_axes_to_2d(axes, n_rows, n_cols)

            for col_idx, (each_condition, each_header_name) in enumerate(valid_group_headers):
                each_header_summary_df = summary_df[summary_df["condition"] == each_condition]
                for row_idx, each_uncaging_power_coherent_mW in enumerate(sorted_uncaging_powers):
                    ax = axes[row_idx, col_idx]
                    plot_df = each_header_summary_df[
                        each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW
                    ]
                    if plot_df.empty:
                        ax.axis("off")
                        continue

                    sns.scatterplot(
                        x=each_plot_info["x"],
                        y=each_plot_info["y"],
                        data=plot_df,
                        palette="tab10",
                        ax=ax,
                    )

                    ax.set_xlim(each_plot_info["xlim"])
                    ax.set_ylim(each_plot_info["ylim"])

                    if row_idx == n_rows - 1:
                        ax.set_xlabel(each_plot_info["xlabel"])
                    else:
                        ax.set_xlabel("")

                    if col_idx == 0:
                        ax.set_ylabel(each_plot_info["ylabel"])
                    else:
                        ax.set_ylabel("")

                    if row_idx == 0:
                        ax.set_title(each_header_name)
                    else:
                        ax.set_title("")

                    if col_idx == 0:
                        ax.annotate(
                            f"{each_uncaging_power_coherent_mW} mW",
                            xy=(0, 0.5),
                            xycoords="axes fraction",
                            xytext=(-55, 0),
                            textcoords="offset points",
                            ha="right",
                            va="center",
                        )

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

            fig.subplots_adjust(left=0.26, right=0.98, top=0.9, bottom=0.18)
            savepath = os.path.join(save_folder, f"panel_{each_plot_type}_time_scatterplot.png")
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
            plt.show()


    # %%
    print("plots were saved to:")
    print(save_folder)
    return save_folder
