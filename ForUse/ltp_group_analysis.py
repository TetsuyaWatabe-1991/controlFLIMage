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
    # Mean-line colors in condition_order. Extra entries are unused.
    condition_styles: Sequence[str] = (
        "k",
        "b",
        "g",
        "r",
        "#E69F00",
        "m",
        "c",
        "#56B4E9",
    )
    ch_1or2: int = 2
    ltp_window_min: Sequence[float] = (25.0, 35.0)
    pre_instability_max_min_ratio_threshold: float = 2.0
    # 25-35 min Δspine volume (F/F0 - 1). None = no cutoff on that side.
    spine_volume_delta_ff0_min: float | None = None
    spine_volume_delta_ff0_max: float | None = None
    aligned_time_xlim_max_sec: float = 2100.0
    # Mean±SEM x-axis in minutes. Negative is pre, positive is post.
    # Uncaging is at 0. Does not change the seconds-scale line plots.
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
    # 0 keeps the mean color; 1 fades individual traces to white. Applied to every condition.
    mean_sem_indiv_lightness: float = 0.65
    mean_sem_mean_color: str = "r"
    # Uncaging frames use a different frame average and no Z projection, so they
    # are omitted from pre/post mean-intensity time courses unless this is True.
    plot_uncaging_on_mean_intensity: bool = False
    # Quantity of the main plots (time course, mean+/-SEM, swarm, stats, incubation time):
    #   "intensity": spine volume = F/F0 - 1 of the ch_1or2 Spine intensity.
    #   "lifetime":  delta lifetime (ns) of the ch_1or2 Spine = lifetime - pre mean.
    # Lifetime plots and summary_df.csv go to <session>/summary_lifetime/.
    # The intensity filters above (pre max/min ratio, spine volume cutoff) apply in both modes.
    signal: str = "intensity"
    # Lifetime filters (ns), used in both modes when set. None = off.
    # pre_lifetime_range_max_ns: drop a set whose pre Spine lifetimes (max - min) exceed it.
    # delta_lifetime_min/max: drop a set whose 25-35 min delta lifetime is outside the range.
    pre_lifetime_range_max_ns: float | None = None
    delta_lifetime_min: float | None = None
    delta_lifetime_max: float | None = None
    # True: groups are split by uncaging protocol (pulses and pulse interval of the
    # uncaging file), e.g. "Culture media, 30 pulses 1.95 Hz". The combined statistics
    # compare the conditions within each protocol.
    split_by_uncaging_protocol: bool = False


SIGNALS = ("intensity", "lifetime")
PROTOCOL_SEP = ", "


def uncaging_protocol_label(statedict: Mapping) -> str:
    """Short name of an uncaging protocol from the FLIM header (pulse count and rate)."""
    n = statedict.get("State.Uncaging.trainRepeat")
    interval_ms = statedict.get("State.Uncaging.trainInterval")
    try:
        n, interval_ms = int(n), float(interval_ms)
    except (TypeError, ValueError):
        return f"{int(statedict.get('State.Acq.nFrames') or 0)} uncaging frames"
    if n <= 1 or interval_ms <= 0:
        return f"{n} pulse" if n == 1 else f"{int(statedict.get('State.Acq.nFrames') or 0)} uncaging frames"
    return f"{n} pulses {1000.0 / interval_ms:.3g} Hz"


def two_line_title(name: str) -> str:
    """'<condition>, <protocol>' as two lines for panel titles (other names unchanged)."""
    return name.replace(PROTOCOL_SEP, "\n", 1) if PROTOCOL_SEP in name else name


def protocol_condition_order(condition_order: Sequence[str], summary_df: pd.DataFrame) -> list[str]:
    """'<condition>, <protocol>': condition first, then protocol (in order of first acquisition)."""
    if "uncaging_protocol" not in summary_df.columns or summary_df.empty:
        return list(condition_order)
    first = summary_df.sort_values("acq_time_str").drop_duplicates("uncaging_protocol")
    protocols = first["uncaging_protocol"].astype(str).tolist()
    return [f"{c}{PROTOCOL_SEP}{p}" for c in condition_order for p in protocols]


def protocol_stat_subsets(group_order: Sequence[str], split: bool) -> list[tuple[str, list[str]]]:
    """(file suffix, groups) compared together: one subset per protocol when split."""
    if not split:
        return [("", list(group_order))]
    out: dict[str, list[str]] = {}
    for g in group_order:
        protocol = g.split(PROTOCOL_SEP, 1)[1] if PROTOCOL_SEP in g else ""
        out.setdefault(protocol, []).append(g)
    safe = lambda t: "".join(ch if ch.isalnum() else "_" for ch in t).strip("_")  # noqa: E731
    return [(f"_{safe(p)}" if p else "", gs) for p, gs in out.items()]


def signal_columns(signal: str, ch_1or2: int) -> tuple[str, str, str]:
    """(time-course column, summary column, y label) of the plotted quantity."""
    if signal == "intensity":
        return (f"Spine_Ch{ch_1or2}_intensity_normalized", f"delta_FF0_intensity_ch{ch_1or2}",
                r"$\Delta$spine volume (a.u.)")
    if signal == "lifetime":
        return (f"Spine_Ch{ch_1or2}_lifetime_normalized", f"delta_lifetime_ch{ch_1or2}",
                r"$\Delta$lifetime (ns)")
    raise ValueError(f"signal must be one of {SIGNALS}, got {signal!r}")


def summary_folder_name(signal: str) -> str:
    """Output folder next to the pkl: intensity keeps the former 'summary'."""
    return "summary" if signal == "intensity" else f"summary_{signal}"


def padded_ylim(values, pad: float = 0.1, min_span: float = 1e-6) -> list[float]:
    """[min, max] of finite values widened by pad * span on both sides."""
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if vals.empty:
        return [-0.1, 0.1]
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, min_span)
    return [lo - pad * span, hi + pad * span]


def mean_sem_color_by_condition(
    conditions: Sequence[str],
    style_colors: Sequence[str],
    fallback: str = "k",
) -> dict[str, str]:
    """Assign mean-line colors in list order. Colors past len(conditions) are ignored."""
    palette = [str(color) for color in style_colors]
    assigned: dict[str, str] = {}
    for index, name in enumerate(conditions):
        if index < len(palette):
            assigned[str(name)] = palette[index]
        elif palette:
            assigned[str(name)] = palette[index % len(palette)]
        else:
            assigned[str(name)] = fallback
    return assigned


def lighten_mpl_color(color: str, lightness: float) -> tuple[float, float, float]:
    """Blend a matplotlib color toward white. lightness 0 is unchanged, 1 is white."""
    rgb = matplotlib.colors.to_rgb(color)
    amount = float(lightness)
    if amount < 0.0:
        amount = 0.0
    if amount > 1.0:
        amount = 1.0
    return tuple((1.0 - amount) * channel + amount for channel in rgb)


def uncaging_frame_period_sec(statedict: Mapping) -> float:
    """Saved-frame period in seconds from the acquisition header."""
    ms_per_line = float(statedict["State.Acq.msPerLine"])
    if statedict.get("State.Acq.fastZScan"):
        fast_ms = statedict.get("State.Acq.FastZ_msPerLine")
        if fast_ms is not None:
            ms_per_line = float(fast_ms)
    lines = float(statedict["State.Acq.linesPerFrame"])
    if (not statedict.get("State.Acq.BiDirectionalScanY", True)) and float(
        statedict.get("State.Acq.SkipFirstLines") or 0
    ) > 0:
        lines += float(statedict["State.Acq.SkipFirstLines"])
    return lines * ms_per_line / 1000.0


def uncaging_pulse_times_aligned_sec(
    statedict: Mapping,
    zero_frame_index: int,
) -> np.ndarray:
    """Laser-on times in seconds on the aligned_time_sec axis.

    Times follow FLIMage's imaging-uncaging waveform:
    baselineBeforeTrain_forFrame, then pulseDelay, repeated every
    pulseSetInterval_forFrame. zero_frame_index is the 0-based frame whose
    timestamp is aligned_time_sec = 0.
    """
    baseline_ms = float(statedict["State.Uncaging.baselineBeforeTrain_forFrame"])
    pulse_delay_ms = float(statedict.get("State.Uncaging.pulseDelay") or 0)
    interval_ms = float(statedict["State.Uncaging.pulseSetInterval_forFrame"])
    pulse_isi_ms = float(statedict.get("State.Uncaging.pulseISI") or 0)
    n_trains = int(statedict["State.Uncaging.trainRepeat"])
    n_pulses = int(statedict["State.Uncaging.nPulses"])
    if n_trains < 1 or n_pulses < 1 or interval_ms <= 0:
        return np.array([], dtype=float)
    zero_sec = int(zero_frame_index) * uncaging_frame_period_sec(statedict)
    times = [
        (baseline_ms + pulse_delay_ms + pulse * pulse_isi_ms + train * interval_ms) / 1000.0
        - zero_sec
        for train in range(n_trains)
        for pulse in range(n_pulses)
    ]
    return np.asarray(times, dtype=float)


def mark_uncaging_pulses(ax, pulse_times_sec) -> None:
    """Draw one point per uncaging pulse. Does not extend the x-axis."""
    times = np.asarray(list(pulse_times_sec), dtype=float)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return
    times = np.unique(np.round(times, 4))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y = ylim[0] + (ylim[1] - ylim[0]) * 0.9
    ax.plot(
        times,
        np.full(times.shape, y),
        linestyle="None",
        marker="o",
        color="k",
        markersize=3,
        zorder=4,
    )
    ax.annotate(
        "uncaging",
        xy=(float(times[0]), y),
        xytext=(0, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    ax.set_xlim(xlim)


def pulse_times_for_plot(plot_df: pd.DataFrame, pulse_times_by_file: Mapping[str, np.ndarray]) -> np.ndarray:
    """Unique pulse times for the files drawn on one transient panel."""
    if plot_df.empty or "file_path" not in plot_df.columns:
        return np.array([], dtype=float)
    chunks = []
    for path in plot_df["file_path"].dropna().unique():
        times = pulse_times_by_file.get(path)
        if times is None:
            continue
        chunks.append(np.asarray(times, dtype=float))
    if not chunks:
        return np.array([], dtype=float)
    return np.unique(np.round(np.concatenate(chunks), 4))


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


def pin_last_pre_at_uncaging(
    df: pd.DataFrame,
    time_col: str = "aligned_time_sec",
) -> pd.DataFrame:
    """Put each set's final pre acquisition on the uncaging bin (0 min).

    Floor bins are several minutes wide, so a frame taken seconds before
    uncaging would otherwise land on the previous bin (for example -5 min).
    That last pre is always drawn at 0 min, the left edge of the uncaging mark.
    """
    if df.empty or "phase" not in df.columns or "group_set_id" not in df.columns:
        return df
    if time_col not in df.columns:
        return df
    out = df.copy()
    for _, each_df in out.groupby("group_set_id"):
        pre = each_df[each_df["phase"] == "pre"]
        if pre.empty:
            continue
        times = pre[time_col].astype(float)
        if times.isna().all():
            continue
        last_time = float(times.max())
        if "file_path" in pre.columns:
            last_path = pre.loc[times.idxmax(), "file_path"]
            last_idx = pre.index[pre["file_path"] == last_path]
        else:
            last_idx = pre.index[np.isclose(times.to_numpy(dtype=float), last_time)]
        out.loc[last_idx, "binned_sec"] = 0.0
        out.loc[last_idx, "binned_min"] = 0.0
    return out


def assign_uncaging_frame_bins(df: pd.DataFrame, time_col: str = "aligned_time_sec") -> pd.DataFrame:
    """Every uncaging frame on its own time point instead of a minute bin.

    The frames of one protocol (n_unc_frames) are taken at the same times relative to the
    first pulse, so frame k of every set gets the mean time of frame k (binned_min, in
    minutes). The last pre of each set, otherwise pinned to 0 min (on the first pulse),
    goes to the mean time of the last pre frames. Rows set here have own_time_bin=True.
    """
    out = df.copy()
    out["own_time_bin"] = False
    if out.empty or "phase" not in out.columns or time_col not in out.columns:
        return out
    unc = out["phase"] == "unc"
    if unc.any():
        keys = [k for k in ("n_unc_frames", "slice") if k in out.columns]
        t = out.loc[unc].groupby(keys)[time_col].transform("mean")
        out.loc[unc, "binned_sec"] = t
        out.loc[unc, "binned_min"] = t / 60.0
        out.loc[unc, "own_time_bin"] = True
    last_pre = []
    for _, each_df in out.groupby("group_set_id"):
        pre = each_df[each_df["phase"] == "pre"]
        if len(pre) and pre[time_col].notna().any():
            last_pre.append(pre[time_col].astype(float).idxmax())
    if last_pre:
        t_last = float(out.loc[last_pre, time_col].mean())
        out.loc[last_pre, "binned_sec"] = t_last
        out.loc[last_pre, "binned_min"] = t_last / 60.0
        out.loc[last_pre, "own_time_bin"] = True
    return out


def exclude_uncaging_from_mean_intensity(
    df: pd.DataFrame,
    include_uncaging: bool = False,
) -> pd.DataFrame:
    """Drop phase=='unc' rows from a pre/post mean-intensity time course.

    Imaging frames are averaged (typically 3 frames) and Z-projected. Uncaging
    frames are not, so they are left off the same plot unless include_uncaging
    is True. GCaMP transient plots that use only the uncaging phase are unchanged.
    """
    if include_uncaging or df.empty or "phase" not in df.columns:
        return df
    return df.loc[df["phase"] != "unc"].copy()


def plot_df_for_binned_ltp(plot_df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Drop the t=0 bin except the pinned last-pre point.

    Uncaging frames and early post frames that floor into 0 min are omitted.
    The final pre acquisition is pinned to that same bin and kept.
    """
    if plot_df.empty or "binned_min" not in plot_df.columns:
        return plot_df.iloc[0:0].copy()
    out = plot_df.dropna(subset=[y_col, "binned_min"])
    at_zero = out["binned_min"] == 0
    if "phase" in out.columns:
        at_zero = at_zero & (out["phase"] != "pre")
    if "own_time_bin" in out.columns:
        at_zero = at_zero & ~out["own_time_bin"].fillna(False).astype(bool)  # assign_uncaging_frame_bins
    return out.loc[~at_zero].copy()


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


def drop_legend(ax) -> None:
    """Remove the axes legend (n is shown in the title instead, clear of the uncaging label)."""
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def legend_outside(ax, fontsize: float = 8) -> None:
    """Legend to the right of the axes so it never covers the uncaging label or the data."""
    ax.legend(frameon=False, fontsize=fontsize, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


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


UNCAGING_LIFETIME_BASELINES = (
    # (column template, panel title): delta lifetime of the uncaging frames (ns)
    ("transient_Spine_Ch{ch}_lifetime", "vs frames before 1st pulse"),
    ("Spine_Ch{ch}_lifetime_normalized", "vs pre mean"),
)


def uncaging_lifetime_mean_sem(unc_df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Per uncaging frame (slice): time = mean aligned_time_sec, mean, SEM and n of sets."""
    d = unc_df.dropna(subset=[y_col])
    if d.empty:
        return pd.DataFrame(columns=["slice", "time_sec", "mean", "sem", "n"])
    g = d.groupby("slice")
    out = pd.DataFrame({
        "time_sec": g["aligned_time_sec"].mean(),
        "mean": g[y_col].mean(),
        "sem": g[y_col].sem(ddof=1),
        "n": g["group_set_id"].nunique(),
    }).reset_index()
    return out


def plot_uncaging_lifetime(
    fulltimeseries_df: pd.DataFrame,
    *,
    ch_1or2: int,
    condition_group_specs: Sequence[tuple[str, str]],
    condition_mean_colors: Mapping[str, str],
    indiv_lightness: float,
    pulse_times_by_file: Mapping[str, np.ndarray],
    save_folder: str,
) -> list[str]:
    """Spine delta lifetime during the uncaging acquisition only (phase == 'unc').

    One figure per condition and uncaging power with two panels (two baselines):
    left = lifetime minus the mean of the uncaging frames before the first pulse,
    right = lifetime minus the pre-phase mean. Faint lines: each set; thick line and
    band: mean +/- SEM over sets per uncaging frame. Returns the saved paths.
    """
    unc = fulltimeseries_df[fulltimeseries_df["phase"] == "unc"]
    cols = [(tmpl.format(ch=ch_1or2), title) for tmpl, title in UNCAGING_LIFETIME_BASELINES]
    cols = [(c, t) for c, t in cols if c in unc.columns]
    if unc.empty or not cols:
        print("Uncaging lifetime plot: no uncaging rows or lifetime columns")
        return []
    # plain values: pd.concat would compare the frame attrs (arrays) and fail
    ylim = padded_ylim([v for c, _ in cols for v in unc[c].to_numpy(dtype=float)] + [0.0])
    saved = []
    for header, condition in condition_group_specs:
        cond_df = unc[unc["condition"] == condition]
        color = condition_mean_colors.get(condition, "k")
        for power in sorted(cond_df["uncaging_power_coherent_mW"].dropna().unique()):
            plot_df = cond_df[cond_df["uncaging_power_coherent_mW"] == power]
            fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.2), dpi=200, sharey=True)
            axes = np.atleast_1d(axes)
            for ax, (y_col, title) in zip(axes, cols):
                for _, s in plot_df.dropna(subset=[y_col]).groupby("group_set_id"):
                    s = s.sort_values("aligned_time_sec")
                    ax.plot(s["aligned_time_sec"], s[y_col], color=lighten_mpl_color(color, indiv_lightness),
                            lw=0.6, alpha=0.5, zorder=1)
                ms = uncaging_lifetime_mean_sem(plot_df, y_col)
                n_sets = int(plot_df.dropna(subset=[y_col])["group_set_id"].nunique())
                if len(ms):
                    ax.fill_between(ms["time_sec"], ms["mean"] - ms["sem"].fillna(0), ms["mean"] + ms["sem"].fillna(0),
                                    color=color, alpha=0.25, lw=0, zorder=2)
                    ax.plot(ms["time_sec"], ms["mean"], color=color, lw=2.5, zorder=3,
                            label=f"Mean ± SEM (n={n_sets})")
                    ax.legend(frameon=False, fontsize=7, loc="lower right")
                ax.set_ylim(ylim)
                ax.axhline(0.0, color="gray", lw=0.5, ls="--", zorder=0)
                mark_uncaging_pulses(ax, pulse_times_for_plot(plot_df, pulse_times_by_file))
                ax.set_title(title, fontsize=9)
                ax.set_xlabel("Time (sec)")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            axes[0].set_ylabel(r"$\Delta$lifetime (ns)")
            fig.suptitle(f"{header}, uncaging {power:g} mW, Spine Ch{ch_1or2}", fontsize=10)
            fig.tight_layout()
            path = os.path.join(save_folder, f"{condition}_uncaging_lifetime_{power:g}mW_lineplot.png")
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            print(f"Saved uncaging lifetime: {path}")
            saved.append(path)
            plt.show()
    return saved


def plot_swarm_panels(
    summary_df: pd.DataFrame,
    group_headers: Sequence[tuple[str, str]],
    uncaging_powers: Sequence[float],
    *,
    y_col: str,
    ylabel: str,
    ylim: Sequence[float],
    save_path: str,
) -> None:
    """Tiled swarm plots: columns = (condition, header) groups, rows = uncaging powers.

    Each panel: swarm of the sets, red mean line and "mean ± SD" next to it.
    """
    if len(uncaging_powers) == 0 or len(group_headers) == 0:
        return
    n_rows = len(uncaging_powers)
    n_cols = len(group_headers)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3 * n_rows),
        sharex=False,
        sharey=True,
    )
    axes = reshape_axes_to_2d(axes, n_rows, n_cols)
    for col_idx, (each_condition, each_header_name) in enumerate(group_headers):
        each_header_summary_df = summary_df[summary_df["condition"] == each_condition]
        for row_idx, each_uncaging_power_coherent_mW in enumerate(uncaging_powers):
            ax = axes[row_idx, col_idx]
            plot_df = each_header_summary_df[
                each_header_summary_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW
            ]
            if plot_df.empty:
                ax.axis("off")
                continue
            p = sns.swarmplot(y=y_col, data=plot_df, palette="tab10", ax=ax)
            sns.boxplot(
                showmeans=True,
                meanline=True,
                meanprops={"color": "r", "ls": "-", "lw": 1},
                medianprops={"visible": False},
                whiskerprops={"visible": False},
                zorder=10,
                y=y_col,
                data=plot_df,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p,
            )
            mean = plot_df[y_col].mean()
            std = plot_df[y_col].std()
            ax.text(0.2, mean, f"{mean:.2f} ± {std:.2f}", ha="left", va="bottom")
            ax.set_ylim(ylim)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(two_line_title(each_header_name) if row_idx == 0 else "")
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()



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
    label: str = "spine volume",
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Drop spines whose LTP value in y_col (default Δspine volume) is outside [vmin, vmax]."""
    if vmin is None and vmax is None:
        print(f"{label} extreme filter: disabled ({y_col})")
        return summary_df, fulltimeseries_df, []
    if y_col not in summary_df.columns:
        raise KeyError(f"Missing {label} column: {y_col}")

    mask = pd.Series(False, index=summary_df.index)
    if vmin is not None:
        mask |= summary_df[y_col] < vmin
    if vmax is not None:
        mask |= summary_df[y_col] > vmax
    extreme_ids = summary_df.loc[mask, "group_set_id"].tolist()
    bounds = f"[{vmin}, {vmax}]"
    print(f"Excluded sets ({label} {y_col} outside {bounds}): {extreme_ids}")
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


def inclusion_census(
    summary_df: pd.DataFrame,
    unstable_mask: pd.Series,
    y_col: str,
    vmin: float | None,
    vmax: float | None,
) -> pd.DataFrame:
    """Mark every spine included or rejected, before those rows are dropped."""
    census = summary_df.loc[:, ["group_set_id", "acq_time_str"]].copy()
    census["analysis_status"] = "included"
    census.loc[unstable_mask, "analysis_status"] = "rejected_pre_instability"
    volume_mask = pd.Series(False, index=summary_df.index)
    if vmin is not None:
        volume_mask = volume_mask | (summary_df[y_col] < vmin)
    if vmax is not None:
        volume_mask = volume_mask | (summary_df[y_col] > vmax)
    volume_mask = volume_mask & census["analysis_status"].eq("included")
    census.loc[volume_mask, "analysis_status"] = "rejected_spine_volume"
    return census.reset_index(drop=True)


def roi_gui_spine_sets(combined_df: pd.DataFrame) -> pd.DataFrame:
    """One row per spine in the quantitative-ROI table.

    A ``reject`` value of 1 means the ROI GUI rejected that spine, so
    quantification was skipped and the spine never reached the intensity table.
    """
    columns = ["group_set_id", "acq_time_str", "gui_rejected"]
    if (
        combined_df is None
        or "group" not in combined_df.columns
        or "nth_set_label" not in combined_df.columns
    ):
        return pd.DataFrame(columns=columns)
    work = combined_df.loc[combined_df["nth_set_label"] != -1]
    if work.empty:
        return pd.DataFrame(columns=columns)
    has_reject = "reject" in work.columns
    time_col = "dt_str" if "dt_str" in work.columns else None
    rows = []
    for (group, set_label), group_df in work.groupby(["group", "nth_set_label"], sort=False):
        if has_reject:
            flags = pd.to_numeric(group_df["reject"], errors="coerce").fillna(0)
            gui_rejected = bool(flags.max() >= 1)
        else:
            gui_rejected = False
        if time_col is not None and group_df[time_col].notna().any():
            acq_time = str(group_df[time_col].dropna().min())
        else:
            acq_time = ""
        rows.append(
            {
                "group_set_id": f"{group}_{set_label}",
                "acq_time_str": acq_time,
                "gui_rejected": gui_rejected,
            }
        )
    return pd.DataFrame(rows)


def apply_roi_gui_rejects(
    roi_sets: pd.DataFrame,
    analysis_census: pd.DataFrame,
) -> pd.DataFrame:
    """Keep a spine only when the ROI GUI and the later filters both keep it.

    Every other spine is ``rejected``. GUI rejects and analysis rejects stay
    in one count.
    """
    if roi_sets is None or len(roi_sets) == 0:
        out = analysis_census.loc[:, ["group_set_id", "acq_time_str"]].copy()
        out["analysis_status"] = np.where(
            analysis_census["analysis_status"].astype(str).eq("included"),
            "included",
            "rejected",
        )
        return out.reset_index(drop=True)

    later_status: dict[str, str] = {}
    later_time: dict[str, str] = {}
    if analysis_census is not None and len(analysis_census) > 0:
        for _, row in analysis_census.iterrows():
            spine_id = str(row["group_set_id"])
            later_status[spine_id] = str(row["analysis_status"])
            later_time[spine_id] = row["acq_time_str"]

    rows = []
    seen: set[str] = set()
    for _, row in roi_sets.iterrows():
        spine_id = str(row["group_set_id"])
        seen.add(spine_id)
        if bool(row["gui_rejected"]):
            status = "rejected"
        elif later_status.get(spine_id, "included") == "included":
            status = "included"
        else:
            status = "rejected"
        acq_time = row["acq_time_str"] or later_time.get(spine_id, "")
        rows.append(
            {
                "group_set_id": spine_id,
                "acq_time_str": acq_time,
                "analysis_status": status,
            }
        )
    if analysis_census is not None and len(analysis_census) > 0:
        for _, row in analysis_census.iterrows():
            spine_id = str(row["group_set_id"])
            if spine_id in seen:
                continue
            status = "included" if str(row["analysis_status"]) == "included" else "rejected"
            rows.append(
                {
                    "group_set_id": spine_id,
                    "acq_time_str": row["acq_time_str"],
                    "analysis_status": status,
                }
            )
    return pd.DataFrame(rows)


def inclusion_counts_by_incubation_bin(
    census: pd.DataFrame,
    acquisition_start: datetime.datetime,
    bin_hours: float = 2.0,
) -> pd.DataFrame:
    """Count imaged, included, and rejected spines in incubation-time bins.

    Conditions are pooled. Anything other than ``included`` is one rejected count.
    """
    if bin_hours <= 0:
        raise ValueError(f"bin_hours must be positive, got {bin_hours}")
    hours = (
        pd.to_datetime(census["acq_time_str"]) - pd.Timestamp(acquisition_start)
    ).dt.total_seconds() / 3600.0
    frame = census.copy()
    frame["bin_start_h"] = np.floor(hours / bin_hours) * bin_hours
    rows = []
    for bin_start, group in frame.groupby("bin_start_h", sort=True):
        included = int((group["analysis_status"] == "included").sum())
        imaged = int(len(group))
        rows.append(
            {
                "bin_start_h": float(bin_start),
                "bin_end_h": float(bin_start) + float(bin_hours),
                "imaged": imaged,
                "included": included,
                "rejected": imaged - included,
            }
        )
    return pd.DataFrame(rows)


def plot_inclusion_counts_by_incubation(
    census: pd.DataFrame,
    acquisition_start: datetime.datetime,
    savepath: str,
    bin_hours: float = 2.0,
    show: bool = True,
) -> pd.DataFrame:
    """Stacked counts of included and rejected spines, all conditions together."""
    counts = inclusion_counts_by_incubation_bin(
        census, acquisition_start, bin_hours=bin_hours
    )
    if counts.empty:
        return counts
    centers = (counts["bin_start_h"] + counts["bin_end_h"]) / 2.0
    labels = [
        f"{int(start)}–{int(end)}"
        for start, end in zip(counts["bin_start_h"], counts["bin_end_h"])
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    included = counts["included"].to_numpy()
    rejected = counts["rejected"].to_numpy()
    ax.bar(centers, included, width=bin_hours * 0.72, color="#4C78A8", label="Included")
    ax.bar(
        centers,
        rejected,
        width=bin_hours * 0.72,
        bottom=included,
        color="#E45756",
        label="Rejected",
    )
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Incubation time (hours)")
    ax.set_ylabel("Spines per bin")
    ax.set_title(f"Imaged spines, {bin_hours:g}-hour bins (all conditions)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved inclusion counts: {savepath}")
    print(counts.to_string(index=False))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return counts


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

    signal_columns(cfg.signal, ch_1or2)  # validates cfg.signal
    save_folder = os.path.join(os.path.dirname(df_save_path_1), summary_folder_name(cfg.signal))
    os.makedirs(save_folder, exist_ok=True)



    #%% get uncaging power
    combined_df["uncaging_power"] = np.nan
    fulltimeseries_df["uncaging_power"] = np.nan

    unc_df = combined_df[combined_df['phase'] == 'unc']
    unc_pulse_times_by_file: dict[str, np.ndarray] = {}
    unc_protocol_by_file: dict[str, str] = {}
    for each_file in unc_df["file_path"].unique():
        statedict = unc_df[unc_df["file_path"] == each_file]["statedict"].values[0]
        unc_protocol_by_file[each_file] = uncaging_protocol_label(statedict)
        uncaging_power = statedict["State.Uncaging.Power"]
        combined_df.loc[combined_df["file_path"] == each_file, "uncaging_power"] = uncaging_power
        fulltimeseries_df.loc[fulltimeseries_df["file_path"] == each_file, "uncaging_power"] = uncaging_power
        n_unc_frames = int(statedict.get("State.Acq.nFrames") or 0)
        if n_unc_frames in unc_total_frame_first_unc_dict:
            zero_frame_index = unc_total_frame_first_unc_dict[n_unc_frames] - 1
        else:
            zero_frame_index = int(statedict.get("State.Uncaging.FramesBeforeUncage") or 1) - 1
        unc_pulse_times_by_file[each_file] = uncaging_pulse_times_aligned_sec(
            statedict,
            zero_frame_index,
        )

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
            unc_files = each_df.loc[each_df["phase"] == "unc", "file_path"]
            protocol = unc_protocol_by_file.get(unc_files.iloc[0], "unknown protocol") if len(unc_files) else "unknown protocol"
            each_summary_dict["uncaging_protocol"] = protocol
            fulltimeseries_df.loc[each_df.index, "uncaging_protocol"] = protocol
            if cfg.split_by_uncaging_protocol:
                each_condition = f"{each_condition}{PROTOCOL_SEP}{protocol}"
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
            for each_ch in ["Ch1", "Ch2"]:
                col = f"Spine_{each_ch}_lifetime"
                pre_lt = pre_df_phase[col].dropna() if col in pre_df_phase.columns else pd.Series(dtype=float)
                each_summary_dict[f"pre_lifetime_range_ns_Spine_{each_ch}"] = (
                    float(pre_lt.max() - pre_lt.min()) if len(pre_lt) >= 2 else np.nan
                )

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
    if cfg.pre_lifetime_range_max_ns is not None:
        lt_col = f"pre_lifetime_range_ns_Spine_Ch{ch_1or2}"
        lt_unstable = summary_df[lt_col].gt(cfg.pre_lifetime_range_max_ns)
        print(f"Excluded sets (pre lifetime max - min > {cfg.pre_lifetime_range_max_ns} ns on {lt_col}): "
              f"{summary_df.loc[lt_unstable & ~unstable_mask, 'group_set_id'].tolist()}")
        unstable_mask = unstable_mask | lt_unstable
        unstable_ids = summary_df.loc[unstable_mask, "group_set_id"].tolist()
    analysis_census = inclusion_census(
        summary_df,
        unstable_mask,
        y_col=f"delta_FF0_intensity_ch{ch_1or2}",
        vmin=spine_volume_delta_ff0_min,
        vmax=spine_volume_delta_ff0_max,
    )
    analysis_census = apply_roi_gui_rejects(roi_gui_spine_sets(combined_df), analysis_census)
    summary_df = summary_df[~unstable_mask].reset_index(drop=True)
    fulltimeseries_df = fulltimeseries_df[~fulltimeseries_df["group_set_id"].isin(unstable_ids)].reset_index(drop=True)
    summary_df, fulltimeseries_df, _extreme_ids = exclude_extreme_spine_volume(
        summary_df,
        fulltimeseries_df,
        y_col=f"delta_FF0_intensity_ch{ch_1or2}",
        vmin=spine_volume_delta_ff0_min,
        vmax=spine_volume_delta_ff0_max,
    )
    summary_df, fulltimeseries_df, lifetime_ids = exclude_extreme_spine_volume(
        summary_df,
        fulltimeseries_df,
        y_col=f"delta_lifetime_ch{ch_1or2}",
        vmin=cfg.delta_lifetime_min,
        vmax=cfg.delta_lifetime_max,
        label="delta lifetime",
    )
    analysis_census.loc[analysis_census["group_set_id"].isin(lifetime_ids)
                        & analysis_census["analysis_status"].eq("included"),
                        "analysis_status"] = "rejected_delta_lifetime"
    fulltimeseries_df = assign_binned_min_everymin(fulltimeseries_df, time_threshold=BIN_TIME_THRESHOLD_SEC, bin_percent_median=BIN_PERCENT_MEDIAN)
    fulltimeseries_df = pin_last_pre_at_uncaging(fulltimeseries_df)
    summary_csv = os.path.join(save_folder, "summary_df.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary table: {summary_csv}")
    fulltimeseries_df.attrs["unc_pulse_times_by_file"] = unc_pulse_times_by_file
    fulltimeseries_df.attrs["analysis_census"] = analysis_census
    return summary_df, fulltimeseries_df, save_folder


def run_ltp_group_analysis(cfg: LTPGroupAnalysisConfig) -> str:
    """Prepare tables and save the standard LTP plot suite. Returns save_folder."""
    summary_df, fulltimeseries_df, save_folder = prepare_ltp_group_tables(cfg)
    unc_pulse_times_by_file = fulltimeseries_df.attrs.get("unc_pulse_times_by_file", {})
    analysis_census = fulltimeseries_df.attrs.get("analysis_census")
    prefix_map = dict(cfg.condition_prefix_map)
    prefix_names = ", ".join(prefix_map.keys())
    CONDITION_PREFIX_ORDER = list(cfg.condition_order)
    if cfg.split_by_uncaging_protocol:
        CONDITION_PREFIX_ORDER = protocol_condition_order(CONDITION_PREFIX_ORDER, summary_df)
        print("Groups split by uncaging protocol:", CONDITION_PREFIX_ORDER)
    MEAN_SEM_MEAN_COLOR = cfg.mean_sem_mean_color
    MEAN_SEM_INDIV_LIGHTNESS = cfg.mean_sem_indiv_lightness
    ch_1or2 = cfg.ch_1or2
    LTP_data_point_after_min_between = list(cfg.ltp_window_min)
    ALIGNED_TIME_XLIM_MAX_SEC = cfg.aligned_time_xlim_max_sec
    BINNED_TIME_XLIM_MIN = cfg.binned_time_xlim_min
    BINNED_TIME_XLIM_MAX = cfg.binned_time_xlim_max
    acquisiton_start_datetime_str = cfg.acquisition_start_datetime_str
    signal = cfg.signal
    signal_y, signal_summary_y, signal_ylabel = signal_columns(signal, ch_1or2)
    print(f"Plotted signal: {signal} ({signal_y}, {signal_summary_y})")

    # %% condition groups by filename prefix
    condition_group_specs = build_condition_group_specs(fulltimeseries_df, CONDITION_PREFIX_ORDER)
    condition_mean_colors = mean_sem_color_by_condition(
        [condition for _label, condition in condition_group_specs],
        cfg.condition_styles,
        fallback=MEAN_SEM_MEAN_COLOR,
    )
    print("Plot groups by filename prefix:", condition_group_specs)
    print(fulltimeseries_df.groupby("condition")["group_set_id"].nunique())
    unknown_ids = sorted(
        fulltimeseries_df.loc[fulltimeseries_df["condition"] == "unknown", "group_set_id"].dropna().unique()
    )
    if unknown_ids:
        print(f"Warning: unmatched filename prefix (not {prefix_names}): {unknown_ids}")

    # %% line plot
    #plot each data, with light thin color lines
    # Uncaging-phase rows stay in fulltimeseries_df for GCaMP transients.
    mean_intensity_df = exclude_uncaging_from_mean_intensity(
        fulltimeseries_df,
        include_uncaging=cfg.plot_uncaging_on_mean_intensity or signal == "lifetime",
    )
    if signal == "lifetime":
        # lifetime of single uncaging frames is comparable: plot every frame at its own time
        mean_intensity_df = assign_uncaging_frame_bins(mean_intensity_df)
    intensity_y = signal_y
    intensity_vals = mean_intensity_df[intensity_y].dropna()
    if signal == "intensity":
        swarm_ylim = [summary_df[signal_summary_y].min()-0.1,
                        summary_df[signal_summary_y].max()+0.1]
        main_ylim = [-0.4, swarm_ylim[1]]
        if len(intensity_vals) == 0:
            full_range_ylim = [-0.1, 0.1]
        else:
            full_range_ylim = [float(intensity_vals.min()) - 0.1, float(intensity_vals.max()) + 0.1]
    else:
        # lifetime (ns): padded data range, always including 0
        main_ylim = padded_ylim(list(intensity_vals) + [0.0])
        full_range_ylim = main_ylim
    plot_info_dict = {
        signal: {"ylabel": signal_ylabel,
                    "xlabel": "Time (sec)",
                    "x": "aligned_time_sec",
                    "y": intensity_y,
                    "errorbar": "se",
                    # "ylim": [-0.4, 5.9]
                    "ylim": main_ylim,
                    "plot_zero_line": True,
                    },
        f"{signal}_full_range": {"ylabel": signal_ylabel,
                    "xlabel": "Time (sec)",
                    "x": "aligned_time_sec",
                    "y": intensity_y, 
                    "errorbar": "se",
                    # "ylim": [-0.4, 5.9]
                    "ylim": full_range_ylim,
                    "plot_zero_line": True,
                    },    
        }

    for each_header_name, each_condition in condition_group_specs:
        eachgroup_df = mean_intensity_df[mean_intensity_df["condition"] == each_condition]
        if len(eachgroup_df) == 0:
            continue

        for each_uncaging_power_coherent_mW in mean_intensity_df["uncaging_power_coherent_mW"].unique():
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
    sorted_uncaging_powers = sorted(mean_intensity_df["uncaging_power_coherent_mW"].dropna().unique())
    valid_group_headers = []
    for each_header_name, each_condition in condition_group_specs:
        if (mean_intensity_df["condition"] == each_condition).any():
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
                eachgroup_df = mean_intensity_df[mean_intensity_df["condition"] == each_condition]
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
                        ax.set_title(two_line_title(each_header_name))
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
    mean_sem_plot_info = plot_info_dict[signal]
    # (roi title prefix, y column, y label, y limits, file name tag); lifetime: Spine and Shaft
    mean_sem_targets = [("", mean_sem_plot_info["y"], mean_sem_plot_info["ylabel"], mean_sem_plot_info["ylim"], signal)]
    if signal == "lifetime":
        shaft_y = f"DendriticShaft_Ch{ch_1or2}_lifetime_normalized"
        mean_sem_targets = [("Spine: ",) + mean_sem_targets[0][1:]]
        if shaft_y in mean_intensity_df.columns:
            mean_sem_targets.append(("Shaft: ", shaft_y, mean_sem_plot_info["ylabel"],
                                     padded_ylim(mean_intensity_df[shaft_y].tolist() + [0.0]),
                                     "lifetime_DendriticShaft"))
    for roi_title, y_col_mean_sem, ylabel_mean_sem, ylim_mean_sem, file_signal in mean_sem_targets:

        for each_header_name, each_condition in condition_group_specs:
            eachgroup_df = mean_intensity_df[mean_intensity_df["condition"] == each_condition]
            if eachgroup_df.empty:
                continue
            for each_uncaging_power_coherent_mW in sorted_uncaging_powers:
                plot_df = eachgroup_df[eachgroup_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
                if plot_df.empty:
                    continue
                fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=300)
                mean_color = condition_mean_colors[each_condition]
                n = plot_spine_volume_mean_sem(
                    ax,
                    plot_df,
                    y_col_mean_sem,
                    mean_color=mean_color,
                    indiv_color=lighten_mpl_color(mean_color, MEAN_SEM_INDIV_LIGHTNESS),
                    label=f"Mean ± SEM (n={plot_df['group_set_id'].nunique()})",
                )
                decorate_spine_volume_timecourse_ax(ax, ylim_mean_sem, ltp_window_min=LTP_data_point_after_min_between, xlim_min=BINNED_TIME_XLIM_MIN, xlim_max=BINNED_TIME_XLIM_MAX)
                ax.set_xlabel("Time (min)", fontsize=10)
                ax.set_ylabel(ylabel_mean_sem, fontsize=10)
                ax.set_title(f"{roi_title}{each_header_name}, {each_uncaging_power_coherent_mW:g} mW "
                             f"(n={plot_df['group_set_id'].nunique()})", fontsize=10)
                drop_legend(ax)
                fig.tight_layout()
                savepath = os.path.join(
                    save_folder,
                    f"{each_condition}_{file_signal}_{each_uncaging_power_coherent_mW}mW_lineplot_mean_sem.png",
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
                eachgroup_df = mean_intensity_df[mean_intensity_df["condition"] == each_condition]
                for row_idx, each_uncaging_power_coherent_mW in enumerate(sorted_uncaging_powers):
                    ax = axes[row_idx, col_idx]
                    plot_df = eachgroup_df[eachgroup_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW]
                    if plot_df.empty:
                        ax.axis("off")
                        continue
                    mean_color = condition_mean_colors[each_condition]
                    plot_spine_volume_mean_sem(
                        ax,
                        plot_df,
                        y_col_mean_sem,
                        mean_color=mean_color,
                        indiv_color=lighten_mpl_color(mean_color, MEAN_SEM_INDIV_LIGHTNESS),
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
                    n_txt = f"n={plot_df['group_set_id'].nunique()}"
                    ax.set_title(f"{roi_title}{two_line_title(each_header_name)}\n{n_txt}" if row_idx == 0 else n_txt, fontsize=9)
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
                    drop_legend(ax)
            fig.subplots_adjust(left=0.22, right=0.98, top=0.9, bottom=0.14)
            savepath = os.path.join(save_folder, f"panel_{file_signal}_lineplot_mean_sem.png")
            plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved mean±SEM tiled time course: {savepath}")
            plt.show()

        for each_uncaging_power_coherent_mW in sorted_uncaging_powers:
            fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=300)
            n_plotted = 0
            for each_header_name, each_condition in condition_group_specs:
                mean_color = condition_mean_colors[each_condition]
                plot_df = mean_intensity_df[
                    (mean_intensity_df["condition"] == each_condition)
                    & (mean_intensity_df["uncaging_power_coherent_mW"] == each_uncaging_power_coherent_mW)
                ]
                if plot_df.empty:
                    continue
                n = plot_spine_volume_mean_sem(
                    ax,
                    plot_df,
                    y_col_mean_sem,
                    mean_color=mean_color,
                    indiv_color=lighten_mpl_color(mean_color, MEAN_SEM_INDIV_LIGHTNESS),
                    label=f"{each_header_name} (n={plot_df['group_set_id'].nunique()})",
                )
                n_plotted += n
            if n_plotted == 0:
                plt.close(fig)
                continue
            decorate_spine_volume_timecourse_ax(ax, ylim_mean_sem, ltp_window_min=LTP_data_point_after_min_between, xlim_min=BINNED_TIME_XLIM_MIN, xlim_max=BINNED_TIME_XLIM_MAX)
            ax.set_xlabel("Time (min)", fontsize=10)
            ax.set_ylabel(ylabel_mean_sem, fontsize=10)
            ax.set_title(f"{roi_title}{each_uncaging_power_coherent_mW:g} mW", fontsize=10)
            fig.tight_layout()
            legend_outside(ax)  # after the layout: the axes keep their size, the saved image widens
            savepath = os.path.join(
                save_folder,
                f"panel_{file_signal}_lineplot_mean_sem_overlay_{each_uncaging_power_coherent_mW:g}mW.png",
            )
            plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved mean±SEM overlay: {savepath}")
            plt.show()

    # %% swarm plot
    swarm_min = summary_df[signal_summary_y].min()
    swarm_max = summary_df[signal_summary_y].max()
    ten_percent_ylim = (swarm_max - swarm_min) * 0.1
    swarm_ylim = [swarm_min-ten_percent_ylim, swarm_max+ten_percent_ylim]
    plot_info_dict = {
                      signal: {"ylabel": signal_ylabel,
                                    "y": signal_summary_y,
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

    for each_plot_type, each_plot_info in plot_info_dict.items():
        plot_swarm_panels(
            summary_df,
            valid_group_headers,
            sorted_uncaging_powers,
            y_col=each_plot_info["y"],
            ylabel=each_plot_info["ylabel"],
            ylim=each_plot_info["ylim"],
            save_path=os.path.join(save_folder, f"panel_{each_plot_type}_swarmplot.png"),
        )


    # %% combined paper-style swarmplot (all groups in one figure + mean difference test)
    y_col = signal_summary_y
    ylabel = signal_ylabel
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
        stat_subsets = protocol_stat_subsets(group_order, cfg.split_by_uncaging_protocol)
        for each_power, (subset_suffix, subset_groups) in [(p, sg) for p in sorted_powers for sg in stat_subsets]:
            power_df = plot_df[(plot_df["uncaging_power_coherent_mW"] == each_power)
                               & plot_df["condition"].isin(subset_groups)].copy()
            if power_df.empty:
                continue

            present_groups = [g for g in subset_groups if (power_df["condition"] == g).any()]
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
            y_span = max(y_max - y_min, 0.2 if signal == "intensity" else 0.02)
            bracket_y = y_max + 0.12 * y_span
            bracket_h = 0.05 * y_span
            add_significance_bracket(ax, 0, 1, bracket_y, bracket_h, star)

            ax.axhline(0.0, color="0.6", lw=0.7, ls="--", zorder=1)
            ax.set_xlim(-0.55, len(present_groups) - 0.45)
            ax.set_ylim(y_min - 0.12 * y_span, bracket_y + 0.22 * y_span)
            ax.set_xticks(range(len(present_groups)))
            ax.set_xticklabels(
                [f"{g.split(PROTOCOL_SEP, 1)[0] if subset_suffix else g}\n(n={n})"
                 for g, n in zip(present_groups, ns)],
                fontsize=8,
            )
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel("")
            protocol_title = present_groups[0].split(PROTOCOL_SEP, 1)[1] if subset_suffix else ""
            ax.set_title(f"{each_power:g} mW" + (f"\n{protocol_title}" if protocol_title else ""), fontsize=10)
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
                f"panel_{signal}_swarmplot_combined_stats_{each_power:g}mW{subset_suffix}.png",
            )
            plt.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
            stats_csv = os.path.join(
                save_folder,
                f"panel_{signal}_swarmplot_combined_stats_{each_power:g}mW{subset_suffix}.csv",
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

                mark_uncaging_pulses(
                    plt.gca(),
                    pulse_times_for_plot(plot_df, unc_pulse_times_by_file),
                )

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

                    mark_uncaging_pulses(
                        ax,
                        pulse_times_for_plot(plot_df, unc_pulse_times_by_file),
                    )

                    if row_idx == 0:
                        ax.set_title(two_line_title(each_header_name))
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

    # %% lifetime during the uncaging acquisition only
    if signal == "lifetime":
        plot_uncaging_lifetime(
            fulltimeseries_df,
            ch_1or2=ch_1or2,
            condition_group_specs=condition_group_specs,
            condition_mean_colors=condition_mean_colors,
            indiv_lightness=MEAN_SEM_INDIV_LIGHTNESS,
            pulse_times_by_file=unc_pulse_times_by_file,
            save_folder=save_folder,
        )


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
                        ax.set_title(two_line_title(each_header_name))
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
                        ax.set_title(two_line_title(each_header_name))
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

    if analysis_census is not None and len(analysis_census) > 0:
        # One-hour bins are sparse late in a long session, so counts use 2 hours.
        plot_inclusion_counts_by_incubation(
            analysis_census,
            acquisiton_start_datetime,
            os.path.join(save_folder, "imaged_included_rejected_vs_incubation_2h.png"),
            bin_hours=2.0,
        )

    plot_info_dict = {
        "GCaMP_LTP_level_against_vs_time_hours_incubation": 
                                {"ylabel": signal_ylabel,
                                "xlabel": "Incubation time (hours)",
                                "y": signal_summary_y,
                                "x": "time_hours_incubation",
                                "ylim": [summary_df[signal_summary_y].min() - (summary_df[signal_summary_y].max() - summary_df[signal_summary_y].min()) * 0.1,
                                         summary_df[signal_summary_y].max() + (summary_df[signal_summary_y].max() - summary_df[signal_summary_y].min()) * 0.1],
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
                        ax.set_title(two_line_title(each_header_name))
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
