# -*- coding: utf-8 -*-
"""
Side-by-side swarmplots (manual vs auto ROI) matching LTP analysis style.

Equivalent to *_intensity_{power}mW_plot_swarmplot.png in 20260701_tdTom_LTPanalysis_10h_bin.py.

Usage:
    python plot_auto_manual_swarmplot_respan.py --power-mw 4.0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

controlFLIMage_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(controlFLIMage_DIR)
sys.path.append(os.path.join(controlFLIMage_DIR, "ForUse"))

from custom_plot import plt as _plt  # noqa: F401  # side-effect: matplotlib style
from flim_summarize_func import build_group_header_combined, df_matches_group_headers

LTP_DATA_POINT_AFTER_MIN_BETWEEN = [25, 35]
CH_1OR2 = 2
UNC_TOTAL_FRAME_FIRST_UNC_DICT = {33: 2, 55: 5}
PRE_INSTABILITY_MAX_MIN_RATIO_THRESHOLD = 1.5
FROM_THORLAB_TO_COHERENT_FACTOR = 1 / 3
ROI_NAME_LIST = ["Spine", "DendriticShaft"]
GROUP_HEADER_DICT = {"": "Carbogen"}
GROUP_HEADER_COMBINED = build_group_header_combined(GROUP_HEADER_DICT)
Y_COL = f"delta_FF0_intensity_ch{CH_1OR2}"
Y_LABEL = r"$\Delta$spine volume (a.u.)"


def _add_uncaging_power_coherent_mw(
    fulltimeseries_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    powermeter_folder: str,
) -> pd.DataFrame:
    """Attach uncaging_power_coherent_mW (same logic as LTP analysis script)."""
    out = fulltimeseries_df.copy()
    out["uncaging_power"] = np.nan
    out["uncaging_power_coherent_mW"] = np.nan

    unc_df = combined_df[combined_df["phase"] == "unc"]
    for each_file in unc_df["file_path"].unique():
        statedict = unc_df[unc_df["file_path"] == each_file]["statedict"].values[0]
        uncaging_power = statedict["State.Uncaging.Power"]
        out.loc[out["file_path"] == each_file, "uncaging_power"] = uncaging_power

    list_of_uncaging_power = list(
        combined_df["uncaging_power"].dropna().unique()[combined_df["uncaging_power"].dropna().unique() > 0]
    )
    if not list_of_uncaging_power:
        return out

    earliest_acq_time = datetime.strptime(
        out["acq_time_str"].min(), "%Y-%m-%dT%H:%M:%S.%f"
    )
    powermeter_ini_files = glob.glob(os.path.join(powermeter_folder, "*.json")) + glob.glob(
        os.path.join(powermeter_folder, "old/*.json")
    )
    datetime_only_arr = np.array(
        [int(os.path.basename(each_file).replace(".json", "")) for each_file in powermeter_ini_files]
    )
    latest_basename = (
        f"{datetime_only_arr[datetime_only_arr < int(earliest_acq_time.strftime('%Y%m%d%H%M'))].max()}.json"
    )
    latest_files = [
        each_file for each_file in powermeter_ini_files if os.path.basename(each_file) == latest_basename
    ]
    if len(latest_files) != 1:
        raise FileNotFoundError(f"Powermeter json not found: {latest_files}")
    with open(latest_files[0], "r") as fh:
        powermeter_calib = json.load(fh)
    x_percent = np.array(list(powermeter_calib["Laser2"].keys())).astype(float)
    y_mw = np.array(list(powermeter_calib["Laser2"].values())).astype(float)
    power_slope, power_intercept = np.polyfit(x_percent, y_mw, 1)
    power_percent_to_coherent_mW_dict = {
        int(p): round(float(power_slope * p + power_intercept) * FROM_THORLAB_TO_COHERENT_FACTOR, 1)
        for p in list_of_uncaging_power
    }

    for each_file in unc_df["file_path"].unique():
        statedict = unc_df[unc_df["file_path"] == each_file]["statedict"].values[0]
        uncaging_power = int(statedict["State.Uncaging.Power"])
        mw = power_percent_to_coherent_mW_dict[uncaging_power]
        out.loc[out["file_path"] == each_file, "uncaging_power_coherent_mW"] = mw

    return out


def build_ltp_summary_df(fulltimeseries_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-set summary with delta_FF0 (matches LTP analysis script)."""
    df = fulltimeseries_df.copy()
    for roi in ROI_NAME_LIST:
        for ch in ["Ch1", "Ch2"]:
            df[f"{roi}_{ch}_intensity_div_by_nAve"] = (
                df[f"{roi}_{ch}_intensity"] / df["nAveFrame"]
            )

    df["aligned_time_sec"] = np.nan
    summary_rows: list[dict] = []

    for each_group in df["group"].unique():
        for each_set_label in df[df["group"] == each_group]["set_label"].unique():
            group_set_id = f"{each_group}_{each_set_label}"
            each_df = df[(df["group"] == each_group) & (df["set_label"] == each_set_label)].copy()
            df.loc[each_df.index, "group_set_id"] = group_set_id

            length_of_unc_df = len(each_df[each_df["phase"] == "unc"])
            if length_of_unc_df not in UNC_TOTAL_FRAME_FIRST_UNC_DICT:
                raise ValueError(f"Unsupported unc length {length_of_unc_df} for {group_set_id}")
            unc_trigger_time = each_df[each_df["phase"] == "unc"]["elapsed_time_sec"].iloc[
                UNC_TOTAL_FRAME_FIRST_UNC_DICT[length_of_unc_df] - 1
            ]
            aligned_values = each_df["elapsed_time_sec"].values - unc_trigger_time
            each_df["aligned_time_sec"] = aligned_values
            df.loc[each_df.index, "aligned_time_sec"] = aligned_values

            row: dict = {
                "group": each_group,
                "set_label": each_set_label,
                "group_set_id": group_set_id,
            }
            unc_rows = each_df[each_df["phase"] == "unc"]
            if "uncaging_power_coherent_mW" in each_df.columns:
                pow_vals = unc_rows["uncaging_power_coherent_mW"].dropna().unique()
                row["uncaging_power_coherent_mW"] = pow_vals[0] if len(pow_vals) else np.nan

            for each_ch in ["Ch1", "Ch2"]:
                row[f"{each_ch}_pre_intensity"] = each_df[each_df["phase"] == "pre"][
                    f"Spine_{each_ch}_intensity"
                ].mean()
                post_ltp = each_df[
                    (each_df["phase"] == "post")
                    & (each_df["aligned_time_sec"] > LTP_DATA_POINT_AFTER_MIN_BETWEEN[0] * 60)
                    & (each_df["aligned_time_sec"] < LTP_DATA_POINT_AFTER_MIN_BETWEEN[1] * 60)
                ]
                if len(post_ltp) > 0:
                    row[f"{each_ch}_post_intensity"] = post_ltp[f"Spine_{each_ch}_intensity"].mean()
                else:
                    post_fallback = each_df[
                        (each_df["phase"] == "post")
                        & (each_df["aligned_time_sec"] > LTP_DATA_POINT_AFTER_MIN_BETWEEN[0] * 60)
                    ]
                    row[f"{each_ch}_post_intensity"] = (
                        post_fallback[f"Spine_{each_ch}_intensity"].mean()
                        if len(post_fallback) > 0
                        else np.nan
                    )

            row["delta_FF0_intensity_ch1"] = (
                row["Ch1_post_intensity"] / row["Ch1_pre_intensity"] - 1
            )
            row["delta_FF0_intensity_ch2"] = (
                row["Ch2_post_intensity"] / row["Ch2_pre_intensity"] - 1
            )

            pre_df_phase = each_df[each_df["phase"] == "pre"]
            for roi in ROI_NAME_LIST:
                for ch in ["Ch1", "Ch2"]:
                    col = f"{roi}_{ch}_intensity_div_by_nAve"
                    pre_vals = pre_df_phase[col].dropna()
                    if len(pre_vals) >= 2 and pre_vals.min() > 0:
                        row[f"pre_max_min_ratio_{roi}_{ch}"] = pre_vals.max() / pre_vals.min()
                    else:
                        row[f"pre_max_min_ratio_{roi}_{ch}"] = np.nan

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    ratio_cols = [c for c in summary_df.columns if c.startswith("pre_max_min_ratio_")]
    ratio_cols_ch = [c for c in ratio_cols if f"Ch{CH_1OR2}" in c]
    if ratio_cols_ch:
        unstable = summary_df[ratio_cols_ch].gt(PRE_INSTABILITY_MAX_MIN_RATIO_THRESHOLD).any(axis=1)
        summary_df = summary_df[~unstable].reset_index(drop=True)
    return summary_df


def _draw_swarm_ax(ax: plt.Axes, plot_df: pd.DataFrame, title: str, ylim: list[float]) -> None:
    p = sns.swarmplot(y=Y_COL, data=plot_df, palette="tab10", ax=ax)
    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={"color": "r", "ls": "-", "lw": 1},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=10,
        y=Y_COL,
        data=plot_df,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=p,
    )
    mean = plot_df[Y_COL].mean()
    std = plot_df[Y_COL].std()
    ax.text(0.2, mean, f"{mean:.2f} ± {std:.2f}", ha="left", va="bottom")
    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])


def plot_side_by_side_swarm(
    manual_summary: pd.DataFrame,
    auto_summary: pd.DataFrame,
    *,
    header_name: str,
    power_mw: float,
    out_path: str,
) -> str:
    manual_plot = manual_summary[
        manual_summary["uncaging_power_coherent_mW"] == power_mw
    ].copy()
    auto_plot = auto_summary[auto_summary["uncaging_power_coherent_mW"] == power_mw].copy()

    if manual_plot.empty and auto_plot.empty:
        raise ValueError(f"No data at {power_mw} mW")

    all_vals = pd.concat([manual_plot[Y_COL], auto_plot[Y_COL]], ignore_index=True).dropna()
    swarm_min, swarm_max = float(all_vals.min()), float(all_vals.max())
    pad = (swarm_max - swarm_min) * 0.1
    ylim = [swarm_min - pad, swarm_max + pad]

    fig, axes = plt.subplots(1, 2, figsize=(4.5, 3.5), sharey=True)
    _draw_swarm_ax(
        axes[0],
        manual_plot,
        f"Manual ROI\n{header_name}, uncaging {power_mw} mW (n={len(manual_plot)})",
        ylim,
    )
    _draw_swarm_ax(
        axes[1],
        auto_plot,
        f"Auto ROI\n{header_name}, uncaging {power_mw} mW (n={len(auto_plot)})",
        ylim,
    )
    axes[1].set_ylabel("")
    fig.suptitle(f"delta_FF0 Ch{CH_1OR2} (post 25-35 min / pre)", fontsize=11, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual vs auto ROI swarmplot (side by side)")
    parser.add_argument(
        "--df-path",
        default=r"G:\ImagingData\Tetsuya\20260701\auto1\combined_df_respan.pkl",
    )
    parser.add_argument("--power-mw", type=float, default=4.0)
    parser.add_argument(
        "--powermeter-folder",
        default=r"//RY-LAB-WS04/Users/yasudalab/Documents/Tetsuya_Imaging/powermeter",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(args.df_path)
    out_dir = args.out_dir or os.path.join(base_dir, "compare")
    manual_csv = args.df_path.replace(".pkl", "_intensity_lifetime_all_frames.csv")
    auto_csv = args.df_path.replace(".pkl", "_intensity_lifetime_all_frames_AUTO_roi.csv")

    combined_df = pd.read_pickle(args.df_path)
    combined_df = combined_df.copy()
    combined_df["uncaging_power"] = np.nan
    unc_rows = combined_df[combined_df["phase"] == "unc"]
    for each_file in unc_rows["file_path"].unique():
        statedict = unc_rows[unc_rows["file_path"] == each_file]["statedict"].values[0]
        combined_df.loc[combined_df["file_path"] == each_file, "uncaging_power"] = statedict[
            "State.Uncaging.Power"
        ]

    manual_ts = _add_uncaging_power_coherent_mw(
        pd.read_csv(manual_csv), combined_df, args.powermeter_folder
    )
    auto_ts = _add_uncaging_power_coherent_mw(
        pd.read_csv(auto_csv), combined_df, args.powermeter_folder
    )

    manual_summary = build_ltp_summary_df(manual_ts)
    auto_summary = build_ltp_summary_df(auto_ts)

    power_tag = f"{args.power_mw:.1f}"
    for each_header_name, header_list in GROUP_HEADER_COMBINED.items():
        manual_sub = df_matches_group_headers(manual_summary, header_list)
        auto_sub = df_matches_group_headers(auto_summary, header_list)
        if manual_sub.empty and auto_sub.empty:
            continue
        out_name = f"intensity_{power_tag}mW_auto_vs_manual_swarmplot.png"
        out_path = os.path.join(out_dir, out_name)
        saved = plot_side_by_side_swarm(
            manual_sub,
            auto_sub,
            header_name=each_header_name or "all",
            power_mw=args.power_mw,
            out_path=out_path,
        )
        print(f"Saved: {saved}")
        print(f"  Manual n={len(manual_sub[manual_sub.uncaging_power_coherent_mW==args.power_mw])}, "
              f"Auto n={len(auto_sub[auto_sub.uncaging_power_coherent_mW==args.power_mw])}")


if __name__ == "__main__":
    main()
