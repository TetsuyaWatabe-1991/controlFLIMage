# %%
"""Serum-source Delta spine-volume swarm plots.

Base figure: Gibco, Gibco-2, Cytiva (2.7+4.0 mW pooled), Calf, Rabbit.
A separate figure also includes APV (3.9 mW from 20260508).
Saves Nature full-column (183 mm) and three-quarter-column (137 mm) figures.
"""
from __future__ import annotations

import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")

FORUSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLFLIMAGE_DIR = os.path.dirname(FORUSE_DIR)
sys.path.insert(0, CONTROLFLIMAGE_DIR)
sys.path.insert(0, FORUSE_DIR)

from ltp_group_analysis import (  # noqa: E402
    LTPGroupAnalysisConfig,
    NATURE_FULL_COLUMN_MM,
    NATURE_THREE_QUARTER_COLUMN_MM,
    combine_labeled_spine_tables,
    grouped_spine_volume_ylim,
    prepare_ltp_group_tables,
    save_grouped_spine_volume_swarm_figure,
    select_summary_spines,
)

Y_COL = "delta_FF0_intensity_ch2"
SAVE_DIR = r"G:\ImagingData\Tetsuya\20260508\combined"
SPINE_VOLUME_DELTA_FF0_MIN = -2.0
SPINE_VOLUME_DELTA_FF0_MAX = 4.0
GROUP_ORDER = [
    "Gibco",
    "Gibco-2",
    "Cytiva",
    "Calf",
    "Rabbit",
]
GROUP_ORDER_WITH_APV = [
    "Gibco",
    "Gibco-2",
    "Cytiva",
    "Calf",
    "Rabbit",
    "APV",
]
FIGURE_HEIGHT_MM = 72.0
KEEP_COLS = (
    "group_label",
    "delta_spine_volume",
    "condition",
    "uncaging_power_coherent_mW",
    "group_set_id",
    "group",
    "set_label",
)


def _base_cfg(**kwargs) -> LTPGroupAnalysisConfig:
    return LTPGroupAnalysisConfig(
        spine_volume_delta_ff0_min=SPINE_VOLUME_DELTA_FF0_MIN,
        spine_volume_delta_ff0_max=SPINE_VOLUME_DELTA_FF0_MAX,
        **kwargs,
    )


GIBCO_CFG = _base_cfg(
    df_save_path=r"G:/ImagingData/Tetsuya/20260508/auto1\combined_df_respan.pkl",
    out_csv_path=(
        r"G:/ImagingData/Tetsuya/20260508/auto1"
        r"\combined_df_respan_intensity_lifetime_all_frames.csv"
    ),
    acquisition_start_datetime_str="2026-05-08T10:30:00.000",
    condition_prefix_map={"CM_": "CM", "APV_": "APV"},
    condition_order=["CM", "APV"],
    condition_styles={
        "CM": {"indiv": "0.75", "mean": "k"},
        "APV": {"indiv": "#E8B4B0", "mean": "r"},
    },
)

GIBCO_TEST_CFG = _base_cfg(
    df_save_path=r"G:/ImagingData/Tetsuya/20260506/test_gibco_serum\combined_df_respan.pkl",
    out_csv_path=(
        r"G:/ImagingData/Tetsuya/20260506/test_gibco_serum"
        r"\combined_df_respan_intensity_lifetime_all_frames.csv"
    ),
    acquisition_start_datetime_str="2026-05-06T13:30:00.000",
    condition_prefix_map={"": "Gibco"},
    condition_order=["Gibco"],
    condition_styles={"Gibco": {"indiv": "0.75", "mean": "k"}},
)

CYTIVA_CFG = _base_cfg(
    df_save_path=r"G:/ImagingData/Tetsuya/20260430/cytiva_tdTom\combined_df_1.pkl",
    out_csv_path=(
        r"G:/ImagingData/Tetsuya/20260430/cytiva_tdTom"
        r"\combined_df_1_intensity_lifetime_all_frames.csv"
    ),
    acquisition_start_datetime_str="2026-04-30T11:30:00.000",
    condition_prefix_map={"": "Cytiva"},
    condition_order=["Cytiva"],
    condition_styles={"Cytiva": {"indiv": "0.75", "mean": "k"}},
    # Some Cytiva sets have 66 uncaging frames (two 33-frame blocks).
    unc_total_frame_first_unc_dict={33: 2, 55: 5, 66: 2, 80: 8, 144: 8},
)

RAB_CALF_CFG = _base_cfg(
    df_save_path=(
        r"G:/ImagingData/Tetsuya/20260506/RabCalf_tdTGC6s/auto1"
        r"\combined_df_respan.pkl"
    ),
    out_csv_path=(
        r"G:/ImagingData/Tetsuya/20260506/RabCalf_tdTGC6s/auto1"
        r"\combined_df_respan_intensity_lifetime_all_frames.csv"
    ),
    acquisition_start_datetime_str="2026-05-06T17:30:00.000",
    condition_prefix_map={"rab_": "Rab", "calf_": "Calf"},
    condition_order=["Rab", "Calf"],
    condition_styles={
        "Rab": {"indiv": "0.75", "mean": "k"},
        "Calf": {"indiv": "#E8B4B0", "mean": "r"},
    },
)


def _load_summary(cfg: LTPGroupAnalysisConfig):
    summary_csv = os.path.join(os.path.dirname(cfg.df_save_path), "summary", "summary_df.csv")
    if os.path.exists(summary_csv):
        print(f"Loading cached summary: {summary_csv}")
        import pandas as pd

        return pd.read_csv(summary_csv)
    summary_df, _, _ = prepare_ltp_group_tables(cfg)
    return summary_df


def _select(summary_df, label: str, *, condition: str, powers_mw: list[float]):
    selected = select_summary_spines(
        summary_df,
        y_col=Y_COL,
        condition=condition,
        powers_mw=powers_mw,
    )
    print(
        f"{label}: n={len(selected)} "
        f"condition={condition!r} powers={powers_mw}"
    )
    return label, selected


def _save_figure_set(
    plot_df: pd.DataFrame,
    *,
    group_order: list[str],
    stem: str,
) -> pd.DataFrame:
    ylim = grouped_spine_volume_ylim(plot_df, "delta_spine_volume")
    print(f"{stem} shared ylim: {ylim}")
    keep_cols = [c for c in KEEP_COLS if c in plot_df.columns]
    points_csv = os.path.join(SAVE_DIR, f"{stem}_points.csv")
    plot_df[keep_cols].to_csv(points_csv, index=False)
    full_png = os.path.join(SAVE_DIR, f"{stem}_full_column.png")
    three_q_png = os.path.join(SAVE_DIR, f"{stem}_three_quarter_column.png")
    n_groups = len(group_order)
    annotate_fs = 6.5 if n_groups >= 6 else 7.0
    annotate_fs_tq = 6.0 if n_groups >= 6 else 6.5
    stats = save_grouped_spine_volume_swarm_figure(
        plot_df,
        y_col="delta_spine_volume",
        group_order=group_order,
        save_path=full_png,
        width_mm=NATURE_FULL_COLUMN_MM,
        height_mm=FIGURE_HEIGHT_MM,
        ylim=ylim,
        point_size=5.8,
        tick_fontsize=8.0,
        ylabel_fontsize=10.0,
        annotate_fontsize=annotate_fs,
    )
    save_grouped_spine_volume_swarm_figure(
        plot_df,
        y_col="delta_spine_volume",
        group_order=group_order,
        save_path=three_q_png,
        width_mm=NATURE_THREE_QUARTER_COLUMN_MM,
        height_mm=FIGURE_HEIGHT_MM,
        ylim=ylim,
        point_size=5.2,
        tick_fontsize=7.0,
        ylabel_fontsize=9.0,
        annotate_fontsize=annotate_fs_tq,
    )
    stats_csv = os.path.join(SAVE_DIR, f"{stem}_stats.csv")
    stats.to_csv(stats_csv, index=False)
    print(stats.to_string(index=False))
    print(f"Saved points: {points_csv}")
    print(f"Saved stats: {stats_csv}")
    print(f"Saved full-column: {full_png}")
    print(f"Saved 3/4-column: {three_q_png}")
    return stats


def main() -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)

    gibco_summary = _load_summary(GIBCO_CFG)
    gibco_test_summary = _load_summary(GIBCO_TEST_CFG)
    cytiva_summary = _load_summary(CYTIVA_CFG)
    rab_calf_summary = _load_summary(RAB_CALF_CFG)

    labeled = [
        _select(gibco_summary, "Gibco", condition="CM", powers_mw=[3.9]),
        _select(
            gibco_test_summary,
            "Gibco-2",
            condition="Gibco",
            powers_mw=[3.9],
        ),
        _select(cytiva_summary, "Cytiva", condition="Cytiva", powers_mw=[2.7, 4.0]),
        _select(rab_calf_summary, "Calf", condition="Calf", powers_mw=[3.9]),
        _select(rab_calf_summary, "Rabbit", condition="Rab", powers_mw=[3.9]),
    ]
    plot_df = combine_labeled_spine_tables(labeled, Y_COL)
    _save_figure_set(
        plot_df,
        group_order=GROUP_ORDER,
        stem="serum_source_delta_spine_volume_swarm",
    )

    apv = _select(gibco_summary, "APV", condition="APV", powers_mw=[3.9])
    plot_df_apv = combine_labeled_spine_tables(
        [
            labeled[0],
            labeled[1],
            labeled[2],
            labeled[3],
            labeled[4],
            apv,
        ],
        Y_COL,
    )
    _save_figure_set(
        plot_df_apv,
        group_order=GROUP_ORDER_WITH_APV,
        stem="serum_source_delta_spine_volume_swarm_with_APV",
    )
    return SAVE_DIR


if __name__ == "__main__":
    print(main())
