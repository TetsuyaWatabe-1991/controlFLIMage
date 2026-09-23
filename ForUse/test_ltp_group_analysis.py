# -*- coding: utf-8 -*-
"""Headless checks for prefix-based LTP grouping and time binning."""

from __future__ import annotations

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ltp_group_analysis import (  # noqa: E402
    assign_binned_min_everymin,
    assign_condition,
    build_condition_group_specs,
    combine_labeled_spine_tables,
    condition_from_label,
    exclude_extreme_spine_volume,
    format_group_tick_label,
    format_mean_annotation,
    grouped_spine_volume_ylim,
    mm_to_inch,
    plot_grouped_spine_volume_swarm,
    round_uncaging_power_mw,
    select_summary_spines,
)

RAB_CALF_MAP = {"rab_": "Rab", "calf_": "Calf"}
AP5_CNT_MAP = {"ap5_": "AP5", "cnt_": "Control"}


def test_rab_calf_from_group_names() -> None:
    assert condition_from_label("calf_1_pos1__highmag_1", RAB_CALF_MAP) == "Calf"
    assert condition_from_label("Rab_10_pos1__highmag_2", RAB_CALF_MAP) == "Rab"
    assert condition_from_label("Rab_6_pos1__highmag_1", RAB_CALF_MAP) == "Rab"
    assert condition_from_label("CALF_4_pos1__highmag_5", RAB_CALF_MAP) == "Calf"


def test_rab_calf_from_file_path() -> None:
    path = r"G:/ImagingData/Tetsuya/20260506/RabCalf_tdTGC6s/auto1/Rab_8_pos1__highmag_4.flim"
    assert condition_from_label(path, RAB_CALF_MAP) == "Rab"
    path2 = r"G:\ImagingData\Tetsuya\20260506\RabCalf_tdTGC6s\auto1\calf_2_pos1__highmag_7.flim"
    assert condition_from_label(path2, RAB_CALF_MAP) == "Calf"


def test_ap5_cnt_prefixes_still_work() -> None:
    assert condition_from_label("AP5_cell1_pos1", AP5_CNT_MAP) == "AP5"
    assert condition_from_label("cnt_cell2_pos1", AP5_CNT_MAP) == "Control"
    path = r"//server/data/AP5_pos1/file.flim"
    assert condition_from_label(path, AP5_CNT_MAP) == "AP5"


def test_folder_name_rabcalf_does_not_override_filename() -> None:
    calf_path = r"G:/ImagingData/Tetsuya/20260506/RabCalf_tdTGC6s/auto1/calf_1_pos1__highmag_1.flim"
    assert condition_from_label(calf_path, RAB_CALF_MAP) == "Calf"
    rab_path = r"G:/ImagingData/Tetsuya/20260506/RabCalf_tdTGC6s/auto1/Rab_6_pos1__highmag_2.flim"
    assert condition_from_label(rab_path, RAB_CALF_MAP) == "Rab"


def test_unknown_when_no_prefix() -> None:
    assert condition_from_label("highmag_1_pos1", RAB_CALF_MAP) == "unknown"
    assert assign_condition(file_path=None, group=None, prefix_map=RAB_CALF_MAP) == "unknown"


def test_assign_condition_prefers_file_path() -> None:
    cond = assign_condition(
        file_path=r"G:/data/Rab_1_pos1__highmag_1.flim",
        group="calf_1_pos1__highmag_1",
        prefix_map=RAB_CALF_MAP,
    )
    assert cond == "Rab"


def test_build_condition_group_specs_order() -> None:
    df = pd.DataFrame({"condition": ["Calf", "Rab", "Calf", "unknown"]})
    specs = build_condition_group_specs(df, ["Rab", "Calf"])
    assert specs == [("Rab", "Rab"), ("Calf", "Calf"), ("unknown", "unknown")]


def test_assign_binned_min_everymin() -> None:
    df = pd.DataFrame(
        {
            "group_set_id": ["a"] * 6,
            "phase": ["pre", "pre", "unc", "post", "post", "post"],
            "aligned_time_sec": [-120.0, -60.0, 0.0, 60.0, 120.0, 180.0],
        }
    )
    out = assign_binned_min_everymin(df, time_threshold=5.0, bin_percent_median=0.99)
    assert "binned_min" in out.columns
    assert float(out.loc[out["aligned_time_sec"] == 180.0, "binned_min"].iloc[0]) == 3.0
    assert float(out.loc[out["aligned_time_sec"] == -120.0, "binned_min"].iloc[0]) == -2.0


def test_exclude_extreme_spine_volume() -> None:
    summary = pd.DataFrame(
        {
            "group_set_id": ["keep", "too_high", "too_low"],
            "delta_FF0_intensity_ch2": [0.5, 9.0, -3.0],
        }
    )
    ts = pd.DataFrame(
        {
            "group_set_id": ["keep", "keep", "too_high", "too_low"],
            "y": [1, 2, 3, 4],
        }
    )
    out_sum, out_ts, dropped = exclude_extreme_spine_volume(
        summary, ts, "delta_FF0_intensity_ch2", vmin=-2.0, vmax=4.0
    )
    assert dropped == ["too_high", "too_low"]
    assert list(out_sum["group_set_id"]) == ["keep"]
    assert list(out_ts["group_set_id"]) == ["keep", "keep"]


def test_exclude_extreme_spine_volume_disabled() -> None:
    summary = pd.DataFrame(
        {"group_set_id": ["a"], "delta_FF0_intensity_ch2": [99.0]}
    )
    ts = pd.DataFrame({"group_set_id": ["a"], "y": [1]})
    out_sum, out_ts, dropped = exclude_extreme_spine_volume(
        summary, ts, "delta_FF0_intensity_ch2", vmin=None, vmax=None
    )
    assert dropped == []
    assert len(out_sum) == 1
    assert len(out_ts) == 1


def test_round_uncaging_power_mw() -> None:
    assert round_uncaging_power_mw(3.91) == 3.9
    assert round_uncaging_power_mw(2.74) == 2.7
    assert round_uncaging_power_mw(4.0) == 4.0


def test_select_summary_spines_combines_powers() -> None:
    summary = pd.DataFrame(
        {
            "condition": ["Cytiva", "Cytiva", "Cytiva", "Other"],
            "uncaging_power_coherent_mW": [2.71, 3.96, 6.9, 2.7],
            "delta_FF0_intensity_ch2": [0.1, 0.2, 0.3, 0.4],
        }
    )
    out = select_summary_spines(
        summary,
        y_col="delta_FF0_intensity_ch2",
        condition="Cytiva",
        powers_mw=[2.7, 4.0],
    )
    assert len(out) == 2
    rounded = {round_uncaging_power_mw(x) for x in out["uncaging_power_coherent_mW"]}
    assert rounded == {2.7, 4.0}


def test_combine_labeled_spine_tables_and_ylim() -> None:
    a = pd.DataFrame({"delta_FF0_intensity_ch2": [1.0]})
    b = pd.DataFrame({"delta_FF0_intensity_ch2": [2.0, 3.0]})
    out = combine_labeled_spine_tables(
        [("Gibco", a), ("Cytiva", b)],
        "delta_FF0_intensity_ch2",
    )
    assert list(out["group_label"]) == ["Gibco", "Cytiva", "Cytiva"]
    assert list(out["delta_spine_volume"]) == [1.0, 2.0, 3.0]
    lo, hi = grouped_spine_volume_ylim(out, "delta_spine_volume")
    assert lo <= 0.0
    assert hi > 3.0


def test_combine_empty_group_raises() -> None:
    empty = pd.DataFrame({"delta_FF0_intensity_ch2": []})
    try:
        combine_labeled_spine_tables([("x", empty)], "delta_FF0_intensity_ch2")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty group")


def test_mm_to_inch() -> None:
    assert abs(mm_to_inch(25.4) - 1.0) < 1e-9
    assert abs(mm_to_inch(183.0) - 7.204724409448819) < 1e-9


def test_format_group_tick_label() -> None:
    assert format_group_tick_label("Gibco", 13) == "Gibco\n(n=13)"
    assert format_group_tick_label("Gibco-2", 7, show_n=False) == "Gibco-2"
    assert format_group_tick_label("Calf", 19, show_n=False) == "Calf"


def test_format_mean_annotation() -> None:
    assert format_mean_annotation(0.191) == "0.19"
    assert format_mean_annotation(0.191, 0.36) == "0.19 ± 0.36"


def test_plot_grouped_spine_volume_swarm_is_seaborn_swarm() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PathCollection
    from custom_plot import plt

    plot_df = pd.DataFrame(
        {
            "group_label": ["Gibco"] * 6 + ["Gibco-2"] * 6,
            "delta_spine_volume": [0.1, 0.2, 0.0, -0.1, 0.3, 0.15, 0.4, 0.5, 0.2, 0.1, 0.35, 0.25],
        }
    )
    fig, ax = plt.subplots()
    stats = plot_grouped_spine_volume_swarm(
        plot_df,
        y_col="delta_spine_volume",
        group_order=["Gibco", "Gibco-2"],
        ax=ax,
        ylim=(-0.5, 2.0),
        annotate_y=1.8,
        show_n=False,
    )
    texts = [t.get_text() for t in ax.texts]
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    plt.close(fig)
    assert list(stats["group_label"]) == ["Gibco", "Gibco-2"]
    assert "std" in stats.columns
    assert any(isinstance(artist, PathCollection) for artist in ax.collections)
    assert texts == [
        format_mean_annotation(float(stats.loc[0, "mean"]), float(stats.loc[0, "std"])),
        format_mean_annotation(float(stats.loc[1, "mean"]), float(stats.loc[1, "std"])),
    ]
    assert all("±" in t for t in texts)
    assert tick_labels == ["Gibco", "Gibco-2"]
    assert all("(n=" not in t for t in tick_labels)


def main() -> int:
    tests = [
        test_rab_calf_from_group_names,
        test_rab_calf_from_file_path,
        test_folder_name_rabcalf_does_not_override_filename,
        test_ap5_cnt_prefixes_still_work,
        test_unknown_when_no_prefix,
        test_assign_condition_prefers_file_path,
        test_build_condition_group_specs_order,
        test_assign_binned_min_everymin,
        test_exclude_extreme_spine_volume,
        test_exclude_extreme_spine_volume_disabled,
        test_round_uncaging_power_mw,
        test_select_summary_spines_combines_powers,
        test_combine_labeled_spine_tables_and_ylim,
        test_combine_empty_group_raises,
        test_mm_to_inch,
        test_format_group_tick_label,
        test_format_mean_annotation,
        test_plot_grouped_spine_volume_swarm_is_seaborn_swarm,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL: {fn.__name__}: {exc}")
    print(f"Done: {len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
