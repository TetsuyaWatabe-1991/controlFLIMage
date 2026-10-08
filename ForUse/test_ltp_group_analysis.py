# -*- coding: utf-8 -*-
"""Headless checks for prefix-based LTP grouping and time binning."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ltp_group_analysis import (  # noqa: E402
    assign_binned_min_everymin,
    assign_condition,
    build_condition_group_specs,
    exclude_uncaging_from_mean_intensity,
    lighten_mpl_color,
    mean_sem_color_by_condition,
    pin_last_pre_at_uncaging,
    uncaging_pulse_times_aligned_sec,
    plot_df_for_binned_ltp,
    combine_labeled_spine_tables,
    condition_from_label,
    exclude_extreme_spine_volume,
    apply_roi_gui_rejects,
    inclusion_census,
    inclusion_counts_by_incubation_bin,
    format_group_tick_label,
    format_mean_annotation,
    grouped_spine_volume_ylim,
    mm_to_inch,
    padded_ylim,
    plot_grouped_spine_volume_swarm,
    round_uncaging_power_mw,
    select_summary_spines,
    signal_columns,
    summary_folder_name,
    uncaging_lifetime_mean_sem,
    uncaging_protocol_label,
    protocol_condition_order,
    protocol_stat_subsets,
    two_line_title,
    assign_uncaging_frame_bins,
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


def test_pin_last_pre_at_uncaging() -> None:
    df = pd.DataFrame(
        {
            "group_set_id": ["a", "a", "a", "a"],
            "phase": ["pre", "pre", "unc", "post"],
            "file_path": ["pre_early", "pre_last", "unc1", "post1"],
            "aligned_time_sec": [-620.0, -18.0, 0.0, 120.0],
            "binned_min": [-15.0, -5.0, 0.0, 0.0],
            "binned_sec": [-900.0, -300.0, 0.0, 0.0],
            "y": [1.0, 2.0, 9.0, 3.0],
        }
    )
    pinned = pin_last_pre_at_uncaging(df)
    last = pinned[pinned["file_path"] == "pre_last"].iloc[0]
    early = pinned[pinned["file_path"] == "pre_early"].iloc[0]
    assert float(last["binned_min"]) == 0.0
    assert float(early["binned_min"]) == -15.0
    kept = plot_df_for_binned_ltp(pinned, "y")
    assert set(kept["file_path"]) == {"pre_early", "pre_last"}


def test_mean_sem_colors_ignore_extra_entries() -> None:
    colors = mean_sem_color_by_condition(
        ["DMSO", "A4", "A6"],
        ["k", "b", "g", "r", "m"],
        fallback="c",
    )
    assert colors == {"DMSO": "k", "A4": "b", "A6": "g"}
    faded = lighten_mpl_color("k", 1.0)
    assert faded == (1.0, 1.0, 1.0)
    same = lighten_mpl_color("r", 0.0)
    assert same[0] == 1.0 and same[1] == 0.0 and same[2] == 0.0


def test_uncaging_pulse_times_from_80frame_header() -> None:
    # 20260925 uncaging file: 80 frames, 0.256 s/frame, 30 pulses every 512 ms.
    # aligned_time_sec = 0 is frame index 7, the last baseline frame.
    statedict = {
        "State.Acq.msPerLine": 2,
        "State.Acq.linesPerFrame": 128,
        "State.Acq.fastZScan": False,
        "State.Acq.BiDirectionalScanY": False,
        "State.Acq.SkipFirstLines": 0,
        "State.Uncaging.baselineBeforeTrain_forFrame": 2048,
        "State.Uncaging.pulseDelay": 8,
        "State.Uncaging.pulseSetInterval_forFrame": 512,
        "State.Uncaging.pulseISI": 10,
        "State.Uncaging.trainRepeat": 30,
        "State.Uncaging.nPulses": 1,
    }
    times = uncaging_pulse_times_aligned_sec(statedict, zero_frame_index=7)
    assert len(times) == 30
    assert abs(float(times[0]) - 0.264) < 1e-6
    assert abs(float(times[-1]) - 15.112) < 1e-6
    assert float(times[-1]) < 20.0


def test_inclusion_counts_pool_conditions_in_two_hour_bins() -> None:
    import datetime

    summary = pd.DataFrame(
        {
            "group_set_id": ["a", "b", "c", "d"],
            "acq_time_str": [
                "2026-09-25T16:40:00.000",
                "2026-09-25T17:10:00.000",
                "2026-09-25T19:30:00.000",
                "2026-09-25T19:50:00.000",
            ],
            "delta_FF0_intensity_ch2": [1.0, 0.5, 9.0, 0.2],
        }
    )
    unstable = pd.Series([False, True, False, False])
    census = inclusion_census(
        summary,
        unstable,
        y_col="delta_FF0_intensity_ch2",
        vmin=-2.0,
        vmax=6.0,
    )
    assert list(census["analysis_status"]) == [
        "included",
        "rejected_pre_instability",
        "rejected_spine_volume",
        "included",
    ]
    counts = inclusion_counts_by_incubation_bin(
        census,
        datetime.datetime(2026, 9, 25, 14, 30),
        bin_hours=2.0,
    )
    first = counts.iloc[0]
    assert first["bin_start_h"] == 2.0
    assert first["imaged"] == 2
    assert first["included"] == 1
    assert first["rejected"] == 1
    second = counts.iloc[1]
    assert second["imaged"] == second["included"] + second["rejected"]
    assert int(counts["imaged"].sum()) == 4
    assert "rejected_pre_instability" not in counts.columns


def test_roi_gui_rejects_count_as_rejected_without_a_reason_split() -> None:
    import datetime

    analysis = pd.DataFrame(
        {
            "group_set_id": ["kept", "unstable"],
            "acq_time_str": [
                "2026-09-25T16:40:00.000",
                "2026-09-25T16:50:00.000",
            ],
            "analysis_status": ["included", "rejected_pre_instability"],
        }
    )
    roi = pd.DataFrame(
        {
            "group_set_id": ["kept", "unstable", "gui"],
            "acq_time_str": [
                "2026-09-25T16:40:00.000",
                "2026-09-25T16:50:00.000",
                "2026-09-25T17:00:00.000",
            ],
            "gui_rejected": [False, False, True],
        }
    )
    merged = apply_roi_gui_rejects(roi, analysis)
    assert list(merged["analysis_status"]) == ["included", "rejected", "rejected"]
    counts = inclusion_counts_by_incubation_bin(
        merged,
        datetime.datetime(2026, 9, 25, 14, 30),
        bin_hours=2.0,
    )
    assert int(counts["imaged"].sum()) == 3
    assert int(counts["included"].sum()) == 1
    assert int(counts["rejected"].sum()) == 2


def test_exclude_uncaging_from_mean_intensity() -> None:
    df = pd.DataFrame(
        {
            "phase": ["pre", "unc", "unc", "post"],
            "Spine_Ch2_intensity_normalized": [0.0, 8.0, 9.0, 0.4],
        }
    )
    dropped = exclude_uncaging_from_mean_intensity(df, include_uncaging=False)
    assert list(dropped["phase"]) == ["pre", "post"]
    kept = exclude_uncaging_from_mean_intensity(df, include_uncaging=True)
    assert len(kept) == 4
    assert exclude_uncaging_from_mean_intensity(df.iloc[0:0]).empty


def test_signal_columns_and_folder() -> None:
    assert signal_columns("intensity", 2) == (
        "Spine_Ch2_intensity_normalized", "delta_FF0_intensity_ch2", r"$\Delta$spine volume (a.u.)")
    assert signal_columns("lifetime", 1) == (
        "Spine_Ch1_lifetime_normalized", "delta_lifetime_ch1", r"$\Delta$lifetime (ns)")
    assert summary_folder_name("intensity") == "summary"
    assert summary_folder_name("lifetime") == "summary_lifetime"
    try:
        signal_columns("volume", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown signal accepted")


def test_padded_ylim() -> None:
    lo, hi = padded_ylim([0.0, 0.2, None])
    assert abs(lo + 0.02) < 1e-12 and abs(hi - 0.22) < 1e-12, (lo, hi)
    assert padded_ylim([]) == [-0.1, 0.1]


def test_exclude_delta_lifetime() -> None:
    summary = pd.DataFrame({"group_set_id": ["a", "b", "c"], "delta_lifetime_ch1": [0.05, 0.4, -0.3]})
    ts = pd.DataFrame({"group_set_id": ["a", "b", "c"], "y": [1, 2, 3]})
    out_sum, out_ts, dropped = exclude_extreme_spine_volume(
        summary, ts, "delta_lifetime_ch1", vmin=-0.2, vmax=0.3, label="delta lifetime")
    assert dropped == ["b", "c"] and list(out_ts["group_set_id"]) == ["a"]
    # disabled: the column does not even have to exist
    out_sum, _, dropped = exclude_extreme_spine_volume(summary, ts, "missing", vmin=None, vmax=None)
    assert dropped == [] and len(out_sum) == 3


def test_uncaging_lifetime_mean_sem() -> None:
    df = pd.DataFrame({
        "group_set_id": ["a", "a", "b", "b", "c"],
        "slice": [0, 1, 0, 1, 0],
        "aligned_time_sec": [-2.0, 0.0, -2.2, 0.2, -1.8],
        "y": [0.1, 0.3, 0.2, None, 0.3],
    })
    out = uncaging_lifetime_mean_sem(df, "y").set_index("slice")
    assert abs(out.loc[0, "mean"] - 0.2) < 1e-12 and int(out.loc[0, "n"]) == 3
    assert abs(out.loc[0, "time_sec"] + 2.0) < 1e-12
    assert abs(out.loc[1, "mean"] - 0.3) < 1e-12 and int(out.loc[1, "n"]) == 1  # NaN frame skipped


def test_uncaging_protocol_split() -> None:
    sd80 = {"State.Uncaging.trainRepeat": 30, "State.Uncaging.trainInterval": 512, "State.Acq.nFrames": 80}
    sd264 = {"State.Uncaging.trainRepeat": 30, "State.Uncaging.trainInterval": 2048, "State.Acq.nFrames": 264}
    assert uncaging_protocol_label(sd80) == "30 pulses 1.95 Hz"
    assert uncaging_protocol_label(sd264) == "30 pulses 0.488 Hz"
    assert uncaging_protocol_label({"State.Acq.nFrames": 33}) == "33 uncaging frames"
    summ = pd.DataFrame({"uncaging_protocol": ["30 pulses 0.488 Hz", "30 pulses 1.95 Hz", "30 pulses 1.95 Hz"],
                         "acq_time_str": ["2026-10-02T12:00", "2026-10-02T10:00", "2026-10-02T11:00"]})
    order = protocol_condition_order(["CM", "Celex"], summ)
    # condition first, then protocol: the same condition sits side by side
    assert order == ["CM, 30 pulses 1.95 Hz", "CM, 30 pulses 0.488 Hz",
                     "Celex, 30 pulses 1.95 Hz", "Celex, 30 pulses 0.488 Hz"], order
    subsets = protocol_stat_subsets(order, True)  # statistics still compare conditions within a protocol
    assert subsets == [("_30_pulses_1_95_Hz", [order[0], order[2]]),
                       ("_30_pulses_0_488_Hz", [order[1], order[3]])], subsets
    assert two_line_title("CM, 30 pulses 1.95 Hz") == "CM\n30 pulses 1.95 Hz"
    assert two_line_title("Culture media") == "Culture media"
    assert protocol_stat_subsets(["CM", "Celex"], False) == [("", ["CM", "Celex"])]


def test_assign_uncaging_frame_bins() -> None:
    rows = []
    for k, (sid, shift) in enumerate((("a", 0.0), ("b", 0.4))):
        rows += [dict(group_set_id=sid, phase="pre", aligned_time_sec=-300.0 + shift, slice=0, n_unc_frames=3,
                      binned_min=-5.0),
                 dict(group_set_id=sid, phase="pre", aligned_time_sec=-36.0 + 2 * shift, slice=0, n_unc_frames=3,
                      binned_min=0.0)]  # last pre, pinned to 0 before
        rows += [dict(group_set_id=sid, phase="unc", aligned_time_sec=t + shift, slice=i, n_unc_frames=3,
                      binned_min=0.0) for i, t in enumerate((-2.0, 0.0, 2.0))]
        rows += [dict(group_set_id=sid, phase="post", aligned_time_sec=300.0, slice=0, n_unc_frames=3, binned_min=5.0)]
    df = pd.DataFrame(rows).assign(y=1.0)
    out = assign_uncaging_frame_bins(df)
    unc = out[out.phase == "unc"].groupby("slice").binned_min.unique()
    assert [len(v) for v in unc] == [1, 1, 1]  # same frame -> same time point in both sets
    assert np.allclose([v[0] * 60 for v in unc], [-1.8, 0.2, 2.2])
    last = out[(out.phase == "pre") & (out.aligned_time_sec > -100)]
    assert np.allclose(last.binned_min * 60, -35.6) and last.own_time_bin.all()
    kept = plot_df_for_binned_ltp(out, "y")
    assert (kept.phase == "unc").sum() == 6  # frames at ~0 min are kept (own time point)
    # unchanged when frames are not given their own time point: uncaging rows in the 0 bin dropped
    assert (plot_df_for_binned_ltp(df, "y").phase == "unc").sum() == 0


def real_check(pkl: str, csv: str, ch: int = 1) -> None:
    """Run the full lifetime suite on copies of a session's tables in a temporary folder.

    The delta lifetime of every set in summary_df.csv must equal an independent recomputation
    (mean Spine lifetime 25-35 min after uncaging - mean pre lifetime), and the lifetime
    figures must be written. Nothing is written next to the session.
    """
    import shutil
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    from ltp_group_analysis import LTPGroupAnalysisConfig, run_ltp_group_analysis

    with tempfile.TemporaryDirectory() as td:
        pkl_t, csv_t = os.path.join(td, os.path.basename(pkl)), os.path.join(td, os.path.basename(csv))
        shutil.copy2(pkl, pkl_t)
        shutil.copy2(csv, csv_t)
        cfg = LTPGroupAnalysisConfig(
            df_save_path=pkl_t, out_csv_path=csv_t,
            acquisition_start_datetime_str="2026-10-02T18:00:00.000",
            condition_prefix_map={"": "all"}, condition_order=["all"], ch_1or2=ch, signal="lifetime",
        )
        folder = run_ltp_group_analysis(cfg)
        assert folder == os.path.join(td, "summary_lifetime"), folder
        summ = pd.read_csv(os.path.join(folder, "summary_df.csv"))
        q = pd.read_csv(csv)
        n_checked = 0
        for row in summ.itertuples():
            pre = row.__getattribute__(f"Ch{ch}_pre_lifetime")
            post = row.__getattribute__(f"Ch{ch}_post_lifetime")
            got = row.__getattribute__(f"delta_lifetime_ch{ch}")
            if pd.notna(pre) and pd.notna(post):
                assert abs(got - (post - pre)) < 1e-12, (row.group_set_id, got, post - pre)
                sq = q[(q["group"] == row.group) & (q["set_label"] == row.set_label) & (q["phase"] == "pre")]
                assert abs(sq[f"Spine_Ch{ch}_lifetime"].mean() - pre) < 1e-9, row.group_set_id
                n_checked += 1
        pngs = [f for f in os.listdir(folder) if f.endswith(".png")]
        lt_pngs = [f for f in pngs if "lifetime" in f]
        assert any(f.endswith("_uncaging_lifetime_2.8mW_lineplot.png") or "_uncaging_lifetime_" in f for f in pngs), pngs
        assert lt_pngs and not any("_intensity_" in f and "lineplot_mean_sem" in f for f in pngs), pngs
        vals = summ[f"delta_lifetime_ch{ch}"].dropna()
        print(f"real check passed: {len(summ)} sets, {n_checked} delta lifetimes recomputed, "
              f"delta lifetime mean {vals.mean():.3f} ns (min {vals.min():.3f}, max {vals.max():.3f}), "
              f"{len(lt_pngs)} lifetime figures")
        for f in sorted(lt_pngs):
            print("  ", f)


def main() -> int:
    tests = [
        test_signal_columns_and_folder,
        test_padded_ylim,
        test_exclude_delta_lifetime,
        test_uncaging_lifetime_mean_sem,
        test_uncaging_protocol_split,
        test_assign_uncaging_frame_bins,
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
        test_pin_last_pre_at_uncaging,
        test_mean_sem_colors_ignore_extra_entries,
        test_uncaging_pulse_times_from_80frame_header,
        test_inclusion_counts_pool_conditions_in_two_hour_bins,
        test_roi_gui_rejects_count_as_rejected_without_a_reason_split,
        test_exclude_uncaging_from_mean_intensity,
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
    # python test_ltp_group_analysis.py [--real PKL CSV [CH]]  (real: lifetime suite in a temp folder)
    if len(sys.argv) >= 4 and sys.argv[1] == "--real":
        code = main()
        real_check(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
        raise SystemExit(code)
    raise SystemExit(main())
