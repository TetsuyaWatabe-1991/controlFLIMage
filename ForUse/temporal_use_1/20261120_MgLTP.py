# %%
"""Intensity summary for the 20261117 NMDAR-antagonist session.

Same pattern as ``20261002_Ins2CKO.py``. Does not open the ROI review GUI.

Filename headers are ``<drug>_<well>_pos1__highmag_<n>_``. Well numbers are
collapsed into one condition per drug. ``Gave_`` is labeled Gavestinel
(same drug name as the 20251218 calcium experiment).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Mapping, Sequence

FORUSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLFLIMAGE_DIR = os.path.dirname(FORUSE_DIR)
sys.path.insert(0, CONTROLFLIMAGE_DIR)
sys.path.insert(0, FORUSE_DIR)

from ltp_group_analysis import LTPGroupAnalysisConfig, run_ltp_group_analysis  # noqa: E402

# %% shared plot / filter settings (intensity summary only)
spine_volume_delta_ff0_min = -2.0
spine_volume_delta_ff0_max = 4.0
mean_sem_xlim_pre_min = -20.0
# mean_sem_xlim_post_min = 35.0
mean_sem_xlim_post_min = 80.0
ch_1or2 = 2
split_by_uncaging_protocol = True
pre_instability_max_min_ratio_threshold = 1.5


@dataclass(frozen=True)
class SessionSpec:
    """One combined_df_respan session + filename-prefix labels."""

    name: str
    df_save_path: str
    out_csv_path: str
    acquisition_start_datetime_str: str
    condition_prefix_map: Mapping[str, str]
    condition_order: Sequence[str]
    note: str = ""


# Longer prefixes win (matcher sorts by length). The underscore after the
# drug token keeps AP5_ from matching a bare "AP" and MK801_ from "MK".

SESSIONS: list[SessionSpec] = [
    SessionSpec(
        name="20261117_auto1",
        df_save_path=r"G:/ImagingData/Tetsuya/20261120/auto1\combined_df_respan.pkl",
        out_csv_path=(
            r"G:/ImagingData/Tetsuya/20261120/auto1\combined_df_respan_intensity_lifetime_all_frames.csv"
        ),
        # First quantified frame in the CSV. Replace with the drug-addition
        # clock time if the incubation axis should start earlier.
        acquisition_start_datetime_str="2026-11-18T00:41:57.683",
        condition_prefix_map={
            "": "0.1 mM MgCl2",       # wells 3, 4, 7, 8, 9
        },
        condition_order=[
            "0.1 mM MgCl2",

        ],
        note="Drug token before the well number; wells pooled per drug.",
    ),
]


def _cfg_for(session: SessionSpec) -> LTPGroupAnalysisConfig:
    return LTPGroupAnalysisConfig(
        df_save_path=session.df_save_path,
        out_csv_path=session.out_csv_path,
        acquisition_start_datetime_str=session.acquisition_start_datetime_str,
        condition_prefix_map=dict(session.condition_prefix_map),
        condition_order=list(session.condition_order),
        split_by_uncaging_protocol=split_by_uncaging_protocol,
        condition_styles=["k", "b", "g", "r", "m", "c", "#E69F00", "#56B4E9"],
        mean_sem_indiv_lightness=0.65,
        spine_volume_delta_ff0_min=spine_volume_delta_ff0_min,
        spine_volume_delta_ff0_max=spine_volume_delta_ff0_max,
        binned_time_xlim_min=mean_sem_xlim_pre_min,
        binned_time_xlim_max=mean_sem_xlim_post_min,
        plot_uncaging_on_mean_intensity=False,
        pre_instability_max_min_ratio_threshold=pre_instability_max_min_ratio_threshold,
        ch_1or2=ch_1or2,
        signal="intensity",
    )


# %%
if __name__ == "__main__":
    # Set to a session name to run one, or None to run all.
    only: str | None = None  # e.g. "20261117_auto1"

    for session in SESSIONS:
        if only is not None and session.name != only:
            continue
        print("=" * 72)
        print(f"[{session.name}] {session.note}")
        print(f"  labels: {dict(session.condition_prefix_map)}")
        assert os.path.exists(session.df_save_path), session.df_save_path
        assert os.path.exists(session.out_csv_path), session.out_csv_path
        save_folder = run_ltp_group_analysis(_cfg_for(session))
        print(f"  saved -> {save_folder}")
