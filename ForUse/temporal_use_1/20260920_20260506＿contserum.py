# %%
"""Rab vs Calf grouped LTP analysis.

Shared processing/plotting lives in ``ForUse/ltp_group_analysis.py``.
Paste the two path lines printed at the end of ROI analysis below.
"""
import os
import sys

FORUSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLFLIMAGE_DIR = os.path.dirname(FORUSE_DIR)
sys.path.insert(0, CONTROLFLIMAGE_DIR)
sys.path.insert(0, FORUSE_DIR)

from ltp_group_analysis import LTPGroupAnalysisConfig, run_ltp_group_analysis  # noqa: E402

# %% paste from ROI analysis
df_save_path_1 = r"G:/ImagingData/Tetsuya/20260506/test_gibco_serum\combined_df_respan.pkl"
out_csv_path = r"G:/ImagingData/Tetsuya/20260506/test_gibco_serum\combined_df_respan_intensity_lifetime_all_frames.csv"

# %% experiment settings
acquisiton_start_datetime_str = "2026-05-06T13:30:00.000"
# 25-35 min Δspine volume (F/F0 - 1). None = no cutoff on that side.
spine_volume_delta_ff0_min = -2.0
spine_volume_delta_ff0_max = 4.0

cfg = LTPGroupAnalysisConfig(
    df_save_path=df_save_path_1,
    out_csv_path=out_csv_path,
    acquisition_start_datetime_str=acquisiton_start_datetime_str,
    condition_prefix_map={
        "": "Gibco",
        # "calf_": "Calf",
    },
    condition_order=["Gibco"],
    condition_styles={せ
        "Gibco": {"indiv": "0.75", "mean": "k"},
        # "Calf": {"indiv": "#E8B4B0", "mean": "r"},
    },
    spine_volume_delta_ff0_min=spine_volume_delta_ff0_min,
    spine_volume_delta_ff0_max=spine_volume_delta_ff0_max,
)

# %%
if __name__ == "__main__":
    save_folder = run_ltp_group_analysis(cfg)
    print(save_folder)
