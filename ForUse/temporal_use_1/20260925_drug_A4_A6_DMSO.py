# %%
"""AP5 vs Control grouped LTP analysis.

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
# df_save_path_1 = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260909/auto1/combined_df_respan.pkl"
# out_csv_path = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260909/auto1/combined_df_respan_intensity_lifetime_all_frames.csv"
df_save_path_1 = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260925/auto1\combined_df_respan.pkl"
out_csv_path = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260925/auto1\combined_df_respan_intensity_lifetime_all_frames.csv"

# %% experiment settings
acquisiton_start_datetime_str = "2026-09-25T14:30:00.000"
# 25-35 min Δspine volume (F/F0 - 1). None = no cutoff on that side.
spine_volume_delta_ff0_min = -2.0
spine_volume_delta_ff0_max = 6.0
# Mean±SEM time axis in minutes. Pre is negative, post is positive, uncaging is 0.
mean_sem_xlim_pre_min = -20.0
mean_sem_xlim_post_min = 35.0

cfg = LTPGroupAnalysisConfig(
    df_save_path=df_save_path_1,
    out_csv_path=out_csv_path,
    acquisition_start_datetime_str=acquisiton_start_datetime_str,
    condition_prefix_map={
        "DMSO_": "DMSO",
        "A4_": "EGFRi",
        "A6_": "AuroraAi",
    },
    condition_order=["DMSO", "EGFRi", "AuroraAi"],
    # Mean-line colors in condition_order. Extra colors are unused.
    condition_styles=["k", "b", "g", "r", "m", "c"],
    # 0 = same as the mean color, 1 = white. Applied to every individual trace.
    mean_sem_indiv_lightness=0.65,
    spine_volume_delta_ff0_min=spine_volume_delta_ff0_min,
    spine_volume_delta_ff0_max=spine_volume_delta_ff0_max,
    binned_time_xlim_min=mean_sem_xlim_pre_min,
    binned_time_xlim_max=mean_sem_xlim_post_min,
    # Uncaging frames are a different average than pre/post imaging. Set True to plot them.
    plot_uncaging_on_mean_intensity=False,
)

# %%
if __name__ == "__main__":
    save_folder = run_ltp_group_analysis(cfg)
    print(save_folder)
