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
df_save_path_1 = r"G:/ImagingData/Tetsuya/20261002/auto2\combined_df_respan.pkl"
out_csv_path = r"G:/ImagingData/Tetsuya/20261002/auto2\combined_df_respan_intensity_lifetime_all_frames.csv"

# %% experiment settings
acquisiton_start_datetime_str = "2026-10-02T18:00:00.000"
# 25-35 min Δspine volume (F/F0 - 1). None = no cutoff on that side.
spine_volume_delta_ff0_min = -2.0
spine_volume_delta_ff0_max = 6.0
# Mean±SEM time axis in minutes. Pre is negative, post is positive, uncaging is 0.
mean_sem_xlim_pre_min = -20.0
mean_sem_xlim_post_min = 35.0
# Plotted quantity: "intensity" = spine volume (F/F0 - 1), "lifetime" = delta lifetime (ns).
# Lifetime plots go to <session>/summary_lifetime/ (intensity keeps <session>/summary/).
signal = "lifetime"
# Lifetime filters (ns). None = off.
pre_lifetime_range_max_ns = None  # drop a set whose pre Spine lifetimes vary more than this
delta_lifetime_min = None  # drop a set whose 25-35 min delta lifetime is below this
delta_lifetime_max = None  # ... or above this

cfg = LTPGroupAnalysisConfig(
    df_save_path=df_save_path_1,
    out_csv_path=out_csv_path,
    acquisition_start_datetime_str=acquisiton_start_datetime_str,
    signal=signal,
    pre_lifetime_range_max_ns=pre_lifetime_range_max_ns,
    delta_lifetime_min=delta_lifetime_min,
    delta_lifetime_max=delta_lifetime_max,
    condition_prefix_map={
        "": "Camui IUE",
        # "DMSO_": "DMSO",
        # "A4_": "EGFRi",
        # "A6_": "AuroraAi",
    },
    condition_order=["Camui IUE"],
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
    pre_instability_max_min_ratio_threshold=1.5,
    ch_1or2=1, 
)

# %%
if __name__ == "__main__":
    save_folder = run_ltp_group_analysis(cfg)
    print(save_folder)
