# %% import libraries
from gui_roi_respan_seg_masks import (
    GLOBAL_ALIGN_METHOD,
    LOCAL_ALIGN_MODE,
    run_tiff_uncaging_roi_respan,
)

# %% set parameters (aligned with tpem_low_high_spine_multi_merged_titrate_uncaging_pow_respan.py)
ch_1or2 = 2
z_plus_minus = 2
uncaging_at_nth = 2
pre_length = uncaging_at_nth + 1

# Explicit alignment policy (same as live respan titration + quant replay)
global_align_method = GLOBAL_ALIGN_METHOD  # roi_adjacent
local_align_mode = LOCAL_ALIGN_MODE  # adjacent == LocalAlignMode.ADJACENT
local_crop_half_size = 60  # quant_small_region_size in titration script

print("Running respan ROI analysis.\nExplorer will pop up to select the FLIM file.")
df_save_path_1, out_csv_path = run_tiff_uncaging_roi_respan(
    ch_1or2=ch_1or2,
    z_plus_minus=z_plus_minus,
    pre_length=pre_length,
    global_align_method=global_align_method,
    local_align_mode=local_align_mode,
    local_crop_half_size=local_crop_half_size,
    # predefined_df_path=r"G:\ImagingData\Tetsuya\20260610\auto3\combined_df_respan.pkl",
    # flim_path=r"G:\ImagingData\Tetsuya\20260610\auto3\pos3__highmag_1_002.flim",
)

# %%
print("df_save_path_1 =", f'r"{df_save_path_1}"')
print("out_csv_path =", f'r"{out_csv_path}"')

# %%
