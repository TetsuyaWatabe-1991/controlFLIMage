# %% import libraries
import os
import sys

_FORUSE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ForUse"))
if _FORUSE_DIR not in sys.path:
    sys.path.insert(0, _FORUSE_DIR)

from flim_summarize_func import format_respan_path_assignments  # noqa: E402
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

# Uncaging ROI editing: None or >= n_unc -> seek bar uses every uncaging frame (legacy).
# 1/2/3+ -> seek bar stops at that many uncaging keyframes only (first, last, evenly spaced).
# Quantification and plot still use all uncaging frames (interpolated ROI positions).
# Set overwrite_seg_roi_masks=True to reset all ROI masks from seg_masks on re-run.
uncaging_roi_keyframe_count = 3
overwrite_seg_roi_masks = False
skip_lifetime_analysis = True
# induction (~33 or 36 averaged frames), common TS (55), uncaging_2Hz30pulses (80), uncaging1hz (144)
uncaging_frame_num = [33, 34, 35, 36, 55, 80, 144]

# Default False keeps the original (slower) path. True: skip repeated FLIM decode,
# header-only shape peek, intensity cache, no PNG/small TIFF, lighter align.
fast_mode = True

print("Running respan ROI analysis.\nExplorer will pop up to select the FLIM file.")
df_save_path_1, out_csv_path = run_tiff_uncaging_roi_respan(
    ch_1or2=ch_1or2,
    z_plus_minus=z_plus_minus,
    pre_length=pre_length,
    global_align_method=global_align_method,
    local_align_mode=local_align_mode,
    local_crop_half_size=local_crop_half_size,
    uncaging_frame_num=uncaging_frame_num,
    uncaging_roi_keyframe_count=uncaging_roi_keyframe_count,
    overwrite_seg_roi_masks=overwrite_seg_roi_masks,
    skip_lifetime_analysis=skip_lifetime_analysis,
    fast_mode=fast_mode,
    # predefined_df_path=r"G:\ImagingData\Tetsuya\20260610\auto3\combined_df_respan.pkl",
    # flim_path=r"G:\ImagingData\Tetsuya\20260610\auto3\pos3__highmag_1_002.flim",
)

# %%
print("Paste into the LTP analysis script:")
print(format_respan_path_assignments(df_save_path_1, out_csv_path))

# %%
