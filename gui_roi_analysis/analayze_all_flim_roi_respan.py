# %% import libraries
import os
import sys

_FORUSE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ForUse"))
if _FORUSE_DIR not in sys.path:
    sys.path.insert(0, _FORUSE_DIR)

from flim_summarize_func import format_respan_path_assignments  # noqa: E402
from gui_roi_respan_seg_masks import run_tiff_uncaging_roi_respan

# %% set parameters (aligned with tpem_low_high_spine_multi_merged_titrate_uncaging_pow_respan.py)
ch_1or2 = 1
z_plus_minus = 1
uncaging_at_nth = 2
pre_length = uncaging_at_nth + 1


# Whole-field highpass against frame 0. Spine-window local align is skipped
# (only local_align_mode == "adjacent" writes local_shift_*).
global_align_method = "highpass"
# Reference of the set TIFF: "pre1" = every pre/post frame registered directly to the set's
# Pre 1 (3D phase correlation, no filter); "002" = the global registration above (former).
# Changing it rebuilds the TIFF stacks; existing ROI masks are moved with the frames.
align_reference = "pre1"
local_align_mode = "A"
local_crop_half_size = 60  # unused while local align is off

# Uncaging ROI editing: None or >= n_unc -> seek bar uses every uncaging frame (legacy).
# 1/2/3+ -> seek bar stops at that many uncaging keyframes only (first, last, evenly spaced).
# Quantification and plot still use all uncaging frames (interpolated ROI positions).
# Set overwrite_seg_roi_masks=True to reset all ROI masks from seg_masks on re-run.
uncaging_roi_keyframe_count = 3
overwrite_seg_roi_masks = False
skip_lifetime_analysis = True
# Lifetime fit per ROI and frame: pixels with <= photon_threshold photons are left out of
# the decay; the lifetime is NaN when the ROI total of the remaining pixels is below
# total_photon_threshold (former default 1000).
photon_threshold = 15
total_photon_threshold = 300
# induction (~33 or 36 averaged frames), common TS (55), uncaging_2Hz30pulses (80), uncaging1hz (144)
uncaging_frame_num = [33, 34, 35, 36, 55, 80, 144]

# Default False keeps the original (slower) path. True: skip repeated FLIM decode,
# header-only shape peek, intensity cache, no PNG/small TIFF, lighter align.
fast_mode = True

# Spine ROI prefill: "seg" = seg_masks (as before); "respan_tracked" = uncaged spine
# tracked by per-frame RESPAN (run ongoing/ASIcontroller/respan_track_session.py on the
# session first). ROIs edited in the GUI are never overwritten.
spine_roi_source = "respan_tracked"

# Time window (minutes from uncaging). Pre/post frames outside it are removed from their
# set: they are not in the TIFF, the viewer or the quantification. None keeps all frames.
# Changing it rebuilds the TIFF stacks of an existing combined_df_respan.pkl.
time_window_min = (-40, 50)

# ROI review:
#   "viewer": quantify every set with the pre-filled ROIs first, then open the set review
#             viewer (Left/Right: move, E: edit ROIs in the ROI GUI -> this set is
#             re-quantified, R: reject / un-reject, U: uncaging position).
#             Only the sets that need it are edited.
#   "table" : open the ROI table GUI before quantification (former flow).
roi_review = "viewer"

# True: skip the analysis and open the viewer on an existing combined_df_respan.pkl
# (a dialog asks for it unless viewer_pkl_path is set). Sets without quantification
# are quantified when they are shown.
start_with_viewer = False
viewer_pkl_path = None  # e.g. r"G:\ImagingData\Tetsuya\20260929\auto1\combined_df_respan.pkl"

# Pearson r of the aligned images (no filtering), shown below each Pre/Post image of the
# viewer as "H r  L r": H = highmag frame vs the set's Pre 1 (as shown), L = lowmag of the same cycle vs the
# first lowmag (_001). After the analysis, all sets are computed and saved to
# <pkl stem>_alignment_r.csv next to the pkl (with start_with_viewer, sets are computed
# when they are shown and cached beside each TIFF).
show_alignment_r = True

viewer_kwargs = dict(
    ch_1or2=ch_1or2,
    z_plus_minus=z_plus_minus,
    skip_lifetime_analysis=skip_lifetime_analysis,
    photon_threshold=photon_threshold,
    total_photon_threshold=total_photon_threshold,
    uncaging_roi_keyframe_count=uncaging_roi_keyframe_count,
    show_alignment_r=show_alignment_r,
    time_window_min=time_window_min,
)

if start_with_viewer:
    from set_review_viewer import launch_set_review_viewer

    _session = launch_set_review_viewer(viewer_pkl_path, None, **viewer_kwargs)
    df_save_path_1 = _session.pkl_path if _session else None
    out_csv_path = _session.csv_path if _session else None
else:
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
        photon_threshold=photon_threshold,
        total_photon_threshold=total_photon_threshold,
        fast_mode=fast_mode,
        spine_roi_source=spine_roi_source,
        time_window_min=time_window_min,
        align_reference=align_reference,
        skip_roi_gui=(roi_review == "viewer"),
        # predefined_df_path=r"G:\ImagingData\Tetsuya\20260610\auto3\combined_df_respan.pkl",
        # flim_path=r"G:\ImagingData\Tetsuya\20260610\auto3\pos3__highmag_1_002.flim",
    )

# %% alignment correlation of all sets (cached per set, so the viewer shows it at once)
if show_alignment_r and not start_with_viewer and df_save_path_1:
    from alignment_correlation import session_alignment_r

    session_alignment_r(df_save_path_1, ch=ch_1or2)

# %% set review viewer after the analysis (re-open any time by running this cell)
if roi_review == "viewer" and not start_with_viewer and df_save_path_1 and out_csv_path:
    from set_review_viewer import launch_set_review_viewer

    launch_set_review_viewer(df_save_path_1, out_csv_path, **viewer_kwargs)

# %%
print("Paste into the LTP analysis script:")
print(format_respan_path_assignments(df_save_path_1, out_csv_path))

# %%
