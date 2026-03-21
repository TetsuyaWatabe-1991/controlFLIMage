#%%
from gui_roi_fast_simple import run_tiff_uncaging_roi_no_zstack, create_initial_roi_masks_from_ini

df_save_path_1, out_csv_path = run_tiff_uncaging_roi_no_zstack(
    ch_1or2=2,
    z_plus_minus=1,
    pre_length=1,
    photon_threshold=15,
    total_photon_threshold=1000,
    uncaging_frame_num=[32, 33, 34, 35, 55],
    titration_frame_num=[],
)

#%%
import pandas as pd
from simple_dialog import ask_yes_no_gui
ask_yn = ask_yes_no_gui("Do you want to create initial ROI masks?")
if ask_yn:
    create_initial_roi_masks_from_ini(
        combined_df=pd.read_pickle(df_save_path_1),
    )

#%%
df_save_path_1, out_csv_path = run_tiff_uncaging_roi_no_zstack(
    ch_1or2=2,
    z_plus_minus=1,
    pre_length=1,
    photon_threshold=15,
    total_photon_threshold=1000,
    uncaging_frame_num=[32, 33, 34, 35, 55],
    titration_frame_num=[],
)

# %%
