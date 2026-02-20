# %% import libraries
from gui_roi_fast_simple import run_tiff_uncaging_roi

# %% set parameters and read image, define ROI, quantify intensity and lifetime
ch_1or2 = 2
z_plus_minus = 2
pre_length = 2    # uncaging_at_nth + 1

df_save_path_1, out_csv_path = run_tiff_uncaging_roi(ch_1or2 = ch_1or2,
                                    z_plus_minus = z_plus_minus,
                                    pre_length = pre_length)

# %%
display("  ")

print("df_save_path_1 =",f"r\"{df_save_path_1}\"")
print("out_csv_path =",f"r\"{out_csv_path}\"")


# %%
