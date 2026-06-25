# %% import libraries
import os
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTROLFLIMAGE = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "controlFLIMage"))
for _path in (_SCRIPT_DIR, _CONTROLFLIMAGE):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from gui_roi_fast_simple import run_tiff_uncaging_roi

# %% set parameters and read image, define ROI, quantify intensity and lifetime
ch_1or2 = 2
z_plus_minus = 2
uncaging_at_nth = 2
skip_lifetime_analysis = True

pre_length = uncaging_at_nth + 1
df_save_path_1, out_csv_path = run_tiff_uncaging_roi(ch_1or2 = ch_1or2,
                                    z_plus_minus = z_plus_minus,
                                    pre_length = pre_length,
                                    # predefined_df_path=r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260318\auto1\combined_df_1.pkl",
                                    # skip_roi_gui=True
                                    skip_lifetime_analysis=skip_lifetime_analysis,
                                    )

# %%

try:
    display("  ")
except:
    print("  ")
print("df_save_path_1 =",f"r\"{df_save_path_1}\"")
print("out_csv_path =",f"r\"{out_csv_path}\"")


# %%
