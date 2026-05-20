# Interactive: parameters auto-selected from .flim State.Acq.zoom (2x / 4x presets)
from run_branch_segment_score import run_interactive

FLIM = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260508\auto1\CM_1_pos1_001.flim"


list_flim = [
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260508\auto1\APV_9_pos1_001.flim",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260508\auto1\APV_7_pos1_001.flim",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260508\auto1\APV_6_pos1_001.flim",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260508\auto1\CM_5_pos1_001.flim",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260508\auto1\CM_4_pos1_001.flim",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260508\auto1\CM_3_pos1_001.flim",
    r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260508\auto1\CM_1_pos1_001.flim"
]
for flim in list_flim:
    result, fig_ov, fig_roi = run_interactive(
        flim,
        top_n=12,      # 上位12セグメントをROIタイルに表示（デフォルト 8）
        roi_um=16.0, 
        min_roi_xy_sep_um=20.0,  # デフォルト
        min_roi_z_sep_um=10.0,   # デフォルト
    )

# result, fig_ov, fig_roi = run_interactive(
#     FLIM,
#     # use_zoom_presets=True,  # zoom=2 -> width 1.0-1.8, pct=98; zoom=4 -> 0.6-1.0, pct=94
#     # save=True,
# )
