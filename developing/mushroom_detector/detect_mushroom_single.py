# -*- coding: utf-8 -*-
"""
Detect mushroom spines from a single .flim file.

Uses controlFLIMage/deepd3_mushroom_spine_assign_save.py (DeepD3 + shaft-to-head filter).
Outputs are written next to the .flim (same savefolder convention as multi_spine).

Run with the deepd3 venv, e.g.:
  C:\\Users\\yasudalab\\Documents\\Tetsuya_GIT\\deepd3\\Scripts\\python.exe detect_mushroom_single.py
"""
import os
import sys

_CONTROLFLIMAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _CONTROLFLIMAGE_DIR not in sys.path:
    sys.path.insert(0, _CONTROLFLIMAGE_DIR)

from deepd3_mushroom_spine_assign_save import (  # noqa: E402
    DEEPD3_MODEL_PATH,
    MIN_SHAFT_TO_HEAD_UM,
    loop_mushroom_spine_assign_save,
)

# --- edit here ---
FLIM_PATH = (
    r"G:\ImagingData\Tetsuya\20260608\mushroom_1dend\pos1__highmag_1_002.flim"
)

# Override if the default path in deepd3_mushroom_spine_assign_save.py is missing.
DEEPD3_MODEL = (
    r"C:\Users\yasudalab\Documents\Tetsuya_GIT\ongoing\deepd3\DeepD3_8F.h5"
)

MIN_SHAFT_TO_HEAD_UM_OVERRIDE = MIN_SHAFT_TO_HEAD_UM
SAVE_PER_SPINE_PNG = True
SAVE_PER_SPINE_INI = False


def detect_mushroom_from_flim(
    flim_path: str,
    *,
    min_shaft_to_head_um: float = MIN_SHAFT_TO_HEAD_UM,
    deepd3_model_path: str | None = None,
    save_per_spine_png: bool = SAVE_PER_SPINE_PNG,
    save_per_spine_ini: bool = SAVE_PER_SPINE_INI,
):
    """Run mushroom detection on one .flim file."""
    if not os.path.isfile(flim_path):
        raise FileNotFoundError(f"FLIM file not found: {flim_path}")

    folder = os.path.dirname(flim_path)
    filename = os.path.basename(flim_path)
    model_path = deepd3_model_path or DEEPD3_MODEL_PATH

    if not os.path.isfile(model_path):
        fallback = DEEPD3_MODEL
        if os.path.isfile(fallback):
            print(f"DeepD3 model not found at {model_path!r}; using {fallback!r}")
            model_path = fallback
        else:
            raise FileNotFoundError(
                f"DeepD3 model not found: {model_path!r} (fallback {fallback!r} also missing)"
            )

    print("FLIM:", flim_path)
    print(f"Mushroom threshold: shaft-to-head XY >= {min_shaft_to_head_um} um")

    summary_rows = loop_mushroom_spine_assign_save(
        highmag_folder=folder,
        highmag_filename=filename,
        exclude_ini_saved=False,
        from_latest_file=False,
        min_shaft_to_head_um=min_shaft_to_head_um,
        min_seconds_since_modification=0,
        deepd3_model_path=model_path,
        save_per_spine_ini=save_per_spine_ini,
        save_per_spine_png=save_per_spine_png,
    )
    return summary_rows


def main():
    summary_rows = detect_mushroom_from_flim(
        FLIM_PATH,
        min_shaft_to_head_um=MIN_SHAFT_TO_HEAD_UM_OVERRIDE,
    )
    mushroom_rows = [row for row in summary_rows if row.get("deepd3_label") is not None]
    print(f"\ndone — {len(mushroom_rows)} mushroom spine(s) detected")
    for row in mushroom_rows:
        print(
            f"  label {row['deepd3_label']}: "
            f"shaft-to-head {row.get('shaft_to_head_um', float('nan')):.3f} um"
        )

    print("=========== done ============")


if __name__ == "__main__":
    main()
