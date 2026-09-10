# -*- coding: utf-8 -*-
"""Launch the bead FWHM GUI with a .flim path set below.

Edit FLIM_PATH, then run this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Edit here
# ---------------------------------------------------------------------------
FLIM_PATH = r"G:\ImagingData\Tetsuya\20260903\0,2beads_067.flim"
# Example:
# FLIM_PATH = r"G:\ImagingData\Tetsuya\20260820\auto2\something_001.flim"
CHANNEL = 1  # 1 = Ch1, 2 = Ch2
# ---------------------------------------------------------------------------

_CALIB_DIR = Path(__file__).resolve().parent
if str(_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIB_DIR))


def resolve_flim_path(path: str | Path) -> Path:
    """Return an existing .flim path, or raise if FLIM_PATH is empty/missing."""
    text = str(path).strip().strip('"').strip("'")
    if not text:
        raise ValueError("Set FLIM_PATH to a .flim file at the top of this script.")
    resolved = Path(text)
    if not resolved.is_file():
        raise FileNotFoundError(f"FLIM file not found: {resolved}")
    return resolved


def main() -> int:
    flim_path = resolve_flim_path(FLIM_PATH)
    from bead_fwhm_gui import launch_window

    print(f"Opening Ch{CHANNEL}: {flim_path}")
    return launch_window(flim_path=str(flim_path), channel_1based=int(CHANNEL))


if __name__ == "__main__":
    raise SystemExit(main())
