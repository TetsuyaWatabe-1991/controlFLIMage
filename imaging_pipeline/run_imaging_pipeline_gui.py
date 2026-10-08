"""Open the imaging-pipeline window.

Launch with the project Python, for example:
  python controlFLIMage/imaging_pipeline/run_imaging_pipeline_gui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_CONTROLFLIMAGE = Path(__file__).resolve().parents[1]
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

from imaging_pipeline.gui import main  # noqa: E402

if __name__ == "__main__":
    main()
