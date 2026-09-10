"""Insert lahammond/RESPAN and lab dependencies on sys.path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from respan_runner.paths import (
    conda_gpu_env,
    controlflimage_root,
    deepd3_export_root,
    respan_root,
    respan_scripts_dir,
)


def configure_cuda_path() -> None:
    cuda_lib = conda_gpu_env() / "Library"
    cuda_bin = cuda_lib / "bin"
    if cuda_lib.is_dir():
        os.environ.setdefault("CUDA_PATH", str(cuda_lib))
        os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")


def ensure_respan_syspath() -> Path:
    """Add RESPAN upstream and FLIM export helpers to sys.path; return clone root."""
    configure_cuda_path()
    root = respan_root()
    for path in (
        respan_scripts_dir(),
        root,
        controlflimage_root(),
        deepd3_export_root(),
    ):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return root
