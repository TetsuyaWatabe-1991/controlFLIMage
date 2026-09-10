"""Path resolution for external lahammond/RESPAN and lab conda environments."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Return Tetsuya_GIT root (parent of controlFLIMage)."""
    return Path(__file__).resolve().parents[4]


def controlflimage_root() -> Path:
    """Return controlFLIMage repository root."""
    return Path(__file__).resolve().parents[3]


def deepd3_export_root() -> Path:
    """Return controlFLIMage developing/deepd3 (FLIM stack export helpers)."""
    return controlflimage_root() / "developing" / "deepd3"


def respan_root() -> Path:
    """Return lahammond/RESPAN clone root (directory that contains RESPAN/ package)."""
    env = os.environ.get("RESPAN_ROOT")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

    ws = workspace_root()
    for candidate in (
        ws / "third_party" / "RESPAN",
        ws / "ongoing" / "RESPAN",
    ):
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "RESPAN clone not found. Clone lahammond/RESPAN under "
        f"{ws / 'third_party' / 'RESPAN'} or set RESPAN_ROOT."
    )


def respan_repo_root() -> Path:
    """Return official RESPAN package directory (clone_root/RESPAN)."""
    return respan_root() / "RESPAN"


def respan_scripts_dir() -> Path:
    return respan_repo_root() / "Scripts"


def conda_root() -> Path:
    env = os.environ.get("CONDA_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / "AppData" / "Local" / "miniconda3"


def conda_gpu_env() -> Path:
    return conda_root() / "envs" / "respan_gpu"


def conda_nnunet_env() -> Path:
    return conda_root() / "envs" / "respan_nnunet"


def respan_gpu_python() -> Path:
    env = os.environ.get("RESPAN_GPU_PYTHON")
    if env:
        return Path(env).expanduser()
    return conda_gpu_env() / "python.exe"


def respan_nnunet_python() -> Path:
    env = os.environ.get("RESPAN_NNUNET_PYTHON")
    if env:
        return Path(env).expanduser()
    return conda_nnunet_env() / "python.exe"


def nnunet_predict_bat() -> Path:
    return conda_nnunet_env() / "Scripts" / "nnUNetv2_predict.bat"


def clean_launcher_path() -> Path:
    return respan_scripts_dir() / "clean_launcher.py"


def respan_model_dir() -> Path:
    """Return pretrained nnU-Net model folder (RESPAN Model 3 for 2P in vivo)."""
    env = os.environ.get("RESPAN_MODEL_DIR")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

    legacy = Path(
        r"G:\ImagingData\Model_3_2P_XY102nm_Z1um\Dataset220_XY102nm_Z1000nm_2P"
    )
    if legacy.is_dir():
        return legacy

    raise FileNotFoundError(
        "Pretrained model folder not found. Download weights from the RESPAN README "
        "Google Form and set RESPAN_MODEL_DIR."
    )


def nnunet_trainer_dir() -> Path:
    return respan_model_dir() / "nnUNetTrainer__nnUNetPlans__3d_fullres"


def run_subfolder_name_from_tiff(tiff_path: Path) -> str:
    """Strip channel export suffix so Ch1/Ch2 runs share one folder per FLIM field."""
    stem = tiff_path.stem
    for suffix in ("_ch1_zyx", "_ch2_zyx"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem
