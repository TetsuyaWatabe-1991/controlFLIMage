"""
Per-.flim file: sum decay from ch1 pixels with intensity > threshold, fit with FLIMLifetimeFitter.
"""
import os
import sys
import glob

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_control_flimage_root = os.path.normpath(os.path.join(_script_dir, "..", ".."))
if _control_flimage_root not in sys.path:
    sys.path.insert(0, _control_flimage_root)

from FLIMageFileReader2 import FileReader
from fitting.flim_lifetime_fitting import FLIMLifetimeFitter

# --- settings ---
flim_folder = r"G:\ImagingData\Tetsuya\20260320\auto1"
GLOB_PATTERN = "*_[0-9][0-9][0-9].flim"
# Match flim_whole_frame_decay_plot.py: UI ch1 uses channel index 1 here
CH1_INDEX = 1
INTENSITY_THRESHOLD = 30  # sum over time bins > this
DEFAULT_SYNC_RATE_HZ = 80e6


def _get_ps_per_unit(reader: FileReader, channel: int) -> float:
    """Picoseconds per time bin for the given channel."""
    res = getattr(reader, "resolution", 250)
    res = np.atleast_1d(np.asarray(res, dtype=float))
    return float(res[channel] if channel < len(res) else res[0])


def _get_sync_rate(reader: FileReader, channel: int) -> float:
    """Laser sync rate (Hz) from file metadata if present."""
    try:
        state = getattr(reader, "State", None)
        if state is None:
            return DEFAULT_SYNC_RATE_HZ
        spc = getattr(state, "Spc", None)
        if spc is None:
            return DEFAULT_SYNC_RATE_HZ
        datainfo = getattr(spc, "datainfo", None)
        if datainfo is None:
            return DEFAULT_SYNC_RATE_HZ
        sync_rate = getattr(datainfo, "syncRate", None)
        if sync_rate is not None:
            sync_rate = np.atleast_1d(np.asarray(sync_rate, dtype=float))
            if len(sync_rate) > channel:
                return float(sync_rate[channel])
            return float(sync_rate[0])
    except Exception:
        pass
    return DEFAULT_SYNC_RATE_HZ


def _decay_from_masked_pixels(flim3d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum decay over pixels where mask is True; shape (n_time,)."""
    m = np.broadcast_to(mask[:, :, np.newaxis], flim3d.shape)
    return np.sum(flim3d.astype(np.float64) * m, axis=(0, 1))


def fitted_tau_ns_for_file(
    path: str,
    channel: int = CH1_INDEX,
    intensity_thr: float = INTENSITY_THRESHOLD,
) -> tuple[float, int, bool, float]:
    """
    Returns (tau_ns from fit, n_masked_pixels, fit_success, chi_square).
    """
    reader = FileReader()
    reader.read_imageFile(path, readImage=True)

    if not reader.flim:
        raise ValueError("not a FLIM file")

    if channel >= reader.nChannels:
        raise ValueError(f"channel {channel} out of range (nChannels={reader.nChannels})")

    nt = reader.n_time[channel]
    if nt < 2:
        raise ValueError(f"no time histogram for channel {channel}")

    y_total = np.zeros(nt, dtype=np.float64)
    n_pixels = 0

    n_fast = reader.nFastZSlices if reader.FastZStack else 1
    for page in range(reader.n_images):
        for fz in range(n_fast):
            reader.LoadFLIMFromMemory(page, fz, channel)
            flim3d = np.asarray(reader.FLIM3D, dtype=np.float64)
            intensity = np.sum(flim3d, axis=2)
            mask = intensity > intensity_thr
            if not np.any(mask):
                continue
            n_pixels += int(np.sum(mask))
            y_total += _decay_from_masked_pixels(flim3d, mask)

    if n_pixels == 0 or np.sum(y_total) <= 0:
        return float("nan"), 0, False, float("nan")

    x = np.arange(nt, dtype=np.float64)
    ps_per_unit = _get_ps_per_unit(reader, channel)
    sync_rate = _get_sync_rate(reader, channel)
    if not sync_rate or sync_rate <= 0:
        sync_rate = DEFAULT_SYNC_RATE_HZ

    fitter = FLIMLifetimeFitter()
    result = fitter.fit_single_exponential(x, y_total, ps_per_unit, sync_rate)

    tau = float(result["lifetime"])
    success = bool(result.get("success", False))
    chi = float(result.get("chi_square", float("nan")))
    return tau, n_pixels, success, chi


def main() -> None:
    paths = sorted(glob.glob(os.path.join(flim_folder, GLOB_PATTERN)))
    if not paths:
        print(f"No files: {os.path.join(flim_folder, GLOB_PATTERN)}")
        return

    print(
        f"Fitter: fitting.flim_lifetime_fitting.FLIMLifetimeFitter "
        f"(single exponential)\n"
        f"channel index={CH1_INDEX}, intensity sum > {INTENSITY_THRESHOLD}, "
        f"folder={flim_folder}\n"
    )
    print("file\ttau_ns\tn_pixels\tfit_ok\tchi_square")

    for path in paths:
        try:
            tau, n_pix, ok, chi = fitted_tau_ns_for_file(path)
            chi_str = f"{chi:.6g}" if np.isfinite(chi) else "nan"
            print(
                f"{os.path.basename(path)}\t{tau:.6f}\t{n_pix}\t{ok}\t{chi_str}"
            )
        except Exception as exc:
            print(f"{os.path.basename(path)}\tERROR\t-\t-\t{exc}")


if __name__ == "__main__":
    main()
