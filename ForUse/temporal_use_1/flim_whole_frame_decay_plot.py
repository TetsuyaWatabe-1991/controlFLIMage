"""
FLIM whole-frame decay analysis and plotting.

For IPython: run the script with %run or execute the cell; it loads the default
.flim file and displays plots (display only, no save).

Loads a .flim file, takes channel 1 (ch1), builds a decay curve from all pixels
with intensity above a threshold, fits a single-exponential convolved with
Gaussian IRF, and plots:
  - Measured decay, convolved fit, IRF, and deconvolved decay (linear and log).
  - A separate figure for the deconvolved decay only.

Does not modify any existing code. All text in this file is in English.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure controlFLIMage root is on path so imports work from any cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
_control_flimage = os.path.dirname(os.path.dirname(_script_dir))
if _control_flimage not in sys.path:
    sys.path.insert(0, _control_flimage)

from FLIMageFileReader2 import FileReader
from fitting.flim_lifetime_fitting import FLIMLifetimeFitter


DEFAULT_THRESHOLD = 10
DEFAULT_SYNC_RATE_HZ = 80e6
CHANNEL = 1  # ch1

# Default .flim file to load when run in IPython (display only, no save)
# DEFAULT_FLIM_PATH = "/Users/watabetetsuya/Downloads/20240315/mStayGold_ftractin_004.flim"
DEFAULT_FLIM_PATH = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20240315\mStayGold_ftractin_004.flim"

def _get_ps_per_unit(reader, channel: int) -> float:
    """Resolution in picoseconds per time bin for the given channel."""
    res = getattr(reader, "resolution", 250)
    res = np.atleast_1d(res)
    return float(res[channel] if channel < len(res) else res[0])


def _get_sync_rate(reader) -> float:
    """Laser sync rate in Hz from file state if available."""
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
            sync_rate = np.atleast_1d(sync_rate)
            if len(sync_rate) > CHANNEL:
                return float(sync_rate[CHANNEL])
            return float(sync_rate[0])
    except Exception:
        pass
    return DEFAULT_SYNC_RATE_HZ


def _intensity_mask(flim3d: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask (H, W) where intensity >= threshold."""
    intensity = np.sum(flim3d, axis=2)
    return intensity >= threshold


def _decay_from_masked_pixels(flim3d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum decay over all pixels where mask is True. Returns shape (n_time,)"""
    # mask: (H, W) -> broadcast to (H, W, T)
    mask_3d = np.broadcast_to(mask[:, :, np.newaxis], flim3d.shape)
    return np.sum(flim3d * mask_3d, axis=(0, 1))


def _irf_gaussian(x: np.ndarray, t0: float, tau_g: float, amplitude: float) -> np.ndarray:
    """Gaussian IRF in same time-bin units as x. amplitude scales height."""
    y = amplitude * np.exp(-0.5 * ((x - t0) / tau_g) ** 2)
    return y


def _deconvolved_single_exp(x: np.ndarray, amplitude: float, decay_rate: float, t0: float) -> np.ndarray:
    """Pure single exponential (no convolution): A * exp(-decay_rate * (x - t0)) for x >= t0."""
    y = np.zeros_like(x, dtype=float)
    valid = x >= t0
    y[valid] = amplitude * np.exp(-decay_rate * (x[valid] - t0))
    return y


def run(
    flim_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    sync_rate: float = None,
) -> None:
    """
    Load .flim ch1, build thresholded decay, fit, and plot decay + IRF + deconvolved.
    """
    if not os.path.isfile(flim_path):
        raise FileNotFoundError(f"Not a file: {flim_path}")
    if not flim_path.lower().endswith(".flim"):
        print("Warning: file does not have .flim extension.")

    reader = FileReader()
    reader.read_imageFile(flim_path, readImage=True)

    if not getattr(reader, "flim", False):
        raise ValueError("File is not read as FLIM data.")

    if CHANNEL >= reader.nChannels or reader.n_time[CHANNEL] < 2:
        raise ValueError(
            f"Channel {CHANNEL} not available or too few time bins. "
            f"nChannels={reader.nChannels}, n_time[CHANNEL]={reader.n_time[CHANNEL]}."
        )

    reader.LoadFLIMFromMemory(0, 0, CHANNEL)
    flim3d = np.asarray(reader.FLIM3D, dtype=float)
    n_time = flim3d.shape[2]

    mask = _intensity_mask(flim3d, threshold)
    n_pixels = int(np.sum(mask))
    if n_pixels == 0:
        raise ValueError(
            f"No pixels with intensity >= {threshold}. Try a lower threshold."
        )

    y_decay = _decay_from_masked_pixels(flim3d, mask)
    x = np.arange(n_time, dtype=float)

    ps_per_unit = _get_ps_per_unit(reader, CHANNEL)
    if sync_rate is None:
        sync_rate = _get_sync_rate(reader)
    if not sync_rate or sync_rate <= 0:
        sync_rate = DEFAULT_SYNC_RATE_HZ

    fitter = FLIMLifetimeFitter()
    result = fitter.fit_single_exponential(x, y_decay, ps_per_unit, sync_rate)

    if not result["success"]:
        print("Warning: fit did not converge:", result.get("message", ""))

    beta = result["beta"]
    amplitude, decay_rate, tau_g, t0 = beta
    pulse_interval = 1e12 / sync_rate / ps_per_unit

    # Convolved fit (model = decay * IRF)
    y_fit_convolved = result["fit_curve"]

    # IRF: Gaussian with fitted tau_g, t0; scale to data for visibility
    irf_scale = np.max(y_decay) if np.max(y_decay) > 0 else 1.0
    y_irf = _irf_gaussian(x, t0, tau_g, irf_scale)

    # Deconvolved decay: pure exponential
    y_deconv = _deconvolved_single_exp(x, amplitude, decay_rate, t0)
    # Avoid zeros for log scale
    y_deconv_log = np.where(y_deconv > 0, y_deconv, np.nan)

    lifetime_ns = result["lifetime"]
    print(f"Threshold: {threshold}, pixels above: {n_pixels}")
    print(f"ps_per_unit: {ps_per_unit}, sync_rate: {sync_rate:.2e} Hz")
    print(f"Fitted lifetime: {lifetime_ns:.3f} ns")
    print(f"Fit success: {result['success']}, chi_square: {result['chi_square']:.4f}")

    # Time in ns for axis (optional; we keep bins for consistency with fitter)
    x_ns = x * ps_per_unit / 1000.0

    # ----- Plot 1: Linear and log scale (decay + IRF + convolved fit + deconvolved) -----
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear
    ax = axes[0]
    ax.plot(x_ns, y_decay, "b.", alpha=0.6, label="Measured decay")
    ax.plot(x_ns, y_fit_convolved, "r-", label="Convolved fit")
    ax.plot(x_ns, y_irf, "c-", alpha=0.8, label="IRF (Gaussian)")
    ax.plot(x_ns, y_deconv, "g-", alpha=0.8, label="Deconvolved decay")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Counts")
    ax.set_title("Decay (linear scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log (ylim from actual data min/max so the curve is visible)
    ax = axes[1]
    ax.semilogy(x_ns, np.where(y_decay > 0, y_decay, np.nan), "b.", alpha=0.6, label="Measured decay")
    ax.semilogy(x_ns, np.where(y_fit_convolved > 0, y_fit_convolved, np.nan), "r-", label="Convolved fit")
    ax.semilogy(x_ns, np.where(y_irf > 0, y_irf, np.nan), "c-", alpha=0.8, label="IRF (Gaussian)")
    ax.semilogy(x_ns, y_deconv_log, "g-", alpha=0.8, label="Deconvolved decay")
    _yd = y_decay[y_decay > 0]
    if len(_yd) > 0:
        _ymin, _ymax = np.min(_yd), np.max(_yd)
        _r = np.log10(_ymax) - np.log10(_ymin)
        ax.set_ylim(10 ** (np.log10(_ymin) - 0.20 * _r), 10 ** (np.log10(_ymax) + 0.10 * _r))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Counts")
    ax.set_title("Decay (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig1.suptitle(f"FLIM ch1 whole-frame decay (intensity ≥ {threshold}, τ = {lifetime_ns:.2f} ns)")
    fig1.tight_layout()

    # ----- Plot 2: Deconvolved decay only (linear and log) -----
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes2[0]
    ax.plot(x_ns, y_deconv, "g-", label="Deconvolved decay")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Counts")
    ax.set_title("Deconvolved decay (linear)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.semilogy(x_ns, y_deconv_log, "g-", label="Deconvolved decay")
    _yd = y_decay[y_decay > 0]
    if len(_yd) > 0:
        _ymin, _ymax = np.min(_yd), np.max(_yd)
        _r = np.log10(_ymax) - np.log10(_ymin)
        ax.set_ylim(10 ** (np.log10(_ymin) - 0.20 * _r), 10 ** (np.log10(_ymax) + 0.10 * _r))
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Counts")
    ax.set_title("Deconvolved decay (log)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig2.suptitle("Deconvolved decay (single exponential, no IRF)")
    fig2.tight_layout()

    plt.show()


# IPython: run this file to load the default .flim and display plots (no save)
if __name__ == "__main__":
    if "--test" in sys.argv:
        # Test: missing .flim file must raise FileNotFoundError
        try:
            run(os.path.join(_script_dir, "nonexistent.flim"), threshold=DEFAULT_THRESHOLD)
        except FileNotFoundError as e:
            print("Test passed: FileNotFoundError as expected:", e)
        else:
            sys.exit("Test failed: expected FileNotFoundError for missing file")
    else:
        run(DEFAULT_FLIM_PATH, threshold=DEFAULT_THRESHOLD)
