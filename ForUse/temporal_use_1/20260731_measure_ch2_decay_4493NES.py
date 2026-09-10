"""
Independent Ch2 lifetime decay measurement for one .flim file (no FLIMage GUI).

Reads photon histogram via FLIMageFileReader2, sums all Ch2 pixels across pages,
reports centroid / single-exp / double-exp fits, and saves CSV + PNG next to the file.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_control_flimage_root = os.path.normpath(os.path.join(_script_dir, "..", ".."))
if _control_flimage_root not in sys.path:
    sys.path.insert(0, _control_flimage_root)

from FLIMageFileReader2 import FileReader
from fitting.flim_lifetime_fitting import FLIMLifetimeFitter

FLIM_PATH = (
    r"\\RY-LAB-YAS15\Users\Yasudalab\Documents\Tetsuya_Imaging"
    r"\20260726\4493NES_dish1_001.flim"
)
# UI Ch2 -> 0-based channel index 1
CHANNEL = 1
DEFAULT_SYNC_RATE_HZ = 80e6
INTENSITY_THRESHOLD = 0  # 0 = use all pixels


def _get_ps_per_unit(reader: FileReader, channel: int) -> float:
    res = getattr(reader, "resolution", 250)
    res = np.atleast_1d(np.asarray(res, dtype=float))
    return float(res[channel] if channel < len(res) else res[0])


def _get_sync_rate(reader: FileReader, channel: int) -> float:
    try:
        sync_rate = np.atleast_1d(
            np.asarray(reader.State.Spc.datainfo.syncRate, dtype=float)
        )
        if len(sync_rate) > channel:
            return float(sync_rate[channel])
        return float(sync_rate[0])
    except Exception:
        return DEFAULT_SYNC_RATE_HZ


def main() -> None:
    reader = FileReader()
    reader.read_imageFile(FLIM_PATH, True)

    print(f"file: {FLIM_PATH}")
    print(f"flim={reader.flim} nChannels={reader.nChannels} n_images={reader.n_images}")
    print(f"n_time={reader.n_time} resolution_ps={reader.resolution}")
    print(f"size={reader.width}x{reader.height} FastZ={reader.FastZStack}")

    if not reader.flim:
        raise SystemExit("Not a FLIM file")
    if CHANNEL >= reader.nChannels:
        raise SystemExit(f"Channel {CHANNEL} out of range")

    nt = int(reader.n_time[CHANNEL])
    if nt < 2:
        raise SystemExit(f"No time histogram for channel {CHANNEL}")

    ps_per_unit = _get_ps_per_unit(reader, CHANNEL)
    sync_rate = _get_sync_rate(reader, CHANNEL)
    if sync_rate <= 0:
        sync_rate = DEFAULT_SYNC_RATE_HZ

    y_total = np.zeros(nt, dtype=np.float64)
    n_pix = 0
    n_fast = reader.nFastZSlices if reader.FastZStack else 1

    for page in range(reader.n_images):
        for fz in range(n_fast):
            reader.LoadFLIMFromMemory(page, fz, CHANNEL)
            flim3d = np.asarray(reader.FLIM3D, dtype=np.float64)
            intensity = flim3d.sum(axis=2)
            if INTENSITY_THRESHOLD > 0:
                mask = intensity > INTENSITY_THRESHOLD
            else:
                mask = np.ones(intensity.shape, dtype=bool)
            n_pix += int(mask.sum())
            if mask.any():
                y_total += flim3d[mask].sum(axis=0)
            print(
                f"  page={page} fz={fz} shape={flim3d.shape} "
                f"photons={flim3d.sum():.0f} masked_pix={int(mask.sum())}"
            )

    total_photons = float(y_total.sum())
    t_ns = np.arange(nt, dtype=np.float64) * ps_per_unit / 1000.0
    peak_idx = int(np.argmax(y_total))
    centroid_ns = float(np.sum(t_ns * y_total) / total_photons) if total_photons > 0 else float("nan")

    print("--- metadata used for fit ---")
    print(f"channel index (Ch2): {CHANNEL}")
    print(f"ps_per_unit: {ps_per_unit}")
    print(f"sync_rate_Hz: {sync_rate}")
    print(f"n_bins: {nt}")
    print(f"total_photons: {total_photons:.0f}")
    print(f"n_pixels_used: {n_pix}")
    print(f"peak_bin: {peak_idx} ({t_ns[peak_idx]:.4f} ns)")
    print(f"intensity-weighted mean arrival time: {centroid_ns:.4f} ns")

    x = np.arange(nt, dtype=np.float64)
    fitter = FLIMLifetimeFitter()

    res1 = fitter.fit_single_exponential(x, y_total, ps_per_unit, sync_rate)
    print("--- single exponential ---")
    print(f"success={res1.get('success')} message={res1.get('message')}")
    print(f"tau_ns={res1.get('lifetime'):.6f}")
    print(f"chi_square={res1.get('chi_square'):.6g} iterations={res1.get('iterations')}")
    b1 = res1["beta"]
    print(
        f"amp={b1[0]:.3g} tau_g_ps={b1[2]*ps_per_unit:.2f} "
        f"t0_ns={b1[3]*ps_per_unit/1000:.4f}"
    )

    res2 = fitter.fit_double_exponential(x, y_total, ps_per_unit, sync_rate)
    print("--- double exponential ---")
    print(f"success={res2.get('success')} message={res2.get('message')}")
    print(f"mean_tau_ns={res2.get('lifetime'):.6f}")
    print(f"chi_square={res2.get('chi_square'):.6g} iterations={res2.get('iterations')}")
    b2 = res2["beta"]
    tau1 = ps_per_unit / b2[1] / 1000.0
    tau2 = ps_per_unit / b2[3] / 1000.0
    pop1 = b2[0] / (b2[0] + b2[2])
    print(
        f"tau1={tau1:.4f} ns  tau2={tau2:.4f} ns  "
        f"pop1={pop1:.3f} pop2={1.0 - pop1:.3f}"
    )

    out_dir = os.path.dirname(FLIM_PATH)
    stem = os.path.splitext(os.path.basename(FLIM_PATH))[0]
    out_csv = os.path.join(out_dir, f"{stem}_Ch2_decay_independent.csv")
    out_png = os.path.join(out_dir, f"{stem}_Ch2_decay_independent.png")

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("bin,time_ns,counts,fit_single,fit_double\n")
        for i in range(nt):
            f.write(
                f"{i},{t_ns[i]:.6f},{y_total[i]:.0f},"
                f"{res1['fit_curve'][i]:.6f},{res2['fit_curve'][i]:.6f}\n"
            )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].semilogy(t_ns, np.maximum(y_total, 0.5), "k.", ms=3, label="data (all pixels)")
    axes[0].semilogy(
        t_ns,
        np.maximum(res1["fit_curve"], 0.5),
        "r-",
        lw=1.5,
        label=f"single exp tau={res1['lifetime']:.3f} ns",
    )
    axes[0].semilogy(
        t_ns,
        np.maximum(res2["fit_curve"], 0.5),
        "b-",
        lw=1.5,
        label=f"double mean tau={res2['lifetime']:.3f} ns",
    )
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Photon counts")
    axes[0].set_title("Ch2 decay (log)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ns, y_total, "k.", ms=3, label="data")
    axes[1].plot(t_ns, res1["fit_curve"], "r-", lw=1.5, label="single")
    axes[1].plot(t_ns, res2["fit_curve"], "b-", lw=1.5, label="double")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Photon counts")
    axes[1].set_title("Ch2 decay (linear)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{os.path.basename(FLIM_PATH)} Ch2 — independent of FLIMage GUI")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved: {out_csv}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
