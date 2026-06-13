# -*- coding: utf-8 -*-
"""
Compare edge-artifact mitigation strategies and plot alignment results.

Run:
  C:\\Users\\yasudalab\\AppData\\Local\\anaconda3\\python.exe run_edge_mitigation_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _SCRIPT_DIR.parents[1]
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from FLIMageAlignment import (  # noqa: E402
    flim_files_to_nparray,
    get_flimfile_list,
    get_xyz_pixel_um,
)
from FLIMageFileReader2 import FileReader  # noqa: E402
from align_methods import (  # noqa: E402
    RoiCropSpec,
    alignment_mse_vs_ref,
    crop_zyx,
    shifts_to_um,
)
from edge_mitigation_methods import run_edge_mitigation_methods  # noqa: E402
from uncaging_utils import uncaging_info_from_series  # noqa: E402

FLIM_PATHS = [
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_4_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_2_002.flim",
]
CHANNEL_1OR2 = 2
SMALL_REGION_SIZE = 60
SMALL_Z_PLUS_MINUS = 2
PLOT_FRAMES = (0, 6, 7, 8, 16)


def _load_stack_and_motor(filelist: list[str], ch: int) -> tuple[np.ndarray, list, list]:
    """Load aligned stack and motor positions for frames kept by flim_files_to_nparray."""
    ref_shape = None
    motors = []
    used_paths = []
    for fp in filelist:
        im = FileReader()
        im.read_imageFile(fp, readImage=True)
        shape = np.array(im.image).shape
        if ref_shape is None:
            ref_shape = shape
        if shape != ref_shape:
            continue
        pos = im.statedict.get("State.Motor.motorPosition", [0, 0, 0])
        motors.append((float(pos[0]), float(pos[1]), float(pos[2])))
        used_paths.append(fp)

    stack, _, time_sec = flim_files_to_nparray(filelist, ch=ch)
    if len(motors) != stack.shape[0]:
        motors = motors[: stack.shape[0]]
    return stack, motors, time_sec


def _save_drift_plot(
    results_um: dict,
    time_sec: list,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    labels = ["Z drift (um)", "Y drift (um)", "X drift (um)"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_um)))
    for (name, sh_um), color in zip(results_um.items(), colors):
        for ax_i in range(3):
            axes[ax_i].plot(time_sec, sh_um[:, ax_i], label=name, color=color, alpha=0.85)
    for ax, lab in zip(axes, labels):
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (sec)")
    axes[0].legend(fontsize=6, loc="upper left", ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_mse_bar(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df = summary_df.sort_values("mse_full")
    axes[0].barh(df["method"], df["mse_full"], color="steelblue")
    axes[0].set_xlabel("MSE full FOV")
    axes[0].set_title("Full max-proj MSE")
    df2 = summary_df.sort_values("mse_roi")
    axes[1].barh(df2["method"], df2["mse_roi"], color="coral")
    axes[1].set_xlabel("MSE ROI trim")
    axes[1].set_title("ROI max-proj MSE")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_roi_overlay_grid(
    raw_stack: np.ndarray,
    results: dict,
    roi_bounds: tuple,
    plot_frames: tuple[int, ...],
    out_path: Path,
    title: str,
) -> None:
    z0, z1, y0, y1, x0, x1 = roi_bounds
    ref_roi = np.max(raw_stack[0, z0:z1, y0:y1, x0:x1], axis=0)
    vmax = float(np.percentile(ref_roi, 99.5))

    methods = ["00_raw"] + list(results.keys())
    n_rows = len(plot_frames)
    n_cols = len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for row, t in enumerate(plot_frames):
        if t >= raw_stack.shape[0]:
            continue
        for col, name in enumerate(methods):
            ax = axes[row, col]
            if name == "00_raw":
                img = np.max(raw_stack[t, z0:z1, y0:y1, x0:x1], axis=0)
            else:
                _, aligned = results[name]
                img = np.max(aligned[t, z0:z1, y0:y1, x0:x1], axis=0)
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            if row == 0:
                ax.set_title(name, fontsize=6)
            if col == 0:
                ax.set_ylabel(f"t={t}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_shift_jump_plot(
    results_um: dict,
    jump_frame: int,
    out_path: Path,
    title: str,
) -> None:
    """Bar plot of Y/X shift at a problematic frame (e.g. motor jump)."""
    names = list(results_um.keys())
    y_vals = [results_um[n][jump_frame, 1] for n in names]
    x_vals = [results_um[n][jump_frame, 2] for n in names]
    x_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x_pos - 0.2, y_vals, width=0.4, label="Y drift (um)")
    ax.bar(x_pos + 0.2, x_vals, width=0.4, label="X drift (um)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel("Shift vs frame 0 (um)")
    ax.legend()
    ax.set_title(f"{title} — frame {jump_frame}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_one(flim_path: str) -> None:
    ch = CHANNEL_1OR2 - 1
    stem = Path(flim_path).stem
    out_dir = _SCRIPT_DIR / "output" / stem / "edge_mitigation"
    out_dir.mkdir(parents=True, exist_ok=True)

    filelist = get_flimfile_list(flim_path)
    stack, motors, time_sec = _load_stack_and_motor(filelist, ch)
    _, iminfo, _ = flim_files_to_nparray(filelist, ch=ch)
    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)

    unc_info, roi_spec = uncaging_info_from_series(
        filelist,
        ch,
        half_z=SMALL_Z_PLUS_MINUS,
        half_y=SMALL_REGION_SIZE // 2,
        half_x=SMALL_REGION_SIZE // 2,
    )
    if roi_spec is None:
        print(f"Skip {stem}: no uncaging ROI")
        return

    _, roi_bounds = crop_zyx(stack[0], roi_spec)
    print(f"\n=== {stem} ===")
    print(f"ROI center ZYX=({roi_spec.z},{roi_spec.y},{roi_spec.x})")

    results = run_edge_mitigation_methods(
        stack, roi_spec, motors, z_um, y_um, x_um
    )

    summary_rows = []
    results_um = {}
    max_shift_px = int(np.ceil(np.abs(
        np.array([s for s, _ in results.values()])
    ).max()))

    for name, (shifts, aligned) in results.items():
        sh_um = shifts_to_um(shifts, z_um, y_um, x_um)
        results_um[name] = sh_um
        mse_full = alignment_mse_vs_ref(
            aligned,
            exclude_shift_margin_px=max_shift_px,
        )
        mse_roi = alignment_mse_vs_ref(aligned, roi_bounds=roi_bounds)
        summary_rows.append({
            "method": name,
            "mse_full": mse_full,
            "mse_roi": mse_roi,
            "final_y_um": sh_um[-1, 1],
            "final_x_um": sh_um[-1, 2],
            "max_y_um": np.abs(sh_um[:, 1]).max(),
            "max_x_um": np.abs(sh_um[:, 2]).max(),
        })
        print(
            f"  {name}: mse_roi={mse_roi:.1f}  "
            f"final_YX=({sh_um[-1,1]:.2f},{sh_um[-1,2]:.2f}) um"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    frames_plot = tuple(t for t in PLOT_FRAMES if t < stack.shape[0])
    _save_drift_plot(
        results_um, time_sec,
        out_dir / "drift_traces.png",
        f"{stem} — edge mitigation drift",
    )
    _save_mse_bar(
        summary_df, out_dir / "mse_comparison.png", stem
    )
    _save_roi_overlay_grid(
        stack, results, roi_bounds, frames_plot,
        out_dir / "roi_overlay_grid.png",
        f"{stem} ROI trim (rows=frames, cols=methods)",
    )
    if 7 in frames_plot:
        _save_shift_jump_plot(
            results_um, 7, out_dir / "shift_at_frame7.png", stem
        )

    print(f"Plots saved to: {out_dir}")


def main() -> None:
    for flim_path in FLIM_PATHS:
        process_one(flim_path)


if __name__ == "__main__":
    main()
