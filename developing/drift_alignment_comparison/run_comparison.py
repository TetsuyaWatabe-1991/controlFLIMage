# -*- coding: utf-8 -*-
"""
Compare drift alignment methods on FLIM time series.

Run (example):
  C:\\Users\\yasudalab\\AppData\\Local\\anaconda3\\python.exe run_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
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
from align_methods import (  # noqa: E402
    RoiCropSpec,
    alignment_mse_vs_ref,
    crop_zyx,
    run_selected_methods,
    shifts_to_um,
)
from uncaging_utils import uncaging_info_from_series  # noqa: E402

# --- edit here ---
FLIM_PATHS = [
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_1_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_2_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_3_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_4_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_5_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_6_002.flim",
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy\AP5_pos6__highmag_7_002.flim",
]
CHANNEL_1OR2 = 2  # 1=Ch1/GFP, 2=Ch2/RFP
SELECTED_METHODS = [
    "01_traditional",
    "02_xy_z_split",
    "04_structure_laplacian",
]

# ROI trim size (half-extents per axis: Z, Y, X).
# Matches gui_integration.process_small_region defaults:
#   small_region_size=60  -> XY half = 30 px (~4 um at high mag)
#   small_z_plus_minus=2  -> Z half = 2 slices
# See also controlflimage_threading.cuboid_ZYX=[1,20,20] for live AlignSmallRegion.
SMALL_REGION_SIZE = 60
SMALL_Z_PLUS_MINUS = 2
ROI_HALF_ZYX = (
    SMALL_Z_PLUS_MINUS,
    SMALL_REGION_SIZE // 2,
    SMALL_REGION_SIZE // 2,
)


def _save_drift_plot(
    results_um: dict,
    time_sec: list,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["Z drift (um)", "Y drift (um)", "X drift (um)"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_um)))

    for (name, shifts_um), color in zip(results_um.items(), colors):
        for ax_idx in range(3):
            axes[ax_idx].plot(
                time_sec,
                shifts_um[:, ax_idx],
                label=name,
                color=color,
                alpha=0.85,
            )

    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (sec)")
    axes[0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Drift estimates by method")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_maxproj_comparison(
    raw_stack: np.ndarray,
    aligned_results: dict,
    out_path: Path,
    vmax_percentile: float = 99.5,
) -> None:
    ref_mp = np.max(raw_stack[0], axis=0)
    last_mp_raw = np.max(raw_stack[-1], axis=0)
    vmax = np.percentile(ref_mp, vmax_percentile)

    n_methods = len(aligned_results)
    n_cols = 3
    n_rows = int(np.ceil((n_methods + 2) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    panels = [
        ("raw frame 0", ref_mp),
        ("raw last frame", last_mp_raw),
    ]
    for name, (_, aligned) in aligned_results.items():
        panels.append((f"{name}\nlast aligned", np.max(aligned[-1], axis=0)))

    for idx, (title, img) in enumerate(panels):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for idx in range(len(panels), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("Max projection: reference vs last frame (aligned)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_gray_png(
    path: Path,
    img2d: np.ndarray,
    vmax: float,
) -> None:
    """Save a 2D grayscale image as PNG (scaled to uint8 using shared vmax)."""
    if vmax <= 0:
        vmax = 1.0
    scaled = np.clip(img2d.astype(np.float64) / vmax * 255.0, 0, 255).astype(np.uint8)
    mpimg.imsave(path, scaled, cmap="gray")


def _save_all_frames(
    method_name: str,
    aligned_stack: np.ndarray,
    out_frames_dir: Path,
    *,
    roi_bounds: tuple[int, int, int, int, int, int] | None = None,
    vmax_percentile: float = 99.5,
) -> None:
    """
    Save every frame under method-specific subfolders (one output type per folder).

    frames/<method>/
      zproj/     full-FOV max-Z projection per frame (PNG)
      roi_trim/  trimmed ROI max-Z projection per frame (PNG)
      zyx/       per-Z-slice PNG per frame (frame_XXX_zYY.png)
    """
    method_dir = out_frames_dir / method_name
    zproj_dir = method_dir / "zproj"
    roi_dir = method_dir / "roi_trim"
    zyx_dir = method_dir / "zyx"
    zproj_dir.mkdir(parents=True, exist_ok=True)
    zyx_dir.mkdir(parents=True, exist_ok=True)
    if roi_bounds is not None:
        roi_dir.mkdir(parents=True, exist_ok=True)

    z0, z1, y0, y1, x0, x1 = roi_bounds if roi_bounds is not None else (0, 0, 0, 0, 0, 0)

    zproj_vmax = float(np.percentile(np.max(aligned_stack[0], axis=0), vmax_percentile))
    roi_vmax = None
    if roi_bounds is not None:
        roi_vmax = float(
            np.percentile(
                np.max(aligned_stack[0, z0:z1, y0:y1, x0:x1], axis=0),
                vmax_percentile,
            )
        )
    zyx_vmax = float(np.percentile(aligned_stack[0], vmax_percentile))

    for t_idx in range(aligned_stack.shape[0]):
        vol = aligned_stack[t_idx]
        tag = f"frame_{t_idx:03d}"

        _save_gray_png(zproj_dir / f"{tag}.png", np.max(vol, axis=0), zproj_vmax)

        if roi_bounds is not None and roi_vmax is not None:
            _save_gray_png(
                roi_dir / f"{tag}.png",
                np.max(vol[z0:z1, y0:y1, x0:x1], axis=0),
                roi_vmax,
            )

        for z_idx in range(vol.shape[0]):
            _save_gray_png(
                zyx_dir / f"{tag}_z{z_idx:02d}.png",
                vol[z_idx],
                zyx_vmax,
            )


def _save_raw_all_frames(
    raw_stack: np.ndarray,
    out_frames_dir: Path,
    *,
    roi_bounds: tuple[int, int, int, int, int, int] | None = None,
) -> None:
    """Save unaligned raw frames for side-by-side comparison."""
    _save_all_frames("00_raw", raw_stack, out_frames_dir, roi_bounds=roi_bounds)


def _save_roi_zoom_comparison(
    raw_stack: np.ndarray,
    aligned_results: dict,
    roi_spec: RoiCropSpec,
    out_path: Path,
) -> None:
    _, (z0, z1, y0, y1, x0, x1) = crop_zyx(raw_stack[0], roi_spec)
    ref_roi = np.max(raw_stack[0, z0:z1, y0:y1, x0:x1], axis=0)
    vmax = np.percentile(ref_roi, 99.5)

    n_methods = len(aligned_results)
    n_cols = 4
    n_rows = int(np.ceil((n_methods + 2) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_2d(axes)

    panels = [
        ("raw frame 0 ROI", ref_roi),
        ("raw last ROI", np.max(raw_stack[-1, z0:z1, y0:y1, x0:x1], axis=0)),
    ]
    for name, (_, aligned) in aligned_results.items():
        panels.append(
            (
                name,
                np.max(aligned[-1, z0:z1, y0:y1, x0:x1], axis=0),
            )
        )

    for idx, (title, img) in enumerate(panels):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    for idx in range(len(panels), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"ROI max-proj Z[{z0}:{z1}] Y[{y0}:{y1}] X[{x0}:{x1}] "
        f"(center ZYX=({roi_spec.z},{roi_spec.y},{roi_spec.x}))"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_one_flim(flim_path: str) -> dict | None:
    """Run selected alignment methods for one FLIM series."""
    ch = CHANNEL_1OR2 - 1
    out_dir = _SCRIPT_DIR / "output" / Path(flim_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    filelist = get_flimfile_list(flim_path)
    print(f"\n{'=' * 60}")
    print(f"Processing: {flim_path}")
    print(f"Series files: {len(filelist)}")

    unc_info, roi_spec = uncaging_info_from_series(
        filelist,
        ch,
        half_z=ROI_HALF_ZYX[0],
        half_y=ROI_HALF_ZYX[1],
        half_x=ROI_HALF_ZYX[2],
    )
    if unc_info is None or roi_spec is None:
        print("WARNING: No uncaging file found; skipping.")
        return None

    print(f"Uncaging file: {unc_info.file_path}")
    print(
        f"Uncaging point (frame {unc_info.frame_index}): "
        f"Z={unc_info.center_z}, Y={unc_info.center_y:.1f}, X={unc_info.center_x:.1f}"
    )
    print(f"ROI trim center ZYX=({roi_spec.z}, {roi_spec.y}, {roi_spec.x})")

    stack, iminfo, time_sec = flim_files_to_nparray(filelist, ch=ch)
    print(f"Stack shape (T,Z,Y,X): {stack.shape}")

    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)

    results = run_selected_methods(stack, SELECTED_METHODS)

    summary_rows = []
    shift_csv_rows = []
    results_um = {}

    for name, (shifts, aligned) in results.items():
        shifts_um = shifts_to_um(shifts, z_um, y_um, x_um)
        results_um[name] = shifts_um
        mse = alignment_mse_vs_ref(aligned, ref_index=0)
        summary_rows.append(
            {
                "method": name,
                "mse_maxproj_vs_ref": mse,
                "final_drift_z_um": shifts_um[-1, 0],
                "final_drift_y_um": shifts_um[-1, 1],
                "final_drift_x_um": shifts_um[-1, 2],
            }
        )
        for t_idx in range(shifts.shape[0]):
            shift_csv_rows.append(
                {
                    "method": name,
                    "frame": t_idx,
                    "time_sec": time_sec[t_idx] if t_idx < len(time_sec) else t_idx,
                    "shift_z_px": shifts[t_idx, 0],
                    "shift_y_px": shifts[t_idx, 1],
                    "shift_x_px": shifts[t_idx, 2],
                    "shift_z_um": shifts_um[t_idx, 0],
                    "shift_y_um": shifts_um[t_idx, 1],
                    "shift_x_um": shifts_um[t_idx, 2],
                }
            )
        print(
            f"  {name}: MSE={mse:.2f}, "
            f"final drift (um) Z={shifts_um[-1,0]:.2f} "
            f"Y={shifts_um[-1,1]:.2f} X={shifts_um[-1,2]:.2f}"
        )

    pd.DataFrame(
        [{
            "flim_series": Path(flim_path).stem,
            "uncaging_file": unc_info.file_path,
            "uncaging_frame_index": unc_info.frame_index,
            "uncaging_center_z": unc_info.center_z,
            "uncaging_center_y": unc_info.center_y,
            "uncaging_center_x": unc_info.center_x,
            "roi_center_z": roi_spec.z,
            "roi_center_y": roi_spec.y,
            "roi_center_x": roi_spec.x,
            "roi_half_z": roi_spec.half_z,
            "roi_half_y": roi_spec.half_y,
            "roi_half_x": roi_spec.half_x,
        }]
    ).to_csv(out_dir / "uncaging_roi.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("mse_maxproj_vs_ref")
    shifts_df = pd.DataFrame(shift_csv_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    shifts_df.to_csv(out_dir / "shifts_all_methods.csv", index=False)

    _save_drift_plot(results_um, time_sec, out_dir / "drift_traces.png")
    _save_maxproj_comparison(stack, results, out_dir / "maxproj_last_frame.png")
    _save_roi_zoom_comparison(stack, results, roi_spec, out_dir / "roi_zoom_last_frame.png")

    frames_dir = out_dir / "frames"
    _, roi_bounds = crop_zyx(stack[0], roi_spec)
    print(f"Saving all frames to: {frames_dir}")
    _save_raw_all_frames(stack, frames_dir, roi_bounds=roi_bounds)
    for name, (_, aligned) in results.items():
        _save_all_frames(name, aligned, frames_dir, roi_bounds=roi_bounds)

    print(f"Results saved to: {out_dir}")
    return {
        "flim_series": Path(flim_path).stem,
        "out_dir": str(out_dir),
        "summary": summary_df,
    }


def main() -> None:
    batch_summary = []
    for flim_path in FLIM_PATHS:
        result = process_one_flim(flim_path)
        if result is not None:
            best = result["summary"].iloc[0]
            batch_summary.append(
                {
                    "flim_series": result["flim_series"],
                    "best_method": best["method"],
                    "best_mse": best["mse_maxproj_vs_ref"],
                }
            )

    if batch_summary:
        batch_df = pd.DataFrame(batch_summary)
        batch_out = _SCRIPT_DIR / "output" / "batch_summary.csv"
        batch_df.to_csv(batch_out, index=False)
        print(f"\nBatch summary saved to: {batch_out}")
        print(batch_df.to_string(index=False))


if __name__ == "__main__":
    main()
