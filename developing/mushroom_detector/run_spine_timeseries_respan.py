# -*- coding: utf-8 -*-
"""
Align a high-mag FLIM time series and quantify spine ROI intensity over time.

Uses seg_masks/{stem}_spine_outline_mask.tif (dilated spine outline, 2D) stacked
over head Z +/- z_half_window. FLIMageAlignment global + per-spine local crop
re-alignment (gui_integration.process_small_region pattern).

Outputs:
  - spine_timeseries.csv
  - alignment_shifts.csv (global)
  - local_alignment_shifts.csv (per-spine crop re-alignment)
  - zproj_overlays/{spine_stem}/tNNN.png  (global align)
  - zproj_overlays_local/{spine_stem}/tNNN.png  (local re-align)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf
from skimage.measure import find_contours

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _SCRIPT_DIR.parents[1]
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

from FLIMageAlignment import Align_4d_array, get_flimfile_list  # noqa: E402


def load_and_align_flim_series(
    filelist: list[str], ch: int
) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
    """Load FLIM series, skipping shape-mismatched files, then 3D-align to frame 0."""
    used_paths: list[str] = []
    arrays: list[np.ndarray] = []
    relative_sec_list: list[float] = []
    ref_shape = None

    from FLIMageFileReader2 import FileReader
    from datetime import datetime

    for file_path in filelist:
        iminfo = FileReader()
        print(file_path)
        iminfo.read_imageFile(file_path, True)
        imagearray = np.array(iminfo.image)
        div_by = iminfo.State.Acq.nAveFrame
        if ref_shape is None:
            ref_shape = imagearray.shape
        if imagearray.shape != ref_shape:
            print(f"{file_path} <- skipped read")
            continue
        intensityarray = (12 * np.sum(imagearray, axis=-1)) / div_by
        used_paths.append(file_path)
        arrays.append(intensityarray)
        acq_time = datetime.strptime(iminfo.acqTime[0], "%Y-%m-%dT%H:%M:%S.%f")
        if not relative_sec_list:
            t0 = acq_time
            relative_sec_list.append(0.0)
        else:
            relative_sec_list.append((acq_time - t0).total_seconds())

    if not arrays:
        raise ValueError("No FLIM files loaded with a consistent shape.")

    print("ch", ch)
    tiff_multi = np.array(arrays, dtype=np.uint16)[:, :, 0, ch, :, :]
    shifts, aligned = Align_4d_array(tiff_multi)
    return aligned, shifts, relative_sec_list, used_paths

DEFAULT_REFERENCE_FLIM = (
    r"G:\ImagingData\Tetsuya\20260608\mushroom_1dend\pos1__highmag_1_002.flim"
)
DEFAULT_CHANNEL = 2
DEFAULT_Z_HALF_WINDOW = 2
DEFAULT_SMALL_REGION_SIZE = 60
DEFAULT_SMALL_Z_HALF_WINDOW = 2
DEFAULT_NORM_PERCENTILE = 10.0
INTENSITY_COL_LOCAL = "spine_mean_intensity_local"
INTENSITY_COL_GLOBAL = "spine_mean_intensity"
MASK_CONTOUR_COLOR = "cyan"


def _spine_stems_from_feature_csv(savefolder: Path, base_name: str) -> list[str]:
    csv_path = savefolder / f"{base_name}_respan_mushroom_features.csv"
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    stems: list[str] = []
    for _, row in df.iterrows():
        ini_path = Path(str(row["ini_path"]))
        stems.append(ini_path.stem)
    return stems


def _discover_spine_stems(savefolder: Path, base_name: str) -> list[str]:
    stems = _spine_stems_from_feature_csv(savefolder, base_name)
    if stems:
        return stems
    seg_dir = savefolder / "seg_masks"
    paths = sorted(seg_dir.glob(f"{base_name}_*_spine_outline_mask.tif"))
    return [p.name.replace("_spine_outline_mask.tif", "") for p in paths]


def _load_spine_outline_mask_2d(mask_path: Path) -> np.ndarray:
    mask = tf.imread(mask_path)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = np.any(arr, axis=0)
    return arr > 0


def _outline_mask_to_3d(
    outline_2d: np.ndarray,
    z0: int,
    z1: int,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Replicate 2D dilated outline across Z slices in [z0, z1)."""
    n_z, height, width = shape_zyx
    mask_3d = np.zeros((n_z, height, width), dtype=bool)
    oh = min(outline_2d.shape[0], height)
    ow = min(outline_2d.shape[1], width)
    mask_3d[z0:z1, :oh, :ow] = outline_2d[:oh, :ow]
    return mask_3d


def _z_window(z_center: int, n_z: int, z_half: int) -> tuple[int, int]:
    z0 = max(0, int(z_center) - z_half)
    z1 = min(n_z, int(z_center) + z_half + 1)
    return z0, max(z1, z0 + 1)


def _spine_crop_bounds(
    head_y: float,
    head_x: float,
    head_z: float,
    shape_zyx: tuple[int, int, int],
    *,
    small_region_size: int,
    z_half_window: int,
) -> tuple[int, int, int, int, int, int]:
    """Crop box (z0,z1,y0,y1,x0,x1) centered on spine head (gui_integration style)."""
    half = int(small_region_size / 2)
    n_z, height, width = shape_zyx
    z0, z1 = _z_window(int(round(head_z)), n_z, z_half_window)
    cy = int(round(head_y))
    cx = int(round(head_x))
    y0 = int(max(0, cy - half))
    y1 = int(min(height, cy + half + 1))
    x0 = int(max(0, cx - half))
    x1 = int(min(width, cx + half + 1))
    return z0, z1, y0, y1, x0, x1


def align_spine_local_crop(
    aligned_4d: np.ndarray,
    bounds: tuple[int, int, int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Second-stage Align_4d_array on a spine-centered crop (process_small_region pattern)."""
    z0, z1, y0, y1, x0, x1 = bounds
    crop = aligned_4d[:, z0:z1, y0:y1, x0:x1]
    small_shifts, small_aligned = Align_4d_array(crop)
    return small_shifts, small_aligned


def _mask_for_z_window(mask_3d: np.ndarray, z0: int, z1: int) -> np.ndarray:
    out = np.zeros_like(mask_3d, dtype=bool)
    out[z0:z1] = mask_3d[z0:z1]
    return out


def _plot_zproj_with_mask(
    zproj: np.ndarray,
    mask_2d: np.ndarray,
    *,
    title: str,
    savepath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    vmax = float(np.percentile(zproj, 99.5)) if zproj.size else 1.0
    ax.imshow(zproj, cmap="gray", vmin=0, vmax=max(vmax, 1.0), origin="upper")
    if mask_2d.any():
        for contour in find_contours(mask_2d.astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color=MASK_CONTOUR_COLOR, linewidth=1.2)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.savefig(savepath, dpi=120, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _save_montage(image_paths: list[Path], savepath: Path, *, ncols: int = 6) -> None:
    if not image_paths:
        return
    import matplotlib.image as mpimg

    n = len(image_paths)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    axes = np.atleast_2d(axes)
    for idx, img_path in enumerate(image_paths):
        r, c = divmod(idx, ncols)
        axes[r, c].imshow(mpimg.imread(str(img_path)))
        axes[r, c].set_title(img_path.stem, fontsize=7)
        axes[r, c].axis("off")
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")
    fig.savefig(savepath, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _pick_intensity_column(ts_df: pd.DataFrame, prefer_local: bool = True) -> str:
    if prefer_local and INTENSITY_COL_LOCAL in ts_df.columns:
        if ts_df[INTENSITY_COL_LOCAL].notna().any():
            return INTENSITY_COL_LOCAL
    return INTENSITY_COL_GLOBAL


def _normalize_by_lower_percentile(
    values: np.ndarray,
    percentile: float = DEFAULT_NORM_PERCENTILE,
) -> tuple[np.ndarray, float]:
    """Divide by the lower percentile so that percentile value equals 1.0."""
    clean = values.astype(float)
    ref = float(np.nanpercentile(clean, percentile))
    if not np.isfinite(ref) or ref <= 0:
        ref = 1.0
    return clean / ref, ref


def save_spine_quant_plots(
    ts_df: pd.DataFrame,
    overlay_local_dir: Path,
    *,
    intensity_col: str | None = None,
    norm_percentile: float = DEFAULT_NORM_PERCENTILE,
) -> list[Path]:
    """Save per-spine normalized quant plots under zproj_overlays_local/{stem}/."""
    if ts_df.empty:
        return []

    overlay_local_dir = Path(overlay_local_dir)
    overlay_local_dir.mkdir(parents=True, exist_ok=True)
    if intensity_col is None:
        intensity_col = _pick_intensity_column(ts_df)

    saved: list[Path] = []

    for stem in sorted(ts_df["spine_stem"].unique()):
        sub = ts_df[ts_df["spine_stem"] == stem].sort_values("frame_index").copy()
        y_raw = sub[intensity_col].to_numpy(dtype=float)
        y_norm, p_ref = _normalize_by_lower_percentile(y_raw, norm_percentile)
        sub["intensity_norm_p10"] = y_norm
        sub["norm_ref_p10"] = p_ref

        x = sub["relative_sec"].to_numpy(dtype=float)
        if not np.any(np.isfinite(x)):
            x = sub["frame_index"].to_numpy(dtype=float)
            x_label = "Frame index"
        else:
            x_label = "Time (s)"

        spine_dir = overlay_local_dir / stem
        spine_dir.mkdir(parents=True, exist_ok=True)
        plot_path = spine_dir / "quant_timeseries_norm_p10.png"
        csv_path = spine_dir / "quant_timeseries_norm_p10.csv"

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y_norm, "o-", color="#1f77b4", linewidth=1.5, markersize=5)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="p10 = 1")
        respan_id = sub["respan_spine_id"].iloc[0] if "respan_spine_id" in sub.columns else ""
        ax.set_title(
            f"{stem} (RESPAN {respan_id})\n"
            f"norm = {intensity_col} / p{norm_percentile:g} ({p_ref:.4g})",
            fontsize=10,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"Normalized intensity (p{norm_percentile:g} = 1)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        sub.to_csv(csv_path, index=False)
        saved.append(plot_path)
        print(f"  quant plot: {plot_path}")

    return saved


def plot_timeseries_from_csv(
    timeseries_csv: str | Path,
    overlay_local_dir: str | Path | None = None,
    *,
    norm_percentile: float = DEFAULT_NORM_PERCENTILE,
) -> list[Path]:
    """Regenerate quant plots from an existing spine_timeseries.csv."""
    timeseries_csv = Path(timeseries_csv)
    ts_df = pd.read_csv(timeseries_csv)
    if overlay_local_dir is None:
        overlay_local_dir = timeseries_csv.parent / "zproj_overlays_local"
    return save_spine_quant_plots(
        ts_df,
        Path(overlay_local_dir),
        norm_percentile=norm_percentile,
    )


def run_spine_timeseries(
    reference_flim: str | Path,
    *,
    channel: int = DEFAULT_CHANNEL,
    z_half_window: int = DEFAULT_Z_HALF_WINDOW,
    small_region_size: int = DEFAULT_SMALL_REGION_SIZE,
    local_align: bool = True,
    output_subdir: str = "spine_timeseries",
) -> Path:
    reference_flim = Path(reference_flim)
    if not reference_flim.is_file():
        raise FileNotFoundError(f"Reference FLIM not found: {reference_flim}")

    savefolder = Path(str(reference_flim)[:-9])
    if not savefolder.is_dir():
        savefolder = reference_flim.parent / reference_flim.stem
    base_name = os.path.basename(str(reference_flim)[:-9])

    filelist = get_flimfile_list(str(reference_flim))
    if reference_flim.as_posix() not in [os.path.normpath(p) for p in filelist]:
        filelist = [str(reference_flim)] + filelist
    filelist = sorted(set(filelist), key=lambda p: os.path.basename(p))

    print(f"Reference FLIM: {reference_flim}")
    print(f"Time series files: {len(filelist)}")
    for path in filelist:
        print(f"  {os.path.basename(path)}")

    ch_idx = channel - 1
    aligned_4d, shifts, relative_sec_list, used_filelist = load_and_align_flim_series(
        filelist, ch_idx
    )
    if len(used_filelist) != len(filelist):
        skipped = set(filelist) - set(used_filelist)
        print(f"WARNING: skipped {len(skipped)} file(s) with mismatched shape:")
        for path in sorted(skipped):
            print(f"  {os.path.basename(path)}")
    filelist = used_filelist
    n_time, n_z, height, width = aligned_4d.shape
    print(f"Aligned stack shape (T,Z,Y,X): {aligned_4d.shape}")

    out_dir = savefolder / output_subdir
    overlay_dir = out_dir / "zproj_overlays"
    overlay_local_dir = out_dir / "zproj_overlays_local"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if local_align:
        overlay_local_dir.mkdir(parents=True, exist_ok=True)

    shift_rows = []
    for t, flim_path in enumerate(filelist):
        shift_rows.append(
            {
                "frame_index": t,
                "flim_path": flim_path,
                "flim_name": os.path.basename(flim_path),
                "shift_z": float(shifts[t, 0]),
                "shift_y": float(shifts[t, 1]),
                "shift_x": float(shifts[t, 2]),
                "relative_sec": float(relative_sec_list[t]) if t < len(relative_sec_list) else np.nan,
            }
        )
    pd.DataFrame(shift_rows).to_csv(out_dir / "alignment_shifts.csv", index=False)

    feature_csv = savefolder / f"{base_name}_respan_mushroom_features.csv"
    feature_df = pd.read_csv(feature_csv) if feature_csv.is_file() else pd.DataFrame()
    spine_stems = _discover_spine_stems(savefolder, base_name)
    if not spine_stems:
        raise FileNotFoundError(f"No spine outline masks found under {savefolder / 'seg_masks'}")

    ts_rows: list[dict] = []
    local_shift_rows: list[dict] = []
    for stem in spine_stems:
        mask_path = savefolder / "seg_masks" / f"{stem}_spine_outline_mask.tif"
        if not mask_path.is_file():
            print(f"  skip missing outline mask: {mask_path.name}")
            continue

        outline_2d = _load_spine_outline_mask_2d(mask_path)
        stack_shape = (n_z, height, width)
        oh = min(outline_2d.shape[0], height)
        ow = min(outline_2d.shape[1], width)
        outline_2d = outline_2d[:oh, :ow]
        aligned_use = aligned_4d

        ys_x, xs_x = np.where(outline_2d)
        head_y = float(np.mean(ys_x)) if len(ys_x) else height / 2
        head_x = float(np.mean(xs_x)) if len(xs_x) else width / 2
        head_z = n_z // 2
        if not feature_df.empty and "ini_path" in feature_df.columns:
            match = feature_df[feature_df["ini_path"].astype(str).str.endswith(f"{stem}.ini")]
            if len(match):
                head_z = int(round(float(match.iloc[0]["head_z_pix"])))
                head_y = float(match.iloc[0]["head_y_pix"])
                head_x = float(match.iloc[0]["head_x_pix"])

        z0, z1 = _z_window(head_z, n_z, z_half_window)
        mask_3d = _outline_mask_to_3d(outline_2d, z0, z1, stack_shape)
        mask_z = mask_3d
        mask_2d_mip = outline_2d

        crop_bounds = _spine_crop_bounds(
            head_y,
            head_x,
            head_z,
            stack_shape,
            small_region_size=small_region_size,
            z_half_window=z_half_window,
        )
        cz0, cz1, cy0, cy1, cx0, cx1 = crop_bounds
        mask_crop = mask_3d[cz0:cz1, cy0:cy1, cx0:cx1]
        mask_crop_z = mask_crop
        mask_crop_2d_mip = outline_2d[cy0:cy1, cx0:cx1]

        small_shifts: np.ndarray | None = None
        small_aligned: np.ndarray | None = None
        if local_align:
            small_shifts, small_aligned = align_spine_local_crop(aligned_use, crop_bounds)
            print(
                f"  local align {stem}: crop Z[{cz0}:{cz1}] Y[{cy0}:{cy1}] X[{cx0}:{cx1}] "
                f"-> {small_aligned.shape}"
            )

        spine_overlay_dir = overlay_dir / stem
        spine_overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_paths: list[Path] = []
        overlay_local_paths: list[Path] = []
        if local_align:
            spine_overlay_local_dir = overlay_local_dir / stem
            spine_overlay_local_dir.mkdir(parents=True, exist_ok=True)

        respan_id = np.nan
        if not feature_df.empty and "ini_path" in feature_df.columns:
            match = feature_df[feature_df["ini_path"].astype(str).str.endswith(f"{stem}.ini")]
            if len(match) and "respan_spine_id" in match.columns:
                respan_id = int(match.iloc[0]["respan_spine_id"])

        for t, flim_path in enumerate(filelist):
            vol = aligned_use[t]
            masked_vals = vol[mask_z]
            mean_int = float(masked_vals.mean()) if masked_vals.size else np.nan
            sum_int = float(masked_vals.sum()) if masked_vals.size else np.nan
            n_vox = int(mask_z.sum())

            mean_int_local = np.nan
            sum_int_local = np.nan
            if local_align and small_aligned is not None:
                vol_local = small_aligned[t]
                masked_local = vol_local[mask_crop_z]
                mean_int_local = float(masked_local.mean()) if masked_local.size else np.nan
                sum_int_local = float(masked_local.sum()) if masked_local.size else np.nan
                local_shift_rows.append(
                    {
                        "spine_stem": stem,
                        "respan_spine_id": respan_id,
                        "frame_index": t,
                        "flim_name": os.path.basename(flim_path),
                        "small_shift_z": float(small_shifts[t, 0]),
                        "small_shift_y": float(small_shifts[t, 1]),
                        "small_shift_x": float(small_shifts[t, 2]),
                        "crop_z_from": cz0,
                        "crop_z_to": cz1 - 1,
                        "crop_y_from": cy0,
                        "crop_y_to": cy1 - 1,
                        "crop_x_from": cx0,
                        "crop_x_to": cx1 - 1,
                        "small_region_size": small_region_size,
                    }
                )

            zproj = vol[z0:z1].max(axis=0)
            flim_name = os.path.basename(flim_path)
            overlay_path = spine_overlay_dir / f"t{t:03d}_{Path(flim_name).stem}.png"
            _plot_zproj_with_mask(
                zproj,
                mask_2d_mip,
                title=f"{stem} global t={t} {flim_name[-7:-5]}",
                savepath=overlay_path,
            )
            overlay_paths.append(overlay_path)

            if local_align and small_aligned is not None:
                lz0, lz1 = z0 - cz0, z1 - cz0
                zproj_local = small_aligned[t, lz0:lz1].max(axis=0)
                overlay_local_path = spine_overlay_local_dir / f"t{t:03d}_{Path(flim_name).stem}.png"
                _plot_zproj_with_mask(
                    zproj_local,
                    mask_crop_2d_mip,
                    title=f"{stem} local t={t} {flim_name[-7:-5]}",
                    savepath=overlay_local_path,
                )
                overlay_local_paths.append(overlay_local_path)

            ts_rows.append(
                {
                    "spine_stem": stem,
                    "respan_spine_id": respan_id,
                    "roi_mask": "spine_outline",
                    "frame_index": t,
                    "flim_path": flim_path,
                    "flim_name": flim_name,
                    "relative_sec": float(relative_sec_list[t])
                    if t < len(relative_sec_list)
                    else np.nan,
                    "head_z_pix": head_z,
                    "head_y_pix": head_y,
                    "head_x_pix": head_x,
                    "z_from": z0,
                    "z_to": z1 - 1,
                    "n_mask_voxels": n_vox,
                    "shift_z": float(shifts[t, 0]),
                    "shift_y": float(shifts[t, 1]),
                    "shift_x": float(shifts[t, 2]),
                    "small_shift_z": float(small_shifts[t, 0]) if small_shifts is not None else np.nan,
                    "small_shift_y": float(small_shifts[t, 1]) if small_shifts is not None else np.nan,
                    "small_shift_x": float(small_shifts[t, 2]) if small_shifts is not None else np.nan,
                    "spine_mean_intensity": mean_int,
                    "spine_sum_intensity": sum_int,
                    "spine_mean_intensity_local": mean_int_local,
                    "spine_sum_intensity_local": sum_int_local,
                }
            )

        montage_path = overlay_dir / f"{stem}_montage.png"
        _save_montage(overlay_paths, montage_path)
        print(f"  spine {stem}: {len(overlay_paths)} global frames -> {montage_path.name}")
        if overlay_local_paths:
            montage_local_path = overlay_local_dir / f"{stem}_montage.png"
            _save_montage(overlay_local_paths, montage_local_path)
            print(f"  spine {stem}: {len(overlay_local_paths)} local frames -> {montage_local_path.name}")

    ts_df = pd.DataFrame(ts_rows)
    ts_csv = out_dir / "spine_timeseries.csv"
    ts_df.to_csv(ts_csv, index=False)
    if local_shift_rows:
        pd.DataFrame(local_shift_rows).to_csv(out_dir / "local_alignment_shifts.csv", index=False)
    if local_align and not ts_df.empty:
        save_spine_quant_plots(ts_df, overlay_local_dir)
    print(f"Saved: {ts_csv} ({len(ts_df)} rows)")
    print(f"Overlays (global): {overlay_dir}")
    if local_align:
        print(f"Overlays (local):  {overlay_local_dir}")
    return out_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spine-only FLIM time series with alignment.")
    parser.add_argument("--reference-flim", default=DEFAULT_REFERENCE_FLIM)
    parser.add_argument("--channel", type=int, default=DEFAULT_CHANNEL, choices=[1, 2])
    parser.add_argument("--z-half-window", type=int, default=DEFAULT_Z_HALF_WINDOW)
    parser.add_argument("--small-region-size", type=int, default=DEFAULT_SMALL_REGION_SIZE)
    parser.add_argument("--no-local-align", action="store_true")
    parser.add_argument("--output-subdir", default="spine_timeseries")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate quant plots from existing spine_timeseries.csv",
    )
    parser.add_argument("--timeseries-csv", default=None)
    parser.add_argument("--norm-percentile", type=float, default=DEFAULT_NORM_PERCENTILE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.plot_only:
        csv_path = args.timeseries_csv
        if csv_path is None:
            ref = Path(args.reference_flim)
            savefolder = Path(str(ref)[:-9])
            csv_path = savefolder / args.output_subdir / "spine_timeseries.csv"
        plot_timeseries_from_csv(
            csv_path,
            norm_percentile=args.norm_percentile,
        )
        return
    run_spine_timeseries(
        args.reference_flim,
        channel=args.channel,
        z_half_window=args.z_half_window,
        small_region_size=args.small_region_size,
        local_align=not args.no_local_align,
        output_subdir=args.output_subdir,
    )


if __name__ == "__main__":
    main()
