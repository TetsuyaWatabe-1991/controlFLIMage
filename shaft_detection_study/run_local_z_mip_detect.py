# -*- coding: utf-8 -*-
"""
Compare DeepD3 on native Z slices vs local Z-MIP input stacks (2*radius+1 planes).

Outputs next to the .flim (under resolve_savefolder / <subdir>/):
  compare_single_vs_localzmip.png
  S_shaft_single.tif, S_spine_single.tif
  S_shaft_localzmip.tif, S_spine_localzmip.tif
  metrics.txt

Example:
  python controlFLIMage/shaft_detection_study/run_local_z_mip_detect.py ^
    --flim "\\\\server\\path\\sample_256_4x_001.flim"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

_STUDY_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _STUDY_DIR.parent
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ.setdefault("PYTHONUTF8", "1")

from deepd3_spine_head_detector import resolve_savefolder
from shaft_detection_study.shaft_inference import (
    load_flim_stack,
    local_z_mip_stack,
    predict_deepd3_stack,
    raw_z_mip,
)
from shaft_detection_study.shaft_metrics import summarize_shaft_map
from shaft_detection_study.shaft_viz import save_single_vs_localzmip_compare

DEFAULT_MODEL = r"C:\Users\WatabeT\Documents\Git\DeepD3\deepd3\DeepD3_32F_94nm.h5"
DEFAULT_SUBDIR = "local_z_mip_r2"
DEFAULT_Z_RADIUS = 2
DEFAULT_STACK_PREPROCESS = "tophat_clahe"


def output_dir_for_flim(flim_path: str, subdir: str = DEFAULT_SUBDIR) -> str:
    """Folder beside .flim series (resolve_savefolder + subdir name)."""
    base = resolve_savefolder(flim_path)
    out = os.path.join(base, subdir)
    os.makedirs(out, exist_ok=True)
    return out


def resolve_flim_paths(
    flim: str | None,
    folder: str | None,
    pattern: str,
) -> list[str]:
    if flim:
        if not os.path.isfile(flim):
            raise FileNotFoundError(flim)
        return [flim]
    if folder:
        paths = sorted(glob.glob(os.path.join(folder, pattern)))
        if not paths:
            raise FileNotFoundError(
                f"No files for folder={folder!r} pattern={pattern!r}"
            )
        return paths
    raise ValueError("Provide --flim or --folder")


def write_metrics_txt(
    path: str,
    *,
    flim_path: str,
    z_radius: int,
    stack_preprocess: str,
    shaft_single: np.ndarray,
    shaft_local: np.ndarray,
    spine_single: np.ndarray,
    spine_local: np.ndarray,
) -> None:
    lines = [
        f"flim={flim_path}",
        f"z_radius={z_radius} (window={2 * z_radius + 1} planes)",
        f"stack_preprocess={stack_preprocess}",
        f"timestamp={datetime.now().isoformat()}",
        "",
        "--- shaft (dendrite) ---",
    ]
    for tag, arr in ("single_plane", shaft_single), ("local_z_mip", shaft_local):
        m = summarize_shaft_map(arr, tag=tag)
        lines.append(f"[{tag}]")
        for k, v in m.items():
            if k != "map_tag":
                lines.append(f"  {k}={v}")
        lines.append("")
    lines.append("--- spine ---")
    for tag, arr in ("single_plane", spine_single), ("local_z_mip", spine_local):
        m = summarize_shaft_map(arr, tag=tag)
        lines.append(f"[{tag}]")
        for k, v in m.items():
            if k != "map_tag":
                lines.append(f"  {k}={v}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_for_flim(
    flim_path: str,
    model_path: str,
    *,
    z_radius: int = DEFAULT_Z_RADIUS,
    stack_preprocess: str = DEFAULT_STACK_PREPROCESS,
    subdir: str = DEFAULT_SUBDIR,
    skip_single_plane: bool = False,
    save_input_local_mip_stack: bool = False,
    raw_vmax_percentile: float = 95.0,
    contour_level: float = 0.2,
    ch_1or2: int | None = None,
) -> str:
    out_dir = output_dir_for_flim(flim_path, subdir=subdir)
    print(f"\n=== {flim_path} ===")
    print(f"Output: {out_dir}")

    zyx, xy_um, z_um = load_flim_stack(flim_path, ch_1or2=ch_1or2)
    print(f"  ZYX={zyx.shape}  xy={xy_um:.3f} um  z={z_um:.3f} um")
    print(f"  local Z-MIP radius={z_radius} ({2 * z_radius + 1} planes)")

    zyx_local = local_z_mip_stack(zyx, radius=z_radius)
    mip_raw = raw_z_mip(zyx)

    if save_input_local_mip_stack:
        tif_path = os.path.join(out_dir, "input_local_z_mip_stack.tif")
        tifffile.imwrite(tif_path, zyx_local)
        print(f"  saved {tif_path}")

    if skip_single_plane:
        shaft_single = spine_single = None
    else:
        print("  DeepD3 on single-Z planes...")
        shaft_single, spine_single = predict_deepd3_stack(
            zyx, xy_um, z_um, model_path, stack_preprocess=stack_preprocess
        )
        tifffile.imwrite(
            os.path.join(out_dir, "S_shaft_single_plane.tif"), shaft_single
        )
        tifffile.imwrite(
            os.path.join(out_dir, "S_spine_single_plane.tif"), spine_single
        )

    print("  DeepD3 on local Z-MIP stack...")
    shaft_local, spine_local = predict_deepd3_stack(
        zyx_local, xy_um, z_um, model_path, stack_preprocess=stack_preprocess
    )
    tifffile.imwrite(os.path.join(out_dir, "S_shaft_localzmip.tif"), shaft_local)
    tifffile.imwrite(os.path.join(out_dir, "S_spine_localzmip.tif"), spine_local)

    compare_path = os.path.join(out_dir, "compare_single_vs_localzmip.png")
    if skip_single_plane:
        from shaft_detection_study.shaft_viz import save_triple_mip_png

        save_triple_mip_png(
            mip_raw,
            shaft_local,
            shaft_local,
            compare_path,
            title_suffix=f"local Z-MIP r={z_radius} (no single-plane baseline)",
            raw_vmax_percentile=raw_vmax_percentile,
            contour_level=contour_level,
        )
    else:
        save_single_vs_localzmip_compare(
            mip_raw,
            shaft_single,
            shaft_local,
            spine_single,
            spine_local,
            compare_path,
            z_radius=z_radius,
            raw_vmax_percentile=raw_vmax_percentile,
            contour_level=contour_level,
        )
    print(f"  saved {compare_path}")

    metrics_path = os.path.join(out_dir, "metrics.txt")
    if not skip_single_plane:
        write_metrics_txt(
            metrics_path,
            flim_path=flim_path,
            z_radius=z_radius,
            stack_preprocess=stack_preprocess,
            shaft_single=shaft_single,
            shaft_local=shaft_local,
            spine_single=spine_single,
            spine_local=spine_local,
        )
        print(f"  saved {metrics_path}")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepD3 on single Z planes vs local Z-MIP (radius window)."
    )
    parser.add_argument("--flim", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*256_4x_001.flim")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--z-radius", type=int, default=DEFAULT_Z_RADIUS)
    parser.add_argument(
        "--stack-preprocess",
        type=str,
        default=DEFAULT_STACK_PREPROCESS,
        choices=["none", "median", "tophat_clahe"],
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=DEFAULT_SUBDIR,
        help="Subfolder under flim savefolder for outputs",
    )
    parser.add_argument(
        "--skip-single-plane",
        action="store_true",
        help="Only run local Z-MIP path (faster)",
    )
    parser.add_argument(
        "--save-input-stack",
        action="store_true",
        help="Write input_local_z_mip_stack.tif",
    )
    parser.add_argument("--raw-vmax-p", type=float, default=95.0)
    parser.add_argument("--contour", type=float, default=0.2)
    parser.add_argument("--ch", type=int, default=None)
    args = parser.parse_args()

    flim_paths = resolve_flim_paths(args.flim, args.folder, args.pattern)
    for flim_path in flim_paths:
        run_for_flim(
            flim_path,
            args.model,
            z_radius=args.z_radius,
            stack_preprocess=args.stack_preprocess,
            subdir=args.subdir,
            skip_single_plane=args.skip_single_plane,
            save_input_local_mip_stack=args.save_input_stack,
            raw_vmax_percentile=args.raw_vmax_p,
            contour_level=args.contour,
            ch_1or2=args.ch,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
