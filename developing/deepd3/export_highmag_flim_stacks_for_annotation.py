# -*- coding: utf-8 -*-
"""Export 3D ZYX TIFF stacks from high-mag .flim files for DeepD3 annotation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

_CONTROLFLIMAGE = Path(__file__).resolve().parents[2]
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

from FLIMageAlignment import get_xyz_pixel_um  # noqa: E402
from FLIMageFileReader2 import FileReader  # noqa: E402

DEFAULT_FOLDERS = [
    r"G:\ImagingData\Tetsuya\20260530\auto1",
    r"G:\ImagingData\Tetsuya\20260528\auto1",
    r"G:\ImagingData\Tetsuya\20260526\auto1",
    r"G:\ImagingData\Tetsuya\20260510\auto1",
    r"G:\ImagingData\Tetsuya\20260429\auto1",
]
DEFAULT_OUTPUT_SUBDIR = "deepd3_annotation_stacks"


def find_target_flim_files(folder: str | Path) -> list[Path]:
    """Return *highmag*_002.flim paths, excluding for_aling* names."""
    root = Path(folder)
    if not root.is_dir():
        return []
    files = sorted(root.glob("*highmag*_002.flim"))
    return [p for p in files if not p.name.lower().startswith("for_aling")]


def load_flim_zyx_uint16(flim_path: str | Path, ch_1or2: int = 2) -> tuple[np.ndarray, dict]:
    """Load a ZYX stack (uint16) and acquisition metadata from one .flim file."""
    flim_path = Path(flim_path)
    iminfo = FileReader()
    iminfo.read_imageFile(str(flim_path), True)

    imagearray = np.array(iminfo.image)
    if imagearray.ndim != 6:
        raise ValueError(f"Unexpected image shape {imagearray.shape} in {flim_path}")

    ch_idx = ch_1or2 - 1
    n_channels = imagearray.shape[2]
    if ch_idx < 0 or ch_idx >= n_channels:
        raise ValueError(f"Channel {ch_1or2} not available (n_channels={n_channels}) in {flim_path}")

    n_ave = int(getattr(iminfo.State.Acq, "nAveFrame", 1) or 1)
    zyx = (12 * np.sum(imagearray[:, :, ch_idx, :, :, :], axis=-1)) // max(n_ave, 1)
    zyx = np.squeeze(zyx, axis=1)
    zyx = np.asarray(zyx, dtype=np.uint16)
    if zyx.ndim != 3:
        raise ValueError(f"Expected ZYX after squeeze, got {zyx.shape} in {flim_path}")

    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
    z_n, y_n, x_n = zyx.shape
    meta = {
        "source_flim": str(flim_path.resolve()),
        "channel_1based": ch_1or2,
        "shape_zyx": [int(z_n), int(y_n), int(x_n)],
        "x_pixel_um": float(x_um),
        "y_pixel_um": float(y_um),
        "z_pixel_um": float(z_um),
        "field_size_um_xyz": [float(x_n * x_um), float(y_n * y_um), float(z_n * z_um)],
        "zoom": float(iminfo.State.Acq.zoom),
        "slice_step_um": float(iminfo.State.Acq.sliceStep),
        "n_ave_frame": n_ave,
        "dtype": "uint16",
        "intensity": "12 * photon_sum / nAveFrame",
    }
    return zyx, meta


def save_annotation_stack(
    zyx: np.ndarray,
    meta: dict,
    out_tif: str | Path,
    out_json: str | Path | None = None,
    save_mip: bool = True,
) -> None:
    """Save ZYX stack as OME-TIFF with physical pixel sizes and sidecar JSON."""
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    x_um = meta["x_pixel_um"]
    y_um = meta["y_pixel_um"]
    z_um = meta["z_pixel_um"]

    tifffile.imwrite(
        out_tif,
        zyx,
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": x_um,
            "PhysicalSizeY": y_um,
            "PhysicalSizeZ": z_um,
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeYUnit": "um",
            "PhysicalSizeZUnit": "um",
        },
        ome=True,
    )

    if out_json is None:
        out_json = out_tif.with_suffix(".json")
    out_json = Path(out_json)
    payload = dict(meta)
    payload["output_tif"] = str(out_tif.resolve())
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if save_mip:
        mip_path = out_tif.with_name(out_tif.stem + "_zmax_mip.png")
        try:
            import matplotlib.pyplot as plt

            vmax = float(np.percentile(zyx, 99.5))
            plt.imsave(mip_path, zyx.max(axis=0), cmap="gray", vmin=0, vmax=max(vmax, 1.0))
            payload["zmax_mip_png"] = str(mip_path.resolve())
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"  WARNING: could not save MIP preview: {exc}")


def export_folder(
    folder: str | Path,
    *,
    ch_1or2: int = 2,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    overwrite: bool = False,
    save_mip: bool = True,
) -> list[dict]:
    """Export all matching .flim files in one folder."""
    folder = Path(folder)
    out_dir = folder / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    targets = find_target_flim_files(folder)
    print(f"\n{folder}  ({len(targets)} files)")

    for flim_path in targets:
        stem = flim_path.stem
        out_tif = out_dir / f"{stem}_ch{ch_1or2}_zyx.tif"
        out_json = out_dir / f"{stem}_ch{ch_1or2}_zyx.json"

        if out_tif.exists() and not overwrite:
            print(f"  skip (exists): {out_tif.name}")
            if out_json.exists():
                meta = json.loads(out_json.read_text(encoding="utf-8"))
            else:
                meta = {"source_flim": str(flim_path)}
            rows.append(
                {
                    "folder": str(folder),
                    "flim_path": str(flim_path),
                    "output_tif": str(out_tif),
                    "status": "skipped",
                    **{k: meta.get(k) for k in ("x_pixel_um", "y_pixel_um", "z_pixel_um", "shape_zyx")},
                }
            )
            continue

        try:
            zyx, meta = load_flim_zyx_uint16(flim_path, ch_1or2=ch_1or2)
            save_annotation_stack(
                zyx,
                meta,
                out_tif,
                out_json=out_json,
                save_mip=save_mip,
            )
            print(
                f"  saved {out_tif.name}  ZYX={zyx.shape}  "
                f"xy={meta['x_pixel_um']:.4f} um  z={meta['z_pixel_um']:.4f} um"
            )
            rows.append(
                {
                    "folder": str(folder),
                    "flim_path": str(flim_path),
                    "output_tif": str(out_tif),
                    "status": "ok",
                    "x_pixel_um": meta["x_pixel_um"],
                    "y_pixel_um": meta["y_pixel_um"],
                    "z_pixel_um": meta["z_pixel_um"],
                    "shape_z": meta["shape_zyx"][0],
                    "shape_y": meta["shape_zyx"][1],
                    "shape_x": meta["shape_zyx"][2],
                }
            )
        except Exception as exc:
            print(f"  ERROR {flim_path.name}: {exc}")
            rows.append(
                {
                    "folder": str(folder),
                    "flim_path": str(flim_path),
                    "output_tif": str(out_tif),
                    "status": f"error: {exc}",
                }
            )

    manifest_path = out_dir / "export_manifest.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  manifest: {manifest_path}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export high-mag *_002.flim files to 3D annotation TIFF stacks."
    )
    parser.add_argument(
        "folders",
        nargs="*",
        default=DEFAULT_FOLDERS,
        help="Folders to scan (default: predefined Tetsuya auto1 paths)",
    )
    parser.add_argument("--ch", type=int, default=2, choices=[1, 2], help="FLIM channel (1 or 2)")
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help=f"Subfolder name under each input folder (default: {DEFAULT_OUTPUT_SUBDIR})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing TIFF files")
    parser.add_argument("--no-mip", action="store_true", help="Skip z-max MIP PNG preview")
    args = parser.parse_args()

    all_rows: list[dict] = []
    for folder in args.folders:
        all_rows.extend(
            export_folder(
                folder,
                ch_1or2=args.ch,
                output_subdir=args.output_subdir,
                overwrite=args.overwrite,
                save_mip=not args.no_mip,
            )
        )

    ok = sum(1 for r in all_rows if r.get("status") == "ok")
    skipped = sum(1 for r in all_rows if r.get("status") == "skipped")
    errors = sum(1 for r in all_rows if str(r.get("status", "")).startswith("error"))
    print(f"\nDone: {ok} exported, {skipped} skipped, {errors} errors, {len(all_rows)} total")


if __name__ == "__main__":
    main()
