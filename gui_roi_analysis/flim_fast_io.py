# -*- coding: utf-8 -*-
"""Optional fast FLIM IO for GUI prep (header peek, intensity-only, disk cache).

Default analysis still uses FileReader / flim_files_to_nparray unchanged.
Enable via run_tiff_uncaging_roi_respan(fast_mode=True).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from FLIMageFileReader2 import FileReader

FAST_CACHE_SUBDIR = os.path.join("tif", "_fast_intensity")


def peek_flim_header_shape(flim_path: str) -> tuple[int, ...] | None:
    """Shape from header / page count only (no photon decode)."""
    try:
        iminfo = FileReader()
        iminfo.read_imageFile(flim_path, False)
        n_tau = int(iminfo.n_time[0]) if iminfo.n_time else 1
        return (
            int(iminfo.n_images),
            int(iminfo.nFastZSlices),
            int(iminfo.nChannels),
            int(iminfo.height),
            int(iminfo.width),
            n_tau,
        )
    except Exception:
        return None


def header_yxzn(flim_path: str) -> tuple[int, int, int] | None:
    """Return (Z_pages, Y, X) from header, or None."""
    shape = peek_flim_header_shape(flim_path)
    if shape is None:
        return None
    n_images, _n_fastz, _n_ch, height, width, _n_tau = shape
    return int(n_images), int(height), int(width)


def _cache_dir_for_flim(flim_path: str) -> str:
    return os.path.join(os.path.dirname(flim_path), FAST_CACHE_SUBDIR)


def _cache_paths(flim_path: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(flim_path))[0]
    cache_dir = _cache_dir_for_flim(flim_path)
    return (
        os.path.join(cache_dir, stem + "_intensity.npy"),
        os.path.join(cache_dir, stem + "_intensity.json"),
    )


def _source_mtime_ns(flim_path: str) -> int:
    return int(os.stat(flim_path).st_mtime_ns)


def _load_cached_intensity(flim_path: str) -> np.ndarray | None:
    npy_path, meta_path = _cache_paths(flim_path)
    if not (os.path.isfile(npy_path) and os.path.isfile(meta_path)):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if int(meta.get("mtime_ns", -1)) != _source_mtime_ns(flim_path):
            return None
        arr = np.load(npy_path)
        if tuple(arr.shape) != tuple(meta.get("shape", ())):
            return None
        return arr
    except Exception:
        return None


def _save_cached_intensity(flim_path: str, intensity: np.ndarray) -> None:
    npy_path, meta_path = _cache_paths(flim_path)
    try:
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, np.asarray(intensity, dtype=np.float32))
        meta = {
            "mtime_ns": _source_mtime_ns(flim_path),
            "shape": list(np.asarray(intensity).shape),
            "source": os.path.basename(flim_path),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    except Exception as e:
        print(f"  fast_mode: could not cache intensity for {flim_path}: {e}")


def load_flim_intensity(
    flim_path: str,
    *,
    use_cache: bool = True,
) -> tuple[np.ndarray, FileReader]:
    """Load FLIM as intensity (tau summed), optionally from disk cache.

    Returns (intensity, iminfo) where intensity is float32
    (n_pages, nFastZ, C, Y, X) = (12 * sum(bins)) / nAveFrame.
    iminfo is header-only (readImage=False) so nAve / acqTime are available.
    """
    iminfo = FileReader()
    iminfo.read_imageFile(flim_path, False)
    if use_cache:
        cached = _load_cached_intensity(flim_path)
        if cached is not None:
            return cached, iminfo

    decoded = FileReader()
    decoded.read_imageFile(flim_path, True, intensity_only=True)
    imagearray = np.array(decoded.image)
    n_ave = getattr(decoded.State.Acq, "nAveFrame", 1) or 1
    intensity = ((12.0 * np.sum(imagearray, axis=-1)) / n_ave).astype(np.float32)
    if use_cache:
        _save_cached_intensity(flim_path, intensity)
    return intensity, decoded if decoded.acqTime else iminfo


def zproj_from_intensity(
    intensity: np.ndarray,
    ch_1or2: int,
    z_from: int,
    z_to: int,
) -> np.ndarray:
    """Max projection over axis-0, matching _load_flim_zproj_full."""
    intensity_raw = np.asarray(intensity, dtype=np.float32)
    axis0_len = intensity_raw.shape[0]
    z0 = max(0, min(int(z_from), axis0_len - 1))
    z1 = min(axis0_len, int(z_to))
    z1 = max(z1, z0 + 1)
    return intensity_raw[z0:z1, 0, ch_1or2 - 1, :, :].max(axis=0)


def uncaging_frames_from_intensity(
    intensity: np.ndarray,
    ch_1or2: int,
) -> list[np.ndarray]:
    """Match _load_uncaging_full: Z slices if Z>1 else time frames."""
    tyx = np.asarray(intensity, dtype=np.float32)[:, :, ch_1or2 - 1, :, :]
    t_len, z_len, _h, _w = tyx.shape
    if z_len > 1:
        return [tyx[0, z].copy() for z in range(z_len)]
    return [tyx[t, 0].copy() for t in range(t_len)]


def flim_files_to_nparray_fast(
    filelist: list[str],
    ch: int = 0,
    *,
    intensity_cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, Any, list[float]]:
    """Like flim_files_to_nparray, but intensity-only + disk cache."""
    four_dim: list[np.ndarray] = []
    timestamp_list: list[datetime] = []
    relative_sec_list: list[float] = []
    first_shape = None
    last_iminfo: Any = None
    for file_path in filelist:
        print(file_path)
        if intensity_cache is not None and file_path in intensity_cache:
            intensity = intensity_cache[file_path]
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, False)
        else:
            intensity, iminfo = load_flim_intensity(file_path, use_cache=True)
            if intensity_cache is not None:
                intensity_cache[file_path] = intensity
        last_iminfo = iminfo
        if first_shape is None:
            first_shape = intensity.shape
        if intensity.shape != first_shape:
            print(file_path, "<- skipped read")
            continue
        four_dim.append(intensity)
        if iminfo.acqTime:
            ts = datetime.strptime(iminfo.acqTime[0], "%Y-%m-%dT%H:%M:%S.%f")
            timestamp_list.append(ts)
            relative_sec_list.append((ts - timestamp_list[0]).seconds)
        else:
            relative_sec_list.append(0.0)
    print("ch", ch)
    stacked = np.array(four_dim, dtype=np.float32)[:, :, 0, ch, :, :]
    return stacked, last_iminfo, relative_sec_list
