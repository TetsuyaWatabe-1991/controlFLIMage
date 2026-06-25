# -*- coding: utf-8 -*-
"""Find uncaging files and derive ROI center from uncaging metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from FLIMageFileReader2 import FileReader
from align_methods import RoiCropSpec

UNCAGING_FRAME_NUMS = [33, 34, 35, 55]
TITRATION_FRAME_NUMS = [32]


@dataclass
class UncagingInfo:
    """Uncaging file metadata used for ROI trimming."""

    file_path: str
    frame_index: int
    center_x: float
    center_y: float
    center_z: int
    n_images: int


def _intensity_zyx_from_iminfo(iminfo: FileReader, ch: int) -> np.ndarray:
    imagearray = np.array(iminfo.image)
    n_ave = iminfo.State.Acq.nAveFrame
    div_by = n_ave if n_ave else 1
    intensity = (12 * np.sum(imagearray, axis=-1)) / div_by
    return intensity[:, 0, ch, :, :]


def get_uncaging_xy_pixels(iminfo: FileReader) -> Tuple[float, float]:
    """Read uncaging center in pixel coordinates from FLIM metadata."""
    pos = iminfo.statedict["State.Uncaging.Position"]
    x_pix = iminfo.statedict["State.Acq.pixelsPerLine"]
    y_pix = iminfo.statedict["State.Acq.linesPerFrame"]
    return float(x_pix * pos[0]), float(y_pix * pos[1])


def _uncaging_frame_index(iminfo: FileReader, n_time: int) -> int:
    """Pick the time index closest to the uncaging event."""
    statedict = iminfo.statedict
    for key in (
        "State.Uncaging.FramesBeforeUncage",
        "State.Uncaging.SlicesBeforeUncage",
    ):
        if key in statedict:
            idx = int(statedict[key])
            return int(np.clip(idx, 0, n_time - 1))
    return n_time // 2


def find_uncaging_file(
    filelist: List[str],
) -> Tuple[Optional[str], Optional[FileReader], Optional[int]]:
    """Return uncaging file path, metadata reader, and index in filelist."""
    first_n_images: Optional[int] = None
    for idx, file_path in enumerate(filelist):
        iminfo = FileReader()
        iminfo.read_imageFile(file_path, readImage=False)
        if first_n_images is None:
            first_n_images = iminfo.n_images
        if (
            iminfo.n_images in UNCAGING_FRAME_NUMS
            and iminfo.n_images != first_n_images
        ):
            return file_path, iminfo, idx
    return None, None, None


def _load_single_stack_zyx(file_path: str, ch: int) -> np.ndarray:
    """Load one FLIM file as a ZYX volume (first time point)."""
    iminfo = FileReader()
    iminfo.read_imageFile(file_path, readImage=True)
    intensity = _intensity_zyx_from_iminfo(iminfo, ch)
    if intensity.ndim == 4:
        return intensity[0]
    if intensity.ndim == 3:
        return intensity
    raise ValueError(f"Unexpected intensity shape {intensity.shape} for {file_path}")


def estimate_z_from_volume(
    vol_zyx: np.ndarray,
    center_y: float,
    center_x: float,
    *,
    xy_radius: int = 8,
) -> int:
    """Estimate best Z slice near the uncaging XY position."""
    cy = int(np.clip(round(center_y), 0, vol_zyx.shape[1] - 1))
    cx = int(np.clip(round(center_x), 0, vol_zyx.shape[2] - 1))
    y0 = max(0, cy - xy_radius)
    y1 = min(vol_zyx.shape[1], cy + xy_radius + 1)
    x0 = max(0, cx - xy_radius)
    x1 = min(vol_zyx.shape[2], cx + xy_radius + 1)
    z_profile = vol_zyx[:, y0:y1, x0:x1].max(axis=(1, 2))
    return int(np.argmax(z_profile))


def uncaging_info_from_series(
    filelist: List[str],
    ch: int,
    *,
    half_z: int = 2,
    half_y: int = 30,
    half_x: int = 30,
) -> Tuple[Optional[UncagingInfo], Optional[RoiCropSpec]]:
    """
    Locate uncaging file and build ROI spec centered on uncaging point.

    XY comes from State.Uncaging.Position (same as flim_ini_match_by_uncaging_pos).
    Z is estimated from the last pre-uncaging Z-stack near that XY (uncaging
    acquisitions are often single-plane).
    """
    unc_path, unc_meta, unc_idx = find_uncaging_file(filelist)
    if unc_path is None:
        # Fallback: use uncaging metadata stored on a regular Z-stack file.
        for idx, file_path in enumerate(filelist):
            iminfo = FileReader()
            iminfo.read_imageFile(file_path, readImage=False)
            if "State.Uncaging.Position" not in iminfo.statedict:
                continue
            pos = iminfo.statedict["State.Uncaging.Position"]
            if pos is None or len(pos) < 2:
                continue
            unc_path = file_path
            unc_idx = idx
            unc_meta = iminfo
            break
        if unc_path is None:
            return None, None

    iminfo = FileReader()
    iminfo.read_imageFile(unc_path, readImage=False)
    n_time = iminfo.n_images
    frame_idx = _uncaging_frame_index(iminfo, n_time)
    center_x, center_y = get_uncaging_xy_pixels(iminfo)

    if unc_idx is not None and unc_idx > 0 and n_time in UNCAGING_FRAME_NUMS:
        z_source_path = filelist[unc_idx - 1]
    else:
        z_source_path = unc_path
    z_vol = _load_single_stack_zyx(z_source_path, ch)
    if z_vol.ndim != 3:
        return None, None
    center_z = estimate_z_from_volume(z_vol, center_y, center_x)

    info = UncagingInfo(
        file_path=unc_path,
        frame_index=frame_idx,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        n_images=n_time,
    )
    roi_spec = RoiCropSpec(
        z=center_z,
        y=int(round(center_y)),
        x=int(round(center_x)),
        half_z=half_z,
        half_y=half_y,
        half_x=half_x,
    )
    return info, roi_spec
