"""Combined figure for one gentle noise-ceiling surface.

Top: XY max projection, and an XY image sampled on the fitted surface.
Bottom: XZ max projections at Y=5, mid-field, and Y=max-5, each using
+/- 5 pixels in Y. Axes are in micrometers. Z = 0 is the bottom slice.
The fitted surface is drawn on the XZ panels.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from utility.gentle_surface import (
    MIN_BLOB_PX,
    NOISE_THRESHOLD,
    XY_SMOOTH_SIGMA,
    GentleSurfaceResult,
    _noise_ceiling_z,
    _signal_mask,
)

Y_HALF_WIDTH = 5
NEAR_SURFACE_Z = 1.0


def _sample_extent(n: int, um_per_sample: float) -> tuple[float, float]:
    """Data limits that put sample index i at i * um_per_sample."""
    half = 0.5 * float(um_per_sample)
    return -half, (int(n) - 0.5) * float(um_per_sample)


def _robust_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _y_band(center: int, n_y: int, half: int) -> tuple[int, int]:
    y0 = max(0, int(center) - half)
    y1 = min(n_y, int(center) + half + 1)
    return y0, y1


def _sample_along_surface(zyx: np.ndarray, surface_z: np.ndarray) -> np.ndarray:
    n_z = zyx.shape[0]
    z_clip = np.clip(surface_z, 0.0, n_z - 1.0)
    z0 = np.floor(z_clip).astype(int)
    z1 = np.clip(z0 + 1, 0, n_z - 1)
    weight = z_clip - z0
    yy, xx = np.indices(surface_z.shape)
    return (1.0 - weight) * zyx[z0, yy, xx] + weight * zyx[z1, yy, xx]


def _y_centers(n_y: int) -> dict[str, int]:
    return {
        "Y=5": min(5, n_y - 1),
        "Y=middle": n_y // 2,
        "Y=max-5": max(0, (n_y - 1) - 5),
    }


def pixel_um_from_reader(iminfo) -> tuple[float, float, float]:
    """Return (x_um_per_pixel, y_um_per_pixel, z_um_per_slice) from a FLIMage reader."""
    state = iminfo.State.Acq
    x_um = float(state.FOV_default[0]) / float(state.zoom) / float(state.pixelsPerLine)
    y_um = float(state.FOV_default[1]) / float(state.zoom) / float(state.linesPerFrame)
    z_um = float(state.sliceStep)
    return x_um, y_um, z_um


def save_surface_summary(
    zyx: np.ndarray,
    surface: GentleSurfaceResult,
    out_path: str,
    title: str,
    x_um_per_px: float = 1.0,
    z_um_per_slice: float = 1.0,
    y_um_per_px: float | None = None,
) -> str:
    """Save one summary PNG and return its path.

    Axes are in micrometers. Z = 0 is the bottom slice of this stack.
    """
    volume = np.asarray(zyx, dtype=np.float32)
    surface_z = np.asarray(surface.surface_z, dtype=float)
    n_z, n_y, n_x = volume.shape
    if surface_z.shape != (n_y, n_x):
        raise ValueError("surface_z shape must be (Y, X)")

    smoothed = gaussian_filter(volume, sigma=(0.0, XY_SMOOTH_SIGMA, XY_SMOOTH_SIGMA))
    signal = _signal_mask(smoothed, NOISE_THRESHOLD, MIN_BLOB_PX)
    z_top = _noise_ceiling_z(signal)
    on_sheet = (z_top >= 0) & (np.abs(z_top.astype(float) - surface_z) <= NEAR_SURFACE_Z)

    x_um = float(x_um_per_px) if x_um_per_px else 1.0
    y_um = x_um if y_um_per_px is None else float(y_um_per_px)
    z_um = float(z_um_per_slice) if z_um_per_slice else 1.0
    if y_um <= 0:
        y_um = x_um
    y_centers = _y_centers(n_y)
    band_colors = ["#00E5FF", "#FFD54F", "#69F0AE"]
    x0, x1 = _sample_extent(n_x, x_um)
    y0_axis, y1_axis = _sample_extent(n_y, y_um)
    z0, z1 = _sample_extent(n_z, z_um)

    fig = plt.figure(figsize=(14, 8), dpi=140)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.55)
    ax_xy = fig.add_subplot(grid[0, 0:3])
    ax_surf = fig.add_subplot(grid[0, 3:6])
    xz_axes = [
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[1, 4:6]),
    ]

    xy_extent = (x0, x1, y1_axis, y0_axis)
    xy_max = volume.max(axis=0)
    vmin, vmax = _robust_limits(xy_max)
    ax_xy.imshow(
        xy_max,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=xy_extent,
        aspect="equal",
    )
    for (_label, center), color in zip(y_centers.items(), band_colors):
        y0, y1 = _y_band(center, n_y, Y_HALF_WIDTH)
        ax_xy.axhspan((y0 - 0.5) * y_um, (y1 - 0.5) * y_um, color=color, alpha=0.18, linewidth=0)
        ax_xy.axhline(center * y_um, color=color, linewidth=0.8)
        ax_xy.text(
            x0 + 2.0 * x_um,
            center * y_um,
            f"Y={center * y_um:.1f} um",
            color=color,
            fontsize=8,
            va="bottom",
            ha="left",
        )
    ax_xy.set_title("XY max projection")
    ax_xy.set_xlabel("X (um)")
    ax_xy.set_ylabel("Y (um)")

    xy_on_surface = _sample_along_surface(volume, surface_z)
    vmin, vmax = _robust_limits(xy_on_surface)
    center_z = float(surface_z[n_y // 2, n_x // 2])
    ax_surf.imshow(
        xy_on_surface,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        extent=xy_extent,
        aspect="equal",
    )
    ax_surf.plot(
        [((n_x - 1) / 2.0) * x_um],
        [((n_y - 1) / 2.0) * y_um],
        marker="+",
        color="red",
        markersize=10,
        markeredgewidth=1.2,
    )
    ax_surf.set_title(f"XY on fitted surface    center z={center_z * z_um:.1f} um")
    ax_surf.set_xlabel("X (um)")
    ax_surf.set_ylabel("Y (um)")

    x_line_um = np.arange(n_x, dtype=float) * x_um
    for ax, (_label, center), color in zip(xz_axes, y_centers.items(), band_colors):
        y0, y1 = _y_band(center, n_y, Y_HALF_WIDTH)
        xz = volume[:, y0:y1, :].max(axis=1)
        vmin, vmax = _robust_limits(xz)
        ax.imshow(
            xz,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            extent=(x0, x1, z0, z1),
        )
        ax.plot(x_line_um, surface_z[int(center)] * z_um, color="red", linewidth=1.2)
        in_band = np.zeros_like(on_sheet)
        in_band[y0:y1, :] = on_sheet[y0:y1, :]
        py, px = np.where(in_band)
        if px.size:
            ax.scatter(
                px.astype(float) * x_um,
                z_top[py, px].astype(float) * z_um,
                s=10,
                c=color,
                marker="o",
                linewidths=0,
                zorder=3,
            )
        ax.set_ylim(z0, z1)
        ax.set_xlim(x0, x1)
        ax.set_title(
            f"XZ max   Y {(y0 - 0.5) * y_um:.1f}-{(y1 - 0.5) * y_um:.1f} um"
        )
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Z (um)")

    status_line = f"status={surface.status}"
    if surface.warning:
        status_line = f"{status_line}    {surface.warning}"
    fig.suptitle(
        f"{title}\n"
        f"surface Z {float(surface_z.min()) * z_um:.1f}-{float(surface_z.max()) * z_um:.1f} um"
        f"    {status_line}",
        fontsize=11,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
