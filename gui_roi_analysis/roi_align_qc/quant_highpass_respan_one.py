"""Re-quantify one highpass-aligned set with a new RESPAN ROI.

The highpass max projection of pre and post (uncaging excluded) is given to
RESPAN. Two ROIs are built from the spine that overlaps the frame-0 manual
ROI:

- raw: that spine mask unchanged
- outline: spine dilated by 4 px, dendrite dilated by 2 px, overlap removed

Those radii match the imaging-time outline (SPINE_OUTLINE_DILATION_PX and
SHAFT_OUTLINE_DILATION_PX).

Four F/F0 traces are written. Each trace is divided by its own pre mean.

- manual: per-frame manual mask on the saved aligned TIFF
- highpass + frame-0 manual ROI
- highpass + raw RESPAN spine
- highpass + dilated outline

RESPAN is a 3D model (patch depth 10). The 2D max projection is copied onto
the central 3 slices of a 10-slice volume. The ROI is the Z max of the
labels. If that yields no spine, the same image is copied onto every slice
and RESPAN is run again.

Usage:
    python quant_highpass_respan_one.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_MUSHROOM = Path(__file__).resolve().parents[2] / "developing" / "mushroom_detector"
if str(_MUSHROOM) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM))

from export_align_frames import (  # noqa: E402
    DEFAULT_OUT_DIR,
    SESSIONS,
    _safe,
    load_set,
    roi_mean,
    to_ff0,
)
from export_highpass_summary import SCALE, highpass_view_shifts, summary_frame_indices  # noqa: E402
from export_maxproj_compare import max_projection  # noqa: E402
from respan_runner.paths import respan_gpu_python  # noqa: E402
from tracking import translate_image  # noqa: E402

SESSION = "auto1_20260909"
GROUP = "cnt_2_pos1__highmag_3_"
SET_LABEL = 0
PIXEL_JSON = (
    "//RY-LAB-WS04/ImagingData/Tetsuya/20260909/auto1/"
    "deepd3_annotation_stacks/cnt_2_pos1__highmag_3_002_ch2_zyx.json"
)
N_Z = 10
CENTER_THICKNESS = 3
SPINE_DILATION_PX = 4
SHAFT_DILATION_PX = 2
SPINE_LABEL = 1
DENDRITE_LABEL = 2

OUT_DIR = os.path.join(
    os.path.dirname(DEFAULT_OUT_DIR),
    "highpass_respan_requant",
    f"{_safe(SESSION)}_{_safe(GROUP)}_set{SET_LABEL}",
)
RUNNER = os.path.join(_THIS_DIR, "run_respan_on_prepared_tiff.py")


def disk(radius_px: int) -> np.ndarray:
    """Boolean disk used by the imaging-time outline dilation."""
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (xx * xx + yy * yy) <= radius_px * radius_px


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Disk dilation. Radius 0 leaves the mask unchanged."""
    binary = np.asarray(mask, dtype=bool)
    if radius_px <= 0 or not binary.any():
        return binary
    return ndimage.binary_dilation(binary, structure=disk(radius_px))


def outline_mask(spine: np.ndarray, dendrite: np.ndarray) -> np.ndarray:
    """Imaging-time ROI: dilated spine minus dilated dendrite."""
    return dilate_mask(spine, SPINE_DILATION_PX) & ~dilate_mask(dendrite, SHAFT_DILATION_PX)


def maxproj_zyx(image: np.ndarray, *, n_z: int = N_Z, thickness: int = CENTER_THICKNESS) -> np.ndarray:
    """Place a 2D image on the central slices of a ZYX uint16 volume."""
    if thickness > n_z:
        raise ValueError(f"thickness {thickness} exceeds n_z {n_z}")
    slab = np.clip(np.rint(np.asarray(image, dtype=np.float64)), 0, 65535).astype(np.uint16)
    volume = np.zeros((n_z, slab.shape[0], slab.shape[1]), dtype=np.uint16)
    start = (n_z - thickness) // 2
    volume[start : start + thickness] = slab
    return volume


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def pick_overlapping_component(candidates: np.ndarray, manual: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return the labeled component with the largest overlap with the manual ROI.

    candidates is an integer label image (0 = background). When no component
    overlaps the manual ROI, the nearest centroid is used. The third return
    value is how many components were present.
    """
    labels = np.asarray(candidates, dtype=np.int32)
    manual_bool = np.asarray(manual, dtype=bool)
    ids = [int(v) for v in np.unique(labels) if int(v) != 0]
    if not ids:
        return np.zeros(labels.shape, dtype=bool), 0, 0
    best_id = ids[0]
    best_overlap = -1
    for label_id in ids:
        overlap = int(np.count_nonzero((labels == label_id) & manual_bool))
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = label_id
    if best_overlap <= 0:
        manual_center = _centroid(manual_bool)
        if manual_center is not None:
            best_dist = float("inf")
            for label_id in ids:
                center = _centroid(labels == label_id)
                if center is None:
                    continue
                dist = (center[0] - manual_center[0]) ** 2 + (center[1] - manual_center[1]) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_id = label_id
    return labels == best_id, max(best_overlap, 0), len(ids)


def label_components(mask: np.ndarray) -> np.ndarray:
    """Integer labels for a binary mask. Background stays 0."""
    labeled, _n = ndimage.label(np.asarray(mask, dtype=bool))
    return labeled.astype(np.int32)


def _mip_labels(volume: np.ndarray) -> np.ndarray:
    """Max over Z. A 2D array is returned unchanged."""
    array = np.asarray(volume)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"Expected ZYX or YX labels, got shape {array.shape}")
    return np.max(array, axis=0)


def spine_and_dendrite_mips(label_zyx: np.ndarray, filtered_zyx: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Integer spine labels and a binary dendrite MIP.

    Filtered spine instances are preferred. Otherwise spine-class pixels are
    split into connected components.
    """
    class_mip = _mip_labels(label_zyx)
    dendrite = class_mip == DENDRITE_LABEL
    if filtered_zyx is not None:
        filtered = _mip_labels(filtered_zyx)
        if int(np.max(filtered)) > 0:
            if int(np.max(filtered)) == 1:
                return label_components(filtered > 0), dendrite
            return filtered.astype(np.int32), dendrite
    return label_components(class_mip == SPINE_LABEL), dendrite


def fixed_mask_means(before: np.ndarray, shifts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean intensity of one mask on every highpass-aligned frame."""
    means = np.empty(len(before), dtype=np.float64)
    for t in range(len(before)):
        view = translate_image(before[t], float(shifts[t, 0]), float(shifts[t, 1]))
        means[t] = roi_mean(view, mask)
    return means


def manual_means(after: np.ndarray, manual: np.ndarray) -> np.ndarray:
    """Mean intensity of the per-frame manual mask on the saved aligned TIFF."""
    return np.array(
        [roi_mean(after[t], manual[t]) for t in range(len(after))],
        dtype=np.float64,
    )


def _pre_index(phases: list[str]) -> np.ndarray:
    index = np.array(
        [i for i, phase in enumerate(phases) if str(phase).strip().lower() == "pre"],
        dtype=int,
    )
    if index.size == 0:
        return np.array([0], dtype=int)
    return index


def _is_uncaging(phase: str) -> bool:
    return str(phase).strip().lower() in {"unc", "uncaging"}


def minutes_from_uncaging(elapsed_sec: np.ndarray, phases: list[str]) -> np.ndarray:
    """Minutes relative to the first uncaging frame."""
    unc = [i for i, phase in enumerate(phases) if _is_uncaging(phase)]
    origin = float(elapsed_sec[unc[0]]) if unc else float(elapsed_sec[0])
    return (np.asarray(elapsed_sec, dtype=np.float64) - origin) / 60.0


def _load_filtered_spines(run_dir: Path, tiff_stem: str) -> np.ndarray | None:
    vol_path = run_dir / "Validation_Data" / "Validation_Vols" / f"{tiff_stem}.tif"
    if not vol_path.is_file():
        return None
    import tifffile

    volume = tifffile.imread(vol_path)
    if volume.ndim != 4:
        return None
    return volume[:, 2]


def _find_label_tiff(run_parent: Path, tiff_path: Path) -> Path:
    matches = list(run_parent.rglob(tiff_path.name))
    labels = [path for path in matches if "Segmentation_Labels" in path.parts]
    if not labels:
        raise FileNotFoundError(f"RESPAN labels not found under {run_parent}")
    return labels[0]


def _write_input_tiff(out_dir: Path, image: np.ndarray, *, thickness: int) -> Path:
    import tifffile

    tiff_path = out_dir / "highpass_maxproj_ch2_zyx.tif"
    volume = maxproj_zyx(image, thickness=thickness)
    tifffile.imwrite(tiff_path, volume)
    with open(PIXEL_JSON, encoding="utf-8") as handle:
        meta = json.load(handle)
    sidecar = {
        "source": "highpass max projection of pre and post, uncaging excluded",
        "x_pixel_um": meta["x_pixel_um"],
        "y_pixel_um": meta["y_pixel_um"],
        "z_pixel_um": meta["z_pixel_um"],
        "shape_zyx": list(volume.shape),
        "filled_z_thickness": thickness,
        "pixel_size_copied_from": PIXEL_JSON,
    }
    tiff_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return tiff_path


def run_respan(tiff_path: Path, run_parent: Path, *, rerun: bool) -> None:
    """Call RESPAN with the respan_gpu interpreter."""
    if rerun and run_parent.exists():
        shutil.rmtree(run_parent)
    run_parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(respan_gpu_python()),
        RUNNER,
        "--tiff",
        str(tiff_path),
        "--run-parent",
        str(run_parent),
    ]
    if rerun:
        command.append("--rerun")
    log_path = run_parent.parent / "respan_stdout.log"
    with open(log_path, "w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"RESPAN failed (code {completed.returncode}).\n{tail}")


def _contour(mask: np.ndarray, color: tuple[int, int, int], canvas: np.ndarray) -> None:
    big = cv2.resize(
        mask.astype(np.uint8),
        (mask.shape[1] * SCALE, mask.shape[0] * SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _hier = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, 1)


def render_overlay(
    maxproj: np.ndarray,
    manual: np.ndarray,
    raw_spine: np.ndarray,
    outline: np.ndarray,
) -> np.ndarray:
    """Gray max projection with the three ROI outlines."""
    finite = maxproj[np.isfinite(maxproj)]
    low, high = np.percentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0)
    gray = np.clip((maxproj - low) / max(high - low, 1e-6), 0, 1)
    gray_u8 = (gray * 255).astype(np.uint8)
    big = cv2.resize(
        gray_u8,
        (gray_u8.shape[1] * SCALE, gray_u8.shape[0] * SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    _contour(manual, (0, 255, 0), canvas)
    _contour(raw_spine, (255, 255, 0), canvas)
    _contour(outline, (255, 0, 255), canvas)
    return canvas


TRACE_ORDER = (
    "manual",
    "highpass + manual ROI",
    "highpass + RESPAN raw",
    "highpass + RESPAN outline",
)
# RGB, shared by the plot and the image outlines.
TRACE_RGB = {
    "manual": (255, 165, 0),
    "highpass + manual ROI": (255, 40, 40),
    "highpass + RESPAN raw": (30, 144, 255),
    "highpass + RESPAN outline": (0, 200, 0),
}
TRACE_NOTE = {
    "manual": "saved align, per-frame ROI",
    "highpass + manual ROI": "highpass, fixed frame-0 ROI",
    "highpass + RESPAN raw": "highpass, RESPAN spine as-is",
    "highpass + RESPAN outline": "highpass, dilate spine minus dendrite",
}


def _bgr(name: str) -> tuple[int, int, int]:
    red, green, blue = TRACE_RGB[name]
    return blue, green, red


def representative_frame_indices(phases: list[str]) -> list[tuple[str, int]]:
    """Pre first, pre last, post first, post middle, and post last."""
    pre = [i for i, phase in enumerate(phases) if str(phase).strip().lower() == "pre"]
    post = [i for i, phase in enumerate(phases) if str(phase).strip().lower() == "post"]
    if not pre or not post:
        return []
    middle = post[(len(post) - 1) // 2]
    return [
        ("pre first", pre[0]),
        ("pre last", pre[-1]),
        ("post first", post[0]),
        ("post middle", middle),
        ("post last", post[-1]),
    ]


def _display_limits(images: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.concatenate([np.asarray(image, dtype=np.float64).ravel() for image in images])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1, 99.5])
    return float(low), float(high)


def _gray_bgr(image: np.ndarray, low: float, high: float, scale: int) -> np.ndarray:
    gray = np.clip((np.asarray(image, dtype=np.float64) - low) / max(high - low, 1e-6), 0, 1)
    u8 = (gray * 255).astype(np.uint8)
    big = cv2.resize(u8, (u8.shape[1] * scale, u8.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def _draw_contour(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], scale: int) -> None:
    big = cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (mask.shape[1] * scale, mask.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _hier = cv2.findContours((big > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, 2)


def _plot_bgr(
    minutes: np.ndarray,
    phases: list[str],
    traces: dict[str, np.ndarray],
    marks: list[tuple[str, int]],
    title: str,
) -> np.ndarray:
    """F/F0 plot as BGR. Vertical lines mark the frames shown in the tiles."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = [i for i, phase in enumerate(phases) if not _is_uncaging(phase)]
    fig, ax = plt.subplots(figsize=(7.2, 8.2), dpi=140)
    x = minutes[keep]
    for name in TRACE_ORDER:
        red, green, blue = TRACE_RGB[name]
        post = _post_mean(traces[name], phases)
        ax.plot(
            x,
            traces[name][keep],
            color=(red / 255, green / 255, blue / 255),
            lw=1.6,
            marker="o",
            ms=3.5,
            label=f"{name}   post F/F0 {post:.2f}",
        )
    for label, index in marks:
        ax.axvline(minutes[index], color="0.75", lw=0.8)
        ax.text(minutes[index], ax.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=7, color="0.35")
    ax.axhline(1.0, color="0.7", lw=0.8)
    ax.set_xlabel("minutes from first uncaging frame")
    ax.set_ylabel("F/F0")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    encoded = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not encode the F/F0 plot")
    return image


def render_comparison_sheet(
    before: np.ndarray,
    after: np.ndarray,
    manual: np.ndarray,
    shifts: np.ndarray,
    phases: list[str],
    minutes: np.ndarray,
    traces: dict[str, np.ndarray],
    raw_spine: np.ndarray,
    outline: np.ndarray,
    *,
    scale: int = SCALE,
    title: str | None = None,
) -> np.ndarray:
    """One sheet: a row per ROI, five time points, and the F/F0 plot."""
    if title is None:
        title = f"{SESSION}  {GROUP}  set {SET_LABEL}"
    marks = representative_frame_indices(phases)
    if len(marks) != 5:
        raise ValueError("pre and post frames are required")
    views: dict[str, list[np.ndarray]] = {name: [] for name in TRACE_ORDER}
    masks: dict[str, list[np.ndarray]] = {name: [] for name in TRACE_ORDER}
    shown: list[np.ndarray] = []
    for _label, index in marks:
        saved = np.asarray(after[index], dtype=np.float64)
        aligned = translate_image(before[index], float(shifts[index, 0]), float(shifts[index, 1]))
        views["manual"].append(saved)
        masks["manual"].append(manual[index])
        for name, mask in (
            ("highpass + manual ROI", manual[0]),
            ("highpass + RESPAN raw", raw_spine),
            ("highpass + RESPAN outline", outline),
        ):
            views[name].append(aligned)
            masks[name].append(mask)
        shown.extend((saved, aligned))
    low, high = _display_limits(shown)

    label_w = 360
    header_h = 36
    value_h = 26
    panels: list[list[np.ndarray]] = []
    for name in TRACE_ORDER:
        row = []
        color = _bgr(name)
        for column, (_label, index) in enumerate(marks):
            image = _gray_bgr(views[name][column], low, high, scale)
            _draw_contour(image, masks[name][column], color, scale)
            value = traces[name][index]
            bar = np.zeros((value_h, image.shape[1], 3), dtype=np.uint8)
            text = f"F/F0 {value:.2f}" if np.isfinite(value) else "F/F0 nan"
            cv2.putText(bar, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            row.append(np.vstack([image, bar]))
        panels.append(row)

    panel_h, panel_w = panels[0][0].shape[:2]
    header = np.zeros((header_h, label_w + panel_w * 5, 3), dtype=np.uint8)
    for column, (label, index) in enumerate(marks):
        x = label_w + column * panel_w + 8
        cv2.putText(
            header,
            f"{label}   {minutes[index]:+.1f} min",
            (x, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    rows = [header]
    for name, row_panels in zip(TRACE_ORDER, panels):
        gutter = np.zeros((panel_h, label_w, 3), dtype=np.uint8)
        color = _bgr(name)
        cv2.rectangle(gutter, (8, 16), (28, 36), color, -1)
        cv2.putText(gutter, name, (36, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        note = TRACE_NOTE[name]
        cv2.putText(gutter, note, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        rows.append(np.hstack([gutter, *row_panels]))
    tiles = np.vstack(rows)
    plot = _plot_bgr(minutes, phases, traces, marks, title)
    plot = cv2.resize(
        plot,
        (int(round(plot.shape[1] * tiles.shape[0] / plot.shape[0])), tiles.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    body = np.hstack([tiles, plot])
    banner = np.zeros((34, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        banner,
        f"{title}    outline color matches the plot",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([banner, body])


def render_plot(
    minutes: np.ndarray,
    phases: list[str],
    traces: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """F/F0 of pre and post. Uncaging samples are omitted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = [i for i, phase in enumerate(phases) if not _is_uncaging(phase)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    x = minutes[keep]
    for name in TRACE_ORDER:
        red, green, blue = TRACE_RGB[name]
        ax.plot(
            x,
            traces[name][keep],
            color=(red / 255, green / 255, blue / 255),
            marker="o",
            ms=4,
            label=name,
        )
    ax.axhline(1.0, color="0.7", lw=0.8)
    ax.set_xlabel("minutes from first uncaging frame")
    ax.set_ylabel("F/F0")
    ax.set_title(f"{SESSION}  {GROUP}  set {SET_LABEL}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _post_mean(values: np.ndarray, phases: list[str]) -> float:
    index = [i for i, phase in enumerate(phases) if str(phase).strip().lower() == "post"]
    if not index:
        return float("nan")
    return float(np.nanmean(values[index]))


def quantify_one(out_dir: str = OUT_DIR) -> Path:
    """Run the one-set comparison. Returns the output folder."""
    import tifffile

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined = pd.read_pickle(SESSIONS[SESSION])
    arrays = load_set(combined, SESSION, GROUP, SET_LABEL)
    if arrays is None:
        raise RuntimeError(f"Could not load {SESSION} {GROUP} set {SET_LABEL}")
    chosen = summary_frame_indices(arrays.phase)
    if chosen is None:
        raise RuntimeError("Set has no pre or post frames")
    shifts = highpass_view_shifts(arrays.before)
    _pre_first, _post_first, _post_last, max_frames = chosen
    projection = max_projection(arrays.before, shifts, max_frames)

    run_parent = out / "respan_runs"
    tiff_path = _write_input_tiff(out, projection, thickness=CENTER_THICKNESS)
    run_respan(tiff_path, run_parent, rerun=True)
    label_path = _find_label_tiff(run_parent, tiff_path)
    import tifffile as _tiff

    labels = _tiff.imread(label_path)
    filtered = _load_filtered_spines(label_path.parents[2], tiff_path.stem)
    spine_labels, dendrite = spine_and_dendrite_mips(labels, filtered)
    note_fill = f"central {CENTER_THICKNESS} of {N_Z} slices"
    if int(spine_labels.max()) == 0:
        tiff_path = _write_input_tiff(out, projection, thickness=N_Z)
        run_respan(tiff_path, run_parent, rerun=True)
        label_path = _find_label_tiff(run_parent, tiff_path)
        labels = _tiff.imread(label_path)
        filtered = _load_filtered_spines(label_path.parents[2], tiff_path.stem)
        spine_labels, dendrite = spine_and_dendrite_mips(labels, filtered)
        note_fill = f"all {N_Z} slices (central slab had no spine)"

    raw, overlap, n_spines = pick_overlapping_component(spine_labels, arrays.manual[0])
    outline = outline_mask(raw, dendrite)
    tifffile.imwrite(out / "roi_respan_raw.tif", raw.astype(np.uint8))
    tifffile.imwrite(out / "roi_respan_outline.tif", outline.astype(np.uint8))
    tifffile.imwrite(out / "roi_dendrite.tif", dendrite.astype(np.uint8))
    cv2.imwrite(str(out / "roi_overlay.png"), render_overlay(projection, arrays.manual[0], raw, outline))

    after_path = combined.loc[
        (combined["group"].astype(str) == GROUP)
        & (pd.to_numeric(combined["nth_set_label"], errors="coerce") == SET_LABEL),
        "after_align_full_save_path",
    ].iloc[0]
    info = pd.read_csv(os.path.join(os.path.dirname(after_path), f"{Path(after_path).stem}_frame_info.csv"))
    info = info.iloc[: len(arrays.before)]
    phases = arrays.phase
    elapsed = pd.to_numeric(info["elapsed_time_sec"], errors="coerce").to_numpy(dtype=float)
    minutes = minutes_from_uncaging(elapsed, phases)
    pre_index = _pre_index(phases)

    columns = {
        "manual": manual_means(arrays.after, arrays.manual),
        "highpass + manual ROI": fixed_mask_means(arrays.before, shifts, arrays.manual[0]),
        "highpass + RESPAN raw": fixed_mask_means(arrays.before, shifts, raw),
        "highpass + RESPAN outline": fixed_mask_means(arrays.before, shifts, outline),
    }
    ff0 = {name: to_ff0(values, pre_index) for name, values in columns.items()}
    table = pd.DataFrame(
        {
            "frame": np.arange(len(phases)),
            "phase": phases,
            "minutes_from_uncaging": minutes,
            **{f"{name} mean": values for name, values in columns.items()},
            **{f"{name} F/F0": values for name, values in ff0.items()},
        }
    )
    table.to_csv(out / "traces.csv", index=False)
    summary_lines = [
        f"set: {SESSION} {GROUP} set {SET_LABEL}",
        f"RESPAN input fill: {note_fill}",
        f"spine components: {n_spines}",
        f"chosen spine overlap with frame-0 manual ROI: {overlap} px",
        f"raw spine area: {int(raw.sum())} px",
        f"outline area: {int(outline.sum())} px",
        f"dendrite area: {int(dendrite.sum())} px",
        "F/F0 uses each trace's own pre mean. Uncaging frames are in the CSV but not the plot.",
        "",
        "post mean F/F0 and (F/F0 - 1):",
    ]
    for name, values in ff0.items():
        post = _post_mean(values, phases)
        summary_lines.append(f"  {name}: post F/F0 {post:.4f}   delta {post - 1:.4f}")
    (out / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    render_plot(minutes, phases, ff0, out / "ff0_comparison.png")
    sheet = render_comparison_sheet(
        arrays.before,
        arrays.after,
        arrays.manual,
        shifts,
        phases,
        minutes,
        ff0,
        raw,
        outline,
    )
    cv2.imwrite(str(out / "comparison_sheet.png"), sheet)
    print("\n".join(summary_lines))
    print(f"wrote {out}")
    return out


def write_comparison_sheet(out_dir: str = OUT_DIR) -> Path:
    """Rebuild the one-page sheet from the saved RESPAN masks. Does not rerun RESPAN."""
    import tifffile

    out = Path(out_dir)
    combined = pd.read_pickle(SESSIONS[SESSION])
    arrays = load_set(combined, SESSION, GROUP, SET_LABEL)
    if arrays is None:
        raise RuntimeError(f"Could not load {SESSION} {GROUP} set {SET_LABEL}")
    shifts = highpass_view_shifts(arrays.before)
    raw = np.asarray(tifffile.imread(out / "roi_respan_raw.tif")) > 0
    outline = np.asarray(tifffile.imread(out / "roi_respan_outline.tif")) > 0
    after_path = combined.loc[
        (combined["group"].astype(str) == GROUP)
        & (pd.to_numeric(combined["nth_set_label"], errors="coerce") == SET_LABEL),
        "after_align_full_save_path",
    ].iloc[0]
    info = pd.read_csv(os.path.join(os.path.dirname(after_path), f"{Path(after_path).stem}_frame_info.csv"))
    info = info.iloc[: len(arrays.before)]
    phases = arrays.phase
    elapsed = pd.to_numeric(info["elapsed_time_sec"], errors="coerce").to_numpy(dtype=float)
    minutes = minutes_from_uncaging(elapsed, phases)
    pre_index = _pre_index(phases)
    means = {
        "manual": manual_means(arrays.after, arrays.manual),
        "highpass + manual ROI": fixed_mask_means(arrays.before, shifts, arrays.manual[0]),
        "highpass + RESPAN raw": fixed_mask_means(arrays.before, shifts, raw),
        "highpass + RESPAN outline": fixed_mask_means(arrays.before, shifts, outline),
    }
    traces = {name: to_ff0(values, pre_index) for name, values in means.items()}
    sheet = render_comparison_sheet(
        arrays.before,
        arrays.after,
        arrays.manual,
        shifts,
        phases,
        minutes,
        traces,
        raw,
        outline,
    )
    path = out / "comparison_sheet.png"
    cv2.imwrite(str(path), sheet)
    render_plot(minutes, phases, traces, out / "ff0_comparison.png")
    print(f"wrote {path}")
    return path


def main() -> None:
    if "--sheet-only" in sys.argv:
        write_comparison_sheet(OUT_DIR)
        return
    quantify_one(OUT_DIR)


if __name__ == "__main__":
    main()
