"""Load one spine set with its auto (RESPAN seg_mask) ROI and the manual ROI per frame.

Everything is expressed in the coordinates of the saved aligned TIFF
(``*_after_align_full.tif``), which is where the auto ROI is pasted without
any shift and where the human edited it. The manual correction of frame ``t``
is the integer translation that carries the auto ROI onto the manual ROI of
that frame.

Usage:
    from manual_shift_dataset import iter_sets, SESSION_DF_PATHS
    for spine_set in iter_sets(SESSION_DF_PATHS):
        ...
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GUI_DIR = os.path.dirname(_THIS_DIR)
_ASI = Path(_THIS_DIR).resolve().parents[2] / "ongoing" / "ASIcontroller"
for _p in (_THIS_DIR, _GUI_DIR, str(_ASI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from respan_uncaging_log import parse_uncaging_records  # noqa: E402

SESSION_DF_PATHS: list[str] = [
    r"G:\ImagingData\Tetsuya\20260610\auto3\combined_df_respan.pkl",
    r"G:\ImagingData\Tetsuya\20260611\auto1\combined_df_respan.pkl",
    r"G:\ImagingData\Tetsuya\20260622\auto1\combined_df_respan.pkl",
    r"G:\ImagingData\Tetsuya\20260623\auto1\combined_df_respan.pkl",
    r"G:\ImagingData\Tetsuya\20260701\auto1\combined_df_respan.pkl",
]
QC_SESSION_DF_PATHS: list[str] = [
    r"G:\ImagingData\Tetsuya\20260909\auto1\combined_df_respan.pkl",
]

SEG_MASK_SUBDIR = "seg_masks"
SEG_FILES = {
    "spine_outline": "{stem}_spine_outline_mask.tif",
    "spine": "{stem}_spine_mask.tif",
    "shaft_fit": "{stem}_shaft_fit_radius_mask.tif",
    "shaft": "{stem}_shaft_mask.tif",
    "bg": "{stem}_bg_mask.tif",
}


@dataclass
class SpineSet:
    """One (session, group, set) with stacks, auto masks, and manual masks."""

    session: str
    group: str
    set_label: int
    spine_stem: str
    before: np.ndarray
    after: np.ndarray
    stored_shift: np.ndarray
    manual_spine: np.ndarray
    manual_shaft: np.ndarray | None
    auto_spine: np.ndarray
    auto_shaft: np.ndarray | None
    auto_spine_core: np.ndarray | None
    phases: list[str]
    elapsed_sec: np.ndarray
    flim_paths: list[str]
    z_from: np.ndarray
    z_to: np.ndarray
    uncaging_yx: tuple[float, float] | None
    reference_flim_path: str | None
    other_spines: np.ndarray | None = None
    after_path: str | None = None
    correction: np.ndarray = field(init=False)
    pure_shift: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.manual_spine)
        self.correction = np.zeros((n, 2), dtype=np.int64)
        self.pure_shift = np.zeros(n, dtype=bool)
        for t in range(n):
            shift, pure = mask_translation(self.auto_spine, self.manual_spine[t])
            self.correction[t] = shift
            self.pure_shift[t] = pure

    @property
    def set_id(self) -> str:
        return f"{self.session}|{self.group}|{self.set_label}"

    @property
    def n_frames(self) -> int:
        return len(self.phases)

    def phase_index(self, name: str) -> np.ndarray:
        wanted = {"unc", "uncaging"} if name in ("unc", "uncaging") else {name}
        return np.array([i for i, p in enumerate(self.phases) if str(p).strip().lower() in wanted], dtype=int)


def translate_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer translation. Vacated pixels are False."""
    dy, dx = int(dy), int(dx)
    height, width = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    y_src0, y_src1 = max(0, -dy), min(height, height - dy)
    x_src0, x_src1 = max(0, -dx), min(width, width - dx)
    if y_src1 <= y_src0 or x_src1 <= x_src0:
        return out
    out[y_src0 + dy : y_src1 + dy, x_src0 + dx : x_src1 + dx] = mask[y_src0:y_src1, x_src0:x_src1]
    return out


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return float("nan"), float("nan")
    return float(ys.mean()), float(xs.mean())


def mask_translation(source: np.ndarray, target: np.ndarray) -> tuple[tuple[int, int], bool]:
    """Integer (dy, dx) carrying ``source`` onto ``target`` and whether it is exact.

    The rounded centroid difference is used. ``pure`` is True when the
    translated source equals the target pixel for pixel.
    """
    source = np.asarray(source, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not source.any() or not target.any():
        return (0, 0), False
    sy, sx = mask_centroid(source)
    ty, tx = mask_centroid(target)
    dy = int(round(ty - sy))
    dx = int(round(tx - sx))
    pure = bool(np.array_equal(translate_mask(source, dy, dx), target))
    return (dy, dx), pure


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b)) / union


def session_name(df_path: str) -> str:
    path = Path(df_path)
    return f"{path.parent.name}_{path.parents[1].name}"


def highmag_folder(filepath_without_number: str) -> str:
    folder = os.path.dirname(str(filepath_without_number))
    stem = os.path.basename(str(filepath_without_number)).rstrip("_")
    return os.path.join(folder, stem)


def _match_record(set_df: pd.DataFrame, records: list):
    pre = set_df[set_df["phase"] == "pre"].sort_values("nth_omit_induction")
    if len(pre) == 0:
        return None
    target = os.path.basename(str(pre.iloc[-1]["file_path"])).lower()
    for rec in records:
        if os.path.basename(str(rec.flim_path)).lower() == target:
            return rec
    return None


def _read_stack(path: str) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        return None
    return np.asarray(arr, dtype=np.float32)


def _read_mask_stack(path: str, n_frames: int) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[None], n_frames, axis=0)
    return np.asarray(arr[:n_frames] > 0)


def _seg_mask(folder: str, stem: str, key: str) -> np.ndarray | None:
    path = Path(folder) / SEG_MASK_SUBDIR / SEG_FILES[key].format(stem=stem)
    if not path.is_file():
        return None
    return np.asarray(tifffile.imread(str(path)) > 0)


def _after_path(set_df: pd.DataFrame) -> str | None:
    for col in ("after_align_full_save_path", "after_align_save_path"):
        if col in set_df.columns:
            value = set_df[col].iloc[0]
            if isinstance(value, str) and os.path.exists(value):
                return value
    return None


def _before_path(after_path: str, set_df: pd.DataFrame) -> str | None:
    if "before_align_full_save_path" in set_df.columns:
        value = set_df["before_align_full_save_path"].iloc[0]
        if isinstance(value, str) and os.path.exists(value):
            return value
    guess = after_path.replace("after_align", "before_align")
    return guess if guess != after_path and os.path.exists(guess) else None


def reference_flim_path(group_df: pd.DataFrame) -> str | None:
    """FLIM file of the group's reference stack (set label -1, stored shift 0)."""
    labels = pd.to_numeric(group_df["nth_set_label"], errors="coerce")
    rows = group_df[labels == -1]
    if len(rows) == 0:
        return None
    path = str(rows["file_path"].iloc[0])
    return path if os.path.exists(path) else None


def load_flim_projection(path: str, z_from: float, z_to: float, ch: int = 2) -> np.ndarray:
    """Max projection of one FLIM Z stack over [z_from, z_to), matching the TIFF build."""
    from FLIMageFileReader2 import FileReader

    reader = FileReader()
    reader.read_imageFile(path, True)
    array = np.asarray(reader.image)
    n_ave = float(getattr(reader.State.Acq, "nAveFrame", 1) or 1)
    intensity = (12.0 * array.sum(axis=-1)) / n_ave
    volume = intensity[:, 0, ch - 1].astype(np.float32)
    n_z = volume.shape[0]
    if not np.isfinite(z_from) or not np.isfinite(z_to):
        return volume.max(axis=0)
    z0 = max(0, min(int(z_from), n_z - 1))
    z1 = max(z0 + 1, min(n_z, int(z_to)))
    return volume[z0:z1].max(axis=0)


def load_flim_volume(path: str, ch: int = 2) -> np.ndarray:
    """Full (Z, Y, X) intensity volume of one FLIM file."""
    from FLIMageFileReader2 import FileReader

    reader = FileReader()
    reader.read_imageFile(path, True)
    array = np.asarray(reader.image)
    n_ave = float(getattr(reader.State.Acq, "nAveFrame", 1) or 1)
    intensity = (12.0 * array.sum(axis=-1)) / n_ave
    return intensity[:, 0, ch - 1].astype(np.float32)


def reference_projection(spine_set: SpineSet, ch: int = 2) -> np.ndarray | None:
    """Reference image in the auto-mask coordinates, projected like the pre frames."""
    if not spine_set.reference_flim_path:
        return None
    pre = spine_set.phase_index("pre")
    index = int(pre[0]) if len(pre) else 0
    return load_flim_projection(spine_set.reference_flim_path, spine_set.z_from[index], spine_set.z_to[index], ch)


def load_set(
    set_df: pd.DataFrame,
    session: str,
    group: str,
    set_label: int,
    records: list,
    reference_path: str | None = None,
    allow_empty_manual: bool = False,
) -> SpineSet | None:
    """Build a SpineSet. Returns None when a required file is missing.

    Args:
        allow_empty_manual: Accept sets whose existing ``_Spine_roi_mask.tif`` has
            empty frames (e.g. an automatic tracker that ran off the field). Used
            when the existing mask is only a reference and is about to be replaced.
    """
    after_path = _after_path(set_df)
    if after_path is None:
        return None
    before_path = _before_path(after_path, set_df)
    if before_path is None:
        return None
    after = _read_stack(after_path)
    before = _read_stack(before_path)
    if after is None or before is None:
        return None
    tiff_dir = os.path.dirname(after_path)
    base = os.path.splitext(os.path.basename(after_path))[0]
    info_path = os.path.join(tiff_dir, f"{base}_frame_info.csv")
    if not os.path.exists(info_path):
        return None
    info = pd.read_csv(info_path)
    n_frames = min(len(before), len(after), len(info))
    manual_spine = _read_mask_stack(os.path.join(tiff_dir, f"{base}_Spine_roi_mask.tif"), n_frames)
    if manual_spine is None or len(manual_spine) < n_frames:
        return None
    manual_shaft = _read_mask_stack(os.path.join(tiff_dir, f"{base}_DendriticShaft_roi_mask.tif"), n_frames)

    record = _match_record(set_df, records)
    if record is None:
        return None
    folder = highmag_folder(str(set_df["filepath_without_number"].iloc[0]))
    auto_spine = _seg_mask(folder, record.spine_stem, "spine_outline")
    if auto_spine is None or auto_spine.shape != after.shape[1:]:
        return None
    if not auto_spine.any():
        return None
    manual_nonempty = manual_spine.reshape(n_frames, -1).any(axis=1)
    if not (manual_nonempty.all() if not allow_empty_manual else manual_nonempty.any()):
        return None
    other_spines = np.zeros_like(auto_spine)
    for other in records:
        if getattr(other, "spine_stem", None) in (None, record.spine_stem):
            continue
        mask = _seg_mask(folder, other.spine_stem, "spine_outline")
        if mask is not None and mask.shape == auto_spine.shape:
            other_spines |= mask
    other_spines &= ~auto_spine

    info = info.iloc[:n_frames].reset_index(drop=True)
    stored = np.column_stack(
        [
            pd.to_numeric(info["shift_y"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(info["shift_x"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    folder_of_flim = os.path.dirname(str(set_df["file_path"].iloc[0]))
    flim_paths = [os.path.join(folder_of_flim, str(name)) for name in info["filename"].astype(str)]
    z_from = pd.to_numeric(info["z_from"], errors="coerce").to_numpy(dtype=float) if "z_from" in info else np.full(n_frames, np.nan)
    z_to = pd.to_numeric(info["z_to"], errors="coerce").to_numpy(dtype=float) if "z_to" in info else np.full(n_frames, np.nan)
    elapsed = (
        pd.to_numeric(info["elapsed_time_sec"], errors="coerce").to_numpy(dtype=float)
        if "elapsed_time_sec" in info
        else np.arange(n_frames, dtype=float)
    )
    return SpineSet(
        session=session,
        group=str(group),
        set_label=int(set_label),
        spine_stem=record.spine_stem,
        before=before[:n_frames],
        after=after[:n_frames],
        stored_shift=stored,
        manual_spine=manual_spine,
        manual_shaft=manual_shaft,
        auto_spine=auto_spine,
        auto_shaft=_seg_mask(folder, record.spine_stem, "shaft_fit"),
        auto_spine_core=_seg_mask(folder, record.spine_stem, "spine"),
        phases=info["phase"].astype(str).tolist(),
        elapsed_sec=elapsed,
        flim_paths=flim_paths,
        z_from=z_from,
        z_to=z_to,
        uncaging_yx=(float(record.uncaging_y_pix), float(record.uncaging_x_pix)),
        reference_flim_path=reference_path,
        other_spines=other_spines,
        after_path=after_path,
    )


def iter_sets(
    df_paths: list[str],
    *,
    skip_rejected: bool = True,
    verbose: bool = True,
    allow_empty_manual: bool = False,
):
    """Yield every loadable SpineSet from the given combined_df pickles."""
    for df_path in df_paths:
        if not os.path.exists(df_path):
            if verbose:
                print(f"missing session {df_path}")
            continue
        session = session_name(df_path)
        combined = pd.read_pickle(df_path)
        n_yielded = 0
        for filepath_wo, filegroup in combined.groupby("filepath_without_number", sort=False):
            folder = highmag_folder(str(filepath_wo))
            try:
                records = parse_uncaging_records(folder) if os.path.isdir(folder) else []
            except Exception:
                records = []
            if not records:
                continue
            for group, group_df in filegroup.groupby("group", sort=False):
                reference_path = reference_flim_path(group_df)
                for set_label, set_df in group_df.groupby("nth_set_label", sort=False):
                    if not np.isfinite(set_label) or int(set_label) < 0:
                        continue
                    if skip_rejected and "reject" in set_df.columns and (set_df["reject"] == 1).any():
                        continue
                    try:
                        spine_set = load_set(
                            set_df,
                            session,
                            str(group),
                            int(set_label),
                            records,
                            reference_path,
                            allow_empty_manual=allow_empty_manual,
                        )
                    except Exception as exc:  # noqa: BLE001
                        if verbose:
                            print(f"  {session} {group} set {int(set_label)}: load failed: {exc}")
                        continue
                    if spine_set is None:
                        continue
                    n_yielded += 1
                    yield spine_set
        if verbose:
            print(f"{session}: {n_yielded} sets", flush=True)
