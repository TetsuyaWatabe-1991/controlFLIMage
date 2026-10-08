"""Pearson correlation of aligned images (no filtering) for the set review viewer.

highmag: each pre/post frame of the set against the set's Pre 1, as shown in the viewer:
    the frames of the full-size TIFF (max projection over each frame's z window, moved by
    the integer shift of frame_info.csv). r is computed over the pixels both frames cover
    (the zero-filled border of a moved frame is left out). Pre 1 itself is 1.
lowmag: the lowmag acquired last before that highmag frame (same cycle), aligned to the
    field's first lowmag (_001) by 3D phase correlation (traditional, as the acquisition
    aligns lowmag); r of the whole-volume max projections over the overlap. _001 is
    resized to the query pixel grid when the resolutions differ.
Results are cached per set in <tiff>_alignment_r.csv (recomputed when the TIFF is newer or
the cache has another reference).
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

R_CSV_SUFFIX = "_alignment_r.csv"
HIGHMAG_REFERENCE = "pre1"


def pearson_overlap(ref: np.ndarray, mov: np.ndarray, dy: int, dx: int) -> float:
    """r between ref and mov moved by (+dy, +dx) whole pixels, over the overlap only."""
    h, w = ref.shape
    ry = slice(max(0, dy), min(h, h + dy))
    rx = slice(max(0, dx), min(w, w + dx))
    my = slice(max(0, -dy), min(h, h - dy))
    mx = slice(max(0, -dx), min(w, w - dx))
    a = np.asarray(ref, float)[ry, rx].ravel()
    b = np.asarray(mov, float)[my, mx].ravel()
    if a.size < 16 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _round(v) -> int:
    return int(np.floor(float(v) + 0.5))


def _volume(flim_path: str, ch: int) -> np.ndarray:
    """(Z, Y, X) intensity of one channel (fast_mode intensity cache)."""
    from flim_fast_io import load_flim_intensity

    return np.asarray(load_flim_intensity(flim_path, use_cache=True)[0][:, 0, ch - 1], dtype=np.float64)


def _acq_time(flim_path: str):
    from FLIMageFileReader2 import FileReader

    from gui_roi_fast_simple import _parse_acq_time

    r = FileReader()
    r.read_imageFile(flim_path, False)
    return _parse_acq_time(str(r.acqTime[0]).strip()) if r.acqTime else None


def _valid_region(shape: tuple[int, int], dy: int, dx: int) -> tuple[slice, slice]:
    """Pixels of a frame moved by (dy, dx) that hold image data (not the zero fill)."""
    h, w = shape
    return slice(max(0, dy), h + min(0, dy)), slice(max(0, dx), w + min(0, dx))


def pearson_in_tiff(ref: np.ndarray, img: np.ndarray, ref_shift: tuple[int, int],
                    img_shift: tuple[int, int]) -> float:
    """r of two TIFF frames over the pixels both cover (no filtering)."""
    ry, rx = _valid_region(ref.shape, *ref_shift)
    iy, ix = _valid_region(img.shape, *img_shift)
    ys = slice(max(ry.start, iy.start), min(ry.stop, iy.stop))
    xs = slice(max(rx.start, ix.start), min(rx.stop, ix.stop))
    a = np.asarray(ref, float)[ys, xs].ravel()
    b = np.asarray(img, float)[ys, xs].ravel()
    if a.size < 16 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def highmag_r(tiff_path: str, frame_info: pd.DataFrame) -> dict[str, float]:
    """filename (lower case) -> r against Pre 1 (first pre frame) of the set's TIFF."""
    import tifffile

    rows = frame_info[frame_info["phase"].astype(str).str.lower().isin(["pre", "post"])]
    pre = rows[rows["phase"].astype(str).str.lower() == "pre"]
    if not len(pre):
        return {}
    stack = tifffile.imread(tiff_path)
    shift = lambda fr: (_round(fr.shift_y if pd.notna(fr.shift_y) else 0),  # noqa: E731
                        _round(fr.shift_x if pd.notna(fr.shift_x) else 0))
    ref_fr = pre.sort_values("frame").iloc[0]
    ref = stack[int(ref_fr.frame)]
    out = {}
    for fr in rows.itertuples():
        out[str(fr.filename).lower()] = pearson_in_tiff(ref, stack[int(fr.frame)], shift(ref_fr), shift(fr))
    return out


class LowmagIndex:
    """Lowmag files of one field with acquisition times; aligned r against _001."""

    def __init__(self, folder: str, prefix: str, ch: int):
        self.ch = ch
        self.files = sorted(p for p in glob.glob(os.path.join(folder, f"{prefix}[0-9][0-9][0-9].flim"))
                            if re.fullmatch(re.escape(prefix) + r"\d{3}\.flim", os.path.basename(p)))
        self.times = [_acq_time(p) for p in self.files]
        self._r: dict[str, float] = {}
        self._ref = None

    def last_before(self, t) -> str | None:
        best = None
        for p, tp in zip(self.files, self.times):
            if tp is not None and t is not None and tp <= t:
                best = p
        return best

    def r_against_first(self, path: str) -> float:
        if path in self._r:
            return self._r[path]
        from FLIMageAlignment import Align_4d_array
        from skimage.transform import resize

        if self._ref is None:
            self._ref = _volume(self.files[0], self.ch)
        mov = _volume(path, self.ch)
        ref = self._ref
        if ref.shape != mov.shape:
            ref = resize(ref, mov.shape, order=1, preserve_range=True, anti_aliasing=True)
        shifts, _ = Align_4d_array(np.stack([ref, mov]), method="traditional", apply_shifts=False)
        s = shifts[-1]
        r = pearson_overlap(ref.max(0), mov.max(0), _round(s[1]), _round(s[2]))
        self._r[path] = r
        return r


_lowmag_cache: dict[tuple, LowmagIndex] = {}


def set_alignment_r(set_df: pd.DataFrame, group_df: pd.DataFrame | None, tiff_path: str, ch: int = 2,
                    write_cache: bool = True) -> pd.DataFrame:
    """Per pre/post frame: filename, r_high (vs Pre 1), lowmag file, r_low (vs _001). Cached CSV.

    group_df is not used any more (the highmag reference was the group's 002); kept for callers.
    """
    base = os.path.splitext(tiff_path)[0]
    csv = base + R_CSV_SUFFIX
    if write_cache and os.path.exists(csv) and os.path.getmtime(csv) >= os.path.getmtime(tiff_path):
        cached = pd.read_csv(csv)
        if "reference" in cached.columns and (cached["reference"] == HIGHMAG_REFERENCE).all():
            return cached
    frame_info = pd.read_csv(base + "_frame_info.csv")
    rh = highmag_r(tiff_path, frame_info)
    first = str(set_df.file_path.iloc[0])
    folder = os.path.dirname(first)
    prefix = os.path.basename(first).split("_highmag_")[0]  # "4_pos1_" -> 4_pos1_NNN.flim
    key = (folder, prefix, ch)
    if key not in _lowmag_cache:
        _lowmag_cache[key] = LowmagIndex(folder, prefix, ch)
    low = _lowmag_cache[key]
    rows = []
    by_name = {os.path.basename(str(p)).lower(): str(p) for p in set_df.file_path}
    for name, r in rh.items():
        lp = low.last_before(_acq_time(by_name[name])) if low.files else None
        rows.append(dict(filename=name, r_high=round(r, 3) if np.isfinite(r) else np.nan,
                         lowmag=os.path.basename(lp) if lp else "",
                         r_low=round(low.r_against_first(lp), 3) if lp else np.nan))
    df = pd.DataFrame(rows, columns=["filename", "r_high", "lowmag", "r_low"])
    df["reference"] = HIGHMAG_REFERENCE
    if write_cache:
        try:
            df.to_csv(csv, index=False)
        except OSError:
            pass
    return df


def session_alignment_r(pkl_path: str, ch: int = 2, out_csv: str | None = None, write_cache: bool = True,
                        max_sets: int | None = None, log=print) -> str:
    """Correlations of every set of a combined_df pkl, in one CSV next to the pkl.

    Columns: group, set_label, filename, r_high, lowmag, r_low. Per-set results are also
    cached beside each TIFF (set_alignment_r), so the viewer shows them without waiting.
    Returns the CSV path (default <pkl stem>_alignment_r.csv).
    """
    from set_review_viewer import list_sets, set_rows

    df = pd.read_pickle(pkl_path)
    out_csv = out_csv or os.path.splitext(pkl_path)[0] + R_CSV_SUFFIX
    sets = list_sets(df)[:max_sets]
    parts = []
    for k, (g, s, tiff) in enumerate(sets):
        try:
            sdf = df[set_rows(df, g, s)]
            gdf = df[(df["group"].astype(str) == str(g))
                     & (df["filepath_without_number"] == sdf["filepath_without_number"].iloc[0])]
            r = set_alignment_r(sdf, gdf, tiff, ch=ch, write_cache=write_cache)
        except Exception as exc:  # one broken set must not stop the others
            log(f"alignment r [{k + 1}/{len(sets)}] {g} set {s:g}: failed {exc!r}")
            continue
        parts.append(r.assign(group=g, set_label=s))
        log(f"alignment r [{k + 1}/{len(sets)}] {g} set {s:g}: "
            f"H median {r.r_high.median():.3f}, L median {r.r_low.median():.3f}")
    cols = ["group", "set_label", "filename", "r_high", "lowmag", "r_low"]
    out = pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)
    out.to_csv(out_csv, index=False)
    log(f"alignment r saved: {out_csv} ({len(out)} frames, {len(parts)} sets)")
    return out_csv
