"""Spine ROIs from per-frame RESPAN tracking, written as GUI (Type A) masks.

Prerequisite: RESPAN on 64 px crops around the uncaged spine in every pre/post frame,
either queued during acquisition (one job per frame, <session>/respan_online_queue,
tpem_low_high_spine_multi_respan_online.py) or afterwards by
ongoing/ASIcontroller/respan_track_session.py (<session>/respan_track_queue).

For every set (after create_roi_masks_from_seg_masks, before the ROI GUI):
1. Track the uncaged spine through pre + post frames in time order: the first frame
   starts from the uncaging position, every later frame from the spine picked in the
   previous frame carried by the global shift; a candidate must lie within
   MAX_STEP_XY_UM in xy and MAX_STEP_DZ slices in z, otherwise the frame is "lost" and
   the previous ROI is carried by the global shift.
2. Reference ROI = RESPAN outline of the LAST pre frame, dilated by 1 px (fixed_d1).
   Every frame gets the same shape, translated by the rounded head displacement.
3. Convert to the after_align_full frame of the GUI TIFF (integer shift relative to
   the first pre of the set, same as _rebuild_one_set_full_size). Uncaging frames use
   the last-pre ROI (they are aligned to the last pre there).
4. Write <base>_Spine_roi_mask.tif only when the existing Spine mask is still the
   seg_masks initialisation (or was written by this module and not edited since).
   A mask edited in the GUI is never overwritten. A sidecar
   <base>_Spine_roi_mask_source.json records the source, hash, lost frames and heads.

DendriticShaft and Background masks are not touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "ongoing" / "ASIcontroller"))
sys.path.insert(0, str(_REPO / "controlFLIMage" / "developing" / "mushroom_detector"))

MAX_STEP_XY_UM = 1.0
MAX_STEP_DZ = 3
FIRST_FRAME_MAX_XY_UM = 2.0
ROI_DILATE_PX = 1
SOURCE = "respan_tracked_lastpre_d1"


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def _int_shift(mask: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Move content by (+dy, +dx) whole pixels, zero fill (same as ndimage.shift, order 0)."""
    dy, dx = int(round(dy)), int(round(dx))
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    if ys.start < ys.stop and xs.start < xs.stop:
        out[yd, xd] = mask[ys, xs]
    return out


def _sha(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a.astype(np.uint8)).tobytes()).hexdigest()


def _rows(run_dir: Path, cstem: str, y0: int, x0: int) -> list[dict]:
    """detected_spines rows of one crop, with full-frame y/x and crop-local _cy/_cx."""
    import respan_mushroom_core as rmc

    rows = rmc.load_spine_rows(run_dir / "Tables" / f"{cstem}_detected_spines.csv", missing_ok=True) or []
    out = []
    for r in rows:
        cy, cx = float(r["y"]), float(r["x"])
        out.append(dict(r, _cy=cy, _cx=cx, y=cy + y0, x=cx + x0, z=float(r["z"])))
    return out


def track_frames(frames: list[dict], rows_by_stem: dict[str, list[dict]], uncaging_yx: tuple[float, float],
                 last_pre_flim: str, xy_um: float) -> list[dict]:
    """Track the uncaged spine through frames (time order). Pure function (tests).

    frames: [{flim, crop_stem, shift_zyx}]; rows_by_stem: full-frame rows per crop.
    Returns one pick per frame with the raw-frame head (carried by the shift if lost).
    """
    last_shift = next(f["shift_zyx"] for f in frames if _norm(f["flim"]) == _norm(last_pre_flim))
    picks, prev = [], None
    for f in frames:
        sz, sy, sx = f["shift_zyx"]
        if prev is None:
            tz = None
            ty = uncaging_yx[0] + last_shift[1] - sy
            tx = uncaging_yx[1] + last_shift[2] - sx
            max_xy, max_dz = FIRST_FRAME_MAX_XY_UM, None
        else:
            (pz, py, px), (qz, qy, qx) = prev
            tz, ty, tx = pz + qz - sz, py + qy - sy, px + qx - sx
            max_xy, max_dz = MAX_STEP_XY_UM, MAX_STEP_DZ
        best, best_d = None, None
        for r in rows_by_stem.get(f["crop_stem"], []):
            dxy = float(np.hypot(r["y"] - ty, r["x"] - tx)) * xy_um
            if dxy > max_xy or (max_dz is not None and abs(r["z"] - tz) > max_dz):
                continue
            if best_d is None or dxy < best_d:
                best, best_d = r, dxy
        if best is not None:
            prev = ((best["z"], best["y"], best["x"]), (sz, sy, sx))
            head = [best["z"], best["y"], best["x"]]
        elif prev is not None:
            head = [tz, ty, tx]
        else:
            head = None
        picks.append({"flim": f["flim"], "crop_stem": f["crop_stem"], "shift_zyx": [sz, sy, sx], "row": best,
                      "step_um": best_d, "status": "tracked" if best is not None else "lost", "head": head})
    return picks


def track_set(job_dir: Path) -> dict:
    """Load one finished set-level track job (respan_track_session.py) and track the spine."""
    in_dir = Path(job_dir) / "input"
    info = json.loads((in_dir / "track_info.json").read_text(encoding="utf-8"))
    crops = json.loads((in_dir / "crops.json").read_text(encoding="utf-8"))
    return track_source(dict(info=info, crops=crops, dir_of={st: Path(job_dir) for st in crops}, missing=[]))


def load_track_source(last_pre_flim: str, queue_roots) -> dict | None:
    """Tracking crops of one uncaging set, from either layout.

    1. set-level job (respan_track_session.py, key track_key(last pre)), or
    2. per-frame jobs queued during acquisition (respan_online_roi.write_track_frame_job,
       manifest <queue>/track_sets/<key>/track_info.json); frames whose job is not done
       yet are listed in "missing".
    Returns dict(info, crops, dir_of: crop_stem -> job dir, missing) or None.
    """
    from respan_online_queue import TRACK_SETS_DIR, JobQueue, track_key

    roots = [Path(r) for r in queue_roots if r and Path(r).is_dir()]  # never create a queue here
    key = track_key(last_pre_flim)
    for root in roots:
        q = JobQueue(root)
        job = q.result(key)
        if job is not None and not job.error and (q.job_dir(key) / "input" / "track_info.json").exists():
            in_dir = q.job_dir(key) / "input"
            crops = json.loads((in_dir / "crops.json").read_text(encoding="utf-8"))
            return dict(info=json.loads((in_dir / "track_info.json").read_text(encoding="utf-8")), crops=crops,
                        dir_of={st: q.job_dir(key) for st in crops}, missing=[])
    for root in roots:
        mpath = root / TRACK_SETS_DIR / key / "track_info.json"
        if not mpath.exists():
            continue
        q = JobQueue(root)
        info = json.loads(mpath.read_text(encoding="utf-8"))
        crops, dir_of, missing = {}, {}, []
        for f in info["frames"]:
            crops[f["crop_stem"]] = {k: f[k] for k in ("y0", "x0", "full_shape_yx")}
            job = q.result(f["job_key"])
            if job is not None and not job.error:
                dir_of[f["crop_stem"]] = q.job_dir(f["job_key"])
            else:
                missing.append(f["crop_stem"])
        return dict(info=info, crops=crops, dir_of=dir_of, missing=missing)
    return None


def track_source(src: dict) -> dict:
    """Track the uncaged spine over the frames of load_track_source / track_set."""
    info, crops = src["info"], src["crops"]
    rows = {st: _rows(Path(d) / "run", st, crops[st]["y0"], crops[st]["x0"]) for st, d in src["dir_of"].items()}
    picks = track_frames(info["frames"], rows, (info["uncaging_y_pix"], info["uncaging_x_pix"]),
                         info["last_pre_flim"], float(info["xy_um"]))
    for p in picks:
        c = crops[p["crop_stem"]]
        p.update(y0=c["y0"], x0=c["x0"], full_shape_yx=c["full_shape_yx"], job_dir=src["dir_of"].get(p["crop_stem"]))
        if p["crop_stem"] in src["missing"]:
            p["status"] = "not processed"
    return {"info": info, "picks": picks, "xy_um": float(info["xy_um"]), "missing": list(src["missing"])}


def build_type_a_stack(picks: list[dict], ref_pick: dict, ref_roi: np.ndarray,
                       frame_info: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """GUI (after_align_full) Spine mask stack. Pure function (tests).

    Raw ROI of frame f = ref_roi moved by round(head_f - head_ref); the GUI frame is the
    raw frame moved by the integer shift_y/x in frame_info (as written for the TIFF).
    Uncaging rows get the mask of the last pre row. Frames without a pick (not in the
    track job) inherit the previous mask.
    """
    by_base = {os.path.basename(p["flim"]).lower(): p for p in picks}
    ry, rx = ref_pick["head"][1], ref_pick["head"][2]
    masks, notes, last = [], [], None
    for fr in frame_info.itertuples():
        phase = str(fr.phase).lower()
        if phase.startswith("unc"):
            m = last
            notes.append("uncaging=last pre")
        else:
            p = by_base.get(str(fr.filename).lower())
            if p is None or p["head"] is None:
                m = last
                notes.append("no pick, previous ROI")
            else:
                raw = _int_shift(ref_roi, round(p["head"][1] - ry), round(p["head"][2] - rx))
                m = _int_shift(raw, int(fr.shift_y), int(fr.shift_x))
                notes.append(p["status"])
        if m is None:
            m = np.zeros_like(ref_roi)
        masks.append(m)
        last = m
    return np.stack(masks, 0).astype(np.uint8), notes


def reference_roi(job_dir: Path, pick: dict, xy_um: float) -> np.ndarray:
    """RESPAN outline of one tracked frame (raw full-frame coordinates), dilated by 1 px."""
    import respan_mushroom_core as rmc

    run_dir = job_dir / "run"
    stem = pick["crop_stem"]
    zyx = tifffile.imread(job_dir / "input" / f"{stem}.tif")
    labels = tifffile.imread(run_dir / "Validation_Data" / "Segmentation_Labels" / f"{stem}.tif")
    vols = rmc.load_respan_volumes(run_dir, stem)
    low = rmc.build_low_intensity_mask_2d(zyx)
    r = pick["row"]
    g = rmc._compute_spine_geometry(dict(r, y=r["_cy"], x=r["_cx"]), zyx, labels, xy_um, vols,
                                    int(float(r["spine_id"])), low_int_mask_2d=low)
    roi_c = ndi.binary_dilation(np.asarray(g["spine_outline_mask_2d"]) > 0, iterations=ROI_DILATE_PX)
    full = np.zeros(tuple(pick["full_shape_yx"]), bool)
    y0, x0 = pick["y0"], pick["x0"]
    h, w = min(roi_c.shape[0], full.shape[0] - y0), min(roi_c.shape[1], full.shape[1] - x0)
    full[y0:y0 + h, x0:x0 + w] = roi_c[:h, :w]
    return full.astype(np.uint8)


def _is_replaceable(path: str, seg_mask: np.ndarray | None) -> tuple[bool, str]:
    """True when the current Spine mask is untouched (seg initialisation or our own, unedited)."""
    if not os.path.exists(path):
        return True, "no mask yet"
    cur = tifffile.imread(path) > 0
    if cur.ndim == 2:
        cur = cur[None]
    if seg_mask is not None and cur.shape[1:] == seg_mask.shape and all(
            np.array_equal(cur[i], seg_mask > 0) for i in range(cur.shape[0])):
        return True, "replaced seg_masks initialisation"
    side = Path(path).with_name(Path(path).stem + "_source.json")
    if side.exists():
        meta = json.loads(side.read_text(encoding="utf-8"))
        if meta.get("sha1") == _sha(cur):
            return True, "refreshed own unedited mask"
        return False, "kept: edited after respan tracking"
    return False, "kept: edited (differs from seg_masks initialisation)"


def apply_tracked_spine_rois(combined_df: pd.DataFrame, *, queue_root: str | None = None,
                             overwrite_edited: bool = False) -> pd.DataFrame:
    """Replace Spine Type-A masks by RESPAN-tracked ROIs where allowed. Returns a summary."""
    from gui_roi_respan_seg_masks import (
        _load_mask_2d,
        highmag_savefolder_from_filepath_without_number,
        match_uncaging_record_for_set,
        seg_mask_paths,
    )
    from respan_online_queue import track_key
    from respan_track_session import QUEUE_NAME
    from respan_uncaging_log import parse_uncaging_records

    summary = []
    print("Spine ROIs from RESPAN tracking (last pre outline + 1 px)...")
    for fp_wo in combined_df["filepath_without_number"].unique():
        fgroup = combined_df[combined_df["filepath_without_number"] == fp_wo]
        hdir = highmag_savefolder_from_filepath_without_number(fp_wo)
        records = parse_uncaging_records(hdir)
        if not records:
            continue
        session = Path(hdir).parent
        # set-level jobs (respan_track_session.py) or per-frame jobs queued during acquisition
        roots = [queue_root] if queue_root else [session / QUEUE_NAME, session / "respan_online_queue"]
        for group in fgroup["group"].unique():
            gdf = fgroup[fgroup["group"] == group]
            for sl in gdf["nth_set_label"].unique():
                if sl == -1:
                    continue
                sdf = gdf[gdf["nth_set_label"] == sl]
                name = f"{group}_{sl}"
                row = dict(set=name, status="", reason="")
                try:
                    tiff = sdf["after_align_save_path"].iloc[0]
                    rec = match_uncaging_record_for_set(sdf, records)
                    if rec is None or pd.isna(tiff):
                        row.update(status="skip", reason="no uncaging record / TIFF")
                        summary.append(row)
                        continue
                    key = track_key(rec.flim_path)
                    src = load_track_source(rec.flim_path, roots)
                    if src is None or not src["dir_of"]:
                        row.update(status="skip", reason="RESPAN tracking not available; seg_masks ROI kept")
                        summary.append(row)
                        continue
                    base = os.path.splitext(os.path.basename(tiff))[0]
                    out_path = os.path.join(os.path.dirname(tiff), f"{base}_Spine_roi_mask.tif")
                    segp = seg_mask_paths(hdir, rec.spine_stem).get("Spine")
                    seg = _load_mask_2d(segp) if segp else None
                    ok, why = _is_replaceable(out_path, seg)
                    if not ok and not overwrite_edited:
                        row.update(status="skip", reason=why)
                        summary.append(row)
                        continue
                    fi_path = os.path.join(os.path.dirname(tiff), f"{base}_frame_info.csv")
                    if not os.path.exists(fi_path):
                        row.update(status="skip", reason="frame_info.csv missing")
                        summary.append(row)
                        continue
                    frame_info = pd.read_csv(fi_path)
                    tr = track_source(src)
                    picks = tr["picks"]
                    by_base = {os.path.basename(p["flim"]).lower(): p for p in picks}
                    pre_names = [str(f).lower() for f in frame_info[frame_info.phase == "pre"].filename]
                    ref = next((by_base[n] for n in reversed(pre_names)
                                if n in by_base and by_base[n]["status"] == "tracked"), None)
                    if ref is None:
                        row.update(status="skip", reason="spine not found by RESPAN in any pre frame")
                        summary.append(row)
                        continue
                    ref_roi = reference_roi(Path(ref["job_dir"]), ref, tr["xy_um"])
                    stack, notes = build_type_a_stack(picks, ref, ref_roi, frame_info)
                    n_expected = (int(sdf["n_pre_frames"].iloc[0]) + int(sdf["n_unc_frames"].iloc[0])
                                  + int(sdf["n_post_frames"].iloc[0]))
                    if stack.shape[0] != n_expected or stack.shape[1:] != tifffile.imread(tiff, key=0).shape:
                        row.update(status="skip", reason=f"shape {stack.shape} does not match TIFF ({n_expected})")
                        summary.append(row)
                        continue
                    tifffile.imwrite(out_path, stack, photometric="minisblack")
                    lost = [os.path.basename(p["flim"]) for p in picks if p["status"] != "tracked"]
                    Path(out_path).with_name(Path(out_path).stem + "_source.json").write_text(json.dumps(
                        {"source": SOURCE, "sha1": _sha(stack > 0), "spine_stem": rec.spine_stem,
                         "reference_frame": os.path.basename(ref["flim"]), "roi_area_px": int(ref_roi.sum()),
                         "frame_notes": notes,
                         "lost_frames": lost, "job_key": key, "not_processed": tr["missing"],
                         "frames": [{"flim": os.path.basename(p["flim"]), "status": p["status"],
                                     "step_um": p["step_um"]} for p in picks]}, indent=1), encoding="utf-8")
                    row.update(status="written", reason=why, lost=len(lost), area=int(ref_roi.sum()), ref=os.path.basename(ref['flim']))
                    print(f"    {base}: Spine <- RESPAN tracking ({why}; lost {len(lost)})")
                except Exception as exc:
                    row.update(status="error", reason=repr(exc))
                    print(f"    {name}: RESPAN tracking ROI failed, seg_masks ROI kept: {exc!r}")
                summary.append(row)
    df = pd.DataFrame(summary)
    if len(df):
        print("RESPAN tracked Spine ROIs:", df.status.value_counts().to_dict())
        for r in df[df.status != "written"].itertuples():
            print(f"    {r.set}: {r.status} ({r.reason})")
    return df
