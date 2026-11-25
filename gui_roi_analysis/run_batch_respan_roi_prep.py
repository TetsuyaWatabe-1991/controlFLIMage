# -*- coding: utf-8 -*-
"""
Headless prep for analayze_all_flim_roi_respan.py (same parameters, no GUI).

Runs first_processing + TIFF build + ROI prefill + quantification + alignment_r
so the set review viewer can open immediately afterward.

Skip a session when combined_df_respan.pkl and the all-frames CSV already exist.
"""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTROL = os.path.normpath(os.path.join(_GUI_DIR, ".."))
_FORUSE = os.path.join(_CONTROL, "ForUse")
for _p in (_GUI_DIR, _FORUSE, _CONTROL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alignment_correlation import session_alignment_r  # noqa: E402
from gui_roi_respan_seg_masks import run_tiff_uncaging_roi_respan  # noqa: E402

# Match analayze_all_flim_roi_respan.py
ch_1or2 = 2
z_plus_minus = 1
uncaging_at_nth = 2
pre_length = uncaging_at_nth + 1
global_align_method = "highpass"
align_reference = "pre1"
local_align_mode = "A"
local_crop_half_size = 60
uncaging_roi_keyframe_count = 3
overwrite_seg_roi_masks = False
skip_lifetime_analysis = True
uncaging_lifetime_bin_frames = 8
photon_threshold = 15
total_photon_threshold = 300
uncaging_frame_num = [33, 34, 35, 36, 55, 80, 144, 264]
fast_mode = True
spine_roi_source = "respan_tracked"
time_window_min = (-40, 60)
show_alignment_r = True

PKL_NAME = "combined_df_respan.pkl"
CSV_NAME = "combined_df_respan_intensity_lifetime_all_frames.csv"

DEFAULT_DATE_ROOTS = [
    r"G:\ImagingData\Tetsuya\20261102",
    r"G:\ImagingData\Tetsuya\20261031",
    r"G:\ImagingData\Tetsuya\20261030",
    r"G:\ImagingData\Tetsuya\20261029",
    r"G:\ImagingData\Tetsuya\20261027",
    r"G:\ImagingData\Tetsuya\20261024",
]


def _session_done(session_dir: Path) -> bool:
    return (session_dir / PKL_NAME).is_file() and (session_dir / CSV_NAME).is_file()


def _pick_flim(session_dir: Path) -> Path | None:
    preferred = sorted(session_dir.glob("*__highmag_*_002.flim"))
    if preferred:
        return preferred[0]
    any_flim = sorted(session_dir.glob("*.flim"))
    return any_flim[0] if any_flim else None


def discover_auto_sessions(date_roots: list[str]) -> list[Path]:
    sessions: list[Path] = []
    for root in date_roots:
        root_p = Path(root)
        if not root_p.is_dir():
            print(f"[WARN] missing date folder: {root}")
            continue
        for child in sorted(root_p.iterdir()):
            if child.is_dir() and child.name.startswith("auto"):
                sessions.append(child)
    return sessions


def run_one_session(session_dir: Path) -> dict:
    result = {
        "session": str(session_dir),
        "status": "unknown",
        "pkl": None,
        "csv": None,
        "error": None,
    }
    if _session_done(session_dir):
        result["status"] = "skipped"
        result["pkl"] = str(session_dir / PKL_NAME)
        result["csv"] = str(session_dir / CSV_NAME)
        print(f"[SKIP] already done: {session_dir}")
        return result

    flim = _pick_flim(session_dir)
    if flim is None:
        result["status"] = "error"
        result["error"] = "no .flim files"
        print(f"[ERROR] no .flim in {session_dir}")
        return result

    pkl_path = session_dir / PKL_NAME
    predefined = str(pkl_path) if pkl_path.is_file() else None
    print("=" * 72)
    print(f"[RUN] {session_dir}")
    print(f"  flim_path={flim}")
    print(f"  predefined_df={predefined}")
    print("=" * 72)

    try:
        df_save_path, out_csv = run_tiff_uncaging_roi_respan(
            ch_1or2=ch_1or2,
            z_plus_minus=z_plus_minus,
            pre_length=pre_length,
            global_align_method=global_align_method,
            local_align_mode=local_align_mode,
            local_crop_half_size=local_crop_half_size,
            uncaging_frame_num=uncaging_frame_num,
            uncaging_roi_keyframe_count=uncaging_roi_keyframe_count,
            overwrite_seg_roi_masks=overwrite_seg_roi_masks,
            skip_lifetime_analysis=skip_lifetime_analysis,
            uncaging_lifetime_bin_frames=uncaging_lifetime_bin_frames,
            photon_threshold=photon_threshold,
            total_photon_threshold=total_photon_threshold,
            fast_mode=fast_mode,
            spine_roi_source=spine_roi_source,
            time_window_min=time_window_min,
            align_reference=align_reference,
            skip_roi_gui=True,
            flim_path=None if predefined else str(flim),
            predefined_df_path=predefined,
        )
        if not df_save_path:
            result["status"] = "error"
            result["error"] = "run returned None (no valid sets?)"
            return result
        if show_alignment_r:
            session_alignment_r(df_save_path, ch=ch_1or2)
        result["status"] = "ok"
        result["pkl"] = df_save_path
        result["csv"] = out_csv
        print(f"[OK] {session_dir}")
        print(f"  pkl={df_save_path}")
        print(f"  csv={out_csv}")
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {session_dir}: {result['error']}")
        traceback.print_exc()
    return result


def main(date_roots: list[str] | None = None) -> int:
    roots = date_roots or DEFAULT_DATE_ROOTS
    sessions = discover_auto_sessions(roots)
    log_dir = Path(r"G:\ImagingData\Tetsuya")
    log_path = log_dir / f"batch_respan_roi_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(f"Sessions ({len(sessions)}):")
    for s in sessions:
        print(f"  {s}")
    print(f"Log: {log_path}")

    results: list[dict] = []
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"started {datetime.now().isoformat()}\n")
        for session in sessions:
            res = run_one_session(session)
            results.append(res)
            log.write(
                f"{res['status']}\t{res['session']}\t{res.get('error') or ''}\t"
                f"{res.get('pkl') or ''}\t{res.get('csv') or ''}\n"
            )
            log.flush()
        log.write(f"finished {datetime.now().isoformat()}\n")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_err = sum(1 for r in results if r["status"] == "error")
    print("=" * 72)
    print(f"DONE ok={n_ok} skipped={n_skip} error={n_err}")
    for r in results:
        if r["status"] == "error":
            print(f"  ERROR {r['session']}: {r['error']}")
    print(f"Log: {log_path}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
