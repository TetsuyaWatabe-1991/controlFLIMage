"""Write per-frame Spine ROI masks for one session from an automatic shift estimator.

For every set of a ``combined_df_respan.pkl`` the RESPAN spine outline is
translated per frame by the shifts of one evaluation variant
(``evaluate_local_dendrite_shift.VARIANTS``, default ``gate_s15_d0``) and
saved as ``<base>_Spine_roi_mask.tif`` next to the aligned TIFF, where the
ROI GUI and the quantification read it. The previous mask is kept once as
``<base>_Spine_roi_mask_before_<variant>.tif``; a ``_source.json`` records
the source, hash and shifts like ``respan_tracked_spine_roi`` does.

Reports (CSV of shifts, F/F0 traces of previous vs new ROI, example tiles)
go to ``<session>/roi_align_qc/roi_shift_<variant>/``.

With ``--requantify`` the existing pipeline
(``gui_roi_respan_seg_masks.run_tiff_uncaging_roi_respan``) is run on the
same pickle with ``spine_roi_source="seg"`` and ``overwrite_seg_roi_masks=False``
so the new masks are kept and the intensity CSV / pickle are regenerated.

Usage:
    python apply_roi_shift_to_session.py --df-path G:/ImagingData/Tetsuya/20261029/auto1/combined_df_respan.pkl
    python apply_roi_shift_to_session.py --df-path ... --variant gate_s25_d0 --dry-run
    python apply_roi_shift_to_session.py --df-path ... --requantify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from evaluate_local_dendrite_shift import (  # noqa: E402
    VARIANTS,
    load_reference,
    normalized_trace,
    predict_variant,
    render_example,
)
from local_dendrite_shift import masks_from_shifts  # noqa: E402
from manual_shift_dataset import SpineSet, _read_mask_stack, iter_sets  # noqa: E402

DEFAULT_VARIANT = "gate_s15_d0"


def _sha(mask: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).hexdigest()


def mask_path(spine_set: SpineSet) -> str:
    base = os.path.splitext(os.path.basename(spine_set.after_path))[0]
    return os.path.join(os.path.dirname(spine_set.after_path), f"{base}_Spine_roi_mask.tif")


def write_masks(spine_set: SpineSet, shifts: np.ndarray, variant: str, *, backup: bool = True) -> dict:
    """Save the translated auto mask stack and its source json. Returns a summary row."""
    masks = masks_from_shifts(spine_set.auto_spine, shifts)
    out_path = mask_path(spine_set)
    side = Path(out_path).with_name(Path(out_path).stem + "_source.json")
    previous_source = None
    if side.exists():
        try:
            previous_source = json.loads(side.read_text(encoding="utf-8")).get("source")
        except Exception:  # noqa: BLE001
            previous_source = "unreadable"
    if backup and os.path.exists(out_path):
        backup_path = Path(out_path).with_name(Path(out_path).stem + f"_before_{variant}.tif")
        if not backup_path.exists():
            shutil.copy2(out_path, backup_path)
        if side.exists():
            backup_side = side.with_name(side.stem + f"_before_{variant}.json")
            if not backup_side.exists():
                shutil.copy2(side, backup_side)
    stack = masks.astype(np.uint8)
    tifffile.imwrite(out_path, stack, photometric="minisblack")
    side.write_text(
        json.dumps(
            {
                "source": f"local_dendrite_shift:{variant}",
                "sha1": _sha(stack > 0),
                "spine_stem": spine_set.spine_stem,
                "variant_params": VARIANTS.get(variant, {}),
                "roi_area_px": int(spine_set.auto_spine.sum()),
                "shifts_dy_dx": [[int(a), int(b)] for a, b in shifts],
                "phases": [str(p) for p in spine_set.phases],
                "previous_source": previous_source,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return {"mask_path": out_path, "previous_source": previous_source, "n_frames": int(len(masks))}


def apply_session(
    df_path: str,
    variant: str,
    *,
    out_dir: str | None = None,
    cache_dir: str | None = None,
    dry_run: bool = False,
    backup: bool = True,
    n_examples: int = 20,
) -> pd.DataFrame:
    """Compute and write ROI masks for every set of one session. Returns the per-frame shift table."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    session_dir = os.path.dirname(df_path)
    out_dir = out_dir or os.path.join(session_dir, "roi_align_qc", f"roi_shift_{variant}")
    os.makedirs(out_dir, exist_ok=True)
    example_dir = os.path.join(out_dir, "examples")
    rows: list[dict] = []
    set_rows: list[dict] = []
    started = time.time()
    n_sets = 0
    n_examples_done = 0
    for spine_set in iter_sets([df_path], skip_rejected=False, allow_empty_manual=True):
        # On a rerun the current mask is our own output; compare against the
        # backup taken on the first run so "previous" keeps meaning the original.
        backup_path = Path(mask_path(spine_set)).with_name(Path(mask_path(spine_set)).stem + f"_before_{variant}.tif")
        if backup_path.exists():
            original = _read_mask_stack(str(backup_path), spine_set.n_frames)
            if original is not None and len(original) >= spine_set.n_frames:
                spine_set.manual_spine = original[: spine_set.n_frames]
        reference = load_reference(spine_set, cache_dir)
        if reference is None:
            print(f"  {spine_set.set_id}: no reference stack, skipped")
            set_rows.append({"set_id": spine_set.set_id, "status": "skip: no reference"})
            continue
        shifts = predict_variant(variant, spine_set, reference)
        pre = spine_set.phase_index("pre")
        post = spine_set.phase_index("post")
        new_masks = masks_from_shifts(spine_set.auto_spine, shifts)
        old_trace = normalized_trace(spine_set.after, spine_set.manual_spine, pre)
        new_trace = normalized_trace(spine_set.after, new_masks, pre)
        auto_trace = normalized_trace(spine_set.after, np.repeat(spine_set.auto_spine[None], spine_set.n_frames, axis=0), pre)
        for t in range(spine_set.n_frames):
            rows.append(
                {
                    "set_id": spine_set.set_id,
                    "group": spine_set.group,
                    "set_label": spine_set.set_label,
                    "spine_stem": spine_set.spine_stem,
                    "frame": t,
                    "phase": str(spine_set.phases[t]),
                    "shift_dy": int(shifts[t, 0]),
                    "shift_dx": int(shifts[t, 1]),
                    "previous_dy": int(spine_set.correction[t, 0]),
                    "previous_dx": int(spine_set.correction[t, 1]),
                    "previous_pure_shift": bool(spine_set.pure_shift[t]),
                    "ff0_auto": float(auto_trace[t]),
                    "ff0_previous": float(old_trace[t]),
                    "ff0_new": float(new_trace[t]),
                }
            )
        moved = float(np.max(np.hypot(shifts[:, 0], shifts[:, 1]))) if len(shifts) else 0.0
        info = {"status": "dry-run"}
        if not dry_run:
            info = write_masks(spine_set, shifts, variant, backup=backup)
            info["status"] = "written"
        set_rows.append(
            {
                "set_id": spine_set.set_id,
                "spine_stem": spine_set.spine_stem,
                "n_frames": spine_set.n_frames,
                "max_shift_px": moved,
                "static_dy": int(shifts[pre[0], 0]) if len(pre) else 0,
                "static_dx": int(shifts[pre[0], 1]) if len(pre) else 0,
                "post_ff0_previous": float(np.nanmean(old_trace[post])) if len(post) else float("nan"),
                "post_ff0_new": float(np.nanmean(new_trace[post])) if len(post) else float("nan"),
                "post_ff0_auto": float(np.nanmean(auto_trace[post])) if len(post) else float("nan"),
                **info,
            }
        )
        if n_examples_done < n_examples and moved >= 1.0:
            os.makedirs(example_dir, exist_ok=True)
            safe = spine_set.set_id.replace("|", "_").replace("/", "_").replace("\\", "_")
            render_example(spine_set, shifts, variant, os.path.join(example_dir, f"{safe}.png"))
            n_examples_done += 1
        n_sets += 1
        if n_sets % 10 == 0:
            print(f"  {n_sets} sets, {time.time() - started:.0f}s", flush=True)
    frames = pd.DataFrame(rows)
    sets = pd.DataFrame(set_rows)
    frames.to_csv(os.path.join(out_dir, "shifts_per_frame.csv"), index=False)
    sets.to_csv(os.path.join(out_dir, "sets.csv"), index=False)
    if len(sets):
        written = sets[sets["status"].isin(["written", "dry-run"])]
        print(
            f"{variant}: {len(written)} sets, max shift >= 2 px in {(written['max_shift_px'] >= 2).sum()} sets, "
            f"static offset != 0 in {((written['static_dy'] != 0) | (written['static_dx'] != 0)).sum()} sets"
        )
        delta = (written["post_ff0_new"] - written["post_ff0_previous"]).abs()
        print(f"  |post F/F0 new - previous| mean {delta.mean():.3f}, median {delta.median():.3f}, max {delta.max():.3f}")
    print(f"wrote {out_dir} ({'dry run, masks untouched' if dry_run else 'masks written'})")
    return frames


def requantify(df_path: str) -> tuple[str | None, str | None]:
    """Re-run the standard quantification on the pickle, keeping the ROI masks on disk."""
    gui_dir = os.path.dirname(_THIS_DIR)
    foruse_dir = os.path.normpath(os.path.join(gui_dir, "..", "ForUse"))
    for p in (gui_dir, foruse_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    from gui_roi_respan_seg_masks import run_tiff_uncaging_roi_respan

    return run_tiff_uncaging_roi_respan(
        ch_1or2=2,
        z_plus_minus=1,
        pre_length=3,
        photon_threshold=15,
        total_photon_threshold=300,
        global_align_method="highpass",
        local_align_mode="A",
        local_crop_half_size=60,
        predefined_df_path=df_path,
        uncaging_frame_num=[33, 34, 35, 36, 55, 80, 144, 264],
        skip_roi_gui=True,
        uncaging_roi_keyframe_count=3,
        overwrite_seg_roi_masks=False,
        skip_lifetime_analysis=True,
        uncaging_lifetime_bin_frames=8,
        fast_mode=True,
        spine_roi_source="seg",
        time_window_min=(-40, 60),
        align_reference="pre1",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write automatic per-frame Spine ROI masks for one session")
    parser.add_argument("--df-path", required=True, help="combined_df_respan.pkl of the session")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help=f"estimator variant (default {DEFAULT_VARIANT})")
    parser.add_argument("--out-dir", default=None, help="report folder (default <session>/roi_align_qc/roi_shift_<variant>)")
    parser.add_argument("--cache-dir", default=None, help="scratch folder for reference projections")
    parser.add_argument("--dry-run", action="store_true", help="compute and report only; do not touch the masks")
    parser.add_argument("--no-backup", action="store_true", help="do not keep a copy of the previous masks")
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--requantify", action="store_true", help="re-run the standard quantification afterwards")
    parser.add_argument(
        "--requantify-only",
        action="store_true",
        help="skip mask writing and only re-run the quantification on the masks already on disk",
    )
    args = parser.parse_args()
    if not args.requantify_only:
        apply_session(
            args.df_path,
            args.variant,
            out_dir=args.out_dir,
            cache_dir=args.cache_dir,
            dry_run=args.dry_run,
            backup=not args.no_backup,
            n_examples=args.n_examples,
        )
    if (args.requantify or args.requantify_only) and not args.dry_run:
        pkl, csv = requantify(args.df_path)
        print(f"requantified: {pkl} -> {csv}")


if __name__ == "__main__":
    main()
