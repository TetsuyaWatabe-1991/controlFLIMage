"""Highpass max-projection RESPAN quantification for every 20260909 cnt set.

RESPAN stays loaded for the whole folder. Comparison sheets are written into
one directory so they can be browsed together.

Usage:
    python quant_highpass_respan_cnt_batch.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_MUSHROOM = Path(__file__).resolve().parents[2] / "developing" / "mushroom_detector"
if str(_MUSHROOM) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM))

from export_align_frames import SESSIONS, _safe, _set_keys, load_set, to_ff0  # noqa: E402
from export_highpass_summary import highpass_view_shifts, summary_frame_indices  # noqa: E402
from export_maxproj_compare import max_projection  # noqa: E402
from quant_highpass_respan_one import (  # noqa: E402
    OUT_DIR,
    PIXEL_JSON,
    SESSION,
    _load_filtered_spines,
    _pre_index,
    fixed_mask_means,
    manual_means,
    maxproj_zyx,
    minutes_from_uncaging,
    outline_mask,
    pick_overlapping_component,
    render_comparison_sheet,
    spine_and_dendrite_mips,
)
from respan_runner.paths import respan_gpu_python  # noqa: E402

PARENT = Path(OUT_DIR).parent
INPUT_DIR = PARENT / "cnt_batch_input"
RUN_DIR = PARENT / "cnt_batch_run"
SHEET_DIR = PARENT / "cnt_sheets"
RUNNER = Path(_THIS_DIR) / "run_respan_folder.py"


def cnt_set_keys(combined: pd.DataFrame) -> list[tuple[str, int]]:
    """cnt groups only, in the combined table's order."""
    return [(group, set_label) for group, set_label in _set_keys(combined) if str(group).startswith("cnt_")]


def _stem(group: str, set_label: int) -> str:
    return f"{_safe(SESSION)}_{_safe(group)}_set{set_label}"


def _tiff_name(stem: str) -> str:
    return f"{stem}_ch2_zyx.tif"


def find_label_tiff(label_dir: Path, tiff_name: str) -> Path | None:
    """RESPAN may keep the original name or the nnUNet _0000 suffix."""
    direct = label_dir / tiff_name
    if direct.is_file():
        return direct
    stem = Path(tiff_name).stem
    suffixed = label_dir / f"{stem}_0000.tif"
    if suffixed.is_file():
        return suffixed
    return None


def prepare_inputs(combined: pd.DataFrame) -> list[dict[str, object]]:
    """Write one highpass max-projection volume per cnt set. Returns the manifest."""
    import tifffile

    if INPUT_DIR.exists():
        for old in INPUT_DIR.glob("*.tif"):
            old.unlink()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PIXEL_JSON, encoding="utf-8") as handle:
        pixel = json.load(handle)
    items: list[dict[str, object]] = []
    keys = cnt_set_keys(combined)
    for index, (group, set_label) in enumerate(keys, start=1):
        stem = _stem(group, set_label)
        arrays = load_set(combined, SESSION, group, set_label)
        chosen = None if arrays is None else summary_frame_indices(arrays.phase)
        if arrays is None or chosen is None:
            print(f"  skip {stem}", flush=True)
            items.append({"stem": stem, "group": group, "set_label": set_label, "status": "skip"})
            continue
        _pre, _post0, _post1, max_frames = chosen
        shifts = highpass_view_shifts(arrays.before)
        projection = max_projection(arrays.before, shifts, max_frames)
        tiff_name = _tiff_name(stem)
        tifffile.imwrite(INPUT_DIR / tiff_name, maxproj_zyx(projection))
        items.append(
            {
                "stem": stem,
                "group": group,
                "set_label": set_label,
                "tiff": tiff_name,
                "status": "ready",
            }
        )
        print(f"  prepared {index}/{len(keys)} {stem}", flush=True)
    meta = {
        "x_pixel_um": pixel["x_pixel_um"],
        "y_pixel_um": pixel["y_pixel_um"],
        "z_pixel_um": pixel["z_pixel_um"],
        "items": items,
    }
    (INPUT_DIR / "batch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return items


def _labels_complete(items: list[dict[str, object]]) -> bool:
    label_dir = RUN_DIR / "Validation_Data" / "Segmentation_Labels"
    if not label_dir.is_dir():
        return False
    for item in items:
        if item.get("status") != "ready":
            continue
        if find_label_tiff(label_dir, str(item["tiff"])) is None:
            return False
    return True


def run_respan_once(items: list[dict[str, object]]) -> None:
    """One RESPAN process for every prepared stack."""
    if _labels_complete(items):
        print("RESPAN labels already present; skipping segmentation", flush=True)
        return
    log_path = PARENT / "cnt_batch_respan.log"
    command = [
        str(respan_gpu_python()),
        "-u",
        str(RUNNER),
        "--input-dir",
        str(INPUT_DIR),
        "--run-dir",
        str(RUN_DIR),
    ]
    with open(log_path, "w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"RESPAN batch failed (code {completed.returncode}).\n{tail}")


def write_sheets(combined: pd.DataFrame, items: list[dict[str, object]]) -> list[str]:
    """Write one comparison sheet per ready set into SHEET_DIR."""
    import tifffile

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    label_dir = RUN_DIR / "Validation_Data" / "Segmentation_Labels"
    written: list[str] = []
    lines = [
        "Rows, top to bottom: manual, highpass + manual ROI, highpass + RESPAN raw, highpass + RESPAN outline.",
        "Columns: pre first, pre last, post first, post middle, post last.",
        "Outline color matches the F/F0 plot.",
        "",
    ]
    ready = [item for item in items if item.get("status") == "ready"]
    for index, item in enumerate(ready, start=1):
        stem = str(item["stem"])
        group = str(item["group"])
        set_label = int(item["set_label"])
        label_path = find_label_tiff(label_dir, str(item["tiff"]))
        if label_path is None:
            lines.append(f"FAIL {stem} missing labels")
            print(f"  missing labels {stem}", flush=True)
            continue
        arrays = load_set(combined, SESSION, group, set_label)
        if arrays is None:
            lines.append(f"FAIL {stem} could not reload")
            continue
        shifts = highpass_view_shifts(arrays.before)
        labels = tifffile.imread(label_path)
        filtered = _load_filtered_spines(label_path.parents[2], Path(str(item["tiff"])).stem)
        spine_labels, dendrite = spine_and_dendrite_mips(labels, filtered)
        raw, overlap, n_spines = pick_overlapping_component(spine_labels, arrays.manual[0])
        outline = outline_mask(raw, dendrite)
        after_path = combined.loc[
            (combined["group"].astype(str) == group)
            & (pd.to_numeric(combined["nth_set_label"], errors="coerce") == set_label),
            "after_align_full_save_path",
        ].iloc[0]
        info = pd.read_csv(
            os.path.join(os.path.dirname(after_path), f"{Path(after_path).stem}_frame_info.csv")
        )
        info = info.iloc[: len(arrays.before)]
        elapsed = pd.to_numeric(info["elapsed_time_sec"], errors="coerce").to_numpy(dtype=float)
        minutes = minutes_from_uncaging(elapsed, arrays.phase)
        pre_index = _pre_index(arrays.phase)
        means = {
            "manual": manual_means(arrays.after, arrays.manual),
            "highpass + manual ROI": fixed_mask_means(arrays.before, shifts, arrays.manual[0]),
            "highpass + RESPAN raw": fixed_mask_means(arrays.before, shifts, raw),
            "highpass + RESPAN outline": fixed_mask_means(arrays.before, shifts, outline),
        }
        traces = {name: to_ff0(values, pre_index) for name, values in means.items()}
        title = f"{SESSION}  {group}  set {set_label}"
        sheet = render_comparison_sheet(
            arrays.before,
            arrays.after,
            arrays.manual,
            shifts,
            arrays.phase,
            minutes,
            traces,
            raw,
            outline,
            title=title,
        )
        cv2.imwrite(str(SHEET_DIR / f"{stem}.png"), sheet)
        written.append(stem)
        lines.append(
            f"{stem}  spines {n_spines}  overlap {overlap} px  "
            f"raw {int(raw.sum())} px  outline {int(outline.sum())} px"
        )
        print(f"  sheet {index}/{len(ready)} {stem}", flush=True)
    (SHEET_DIR / "index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def main() -> None:
    combined = pd.read_pickle(SESSIONS[SESSION])
    print(f"preparing {len(cnt_set_keys(combined))} cnt sets", flush=True)
    items = prepare_inputs(combined)
    ready = sum(1 for item in items if item.get("status") == "ready")
    print(f"running RESPAN on {ready} stacks", flush=True)
    run_respan_once(items)
    written = write_sheets(combined, items)
    print(f"sheets {len(written)} -> {SHEET_DIR}", flush=True)


if __name__ == "__main__":
    main()
