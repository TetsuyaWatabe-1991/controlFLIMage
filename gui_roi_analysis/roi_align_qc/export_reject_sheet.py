"""One-page QC sheet for a manually rejected or kept highmag set.

Left: first pre frame, large. Right: aligned max projections of every pre and
post frame, plus the first and last uncaging frames, tiled smaller. The title
states REJECT or KEEP. Spine ROI and the uncaging position are drawn on every
panel.
"""

from __future__ import annotations

import configparser
import os
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile

_CONTROL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CONTROL not in sys.path:
    sys.path.insert(0, _CONTROL)

from FLIMageAlignment import flim_files_to_nparray  # noqa: E402

OUT_DIR = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260701/auto1/roi_align_qc/reject_qc"
SESSIONS = (
    ("20260701", r"G:/ImagingData/Tetsuya/20260701/auto1/combined_df_respan.pkl"),
    ("20260623", r"G:/ImagingData/Tetsuya/20260623/auto1/combined_df_respan.pkl"),
    ("20260909", r"//RY-LAB-WS04/ImagingData/Tetsuya/20260909/auto1/combined_df_respan.pkl"),
)
LARGE = 512
TILE = 128
TILE_COLS = 4
SPINE_BGR = (0, 165, 255)
UNC_BGR = (0, 0, 255)


def _is_reject(value) -> bool:
    return value is True or value == 1 or str(value).lower() in ("true", "1")


def load_set(pkl_path: str, group: str, set_label: float) -> pd.DataFrame:
    df = pd.read_pickle(pkl_path)
    rows = df[(df["group"].astype(str) == group) & (df["nth_set_label"] == set_label)]
    if rows.empty:
        raise FileNotFoundError(f"No rows for {group} set {set_label}")
    return rows


def _frame_order(info: pd.DataFrame) -> list[tuple[int, str]]:
    """Pre frames, first and last uncaging, then post. Indices follow frame_info."""
    phase = info["phase"].astype(str).str.lower()
    pre = info.index[phase.eq("pre")].tolist()
    unc = info.index[phase.isin(["unc", "uncaging"])].tolist()
    post = info.index[phase.eq("post")].tolist()
    chosen_unc = []
    if unc:
        chosen_unc.append(unc[0])
        if unc[-1] != unc[0]:
            chosen_unc.append(unc[-1])
    order = pre + chosen_unc + post
    counts = {"pre": 0, "post": 0}
    labels = []
    for index in order:
        name = str(info.loc[index, "phase"]).lower()
        if name in ("unc", "uncaging"):
            name = "unc first" if index == unc[0] else "unc last"
        else:
            counts[name] = counts.get(name, 0) + 1
            name = f"{name} {counts[name]}"
        labels.append((int(index), name))
    return labels


def _to_u8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    gray = np.clip((image.astype(np.float64) - low) / max(high - low, 1e-6), 0, 1)
    return (gray * 255).astype(np.uint8)


def _draw_roi(canvas: np.ndarray, mask: np.ndarray | None, unc_xy: tuple[float, float] | None, scale: float) -> None:
    if mask is not None and mask.shape[:2] == canvas.shape[:2] or (
        mask is not None and mask.ndim == 2
    ):
        small = mask.astype(np.uint8)
        if small.shape != canvas.shape[:2]:
            small = cv2.resize(small, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = 2 if canvas.shape[0] >= LARGE else 1
        cv2.drawContours(canvas, contours, -1, SPINE_BGR, thickness)
    if unc_xy is None:
        return
    x = int(round(unc_xy[0] * scale))
    y = int(round(unc_xy[1] * scale))
    radius = max(3, int(round(4 * scale)))
    cv2.circle(canvas, (x, y), radius, UNC_BGR, 1)
    cv2.drawMarker(canvas, (x, y), UNC_BGR, cv2.MARKER_CROSS, radius * 2, 1)


def _panel(image: np.ndarray, mask: np.ndarray | None, unc_xy, low: float, high: float, side: int, caption: str) -> np.ndarray:
    scale = side / image.shape[0]
    gray = _to_u8(image, low, high)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = cv2.resize(color, (side, side), interpolation=cv2.INTER_NEAREST)
    _draw_roi(color, mask, unc_xy, scale)
    bar = np.full((18, side, 3), 255, np.uint8)
    cv2.putText(bar, caption, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, color])


def render_sheet(rows: pd.DataFrame, session: str) -> np.ndarray:
    after = str(rows["after_align_full_save_path"].dropna().iloc[0])
    stack = np.asarray(tifffile.imread(after))
    if stack.ndim == 4:
        stack = stack.max(axis=1)
    base = os.path.splitext(os.path.basename(after))[0]
    info = pd.read_csv(os.path.join(os.path.dirname(after), base + "_frame_info.csv"))
    spine_path = os.path.join(os.path.dirname(after), base + "_Spine_roi_mask.tif")
    spine = tifffile.imread(spine_path) if os.path.exists(spine_path) else None
    if spine is not None:
        spine = np.asarray(spine > 0)
    row0 = rows.iloc[0]
    unc_x = row0.get("uncaging_display_x", np.nan)
    unc_y = row0.get("uncaging_display_y", np.nan)
    if pd.isna(unc_x) or pd.isna(unc_y):
        unc_x = row0.get("corrected_uncaging_x", np.nan)
        unc_y = row0.get("corrected_uncaging_y", np.nan)
    unc_xy = None if pd.isna(unc_x) or pd.isna(unc_y) else (float(unc_x), float(unc_y))

    order = _frame_order(info)
    shown = [stack[index] for index, _label in order]
    finite = np.concatenate([frame.ravel() for frame in shown])
    finite = finite[np.isfinite(finite)]
    low, high = np.percentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0)

    def mask_at(index: int):
        if spine is None:
            return None
        if spine.ndim == 2:
            return spine
        if index < spine.shape[0]:
            return spine[index]
        return None

    pre_index = next(index for index, label in order if label.startswith("pre"))
    large = _panel(stack[pre_index], mask_at(pre_index), unc_xy, low, high, LARGE, "pre first")
    tiles = [
        _panel(stack[index], mask_at(index), unc_xy, low, high, TILE, label)
        for index, label in order
    ]
    rows_n = int(np.ceil(len(tiles) / TILE_COLS))
    blank = np.full_like(tiles[0], 255)
    while len(tiles) < rows_n * TILE_COLS:
        tiles.append(blank)
    grid_rows = []
    for r in range(rows_n):
        grid_rows.append(np.hstack(tiles[r * TILE_COLS:(r + 1) * TILE_COLS]))
    grid = np.vstack(grid_rows)
    if grid.shape[0] < large.shape[0]:
        pad = np.full((large.shape[0] - grid.shape[0], grid.shape[1], 3), 255, np.uint8)
        grid = np.vstack([grid, pad])
    elif large.shape[0] < grid.shape[0]:
        pad = np.full((grid.shape[0] - large.shape[0], large.shape[1], 3), 255, np.uint8)
        large = np.vstack([large, pad])
    body = np.hstack([large, np.full((large.shape[0], 8, 3), 255, np.uint8), grid])

    rejected = bool(rows["reject"].map(_is_reject).any()) if "reject" in rows.columns else False
    label = "REJECT" if rejected else "KEEP"
    comment = ""
    if "comment" in rows.columns and pd.notna(rows["comment"].iloc[0]):
        comment = str(rows["comment"].iloc[0]).strip()
    title = f"{label}   {session}   {rows['group'].iloc[0]} set {rows['nth_set_label'].iloc[0]}"
    if comment:
        title = f"{title}   {comment}"
    banner = np.full((36, body.shape[1], 3), 255, np.uint8)
    cv2.putText(banner, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180) if rejected else (0, 120, 0), 2, cv2.LINE_AA)
    return np.vstack([banner, body])


def _set_stem(session: str, group: str, set_label: float) -> str:
    safe_group = str(group).strip().replace(" ", "_")
    return f"{session}_{safe_group}set{int(set_label)}"


def export_sessions(out_dir: str = OUT_DIR) -> None:
    """Write one sheet per set that has an aligned stack, split into reject and keep."""
    reject_dir = os.path.join(out_dir, "reject")
    keep_dir = os.path.join(out_dir, "keep")
    os.makedirs(reject_dir, exist_ok=True)
    os.makedirs(keep_dir, exist_ok=True)
    lines = ["session\tgroup\tset\tlabel\tstatus\tpath"]
    for session, pkl_path in SESSIONS:
        df = pd.read_pickle(pkl_path)
        keys = (
            df.groupby(["group", "nth_set_label"], dropna=False)
            .size()
            .reset_index()[["group", "nth_set_label"]]
        )
        print(f"{session} sets {len(keys)}", flush=True)
        for _i, key in keys.iterrows():
            group = key["group"]
            set_label = key["nth_set_label"]
            rows = df[(df["group"] == group) & (df["nth_set_label"] == set_label)]
            rejected = bool(rows["reject"].map(_is_reject).any()) if "reject" in rows.columns else False
            folder = reject_dir if rejected else keep_dir
            stem = _set_stem(session, str(group), float(set_label))
            path = os.path.join(folder, stem + ".png")
            if os.path.exists(path):
                lines.append(f"{session}\t{group}\t{set_label}\t{'REJECT' if rejected else 'KEEP'}\texists\t{path}")
                continue
            after = rows["after_align_full_save_path"].dropna()
            if after.empty or not os.path.exists(str(after.iloc[0])):
                lines.append(f"{session}\t{group}\t{set_label}\t{'REJECT' if rejected else 'KEEP'}\tskip_no_tiff\t")
                continue
            try:
                canvas = render_sheet(rows, session)
                cv2.imwrite(path, canvas)
            except (OSError, ValueError, FileNotFoundError, KeyError) as exc:
                lines.append(f"{session}\t{group}\t{set_label}\tFAIL\t{exc}\t")
                print("FAIL", stem, exc, flush=True)
                continue
            lines.append(f"{session}\t{group}\t{set_label}\t{'REJECT' if rejected else 'KEEP'}\tok\t{path}")
            print(stem, flush=True)
    with open(os.path.join(out_dir, "index.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote index -> {out_dir}")


def _ini_excluded(path: str) -> int | None:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    try:
        return int(parser["uncaging_settings"]["excluded"])
    except (KeyError, ValueError):
        return None


def _first_highmag_flim(folder: str, base_name: str) -> str | None:
    matches: list[str] = []
    parent = os.path.dirname(os.path.abspath(folder))
    for directory in (folder, parent):
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not name.startswith(base_name + "_") or not name.endswith(".flim"):
                continue
            if "for_align" in name:
                continue
            matches.append(os.path.join(directory, name))
    matches = sorted(set(matches))
    return matches[0] if matches else None


def _groups_that_reached_imaging(pkl_path: str) -> set[str]:
    df = pd.read_pickle(pkl_path)
    labels = pd.to_numeric(df["nth_set_label"], errors="coerce")
    return set(df.loc[labels >= 0, "group"].astype(str))


def render_assignment_mip(folder: str, base_name: str, session: str, kind: str) -> np.ndarray:
    """Z max projection of the assignment FLIM, with rejected spine outlines."""
    flim_path = _first_highmag_flim(folder, base_name)
    if flim_path is None:
        raise FileNotFoundError(f"No FLIM for {base_name}")
    stack, _info, _times = flim_files_to_nparray([flim_path], ch=1)
    volume = np.asarray(stack[0])
    image = volume.max(axis=0) if volume.ndim == 3 else volume
    finite = image[np.isfinite(image)]
    low, high = (np.percentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0))
    panel = _panel(image, None, None, float(low), float(high), LARGE, "assignment")
    seg_dir = os.path.join(folder, "seg_masks")
    if os.path.isdir(seg_dir):
        for name in sorted(os.listdir(seg_dir)):
            if not name.endswith("_spine_outline_mask.tif"):
                continue
            mask = np.asarray(tifffile.imread(os.path.join(seg_dir, name)) > 0)
            _draw_roi(panel[18:], mask, None, LARGE / image.shape[0])
    n_rejected = 0
    for name in os.listdir(folder):
        if name.endswith(".ini") and _ini_excluded(os.path.join(folder, name)) == 1:
            n_rejected += 1
    label = "NONE_FOUND" if kind == "none" else "ALL_REJECTED"
    banner = np.full((52, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(
        banner, f"{label}   {session}   rejected spines {n_rejected}",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 2, cv2.LINE_AA,
    )
    cv2.putText(
        banner, base_name, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )
    return np.vstack([banner, panel])


def export_failed_assignments(out_dir: str = OUT_DIR) -> None:
    """Highmag fields whose RESPAN spines were all rejected or missing, and never imaged."""
    dest = os.path.join(out_dir, "no_assignment")
    os.makedirs(dest, exist_ok=True)
    lines = ["session\tfolder\tkind\tpath"]
    for session, pkl_path in SESSIONS:
        root = os.path.dirname(pkl_path)
        reached = _groups_that_reached_imaging(pkl_path)
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder) or "highmag" not in name:
                continue
            if (name + "_") in reached:
                continue
            inis = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".ini")]
            flags = [_ini_excluded(path) for path in inis]
            n_accepted = sum(1 for flag in flags if flag == 0)
            if n_accepted > 0:
                continue
            csvs = [f for f in os.listdir(folder) if f.endswith("_respan_mushroom_features.csv")]
            if not inis and not csvs:
                continue
            kind = "none" if not inis else "all_rejected"
            path = os.path.join(dest, f"{session}_{name}.png")
            try:
                canvas = render_assignment_mip(folder, name, session, kind)
                cv2.imwrite(path, canvas)
            except (OSError, ValueError, FileNotFoundError) as exc:
                lines.append(f"{session}\t{name}\tFAIL\t{exc}")
                print("FAIL", name, exc, flush=True)
                continue
            lines.append(f"{session}\t{name}\t{kind}\t{path}")
            print(kind, path, flush=True)
    with open(os.path.join(dest, "index.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote no_assignment -> {dest}")


def main() -> None:
    export_sessions()
    export_failed_assignments()


if __name__ == "__main__":
    main()
