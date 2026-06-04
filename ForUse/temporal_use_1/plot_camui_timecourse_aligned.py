"""Plot TimeCourse CSVs aligned to uncaging start (FileNumber block >= 30)."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATHS = [
    Path(r"g:\ImagingData\Tetsuya\20260522\Analysis\camui_dend3__TimeCourse - Copy.csv"),
    Path(r"g:\ImagingData\Tetsuya\20260522\Analysis\camui_dend4__TimeCourse - Copy.csv"),
    Path(r"g:\ImagingData\Tetsuya\20260522\Analysis\camui_dend5__TimeCourse - Copy.csv"),
    Path(r"g:\ImagingData\Tetsuya\20260522\Analysis\camui_dend6__TimeCourse - Copy.csv"),
]

OUTPUT_DIR = Path(r"g:\ImagingData\Tetsuya\20260522\Analysis")
TIME_ROUND_DECIMALS = 0
MIN_SAME_FILENUMBER = 30

LIFETIME_COL = "Lifetime-ROI1-ch1"
INTENSITY_COL = "meanIntensity-ROI1-ch1"


def parse_timecourse_csv(path: Path) -> pd.DataFrame:
    """Parse wide Multi-ROI TimeCourse export into long-format DataFrame."""
    rows: dict[str, list[float]] = {}
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Multi-ROI") or line.startswith("nROIs"):
                continue
            parts = line.split(",")
            key = parts[0].strip()
            if not key:
                continue
            values = []
            for v in parts[1:]:
                v = v.strip()
                if v == "":
                    continue
                try:
                    values.append(float(v))
                except ValueError:
                    continue
            if values:
                rows[key] = values

    n = min(len(v) for v in rows.values())
    data = {k: v[:n] for k, v in rows.items()}
    df = pd.DataFrame(data)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_uncaging_start_index(filenumber: np.ndarray, min_count: int = 30) -> int:
    """First index where FileNumber stays constant for at least min_count points."""
    i = 0
    n = len(filenumber)
    while i < n:
        j = i + 1
        while j < n and filenumber[j] == filenumber[i]:
            j += 1
        if j - i >= min_count:
            return i
        i = j
    raise ValueError(f"No FileNumber run >= {min_count} found")


def align_to_uncaging(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Align full trace so uncaging block starts at t=0 (includes pre-uncaging)."""
    time_col = "Time (s)"
    fn_col = "FileNumber"
    if time_col not in df.columns:
        raise KeyError(f"Missing {time_col}")
    if fn_col not in df.columns:
        raise KeyError(f"Missing {fn_col}")

    idx0 = find_uncaging_start_index(df[fn_col].values.astype(int), MIN_SAME_FILENUMBER)
    t0 = df[time_col].iloc[idx0]
    out = df.copy()
    out["aligned_time_s"] = out[time_col] - t0
    out["time_rounded"] = out["aligned_time_s"].round(TIME_ROUND_DECIMALS)
    return out, idx0


def aggregate_mean_sem(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean and SEM per rounded aligned time."""
    grouped = df.groupby("time_rounded", as_index=False)[value_col]
    agg = grouped.agg(["mean", "sem", "count"])
    agg.columns = ["time_rounded", "mean", "sem", "n"]
    return agg


def dend_label(path: Path) -> str:
    m = re.search(r"dend(\d+)", path.stem, re.I)
    return f"dend{m.group(1)}" if m else path.stem


def main() -> None:
    plt.switch_backend("Agg")
    aligned: list[tuple[str, pd.DataFrame]] = []
    for path in CSV_PATHS:
        raw = parse_timecourse_csv(path)
        al, idx0 = align_to_uncaging(raw)
        label = dend_label(path)
        al["dend"] = label
        aligned.append((label, al))
        n_pre = idx0
        print(
            f"{label}: uncaging at index {idx0}, "
            f"FileNumber={int(raw['FileNumber'].iloc[idx0])}, "
            f"t0={raw['Time (s)'].iloc[idx0]:.2f} s, "
            f"pre_frames={n_pre}, post_frames={len(al) - n_pre}, total={len(al)}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 0.4, len(aligned)))

    for (label, df), color in zip(aligned, colors):
        for ax, col in zip(axes, [LIFETIME_COL, INTENSITY_COL]):
            agg = aggregate_mean_sem(df, col)
            sem = agg["sem"].fillna(0)
            ax.errorbar(
                agg["time_rounded"],
                agg["mean"],
                yerr=sem,
                fmt="o-",
                capsize=3,
                markersize=5,
                linewidth=1.5,
                label=label,
                color=color,
                alpha=0.9,
            )

    axes[0].set_ylabel("Lifetime (ns)")
    axes[0].set_title(
        "ROI1-ch1, pre- and post-uncaging (aligned t=0, mean ± SEM per rounded time)"
    )
    axes[0].axvline(0, color="gray", ls="--", lw=0.8)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Mean intensity (a.u.)")
    axes[1].set_xlabel("Time from uncaging start (s)")
    axes[1].axvline(0, color="gray", ls="--", lw=0.8)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "camui_dend3-6_timecourse_aligned_uncaging.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
