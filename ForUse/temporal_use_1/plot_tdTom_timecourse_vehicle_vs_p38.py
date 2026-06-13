"""Plot TimeCourse CSVs: Vehicle vs p38 inhibitor (2x2) and per-folder summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LIFETIME_COL = "Lifetime-ROI1-ch1"
INTENSITY_COL = "meanIntensity-ROI1-ch1"
MIN_SAME_FILENUMBER = 30
MIN_POST_FRACTION_OF_MEDIAN = 0.85
LIFETIME_YLIM = (-0.050, 0.02)

FOLDER_CONFIGS = [
    {
        "analysis_dir": Path(r"G:/ImagingData/Tetsuya/20260526/Analysis"),
        "row_label": "Vehicle",
        "tag": "20260526",
        "manual_exclude": ["dend1"],
    },
    {
        "analysis_dir": Path(r"G:/ImagingData/Tetsuya/20260528/Analysis"),
        "row_label": "p38 inhibitor",
        "tag": "20260528",
        "manual_exclude": [],
    },
]


@dataclass
class ProcessedTrace:
    """One TimeCourse file after normalization and uncaging-aligned windowing."""

    label: str
    time_min: np.ndarray
    intensity_norm: np.ndarray
    lifetime_norm: np.ndarray
    uncaging_idx: int
    n_post: int


def parse_timecourse_csv(path: Path) -> pd.DataFrame:
    """Parse wide Multi-ROI TimeCourse export into a DataFrame."""
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
            values: list[float] = []
            for value in parts[1:]:
                value = value.strip()
                if value == "":
                    continue
                try:
                    values.append(float(value))
                except ValueError:
                    continue
            if values:
                rows[key] = values

    n_frames = min(len(v) for v in rows.values())
    return pd.DataFrame({k: v[:n_frames] for k, v in rows.items()})


def find_uncaging_start_index(filenumber: np.ndarray, min_count: int = MIN_SAME_FILENUMBER) -> int:
    """Return the first index of a FileNumber run with at least min_count repeats."""
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


def trace_label(path: Path) -> str:
    """Short label from filename (e.g. dend3)."""
    match = re.search(r"dend(\d+)", path.stem, re.I)
    return f"dend{match.group(1)}" if match else path.stem


def select_reference_trace(
    parsed: list[tuple[str, pd.DataFrame, int]],
) -> tuple[str, pd.DataFrame, int]:
    """Pick a reference trace with the most common uncaging index and longest post."""
    unc_counts: dict[int, int] = {}
    for _, _, unc_idx in parsed:
        unc_counts[unc_idx] = unc_counts.get(unc_idx, 0) + 1
    mode_unc_idx = max(unc_counts, key=unc_counts.get)

    candidates = [(label, df, unc_idx) for label, df, unc_idx in parsed if unc_idx == mode_unc_idx]
    return max(candidates, key=lambda item: len(item[1]) - item[2])


def filter_short_post_traces(
    parsed: list[tuple[str, pd.DataFrame, int]],
    min_fraction: float = MIN_POST_FRACTION_OF_MEDIAN,
) -> tuple[list[tuple[str, pd.DataFrame, int]], list[str]]:
    """Drop traces whose post-uncaging frame count is unusually short."""
    post_counts = [len(df) - unc_idx for _, df, unc_idx in parsed]
    median_post = float(np.median(post_counts))
    threshold = median_post * min_fraction

    kept: list[tuple[str, pd.DataFrame, int]] = []
    excluded: list[str] = []
    for item in parsed:
        label, df, unc_idx = item
        n_post = len(df) - unc_idx
        if n_post < threshold:
            excluded.append(f"{label} (post={n_post} < {threshold:.0f})")
        else:
            kept.append(item)

    if not kept:
        raise ValueError("All traces were excluded as short-post datasets")

    return kept, excluded


def load_folder_traces(
    analysis_dir: Path,
    manual_exclude: list[str] | None = None,
) -> tuple[list[ProcessedTrace], np.ndarray]:
    """Load, exclude short-post traces, align to uncaging, and normalize."""
    csv_paths = sorted(analysis_dir.glob("*__TimeCourse.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *__TimeCourse.csv in {analysis_dir}")

    exclude_labels = set(manual_exclude or [])
    parsed: list[tuple[str, pd.DataFrame, int]] = []
    manual_excluded: list[str] = []
    for path in csv_paths:
        label = trace_label(path)
        if label in exclude_labels:
            manual_excluded.append(label)
            continue
        df = parse_timecourse_csv(path)
        unc_idx = find_uncaging_start_index(df["FileNumber"].values.astype(int))
        parsed.append((label, df, unc_idx))

    if manual_excluded:
        print(f"{analysis_dir.name}: manually excluded: {', '.join(manual_excluded)}")

    kept, excluded = filter_short_post_traces(parsed)
    if excluded:
        print(f"{analysis_dir.name}: excluded short-post traces: {', '.join(excluded)}")

    ref_label, ref_df, ref_unc_idx = select_reference_trace(kept)
    min_post = min(len(df) - unc_idx for _, df, unc_idx in kept)
    rel_frames = np.arange(-ref_unc_idx, min_post)

    ref_time_min = (
        ref_df["Time (s)"].values[ref_unc_idx + rel_frames] - ref_df["Time (s)"].iloc[ref_unc_idx]
    ) / 60.0

    traces: list[ProcessedTrace] = []
    for label, df, unc_idx in kept:
        pre_mask = np.arange(len(df)) < unc_idx
        if not np.any(pre_mask):
            raise ValueError(f"{label}: no pre-uncaging frames before index {unc_idx}")

        pre_intensity = df.loc[pre_mask, INTENSITY_COL].mean()
        pre_lifetime = df.loc[pre_mask, LIFETIME_COL].mean()
        if pre_intensity <= 0:
            raise ValueError(f"{label}: non-positive pre-uncaging intensity")

        intensity_norm = np.full(len(rel_frames), np.nan)
        lifetime_norm = np.full(len(rel_frames), np.nan)
        for rel_i, rel_frame in enumerate(rel_frames):
            abs_idx = unc_idx + rel_frame
            if 0 <= abs_idx < len(df):
                intensity_norm[rel_i] = df[INTENSITY_COL].iloc[abs_idx] / pre_intensity - 1.0
                lifetime_norm[rel_i] = df[LIFETIME_COL].iloc[abs_idx] - pre_lifetime

        if not np.any(np.isfinite(intensity_norm[rel_frames >= 0])):
            raise ValueError(f"{label}: no valid post-uncaging frames in aligned window")

        traces.append(
            ProcessedTrace(
                label=label,
                time_min=ref_time_min.copy(),
                intensity_norm=intensity_norm,
                lifetime_norm=lifetime_norm,
                uncaging_idx=unc_idx,
                n_post=min_post,
            )
        )

    print(
        f"{analysis_dir.name}: n={len(traces)}, min_post={min_post}, "
        f"time_axis_ref={ref_label}, uncaging_idx_ref={ref_unc_idx}"
    )
    return traces, ref_time_min


def stack_values(traces: list[ProcessedTrace], field: str) -> np.ndarray:
    """Stack trace values into shape (n_traces, n_frames)."""
    return np.vstack([getattr(trace, field) for trace in traces])


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM across traces (axis 0)."""
    mean = np.nanmean(values, axis=0)
    sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(values), axis=0))
    sem = np.where(np.isfinite(sem), sem, 0.0)
    return mean, sem


def trace_colors(n_traces: int) -> list:
    """Distinct colors for individual traces."""
    cmap = plt.get_cmap("tab10")
    if n_traces <= 10:
        return [cmap(i) for i in range(n_traces)]
    cmap20 = plt.get_cmap("tab20")
    return [cmap20(i) for i in range(n_traces)]


def style_axis(ax: plt.Axes, ylabel: str, show_xlabel: bool) -> None:
    """Apply shared axis styling."""
    ax.axvline(0.0, color="k", linewidth=1.0, zorder=1)
    ax.annotate(
        "Uncaging",
        xy=(0.0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0.0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel("Time (min)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_mean_sem_panel(
    ax: plt.Axes,
    time_min: np.ndarray,
    values: np.ndarray,
    color: str = "#1f4e79",
) -> None:
    """Plot mean line with SEM error bars."""
    mean, sem = mean_sem(values)
    ax.errorbar(
        time_min,
        mean,
        yerr=sem,
        fmt="-",
        color=color,
        ecolor=color,
        elinewidth=1.0,
        capsize=2.0,
        capthick=1.0,
        linewidth=1.8,
        markersize=0,
    )


def plot_individual_panel(
    ax: plt.Axes,
    traces: list[ProcessedTrace],
    field: str,
    show_legend: bool = True,
) -> None:
    """Overlay normalized traces with distinct colors and legend."""
    colors = trace_colors(len(traces))
    for trace, color in zip(traces, colors):
        values = getattr(trace, field)
        valid = np.isfinite(values)
        ax.plot(
            trace.time_min[valid],
            values[valid],
            color=color,
            linewidth=1.0,
            alpha=0.85,
            label=trace.label,
        )
    if show_legend and traces:
        ax.legend(
            fontsize=7,
            loc="upper right",
            frameon=False,
            ncol=2 if len(traces) > 6 else 1,
        )


def save_combined_figure(
    folder_traces: list[tuple[str, list[ProcessedTrace], np.ndarray]],
    plot_mode: str,
    output_path: Path,
) -> None:
    """Save 2x2 figure (Vehicle top, p38 inhibitor bottom)."""
    legend_on_individual = plot_mode == "individual"
    fig, axes = plt.subplots(2, 2, figsize=(9.5 if legend_on_individual else 8.5, 6.5), sharex="col")

    intensity_ylim: list[float] = []
    for _, traces, _ in folder_traces:
        intensity_ylim.extend(
            [np.nanmin(stack_values(traces, "intensity_norm")),
             np.nanmax(stack_values(traces, "intensity_norm"))]
        )

    intensity_range = intensity_ylim[1] - intensity_ylim[0]
    intensity_limits = (
        intensity_ylim[0] - 0.08 * intensity_range,
        intensity_ylim[1] + 0.08 * intensity_range,
    )

    for row_idx, (row_label, traces, time_min) in enumerate(folder_traces):
        intensity_ax = axes[row_idx, 0]
        lifetime_ax = axes[row_idx, 1]

        if plot_mode == "mean_sem":
            plot_mean_sem_panel(intensity_ax, time_min, stack_values(traces, "intensity_norm"))
            plot_mean_sem_panel(lifetime_ax, time_min, stack_values(traces, "lifetime_norm"))
        else:
            plot_individual_panel(
                intensity_ax, traces, "intensity_norm", show_legend=row_idx == 0
            )
            plot_individual_panel(
                lifetime_ax, traces, "lifetime_norm", show_legend=False
            )

        intensity_ax.set_ylim(intensity_limits)
        lifetime_ax.set_ylim(LIFETIME_YLIM)

        style_axis(intensity_ax, "Average Intensity (AU)", show_xlabel=row_idx == 1)
        style_axis(lifetime_ax, "Average Lifetime (AU)", show_xlabel=row_idx == 1)

        intensity_ax.text(
            -0.22,
            0.5,
            row_label,
            transform=intensity_ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )

    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16, wspace=0.28, hspace=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_folder_figure(
    row_label: str,
    traces: list[ProcessedTrace],
    time_min: np.ndarray,
    plot_mode: str,
    output_path: Path,
) -> None:
    """Save 1x2 figure for a single Analysis folder."""
    fig, axes = plt.subplots(
        1, 2, figsize=(9.5 if plot_mode == "individual" else 8.0, 3.5), sharex=True
    )

    if plot_mode == "mean_sem":
        plot_mean_sem_panel(axes[0], time_min, stack_values(traces, "intensity_norm"))
        plot_mean_sem_panel(axes[1], time_min, stack_values(traces, "lifetime_norm"))
    else:
        plot_individual_panel(axes[0], traces, "intensity_norm")
        plot_individual_panel(axes[1], traces, "lifetime_norm", show_legend=False)

    style_axis(axes[0], "Average Intensity (AU)", show_xlabel=True)
    style_axis(axes[1], "Average Lifetime (AU)", show_xlabel=True)
    axes[1].set_ylim(LIFETIME_YLIM)
    axes[0].set_title(f"{row_label} (n={len(traces)})")

    if plot_mode == "individual":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(traces), 5),
            fontsize=8,
            frameon=False,
        )
        fig.subplots_adjust(top=0.78)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    plt.switch_backend("Agg")

    folder_traces: list[tuple[str, list[ProcessedTrace], np.ndarray]] = []
    for config in FOLDER_CONFIGS:
        traces, time_min = load_folder_traces(
            config["analysis_dir"],
            manual_exclude=config.get("manual_exclude", []),
        )
        folder_traces.append((config["row_label"], traces, time_min))

        for plot_mode, suffix in [("mean_sem", "mean_sem"), ("individual", "individual_traces")]:
            out_path = config["analysis_dir"] / f"timecourse_{config['tag']}_{suffix}.png"
            save_folder_figure(config["row_label"], traces, time_min, plot_mode, out_path)

    combined_outputs = [
        ("mean_sem", "timecourse_Vehicle_vs_p38_inhibitor_mean_sem.png"),
        ("individual", "timecourse_Vehicle_vs_p38_inhibitor_individual_traces.png"),
    ]
    for config in FOLDER_CONFIGS:
        for plot_mode, filename in combined_outputs:
            save_combined_figure(
                folder_traces,
                plot_mode,
                config["analysis_dir"] / filename,
            )


if __name__ == "__main__":
    main()
