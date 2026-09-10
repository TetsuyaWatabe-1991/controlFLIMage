"""
Paper-style lifetime time course from trimmed.xlsx.

Aligns stimulation to t=0 (1012 s absolute) and plots Delta lifetime (ns)
for stimulated and control ROIs in separate panels (individual traces only).
"""

from __future__ import annotations

import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from typing import Any

import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
import matplotlib.pyplot as plt
import pandas as pd

controlFLIMage_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(controlFLIMage_DIR)
from custom_plot import plt  # noqa: E402  # re-bind smart show

XLSX_PATH = (
    r"\\RY-LAB-YAS15\Users\Yasudalab\Documents\Tetsuya_Imaging"
    r"\20260806\trimmed.xlsx"
)
STIM_TIME_SEC = 1012.0
OUT_BASENAME = "trimmed_lifetime_delta_stim0"

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ROI_LABEL_RE = re.compile(r"ROI(\d+)", re.IGNORECASE)


def _col_letters_to_index(col: str) -> int:
    """Convert Excel column letters (A, B, ..., AA) to 0-based index."""
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall("m:si", NS):
        texts = [
            t.text or ""
            for t in si.iter(
                "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
            )
        ]
        strings.append("".join(texts))
    return strings


def _sheet_rows_as_dicts(zf: zipfile.ZipFile, strings: list[str]) -> list[dict[str, Any]]:
    sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    rows: list[dict[str, Any]] = []
    for row in sheet.findall("m:sheetData/m:row", NS):
        cells: dict[str, Any] = {}
        for c in row.findall("m:c", NS):
            ref = c.get("r")
            if not ref:
                continue
            col = "".join(ch for ch in ref if ch.isalpha())
            v_el = c.find("m:v", NS)
            if v_el is None or v_el.text is None:
                continue
            raw = v_el.text
            if c.get("t") == "s":
                val: Any = strings[int(raw)]
            else:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
            cells[col] = val
        rows.append(cells)
    return rows


def parse_trimmed_xlsx(path: str) -> pd.DataFrame:
    """Parse stimulated/control lifetime blocks into long-form DataFrame."""
    with zipfile.ZipFile(path) as zf:
        strings = _read_shared_strings(zf)
        rows = _sheet_rows_as_dicts(zf, strings)

    records: list[dict[str, Any]] = []
    current_group: str | None = None
    current_times: list[float] | None = None

    for cells in rows:
        label = cells.get("A")
        if label is None:
            continue
        if isinstance(label, str):
            label_stripped = label.strip().lower()
            if label_stripped in ("stimulated", "control"):
                current_group = label_stripped
                current_times = None
                continue
            if label_stripped.startswith("time"):
                times: list[float] = []
                for col, val in sorted(
                    ((c, v) for c, v in cells.items() if c != "A"),
                    key=lambda kv: _col_letters_to_index(kv[0]),
                ):
                    if isinstance(val, (int, float)):
                        times.append(float(val))
                current_times = times
                continue
            if label_stripped.startswith("lifetime") and current_group and current_times:
                values: list[float] = []
                for col, val in sorted(
                    ((c, v) for c, v in cells.items() if c != "A"),
                    key=lambda kv: _col_letters_to_index(kv[0]),
                ):
                    if isinstance(val, (int, float)):
                        values.append(float(val))
                if len(values) != len(current_times):
                    raise ValueError(
                        f"Length mismatch for {label}: "
                        f"{len(values)} values vs {len(current_times)} times"
                    )
                for t, y in zip(current_times, values):
                    records.append(
                        {
                            "group": current_group,
                            "roi": label.strip(),
                            "time_sec": t,
                            "lifetime_ns": y,
                        }
                    )

    if not records:
        raise ValueError(f"No lifetime rows parsed from {path}")
    return pd.DataFrame.from_records(records)


def add_aligned_delta(df: pd.DataFrame, stim_time_sec: float) -> pd.DataFrame:
    """Align time to stimulation and subtract pre-stim mean per ROI."""
    out = df.copy()
    out["aligned_time_sec"] = out["time_sec"] - float(stim_time_sec)
    out["aligned_time_min"] = out["aligned_time_sec"] / 60.0

    delta = pd.Series(index=out.index, dtype=float)
    for roi, g in out.groupby("roi"):
        pre = g.loc[g["aligned_time_sec"] < 0, "lifetime_ns"]
        if pre.empty:
            raise ValueError(f"No pre-stimulus points for ROI {roi}")
        delta.loc[g.index] = g["lifetime_ns"] - float(pre.mean())
    out["delta_lifetime_ns"] = delta
    return out


def roi_short_label(roi: str) -> str:
    """Extract ROI number label from names like Lifetime_fit-ROI2-ch2."""
    m = ROI_LABEL_RE.search(str(roi))
    if m:
        return f"ROI{m.group(1)}"
    return str(roi)


def _roi_sort_key(roi: str) -> tuple[int, str]:
    m = ROI_LABEL_RE.search(str(roi))
    return (int(m.group(1)), str(roi)) if m else (10**9, str(roi))


def plot_paper_style(df: pd.DataFrame, out_path_png: str, out_path_pdf: str) -> None:
    """Draw individual ROI traces in separate stimulated / control panels."""
    groups = ("stimulated", "control")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300, sharey=True)

    y_min = float(df["delta_lifetime_ns"].min())
    y_max = float(df["delta_lifetime_ns"].max())
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.02
    ylim = (y_min - y_pad, y_max + y_pad)

    for ax, group in zip(axes, groups):
        gdf = df[df["group"] == group]
        rois = sorted(gdf["roi"].unique(), key=_roi_sort_key)
        cmap = plt.get_cmap("tab10")
        for i, roi in enumerate(rois):
            rdf = gdf[gdf["roi"] == roi].sort_values("aligned_time_min")
            ax.plot(
                rdf["aligned_time_min"],
                rdf["delta_lifetime_ns"],
                color=cmap(i % 10),
                linewidth=1.2,
                alpha=0.9,
                marker="o",
                markersize=3.5,
                label=roi_short_label(roi),
                zorder=2,
            )

        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.6, zorder=1)
        ax.set_ylim(ylim)
        ax.text(
            0.15,
            ylim[0] + 0.92 * (ylim[1] - ylim[0]),
            "stim",
            ha="left",
            va="bottom",
            fontsize=8,
            color="gray",
        )
        ax.set_title(group.capitalize(), fontsize=10)
        ax.set_xlabel("Time (min)")
        ax.legend(frameon=False, fontsize=7, loc="best", title="ROI")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$\Delta$lifetime (ns)")
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path_png}")
    print(f"Saved: {out_path_pdf}")


def main() -> None:
    assert os.path.exists(XLSX_PATH), f"File not found: {XLSX_PATH}"
    df = parse_trimmed_xlsx(XLSX_PATH)
    df = add_aligned_delta(df, STIM_TIME_SEC)

    out_dir = os.path.dirname(XLSX_PATH)
    out_png = os.path.join(out_dir, f"{OUT_BASENAME}.png")
    out_pdf = os.path.join(out_dir, f"{OUT_BASENAME}.pdf")

    print(df.groupby(["group", "roi"]).size().rename("n_points"))
    print(
        "aligned_time_sec range:",
        float(df["aligned_time_sec"].min()),
        "to",
        float(df["aligned_time_sec"].max()),
    )
    plot_paper_style(df, out_png, out_pdf)


if __name__ == "__main__":
    main()
