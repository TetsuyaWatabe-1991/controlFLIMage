# -*- coding: utf-8 -*-
"""
Export bandpass false negatives / false positives for manual visual review.

False negative (rating4_missed): manual rating 4 but fails the bandpass gate.
False positive (rating1_false_pass): manual rating 1 but passes the bandpass gate.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_mushroom_ratings import DEFAULT_ROOT, load_merged_dataset  # noqa: E402
from mushroom_bandpass import (  # noqa: E402
    SEG_AREA_UM2_BAND,
    passes_seg_area_bandpass,
)

DEFAULT_OUTPUT_SUBDIR = "bandpass_review_output"
TEXT_PANEL_WIDTH_IN = 5.0


def evaluate_bandpass_row(
    row: pd.Series,
    *,
    seg_area_band: tuple[float, float],
) -> dict[str, Any]:
    seg_val = float(row["seg_area_um2"])
    seg_ok, seg_reason = passes_seg_area_bandpass(
        seg_val,
        seg_area_band=seg_area_band,
    )
    return {
        "bandpass_pass": seg_ok,
        "bandpass_reject_reason": seg_reason,
        "seg_bandpass_pass": seg_ok,
        "seg_reject_reason": seg_reason,
    }


def add_bandpass_columns(
    df: pd.DataFrame,
    *,
    seg_area_band: tuple[float, float],
) -> pd.DataFrame:
    if "seg_area_um2" not in df.columns:
        raise KeyError("seg_area_um2 column not found in merged dataset.")
    out = df.copy()
    results = [
        evaluate_bandpass_row(row, seg_area_band=seg_area_band)
        for _, row in out.iterrows()
    ]
    for key in results[0]:
        out[key] = [result[key] for result in results]
    out["seg_area_um2_eval"] = pd.to_numeric(out["seg_area_um2"], errors="coerce")
    return out


def print_bandpass_classification_summary(scored: pd.DataFrame) -> None:
    passed = scored[scored["bandpass_pass"]]
    n = len(scored)
    n_pass = len(passed)
    r4 = scored[scored["rating"] == 4]
    r1 = scored[scored["rating"] == 1]
    good = scored[scored["rating"] >= 3]
    print("=== Bandpass classification summary (seg area only) ===")
    print(f"Total rated: {n}")
    print(f"Bandpass pass: {n_pass}/{n} ({100 * n_pass / n:.1f}%)")
    if len(good):
        good_pass = passed[passed["rating"] >= 3]
        print(
            f"Good (rating 3-4) pass: {len(good_pass)}/{len(good)} "
            f"({100 * len(good_pass) / len(good):.1f}%)"
        )
    if len(r4):
        r4_pass = passed[passed["rating"] == 4]
        print(
            f"Rating 4 pass: {len(r4_pass)}/{len(r4)} "
            f"({100 * len(r4_pass) / len(r4):.1f}%)"
        )
        print(f"Rating 4 missed (FN): {len(r4) - len(r4_pass)}")
    if len(r1):
        r1_pass = passed[passed["rating"] == 1]
        print(
            f"Rating 1 false pass (FP): {len(r1_pass)}/{len(r1)} "
            f"({100 * len(r1_pass) / len(r1):.1f}%)"
        )
    print("Pass count by manual rating:")
    print(passed["rating"].value_counts().sort_index().to_string())


def gate_status_label(passed: bool) -> str:
    return "PASS" if passed else "REJECT"


def optional_metric(row: pd.Series, column: str, label: str, unit: str) -> str | None:
    if column not in row.index:
        return None
    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value):
        return None
    return f"{label}: {float(value):.3f} {unit}"


def build_annotation_lines(
    row: pd.Series,
    *,
    review_category: str,
    seg_area_band: tuple[float, float],
) -> tuple[list[str], str]:
    if review_category == "rating4_missed":
        header = "RATING 4 MISSED (bandpass reject)"
        banner_color = "#c0392b"
    else:
        header = "RATING 1 FALSE PASS (bandpass accept)"
        banner_color = "#d68910"

    lines = [
        header,
        "Bandpass metric: seg area only",
        f"Manual rating: {int(row['rating'])} ({row.get('rating_label', '')})",
        f"Bandpass: {'PASS' if row['bandpass_pass'] else 'REJECT'}",
        "",
        f"seg area: {row['seg_area_um2_eval']:.3f} um^2  "
        f"[band {seg_area_band[0]:.2f}-{seg_area_band[1]:.2f}]  "
        f"{gate_status_label(bool(row['seg_bandpass_pass']))}",
    ]
    if row["seg_reject_reason"]:
        lines.append(f"  -> {row['seg_reject_reason']}")
    for extra in (
        optional_metric(row, "head_vol_um3", "head vol", "um^3"),
        optional_metric(row, "head_area_um2", "head area", "um^2"),
        optional_metric(row, "shaft_to_head_um", "shaft-to-head (info)", "um"),
    ):
        if extra:
            lines.append(extra)
    if row["bandpass_reject_reason"]:
        lines.extend(["", "Reject summary:", row["bandpass_reject_reason"]])
    return lines, banner_color


def save_annotated_review_image(
    row: pd.Series,
    dst_path: str,
    annotation_lines: list[str],
    *,
    banner_color: str,
) -> None:
    png_path = row.get("png_path")
    if not isinstance(png_path, str) or not os.path.isfile(png_path):
        raise FileNotFoundError(f"Source 3-panel PNG not found: {png_path}")

    panel_img = mpimg.imread(png_path)
    img_h, img_w = panel_img.shape[0], panel_img.shape[1]
    panel_w_in = max(img_w / 120.0, 8.0)
    panel_h_in = max(img_h / 120.0, 3.0)

    fig = plt.figure(figsize=(panel_w_in + TEXT_PANEL_WIDTH_IN, panel_h_in), dpi=120)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[panel_w_in, TEXT_PANEL_WIDTH_IN],
        wspace=0.04,
    )
    ax_img = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    ax_img.imshow(panel_img)
    ax_img.axis("off")

    ax_text.axis("off")
    ax_text.text(
        0.0,
        0.98,
        annotation_lines[0],
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox={
            "facecolor": banner_color,
            "alpha": 0.95,
            "pad": 5,
            "edgecolor": "none",
        },
    )
    body = "\n".join(annotation_lines[1:])
    ax_text.text(
        0.0,
        0.90,
        body,
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        color="black",
        linespacing=1.4,
    )
    fig.savefig(dst_path, bbox_inches="tight", pad_inches=0.04, dpi=120)
    plt.close(fig)


def safe_export_name(base: str, row: pd.Series, suffix: str) -> str:
    seg = row["seg_area_um2_eval"]
    return f"{base}_r{int(row['rating'])}_seg{seg:.2f}_{suffix}.png"


def export_review_images(
    df: pd.DataFrame,
    category_dir: str,
    suffix: str,
    *,
    review_category: str,
    seg_area_band: tuple[float, float],
) -> int:
    os.makedirs(category_dir, exist_ok=True)
    exported = 0
    for _, row in df.iterrows():
        png_path = row.get("png_path")
        if not isinstance(png_path, str) or not os.path.isfile(png_path):
            print(f"  skip missing source PNG: {png_path}")
            continue
        base = os.path.splitext(os.path.basename(png_path))[0]
        dst_name = safe_export_name(base, row, suffix)
        lines, banner_color = build_annotation_lines(
            row,
            review_category=review_category,
            seg_area_band=seg_area_band,
        )
        save_annotated_review_image(
            row,
            os.path.join(category_dir, dst_name),
            lines,
            banner_color=banner_color,
        )
        exported += 1
    return exported


def run_export(
    root_folder: str,
    output_dir: str | None = None,
    *,
    seg_area_band: tuple[float, float] = SEG_AREA_UM2_BAND,
    export_images: bool = True,
) -> str:
    df = load_merged_dataset(root_folder)
    scored = add_bandpass_columns(df, seg_area_band=seg_area_band)
    print_bandpass_classification_summary(scored)

    rating4_missed = scored[(scored["rating"] == 4) & (~scored["bandpass_pass"])].copy()
    rating1_false_pass = scored[(scored["rating"] == 1) & scored["bandpass_pass"]].copy()

    if output_dir is None:
        output_dir = os.path.join(root_folder, DEFAULT_OUTPUT_SUBDIR)
    images_root = os.path.join(output_dir, "images")
    missed_dir = os.path.join(images_root, "rating4_missed")
    false_pass_dir = os.path.join(images_root, "rating1_false_pass")
    os.makedirs(output_dir, exist_ok=True)

    copied_missed = 0
    copied_false_pass = 0
    if export_images:
        copied_missed = export_review_images(
            rating4_missed,
            missed_dir,
            "missed",
            review_category="rating4_missed",
            seg_area_band=seg_area_band,
        )
        copied_false_pass = export_review_images(
            rating1_false_pass,
            false_pass_dir,
            "false_pass",
            review_category="rating1_false_pass",
            seg_area_band=seg_area_band,
        )

    export_cols = [
        "png_path",
        "flim_path",
        "base_name",
        "spine_index",
        "rating",
        "rating_label",
        "bandpass_pass",
        "bandpass_reject_reason",
        "seg_bandpass_pass",
        "seg_reject_reason",
        "seg_area_um2_eval",
        "head_vol_um3",
        "head_area_um2",
        "shaft_to_head_um",
        "auto_rating",
        "auto_rating_label",
    ]
    export_cols = [col for col in export_cols if col in scored.columns]

    cases = pd.concat(
        [
            rating4_missed.assign(review_category="rating4_missed"),
            rating1_false_pass.assign(review_category="rating1_false_pass"),
        ],
        ignore_index=True,
    )
    csv_path = os.path.join(output_dir, "bandpass_review_cases.csv")
    cases[export_cols + ["review_category"]].to_csv(csv_path, index=False)

    passed = int(scored["bandpass_pass"].sum())
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fobj:
        fobj.write("Bandpass review export (seg area only)\n")
        fobj.write(f"Root folder: {root_folder}\n")
        fobj.write(f"Seg area band (um^2): {seg_area_band[0]} - {seg_area_band[1]}\n")
        fobj.write(f"Total rated rows: {len(scored)}\n")
        fobj.write(f"Bandpass pass: {passed}/{len(scored)}\n")
        fobj.write(f"Rating 4 total: {(scored['rating'] == 4).sum()}\n")
        fobj.write(
            f"Rating 4 missed (FN): {len(rating4_missed)} "
            f"(copied {copied_missed} images)\n"
        )
        fobj.write(f"Rating 1 total: {(scored['rating'] == 1).sum()}\n")
        fobj.write(
            f"Rating 1 false pass (FP): {len(rating1_false_pass)} "
            f"(copied {copied_false_pass} images)\n"
        )

    print(f"Wrote: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Rating 4 missed images: {missed_dir} ({copied_missed} files)")
    print(f"Rating 1 false pass images: {false_pass_dir} ({copied_false_pass} files)")
    print(
        f"Bandpass pass rate: {passed}/{len(scored)}; "
        f"FN={len(rating4_missed)}, FP={len(rating1_false_pass)}"
    )
    return output_dir


def parse_band(value: str) -> tuple[float, float]:
    parts = value.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two numbers, e.g. '0.20 0.55'")
    return float(parts[0]), float(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export bandpass FN/FP spine PNGs for manual review.",
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--seg-area-band",
        type=parse_band,
        default=SEG_AREA_UM2_BAND,
        help="Inclusive seg area band in um^2, e.g. '0.20 0.55'",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip PNG export; print classification summary only.",
    )
    args = parser.parse_args()
    run_export(
        args.root,
        args.output,
        seg_area_band=args.seg_area_band,
        export_images=not args.no_images,
    )


if __name__ == "__main__":
    main()
