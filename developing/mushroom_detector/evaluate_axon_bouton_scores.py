# -*- coding: utf-8 -*-
"""
Score rated mushroom candidates for axon bouton / crossing-fiber likelihood.

Loads merged ratings + features, computes axial intensity scores per spine,
and summarizes bandpass false positives (manual rating 1, seg area pass).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_mushroom_ratings import DEFAULT_ROOT, load_merged_dataset  # noqa: E402
from mushroom_axon_bouton_filter import (  # noqa: E402
    compute_bouton_scores,
    passes_bouton_reject,
    raw_local_z_mip,
)
from mushroom_bandpass import passes_seg_area_bandpass  # noqa: E402
from respan_mushroom_core import respan_export_paths, respan_run_dir  # noqa: E402


class RawMipCache:
    """Cache raw Z-local MIP per FLIM file."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(
        self,
        flim_path: str,
        z_pix: float,
        *,
        channel: int = 2,
    ) -> np.ndarray | None:
        key = f"{flim_path}|ch{channel}"
        if key not in self._cache:
            loaded = _load_raw_stack(flim_path, channel=channel)
            if loaded is None:
                return None
            self._cache[key] = loaded
        raw_zyx, _ = self._cache[key]
        return raw_local_z_mip(raw_zyx, z_pix)


def _load_raw_stack(flim_path: str, *, channel: int = 2) -> tuple[np.ndarray, Path] | None:
    flim = Path(flim_path)
    tiff_path, _ = respan_export_paths(flim, channel=channel)
    if not tiff_path.exists():
        return None
    raw = tf.imread(tiff_path)
    if raw.ndim == 4:
        raw = raw[:, 0]
    return np.asarray(raw, dtype=np.float32), tiff_path


def score_row(
    row: pd.Series,
    cache: RawMipCache,
    *,
    channel: int = 2,
) -> dict[str, float | bool | str]:
    flim_path = str(row["flim_path"])
    z_pix = float(row["head_z_pix"])
    head_x = float(row["head_x_pix"])
    head_y = float(row["head_y_pix"])
    dend_slope = float(row["dend_slope"])
    xy_um = float(row["xy_pixel_um"])

    raw_mip = cache.get(flim_path, z_pix, channel=channel)
    if raw_mip is None:
        return {
            "bouton_score_ok": False,
            "bouton_score_error": "raw stack not found",
        }

    if not np.isfinite(dend_slope):
        return {
            "bouton_score_ok": False,
            "bouton_score_error": "invalid dend_slope",
        }

    scores = compute_bouton_scores(
        raw_mip,
        head_x,
        head_y,
        dend_slope,
        xy_um,
    )
    reject, reject_reason = passes_bouton_reject(scores)
    return {
        "bouton_score_ok": True,
        "bouton_score_error": "",
        "bouton_reject": reject,
        "bouton_reject_reason": reject_reason,
        **scores,
    }


def add_seg_bandpass(df: pd.DataFrame, seg_area_band: tuple[float, float]) -> pd.DataFrame:
    out = df.copy()
    passes: list[bool] = []
    for _, row in out.iterrows():
        ok, _ = passes_seg_area_bandpass(float(row["seg_area_um2"]), seg_area_band=seg_area_band)
        passes.append(ok)
    out["seg_bandpass_pass"] = passes
    return out


def print_summary(scored: pd.DataFrame, *, seg_area_band: tuple[float, float]) -> None:
    ok = scored[scored["bouton_score_ok"]]
    n_ok = len(ok)
    n_total = len(scored)
    print("=== Axon bouton score summary ===")
    print(f"Scored: {n_ok}/{n_total}")
    if n_ok == 0:
        return

    fp = ok[(ok["rating"] == 1) & ok["seg_bandpass_pass"]]
    fn = ok[(ok["rating"] == 4) & ~ok["seg_bandpass_pass"]]
    print(f"Bandpass FP (rating 1 + seg pass): {len(fp)}")
    print(f"Bandpass FN (rating 4 + seg reject): {len(fn)}")

    if len(fp):
        fp_reject = fp[fp["bouton_reject"]]
        print(
            f"Bouton filter would reject {len(fp_reject)}/{len(fp)} FP "
            f"({100 * len(fp_reject) / len(fp):.1f}%)"
        )
        print("\nTop FP by bouton_combined_score (higher = more bouton-like):")
        cols = [
            "base_name",
            "spine_index",
            "rating",
            "seg_area_um2",
            "bouton_combined_score",
            "bright_span_um",
            "neck_contrast",
            "bright_axis_angle_deg",
            "bouton_reject",
        ]
        top = fp.sort_values("bouton_combined_score", ascending=False)[cols]
        print(top.to_string(index=False))

    if len(fn):
        fn_reject = fn[fn["bouton_reject"]]
        print(
            f"\nBouton filter would also reject {len(fn_reject)}/{len(fn)} FN "
            f"(collateral damage)"
        )

    r4_pass = ok[(ok["rating"] == 4) & ok["seg_bandpass_pass"]]
    if len(r4_pass):
        collateral = r4_pass[r4_pass["bouton_reject"]]
        print(
            f"Collateral on rating 4 pass: {len(collateral)}/{len(r4_pass)}"
        )

    combined_pass = ok[ok["seg_bandpass_pass"] & ~ok["bouton_reject"]]
    print(
        f"\nCombined gate (seg pass + bouton keep): {len(combined_pass)}/{n_ok} "
        f"({100 * len(combined_pass) / n_ok:.1f}%)"
    )
    if len(fp):
        fp_kept = fp[~fp["bouton_reject"]]
        print(
            f"FP after combined gate: {len(fp_kept)}/{len(fp)} "
            f"(removed {len(fp) - len(fp_kept)})"
        )
    if len(r4_pass):
        r4_kept = r4_pass[~r4_pass["bouton_reject"]]
        print(
            f"Rating 4 pass after combined gate: {len(r4_kept)}/{len(r4_pass)} "
            f"(lost {len(r4_pass) - len(r4_kept)})"
        )

    print("\nMann-Whitney: bouton_combined_score rating1 vs rating4 (seg pass only):")
    r1 = ok[(ok["rating"] == 1) & ok["seg_bandpass_pass"]]["bouton_combined_score"].dropna()
    r4 = ok[(ok["rating"] == 4) & ok["seg_bandpass_pass"]]["bouton_combined_score"].dropna()
    if len(r1) >= 3 and len(r4) >= 3:
        from scipy import stats

        u, p = stats.mannwhitneyu(r1, r4, alternative="two-sided")
        print(f"  n_r1={len(r1)}, n_r4={len(r4)}, p={p:.4g}, median_r1={r1.median():.3f}, median_r4={r4.median():.3f}")


def parse_band(args_value: str) -> tuple[float, float]:
    parts = args_value.split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two floats, e.g. '0.20 0.55'")
    return float(parts[0]), float(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate axon bouton scores on rated dataset.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Dataset root folder")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <root>/rating_analysis/axon_bouton_scores.csv)",
    )
    parser.add_argument(
        "--seg-area-band",
        default="0.20 0.55",
        type=parse_band,
        help="Seg area bandpass range in um^2",
    )
    parser.add_argument("--channel", type=int, default=2)
    args = parser.parse_args()

    root = args.root
    output = args.output or os.path.join(root, "rating_analysis", "axon_bouton_scores.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    df = load_merged_dataset(root)
    df = add_seg_bandpass(df, args.seg_area_band)

    cache = RawMipCache()
    score_rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        result = score_row(row, cache, channel=args.channel)
        score_rows.append(result)
        if (i + 1) % 20 == 0:
            print(f"Scored {i + 1}/{len(df)}...")

    for key in score_rows[0]:
        df[key] = [r.get(key, np.nan) for r in score_rows]

    df.to_csv(output, index=False)
    print(f"Wrote {output}")
    print_summary(df, seg_area_band=args.seg_area_band)

    summary_path = output.replace(".csv", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_summary(df, seg_area_band=args.seg_area_band)
        fh.write(buf.getvalue())
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
