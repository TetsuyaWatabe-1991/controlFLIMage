# -*- coding: utf-8 -*-
"""
Statistical analysis of manual mushroom spine ratings vs detection features.

Merges mushroom_spine_ratings.csv with per-flim *_mushroom_features.csv files,
then compares parameter distributions across rating groups.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# --- defaults ---
DEFAULT_ROOT = r"G:\ImagingData\Tetsuya\20260608\mushroom_multi_dend - Copy"
RATINGS_FILENAME = "mushroom_spine_ratings.csv"
FEATURES_GLOBS = ("*_respan_mushroom_features.csv", "*_mushroom_features.csv")

META_COLUMNS = {
    "flim_path",
    "base_name",
    "png_path",
    "ini_path",
    "deepd3_label",
    "nearest_neighbor_label",
    "rating_label",
    "rated_at",
}
PARAM_PREFIX = "param_"


def load_merged_dataset(root_folder: str) -> pd.DataFrame:
    ratings_path = os.path.join(root_folder, RATINGS_FILENAME)
    if not os.path.isfile(ratings_path):
        raise FileNotFoundError(f"Ratings CSV not found: {ratings_path}")

    ratings = pd.read_csv(ratings_path)
    if "rated_at" in ratings.columns:
        ratings = ratings.sort_values("rated_at").drop_duplicates("png_path", keep="last")
    feature_paths: list[str] = []
    seen_paths: set[str] = set()
    for pattern in FEATURES_GLOBS:
        for path in glob.glob(os.path.join(root_folder, "**", pattern), recursive=True):
            if path not in seen_paths:
                seen_paths.add(path)
                feature_paths.append(path)
    if not feature_paths:
        raise FileNotFoundError(f"No feature CSV files under {root_folder}")

    features = pd.concat([pd.read_csv(path) for path in sorted(feature_paths)], ignore_index=True)
    merged = features.merge(ratings, on="png_path", how="inner", suffixes=("", "_rating"))
    if merged.empty:
        raise ValueError("No rows after merging ratings with features (check png_path).")
    return merged


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_COLUMNS or col == "rating":
            continue
        if col.startswith(PARAM_PREFIX):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def mannwhitney_row(series_a: pd.Series, series_b: pd.Series) -> dict:
    a = series_a.dropna().astype(float)
    b = series_b.dropna().astype(float)
    if len(a) < 3 or len(b) < 3:
        return {"p_value": np.nan, "effect_rank_biserial": np.nan}
    u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    # rank-biserial correlation
    n1, n2 = len(a), len(b)
    effect = 1 - (2 * u_stat) / (n1 * n2)
    return {"p_value": float(p_value), "effect_rank_biserial": float(effect)}


def compare_groups(df: pd.DataFrame, feature_cols: list[str], group_a, group_b, label: str) -> pd.DataFrame:
    mask_a = df["rating"].isin(group_a)
    mask_b = df["rating"].isin(group_b)
    rows = []
    for col in feature_cols:
        a = df.loc[mask_a, col]
        b = df.loc[mask_b, col]
        mw = mannwhitney_row(a, b)
        rows.append({
            "comparison": label,
            "feature": col,
            "n_group_a": int(mask_a.sum()),
            "n_group_b": int(mask_b.sum()),
            "median_a": float(a.median()) if a.notna().any() else np.nan,
            "median_b": float(b.median()) if b.notna().any() else np.nan,
            "mean_a": float(a.mean()) if a.notna().any() else np.nan,
            "mean_b": float(b.mean()) if b.notna().any() else np.nan,
            "p_value": mw["p_value"],
            "effect_rank_biserial": mw["effect_rank_biserial"],
        })
    out = pd.DataFrame(rows)
    out["abs_effect"] = out["effect_rank_biserial"].abs()
    out = out.sort_values(["p_value", "abs_effect"], ascending=[True, False])
    return out


def kruskal_by_rating(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        groups = [g[col].dropna().astype(float).values for _, g in df.groupby("rating")]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        try:
            stat, p_value = stats.kruskal(*groups)
        except ValueError:
            continue
        rows.append({
            "feature": col,
            "kruskal_h": float(stat),
            "p_value": float(p_value),
        })
    out = pd.DataFrame(rows).sort_values("p_value")
    return out


def try_logistic_importance(df: pd.DataFrame, feature_cols: list[str], positive_ratings: set[int]) -> pd.DataFrame:
    """Logistic regression coefficients: positive_ratings=1, others=0."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return pd.DataFrame()

    y = df["rating"].isin(positive_ratings).astype(int)
    if y.nunique() < 2 or y.sum() < 5 or (1 - y).sum() < 5:
        return pd.DataFrame()

    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid = x.notna().all(axis=1)
    x = x.loc[valid]
    y = y.loc[valid]
    if len(x) < 20:
        return pd.DataFrame()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.5,
        max_iter=5000,
        random_state=0,
    )
    model.fit(x_scaled, y)
    coef = pd.DataFrame({
        "feature": feature_cols,
        "logistic_coef": model.coef_.ravel(),
        "abs_logistic_coef": np.abs(model.coef_.ravel()),
    }).sort_values("abs_logistic_coef", ascending=False)
    return coef


def rating_summary_table(df: pd.DataFrame, feature_cols: list[str], top_features: list[str]) -> pd.DataFrame:
    rows = []
    for rating in sorted(df["rating"].unique()):
        sub = df[df["rating"] == rating]
        row = {"rating": rating, "n": len(sub)}
        for feat in top_features:
            row[f"median_{feat}"] = float(sub[feat].median()) if feat in sub else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(root_folder: str, output_dir: str | None = None) -> str:
    df = load_merged_dataset(root_folder)
    if output_dir is None:
        output_dir = os.path.join(root_folder, "rating_analysis")
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = numeric_feature_columns(df)
    merged_path = os.path.join(output_dir, "merged_ratings_features.csv")
    df.to_csv(merged_path, index=False)

    comparisons = {
        "ideal_vs_reject_4_vs_1": ({4}, {1}),
        "good_vs_bad_34_vs_12": ({3, 4}, {1, 2}),
        "acceptable_plus_vs_reject_34_vs_1": ({3, 4}, {1}),
    }
    comparison_frames = []
    for label, (ga, gb) in comparisons.items():
        comparison_frames.append(compare_groups(df, feature_cols, ga, gb, label))
    all_cmp = pd.concat(comparison_frames, ignore_index=True)
    cmp_path = os.path.join(output_dir, "group_comparison_mannwhitney.csv")
    all_cmp.to_csv(cmp_path, index=False)

    kruskal = kruskal_by_rating(df, feature_cols)
    kruskal_path = os.path.join(output_dir, "kruskal_wallis_by_rating.csv")
    kruskal.to_csv(kruskal_path, index=False)

    logistic = try_logistic_importance(df, feature_cols, positive_ratings={3, 4})
    if not logistic.empty:
        logistic_path = os.path.join(output_dir, "logistic_good_vs_bad_coef.csv")
        logistic.to_csv(logistic_path, index=False)

    # Top features for summary: from 4 vs 1 comparison, p<0.05 and |effect|>0.2
    ideal_cmp = comparison_frames[0]
    top = ideal_cmp.loc[
        (ideal_cmp["p_value"] < 0.05) & (ideal_cmp["abs_effect"] >= 0.2),
        "feature",
    ].head(12).tolist()
    if not top:
        top = ideal_cmp.head(12)["feature"].tolist()
    summary = rating_summary_table(df, feature_cols, top)
    summary_path = os.path.join(output_dir, "rating_group_medians_top_features.csv")
    summary.to_csv(summary_path, index=False)

    counts = df["rating"].value_counts().sort_index()
    report_lines = [
        "Mushroom spine rating parameter analysis",
        f"Root folder: {root_folder}",
        f"Merged rows: {len(df)}",
        "Rating counts:",
        counts.to_string(),
        "",
        f"Outputs written to: {output_dir}",
        f"  - {os.path.basename(merged_path)}",
        f"  - {os.path.basename(cmp_path)}",
        f"  - {os.path.basename(kruskal_path)}",
    ]
    if not logistic.empty:
        report_lines.append(f"  - {os.path.basename(logistic_path)}")
    report_lines.append(f"  - {os.path.basename(summary_path)}")
    report_lines.append("")
    report_lines.append("Top discriminators (rating 4 vs 1, by p-value):")
    for _, row in ideal_cmp.head(10).iterrows():
        direction = "higher in 4" if row["median_a"] > row["median_b"] else "higher in 1"
        report_lines.append(
            f"  {row['feature']}: p={row['p_value']:.4g}, "
            f"med_4={row['median_a']:.4g}, med_1={row['median_b']:.4g} ({direction})"
        )
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as fobj:
        fobj.write(report_text)
    print(report_text)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mushroom spine ratings vs features.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Folder with ratings + feature CSVs")
    parser.add_argument("--output", default=None, help="Output directory (default: <root>/rating_analysis)")
    args = parser.parse_args()
    run_analysis(args.root, args.output)


if __name__ == "__main__":
    main()
