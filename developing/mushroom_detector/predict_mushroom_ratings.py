# -*- coding: utf-8 -*-
"""
Predict 1-4 mushroom spine ratings from detection features (trained on manual ratings).

Outputs CSV plus PNG copies named for easy manual vs predicted comparison.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_mushroom_ratings import (  # noqa: E402
    DEFAULT_ROOT,
    load_merged_dataset,
    numeric_feature_columns,
)

DEFAULT_OUTPUT_SUBDIR = "predicted_ratings_output"
TOP_N_FEATURES = 25
N_SPLITS = 5
RANDOM_STATE = 0


def top_feature_list(logistic_csv: str | None, all_features: list[str]) -> list[str]:
    if logistic_csv and os.path.isfile(logistic_csv):
        coef = pd.read_csv(logistic_csv).sort_values("abs_logistic_coef", ascending=False)
        selected = [f for f in coef["feature"].tolist() if f in all_features][:TOP_N_FEATURES]
        if selected:
            return selected
    return all_features


def build_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


def composite_rank_ratings(
    df: pd.DataFrame,
    logistic_csv: str,
    manual_counts: pd.Series,
) -> np.ndarray:
    """
    Weighted feature score (logistic coefficients) ranked into 1-4 bins.

    Bin sizes match the manual rating counts so the score scale is comparable.
    """
    coef_df = pd.read_csv(logistic_csv).sort_values("abs_logistic_coef", ascending=False)
    score = np.zeros(len(df), dtype=float)
    for _, row in coef_df.iterrows():
        feat = row["feature"]
        if feat not in df.columns:
            continue
        vals = pd.to_numeric(df[feat], errors="coerce")
        std = float(vals.std())
        if std <= 0 or np.isnan(std):
            continue
        z = (vals - vals.median()) / std
        score += float(row["logistic_coef"]) * z.fillna(0).to_numpy()

    # Higher score = better rating; assign bins by rank (worst first).
    order = np.argsort(score, kind="stable")
    ratings = np.ones(len(df), dtype=int)
    boundaries = []
    cum = 0
    for rating_level in sorted(manual_counts.index):
        cum += int(manual_counts[rating_level])
        boundaries.append(cum)
    # boundaries for 1,2,3,4 cumulative: e.g. 99, 133, 151, 169
    for rank_pos, idx in enumerate(order):
        r = 1
        for level, bound in zip(sorted(manual_counts.index), boundaries):
            if rank_pos < bound:
                r = int(level)
                break
        ratings[idx] = r
    return ratings, score


def predict_ratings(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["rating"].astype(int)
    model = build_model()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    predicted_cv = cross_val_predict(model, x, y, cv=cv, method="predict")
    proba_cv = cross_val_predict(model, x, y, cv=cv, method="predict_proba")

    model.fit(x, y)
    predicted_fit = model.predict(x)
    proba_fit = model.predict_proba(x)

    out = df.copy()
    out["predicted_rating_cv"] = predicted_cv.astype(int)
    out["predicted_rating_rf_fit"] = predicted_fit.astype(int)
    out["manual_rating"] = y
    out["rating_match_cv"] = out["predicted_rating_cv"] == out["manual_rating"]
    out["rating_match_rf_fit"] = out["predicted_rating_rf_fit"] == out["manual_rating"]

    classes = sorted(y.unique())
    for idx, cls in enumerate(model.named_steps["clf"].classes_):
        out[f"prob_cv_class_{int(cls)}"] = proba_cv[:, idx]
        out[f"prob_class_{int(cls)}"] = proba_fit[:, idx]
    return out, classes


def export_images(df: pd.DataFrame, output_dir: str) -> None:
    images_root = os.path.join(output_dir, "images")
    flat_dir = os.path.join(images_root, "all_named")
    by_pred_dir = os.path.join(images_root, "by_predicted_rating")
    by_man_dir = os.path.join(images_root, "by_manual_rating")
    os.makedirs(flat_dir, exist_ok=True)
    for score in (1, 2, 3, 4):
        os.makedirs(os.path.join(by_pred_dir, str(score)), exist_ok=True)
        os.makedirs(os.path.join(by_man_dir, str(score)), exist_ok=True)

    for _, row in df.iterrows():
        src = row["png_path"]
        if not isinstance(src, str) or not os.path.isfile(src):
            continue
        base = os.path.splitext(os.path.basename(src))[0]
        man = int(row["manual_rating"])
        pred = int(row["predicted_rating"])
        pred_cv = int(row["predicted_rating_cv"])
        name = f"{base}_man{man}_pred{pred}_rfcv{pred_cv}.png"
        dst_flat = os.path.join(flat_dir, name)
        shutil.copy2(src, dst_flat)
        shutil.copy2(src, os.path.join(by_pred_dir, str(pred), name))
        shutil.copy2(src, os.path.join(by_man_dir, str(man), name))


def run_prediction(
    root_folder: str,
    output_dir: str | None = None,
    logistic_csv: str | None = None,
) -> str:
    df = load_merged_dataset(root_folder)
    all_features = numeric_feature_columns(df)
    if logistic_csv is None:
        logistic_csv = os.path.join(root_folder, "rating_analysis", "logistic_good_vs_bad_coef.csv")
    feature_cols = top_feature_list(logistic_csv, all_features)

    scored, classes = predict_ratings(df, feature_cols)
    manual_counts = df["rating"].value_counts().sort_index()
    composite_rating, composite_score = composite_rank_ratings(
        df, logistic_csv, manual_counts
    )
    scored["composite_score"] = composite_score
    scored["predicted_rating_composite"] = composite_rating
    # Primary label for export: composite (parameter-driven, balanced 1-4 bins)
    scored["predicted_rating"] = composite_rating
    scored["rating_match"] = scored["predicted_rating"] == scored["manual_rating"]
    if output_dir is None:
        output_dir = os.path.join(root_folder, DEFAULT_OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    export_cols = [
        "png_path",
        "flim_path",
        "base_name",
        "spine_index",
        "deepd3_label",
        "manual_rating",
        "predicted_rating",
        "predicted_rating_composite",
        "composite_score",
        "predicted_rating_cv",
        "predicted_rating_rf_fit",
        "rating_match",
        "rating_match_cv",
        "rating_match_rf_fit",
        "shaft_to_head_um",
        "seg_spine_pred_mean",
        "shaft_roi_raw_intensity_percentile_image",
        "seg_aspect_ratio",
    ] + [c for c in scored.columns if c.startswith("prob_")]
    export_cols = [c for c in export_cols if c in scored.columns]
    csv_path = os.path.join(output_dir, "predicted_ratings.csv")
    scored[export_cols].to_csv(csv_path, index=False)

    meta_path = os.path.join(output_dir, "model_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as fobj:
        fobj.write("Predicted mushroom spine ratings\n")
        fobj.write(f"Root folder: {root_folder}\n")
        fobj.write(f"Rows: {len(scored)}\n")
        fobj.write(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}\n")
        fobj.write(f"Classes: {classes}\n")
        fobj.write(f"Match rate (composite): {scored['rating_match'].mean():.3f}\n")
        fobj.write(f"Match rate (RF 5-fold CV): {scored['rating_match_cv'].mean():.3f}\n")
        fobj.write(f"Match rate (RF fitted): {scored['rating_match_rf_fit'].mean():.3f}\n")
        fobj.write("\nPredicted rating counts (composite):\n")
        fobj.write(scored["predicted_rating"].value_counts().sort_index().to_string())
        fobj.write("\n\nPredicted rating counts (RF CV):\n")
        fobj.write(scored["predicted_rating_cv"].value_counts().sort_index().to_string())
        fobj.write("\n")

    export_images(scored, output_dir)

    print(f"Wrote: {csv_path}")
    print(f"Images: {os.path.join(output_dir, 'images')}")
    print(f"Match rate composite: {scored['rating_match'].mean():.1%}")
    print(f"Match rate RF 5-fold CV: {scored['rating_match_cv'].mean():.1%}")
    print("Predicted counts (composite):")
    print(scored["predicted_rating"].value_counts().sort_index().to_string())
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict spine ratings from features.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--logistic-csv", default=None)
    args = parser.parse_args()
    run_prediction(args.root, args.output, args.logistic_csv)


if __name__ == "__main__":
    main()
