#!/usr/bin/env python3
"""
XGBoost model to predict GEMM kernel performance from tile static-analysis metrics.

Workflow (Ansor-style):
  1. Generate tile configurations and compute their features with tile_metrics.py
  2. Simulate them (run.sh / many_gemms.sh) to get cycle counts
  3. Train this model on (features, cycles) pairs          ← you are here
  4. Use a genetic algorithm to propose new candidates
  5. Score candidates with the trained model (predict subcommand)
  6. Simulate only the top-N candidates and retrain

Target column: Global Sim E2E_dma (clock cycles; lower is better).
Sample weights = normalized throughput (min_cycles / cycles) so the model
focuses on fast configurations, matching Ansor's weighting strategy.

Usage:
    python boost.py train <features.csv> <results.csv> [--save model.json]
    python boost.py predict <features.csv> --model model.json
    python boost.py train --help

The features CSV is produced by myrtle/scripts/tile_metrics.py (one row per tile config).
The results CSV is produced by myrtle-experiments/combineTilingSchemeDataIntoSingleCSV.py
(or the per-experiment extract step); it must contain FakeNN JSON Name and
Global Sim E2E_dma columns.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

TARGET_COL = "Global Sim E2E_dma"
JOIN_KEY = "FakeNN JSON Name"
# Columns that are strings and not useful as numeric features
_DROP = {"JSON Name"}


def _encode_remainder_tiles(df: pd.DataFrame) -> pd.DataFrame:
    """Expand 'remainderTiles' (e.g. 'MN0') into three binary indicator columns."""
    rt = df["remainderTiles"].fillna("000").astype(str)
    df = df.copy()
    df["has_M_remainder"] = rt.str.contains("M").astype(int)
    df["has_N_remainder"] = rt.str.contains("N").astype(int)
    df["has_K_remainder"] = rt.str.contains("K").astype(int)
    return df.drop(columns=["remainderTiles"])


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a pure-numeric feature DataFrame from a raw features CSV."""
    df = _encode_remainder_tiles(df)
    drop = _DROP | {JOIN_KEY}
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.select_dtypes(include=[np.number])
    return df.reindex(sorted(df.columns), axis=1)  # canonical column order


def build_dataset(features_path: str, results_path: str):
    """Load, join, and clean features + results into (X, y, cycles, feature_names)."""
    feats_raw = pd.read_csv(features_path)
    results = pd.read_csv(results_path)[[JOIN_KEY, TARGET_COL]].copy()

    print(f"Cycles range: {results[TARGET_COL].min():.0f} – {results[TARGET_COL].max():.0f} ")

    # Discard timeouts (marked as -1); the GA blocklist handles avoidance instead.
    results = results[results[TARGET_COL] > 0]

    merged = feats_raw.merge(results, on=JOIN_KEY, how="inner")
    if merged.empty:
        sys.exit(
            f"No rows after joining on '{JOIN_KEY}'. "
            "Check that both CSVs use the same tile name format."
        )

    y_cycles = merged[TARGET_COL].values.astype(float)

    X = _feature_matrix(merged.drop(columns=[TARGET_COL]))

    # Normalized throughput: best tile → 1.0, slower tiles → values < 1.0
    y = y_cycles.min() / y_cycles

    return X, y, y_cycles, X.columns.tolist()


def train_model(X: pd.DataFrame, y: np.ndarray, y_cycles: np.ndarray) -> xgb.Booster:
    # Weight samples by their normalized throughput so fast configs matter more
    weights = y

    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=X.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=X.columns.tolist())

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    preds = model.predict(dval)
    r2 = r2_score(y_val, preds)
    mape = mean_absolute_percentage_error(y_val, preds) * 100
    print(f"\nValidation  R²: {r2:.4f}   MAPE: {mape:.2f}%")

    return model


def rank_candidates(model: xgb.Booster, features_path: str) -> list[tuple[str, float]]:
    """Score and rank candidate tile configurations, best first."""
    feats_raw = pd.read_csv(features_path)
    names = feats_raw[JOIN_KEY].tolist()
    X = _feature_matrix(feats_raw)
    scores = model.predict(xgb.DMatrix(X, feature_names=X.columns.tolist()))
    ranked = sorted(zip(names, scores.tolist()), key=lambda t: -t[1])
    print(f"\n{'Rank':>4}  {'Score':>8}  Configuration")
    for i, (name, score) in enumerate(ranked, 1):
        print(f"{i:>4}  {score:>8.4f}  {name}")
    return ranked


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost GEMM tile-size predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    tr = sub.add_parser("train", help="Train model from features + results CSVs")
    tr.add_argument("features", help="features CSV (from tile_metrics.py)")
    tr.add_argument("results", help="results CSV (must contain Global Sim E2E_dma)")
    tr.add_argument("--save", default="model.json", metavar="PATH",
                    help="where to save the trained model (default: model.json)")

    pr = sub.add_parser("predict", help="Rank candidate configs with a saved model")
    pr.add_argument("features", help="features CSV for candidates to rank")
    pr.add_argument("--model", default="model.json", metavar="PATH",
                    help="path to saved model (default: model.json)")

    args = parser.parse_args()

    if args.cmd == "train":
        X, y, y_cycles, feat_names = build_dataset(args.features, args.results)
        print(f"Dataset: {len(X)} configurations, {len(feat_names)} features")
        print(f"Cycles range: {y_cycles.min():.0f} – {y_cycles.max():.0f}  "
              f"(best throughput weight = 1.0, median = {np.median(y):.3f})")
        model = train_model(X, y, y_cycles)
        model.save_model(args.save)
        print(f"\nModel saved → {args.save}")

    elif args.cmd == "predict":
        model = xgb.Booster()
        model.load_model(args.model)
        rank_candidates(model, args.features)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
