"""
End-to-end training pipeline.

Usage:
    python src/train.py

Runs preprocessing → split → train (ridge, random_forest, xgboost)
→ evaluate → save plots and metrics report.
"""

import logging
import sys
from pathlib import Path

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from preprocessing import run_pipeline, get_feature_columns
from splitter import time_split
from models import train, predict, get_feature_importance
from evaluation import evaluate_all
from plotting import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_metrics_comparison,
    plot_residuals,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    # ── 1. Preprocessing ───────────────────────────────────────────────────
    logger.info("=== Step 1: Preprocessing ===")
    df = run_pipeline(save=True)
    feature_cols = get_feature_columns(df)

    # ── 2. Split ───────────────────────────────────────────────────────────
    logger.info("=== Step 2: Time-based split ===")
    X_train, X_test, y_train, y_test = time_split(df, feature_cols)
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # ── 3. Train all models ────────────────────────────────────────────────
    logger.info("=== Step 3: Training ===")
    model_names = ["ridge", "random_forest", "xgboost"]
    trained = {name: train(name, X_train, y_train) for name in model_names}

    # ── 4. Evaluate ────────────────────────────────────────────────────────
    logger.info("=== Step 4: Evaluation ===")
    metrics_df = evaluate_all(trained, X_test, y_test, predict)
    print("\n── Model Comparison ──")
    print(metrics_df.to_string())

    # ── 5. Plots ───────────────────────────────────────────────────────────
    logger.info("=== Step 5: Generating plots ===")
    plot_metrics_comparison(metrics_df)

    best_name = metrics_df["R2"].idxmax()
    best_model = trained[best_name]
    best_preds = predict(best_model, X_test)

    plot_actual_vs_predicted(y_test, best_preds, best_name)
    plot_residuals(y_test, best_preds, best_name)

    imp = get_feature_importance(best_model, feature_cols)
    if imp is not None:
        plot_feature_importance(imp, best_name)

    logger.info("=== Done. Outputs in outputs/ ===")
    print(f"\nBest model: {best_name}  R²={metrics_df.loc[best_name,'R2']}")


if __name__ == "__main__":
    main()
