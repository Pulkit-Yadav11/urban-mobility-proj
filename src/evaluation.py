"""
Evaluation utilities: metrics, comparison table, and report generation.
"""

import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import REPORTS_DIR

logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }


def evaluate_all(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predict_fn,             # callable: (model, X) → np.ndarray
) -> pd.DataFrame:
    """
    Evaluate every model in `models` and return a comparison DataFrame.
    Also saves results to outputs/reports/metrics.json.
    """
    records = {}
    for name, model in models.items():
        preds = predict_fn(model, X_test)
        records[name] = compute_metrics(y_test, preds)
        logger.info("%s → %s", name, records[name])

    df = pd.DataFrame(records).T.sort_values("R2", ascending=False)

    out = REPORTS_DIR / "metrics.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Metrics saved → %s", out)

    return df
