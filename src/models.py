"""
Model definitions, training, persistence, and congestion classification.
All three models (Ridge, RandomForest, XGBoost) share the same interface.
"""

import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import RIDGE_ALPHA, XGB_PARAMS, RF_PARAMS, MODELS_DIR, CONGESTION_BANDS

logger = logging.getLogger(__name__)

# ── Registry: name → constructor ───────────────────────────────────────────
MODEL_REGISTRY: dict[str, object] = {
    "ridge": lambda: Ridge(alpha=RIDGE_ALPHA),
    "random_forest": lambda: RandomForestRegressor(**RF_PARAMS),
    "xgboost": lambda: XGBRegressor(**XGB_PARAMS),
}


def train(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save: bool = True,
) -> object:
    """Train a named model and optionally persist it."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")

    model = MODEL_REGISTRY[name]()
    logger.info("Training %s on %d samples …", name, len(X_train))
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    if save:
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        logger.info("Model saved → %s", path)

    return model


def load(name: str) -> object:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run training first.")
    return joblib.load(path)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    preds = model.predict(X)
    return np.clip(preds, 0, None)   # traffic volume cannot be negative


def classify_congestion(volume: float) -> str:
    for lo, hi, label in CONGESTION_BANDS:
        if lo <= volume < hi:
            return label
    return "Unknown"


def get_feature_importance(model, feature_cols: list[str]) -> pd.Series | None:
    """Return a sorted Series of feature importances, or None if unavailable."""
    if hasattr(model, "feature_importances_"):
        return (
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
        )
    if hasattr(model, "coef_"):
        return (
            pd.Series(np.abs(model.coef_), index=feature_cols)
            .sort_values(ascending=False)
        )
    return None
