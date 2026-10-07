"""
Time-based sequential train / test split — no shuffle, no leakage.
"""

import pandas as pd
from config import TARGET, TRAIN_RATIO


def time_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series]:
    """
    Returns X_train, X_test, y_train, y_test in chronological order.
    The DataFrame must already be sorted by time index.
    """
    split_idx = int(len(df) * ratio)
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test
