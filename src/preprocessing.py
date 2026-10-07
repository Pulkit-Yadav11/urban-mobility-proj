"""
Data loading, cleaning, and feature engineering.
Returns a clean, fully-featured DataFrame ready for modelling.
"""

import pandas as pd
import numpy as np
import logging

from config import RAW_DATA, PROCESSED_DIR

logger = logging.getLogger(__name__)


# ── 1. Load ────────────────────────────────────────────────────────────────

def load_raw(path=RAW_DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date_time"])
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ── 2. Clean ───────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df
        .sort_values("date_time")
        .drop_duplicates(subset="date_time", keep="last")
        .set_index("date_time")
        # Enforce an explicit hourly grid; interpolate short gaps linearly
        .asfreq("h")
    )

    # Numerical columns: linear interpolation for short gaps (≤3 h), then ffill
    num_cols = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]
    df[num_cols] = (
        df[num_cols]
        .interpolate(method="linear", limit=3)
        .ffill()
        .bfill()
    )

    # Categorical: forward-fill
    cat_cols = ["weather_main", "weather_description", "holiday"]
    df[cat_cols] = df[cat_cols].ffill().bfill()

    # holiday column uses the string "None" for non-holidays
    df["holiday"] = df["holiday"].fillna("None")

    logger.info("After cleaning: %d rows, %d nulls",
                len(df), df.isnull().sum().sum())
    return df


# ── 3. Feature engineering ─────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["hour"]        = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"]       = idx.month
    df["is_weekend"]  = (idx.dayofweek >= 5).astype(int)
    df["is_peak_hour"] = idx.hour.isin(range(7, 10)).astype(int) | \
                         idx.hour.isin(range(16, 20)).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    tv = df["traffic_volume"]
    df["lag_1h"]   = tv.shift(1)
    df["lag_2h"]   = tv.shift(2)
    df["lag_24h"]  = tv.shift(24)
    df["lag_168h"] = tv.shift(168)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # Shift by 1 first to avoid leakage (rolling uses only past values)
    past = df["traffic_volume"].shift(1)
    df["rolling_mean_3h"]  = past.rolling(3).mean()
    df["rolling_mean_24h"] = past.rolling(24).mean()
    df["rolling_std_3h"]   = past.rolling(3).std()
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["is_holiday"] = (df["holiday"] != "None").astype(int)
    df = pd.get_dummies(df, columns=["weather_main"], prefix="wm", dtype=int)
    df = df.drop(columns=["holiday", "weather_description"], errors="ignore")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = encode_categoricals(df)
    # Drop rows with NaN introduced by lags / rolling (first 168 rows)
    before = len(df)
    df = df.dropna()
    logger.info("Dropped %d NaN rows after feature engineering; %d remain",
                before - len(df), len(df))
    return df


# ── 4. Public API ──────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all model input columns (everything except the target)."""
    return [c for c in df.columns if c != "traffic_volume"]


def run_pipeline(save: bool = True) -> pd.DataFrame:
    df = load_raw()
    df = clean(df)
    df = build_features(df)
    if save:
        out = PROCESSED_DIR / "features.parquet"
        df.to_parquet(out)
        logger.info("Saved processed data → %s", out)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")
    df = run_pipeline()
    print(df.shape, "\n", df.head(2))
