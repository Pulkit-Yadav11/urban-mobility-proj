"""
Central configuration for the Urban Mobility Modelling project.
All paths, constants, and hyperparameters live here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
RAW_DATA      = ROOT / "data" / "raw"   / "Metro_Interstate_Traffic_Volume.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
PLOTS_DIR     = ROOT / "outputs" / "plots"
REPORTS_DIR   = ROOT / "outputs" / "reports"

for _p in [PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Feature lists ──────────────────────────────────────────────────────────
TARGET = "traffic_volume"

CALENDAR_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_peak_hour",
]

LAG_FEATURES = [
    "lag_1h", "lag_2h", "lag_24h", "lag_168h",
]

ROLLING_FEATURES = [
    "rolling_mean_3h", "rolling_mean_24h", "rolling_std_3h",
]

WEATHER_NUM_FEATURES = ["temp", "rain_1h", "snow_1h", "clouds_all"]

# One-hot prefix — actual column names are built at runtime
WEATHER_CAT_COLS = ["weather_main"]

# ── Time split ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80          # first 80 % chronologically

# ── Model hyperparameters ──────────────────────────────────────────────────
RIDGE_ALPHA = 1.0

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
)

# ── Congestion bands ───────────────────────────────────────────────────────
CONGESTION_BANDS = [
    (0,    1_000, "Free Flow"),
    (1_000, 3_500, "Moderate Density"),
    (3_500, float("inf"), "Heavy Congestion"),
]
