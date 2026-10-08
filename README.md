# Urban Mobility Modelling
### Short-Term Traffic Flow & Spatial Density Forecasting

**Domain:** Spatial-Temporal Data Mining / Intelligent Transportation Systems  
**Dataset:** Metro Interstate Traffic Volume (UCI / Kaggle) — 48,204 hourly records, 2012–2018

---

## Directory Layout

```
traffic_project/
├── data/
│   ├── raw/          ← original CSV (not committed to git)
│   └── processed/    ← features.parquet (auto-generated)
├── models/           ← saved .joblib files (auto-generated)
├── outputs/
│   ├── plots/        ← PNG charts (auto-generated)
│   └── reports/      ← metrics.json (auto-generated)
├── src/
│   ├── config.py         — all paths, constants, hyperparameters
│   ├── preprocessing.py  — load → clean → feature engineering
│   ├── splitter.py       — time-based sequential split (no leakage)
│   ├── models.py         — Ridge / RandomForest / XGBoost + congestion classifier
│   ├── evaluation.py     — MAE, RMSE, R² + comparison table
│   ├── plotting.py       — all matplotlib/seaborn charts
│   └── train.py          — end-to-end pipeline entrypoint
└── streamlit_app/
    └── app.py            — interactive prediction UI
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (preprocessing + evaluation + plots run automatically)
python src/train.py

# 3. Launch the Streamlit demo
streamlit run streamlit_app/app.py
```

Outputs land in `outputs/plots/` and `outputs/reports/metrics.json` automatically.

---

## Pipeline Summary

| Step | What happens |
|---|---|
| **Preprocessing** | Parse datetime → resample to hourly grid → interpolate short gaps → ffill categoricals |
| **Feature Engineering** | Calendar (hour, DOW, month, weekend, peak) + Lag (1h, 2h, 24h, 168h) + Rolling stats (3h, 24h mean & std) + Weather one-hot + holiday flag |
| **Split** | First 80% chronologically = train; last 20% = test. No shuffle. |
| **Models** | Ridge (baseline), RandomForest, XGBoost |
| **Evaluation** | MAE, RMSE, R² on held-out test set |
| **Demo** | Streamlit sliders → real-time single prediction + congestion band |

---

## Key Design Decisions

- **No data leakage:** rolling features use `.shift(1)` before `.rolling()`, so no current-hour value is visible to itself.
- **No shuffle in split:** time-series data must be split chronologically .
- **Lag 168h:** same hour last week captures weekly seasonality without a neural network.
- **Congestion classifier** maps continuous predictions to Free Flow / Moderate / Heavy Congestion — directly relevant to urban planning.

---

## Evaluation Metrics (expected ranges)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ridge | ~700 | ~950 | ~0.77 |
| RandomForest | ~200 | ~350 | ~0.97 |
| XGBoost | ~190 | ~330 | ~0.97 |
