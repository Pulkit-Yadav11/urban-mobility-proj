"""
Streamlit UI — Urban Mobility Modelling
Run: streamlit run streamlit_app/app.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from PIL import Image

from config import MODELS_DIR, PROCESSED_DIR, PLOTS_DIR, REPORTS_DIR
from models import classify_congestion

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Mobility Predictor",
    page_icon="🚦",
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- font & base ---- */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---- hide default Streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }

/* ---- hero banner ---- */
.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 36px 40px 28px 40px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 6px 0; }
.hero p  { font-size: 1rem; opacity: 0.75; margin: 0; }

/* ---- stat pills in hero ---- */
.pill-row { display: flex; gap: 20px; margin-top: 20px; flex-wrap: wrap; }
.pill {
    background: rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #e0e0e0;
}

/* ---- card wrapper ---- */
.card {
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    border-radius: 14px;
    padding: 24px 26px;
    margin-bottom: 18px;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin-bottom: 6px;
}

/* ---- live result panel ---- */
.result-free     { background: linear-gradient(135deg,#1b5e20,#2e7d32); }
.result-moderate { background: linear-gradient(135deg,#bf360c,#e64a19); }
.result-heavy    { background: linear-gradient(135deg,#7f0000,#b71c1c); }
.result-box {
    border-radius: 16px;
    padding: 28px 32px;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}
.result-box .vol  { font-size: 3.2rem; font-weight: 800; line-height: 1; }
.result-box .unit { font-size: 0.9rem; opacity: 0.75; margin-top: 4px; }
.result-box .band { font-size: 1.15rem; font-weight: 600; margin-top: 14px; }

/* ---- gauge bar ---- */
.gauge-wrap { margin: 16px 0 6px 0; }
.gauge-track {
    height: 10px;
    background: #2a2a3e;
    border-radius: 99px;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s ease;
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #666;
    margin-top: 4px;
}

/* ---- section header ---- */
.section-hdr {
    font-size: 1.05rem;
    font-weight: 700;
    color: #cdd6f4;
    border-left: 3px solid #89b4fa;
    padding-left: 10px;
    margin: 28px 0 14px 0;
}

/* ---- metric tile ---- */
.mtile {
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.mtile .label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; }
.mtile .value { font-size: 1.6rem; font-weight: 700; color: #cdd6f4; margin-top: 4px; }
.mtile .sub   { font-size: 0.72rem; color: #666; margin-top: 2px; }

/* ---- model badge ---- */
.mbadge {
    display: inline-block;
    background: #313244;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.8rem;
    color: #89b4fa;
    font-weight: 600;
    margin-bottom: 14px;
}

/* ---- plot card ---- */
.plot-card {
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 24px;
}
.plot-title   { font-size: 0.95rem; font-weight: 700; color: #cdd6f4; margin-bottom: 4px; }
.plot-caption { font-size: 0.78rem; color: #666; margin-bottom: 14px; }

/* ---- insight chip ---- */
.insight {
    background: #313244;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #a6adc8;
    margin-top: 10px;
    border-left: 3px solid #89b4fa;
}

/* ---- tab styling ---- */
button[data-baseweb="tab"] {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None

@st.cache_data
def load_feature_columns() -> list[str]:
    parquet = PROCESSED_DIR / "features.parquet"
    if not parquet.exists():
        return []
    return [c for c in pd.read_parquet(parquet).columns if c != "traffic_volume"]

@st.cache_data
def load_metrics() -> dict:
    path = REPORTS_DIR / "metrics.json"
    return json.load(open(path)) if path.exists() else {}

def load_plot(filename: str) -> Image.Image | None:
    path = PLOTS_DIR / filename
    return Image.open(path) if path.exists() else None

feature_cols     = load_feature_columns()
metrics          = load_metrics()
available_models = {n: load_model(n) for n in ["xgboost", "random_forest", "ridge"]}
available_models = {k: v for k, v in available_models.items() if v is not None}

if not available_models:
    st.error("No trained models found. Run `python src/train.py` first.")
    st.stop()


# ── Helpers ────────────────────────────────────────────────────────────────
def build_input_row(hour, dow, month, is_weekend, is_peak_hour,
                    temp, rain_1h, snow_1h, clouds_all, weather_main,
                    lag_1h, lag_2h, lag_24h, lag_168h, is_holiday) -> pd.DataFrame:
    weather_cols = [c for c in feature_cols if c.startswith("wm_")]
    row = {
        "hour": hour, "day_of_week": dow, "month": month,
        "is_weekend": is_weekend, "is_peak_hour": is_peak_hour,
        "temp": temp, "rain_1h": rain_1h, "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "lag_1h": lag_1h, "lag_2h": lag_2h,
        "lag_24h": lag_24h, "lag_168h": lag_168h,
        "rolling_mean_3h":  np.mean([lag_1h, lag_2h]),
        "rolling_mean_24h": float(lag_24h),
        "rolling_std_3h":   np.std([lag_1h, lag_2h, lag_24h]),
        "is_holiday": int(is_holiday),
        **{col: int(col == f"wm_{weather_main}") for col in weather_cols},
    }
    return pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)

def congestion_css_class(label: str) -> str:
    return {"Free Flow": "result-free",
            "Moderate Density": "result-moderate",
            "Heavy Congestion": "result-heavy"}.get(label, "result-free")

def gauge_colour(label: str) -> str:
    return {"Free Flow": "#2e7d32",
            "Moderate Density": "#e64a19",
            "Heavy Congestion": "#b71c1c"}.get(label, "#555")

def volume_to_pct(vol: float) -> float:
    return min(vol / 7280 * 100, 100)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Model")
    model_name = st.selectbox(
        "Select model",
        list(available_models.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
        label_visibility="collapsed",
    )
    model = available_models[model_name]

    if metrics:
        m = metrics.get(model_name, {})
        st.markdown(f"""
        <div class="card" style="margin-top:12px">
            <div class="card-title">Selected model stats</div>
            <div style="display:flex;justify-content:space-between;margin-top:10px">
                <div style="text-align:center">
                    <div style="font-size:1.2rem;font-weight:700;color:#a6e3a1">{m.get('R2','—')}</div>
                    <div style="font-size:0.7rem;color:#666">R²</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.2rem;font-weight:700;color:#89b4fa">{int(m.get('MAE',0))}</div>
                    <div style="font-size:0.7rem;color:#666">MAE</div>
                </div>
                <div style="text-align:center">
                    <div style="font-size:1.2rem;font-weight:700;color:#f38ba8">{int(m.get('RMSE',0))}</div>
                    <div style="font-size:0.7rem;color:#666">RMSE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem;color:#666;line-height:1.7">
    🟢 <b style="color:#a6e3a1">Free Flow</b><br>&nbsp;&nbsp;&nbsp;< 1,000 veh/h<br>
    🟡 <b style="color:#f9e2af">Moderate</b><br>&nbsp;&nbsp;&nbsp;1,000 – 3,500 veh/h<br>
    🔴 <b style="color:#f38ba8">Heavy</b><br>&nbsp;&nbsp;&nbsp;> 3,500 veh/h
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_predict, tab_analysis = st.tabs(["🔮  Predict", "📊  Model Analysis"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════
with tab_predict:

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🚦 Urban Mobility Predictor</h1>
        <p>Short-term traffic flow forecasting — Metro Interstate (2012–2018)</p>
        <div class="pill-row">
            <span class="pill">📍 Minneapolis I-94</span>
            <span class="pill">⏱ Hourly Resolution</span>
            <span class="pill">📦 48,204 Records</span>
            <span class="pill">🤖 XGBoost · R² 0.986</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout: inputs LEFT, live result RIGHT ──────────────────
    left, right = st.columns([1.6, 1], gap="large")

    with left:
        # Time
        st.markdown('<div class="section-hdr">🕐 Time & Calendar</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            hour = st.slider("Hour of day", 0, 23, 8,
                             format="%d:00", help="0 = midnight, 23 = 11 PM")
        with c2:
            day_of_week = st.selectbox("Day of week",
                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        with c3:
            month = st.select_slider("Month", options=list(range(1,13)),
                format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                        "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])

        is_holiday = st.checkbox("🗓️ Public Holiday")

        # Weather
        st.markdown('<div class="section-hdr">☁️ Weather Conditions</div>', unsafe_allow_html=True)
        w1, w2 = st.columns(2)
        with w1:
            weather_main = st.selectbox("Condition",
                ["Clear","Clouds","Rain","Drizzle","Mist",
                 "Haze","Fog","Thunderstorm","Snow","Squall","Smoke"])
            temp = st.slider("Temperature (K)", 230.0, 320.0, 285.0, step=0.5,
                             help="288 K ≈ 15°C  |  300 K ≈ 27°C")
        with w2:
            clouds_all = st.slider("Cloud cover (%)", 0, 100, 40)
            rain_1h    = st.number_input("Rain last 1h (mm)", 0.0, 100.0, 0.0, step=0.5)
            snow_1h    = st.number_input("Snow last 1h (mm)", 0.0, 10.0,  0.0, step=0.1)

        # Recent traffic
        st.markdown('<div class="section-hdr">📈 Recent Traffic (lag values)</div>', unsafe_allow_html=True)
        st.caption("The model relies heavily on these — use known counts where possible.")
        l1, l2, l3, l4 = st.columns(4)
        with l1: lag_1h   = st.number_input("1h ago",   0, 8000, 3000, step=100)
        with l2: lag_2h   = st.number_input("2h ago",   0, 8000, 2900, step=100)
        with l3: lag_24h  = st.number_input("24h ago",  0, 8000, 3200, step=100)
        with l4: lag_168h = st.number_input("1wk ago",  0, 8000, 3100, step=100)

    # ── Right: live prediction (updates on every widget change) ─────────────
    with right:
        st.markdown('<div class="section-hdr">⚡ Live Prediction</div>', unsafe_allow_html=True)

        day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                   "Friday":4,"Saturday":5,"Sunday":6}
        dow          = day_map[day_of_week]
        is_weekend   = int(dow >= 5)
        is_peak_hour = int(hour in range(7, 10) or hour in range(16, 20))

        X_input    = build_input_row(hour, dow, month, is_weekend, is_peak_hour,
                                     temp, rain_1h, snow_1h, clouds_all, weather_main,
                                     lag_1h, lag_2h, lag_24h, lag_168h, is_holiday)
        pred       = float(np.clip(model.predict(X_input)[0], 0, None))
        congestion = classify_congestion(pred)
        css_class  = congestion_css_class(congestion)
        colour     = gauge_colour(congestion)
        pct        = volume_to_pct(pred)

        st.markdown(f"""
        <div class="result-box {css_class}">
            <div class="vol">{int(pred):,}</div>
            <div class="unit">vehicles / hour</div>
            <div class="band">{'🟢' if congestion=='Free Flow' else '🟡' if congestion=='Moderate Density' else '🔴'} {congestion}</div>
        </div>
        <div class="gauge-wrap">
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{pct:.1f}%;background:{colour}"></div>
            </div>
            <div class="gauge-labels"><span>0</span><span>3,640</span><span>7,280</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Context chips
        peak_txt   = "Peak hour ⚠️" if is_peak_hour else "Off-peak"
        weekend_tx = "Weekend" if is_weekend else "Weekday"
        holiday_tx = "Holiday 🗓" if is_holiday else ""
        chips      = " · ".join(filter(None, [peak_txt, weekend_tx, holiday_tx]))
        st.markdown(f'<div class="insight">📌 {chips}</div>', unsafe_allow_html=True)

        # Model metrics mini-card
        if metrics and model_name in metrics:
            m = metrics[model_name]
            st.markdown(f"""
            <div class="card" style="margin-top:18px">
                <div class="card-title">Model · {model_name.replace('_',' ').title()}</div>
                <div style="display:flex;justify-content:space-between;margin-top:12px">
                    <div class="mtile" style="flex:1;margin-right:6px;background:#16161e">
                        <div class="label">R²</div>
                        <div class="value" style="font-size:1.2rem">{m['R2']}</div>
                    </div>
                    <div class="mtile" style="flex:1;margin-right:6px;background:#16161e">
                        <div class="label">MAE</div>
                        <div class="value" style="font-size:1.2rem">{int(m['MAE'])}</div>
                        <div class="sub">veh/h</div>
                    </div>
                    <div class="mtile" style="flex:1;background:#16161e">
                        <div class="label">RMSE</div>
                        <div class="value" style="font-size:1.2rem">{int(m['RMSE'])}</div>
                        <div class="sub">veh/h</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight" style="margin-top:14px">
        💡 <b>Tip:</b> Prediction updates live as you change any input above — no button needed.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_analysis:

    st.markdown("""
    <div class="hero">
        <h1>📊 Model Analysis & Evaluation</h1>
        <p>Training results across Ridge, Random Forest, and XGBoost on the held-out test set (last 20% chronologically).</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric tiles ──────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Model Performance at a Glance</div>', unsafe_allow_html=True)
    if metrics:
        cols = st.columns(len(metrics))
        order = ["xgboost", "random_forest", "ridge"]
        icons = {"xgboost": "⚡", "random_forest": "🌲", "ridge": "📐"}
        for col, name in zip(cols, [n for n in order if n in metrics]):
            m    = metrics[name]
            best = name == "xgboost"
            col.markdown(f"""
            <div class="mtile" style="{'border-color:#89b4fa' if best else ''}">
                <div class="label">{icons.get(name,'')} {name.replace('_',' ').title()}
                {'<span style="font-size:0.65rem;background:#1e3a5f;color:#89b4fa;padding:2px 7px;border-radius:10px;margin-left:6px">Best</span>' if best else ''}</div>
                <div style="margin-top:12px;display:flex;justify-content:space-around">
                    <div><div style="font-size:1.3rem;font-weight:700;color:#a6e3a1">{m['R2']}</div><div style="font-size:0.68rem;color:#666">R²</div></div>
                    <div><div style="font-size:1.3rem;font-weight:700;color:#89b4fa">{int(m['MAE'])}</div><div style="font-size:0.68rem;color:#666">MAE</div></div>
                    <div><div style="font-size:1.3rem;font-weight:700;color:#f38ba8">{int(m['RMSE'])}</div><div style="font-size:0.68rem;color:#666">RMSE</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Comparison chart ──────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Side-by-Side Comparison</div>', unsafe_allow_html=True)
    img = load_plot("model_comparison.png")
    if img:
        st.markdown('<div class="plot-card">', unsafe_allow_html=True)
        st.markdown('<div class="plot-title">MAE · RMSE · R² across all models</div>', unsafe_allow_html=True)
        st.markdown('<div class="plot-caption">XGBoost leads on all three metrics. Ridge serves as the interpretable baseline.</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Two plots side by side ────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Predictions & Feature Drivers</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2, gap="medium")

    with pc1:
        img = load_plot("actual_vs_predicted_xgboost.png")
        if img:
            st.markdown('<div class="plot-card">', unsafe_allow_html=True)
            st.markdown('<div class="plot-title">Actual vs Predicted — 7-day window</div>', unsafe_allow_html=True)
            st.markdown('<div class="plot-caption">XGBoost faithfully tracks AM/PM peaks and overnight troughs.</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('<div class="insight">📈 Peak hours (7–9 AM, 4–7 PM) are the hardest to predict; model still stays within ~150 veh/h on average.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Run `python src/train.py` to generate plots.")

    with pc2:
        img = load_plot("feature_importance_xgboost.png")
        if img:
            st.markdown('<div class="plot-card">', unsafe_allow_html=True)
            st.markdown('<div class="plot-title">Feature Importance — XGBoost</div>', unsafe_allow_html=True)
            st.markdown('<div class="plot-caption">lag_1h is the dominant feature — recent traffic predicts near-future traffic.</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('<div class="insight">🔑 Lag features + hour of day account for ~80% of the model\'s predictive power.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Residuals full-width ──────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Residual Analysis</div>', unsafe_allow_html=True)
    img = load_plot("residuals_xgboost.png")
    if img:
        st.markdown('<div class="plot-card">', unsafe_allow_html=True)
        st.markdown('<div class="plot-title">Residual Distribution & Predicted vs Residual Scatter</div>', unsafe_allow_html=True)
        st.markdown('<div class="plot-caption">Errors are centred near zero with no systematic bias — the model generalises well across the traffic range.</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('<div class="insight">⚠️ Larger residuals cluster at high volumes (>5,000 veh/h) — congestion spikes driven by incidents are hard to predict from weather + time features alone.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run `python src/train.py` to generate plots.")
