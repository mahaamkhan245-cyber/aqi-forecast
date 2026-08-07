import os
import sys
import subprocess
import requests
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

from utils.aqi import get_aqi_category
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
# ══════════════════════════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/* ── Base ── */
.stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #2b154f 0%, #1a0635 100%) !important;
    min-height: 100vh;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

[data-testid="stMain"] {
    background: transparent !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 2rem !important;
}

/* ── Typography ── */
h1 {
    font-family: 'Syne', sans-serif !important;
    color: #ffffff !important;
    font-size: 32px !important;
    font-weight: 800;
    letter-spacing: -0.5px;
}

h2 {
    font-family: 'Syne', sans-serif !important;
    color: #f0e6ff !important;
    font-size: 18px !important;
    font-weight: 700;
    letter-spacing: -0.2px;
}

h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #d6b8ff !important;
    font-size: 13px !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

html, body, p, span, label, li, div, small {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
    color: #f1e8ff !important;
}

div[data-testid="stMarkdownContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #f1e8ff !important;
}

div[data-testid="stCaptionContainer"] {
    color: #c4a8f5 !important;
    font-size: 11px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #23103f 0%, #18092d 100%) !important;
    border-right: 1px solid rgba(160,110,255,0.15);
}

/* ── Metric Cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(160,110,255,0.18), rgba(58,20,110,0.35)) !important;
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 18px 20px;
    border: 1px solid rgba(196,168,245,0.22);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform 0.2s, border-color 0.2s;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-color: rgba(196,168,245,0.45);
}

/* Metric text colors */
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] div,
div[data-testid="metric-container"] span {
    color: #f5ecff !important;
}

/* ── Buttons ── */
button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #a855f7, #7448b8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 8px 20px rgba(116,72,184,0.25);
}

button[kind="primary"]:hover, .stButton > button:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
}

/* ── Section Rule ── */
.section-rule {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(196,168,245,0.35), transparent);
    margin: 28px 0;
}

/* ── Streamlit inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div,
[data-testid="stDateInput"] input {
    background: rgba(255,255,255,0.05) !important;
    color: #ffffff !important;
    border: 1px solid rgba(196,168,245,0.2) !important;
    border-radius: 10px !important;
}

/* ── Charts ── */
.js-plotly-plot, .plotly {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════
# Cache Functions
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_registry():
    """Load the model registry once and reuse everywhere it's needed."""
    return pd.read_csv("data/registry/model_registry.csv")


def get_production_model_row(registry_df):
    return registry_df[registry_df["status"] == "Production"].iloc[-1]


@st.cache_resource
def load_model(model_file):
    model_path = os.path.join("data/models", model_file)
    return joblib.load(model_path)


@st.cache_data
def load_dataset():
    return pd.read_csv("data/processed/final_features.csv")


def safe_image(path, caption=None):
    """Display an image if it exists on disk, otherwise show a warning."""
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")
    else:
        st.warning("Image not found")


def pm25_to_aqi(c):
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= c <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo, 1)
    return 500.0


def fetch_live_aqi():
    """Fetch current AQI from OpenWeather API on every load."""
    if not API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": 24.8607, "lon": 67.0011, "appid": API_KEY},
            timeout=5,
        )
        resp.raise_for_status()
        item = resp.json()["list"][0]
        pm2_5 = item["components"]["pm2_5"]
        return {
            "aqi":               pm25_to_aqi(pm2_5),
            "pm2_5":             pm2_5,
            "pm10":              item["components"]["pm10"],
            "ozone":             item["components"]["o3"],
            "carbon_monoxide":   item["components"]["co"],
            "nitrogen_dioxide":  item["components"]["no2"],
            "sulphur_dioxide":   item["components"]["so2"],
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
# Load Data / Model
# ══════════════════════════════════════════════════════════════════════════
REQUIRED_PATHS = {
    "Model Registry": "data/registry/model_registry.csv",
    "Feature dataset": "data/processed/final_features.csv",
    "Forecast dataset": "data/processed/aqi_forecast.csv",
}
_missing_paths = [
    f"{label} ({path})" for label, path in REQUIRED_PATHS.items() if not os.path.exists(path)
]
if _missing_paths:
    st.error("Missing required file(s):\n- " + "\n- ".join(_missing_paths))
    st.stop()

try:
    registry = load_registry()
    production_model = get_production_model_row(registry)
except Exception as e:
    st.error(f"Failed to load model registry: {e}")
    st.stop()

try:
    model = load_model(production_model["model_file"])
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

try:
    df = load_dataset()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

try:
    forecast = pd.read_csv("data/processed/aqi_forecast.csv")
    forecast["date"] = pd.to_datetime(forecast["date"])
except Exception as e:
    st.error(f"Failed to load forecast data: {e}")
    st.stop()

REQUIRED_COLUMNS = [
    "AQI", "date", "temperature", "humidity", "wind_speed", "rain",
    "pressure", "pm2_5", "pm10", "ozone", "carbon_monoxide",
    "nitrogen_dioxide", "sulphur_dioxide",
]
_missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if _missing_cols:
    st.error(f"Dataset is missing required column(s): {', '.join(_missing_cols)}")
    st.stop()

if "Predicted_AQI" not in forecast.columns:
    st.error("Forecast data is missing required column: Predicted_AQI")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# Compute AQI Values
# ══════════════════════════════════════════════════════════════════════════
try:
    weather_forecast = pd.read_csv("data/processed/weather_forecast.csv")
    latest = weather_forecast.iloc[0]
except Exception:
    latest = df.iloc[-1]

_live = fetch_live_aqi()

if _live:
    actual_aqi        = _live["aqi"]
    current_pollution = pd.Series(_live)
else:
    # fallback to CSV if API call fails or no API key is configured
    try:
        _csv = pd.read_csv("data/processed/current_air_quality.csv")
        current_pollution = _csv.iloc[0]
        actual_aqi = pm25_to_aqi(current_pollution["pm2_5"])
    except Exception as e:
        st.error(f"Failed to load live or fallback air quality data: {e}")
        st.stop()

# -----------------------
# Today's prediction
# -----------------------
prediction = forecast.iloc[0]["Predicted_AQI"]

current_category, current_advice = get_aqi_category(actual_aqi)
predicted_category, predicted_advice = get_aqi_category(prediction)

# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1779/1779940.png", width=60)
    st.markdown("### Pearls AQI Predictor")
    st.markdown("---")
    st.markdown("### Project Details")
    st.markdown("**City** — Karachi")
    st.markdown(f"**Model** — Random Forest v{production_model['version']}")
    st.markdown("**Period** — 2022 – 2024")
    st.markdown(f"**Records** — {len(df)}")
    st.markdown("---")
    st.markdown("### Live Data")
    try:
        _aq = pd.read_csv("data/processed/air_quality_forecast.csv")
        _ts = _aq.iloc[0]["last_updated"]
        st.markdown(
            f"<div style='font-size:10px;color:#d4b8ff;margin-bottom:8px;'>"
            f"🕐 Last updated<br><strong style='color:#FFD700;'>{_ts}</strong></div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    if st.button("🔄 Refresh Live Forecast", use_container_width=True):
        with st.spinner("Fetching latest data..."):
            result = subprocess.run(
                [sys.executable, "refresh_data.py"],
                capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("Forecast Updated!")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Unable to refresh forecast.")
    st.markdown("---")
    st.markdown("### Technology Stack")
    for tech in ["Python", "Scikit-Learn", "Random Forest", "SHAP", "Streamlit"]:
        st.markdown(f"· {tech}")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:rgba(255,255,255,0.7);line-height:1.6;'>"
        "Developed by<br>"
        "<strong style='color:#FFD700;font-size:0.85rem;'>Maham Ahmed</strong><br>"
        "<span>Bachelor of Data Science</span></div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="
  background: linear-gradient(135deg, #1a0635 0%, #2d1060 50%, #1a0635 100%);
  padding: 28px 36px;
  border-radius: 20px;
  border: 1px solid rgba(160,110,255,0.25);
  box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  position: relative;
  overflow: hidden;
">
  <!-- glow blob -->
  <div style="position:absolute;top:-40px;left:-40px;width:200px;height:200px;
    background:radial-gradient(circle,rgba(168,85,247,0.18),transparent 70%);pointer-events:none;"></div>

  <div style="flex:1;min-width:0;position:relative;">
    <p style="font-family:'Plus Jakarta Sans',sans-serif;font-size:10px;font-weight:700;
      letter-spacing:0.18em;text-transform:uppercase;color:#7448b8;margin:0 0 6px;">
      LIVE · KARACHI, PAKISTAN
    </p>
    <p style="font-family:'Syne',sans-serif;font-size:52px;font-weight:800;
      color:#ffffff;margin:0;line-height:1;letter-spacing:-1px;">
      {actual_aqi:.0f}
      <span style="font-size:16px;font-weight:400;color:#9b7ed4;letter-spacing:0;margin-left:4px;">AQI</span>
    </p>
    <div style="margin-top:10px;display:inline-flex;align-items:center;gap:8px;
      background:rgba(116,72,184,0.2);border:1px solid rgba(160,110,255,0.3);
      border-radius:20px;padding:4px 12px;">
      <span style="width:7px;height:7px;border-radius:50%;background:#a855f7;
        box-shadow:0 0 8px #a855f7;display:inline-block;"></span>
      <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:11px;
        font-weight:600;color:#d4c5f0;">{current_category}</span>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;align-items:center;gap:16px;">
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(160,110,255,0.2);
      border-radius:16px;padding:20px 32px;text-align:center;min-width:180px;
      box-shadow:inset 0 1px 0 rgba(255,255,255,0.06);">
      <p style="font-family:'Plus Jakarta Sans',sans-serif;font-size:9px;font-weight:700;
        letter-spacing:0.16em;text-transform:uppercase;color:#7448b8;margin:0 0 6px;">
        Tomorrow's Forecast
      </p>
      <p style="font-family:'Syne',sans-serif;font-size:44px;font-weight:800;
        color:#a855f7;margin:0;line-height:1;letter-spacing:-1px;">{prediction:.0f}</p>
      <p style="font-family:'Plus Jakarta Sans',sans-serif;font-size:11px;
        color:#9b7ed4;margin:6px 0 0;">{predicted_category}</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# AQI Gauge
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

_g1, _g2, _g3 = st.columns([1, 2, 1])
with _g2:
    st.markdown(
        "<p style='text-align:center;font-family:DM Sans,sans-serif;font-size:10px;"
        "font-weight:600;text-transform:uppercase;letter-spacing:0.12em;"
        "color:#3eb1f6;margin-bottom:0;'>AQI Gauge</p>",
        unsafe_allow_html=True
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=actual_aqi,
        number={'font': {'size': 26, 'family': 'Playfair Display'}},
        gauge={
            'axis': {'range': [0, 200], 'tickfont': {'size': 9}},
            'bar': {'color': '#7448b8', 'thickness': 0.22},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0,   50],  'color': '#4ade80'},
                {'range': [50,  100], 'color': '#facc15'},
                {'range': [100, 150], 'color': '#fb923c'},
                {'range': [150, 200], 'color': '#f87171'},
            ]
        }
    ))
    fig.update_layout(
        height=190,
        margin=dict(t=10, b=0, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color="#EBE8F1"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# KPI Cards
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)

k1.metric("Current AQI", f"{actual_aqi:.1f}")
k2.metric("Predicted AQI", f"{prediction:.1f}", delta=f"{prediction - actual_aqi:.1f}")
k3.metric("Records", f"{len(df):,}")
k4.metric("Model", "Random Forest")

difference = prediction - actual_aqi

if difference > 0:
    st.warning(f"The model predicts the AQI may increase by {difference:.1f} points.")
elif difference < 0:
    st.success(f"The model predicts the AQI may decrease by {abs(difference):.1f} points.")
else:
    st.info("The predicted AQI matches the current AQI.")

# ══════════════════════════════════════════════════════════════════════════
# Forecast
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="3" y="4" width="18" height="18" rx="3" stroke="#4a2580" stroke-width="2"/>
    <path d="M3 9h18" stroke="#4a2580" stroke-width="2"/>
    <path d="M8 2v3M16 2v3" stroke="#4a2580" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <h2 style="margin:0;font-family:'Playfair Display',serif;color:#2D1B4E;font-size:18px;">3-Day AQI Forecast</h2>
</div>
""", unsafe_allow_html=True)

future_forecast = forecast.iloc[1:4]
cols = st.columns(3)

for i, (_, row) in enumerate(future_forecast.iterrows()):
    cat_f, adv_f = get_aqi_category(row["Predicted_AQI"])
    with cols[i]:
        st.metric(row["date"].strftime("%d %b"), f"{row['Predicted_AQI']:.1f}")
        st.markdown(
            f"<p style='font-size:11px;color:#4a2580;margin-top:-6px;'>{cat_f}</p>",
            unsafe_allow_html=True
        )

# ── Alerts ──
st.markdown("## 🚨 Air Quality Alert")

if actual_aqi <= 50:
    st.success("🟢 Air quality is Good. Safe for outdoor activities.")
elif actual_aqi <= 100:
    st.info("🟡 Air quality is Moderate. Sensitive people should reduce prolonged outdoor exposure.")
elif actual_aqi <= 150:
    st.warning("🟠 Air quality is Unhealthy for Sensitive Groups.")
elif actual_aqi <= 200:
    st.error("🔴 Air quality is Unhealthy. Everyone should limit prolonged outdoor activities.")
elif actual_aqi <= 300:
    st.error("🟣 Air quality is Very Unhealthy. Avoid outdoor activities.")
else:
    st.error("⚫ Hazardous Air Quality! Stay indoors and wear an N95 mask if you must go outside.")

# ══════════════════════════════════════════════════════════════════════════
# Health Recommendation
# ══════════════════════════════════════════════════════════════════════════
st.info(
    f"""
### Health Recommendation

Current AQI : {actual_aqi:.1f}

Category : {current_category}

Advice :

{current_advice}
"""
)

# ══════════════════════════════════════════════════════════════════════════
# Weather
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Weather Conditions")

w1, w2, w3, w4, w5 = st.columns(5)
w1.metric("Temperature", f"{latest['temperature']:.1f} °C")
w2.metric("Humidity",    f"{latest['humidity']:.0f} %")
w3.metric("Wind Speed",  f"{latest['wind_speed']:.1f} km/h")
w4.metric("Rainfall",    f"{latest['rain']:.2f} mm")
w5.metric("Pressure",    f"{latest['pressure']:.1f} hPa")

# ══════════════════════════════════════════════════════════════════════════
# Pollutants
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Pollutant Concentrations")

p1, p2, p3, p4, p5, p6 = st.columns(6)
p1.metric("PM2.5", f"{current_pollution['pm2_5']:.2f}")
p2.metric("PM10", f"{current_pollution['pm10']:.2f}")
p3.metric("Ozone", f"{current_pollution['ozone']:.2f}")
p4.metric("CO", f"{current_pollution['carbon_monoxide']:.2f}")
p5.metric("NO₂", f"{current_pollution['nitrogen_dioxide']:.2f}")
p6.metric("SO₂", f"{current_pollution['sulphur_dioxide']:.2f}")

# ══════════════════════════════════════════════════════════════════════════
# Explainability
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Model Explainability")

# ── Feature Importance from model ──
importances = model.feature_importances_

# Pull feature names directly from the model or fallback to dataset columns
if hasattr(model, "feature_names_in_"):
    feat_names = list(model.feature_names_in_)
else:
    feat_names = [c for c in df.select_dtypes(include="number").columns if c != "AQI"]
    feat_names = feat_names[:len(importances)]

feat_df = pd.DataFrame({
    "Feature": feat_names,
    "Importance": importances
}).sort_values("Importance", ascending=True)

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("### Feature Importance")
    fig_fi = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#c278c0", "#7448b8", "#2D1B4E"],
        template="none",
    )
    fig_fi.update_traces(hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>")
    fig_fi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(116,72,184,0.15)", title=""),
        yaxis=dict(showgrid=False, title=""),
        font=dict(family="DM Sans", color="#E4DFED", size=11),
        height=360,
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("Features ranked by contribution to AQI prediction. Higher bars indicate greater model influence.")

with c2:
    st.markdown("### Pollutant vs AQI Scatter (SHAP proxy)")
    fig_shap = px.scatter(
        df,
        x="pm2_5",
        y="AQI",
        color="temperature",
        color_continuous_scale=["#4a2580", "#9b59d0", "#f97316"],
        opacity=0.6,
        hover_data=["pm10", "humidity", "wind_speed"],
        template="none",
        labels={"pm2_5": "PM2.5 (µg/m³)", "AQI": "AQI"},
    )
    fig_shap.update_traces(
        marker=dict(size=4),
        hovertemplate="PM2.5: %{x:.1f}<br>AQI: %{y:.1f}<br>Temp: %{marker.color:.1f}°C<extra></extra>"
    )
    fig_shap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(116,72,184,0.15)", title="PM2.5 (µg/m³)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(116,72,184,0.15)", title="AQI"),
        font=dict(family="DM Sans", color="#F1F0F5", size=11),
        coloraxis_colorbar=dict(title="Temp °C", thickness=10),
        height=360,
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption("Each point is one day. Color shows temperature — warmer days tend to push AQI higher.")

# ── Feature contribution for a single prediction ──
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("### Feature Contribution — Single Prediction Breakdown")

last_row = df.iloc[-1]
importance_map = dict(zip(feat_names, importances))

CONTRIBUTION_FEATURES = [
    "pm2_5", "pm10", "ozone", "temperature", "humidity", "wind_speed",
    "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "pressure", "rain",
]

# Look up each feature's importance by name (not position) so this stays correct
# regardless of the order feat_names/importances came back in.
contributions = {
    feat: last_row[feat] * importance_map[feat]
    for feat in CONTRIBUTION_FEATURES
    if feat in last_row.index and feat in importance_map
}

contrib_df = pd.DataFrame({
    "Feature": list(contributions.keys()),
    "Contribution": list(contributions.values()),
}).sort_values("Contribution")

_wf_gap, _wf_col, _wf_gap2 = st.columns([0.1, 0.8, 0.1])
with _wf_col:
    fig_wf = px.bar(
        contrib_df,
        x="Contribution",
        y="Feature",
        orientation="h",
        color="Contribution",
        color_continuous_scale=["#0a4e23", "#1f0a71", "#730707"],
        template="none",
    )
    fig_wf.update_traces(hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2f}<extra></extra>")
    fig_wf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(116,72,184,0.15)", title="Weighted Contribution"),
        yaxis=dict(showgrid=False, title=""),
        font=dict(family="DM Sans", color="#F1EEF5", size=11),
        height=320,
    )
    st.plotly_chart(fig_wf, use_container_width=True)
    st.caption("Each bar shows one feature's weighted contribution to the latest AQI prediction.")

# ══════════════════════════════════════════════════════════════════════════
# Historical Analysis
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Historical Analysis")

# ── 1. AQI Trend ──
st.markdown("### AQI Trend Over Time")
df["date"] = pd.to_datetime(df["date"])
fig_trend = px.area(
    df,
    x="date",
    y="AQI",
    color_discrete_sequence=["#7448b8"],
    template="none",
    labels={"date": "Date", "AQI": "AQI"},
)
fig_trend.update_traces(
    fill="tozeroy",
    fillcolor="rgba(116,72,184,0.18)",
    line=dict(color="#7448b8", width=1.5),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>AQI: %{y:.1f}<extra></extra>",
)
fig_trend.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=10, b=10, l=10, r=10),
    xaxis=dict(showgrid=False, title=""),
    yaxis=dict(showgrid=True, gridcolor="rgba(116,72,184,0.15)", title="AQI"),
    font=dict(family="DM Sans", color="#E8E6EE", size=11),
    height=260,
)
st.plotly_chart(fig_trend, use_container_width=True)
st.caption("Daily AQI values across the dataset period. Hover to inspect exact values.")

# ── 2. Distribution + Boxplot ──
d1, d2 = st.columns(2, gap="large")

with d1:
    st.markdown("### Distribution")
    fig_hist = px.histogram(
        df,
        x="AQI",
        nbins=40,
        color_discrete_sequence=["#7448b8"],
        template="none",
        labels={"AQI": "AQI", "count": "Days"},
    )
    fig_hist.update_traces(
        marker_line_color="rgba(255,255,255,0.3)",
        marker_line_width=0.5,
        hovertemplate="AQI %{x:.0f} – %{x:.0f}<br>Days: %{y}<extra></extra>",
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=False, title="AQI"),
        yaxis=dict(showgrid=True, gridcolor="rgba(116,72,184,0.15)", title="Days"),
        font=dict(family="DM Sans", color="#E6E4EA", size=11),
        height=280,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with d2:
    st.markdown("### Outliers")
    fig_box = px.box(
        df,
        y="AQI",
        color_discrete_sequence=["#9b59d0"],
        template="none",
        points="outliers",
    )
    fig_box.update_traces(
        marker=dict(color="#f87171", size=5, opacity=0.7),
        line=dict(color="#7448b8"),
        hovertemplate="AQI: %{y:.1f}<extra></extra>",
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(showgrid=True, gridcolor="rgba(116,72,184,0.15)", title="AQI"),
        font=dict(family="DM Sans", color="#F9F7FD", size=11),
        height=280,
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.caption("Left: frequency distribution of AQI values. Right: boxplot — red dots are extreme pollution events.")

# ── 3. Correlation Heatmap ──
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("### Correlation Heatmap")

corr_cols = ["AQI", "temperature", "humidity", "wind_speed", "rain",
             "pressure", "pm2_5", "pm10", "ozone",
             "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide"]
corr_matrix = df[corr_cols].corr().round(2)

_ch1, _ch2, _ch3 = st.columns([0.05, 0.9, 0.05])
with _ch2:
    fig_hm = px.imshow(
        corr_matrix,
        color_continuous_scale=["#2D1B4E", "#487ab8", "#bf4ebb", "#AEE3FA"],
        zmin=-1, zmax=1,
        text_auto=True,
        aspect="auto",
        template="none",
    )
    fig_hm.update_traces(
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.2f}<extra></extra>",
        textfont=dict(size=9),
    )
    fig_hm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        font=dict(family="DM Sans", color="#F9F6FE", size=10),
        coloraxis_colorbar=dict(title="r", thickness=12),
        height=420,
    )
    st.plotly_chart(fig_hm, use_container_width=True)
st.caption("Hover any cell to see the exact correlation. Values near +1 = strong positive; near −1 = inverse.")

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Project Summary")

mae = production_model["mae"]
rmse = production_model["rmse"]
r2 = production_model["r2"]
algorithm = production_model["algorithm"]
version = production_model["version"]
train_date = production_model["train_date"]

s1, s2 = st.columns(2, gap="large")

with s1:
    st.markdown(
        "<div style='background:rgba(255,255,255,0.45);"
        "backdrop-filter:blur(6px);border-radius:12px;padding:20px;"
        "border:1px solid rgba(255,255,255,0.6);'>"
        "<p style='font-family:DM Sans,sans-serif;font-size:10px;font-weight:700;"
        "letter-spacing:0.12em;text-transform:uppercase;color:#7448b8;margin-bottom:10px;'>"
        "Production Model</p>"
        "<table style='width:100%;border-collapse:collapse;font-size:12px;font-family:DM Sans,sans-serif;'>"
        f"<tr><td style='padding:6px 0;color:#4a2580;font-weight:600;'>Version</td>"
        f"<td style='text-align:right;color:#2D1B4E;font-weight:700;'>v{version}</td></tr>"
        f"<tr><td style='padding:6px 0;color:#4a2580;font-weight:600;'>Algorithm</td>"
        f"<td style='text-align:right;color:#2D1B4E;font-weight:700;'>{algorithm}</td></tr>"
        f"<tr><td style='padding:6px 0;color:#4a2580;font-weight:600;'>MAE</td>"
        f"<td style='text-align:right;color:#2D1B4E;font-weight:700;'>{mae:.2f}</td></tr>"
        f"<tr><td style='padding:6px 0;color:#4a2580;font-weight:600;'>RMSE</td>"
        f"<td style='text-align:right;color:#2D1B4E;font-weight:700;'>{rmse:.2f}</td></tr>"
        f"<tr><td style='padding:6px 0;color:#4a2580;font-weight:600;'>R² Score</td>"
        f"<td style='text-align:right;color:#7448b8;font-weight:800;font-size:14px;'>{r2:.3f}</td></tr>"
        f"<tr><td style='padding:6px 0;color:#4a2580;font-weight:600;'>Trained On</td>"
        f"<td style='text-align:right;color:#2D1B4E;font-weight:700;'>{train_date}</td></tr>"
        "</table></div>",
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        "<div style='background:rgba(255,255,255,0.45);"
        "backdrop-filter:blur(6px);border-radius:12px;padding:20px;"
        "border:1px solid rgba(255,255,255,0.6);'>"
        "<p style='font-family:DM Sans,sans-serif;font-size:10px;font-weight:700;"
        "letter-spacing:0.12em;text-transform:uppercase;color:#7448b8;margin-bottom:10px;'>"
        "Technologies Used</p>"
        "<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
        + "".join(
            f"<span style='background:#7448b8;color:white;padding:6px 12px;"
            f"border-radius:4px;font-size:10px;font-weight:600;"
            f"font-family:DM Sans,sans-serif;'>{tech}</span>"
            for tech in [
                "Python", "Pandas", "Scikit-learn", "Random Forest",
                "Streamlit", "Plotly", "SHAP", "OpenWeather API", "Joblib"
            ]
        )
        + "</div></div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════
# About
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='background:linear-gradient(135deg,#2D1B4E,#7448b8);border-radius:14px;padding:30px;text-align:center;border:1px solid rgba(212,175,55,0.25);'>
        <p style='font-family:DM Sans,sans-serif;color:#D4AF37;font-size:9px;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:10px;'>End-to-End Machine Learning Application</p>
        <h2 style='font-family:Playfair Display,serif;color:white !important;font-size:20px;margin:0 0 12px;'>Pearls AQI Predictor</h2>
        <p style='font-family:DM Sans,sans-serif;color:#E8D5FF;max-width:480px;margin:0 auto 16px;font-size:11px;line-height:1.7;'>
            An end-to-end ML system integrating weather data and pollutant concentrations to forecast Karachi's Air Quality Index,
            with full model explainability via SHAP and an interactive Streamlit interface.
        </p>
        <p style='font-family:DM Sans,sans-serif;color:#C77DFF;font-size:10px;margin:0;'>
            Developed by <strong style='color:#D4AF37;'>Maham Ahmed</strong> · Bachelor of Data Science · 2026
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)