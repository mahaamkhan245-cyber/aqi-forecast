import sys
import subprocess
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from utils.aqi import get_aqi_category

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Google Fonts + Theme CSS ──────────────────────────────────────────────────

st.markdown("""
<style>

/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* =====================================================
GENERAL
===================================================== */

.stApp {
    background-color: #c77fc5 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
}

/* =====================================================
HEADINGS  — Playfair Display
===================================================== */

h1 {
    font-family: 'Playfair Display', serif !important;
    color: #2D1B4E !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px;
}

h2 {
    font-family: 'Playfair Display', serif !important;
    color: #3b1f6e !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

h3 {
    font-family: 'Playfair Display', serif !important;
    color: #4a2580 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

h4, h5, h6 {
    font-family: 'Playfair Display', serif !important;
    color: #4a2580 !important;
}

/* =====================================================
BODY TEXT  — DM Sans
===================================================== */

html, body, p, span, label, li, div, small {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    color: #1a0a2e !important;
}

div[data-testid="stMarkdownContainer"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    color: #1a0a2e !important;
}

div[data-testid="stCaptionContainer"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
    color: #3d2060 !important;
}

/* =====================================================
METRIC CARDS
===================================================== */

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 14px 16px;
    border: 1px solid rgba(255,255,255,0.7);
    box-shadow: 0 2px 10px rgba(116,72,184,0.12);
}

div[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #7448b8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

div[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    color: #2D1B4E !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {
    background: #7448b8 !important;
}

section[data-testid="stSidebar"] * {
    color: #f0e6ff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', serif !important;
    color: #ffffff !important;
    font-size: 14px !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2) !important;
}

/* =====================================================
BUTTONS
===================================================== */

.stButton > button {
    background: linear-gradient(90deg, #9b59d0, #c77fc5);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 8px 18px;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #c77fc5, #9b59d0);
}

/* =====================================================
INFO / SUCCESS / WARNING
===================================================== */

div[data-testid="stAlert"] {
    border-radius: 12px;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    color: #1a0a2e !important;
}

/* =====================================================
DATAFRAME
===================================================== */

table {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    color: #1a0a2e !important;
}

thead {
    background: rgba(116,72,184,0.15);
}

/* =====================================================
IMAGES
===================================================== */

img {
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}

/* =====================================================
TABS
===================================================== */

button[role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    color: #2D1B4E !important;
    font-weight: 600;
}

/* =====================================================
EXPANDERS
===================================================== */

details {
    background: rgba(255,255,255,0.5);
    border-radius: 10px;
    padding: 8px;
}

/* =====================================================
FOOTER / HEADER
===================================================== */

footer { visibility: hidden; }
header { visibility: hidden; }

/* =====================================================
SCROLLBAR
===================================================== */

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: #9b59d0; border-radius: 8px; }
::-webkit-scrollbar-track { background: #c77fc5; }

/* =====================================================
SECTION RULE
===================================================== */

.section-rule {
    border: none;
    border-top: 1px solid rgba(116,72,184,0.25);
    margin: 18px 0;
}

/* =====================================================
EYEBROW TEXT
===================================================== */

.eyebrow {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #4a2580 !important;
    margin-bottom: 4px !important;
}

</style>
""", unsafe_allow_html=True)

# ── Load model & data ─────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load("data/models/random_forest_aqi.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("data/processed/final_features.csv")

forecast = pd.read_csv("data/processed/aqi_forecast.csv")
forecast["date"] = pd.to_datetime(forecast["date"])

model      = load_model()
df         = load_dataset()
latest     = df.iloc[-1]
X_latest   = latest.drop(labels=["date", "AQI"])
prediction = model.predict(pd.DataFrame([X_latest]))[0]
category, advice = get_aqi_category(prediction)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1779/1779940.png", width=60)
    st.markdown("### Pearls AQI Predictor")
    st.markdown("---")

    st.markdown("### Project Details")
    st.markdown(f"**City** — Karachi")
    st.markdown(f"**Model** — Random Forest")
    st.markdown(f"**Period** — 2022 – 2024")
    st.markdown(f"**Records** — {len(df)}")
    st.markdown("---")

    st.markdown("### Live Data")

    if st.button("Refresh Live Forecast", use_container_width=True):
        with st.spinner("Fetching latest data..."):
            result = subprocess.run(
                [sys.executable, "refresh_data.py"],
                capture_output=True,
                text=True
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
        "<span>Bachelor of Data Science</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("<p class='eyebrow'>Machine Learning · Air Quality Intelligence</p>", unsafe_allow_html=True)

st.markdown(f"""
<div style="
  background: linear-gradient(135deg,#4a1a8a,#7448b8);
  padding: 24px 32px;
  border-radius: 20px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.18);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
">
  <div>
    <h1 style="font-family:'Playfair Display',serif;color:white;margin:0 0 4px;font-size:24px;">
      🌍 Pearls AQI Predictor
    </h1>
    <p style="font-family:'DM Sans',sans-serif;font-size:12px;color:#e0c8ff;margin:0;">
      Live Air Quality Intelligence Platform &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; SHAP
    </p>
  </div>
  <div style="
    background:rgba(255,255,255,0.12);
    border-radius:14px;
    padding:16px 28px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.2);
    min-width:160px;
  ">
    <p style="font-family:'DM Sans',sans-serif;font-size:10px;color:#d4b8ff;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 4px;">Current AQI</p>
    <p style="font-family:'Playfair Display',serif;font-size:36px;font-weight:800;color:#FFD54F;margin:0;line-height:1;">{prediction:.1f}</p>
    <p style="font-family:'DM Sans',sans-serif;font-size:11px;color:white;margin:4px 0 0;">{category}</p>
    <p style="font-family:'DM Sans',sans-serif;font-size:10px;color:#c4a8e8;margin:2px 0 0;">Karachi, Pakistan</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── AQI Gauge (compact) ───────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)

_g1, _g2, _g3 = st.columns([1, 2, 1])
with _g2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Current AQI", 'font': {'size': 13, 'family': 'DM Sans'}},
        number={'font': {'size': 28, 'family': 'Playfair Display'}},
        gauge={
            'axis': {'range': [0, 200], 'tickfont': {'size': 10}},
            'bar': {'color': '#7448b8', 'thickness': 0.25},
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
        height=200,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#2D1B4E'),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── KPI row ───────────────────────────────────────────────────────────────────

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Records",       f"{len(df):,}")
k2.metric("Features",      f"{X_latest.shape[0]}")
k3.metric("Predicted AQI", f"{prediction:.1f}")
k4.metric("Model",         "Random Forest")

# ── 3-Day Forecast ────────────────────────────────────────────────────────────

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

cols = st.columns(3)
for i in range(3):
    row = forecast.iloc[i]
    cat_f, adv_f = get_aqi_category(row["Predicted_AQI"])
    with cols[i]:
        st.metric(
            row["date"].strftime("%d %b"),
            f"{row['Predicted_AQI']:.1f}"
        )
        st.markdown(
            f"<p style='font-size:11px;color:#4a2580;margin-top:-6px;'>{cat_f}</p>",
            unsafe_allow_html=True
        )

highest = forecast["Predicted_AQI"].max()
_, advice = get_aqi_category(highest)
st.info(f"**🩺 Health Recommendation** — {advice}")

# ── Weather ───────────────────────────────────────────────────────────────────

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Weather Conditions")

w1, w2, w3, w4, w5 = st.columns(5)
w1.metric("Temperature", f"{latest['temperature']:.1f} °C")
w2.metric("Humidity",    f"{latest['humidity']} %")
w3.metric("Wind Speed",  f"{latest['wind_speed']:.1f} km/h")
w4.metric("Rainfall",    f"{latest['rain']:.2f} mm")
w5.metric("Pressure",    f"{latest['pressure']:.1f} hPa")

# ── Pollutants ────────────────────────────────────────────────────────────────

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Pollutant Concentrations")

p1, p2, p3, p4, p5, p6 = st.columns(6)
p1.metric("PM2.5", f"{latest['pm2_5']:.2f}")
p2.metric("PM10",  f"{latest['pm10']:.2f}")
p3.metric("Ozone", f"{latest['ozone']:.2f}")
p4.metric("CO",    f"{latest['carbon_monoxide']:.2f}")
p5.metric("NO₂",   f"{latest['nitrogen_dioxide']:.2f}")
p6.metric("SO₂",   f"{latest['sulphur_dioxide']:.2f}")

# ── Visual Analytics ──────────────────────────────────────────────────────────

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Model Explainability")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("### Feature Importance")
    st.image("data/eda/feature_importance.png", use_container_width=True)
    st.caption("Features ranked by contribution to AQI prediction. Higher bars indicate greater model influence.")

with c2:
    st.markdown("### SHAP Summary")
    st.image("data/eda/shap_summary.png", use_container_width=True)
    st.caption("Red points push AQI higher; blue points reduce it. Width shows frequency of impact.")

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("### SHAP Waterfall — Single Prediction Breakdown")
_wf_gap, _wf_col, _wf_gap2 = st.columns([0.2, 0.6, 0.2])
with _wf_col:
    st.image("data/eda/shap_waterfall.png", use_container_width=True)
    st.caption("Each bar shows one feature's contribution. Positive values increase the prediction; negative values decrease it.")

# ── Historical trend ──────────────────────────────────────────────────────────

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Historical Analysis")

st.markdown("### AQI Trend Over Time")
st.image("data/eda/aqi_trend.png", use_container_width=True)
st.caption("Daily AQI values across the dataset period. Peaks indicate pollution events; seasonal patterns are visible in multi-year spans.")

d1, d2 = st.columns(2, gap="large")
with d1:
    st.markdown("### Distribution")
    st.image("data/eda/aqi_distribution.png", use_container_width=True)

with d2:
    st.markdown("### Outliers")
    st.image("data/eda/aqi_boxplot.png", use_container_width=True)

st.caption("Left: frequency distribution of AQI values. Right: boxplot highlighting extreme pollution events.")

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("### Correlation Heatmap")
st.image("data/eda/correlation_heatmap.png", use_container_width=True)
st.caption("Values near +1 indicate strong positive correlation; near −1 indicate inverse; near 0 indicate no linear relationship.")

# ── Project Summary ───────────────────────────────────────────────────────────

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Project Summary")

s1, s2 = st.columns(2, gap="large")
with s1:
    st.markdown(
        """
        <div style='background:rgba(255,255,255,0.45);backdrop-filter:blur(6px);border-radius:12px;padding:20px;border:1px solid rgba(255,255,255,0.6);'>
        <p style='font-family:DM Sans,sans-serif;font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#7448b8;margin-bottom:10px;'>Model Performance</p>
        <table style='width:100%;border-collapse:collapse;font-size:12px;font-family:DM Sans,sans-serif;'>
          <tr><td style='padding:6px 0;color:#4a2580;font-weight:600;'>MAE</td><td style='text-align:right;color:#2D1B4E;font-weight:700;'>5.55</td></tr>
          <tr><td style='padding:6px 0;color:#4a2580;font-weight:600;'>RMSE</td><td style='text-align:right;color:#2D1B4E;font-weight:700;'>8.50</td></tr>
          <tr><td style='padding:6px 0;color:#4a2580;font-weight:600;'>R² Score</td><td style='text-align:right;color:#7448b8;font-weight:800;font-size:14px;'>0.943</td></tr>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s2:
    st.markdown(
        """
        <div style='background:rgba(255,255,255,0.45);backdrop-filter:blur(6px);border-radius:12px;padding:20px;border:1px solid rgba(255,255,255,0.6);'>
        <p style='font-family:DM Sans,sans-serif;font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#7448b8;margin-bottom:10px;'>Technologies Used</p>
        <div style='display:flex;flex-wrap:wrap;gap:6px;'>
        """ + "".join(
            f"<span style='background:#7448b8;color:white;padding:3px 10px;border-radius:20px;font-size:10px;font-weight:600;font-family:DM Sans,sans-serif;'>{t}</span>"
            for t in ["Python", "Scikit-learn", "Random Forest", "SHAP", "Streamlit", "Pandas", "Matplotlib", "Open-Meteo API"]
        ) + """
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── About ─────────────────────────────────────────────────────────────────────

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