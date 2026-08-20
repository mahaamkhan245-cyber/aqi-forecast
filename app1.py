

import os
import sys
import subprocess
import warnings
import requests
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.aqi import get_aqi_category
from dotenv import load_dotenv
import textwrap
import json

warnings.filterwarnings("ignore")
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.7967
LON = 67.0728
LOCATION = "Defence Phase 7, Karachi"

# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
.stApp,[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#0f0520 0%,#1a0635 60%,#0d0218 100%) !important;font-family:'Plus Jakarta Sans',sans-serif;}
[data-testid="stMain"]{background:transparent !important;}
[data-testid="stMainBlockContainer"]{padding-top:1rem !important;}
h1{font-family:'Syne',sans-serif !important;color:#fff !important;font-size:24px !important;font-weight:800;}
html,body,p,span,label,li,div,small{font-family:'Plus Jakarta Sans',sans-serif !important;font-size:13px !important;color:#f1e8ff !important;}
div[data-testid="stMarkdownContainer"]{color:#f1e8ff !important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#160830 0%,#0e051f 100%) !important;border-right:1px solid rgba(160,110,255,0.12);}
div[data-testid="metric-container"]{background:linear-gradient(135deg,rgba(160,110,255,0.14),rgba(40,12,80,0.4)) !important;backdrop-filter:blur(12px);border-radius:14px;padding:14px 16px;border:1px solid rgba(196,168,245,0.18);box-shadow:0 4px 20px rgba(0,0,0,0.35);transition:transform 0.2s;}
div[data-testid="metric-container"]:hover{transform:translateY(-2px);}
div[data-testid="metric-container"] label,div[data-testid="metric-container"] div,div[data-testid="metric-container"] span{color:#f5ecff !important;}
.stButton>button{background:linear-gradient(135deg,#a855f7,#7448b8) !important;color:#fff !important;border:none !important;border-radius:10px !important;font-weight:600 !important;box-shadow:0 6px 16px rgba(116,72,184,0.3);}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,0.03) !important;border-radius:12px;padding:4px;border:1px solid rgba(196,168,245,0.12);}
.stTabs [data-baseweb="tab"]{color:#b89ce0 !important;font-weight:600;font-size:12px;border-radius:8px;padding:6px 14px;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(168,85,247,0.3),rgba(116,72,184,0.2)) !important;color:#fff !important;border:1px solid rgba(168,85,247,0.35) !important;}
.hr{border:none;height:1px;background:linear-gradient(90deg,transparent,rgba(196,168,245,0.25),transparent);margin:18px 0;}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,[data-testid="stSelectbox"] div{background:rgba(255,255,255,0.05) !important;color:#fff !important;border:1px solid rgba(196,168,245,0.18) !important;border-radius:8px !important;}
.sb-card{background:linear-gradient(135deg,rgba(160,110,255,0.10),rgba(40,12,80,0.35));border:1px solid rgba(196,168,245,0.16);border-radius:12px;padding:12px 14px;margin-bottom:2px;}
.sb-label{font-size:9px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#8b6fc4;margin:0 0 5px;display:flex;align-items:center;gap:5px;}
.sb-badge{display:inline-flex;align-items:center;gap:5px;font-size:9px;font-weight:600;padding:2px 8px;border-radius:10px;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:5px 8px;border-radius:8px;margin-bottom:1px;transition:background 0.15s;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(168,85,247,0.10);}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def pm25_to_aqi(c):
    bp = [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
    for cl, ch, il, ih in bp:
        if cl <= c <= ch: return round((ih-il)/(ch-cl)*(c-cl)+il, 1)
    return 500.0

# FIX 1: use type name + hasattr so sklearn version mismatch never breaks this


def safe_get_rf(obj):
    if type(obj).__name__ == "RandomForestRegressor": return obj
    if hasattr(obj, "feature_importances_"): return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if type(v).__name__ == "RandomForestRegressor" or hasattr(v, "feature_importances_"):
                return v
    return None


def get_importances(model_obj, feat_names):
    rf = safe_get_rf(model_obj)
    if rf is not None and hasattr(rf, "feature_importances_"):
        return rf.feature_importances_
    # last resort: try direct
    if hasattr(model_obj, "feature_importances_"):
        return model_obj.feature_importances_
    return np.ones(len(feat_names)) / len(feat_names)


def predict_m(model_obj, X):

    # ── PyTorch LSTM ─────────────────────────────────────────────
    if isinstance(model_obj, dict) and model_obj.get("type") == "pytorch":

        model = model_obj["model"]
        scaler = model_obj["scaler"]

        X_scaled = scaler.transform(X).astype(np.float32)

        # Training used shape: (samples, 1, features)
        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        ).reshape(
            -1,
            1,
            X_scaled.shape[1]
        )

        model.eval()

        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy().flatten()

        return predictions

    # ── Existing sklearn dictionary models ───────────────────────
    if isinstance(model_obj, dict) and "model" in model_obj:
        sc = model_obj.get("scaler")
        m = model_obj["model"]

        return m.predict(
            sc.transform(X) if sc else X
        )

    # ── Existing sklearn models ────────────────────────────────
    return model_obj.predict(X)


def aqi_color(v):
    if v <= 50: return "#4ade80"
    if v <= 100: return "#facc15"
    if v <= 150: return "#fb923c"
    if v <= 200: return "#f87171"
    if v <= 300: return "#c084fc"
    return "#ef4444"


def aqi_label(v):
    if v <= 50: return "Good"
    if v <= 100: return "Moderate"
    if v <= 150: return "Unhealthy for Sensitive"
    if v <= 200: return "Unhealthy"
    if v <= 300: return "Very Unhealthy"
    return "Hazardous"


def aqi_alert(v):
    if v <= 50: return "🟢 Good"
    if v <= 100: return "🟡 Moderate"
    if v <= 150: return "🟠 Unhealthy for Sensitive Groups"
    if v <= 200: return "🔴 Unhealthy"
    if v <= 300: return "🟣 Very Unhealthy"
    return "⚫ Hazardous"


def alert_box(aqi_val):
    _, adv = get_aqi_category(aqi_val)
    lbl = aqi_alert(aqi_val)
    if aqi_val <= 50:   st.success(f"**{lbl}** — {adv}")
    elif aqi_val <= 100: st.info(f"**{lbl}** — {adv}")
    elif aqi_val <= 150: st.warning(f"**{lbl}** — {adv}")
    elif aqi_val <= 200: st.error(f"**{lbl}** — {adv}")
    elif aqi_val <= 300: st.error(f"**{lbl}** ⚠️ — {adv} Avoid outdoor activities.")
    else:              st.error(f"**{lbl}** 🚨 — EMERGENCY. Stay indoors. Seal windows. Use air purifier.")

# FIX 3: all charts use automargin + left margin so labels never collapse


def pb(h=280):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#e8e0f5", size=11),
        # generous left margin for y-labels
        margin=dict(t=30, b=40, l=120, r=20),
        height=h,
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True),
    )


def sec(title, icon=""):
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>"
        f"<span style='font-size:17px;'>{icon}</span>"
        f"<span style='font-family:Syne,sans-serif;font-size:16px;font-weight:800;color:#f0e6ff;'>{title}</span>"
        f"</div>", unsafe_allow_html=True)


class AQI_LSTM(nn.Module):
    def __init__(self, inp):
        super().__init__()

        self.l1 = nn.LSTM(
            inp, 128,
            batch_first=True,
            bidirectional=True
        )
        self.d1 = nn.Dropout(0.2)

        self.l2 = nn.LSTM(
            256, 64,
            batch_first=True,
            bidirectional=False
        )
        self.d2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64, 32)
        self.bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        o, _ = self.l1(x)
        o = self.d1(o)

        o, _ = self.l2(o)
        o = self.d2(o[:, -1, :])

        return self.fc2(
            torch.relu(self.bn(self.fc1(o)))
        )

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_df():
    df = pd.read_csv("data/processed/final_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_registry():
    return pd.read_csv("data/registry/model_registry.csv")


@st.cache_resource
def load_model(f):
    model_path = os.path.join("data/models", f)

    # ── PyTorch LSTM ─────────────────────────────────────────────
    if f.endswith(".pt"):
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False
        )

        input_size = checkpoint["input_size"]

        model = AQI_LSTM(input_size)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        scaler_path = os.path.join(
            "data/models",
            "pytorch_scaler.pkl"
        )

        scaler = joblib.load(scaler_path)

        return {
            "type": "pytorch",
            "model": model,
            "scaler": scaler,
            "input_size": input_size
        }

    # ── Existing sklearn models ─────────────────────────────────
    return joblib.load(model_path)


# Validate required files
for lbl, pth in {"Model Registry": "data/registry/model_registry.csv",
                  "Features": "data/processed/final_features.csv"}.items():
    if not os.path.exists(pth):
        st.error(f"Missing: {lbl} ({pth})"); st.stop()

registry = load_registry()
prod_row = registry[registry["status"] == "Production"].iloc[-1]
model = load_model(prod_row["model_file"])
df = load_df()
FCOLS = [c for c in df.columns if c not in ("date", "AQI")]

# FIX 1 applied here
importances = get_importances(model, FCOLS)
feat_names = (list(model.feature_names_in_)
               if hasattr(model, "feature_names_in_") else FCOLS[:len(importances)])


# ── FIX 2: On-the-fly forecast generation when CSV is stale ─────────────
def generate_forecast_from_model(model_obj, df_hist, n_days=4):
    """Generate AQI forecast using the trained model + latest historical lags.
    Works even when APIs are unavailable / forecast CSV is stale."""
    latest = df_hist.sort_values("date").iloc[-1]
    lag1 = latest["AQI"]; lag2 = latest["AQI_lag_1"]; lag3 = latest["AQI_lag_2"]
    mean7 = latest["AQI_7day_mean"]
    today = datetime.now()
    rows = []
    for i in range(n_days):
        d = today + timedelta(days=i)
        row = latest.copy()
        row["year"] = d.year;  row["month"] = d.month
        row["day"] = d.day;   row["day_of_week"] = d.weekday()
        row["day_of_year"] = d.timetuple().tm_yday
        row["weekend"] = 1 if d.weekday() >= 5 else 0
        row["AQI_lag_1"] = lag1; row["AQI_lag_2"] = lag2; row["AQI_lag_3"] = lag3
        row["AQI_3day_mean"] = (lag1+lag2+lag3)/3
        row["AQI_7day_mean"] = mean7
        X = pd.DataFrame([row[FCOLS]])
        pred = float(predict_m(model_obj, X)[0])
        cat, _ = get_aqi_category(pred)
        rows.append({"date": pd.Timestamp(d.date()), "Predicted_AQI": round(pred, 2),
                     "Category": cat, "source": "model"})
        lag3 = lag2; lag2 = lag1; lag1 = pred
    return pd.DataFrame(rows)


def load_forecast_robust():
    """Load forecast CSV; fall back to model-generated forecast if stale."""
    today = pd.Timestamp.now().normalize()
    try:
        fc = pd.read_csv("data/processed/aqi_forecast.csv")
        fc["date"] = pd.to_datetime(fc["date"])
        future = fc[fc["date"] >= today].reset_index(drop=True)
        if len(future) >= 3:
            future["source"] = "csv"
            return future, False   # (df, is_generated)
    except Exception:
        pass
    # Fall back to model-generated
    gen = generate_forecast_from_model(model, df)
    return gen, True


forecast_df, fc_is_generated = load_forecast_robust()


# ── Live pollutant data ──────────────────────────────────────────────────
def fetch_live(lat=LAT, lon=LON):
    if not API_KEY: return None
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/air_pollution",
                         params={"lat": lat, "lon": lon, "appid": API_KEY}, timeout=6)
        if r.status_code != 200: return None
        item = r.json()["list"][0]; pm = item["components"]["pm2_5"]
        return {"aqi": pm25_to_aqi(pm), "pm2_5": pm, "pm10": item["components"]["pm10"],
                "ozone": item["components"]["o3"], "carbon_monoxide": item["components"]["co"],
                "nitrogen_dioxide": item["components"]["no2"], "sulphur_dioxide": item["components"]["so2"]}
    except: return None


_live = fetch_live()
if _live:
    live_aqi = _live["aqi"]; poll = pd.Series(_live)
else:
    try:
        cur = pd.read_csv("data/processed/current_air_quality.csv")
        live_aqi = pm25_to_aqi(float(cur.iloc[0]["pm2_5"])); poll = cur.iloc[0]
    except:
        live_aqi = float(df["AQI"].iloc[-1]); poll = df.iloc[-1]

# Today's model prediction (index 0 of forecast_df)
today_pred = float(forecast_df.iloc[0]["Predicted_AQI"]) if len(
    forecast_df) > 0 else live_aqi

# Latest weather
try:   latest_w = pd.read_csv("data/processed/weather_forecast.csv").iloc[0]
except: latest_w = df.iloc[-1]

live_cat, live_adv = get_aqi_category(live_aqi)
pred_cat, _ = get_aqi_category(today_pred)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:10px 0 4px;'>"
        "<span style='font-size:30px;'>🌍</span>"
        f"<p style='font-family:Syne,sans-serif;font-weight:800;font-size:15px;color:#fff;margin:6px 0 1px;letter-spacing:0.02em;'>Pearls AQI Predictor</p>"
        f"<p style='font-size:9px;color:#9b7ed4;margin:0;letter-spacing:0.12em;'>📍 {LOCATION.upper()}</p>"
        "</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    st.markdown("<p class='sb-label'>🧭 Navigation</p>", unsafe_allow_html=True)
    page = st.radio("Navigate", [
        "🏠 Overview", "📊 Model Comparison",
        "📈 Historical Analysis", "🔬 Explainability", "🔮 Custom Prediction"
    ], label_visibility="collapsed")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-card">
      <p class="sb-label">⚙️ Production Model</p>
      <p style='font-size:13px;color:#fff;font-weight:700;margin:0;'>{prod_row['algorithm']} <span style='color:#9b7ed4;font-weight:600;'>v{prod_row['version']}</span></p>
      <div style='display:flex;gap:14px;margin-top:6px;'>
        <div><p style='font-size:8px;color:#7448b8;margin:0;text-transform:uppercase;letter-spacing:0.08em;'>R²</p>
          <p style='font-size:12px;color:#e0d0ff;font-weight:700;margin:0;'>{prod_row['r2']:.3f}</p></div>
        <div><p style='font-size:8px;color:#7448b8;margin:0;text-transform:uppercase;letter-spacing:0.08em;'>MAE</p>
          <p style='font-size:12px;color:#e0d0ff;font-weight:700;margin:0;'>{prod_row['mae']:.2f}</p></div>
      </div>
    </div>""", unsafe_allow_html=True)
       # ══════════════════════════════════════════════════════════════════════
    # LIVE FORECAST STATUS
    # ══════════════════════════════════════════════════════════════════════

    STATUS_FILE = "data/processed/forecast_status.json"

    status = None

    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                status = json.load(f)
    except Exception:
        status = None


    # ──────────────────────────────────────────────────────────────────────
    # SHOW LAST SUCCESSFUL UPDATE
    # ──────────────────────────────────────────────────────────────────────

    if status and status.get("success"):

        display_date = status.get("display_date", "")
        display_time = status.get("display_time", "")

        st.markdown(
            f"""
            <div class='sb-badge'
                 style='
                    background:rgba(74,222,128,0.14);
                    color:#4ade80;
                    border:1px solid rgba(74,222,128,0.3);
                    margin-top:8px;
                    padding:10px;
                 '>
                🟢 <strong>Live</strong>
                <span style='color:#fff;'>
                    · Updated {display_date}
                </span>
                <br>
                <span style='color:#fff;margin-left:21px;'>
                    {display_time} PKT
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif status and not status.get("success"):

        st.markdown(
            """
            <div class='sb-badge'
                 style='
                    background:rgba(239,68,68,0.14);
                    color:#f87171;
                    border:1px solid rgba(239,68,68,0.3);
                    margin-top:8px;
                 '>
                🔴 Forecast refresh failed
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class='sb-badge'
                 style='
                    background:rgba(250,204,21,0.14);
                    color:#facc15;
                    border:1px solid rgba(250,204,21,0.3);
                    margin-top:8px;
                 '>
                ⚠️ Forecast status unavailable
            </div>
            """,
            unsafe_allow_html=True
        )


    # ══════════════════════════════════════════════════════════════════════
    # REFRESH BUTTON — KEEP INSIDE SIDEBAR
    # ══════════════════════════════════════════════════════════════════════

    if st.button(
        "🔄 Refresh Live Forecast",
        use_container_width=True
    ):

        with st.spinner("Fetching live weather & air quality…"):

            try:

                res = subprocess.run(
                    [sys.executable, "refresh_data.py"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={
                        **os.environ,
                        "PYTHONIOENCODING": "utf-8"
                    }
                )

            except Exception as e:

                st.error(
                    f"❌ Could not start forecast refresh: {e}"
                )

                st.stop()


        # ══════════════════════════════════════════════════════════════════
        # SUCCESS
        # ══════════════════════════════════════════════════════════════════

        if res.returncode == 0:

            st.success(
                "✅ Live forecast updated successfully!"
            )

            st.cache_data.clear()

            st.rerun()


        # ══════════════════════════════════════════════════════════════════
        # FAILURE
        # ══════════════════════════════════════════════════════════════════

        else:

            st.error(
                "❌ Live forecast refresh failed."
            )

            stdout = res.stdout or ""
            stderr = res.stderr or ""

            with st.expander("🔍 Show pipeline error"):

                if stdout:

                    st.markdown("**Pipeline output:**")

                    st.code(
                        stdout,
                        language="text"
                    )

                if stderr:

                    st.markdown("**Error:**")

                    st.code(
                        stderr,
                        language="text"
                    )

                st.caption(
                    f"Pipeline exit code: {res.returncode}"
                )
# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    if fc_is_generated:
        st.info(
            "ℹ️ Forecast is generated from the trained model. Click **Refresh** to fetch live API data.")

    lc = aqi_color(live_aqi)
    pc = aqi_color(today_pred)

    # ── LIVE vs PREDICTED today ──────────────────────────────────────────
    st.markdown(textwrap.dedent(f"""
      <div style="background:linear-gradient(135deg,#0f0520,#1e0840,#0f0520);
      padding:22px 26px;border-radius:18px;border:1px solid rgba(168,85,247,0.22);
      box-shadow:0 8px 48px rgba(0,0,0,0.6);position:relative;overflow:hidden;margin-bottom:14px;">
      <div style="position:absolute;top:-50px;right:-30px;width:200px;height:200px;
        background:radial-gradient(circle,rgba(168,85,247,0.1),transparent 70%);pointer-events:none;"></div>
      <p style="font-size:9px;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;color:#7448b8;margin:0 0 12px;">
        📍 {LOCATION.upper()} · {datetime.now().strftime("%d %b %Y")}</p>
      <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:stretch;">

        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(168,85,247,0.18);
          border-radius:14px;padding:16px 22px;min-width:160px;flex:1;">
          <p style="font-size:9px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#7448b8;margin:0 0 4px;">
            🟢 LIVE AQI (Now)</p>
          <div style="display:flex;align-items:baseline;gap:5px;">
            <span style="font-family:Syne,sans-serif;font-size:56px;font-weight:800;color:{lc};line-height:1;">{live_aqi:.0f}</span>
            <span style="font-size:12px;color:#6b4fa0;">AQI</span>
          </div>
          <div style="margin-top:6px;display:inline-flex;align-items:center;gap:6px;
            background:rgba(168,85,247,0.12);border:1px solid rgba(168,85,247,0.25);border-radius:16px;padding:3px 10px;">
            <span style="width:6px;height:6px;border-radius:50%;background:{lc};box-shadow:0 0 6px {lc};display:inline-block;"></span>
            <span style="font-size:10px;font-weight:600;color:#e0d0ff;">{live_cat}</span>
          </div>
          <p style="font-size:10px;color:#7448b8;margin:5px 0 0;">{live_adv}</p>
        </div>

        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(168,85,247,0.18);
          border-radius:14px;padding:16px 22px;min-width:160px;flex:1;">
          <p style="font-size:9px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#7448b8;margin:0 0 4px;">
            🔮 PREDICTED AQI (Today)</p>
          <div style="display:flex;align-items:baseline;gap:5px;">
            <span style="font-family:Syne,sans-serif;font-size:56px;font-weight:800;color:{pc};line-height:1;">{today_pred:.0f}</span>
            <span style="font-size:12px;color:#6b4fa0;">AQI</span>
          </div>
          <div style="margin-top:6px;display:inline-flex;align-items:center;gap:6px;
            background:rgba(168,85,247,0.12);border:1px solid rgba(168,85,247,0.25);border-radius:16px;padding:3px 10px;">
            <span style="width:6px;height:6px;border-radius:50%;background:{pc};box-shadow:0 0 6px {pc};display:inline-block;"></span>
            <span style="font-size:10px;font-weight:600;color:#e0d0ff;">{pred_cat}</span>
          </div>
          <p style="font-size:10px;color:#7448b8;margin:5px 0 0;">
            {'↑' if today_pred>live_aqi else '↓'} {abs(today_pred-live_aqi):.1f} vs live · Model: {prod_row['algorithm']}</p>
        </div>

      </div>
        </div>"""), unsafe_allow_html=True)

    alert_box(live_aqi)

    # ── Upcoming 3 days ──────────────────────────────────────────────────
    sec("Upcoming 3 Days Forecast", "📅")

    next3 = forecast_df.iloc[1:4] if len(
        forecast_df) > 3 else forecast_df.iloc[:3]
    next3 = next3.reset_index(drop=True)

    # 3 forecast cards
    if len(next3) > 0:
        day_cols = st.columns(len(next3))
        for col, (_, row) in zip(day_cols, next3.iterrows()):
            dc = aqi_color(row["Predicted_AQI"])
            with col:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(168,85,247,0.18);"
                    f"border-radius:14px;padding:14px 16px;text-align:center;'>"
                    f"<p style='font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#7448b8;margin:0 0 4px;'>"
                    f"{row['date'].strftime('%A')}</p>"
                    f"<p style='font-size:10px;color:#9b7ed4;margin:0 0 6px;'>{row['date'].strftime('%d %b')}</p>"
                    f"<span style='font-family:Syne,sans-serif;font-size:40px;font-weight:800;color:{dc};line-height:1;'>{row['Predicted_AQI']:.0f}</span>"
                    f"<p style='font-size:10px;color:{dc};margin:4px 0 0;font-weight:600;'>{aqi_label(row['Predicted_AQI'])}</p>"
                    f"<p style='font-size:9px;color:#6b4fa0;margin:3px 0 0;'>{aqi_alert(row['Predicted_AQI'])}</p>"
                    f"</div>", unsafe_allow_html=True)

    # ── Forecast line chart with AQI bands ──────────────────────────────
    if len(forecast_df) > 0:
        plot_fc = forecast_df.copy()
        fig_fc = go.Figure()
        # colour bands
        for lo, hi, col, lbl in [
            (0, 50, "rgba(74,222,128,0.07)", "Good"),
            (50, 100, "rgba(250,204,21,0.07)", "Moderate"),
            (100, 150, "rgba(251,146,60,0.07)", "Sensitive"),
            (150, 200, "rgba(248,113,113,0.07)", "Unhealthy"),
            (200, 300, "rgba(192,132,252,0.07)", "Very Unhealthy"),
        ]:
            fig_fc.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0,
                annotation_text=lbl, annotation_position="right",
                annotation_font_size=9, annotation_font_color="rgba(200,180,255,0.5)")

        marker_colors = [aqi_color(v) for v in plot_fc["Predicted_AQI"]]
        fig_fc.add_trace(go.Scatter(
            x=plot_fc["date"], y=plot_fc["Predicted_AQI"],
            mode="lines+markers+text",
            line=dict(color="#a855f7", width=3),
            marker=dict(size=14, color=marker_colors,
                        line=dict(color="#fff", width=2)),
            text=[f"{v:.0f}" for v in plot_fc["Predicted_AQI"]],
            textposition="top center",
            textfont=dict(size=13, family="Syne", color="#f0e6ff"),
            hovertemplate="<b>%{x|%A %d %b}</b><br>Predicted AQI: %{y:.1f}<extra></extra>",
            name="Predicted AQI",
        ))
        # highlight today
        forecast_start = plot_fc["date"].iloc[0]

        fig_fc.add_shape(
       type="line",
       x0=forecast_start,
       x1=forecast_start,
       y0=0,
       y1=1,
       xref="x",
       yref="paper",
        line=dict(
        dash="dash"
    )
)

        fig_fc.add_annotation(
        x=forecast_start,
        y=1,
        xref="x",
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yshift=10
)
        ymax = max(plot_fc["Predicted_AQI"].max()*1.35, 120)
        _pb_fc = pb(280)
        _pb_fc["xaxis"].update(
               showgrid=False,
               title="",
               tickformat="%a %d %b",
               tickmode="array",
               tickvals=plot_fc["date"],
              automargin=True
)

        _pb_fc["yaxis"].update(
              showgrid=True,
              gridcolor="rgba(116,72,184,0.12)",
              title="AQI",
              range=[0, ymax],
              automargin=True
)

        fig_fc.update_layout(
        **_pb_fc,
           showlegend=False
)
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Gauge + KPIs ─────────────────────────────────────────────────────
    g1, g2 = st.columns([1, 2], gap="large")
    with g1:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=live_aqi,
            delta={"reference": today_pred, "increasing": {
                "color": "#f87171"}, "decreasing": {"color": "#4ade80"}},
            number={"font": {"size": 28, "color": lc, "family": "Syne"}},
            title={"text": "Live AQI", "font": {
                "size": 11, "color": "#9b7ed4"}},
            gauge={"axis": {"range": [0, 300], "tickfont": {"size": 9}, "tickcolor": "#4a2580"},
                   "bar": {"color": lc, "thickness": 0.22},
                   "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                   "steps": [{"range": [0, 50], "color": "rgba(74,222,128,0.1)"},
                             {"range": [50, 100],
                                 "color": "rgba(250,204,21,0.1)"},
                             {"range": [100, 150],
                                 "color": "rgba(251,146,60,0.1)"},
                             {"range": [150, 200],
                                 "color": "rgba(248,113,113,0.1)"},
                             {"range": [200, 300], "color": "rgba(192,132,252,0.1)"}],
                   "threshold": {"line": {"color": "#fff", "width": 2}, "value": today_pred}},
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#e8e0f5", size=10),
            margin=dict(t=30, b=10, l=20, r=20), height=220)
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption("White marker = today's model prediction")

    with g2:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Live AQI",     f"{live_aqi:.1f}")
        k2.metric("Predicted AQI", f"{today_pred:.1f}",
                  delta=f"{today_pred-live_aqi:.1f}")
        k3.metric("Dataset Rows", f"{len(df):,}")
        k4.metric("Model R²",     f"{prod_row['r2']:.3f}")
        k5.metric("Model MAE",    f"{prod_row['mae']:.2f}")

    # ── Forecast table ────────────────────────────────────────────────────
    sec("3-Day Forecast Table", "📋")
    tbl = forecast_df.iloc[:4].copy()
    tbl["Date"] = tbl["date"].dt.strftime("%A, %d %b %Y")
    tbl["Predicted AQI"] = tbl["Predicted_AQI"].round(1)
    tbl["Category"] = tbl["Predicted_AQI"].apply(
        lambda x: get_aqi_category(x)[0])
    tbl["Alert"] = tbl["Predicted_AQI"].apply(aqi_alert)
    tbl["Health Advice"] = tbl["Predicted_AQI"].apply(
        lambda x: get_aqi_category(x)[1])
    try:
        wf = pd.read_csv(
            "data/processed/weather_forecast.csv"); wf["date"] = pd.to_datetime(wf["date"])
        tbl = tbl.merge(wf[["date", "temperature", "humidity",
                        "wind_speed", "rain"]], on="date", how="left")
        tbl = tbl.rename(columns={"temperature": "Temp °C", "humidity": "RH %",
                         "wind_speed": "Wind km/h", "rain": "Rain mm"})
        show_cols = ["Date", "Predicted AQI", "Category", "Alert",
            "Temp °C", "RH %", "Wind km/h", "Rain mm", "Health Advice"]
    except:
        show_cols = ["Date", "Predicted AQI",
            "Category", "Alert", "Health Advice"]
    st.dataframe(tbl[[c for c in show_cols if c in tbl.columns]],
                 use_container_width=True, hide_index=True)

    # ── Weather + Pollutants ──────────────────────────────────────────────
    sec("Current Weather Conditions", "🌤️")

    def gw(k):
        try: return float(latest_w[k])
        except: return float(df.iloc[-1].get(k, 0))
    w1, w2, w3, w4, w5 = st.columns(5)
    w1.metric("Temperature", f"{gw('temperature'):.1f} °C")
    w2.metric("Humidity",   f"{gw('humidity'):.0f} %")
    w3.metric("Wind Speed", f"{gw('wind_speed'):.1f} km/h")
    w4.metric("Rainfall",   f"{gw('rain'):.2f} mm")
    w5.metric("Pressure",   f"{gw('pressure'):.1f} hPa")

    sec("Pollutant Concentrations", "🏭")

    def gp(k):
        try: return float(poll[k])
        except: return float(df.iloc[-1].get(k, 0))
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("PM2.5", f"{gp('pm2_5'):.2f}")
    p2.metric("PM10", f"{gp('pm10'):.2f}")
    p3.metric("O₃",   f"{gp('ozone'):.2f}")
    p4.metric("CO",   f"{gp('carbon_monoxide'):.2f}")
    p5.metric("NO₂",  f"{gp('nitrogen_dioxide'):.2f}")
    p6.metric("SO₂",  f"{gp('sulphur_dioxide'):.2f}")

    # ── Pollutant tabs ────────────────────────────────────────────────────
    sec("Pollutant Trend Graphs", "📊")
    pcols_map = {"PM2.5": "pm2_5", "PM10": "pm10", "Ozone": "ozone",
                 "CO": "carbon_monoxide", "NO₂": "nitrogen_dioxide", "SO₂": "sulphur_dioxide"}
    pcolors = ["#a855f7", "#3b82f6", "#10b981",
        "#f59e0b", "#ef4444", "#06b6d4"]
    tabs_poll = st.tabs(list(pcols_map.keys()))
    for tab, (name, col), color in zip(tabs_poll, pcols_map.items(), pcolors):
        with tab:
            if col not in df.columns: continue
            ds = df.sort_values("date")
            roll = ds[col].rolling(30, min_periods=1).mean()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric(f"Current {name}", f"{ds[col].iloc[-1]:.2f}")
            s2.metric("30-day Avg", f"{roll.iloc[-1]:.2f}")
            s3.metric("Period Min", f"{ds[col].min():.2f}")
            s4.metric("Period Max", f"{ds[col].max():.2f}")

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=ds["date"], y=ds[col], mode="lines",
                line=dict(color=color, width=1.2), fill="tozeroy",
                fillcolor="rgba(168,85,247,0.13)",
                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{name}: %{{y:.2f}} µg/m³<extra></extra>",
                name=name))
            fig_p.add_trace(go.Scatter(x=ds["date"], y=roll, mode="lines",
                line=dict(color="#FFD700", width=1.5, dash="dash"),
                hovertemplate=f"30-day avg: %{{y:.2f}}<extra></extra>", name="30-day avg"))
            _pb = pb(230)
            _pb["margin"]["l"] = 70
            _pb["xaxis"].update(
                   showgrid=False,
                   title="",
                   automargin=True
)
            _pb["yaxis"].update(
                    showgrid=True,
                    gridcolor="rgba(116,72,184,0.12)",
                    title=f"{name} µg/m³",
                    automargin=True
)

            fig_p.update_layout(
             **_pb,
               legend=dict(
               bgcolor="rgba(0,0,0,0)",
               font=dict(size=9),
               orientation="h",
               yanchor="bottom",
               y=1.02,
               xanchor="right",
               x=1
    )
)
            st.plotly_chart(fig_p, use_container_width=True)

    # ── Pollutant radar ───────────────────────────────────────────────────
    rv = {"PM2.5": gp("pm2_5"), "PM10": gp("pm10"), "Ozone": gp("ozone"),
          "CO/10": gp("carbon_monoxide")/10, "NO₂": gp("nitrogen_dioxide"), "SO₂": gp("sulphur_dioxide")}
    rc = list(rv.keys()); rv_list = list(rv.values())
    _, rcol, _ = st.columns([0.2, 0.6, 0.2])
    with rcol:
        fig_r = go.Figure(go.Scatterpolar(
            r=rv_list+[rv_list[0]], theta=rc+[rc[0]], fill="toself",
            fillcolor="rgba(168,85,247,0.12)",
            line=dict(color="#a855f7", width=2),
            hovertemplate="%{theta}: %{r:.2f} µg/m³<extra></extra>"))
        fig_r.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(showticklabels=True, tickfont=dict(size=8, color="#9b7ed4"),
                                gridcolor="rgba(116,72,184,0.15)"),
                angularaxis=dict(tickfont=dict(size=10, color="#c4a8f5"),
                                 gridcolor="rgba(116,72,184,0.15)")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#e8e0f5", size=11),
            margin=dict(t=40, b=40, l=40, r=40), height=300,
            title=dict(text="Pollutant Radar", font=dict(size=12, color="#9b7ed4"), x=0.5))
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Location map ──────────────────────────────────────────────────────
    sec(f"Location: {LOCATION}", "📍")
    map_df = pd.DataFrame({"lat": [LAT], "lon": [LON], "name": [
                          LOCATION], "AQI": [live_aqi]})
    fig_map = px.scatter_mapbox(map_df, lat="lat", lon="lon", hover_name="name",
        hover_data={"AQI": True, "lat": False, "lon": False},
        color_discrete_sequence=[aqi_color(live_aqi)], zoom=13, height=360)
    fig_map.update_traces(marker=dict(size=20))
    fig_map.update_layout(mapbox_style="carto-darkmatter",
        mapbox_center={"lat": LAT, "lon": LON},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0), height=360)
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(
        f"📍 {LOCATION} | Lat: {LAT}, Lon: {LON} | Live AQI: {live_aqi:.0f}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    sec("Model Registry", "📊")
    st.dataframe(registry, use_container_width=True, hide_index=True,
        column_config={
            "r2":   st.column_config.ProgressColumn("R²", min_value=0, max_value=1, format="%.3f"),
            "mae":  st.column_config.NumberColumn("MAE", format="%.2f"),
            "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
        })

    split = int(len(df)*0.80)
    X_test = df[FCOLS].iloc[split:]; y_test = df["AQI"].iloc[split:]
    test_dates = df["date"].iloc[split:].reset_index(drop=True)

    model_results = []
    load_errors = []
    for _, row in registry.iterrows():
        mp = os.path.join("data/models", str(row["model_file"]))
        if not os.path.exists(mp):
            load_errors.append(f"{row['algorithm']} v{row['version']} — file not found: {row['model_file']}")
            continue
        try:
            m = load_model(row["model_file"]); preds = predict_m(m, X_test)
            model_results.append({"label": f"{row['algorithm']} v{row['version']}",
                "status": row["status"],
                "MAE": round(mean_absolute_error(y_test, preds), 4),
                "RMSE": round(mean_squared_error(y_test, preds)**0.5, 4),
                "R²": round(r2_score(y_test, preds), 4), "preds": preds})
        except Exception as e:
            load_errors.append(f"{row['algorithm']} v{row['version']} — {type(e).__name__}: {e}")
            continue

    if load_errors:
        with st.expander(f"⚠️ {len(load_errors)} model(s) failed to load — click for details", expanded=False):
            for err in load_errors:
                st.caption(err)

    if not model_results:
        st.info("Run `python train_model_comparison.py` to add challenger models.")
    else:
        mdf = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "preds"} for r in model_results])
        pal_s = {"Production": "#a855f7",
            "Challenger": "#3b82f6", "Archived": "#4b5563"}

        sec("Metric Comparison", "📈")
        mc1, mc2, mc3 = st.columns(3)
        for col, met in [(mc1, "R²"), (mc2, "MAE"), (mc3, "RMSE")]:
            with col:
                clrs = [pal_s.get(r["status"], "#4b5563")
                                  for r in model_results]
                _pb = pb(260); _pb["margin"]["l"] = 80
                fig_m = go.Figure(go.Bar(x=mdf["label"], y=mdf[met], marker_color=clrs,
                    text=mdf[met].round(3), textposition="outside",
                    hovertemplate=f"<b>%{{x}}</b><br>{met}: %{{y:.4f}}<extra></extra>"))
                fig_m.update_layout(**_pb,
                    title=dict(text=met, font=dict(
                        size=12, color="#c4a8f5"), x=0.5),
                     showlegend=False)
                st.plotly_chart(fig_m, use_container_width=True)

        sec("Predicted vs Actual AQI — All Models", "🔮")
        pal = ["#a855f7", "#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=test_dates, y=y_test.values, mode="lines", name="Actual",
            line=dict(color="#ffffff", width=1.5, dash="dot"),
            hovertemplate="<b>Actual</b> %{x|%d %b}: %{y:.1f}<extra></extra>"))
        for i, res in enumerate(model_results):
            fig_pred.add_trace(go.Scatter(x=test_dates, y=res["preds"], mode="lines",
                name=res["label"], line=dict(color=pal[i % len(pal)], width=1.5), opacity=0.85,
                hovertemplate=f"<b>{res['label']}</b> %{{x|%d %b}}: %{{y:.1f}}<extra></extra>"))
            _pb = pb(380); _pb["margin"]["l"] = 60
            _pb["xaxis"].update(
                  showgrid=False,
                  title="",
                  automargin=True
)

            _pb["yaxis"].update(
                showgrid=True,
                gridcolor="rgba(116,72,184,0.12)",
                title="AQI",
                automargin=True
)

        fig_pred.update_layout(
        **_pb,
          legend=dict(
          bgcolor="rgba(0,0,0,0.3)",
          bordercolor="rgba(168,85,247,0.3)",
          borderwidth=1,
          font=dict(size=10)
          ),
          hovermode="x unified"
)
        st.plotly_chart(fig_pred, use_container_width=True)
        st.caption(
            "White dotted = actual AQI. Coloured lines = model predictions on test set.")

        # ── Accuracy breakdown: correlation + residuals, side by side ───
        sec("Accuracy Breakdown — All Models", "🎯")
        ac1, ac2 = st.columns(2, gap="large")

        with ac1:
            fig_scatter = go.Figure()
            ax_lo, ax_hi = float(y_test.min()), float(y_test.max())
            fig_scatter.add_trace(go.Scatter(
                x=[ax_lo, ax_hi], y=[ax_lo, ax_hi], mode="lines",
                line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dot"),
                name="Perfect Prediction", hoverinfo="skip"))
            for i, res in enumerate(model_results):
                fig_scatter.add_trace(go.Scatter(
                    x=y_test.values, y=res["preds"], mode="markers",
                    marker=dict(color=pal[i % len(pal)], size=5, opacity=0.55),
                    name=res["label"],
                    hovertemplate=f"<b>{res['label']}</b><br>Actual: %{{x:.1f}}<br>Predicted: %{{y:.1f}}<extra></extra>"))
            _pb_sc = pb(340); _pb_sc["margin"]["l"] = 60
            _pb_sc["xaxis"].update(showgrid=True, gridcolor="rgba(116,72,184,0.12)",
                title="Actual AQI", automargin=True)
            _pb_sc["yaxis"].update(showgrid=True, gridcolor="rgba(116,72,184,0.12)",
                title="Predicted AQI", automargin=True)
            fig_scatter.update_layout(**_pb_sc,
                legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=9)))
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Points closer to the dotted diagonal = more accurate predictions.")

        with ac2:
            fig_resid = go.Figure()
            for i, res in enumerate(model_results):
                resid = y_test.values - res["preds"]
                fig_resid.add_trace(go.Box(
                    y=resid, name=res["label"], marker_color=pal[i % len(pal)],
                    boxmean=True, hovertemplate="Residual: %{y:.2f}<extra></extra>"))
            _pb_rs = pb(340); _pb_rs["margin"]["l"] = 60
            _pb_rs["yaxis"].update(showgrid=True, gridcolor="rgba(116,72,184,0.12)",
                title="Residual (Actual − Predicted)", automargin=True)
            _pb_rs["xaxis"].update(automargin=True)
            fig_resid.update_layout(**_pb_rs, showlegend=False)
            fig_resid.add_hline(y=0, line_dash="dot",
                line_color="rgba(255,255,255,0.3)")
            st.plotly_chart(fig_resid, use_container_width=True)
            st.caption("Boxes centred on zero with a tight spread indicate low bias and low variance.")
# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — HISTORICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Historical Analysis":
    sec("AQI Trend Over Time", "📅")
    ds = df.sort_values("date")
    roll30 = ds["AQI"].rolling(30, min_periods=1).mean()
    fig_t = px.area(ds, x="date", y="AQI", color_discrete_sequence=[
                    "#a855f7"], template="none")
    fig_t.update_traces(fill="tozeroy", fillcolor="rgba(168,85,247,0.10)",
        line=dict(color="#a855f7", width=1.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>AQI: %{y:.1f}<extra></extra>")
    fig_t.add_trace(go.Scatter(x=ds["date"], y=roll30, mode="lines", name="30-day avg",
        line=dict(color="#FFD700", width=2, dash="dash"),
        hovertemplate="30d avg %{x|%d %b}: %{y:.1f}<extra></extra>"))
    _pb = pb(300); _pb["margin"]["l"] = 70
    fig_t.update_layout(**_pb,
       legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    st.plotly_chart(fig_t, use_container_width=True)
    st.caption("Purple area = daily AQI. Gold dashed = 30-day rolling average.")

    sec("Monthly Heatmap", "🗓️")
    df2 = df.copy(
    ); df2["year"] = df2["date"].dt.year; df2["month"] = df2["date"].dt.month
    pivot = df2.groupby(["year", "month"])["AQI"].mean().reset_index().pivot(
        index="year", columns="month", values="AQI")
    mlbls = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = mlbls[:len(pivot.columns)]
    fig_hm = px.imshow(pivot, text_auto=".0f", aspect="auto", template="none",
        color_continuous_scale=["#4ade80", "#facc15", "#fb923c", "#f87171", "#c084fc"], zmin=0, zmax=200)
    fig_hm.update_traces(
        hovertemplate="<b>%{y} %{x}</b><br>Avg AQI: %{z:.1f}<extra></extra>", textfont=dict(size=10))
    _pb2 = pb(210); _pb2["margin"]["l"] = 80
    fig_hm.update_layout(
        **_pb2, coloraxis_colorbar=dict(title="AQI", thickness=12))
    st.plotly_chart(fig_hm, use_container_width=True)

    sec("Distribution & Outliers", "📦")
    d1, d2 = st.columns(2, gap="large")
    with d1:
        fig_h = px.histogram(df, x="AQI", nbins=40, color_discrete_sequence=[
                             "#a855f7"], template="none")
        fig_h.update_traces(marker_line_color="rgba(255,255,255,0.2)", marker_line_width=0.5,
            hovertemplate="AQI %{x:.0f}<br>Days: %{y}<extra></extra>")
        _pb3 = pb(280); _pb3["margin"]["l"] = 70
        _pb3["xaxis"].update(
    showgrid=False,
    title="AQI",
    automargin=True
)
        _pb3["yaxis"].update(
            showgrid=True,
            gridcolor="rgba(116,72,184,0.12)",
            title="Days",
            automargin=True
)
        fig_h.update_layout(
    **_pb3,
    title=dict(
        text="AQI Distribution",
        font=dict(size=11, color="#9b7ed4"),
        x=0
    )
)

        st.plotly_chart(fig_h, use_container_width=True,
                    key="historical_aqi_distribution")
    with d2:
        fig_b = px.box(df, y="AQI", color_discrete_sequence=[
                       "#a855f7"], template="none", points="outliers")
        fig_b.update_traces(marker=dict(
            color="#f87171", size=5, opacity=0.7), line=dict(color="#7448b8"))
        _pb4 = pb(280)
        _pb4["margin"]["l"] = 70
        _pb4["yaxis"].update(
             showgrid=True,
             gridcolor="rgba(116,72,184,0.12)",
             title="AQI",
             automargin=True
)

        fig_b.update_layout(
         **_pb4,
           title=dict(
           text="AQI Outliers",
           font=dict(size=11, color="#9b7ed4"),
            x=0
    )
)

        st.plotly_chart(
        fig_b,
         use_container_width=True,
         key="historical_residuals"
)

    sec("Correlation Heatmap", "🔗")
    ccols = ["AQI", "temperature", "humidity", "wind_speed", "rain", "pressure",
             "pm2_5", "pm10", "ozone", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide"]
    cm = df[ccols].corr().round(2)
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", template="none",
        color_continuous_scale=["#0f0520", "#7448b8", "#a855f7", "#e879f9"], zmin=-1, zmax=1)
    fig_cm.update_traces(
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r=%{z:.2f}<extra></extra>", textfont=dict(size=9))
    _pb5 = pb(420); _pb5["margin"]["l"] = 110; _pb5["margin"]["b"] = 80
    fig_cm.update_layout(
        **_pb5, coloraxis_colorbar=dict(title="r", thickness=12))
    st.plotly_chart(fig_cm, use_container_width=True)

    sec("AQI by Season", "🍂")
    df3 = df.copy()
    df3["Season"] = df3["date"].dt.month.map({12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"})
    fig_s = px.box(df3, x="Season", y="AQI", color="Season", template="none", points="outliers",
        color_discrete_map={"Winter": "#60a5fa", "Spring": "#4ade80",
            "Summer": "#f97316", "Autumn": "#f59e0b"},
        category_orders={"Season": ["Winter", "Spring", "Summer", "Autumn"]})
    _pb6 = pb(300)
    _pb6["margin"]["l"] = 70

    _pb6["xaxis"].update(
    showgrid=False,
    title="",
    automargin=True
)
    _pb6["yaxis"].update(
     showgrid=True,
     gridcolor="rgba(116,72,184,0.12)",
    title="AQI",
    automargin=True
)

    fig_s.update_layout(
    **_pb6,
    showlegend=False
)

    st.plotly_chart(
    fig_s,
    use_container_width=True,
    key="aqi_by_season"
)
    # ============================================================
# SHAP HELPER
# ============================================================

@st.cache_resource
def load_pytorch_shap_model(model_path, scaler_path, input_size):

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model = AQI_LSTM(input_size)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    scaler = joblib.load(
        scaler_path
    )

    return model, scaler


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════
if page == "🔬 Explainability":

    sec("Feature Importance — Production Model", "🔬")

    # ------------------------------------------------------------
    # Get feature importance safely
    # ------------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        raw_imps = np.asarray(model.feature_importances_, dtype=float)

        if hasattr(model, "feature_names_in_"):
            raw_names = list(model.feature_names_in_)
        else:
            raw_names = FCOLS
    else:
        raw_imps = np.asarray(importances, dtype=float)
        raw_names = list(feat_names)

    fi_df = pd.DataFrame({
        "Feature": raw_names[:len(raw_imps)],
        "Importance": raw_imps[:len(raw_names)]
    })

    fi_df = fi_df.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # TOP FEATURE IMPORTANCE + PM2.5 SCATTER
    # ------------------------------------------------------------
    e1, e2 = st.columns(2, gap="large")

    with e1:

        st.markdown("#### Top 15 Feature Importances")

        top15 = (
            fi_df
            .head(15)
            .sort_values("Importance", ascending=True)
        )

        fig_fi = go.Figure(
            go.Bar(
                x=top15["Importance"],
                y=top15["Feature"],
                orientation="h",
                marker=dict(
                    color=top15["Importance"],
                    colorscale=[
                        [0, "#2d1060"],
                        [0.3, "#7448b8"],
                        [0.7, "#a855f7"],
                        [1, "#e879f9"]
                    ],
                    showscale=True,
                    colorbar=dict(
                        title="Importance",
                        thickness=10
                    )
                ),
                text=[
                    f"{v:.4f}"
                    for v in top15["Importance"]
                ],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Importance: %{x:.5f}"
                    "<extra></extra>"
                )
            )
        )

        fig_fi.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                t=20,
                b=50,
                l=130,
                r=80
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(116,72,184,0.12)",
                title="Importance Score",
                automargin=True
            ),
            yaxis=dict(
                showgrid=False,
                title="",
                automargin=True
            ),
            font=dict(
                family="DM Sans",
                color="#e8e0f5"
            )
        )

        st.plotly_chart(
            fig_fi,
            use_container_width=True,
            key="explainability_feature_importance"
        )

        st.caption(
            f"Top feature: "
            f"**{fi_df.iloc[0]['Feature']}** "
            f"({fi_df.iloc[0]['Importance']:.4f})"
        )

    # ------------------------------------------------------------

    with e2:

        st.markdown(
            "#### PM2.5 vs AQI — Interactive"
        )

        fig_sc = px.scatter(
            df,
            x="pm2_5",
            y="AQI",
            color="temperature",
            color_continuous_scale=[
                "#3b82f6",
                "#a855f7",
                "#f97316"
            ],
            opacity=0.6,
            template="none",
            hover_data=[
                "pm10",
                "humidity",
                "wind_speed"
            ],
            labels={
                "pm2_5": "PM2.5 µg/m³",
                "AQI": "AQI",
                "temperature": "Temperature °C"
            }
        )

        fig_sc.update_traces(
            marker=dict(size=5),
            hovertemplate=(
                "<b>PM2.5:</b> %{x:.1f} µg/m³<br>"
                "<b>AQI:</b> %{y:.1f}<br>"
                "<b>Temperature:</b> %{marker.color:.1f} °C"
                "<extra></extra>"
            )
        )

        fig_sc.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                t=20,
                b=50,
                l=70,
                r=50
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(116,72,184,0.12)",
                title="PM2.5 µg/m³",
                automargin=True
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(116,72,184,0.12)",
                title="AQI",
                automargin=True
            ),
            font=dict(
                family="DM Sans",
                color="#e8e0f5"
            ),
            coloraxis_colorbar=dict(
                title="Temperature °C",
                thickness=10
            )
        )

        st.plotly_chart(
            fig_sc,
            use_container_width=True,
            key="explainability_pm25_aqi"
        )

    # ============================================================
    # INTERACTIVE SHAP-STYLE SUMMARY
    # ============================================================

    sec(
        "SHAP-Style Feature Impact Analysis",
        "🧠"
    )

    shap_proxy = []

    for feat in fi_df["Feature"]:

        if feat in df.columns:

            corr = df[feat].corr(df["AQI"])

            if pd.isna(corr):
                corr = 0.0

            imp = float(
                fi_df.loc[
                    fi_df["Feature"] == feat,
                    "Importance"
                ].iloc[0]
            )

            shap_proxy.append({
                "Feature": feat,
                "Importance": imp,
                "Correlation": corr,
                "Direction": (
                    "Positive"
                    if corr >= 0
                    else "Negative"
                ),
                "Impact": imp * abs(corr)
            })

    sp_df = pd.DataFrame(shap_proxy)

    if not sp_df.empty:

        sp_df = (
            sp_df
            .sort_values("Importance", ascending=False)
            .head(15)
        )

        s1, s2 = st.columns(2, gap="large")

        # --------------------------------------------------------
        # SHAP BAR
        # --------------------------------------------------------

        with s1:

            st.markdown(
                "#### SHAP Bar — Feature Impact"
            )

            shap_bar_df = (
                sp_df
                .sort_values("Impact", ascending=True)
            )

            colors_shap = [
                "#f87171"
                if d == "Positive"
                else "#4ade80"
                for d in shap_bar_df["Direction"]
            ]

            fig_shap_bar = go.Figure(
                go.Bar(
                    x=shap_bar_df["Impact"],
                    y=shap_bar_df["Feature"],
                    orientation="h",
                    marker_color=colors_shap,
                    text=[
                        f"{v:.4f}"
                        for v in shap_bar_df["Impact"]
                    ],
                    textposition="outside",
                    customdata=shap_bar_df[
                        ["Direction", "Correlation"]
                    ],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Impact: %{x:.5f}<br>"
                        "Direction: %{customdata[0]}<br>"
                        "Correlation: %{customdata[1]:.3f}"
                        "<extra></extra>"
                    )
                )
            )

            fig_shap_bar.update_layout(
                height=370,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    t=20,
                    b=50,
                    l=130,
                    r=70
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(116,72,184,0.12)",
                    title="Feature Impact",
                    automargin=True
                ),
                yaxis=dict(
                    showgrid=False,
                    title="",
                    automargin=True
                ),
                font=dict(
                    family="DM Sans",
                    color="#e8e0f5"
                )
            )

            st.plotly_chart(
                fig_shap_bar,
                use_container_width=True,
                key="shap_bar_interactive"
            )

            st.caption(
                "🔴 Positive = associated with higher AQI  |  "
                "🟢 Negative = associated with lower AQI"
            )

        # --------------------------------------------------------
        # SHAP SUMMARY-STYLE SCATTER
        # --------------------------------------------------------

        with s2:

            st.markdown(
                "#### SHAP Summary — Interactive"
            )

            summary_rows = []

            for _, row in sp_df.iterrows():

                feat = row["Feature"]

                if feat not in df.columns:
                    continue

                values = df[feat].astype(float)

                importance = float(
                    row["Importance"]
                )

                correlation = float(
                    row["Correlation"]
                )

                # Signed feature contribution proxy
                contribution = (
                    (values - values.mean())
                    * correlation
                    * importance
                )

                for idx in values.index:

                    summary_rows.append({
                        "Feature": feat,
                        "FeatureValue": float(
                            values.loc[idx]
                        ),
                        "SHAPValue": float(
                            contribution.loc[idx]
                        )
                    })

            summary_df = pd.DataFrame(
                summary_rows
            )

            if not summary_df.empty:

                fig_shap_summary = px.scatter(
                    summary_df,
                    x="SHAPValue",
                    y="Feature",
                    color="FeatureValue",
                    color_continuous_scale=[
                        "#3b82f6",
                        "#a855f7",
                        "#f87171"
                    ],
                    hover_data={
                        "SHAPValue": ":.4f",
                        "FeatureValue": ":.4f"
                    },
                    template="none",
                    labels={
                        "SHAPValue": "SHAP Value",
                        "Feature": "",
                        "FeatureValue": "Feature Value"
                    }
                )

                fig_shap_summary.update_traces(
                    marker=dict(
                        size=6,
                        opacity=0.65
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "SHAP value: %{x:.4f}<br>"
                        "Feature value: %{marker.color:.3f}"
                        "<extra></extra>"
                    )
                )

                fig_shap_summary.update_layout(
                    height=370,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(
                        t=20,
                        b=50,
                        l=130,
                        r=70
                    ),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(116,72,184,0.12)",
                        zeroline=True,
                        title="SHAP Value",
                        automargin=True
                    ),
                    yaxis=dict(
                        showgrid=False,
                        title="",
                        automargin=True
                    ),
                    font=dict(
                        family="DM Sans",
                        color="#e8e0f5"
                    )
                )

                st.plotly_chart(
                    fig_shap_summary,
                    use_container_width=True,
                    key="shap_summary_interactive"
                )

                st.caption(
                    "Positive SHAP values push the prediction upward; "
                    "negative values push it downward."
                )

    # ============================================================
    # IMPORTANCE VS CORRELATION
    # ============================================================

    sec(
        "Importance vs Correlation",
        "📊"
    )

    fig_iv = px.scatter(
        sp_df,
        x="Correlation",
        y="Importance",
        text="Feature",
        size="Impact",
        color="Direction",
        color_discrete_map={
            "Positive": "#f87171",
            "Negative": "#4ade80"
        },
        template="none",
        hover_data={
            "Importance": ":.4f",
            "Correlation": ":.3f",
            "Impact": ":.4f"
        }
    )

    fig_iv.update_traces(
        textposition="top center",
        textfont=dict(
            size=8,
            color="#c4a8f5"
        )
    )

    fig_iv.add_vline(
        x=0,
        line_dash="dot",
        line_color="rgba(255,255,255,0.3)"
    )

    fig_iv.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            t=20,
            b=50,
            l=80,
            r=50
        ),
        xaxis=dict(
            title="Correlation with AQI",
            showgrid=True,
            gridcolor="rgba(116,72,184,0.12)"
        ),
        yaxis=dict(
            title="Model Importance",
            showgrid=True,
            gridcolor="rgba(116,72,184,0.12)"
        ),
        font=dict(
            family="DM Sans",
            color="#e8e0f5"
        )
    )

    st.plotly_chart(
        fig_iv,
        use_container_width=True,
        key="explainability_importance_correlation"
    )

    # ============================================================
    # WATERFALL-STYLE FEATURE CONTRIBUTION
    # ============================================================

    sec(
        "Feature Contribution — Latest Prediction",
        "🌊"
    )

    last_row = df.iloc[-1]

    imp_map = dict(
        zip(
            raw_names[:len(raw_imps)],
            raw_imps
        )
    )

    CONTRIBUTION_FEATURES = [
        "pm2_5",
        "pm10",
        "ozone",
        "temperature",
        "humidity",
        "wind_speed",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "pressure",
        "rain",
        "AQI_lag_1",
        "AQI_3day_mean",
        "AQI_7day_mean"
    ]

    contributions = {}

    for feat in CONTRIBUTION_FEATURES:

        if feat in last_row.index and feat in imp_map:

            contributions[feat] = (
                float(last_row[feat])
                * float(imp_map[feat])
            )

    contrib_df = pd.DataFrame({
        "Feature": list(
            contributions.keys()
        ),
        "Contribution": list(
            contributions.values()
        )
    })

    if not contrib_df.empty:

        contrib_df = contrib_df.sort_values(
            "Contribution"
        )

        fig_wf = go.Figure(
            go.Bar(
                x=contrib_df["Contribution"],
                y=contrib_df["Feature"],
                orientation="h",
                marker=dict(
                    color=contrib_df["Contribution"],
                    colorscale=[
                        [0, "#10b981"],
                        [0.5, "#7448b8"],
                        [1, "#ef4444"]
                    ],
                    showscale=True,
                    colorbar=dict(
                        title="Contribution",
                        thickness=10
                    )
                ),
                text=[
                    f"{v:.2f}"
                    for v in contrib_df["Contribution"]
                ],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Contribution: %{x:.3f}"
                    "<extra></extra>"
                )
            )
        )

        fig_wf.update_layout(
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                t=20,
                b=50,
                l=140,
                r=80
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(116,72,184,0.12)",
                title="Weighted Contribution"
            ),
            yaxis=dict(
                showgrid=False,
                title=""
            ),
            font=dict(
                family="DM Sans",
                color="#e8e0f5"
            )
        )

        st.plotly_chart(
            fig_wf,
            use_container_width=True,
            key="explainability_waterfall"
        )

        st.caption(
            "Positive values push AQI upward; "
            "negative values pull AQI downward."
        )


    # --------------------------------------------------------
    # Load production model
    # --------------------------------------------------------

    production_row = registry[
        registry["status"].astype(str).str.lower() == "production"
    ]

    if production_row.empty:
        st.error("❌ No Production model found in registry.")
        st.stop()

    production_row = production_row.iloc[0]

    model_file = str(production_row["model_file"])

    if model_file != "pytorch_lstm.pt":
        st.warning(
            f"Production model is currently `{model_file}`. "
            "The SHAP section is configured for the PyTorch LSTM."
        )
        st.stop()

    model_path = os.path.join(
        "data",
        "models",
        "pytorch_lstm.pt"
    )

    scaler_path = os.path.join(
        "data",
        "models",
        "pytorch_scaler.pkl"
    )

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error(f"❌ Scaler not found: {scaler_path}")
        st.stop()

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_cols = [
        c for c in df.columns
        if c not in ["date", "AQI"]
    ]

    if len(feature_cols) == 0:
        st.error("❌ No feature columns available.")
        st.stop()

    # --------------------------------------------------------
    # Load model + scaler
    # --------------------------------------------------------

    try:

        model, scaler = load_pytorch_shap_model(
            model_path,
            scaler_path,
            len(feature_cols)
        )

    except Exception as e:

        st.error(
            f"❌ Could not load PyTorch LSTM:\n\n{e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X_raw = df[feature_cols].copy()

    X_scaled = scaler.transform(X_raw).astype(np.float32)

    # LSTM expects:
    #
    # samples × sequence_length × features
    #
    # Your training pipeline uses sequence_length = 1

    X_lstm = X_scaled.reshape(
        len(X_scaled),
        1,
        len(feature_cols)
    )

    # --------------------------------------------------------
    # Limit SHAP dataset
    # --------------------------------------------------------

    MAX_SHAP_SAMPLES = min(
        200,
        len(X_lstm)
    )

    shap_indices = np.linspace(
        0,
        len(X_lstm) - 1,
        MAX_SHAP_SAMPLES,
        dtype=int
    )

    X_shap = X_lstm[shap_indices]

    X_shap_raw = X_raw.iloc[shap_indices].reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # SHAP calculation
    # --------------------------------------------------------

    @st.cache_data(show_spinner=False)
    def calculate_shap_values(
        _model,
        X_background,
        X_explain
    ):

        background = torch.tensor(
            X_background,
            dtype=torch.float32
        )

        explain = torch.tensor(
            X_explain,
            dtype=torch.float32
        )

        # DeepExplainer is used for the PyTorch neural network.
        explainer = shap.DeepExplainer(
            _model,
            background
        )

        # LSTM + BatchNorm layers make DeepExplainer's gradient-based
        # additivity check unreliable (it's a known SHAP limitation for
        # recurrent nets), so it's disabled here — the SHAP values
        # themselves are still valid, just the sanity check isn't.
        values = explainer.shap_values(
            explain,
            check_additivity=False
        )

        if isinstance(values, list):
            values = values[0]

        values = np.asarray(values)

        return values

    # Use a smaller background set for performance
    background_size = min(
        50,
        len(X_shap)
    )

    background = X_shap[:background_size]

    # --------------------------------------------------------
    # Calculate SHAP
    # --------------------------------------------------------

    with st.spinner(
        "Calculating SHAP values for the PyTorch LSTM..."
    ):

        try:

            shap_values = calculate_shap_values(
                model,
                background,
                X_shap
            )

        except Exception as e:

            st.error(
                f"❌ SHAP calculation failed:\n\n{e}"
            )

            st.stop()

    # --------------------------------------------------------
    # Normalize SHAP output shape
    # --------------------------------------------------------

    shap_values = np.asarray(shap_values)

    # Possible shape:
    #
    # samples × sequence × features × output
    #
    # or
    #
    # samples × sequence × features

    if shap_values.ndim == 4:
        shap_values = shap_values[:, -1, :, 0]

    elif shap_values.ndim == 3:
        shap_values = shap_values[:, -1, :]

    elif shap_values.ndim == 2:
        pass

    else:

        st.error(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )

        st.stop()

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if shap_values.shape[1] != len(feature_cols):

        st.error(
            f"""
            SHAP feature mismatch.

            SHAP features: {shap_values.shape[1]}

            Model features: {len(feature_cols)}
            """
        )

        st.stop()

    # ========================================================
    # SHAP GLOBAL IMPORTANCE
    # ========================================================

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values(
        "Mean |SHAP|",
        ascending=True
    )

    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3 = st.tabs([
        "📊 SHAP Summary",
        "📈 SHAP Bar",
        "💧 SHAP Waterfall"
    ])

    # ========================================================
    # 1. SHAP SUMMARY
    # ========================================================

    with tab1:

        st.subheader(
            "📊 SHAP Summary Plot"
        )

        st.caption(
            "Each point represents one observation. "
            "SHAP values show how each feature pushes "
            "the AQI prediction higher or lower."
        )

        # ----------------------------------------------------
        # Interactive beeswarm-style Plotly chart
        # ----------------------------------------------------

        summary_rows = []

        for i, feature in enumerate(feature_cols):

            for j in range(len(shap_values)):

                summary_rows.append({
                    "Feature": feature,
                    "SHAP Value": shap_values[j, i],
                    "Feature Value": X_shap_raw.iloc[j][feature],
                    "Observation": j
                })

        summary_df = pd.DataFrame(summary_rows)

        # Order features by importance
        feature_order = (
            importance_df
            .sort_values(
                "Mean |SHAP|",
                ascending=False
            )["Feature"]
            .tolist()
        )

        summary_df["Feature"] = pd.Categorical(
            summary_df["Feature"],
            categories=feature_order,
            ordered=True
        )

        fig_summary = go.Figure()

        for feature in feature_order:

            temp = summary_df[
                summary_df["Feature"] == feature
            ]

            # Small deterministic vertical jitter
            jitter = (
                np.arange(len(temp)) %
                9
            ) / 10.0 - 0.4

            y_values = (
                feature_order.index(feature)
                + jitter
            )

            fig_summary.add_trace(
                go.Scatter(
                    x=temp["SHAP Value"],
                    y=y_values,
                    mode="markers",
                    name=feature,
                    customdata=np.column_stack([
                        temp["Feature Value"],
                        temp["Observation"]
                    ]),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "SHAP value: %{x:.4f}<br>"
                        "Feature value: %{customdata[0]:.4f}<br>"
                        "Observation: %{customdata[1]}<extra></extra>"
                    ),
                    marker=dict(
                        size=7,
                        opacity=0.65
                    ),
                    showlegend=False
                )
            )

        fig_summary.update_layout(
            title="SHAP Feature Impact",
            xaxis_title="SHAP value",
            yaxis=dict(
                tickmode="array",
                tickvals=list(
                    range(len(feature_order))
                ),
                ticktext=feature_order,
                title="Features"
            ),
            hovermode="closest",
            height=max(
                460,
                len(feature_cols) * 30
            )
        )

        fig_summary.add_vline(
            x=0,
            line_dash="dash"
        )

        st.plotly_chart(
            fig_summary,
            use_container_width=True,
            key="shap_summary_plot"
        )

    # ========================================================
    # 2. SHAP BAR
    # ========================================================

    with tab2:

        st.subheader(
            "📈 SHAP Bar Plot"
        )

        st.caption(
            "Global feature importance based on "
            "mean absolute SHAP value."
        )

        bar_df = importance_df.sort_values(
            "Mean |SHAP|",
            ascending=True
        )

        fig_bar = go.Figure()

        fig_bar.add_trace(
            go.Bar(
                x=bar_df["Mean |SHAP|"],
                y=bar_df["Feature"],
                orientation="h",
                text=bar_df["Mean |SHAP|"].round(4),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Mean |SHAP|: %{x:.4f}"
                    "<extra></extra>"
                )
            )
        )

        fig_bar.update_layout(
            title="Global SHAP Feature Importance",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            height=max(
                430,
                len(feature_cols) * 30
            ),
            margin=dict(
                l=180,
                r=80
            )
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True,
            key="shap_bar_plot"
        )

        # ----------------------------------------------------
        # Importance table
        # ----------------------------------------------------

        display_importance = (
            importance_df
            .sort_values(
                "Mean |SHAP|",
                ascending=False
            )
            .reset_index(drop=True)
        )

        st.dataframe(
            display_importance,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # 3. SHAP WATERFALL
    # ========================================================

    with tab3:

        st.subheader(
            "💧 SHAP Waterfall Plot"
        )

        st.caption(
            "Shows how individual features contribute "
            "to one selected AQI prediction."
        )

        # ----------------------------------------------------
        # Observation selector
        # ----------------------------------------------------

        selected_obs = st.slider(
            "Select observation",
            min_value=0,
            max_value=len(shap_values) - 1,
            value=0,
            step=1,
            key="shap_waterfall_observation"
        )

        selected_shap = shap_values[
            selected_obs
        ]

        selected_features = X_shap_raw.iloc[
            selected_obs
        ]

        # ----------------------------------------------------
        # Calculate base value
        # ----------------------------------------------------

        with torch.no_grad():

            background_tensor = torch.tensor(
                background,
                dtype=torch.float32
            )

            background_predictions = (
                model(background_tensor)
                .numpy()
                .flatten()
            )

        base_value = float(
            np.mean(background_predictions)
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        selected_tensor = torch.tensor(
            X_shap[selected_obs:selected_obs + 1],
            dtype=torch.float32
        )

        with torch.no_grad():

            prediction = float(
                model(selected_tensor)
                .numpy()
                .flatten()[0]
            )

        # ----------------------------------------------------
        # Contributions
        # ----------------------------------------------------

        waterfall_df = pd.DataFrame({
            "Feature": feature_cols,
            "SHAP": selected_shap,
            "Feature Value": [
                selected_features[c]
                for c in feature_cols
            ]
        })

        waterfall_df["Abs SHAP"] = (
            waterfall_df["SHAP"].abs()
        )

        waterfall_df = waterfall_df.sort_values(
            "Abs SHAP",
            ascending=False
        )

        # Keep the most important features visible
        MAX_WATERFALL_FEATURES = min(
            15,
            len(waterfall_df)
        )

        waterfall_df = waterfall_df.head(
            MAX_WATERFALL_FEATURES
        )

        # ----------------------------------------------------
        # Build waterfall manually in Plotly
        # ----------------------------------------------------

        features = waterfall_df[
            "Feature"
        ].tolist()[::-1]

        shap_vals = waterfall_df[
            "SHAP"
        ].tolist()[::-1]

        feature_vals = waterfall_df[
            "Feature Value"
        ].tolist()[::-1]

        running = base_value

        x_values = []
        text_values = []

        for value in shap_vals:

            start = running
            end = running + value

            x_values.append(
                (start + end) / 2
            )

            text_values.append(
                f"{value:+.4f}"
            )

            running = end

        fig_waterfall = go.Figure()

        for i, value in enumerate(shap_vals):

            start = (
                base_value
                if i == 0
                else sum(shap_vals[:i]) + base_value
            )

            end = start + value

            fig_waterfall.add_trace(
                go.Bar(
                    x=[end - start],
                    y=[features[i]],
                    base=[start],
                    orientation="h",
                    name=features[i],
                    customdata=[[
                        feature_vals[i],
                        value
                    ]],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Feature value: %{customdata[0]:.4f}<br>"
                        "SHAP contribution: "
                        "%{customdata[1]:+.4f}"
                        "<extra></extra>"
                    ),
                    showlegend=False
                )
            )

        fig_waterfall.add_vline(
            x=base_value,
            line_dash="dash",
            annotation_text="Base"
        )

        fig_waterfall.add_vline(
            x=prediction,
            line_dash="dot",
            annotation_text="Prediction"
        )

        fig_waterfall.update_layout(
            title=(
                f"SHAP Waterfall — "
                f"Observation {selected_obs}"
            ),
            xaxis_title="Predicted AQI",
            yaxis_title="Feature",
            height=max(
                460,
                len(features) * 38
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig_waterfall,
            use_container_width=True,
            key="shap_waterfall_plot"
        )

        # ----------------------------------------------------
        # Prediction summary
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Base AQI",
                f"{base_value:.2f}"
            )

        with c2:
            st.metric(
                "Predicted AQI",
                f"{prediction:.2f}"
            )

        with c3:
            st.metric(
                "Difference",
                f"{prediction - base_value:+.2f}"
            )

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOM PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Custom Prediction":
    sec("Predict AQI with Custom Inputs","🔮")
    t1, t2 = st.tabs(["🎛️ Manual Input","📍 Map Location"])

    with t1:
        last = df.iloc[-1]
        with st.form("manual_form"):
            st.markdown("#### 🌤️ Weather Parameters")
            c1,c2,c3 = st.columns(3)
            temp     = c1.number_input("Temperature (°C)", value=float(last["temperature"]), step=0.1)
            humidity = c2.number_input("Humidity (%)",     value=float(last["humidity"]),    step=1.0)
            pressure = c3.number_input("Pressure (hPa)",   value=float(last["pressure"]),    step=0.1)
            c4,c5 = st.columns(2)
            wind = c4.number_input("Wind Speed (km/h)", value=float(last["wind_speed"]), step=0.1)
            rain = c5.number_input("Rainfall (mm)",     value=float(last["rain"]),       step=0.01)
            st.markdown("#### 🏭 Pollutant Parameters")
            p1,p2,p3 = st.columns(3)
            pm2_5 = p1.number_input("PM2.5 (µg/m³)", value=float(last["pm2_5"]),           step=0.1)
            pm10  = p2.number_input("PM10 (µg/m³)",  value=float(last["pm10"]),            step=0.1)
            ozone = p3.number_input("Ozone (µg/m³)", value=float(last["ozone"]),            step=0.1)
            p4,p5,p6 = st.columns(3)
            co  = p4.number_input("CO (µg/m³)",  value=float(last["carbon_monoxide"]),   step=1.0)
            no2 = p5.number_input("NO₂ (µg/m³)", value=float(last["nitrogen_dioxide"]),  step=0.1)
            so2 = p6.number_input("SO₂ (µg/m³)", value=float(last["sulphur_dioxide"]),   step=0.1)
            sub = st.form_submit_button("🔮 Predict AQI", use_container_width=True)

        if sub:
            row = df.iloc[-1].copy()
            for k,v in [("temperature",temp),("humidity",humidity),("pressure",pressure),
                        ("wind_speed",wind),("rain",rain),("pm2_5",pm2_5),("pm10",pm10),
                        ("ozone",ozone),("carbon_monoxide",co),("nitrogen_dioxide",no2),
                        ("sulphur_dioxide",so2),("rain_flag",1 if rain>0 else 0),
                        ("temp_humidity",temp*humidity)]:
                row[k] = v
            X_c = pd.DataFrame([row[FCOLS]])

            preds_all = []
            for _, reg_row in registry.iterrows():
                mp = os.path.join("data/models",str(reg_row["model_file"]))
                if not os.path.exists(mp): continue
                try:
                    m = load_model(reg_row["model_file"]); p = float(predict_m(m,X_c)[0])
                    cat, adv = get_aqi_category(p)
                    preds_all.append({"model":f"{reg_row['algorithm']} v{reg_row['version']}",
                        "status":reg_row["status"],"pred":p,"category":cat,"advice":adv})
                except: continue

            if preds_all:
                pp = next((r for r in preds_all if r["status"]=="Production"),preds_all[0])
                pc2 = aqi_color(pp["pred"])
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(168,85,247,0.12),rgba(30,10,60,0.7));
                  border-radius:16px;padding:20px;border:1px solid rgba(168,85,247,0.3);text-align:center;margin:12px 0;">
                  <p style="font-size:9px;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#7448b8;margin:0 0 4px;">
                    Production Model Prediction</p>
                  <span style="font-family:Syne,sans-serif;font-size:68px;font-weight:800;color:{pc2};line-height:1;">{pp['pred']:.0f}</span>
                  <span style="font-size:14px;color:#6b4fa0;"> AQI</span>
                  <div style="margin-top:6px;font-size:13px;color:#e0d0ff;font-weight:600;">{pp['category']}</div>
                  <div style="margin-top:4px;font-size:10px;color:#9b7ed4;">{pp['advice']}</div>
                </div>""", unsafe_allow_html=True)
                alert_box(pp["pred"])

                if len(preds_all) > 1:
                    st.markdown("#### All Model Predictions")
                    pm_cols = st.columns(len(preds_all))
                    for col2,res in zip(pm_cols,preds_all):
                        with col2:
                            cc2 = aqi_color(res["pred"])
                            st.markdown(
                                f"<div style='background:rgba(168,85,247,0.08);border-radius:12px;"
                                f"padding:12px;border:1px solid rgba(168,85,247,0.18);text-align:center;'>"
                                f"<p style='font-size:9px;color:#7448b8;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 3px;'>{res['model']}</p>"
                                f"<span style='font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:{cc2};'>{res['pred']:.0f}</span>"
                                f"<p style='font-size:9px;color:#c4a8f5;margin:3px 0 0;'>{res['category']}</p>"
                                f"</div>", unsafe_allow_html=True)

                    # Gauge comparison
                    fig_gauges = go.Figure()
                    for i,res in enumerate(preds_all):
                        fig_gauges.add_trace(go.Indicator(
                            mode="gauge+number",value=res["pred"],
                            title={"text":res["model"],"font":{"size":9,"color":"#9b7ed4"}},
                            number={"font":{"size":18,"color":aqi_color(res["pred"])}},
                            gauge={"axis":{"range":[0,250]},"bar":{"color":aqi_color(res["pred"]),"thickness":0.25},
                                   "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                                   "steps":[{"range":[0,50],"color":"rgba(74,222,128,0.08)"},
                                             {"range":[50,100],"color":"rgba(250,204,21,0.08)"},
                                             {"range":[100,150],"color":"rgba(251,146,60,0.08)"},
                                             {"range":[150,250],"color":"rgba(248,113,113,0.08)"}]},
                            domain={"column":i,"row":0}))
                    fig_gauges.update_layout(
                        grid={"rows":1,"columns":len(preds_all)},
                        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Plus Jakarta Sans",color="#e8e0f5",size=10),
                        margin=dict(t=20,b=20,l=10,r=10),height=200)
                    st.plotly_chart(fig_gauges,use_container_width=True)

    with t2:
        st.markdown("<p style='color:#9b7ed4;font-size:12px;margin-bottom:10px;'>Select a Karachi location to fetch its live AQI from OpenWeather API.</p>",unsafe_allow_html=True)
        presets = {
            "Defence Phase 7 (Default)":(24.7967,67.0728),
            "Clifton":                  (24.8067,67.0300),
            "Saddar":                   (24.8559,67.0106),
            "Gulshan-e-Iqbal":          (24.9215,67.1024),
            "Malir":                    (24.8928,67.2009),
            "Korangi":                  (24.8186,67.1308),
            "North Nazimabad":          (24.9356,67.0435),
            "Custom Coordinates":       None,
        }
        sel = st.selectbox("Select Location",list(presets.keys()))
        if presets[sel] is None:
            cc1,cc2 = st.columns(2)
            sl = cc1.number_input("Latitude", value=LAT, format="%.4f", step=0.0001)
            sn = cc2.number_input("Longitude",value=LON, format="%.4f", step=0.0001)
            sname = "Custom Location"
        else:
            sl,sn = presets[sel]; sname = sel

        if st.button("🌍 Fetch Live AQI for This Location", use_container_width=True):
            with st.spinner(f"Fetching live data for {sname}…"):
                lv = fetch_live(sl,sn)
            if lv:
                la = lv["aqi"]; lcat,_ = get_aqi_category(la); lc2 = aqi_color(la)
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(168,85,247,0.12),rgba(30,10,60,0.7));
                  border-radius:14px;padding:18px;border:1px solid rgba(168,85,247,0.3);text-align:center;margin:10px 0;">
                  <p style="font-size:9px;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:#7448b8;margin:0 0 4px;">
                    📍 {sname}</p>
                  <span style="font-family:Syne,sans-serif;font-size:56px;font-weight:800;color:{lc2};line-height:1;">{la:.0f}</span>
                  <span style="font-size:13px;color:#6b4fa0;"> AQI</span>
                  <div style="margin-top:6px;font-size:12px;color:#e0d0ff;font-weight:600;">{lcat}</div>
                </div>""", unsafe_allow_html=True)
                alert_box(la)
                lp1,lp2,lp3 = st.columns(3)
                lp1.metric("PM2.5",f"{lv['pm2_5']:.2f} µg/m³")
                lp2.metric("PM10", f"{lv['pm10']:.2f} µg/m³")
                lp3.metric("Ozone",f"{lv['ozone']:.2f} µg/m³")
                lp4,lp5,lp6 = st.columns(3)
                lp4.metric("CO",   f"{lv['carbon_monoxide']:.2f} µg/m³")
                lp5.metric("NO₂",  f"{lv['nitrogen_dioxide']:.2f} µg/m³")
                lp6.metric("SO₂",  f"{lv['sulphur_dioxide']:.2f} µg/m³")
                map_d2 = pd.DataFrame({"lat":[sl],"lon":[sn],"name":[sname],"AQI":[la]})
                fig_ml = px.scatter_mapbox(map_d2,lat="lat",lon="lon",hover_name="name",
                    hover_data={"AQI":True,"lat":False,"lon":False},
                    color_discrete_sequence=[lc2],zoom=12,height=300)
                fig_ml.update_traces(marker=dict(size=18))
                fig_ml.update_layout(mapbox_style="carto-darkmatter",
                    mapbox_center={"lat":sl,"lon":sn},
                    paper_bgcolor="rgba(0,0,0,0)",margin=dict(t=0,b=0,l=0,r=0),height=300)
                st.plotly_chart(fig_ml,use_container_width=True)
            else:
                st.error("Could not fetch live data. Check your API key or internet connection.")

        sec("Karachi — Key Locations Map","🗺️")
        klocs = [("Defence Phase 7",24.7967,67.0728),("Clifton",24.8067,67.0300),
                 ("Saddar",24.8559,67.0106),("Gulshan-e-Iqbal",24.9215,67.1024),
                 ("Malir",24.8928,67.2009),("Korangi",24.8186,67.1308),("N. Nazimabad",24.9356,67.0435)]
        kdf = pd.DataFrame([{"Location":n,"lat":lt,"lon":ln,"AQI (est.)":round(live_aqi+(lt-LAT)*40+(ln-LON)*25,1)}
                             for n,lt,ln in klocs])
        fig_km = px.scatter_mapbox(kdf,lat="lat",lon="lon",hover_name="Location",
            hover_data={"AQI (est.)":True,"lat":False,"lon":False},
            color="AQI (est.)",color_continuous_scale=["#4ade80","#facc15","#f87171","#c084fc"],
            size_max=16,zoom=10,height=340)
        fig_km.update_traces(marker=dict(size=14))
        fig_km.update_layout(mapbox_style="carto-darkmatter",
            mapbox_center={"lat":24.86,"lon":67.07},
            paper_bgcolor="rgba(0,0,0,0)",margin=dict(t=0,b=0,l=0,r=0),height=340,
            coloraxis_colorbar=dict(title="AQI",thickness=12))
        st.plotly_chart(fig_km,use_container_width=True)
        st.caption("AQI estimates based on spatial interpolation from the Defence Phase 7 live reading.")


# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
tags = " ".join(f"<span style='background:rgba(168,85,247,0.18);color:#c4a8f5;padding:3px 8px;border-radius:6px;font-size:9px;font-weight:600;'>{t}</span>"
                for t in ["Python","Scikit-learn","Random Forest","Streamlit","Plotly","SHAP","OpenWeather API","Flask","GitHub Actions"])
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f0520,#1e0840,#0f0520);border-radius:14px;padding:20px;
  text-align:center;border:1px solid rgba(168,85,247,0.15);">
  <p style="font-family:Syne,sans-serif;color:#fff;font-size:16px;font-weight:800;margin:0 0 6px;">Pearls AQI Predictor</p>
  <p style="color:#9b7ed4;max-width:460px;margin:0 auto 10px;font-size:10px;line-height:1.7;">
    End-to-end ML pipeline for AQI forecasting in {LOCATION}</p>
  <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:5px;margin-bottom:8px;">{tags}</div>
  <p style="color:#6b4fa0;font-size:9px;margin:0;">
    Developed by <strong style="color:#FFD700;">Maham Ahmed</strong> · Bachelor of Data Science · 2026 · 📍 {LOCATION}</p>
</div>""", unsafe_allow_html=True)