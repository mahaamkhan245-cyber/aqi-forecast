"""
forecast_air_quality.py
Fetches current + forecast air quality from OpenWeather for Defence Phase 7, Karachi.
FIX: Added graceful fallback to cached/historical data when API is unavailable,
     so a network error no longer fails the entire pipeline.
"""
import os
import sys
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.7967   # Defence Phase 7, Karachi
LON = 67.0728   # Defence Phase 7, Karachi

CURRENT_AQ_PATH  = "data/processed/current_air_quality.csv"
FORECAST_AQ_PATH = "data/processed/air_quality_forecast.csv"
FEATURES_PATH    = "data/processed/final_features.csv"

if not API_KEY:
    print("ERROR: OPENWEATHER_API_KEY not set in environment / .env", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────
# FALLBACK HELPERS
# ─────────────────────────────────────────────────────────────────────────

POLLUTANT_COLS = ["pm2_5", "pm10", "carbon_monoxide",
                  "nitrogen_dioxide", "sulphur_dioxide", "ozone"]


def _historical_means() -> dict:
    """Return mean pollutant values from the feature dataset as a fallback."""
    try:
        df = pd.read_csv(FEATURES_PATH)
        return {c: float(df[c].mean()) for c in POLLUTANT_COLS if c in df.columns}
    except Exception:
        # Hard-coded Karachi typical values if even the CSV is missing
        return {
            "pm2_5": 20.0, "pm10": 35.0, "carbon_monoxide": 400.0,
            "nitrogen_dioxide": 15.0, "sulphur_dioxide": 8.0, "ozone": 60.0,
        }


def _save_current_fallback():
    """Write a single-row current_air_quality.csv from historical means."""
    if os.path.exists(CURRENT_AQ_PATH):
        print("⚠️  Using cached current_air_quality.csv (API unavailable)")
        return
    means = _historical_means()
    row = {"datetime": pd.Timestamp.now(), "aqi": 2, **means}
    pd.DataFrame([row]).to_csv(CURRENT_AQ_PATH, index=False)
    print("⚠️  current_air_quality.csv generated from historical means (API unavailable)")


def _save_forecast_fallback():
    """
    Re-use the existing air_quality_forecast.csv if it covers today,
    otherwise build a 4-day forward frame from historical means.
    """
    today = pd.Timestamp.now().normalize()

    # Try reusing an existing file that still covers today
    if os.path.exists(FORECAST_AQ_PATH):
        try:
            existing = pd.read_csv(FORECAST_AQ_PATH, parse_dates=["date"])
            future = existing[existing["date"] >= today]
            if len(future) >= 1:
                print(f"⚠️  Using cached air_quality_forecast.csv "
                      f"({len(future)} days still valid, API unavailable)")
                return
        except Exception:
            pass

    # Build a synthetic 4-day forecast
    means = _historical_means()
    rows = []
    for i in range(4):
        row = {"date": today + pd.Timedelta(days=i), "aqi": 2,
               "last_updated": datetime.now().strftime("%d %b %Y %I:%M %p"),
               **means}
        rows.append(row)
    pd.DataFrame(rows).to_csv(FORECAST_AQ_PATH, index=False)
    print("⚠️  air_quality_forecast.csv generated from historical means (API unavailable)")


# ─────────────────────────────────────────────────────────────────────────
# CURRENT AQI
# ─────────────────────────────────────────────────────────────────────────
current_ok = False
try:
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/air_pollution",
        params={"lat": LAT, "lon": LON, "appid": API_KEY},
        timeout=15,
    )
    if r.status_code == 401:
        print("ERROR: Invalid OpenWeather API key (401).", file=sys.stderr)
        sys.exit(1)
    r.raise_for_status()
    current_rows = []
    for item in r.json()["list"]:
        current_rows.append({
            "datetime":         pd.to_datetime(item["dt"], unit="s"),
            "pm2_5":            item["components"]["pm2_5"],
            "pm10":             item["components"]["pm10"],
            "carbon_monoxide":  item["components"]["co"],
            "nitrogen_dioxide": item["components"]["no2"],
            "sulphur_dioxide":  item["components"]["so2"],
            "ozone":            item["components"]["o3"],
            "aqi":              item["main"]["aqi"],
        })
    pd.DataFrame(current_rows).to_csv(CURRENT_AQ_PATH, index=False)
    print(f"✅ current_air_quality.csv saved ({len(current_rows)} rows)")
    current_ok = True

except Exception as e:
    print(f"⚠️  Could not fetch current air quality: {e}", file=sys.stderr)
    _save_current_fallback()


# ─────────────────────────────────────────────────────────────────────────
# FORECAST AQI
# ─────────────────────────────────────────────────────────────────────────
forecast_ok = False
try:
    r2 = requests.get(
        "https://api.openweathermap.org/data/2.5/air_pollution/forecast",
        params={"lat": LAT, "lon": LON, "appid": API_KEY},
        timeout=15,
    )
    r2.raise_for_status()
    forecast_rows = []
    for item in r2.json()["list"]:
        forecast_rows.append({
            "datetime":         pd.to_datetime(item["dt"], unit="s"),
            "pm2_5":            item["components"]["pm2_5"],
            "pm10":             item["components"]["pm10"],
            "carbon_monoxide":  item["components"]["co"],
            "nitrogen_dioxide": item["components"]["no2"],
            "sulphur_dioxide":  item["components"]["so2"],
            "ozone":            item["components"]["o3"],
            "aqi":              item["main"]["aqi"],
        })

    df_fc = pd.DataFrame(forecast_rows)
    df_fc["date"] = df_fc["datetime"].dt.normalize()

    df_daily = df_fc.groupby("date").agg({
        "pm2_5": "mean", "pm10": "mean", "carbon_monoxide": "mean",
        "nitrogen_dioxide": "mean", "sulphur_dioxide": "mean",
        "ozone": "mean", "aqi": "mean",
    }).reset_index()

    df_daily = df_daily[df_daily["date"] >= pd.Timestamp.now().normalize()].copy()
    df_daily["last_updated"] = datetime.now().strftime("%d %b %Y %I:%M %p")

    if df_daily.empty:
        print("⚠️  No future dates in API response — using fallback.", file=sys.stderr)
        _save_forecast_fallback()
    else:
        df_daily.to_csv(FORECAST_AQ_PATH, index=False)
        print(f"✅ air_quality_forecast.csv saved ({len(df_daily)} future days)")
        print(df_daily[["date", "pm2_5", "pm10", "ozone"]].to_string())
        forecast_ok = True

except Exception as e:
    print(f"⚠️  Could not fetch forecast air quality: {e}", file=sys.stderr)
    _save_forecast_fallback()


# ─────────────────────────────────────────────────────────────────────────
# EXIT — always 0 so the pipeline continues; warn if both failed live
# ─────────────────────────────────────────────────────────────────────────
if not current_ok and not forecast_ok:
    print("⚠️  Air quality data served entirely from fallback "
          "(no live API data retrieved).")
elif not current_ok or not forecast_ok:
    print("⚠️  Partial fallback used for air quality data.")

sys.exit(0)