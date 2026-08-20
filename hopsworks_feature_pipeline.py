"""
hopsworks_feature_pipeline.py
Runs every hour via GitHub Actions.
Fetches live weather + air quality, engineers features,
and pushes one new row to the Hopsworks Feature Store.

This replaces the manual CSV-based feature store with a proper
cloud feature store — satisfying the project requirement.

Run manually:
    python hopsworks_feature_pipeline.py

Called by GitHub Actions:
    .github/workflows/feature_pipeline.yml
"""

import os
import sys
import warnings
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

API_KEY               = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY     = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT     = os.getenv("HOPSWORKS_PROJECT_NAME", "PearlsAQI")

LAT = 24.7967   # Defence Phase 7, Karachi
LON = 67.0728

print("=" * 60)
print("  Pearls AQI — Hopsworks Feature Pipeline")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────
# 1. FETCH LIVE DATA
# ─────────────────────────────────────────────────────────────────────────
if not API_KEY:
    print("ERROR: OPENWEATHER_API_KEY not set"); sys.exit(1)

print("\n[1/4] Fetching live weather …")
try:
    r = requests.get("https://api.open-meteo.com/v1/forecast",
        params={"latitude": LAT, "longitude": LON,
                "current": ["temperature_2m","relative_humidity_2m",
                            "surface_pressure","wind_speed_10m","rain"],
                "timezone": "Asia/Karachi"},
        timeout=15)
    r.raise_for_status()
    cur = r.json()["current"]
    temperature = float(cur["temperature_2m"])
    humidity    = float(cur["relative_humidity_2m"])
    pressure    = float(cur["surface_pressure"])
    wind_speed  = float(cur["wind_speed_10m"])
    rain        = float(cur["rain"])
    print(f"      Temp={temperature}°C  Humidity={humidity}%  Wind={wind_speed}km/h")
except Exception as e:
    print(f"ERROR fetching weather: {e}"); sys.exit(1)

print("\n[2/4] Fetching live air quality …")
try:
    r2 = requests.get("https://api.openweathermap.org/data/2.5/air_pollution",
        params={"lat": LAT, "lon": LON, "appid": API_KEY}, timeout=15)
    if r2.status_code == 401:
        print("ERROR: Invalid OpenWeather API key"); sys.exit(1)
    r2.raise_for_status()
    comp            = r2.json()["list"][0]["components"]
    pm2_5           = float(comp["pm2_5"])
    pm10            = float(comp["pm10"])
    carbon_monoxide = float(comp["co"])
    nitrogen_dioxide= float(comp["no2"])
    sulphur_dioxide = float(comp["so2"])
    ozone           = float(comp["o3"])
    print(f"      PM2.5={pm2_5}  PM10={pm10}  O3={ozone}")
except Exception as e:
    print(f"ERROR fetching air quality: {e}"); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────
# 2. COMPUTE AQI
# ─────────────────────────────────────────────────────────────────────────
def pm25_to_aqi(c):
    bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
          (55.5,150.4,151,200),(150.5,250.4,201,300),
          (250.5,350.4,301,400),(350.5,500.4,401,500)]
    for cl,ch,il,ih in bp:
        if cl <= c <= ch:
            return round((ih-il)/(ch-cl)*(c-cl)+il, 1)
    return 500.0

aqi_value = pm25_to_aqi(pm2_5)
print(f"\n      Computed AQI: {aqi_value}")

# ─────────────────────────────────────────────────────────────────────────
# 3. ENGINEER FEATURES
# ─────────────────────────────────────────────────────────────────────────
print("\n[3/4] Engineering features …")
try:
    # Load historical data to compute lag features
    hist = pd.read_csv("data/processed/final_features.csv")
    hist["date"] = pd.to_datetime(hist["date"], utc=True, errors="coerce").dt.tz_localize(None)
    hist = hist.sort_values("date")

    lag1  = float(hist["AQI"].iloc[-1])
    lag2  = float(hist["AQI_lag_1"].iloc[-1])
    lag3  = float(hist["AQI_lag_2"].iloc[-1])
    mean3 = (lag1 + lag2 + lag3) / 3
    mean7 = float(hist["AQI_7day_mean"].iloc[-1])

    today = datetime.now()
    new_row = {
        "row_id":           int(hist.index[-1]) + 2,
        "event_time":       pd.Timestamp(today.date()),
        "temperature":      temperature,
        "humidity":         humidity,
        "pressure":         pressure,
        "wind_speed":       wind_speed,
        "rain":             rain,
        "pm10":             pm10,
        "pm2_5":            pm2_5,
        "carbon_monoxide":  carbon_monoxide,
        "nitrogen_dioxide": nitrogen_dioxide,
        "sulphur_dioxide":  sulphur_dioxide,
        "ozone":            ozone,
        "AQI":              aqi_value,
        "year":             today.year,
        "month":            today.month,
        "day":              today.day,
        "day_of_week":      today.weekday(),
        "day_of_year":      today.timetuple().tm_yday,
        "weekend":          1 if today.weekday() >= 5 else 0,
        "AQI_lag_1":        lag1,
        "AQI_lag_2":        lag2,
        "AQI_lag_3":        lag3,
        "AQI_3day_mean":    mean3,
        "AQI_7day_mean":    mean7,
        "rain_flag":        1 if rain > 0 else 0,
        "temp_humidity":    temperature * humidity,
    }
    new_df = pd.DataFrame([new_row])
    print(f"      Row engineered for {today.strftime('%Y-%m-%d')}")
    print(f"      AQI={aqi_value}  lag1={lag1:.1f}  3d_mean={mean3:.1f}")

except Exception as e:
    print(f"ERROR engineering features: {e}"); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────
# 4. PUSH TO HOPSWORKS FEATURE STORE
# ─────────────────────────────────────────────────────────────────────────
print("\n[4/4] Pushing to Hopsworks Feature Store …")

if not HOPSWORKS_API_KEY:
    print("ERROR: HOPSWORKS_API_KEY not set in .env"); sys.exit(1)

try:
    import hopsworks
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    fg.insert(new_df, write_options={"wait_for_job": True})

    print(f"      ✅ Row pushed to Hopsworks Feature Store")
    print(f"         Feature group: aqi_features")
    print(f"         Date: {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"         AQI: {aqi_value}")

    # Also save locally as backup
    hist_updated = pd.concat(
        [hist, new_df.rename(columns={"event_time": "date"}).drop(columns=["row_id"])],
        ignore_index=True)
    hist_updated.to_csv("data/processed/final_features.csv", index=False)
    print("      ✅ Local CSV backup updated")

except ImportError:
    print("ERROR: hopsworks not installed. Run: pip install hopsworks")
    sys.exit(1)
except Exception as e:
    print(f"ERROR pushing to Hopsworks: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  ✅ Feature Pipeline Complete!")
print("=" * 60)