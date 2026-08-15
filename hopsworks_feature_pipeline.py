"""
hopsworks_feature_pipeline.py

Runs every hour via GitHub Actions.

Fetches:
    - Live weather from Open-Meteo
    - Live air quality from OpenWeather

Engineers one new AQI feature row and pushes it to:
    Hopsworks Feature Group: aqi_features v1

Run manually:
    python hopsworks_feature_pipeline.py
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

# ============================================================
# CONFIGURATION
# ============================================================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv(
    "HOPSWORKS_PROJECT_NAME",
    "pearl_aqi"
)

LAT = 24.7967
LON = 67.0728

CSV_PATH = "data/processed/final_features.csv"

print("=" * 60)
print("  Pearls AQI — Hopsworks Feature Pipeline")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)


# ============================================================
# 1. FETCH LIVE WEATHER
# ============================================================

print("\n[1/4] Fetching live weather …")

try:
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT,
            "longitude": LON,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
                "rain",
            ],
            "timezone": "Asia/Karachi",
        },
        timeout=15,
    )

    weather_response.raise_for_status()

    current = weather_response.json()["current"]

    temperature = float(current["temperature_2m"])
    humidity = float(current["relative_humidity_2m"])
    pressure = float(current["surface_pressure"])
    wind_speed = float(current["wind_speed_10m"])
    rain = float(current["rain"])

    print(
        f"      Temp={temperature}°C  "
        f"Humidity={humidity}%  "
        f"Wind={wind_speed}km/h"
    )

except Exception as e:
    print(f"      ❌ Weather API failed: {e}")
    sys.exit(1)


# ============================================================
# 2. FETCH LIVE AIR QUALITY
# ============================================================

print("\n[2/4] Fetching live air quality …")

if not OPENWEATHER_API_KEY:
    print("      ❌ OPENWEATHER_API_KEY not found in .env")
    sys.exit(1)

try:
    air_response = requests.get(
        "https://api.openweathermap.org/data/2.5/air_pollution",
        params={
            "lat": LAT,
            "lon": LON,
            "appid": OPENWEATHER_API_KEY,
        },
        timeout=15,
    )

    if air_response.status_code == 401:
        print("      ❌ Invalid OpenWeather API key")
        sys.exit(1)

    air_response.raise_for_status()

    components = air_response.json()["list"][0]["components"]

    pm2_5 = float(components["pm2_5"])
    pm10 = float(components["pm10"])
    carbon_monoxide = float(components["co"])
    nitrogen_dioxide = float(components["no2"])
    sulphur_dioxide = float(components["so2"])
    ozone = float(components["o3"])

    print(
        f"      PM2.5={pm2_5}  "
        f"PM10={pm10}  "
        f"O3={ozone}"
    )

except Exception as e:
    print(f"      ❌ Air quality API failed: {e}")
    sys.exit(1)


# ============================================================
# AQI CALCULATION
# ============================================================

def pm25_to_aqi(c):
    breakpoints = [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for cl, ch, il, ih in breakpoints:
        if cl <= c <= ch:
            return round(
                ((ih - il) / (ch - cl)) * (c - cl) + il,
                1,
            )

    return 500.0


aqi_value = pm25_to_aqi(pm2_5)

print(f"\n      Computed AQI: {aqi_value}")


# ============================================================
# 3. ENGINEER FEATURES
# ============================================================

print("\n[3/4] Engineering features …")

try:
    hist = pd.read_csv(CSV_PATH)

    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    # --------------------------------------------
    # Historical AQI values
    # --------------------------------------------

    lag1 = float(hist["AQI"].iloc[-1])
    lag2 = float(hist["AQI_lag_1"].iloc[-1])
    lag3 = float(hist["AQI_lag_2"].iloc[-1])

    mean3 = (lag1 + lag2 + lag3) / 3

    mean7 = float(hist["AQI_7day_mean"].iloc[-1])

    today = datetime.now()

    # --------------------------------------------
    # Generate next row ID
    # --------------------------------------------

    # Existing CSV has no row_id, so use row count.
    next_row_id = len(hist) + 1

    # --------------------------------------------
    # IMPORTANT:
    # Hopsworks sanitized uppercase names
    # to lowercase, so use lowercase here.
    # --------------------------------------------
    new_row = {
    # Primary key / event time
    "row_id":           int(hist.index[-1]) + 2,
    "event_time":       pd.Timestamp(today.date()),

    # Weather
    "temperature":      float(temperature),
    "humidity":         int(round(humidity)),
    "pressure":         float(pressure),
    "wind_speed":       float(wind_speed),
    "rain":             float(rain),

    # Air quality
    "pm10":             float(pm10),
    "pm2_5":             float(pm2_5),
    "carbon_monoxide":  float(carbon_monoxide),
    "nitrogen_dioxide": float(nitrogen_dioxide),
    "sulphur_dioxide":  float(sulphur_dioxide),
    "ozone":            float(ozone),

    # Target / AQI
    "AQI":              float(aqi_value),

    # Calendar features
    "year":             int(today.year),
    "month":            int(today.month),
    "day":              int(today.day),
    "day_of_week":      int(today.weekday()),
    "day_of_year":      int(today.timetuple().tm_yday),
    "weekend":          int(1 if today.weekday() >= 5 else 0),

    # Historical AQI features
    "AQI_lag_1":        float(lag1),
    "AQI_lag_2":        float(lag2),
    "AQI_lag_3":        float(lag3),
    "AQI_3day_mean":    float(mean3),
    "AQI_7day_mean":    float(mean7),

    # Other features
    "rain_flag":        int(1 if rain > 0 else 0),
    "temp_humidity":    float(temperature * humidity),

}
    new_df = pd.DataFrame([new_row])
    new_df = new_df.astype({
    "row_id": "int64",
    "temperature": "float64",
    "humidity": "int64",
    "pressure": "float64",
    "wind_speed": "float64",
    "rain": "float64",
    "pm10": "float64",
    "pm2_5": "float64",
    "carbon_monoxide": "float64",
    "nitrogen_dioxide": "float64",
    "sulphur_dioxide": "float64",
    "ozone": "float64",
    "AQI": "float64",
    "year": "int64",
    "month": "int64",
    "day": "int64",
    "day_of_week": "int64",
    "day_of_year": "int64",
    "weekend": "int64",
    "AQI_lag_1": "float64",
    "AQI_lag_2": "float64",
    "AQI_lag_3": "float64",
    "AQI_3day_mean": "float64",
    "AQI_7day_mean": "float64",
    "rain_flag": "int64",
    "temp_humidity": "float64",
})

    print("\n      Data types:")
    print(new_df.dtypes)

    print(
        f"      Row engineered for "
        f"{today.strftime('%Y-%m-%d %H:%M')}"
    )

    print(
        f"      AQI={aqi_value}  "
        f"lag1={lag1:.1f}  "
        f"3d_mean={mean3:.1f}"
    )

except Exception as e:
    print(f"      ❌ Feature engineering failed: {e}")
    sys.exit(1)


# ============================================================
# 4. PUSH TO HOPSWORKS
# ============================================================
# ============================================================
# 4. PUSH TO HOPSWORKS
# ============================================================

print("\n[4/4] Pushing to Hopsworks Feature Store …")

if not HOPSWORKS_API_KEY:
    print("      ❌ HOPSWORKS_API_KEY not found in .env")
    sys.exit(1)

try:
    import hopsworks

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )

    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name="aqi_features",
        version=1,
    )

    if fg is None:
        print("      ❌ Feature Group 'aqi_features' v1 not found")
        sys.exit(1)

    print("      ✅ Connected to Feature Group")

    # --------------------------------------------------------
    # Create dataframe for Hopsworks
    # --------------------------------------------------------

    new_df = pd.DataFrame([new_row])

    print("\nFeature dtypes before upload:")
    print(new_df.dtypes)

    # --------------------------------------------------------
    # PUSH TO HOPSWORKS
    # --------------------------------------------------------

    fg.insert(
        new_df,
        write_options={"wait_for_job": True},
    )

    print("      ✅ Row pushed to Hopsworks Feature Store")
    print("         Feature group: aqi_features v1")
    print(
        f"         Date: {today.strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"         AQI: {aqi_value}")

except ImportError as e:
    print(
        f"      ❌ Hopsworks import failed: {e}"
    )
    sys.exit(1)

except Exception as e:
    print(f"      ❌ Hopsworks push failed: {e}")
    sys.exit(1)


# ============================================================
# 5. LOCAL CSV BACKUP
# ============================================================

print("\n[5/5] Updating local CSV backup …")

try:
    # Start from the SAME dataframe that was successfully
    # uploaded to Hopsworks.
    backup_df = new_df.copy()

    # Hopsworks uses event_time.
    # Original CSV uses date.
    backup_df = backup_df.rename(
        columns={"event_time": "date"}
    )

    # Remove Hopsworks-only primary key.
    backup_df = backup_df.drop(
        columns=["row_id"]
    )

    # Restore original CSV column names.
    backup_df = backup_df.rename(
        columns={
            "aqi": "AQI",
            "aqi_lag_1": "AQI_lag_1",
            "aqi_lag_2": "AQI_lag_2",
            "aqi_lag_3": "AQI_lag_3",
            "aqi_3day_mean": "AQI_3day_mean",
            "aqi_7day_mean": "AQI_7day_mean",
        }
    )

    # IMPORTANT:
    # Your new_df currently has uppercase AQI names already.
    # Therefore the rename above does nothing to those columns,
    # which is perfectly fine.

    # Make sure backup has EXACTLY the same columns
    # as the existing historical CSV.
    backup_df = backup_df[hist.columns]

    # Add new row to historical data.
    hist_updated = pd.concat(
        [hist, backup_df],
        ignore_index=True,
    )

    # Save updated CSV.
    hist_updated.to_csv(
        "data/processed/final_features.csv",
        index=False,
    )

    print("      ✅ Local CSV backup updated")
    print(
        f"      Total local rows: {len(hist_updated)}"
    )

except Exception as e:
    # IMPORTANT:
    # Hopsworks upload already succeeded.
    # Therefore don't report this as a Hopsworks failure.
    print(
        f"      ⚠️ Hopsworks upload succeeded, "
        f"but local CSV backup failed: {e}"
    )
    sys.exit(1)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("  ✅ Feature Pipeline Complete!")
print("=" * 60)