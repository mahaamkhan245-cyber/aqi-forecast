"""
forecast_weather.py
Fetches 4-day weather forecast from Open-Meteo for Defence Phase 7, Karachi.
FIX: Updated coordinates to Defence Phase 7 (was wrong Karachi city centre coords).
"""
import sys
import requests
import pandas as pd

LAT = 24.7967   # Defence Phase 7, Karachi
LON = 67.0728   # Defence Phase 7, Karachi

try:
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":  LAT,
            "longitude": LON,
            "daily": [
                "temperature_2m_mean",
                "relative_humidity_2m_max",
                "relative_humidity_2m_min",
                "pressure_msl_mean",
                "wind_speed_10m_mean",
                "rain_sum",
            ],
            "forecast_days": 4,
            "timezone": "Asia/Karachi",
        },
        timeout=15,
    )
    response.raise_for_status()
    daily = response.json()["daily"]

except requests.exceptions.RequestException as e:
    print(f"ERROR: Open-Meteo API request failed: {e}", file=sys.stderr)
    sys.exit(1)

except KeyError as e:
    print(f"ERROR: Unexpected API response structure: {e}", file=sys.stderr)
    print(f"Response: {response.text[:500]}", file=sys.stderr)
    sys.exit(1)

forecast = pd.DataFrame(daily)

forecast["humidity"] = (
    forecast["relative_humidity_2m_max"] + forecast["relative_humidity_2m_min"]
) / 2

forecast.drop(
    columns=["relative_humidity_2m_max", "relative_humidity_2m_min"],
    inplace=True,
)

forecast.rename(columns={
    "time":                 "date",
    "temperature_2m_mean":  "temperature",
    "pressure_msl_mean":    "pressure",
    "wind_speed_10m_mean":  "wind_speed",
    "rain_sum":             "rain",
}, inplace=True)

if forecast.empty:
    print("ERROR: Empty forecast returned from Open-Meteo.", file=sys.stderr)
    sys.exit(1)

forecast.to_csv("data/processed/weather_forecast.csv", index=False)
print(forecast.to_string())
print(f"\n✅ Weather forecast saved ({len(forecast)} days) for Defence Phase 7 [{LAT}, {LON}]")
