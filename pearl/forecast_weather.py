import requests
import pandas as pd

LATITUDE = 24.8607
LONGITUDE = 67.0011

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "daily": [
        "temperature_2m_mean",
        "relative_humidity_2m_max",
        "relative_humidity_2m_min",
        "pressure_msl_mean",
        "wind_speed_10m_mean",
        "rain_sum"
    ],
    "forecast_days": 4,
    "timezone": "Asia/Karachi"
}

response = requests.get(url, params=params)
data = response.json()

daily = data["daily"]

forecast = pd.DataFrame(daily)

# Average max & min humidity to get daily mean
forecast["relative_humidity_2m_mean"] = (
    forecast["relative_humidity_2m_max"] + forecast["relative_humidity_2m_min"]
) / 2

forecast.drop(columns=["relative_humidity_2m_max", "relative_humidity_2m_min"], inplace=True)

forecast.rename(columns={
    "time":                      "date",
    "temperature_2m_mean":       "temperature",
    "relative_humidity_2m_mean": "humidity",
    "pressure_msl_mean":         "pressure",
    "wind_speed_10m_mean":       "wind_speed",
    "rain_sum":                  "rain"
}, inplace=True)

print(forecast)

forecast.to_csv(
    "data/processed/weather_forecast.csv",
    index=False
)

print("✅ 3-Day Weather Forecast Saved!")