import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.8607
LON = 67.0011

# ----------------------------------
# Current AQI  →  current_air_quality.csv
# ----------------------------------

current_url = "https://api.openweathermap.org/data/2.5/air_pollution"

response = requests.get(current_url, params={"lat": LAT, "lon": LON, "appid": API_KEY})
print("Current AQI Status:", response.status_code)

data = response.json()

current_rows = []
for item in data["list"]:
    current_rows.append({
        "datetime":         pd.to_datetime(item["dt"], unit="s"),
        "pm2_5":            item["components"]["pm2_5"],
        "pm10":             item["components"]["pm10"],
        "carbon_monoxide":  item["components"]["co"],
        "nitrogen_dioxide": item["components"]["no2"],
        "sulphur_dioxide":  item["components"]["so2"],
        "ozone":            item["components"]["o3"],
        "aqi":              item["main"]["aqi"]
    })

pd.DataFrame(current_rows).to_csv(
    "data/processed/current_air_quality.csv", index=False
)
print("✅ current_air_quality.csv saved!")

# ----------------------------------
# Forecast AQI  →  air_quality_forecast.csv
# ----------------------------------

forecast_url = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"

response = requests.get(forecast_url, params={"lat": LAT, "lon": LON, "appid": API_KEY})
print("Forecast AQI Status:", response.status_code)

data = response.json()

forecast_rows = []
for item in data["list"]:
    forecast_rows.append({
        "datetime":         pd.to_datetime(item["dt"], unit="s"),
        "pm2_5":            item["components"]["pm2_5"],
        "pm10":             item["components"]["pm10"],
        "carbon_monoxide":  item["components"]["co"],
        "nitrogen_dioxide": item["components"]["no2"],
        "sulphur_dioxide":  item["components"]["so2"],
        "ozone":            item["components"]["o3"],
        "aqi":              item["main"]["aqi"]
    })

df_forecast = pd.DataFrame(forecast_rows)
from datetime import datetime

# ----------------------------------
# Daily Average  →  one row per date
# ----------------------------------

df_forecast["date"] = df_forecast["datetime"].dt.normalize()

df_daily = df_forecast.groupby("date").agg({
    "pm2_5":            "mean",
    "pm10":             "mean",
    "carbon_monoxide":  "mean",
    "nitrogen_dioxide": "mean",
    "sulphur_dioxide":  "mean",
    "ozone":            "mean",
    "aqi":              "mean"
}).reset_index()

df_daily["last_updated"] = datetime.now().strftime("%d %b %Y %I:%M %p")
df_daily = df_daily[df_daily["date"] >= pd.Timestamp.now().normalize()]

df_daily.to_csv(
    "data/processed/air_quality_forecast.csv", index=False
)
print("✅ air_quality_forecast.csv saved!")
print(df_daily)