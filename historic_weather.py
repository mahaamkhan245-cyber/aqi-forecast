import requests
import pandas as pd
import os

LATITUDE = 24.8607
LONGITUDE = 67.0011

START_DATE = "2022-08-01"
END_DATE = "2024-12-31"

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "pressure_msl_mean",
        "wind_speed_10m_mean",
        "rain_sum"
    ],
    "timezone": "Asia/Karachi"
}

response = requests.get(url, params=params)
response.raise_for_status()   # Better error handling

data = response.json()

daily_data = data["daily"]

weather_df = pd.DataFrame(daily_data)

weather_df.rename(columns={
    "time": "date",
    "temperature_2m_mean": "temperature",
    "relative_humidity_2m_mean": "humidity",
    "pressure_msl_mean": "pressure",
    "wind_speed_10m_mean": "wind_speed",
    "rain_sum": "rain"
}, inplace=True)

print(weather_df.head())
print(weather_df.shape)

# Create processed folder if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

weather_df.to_csv(
    "data/processed/daily_weather.csv",
    index=False
)

print("✅ Daily weather data saved successfully!")