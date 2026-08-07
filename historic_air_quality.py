import pandas as pd

air_df = pd.read_csv("data/air_quality_historical.csv")

# Keep only required columns
air_df = air_df[
    [
        "date",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi"
    ]
]

# Rename target column
air_df.rename(columns={
    "us_aqi": "AQI"
}, inplace=True)

print(air_df.head())

air_df.to_csv(
    "data/processed/daily_air_quality.csv",
    index=False
)

print("Daily AQI dataset saved successfully!")