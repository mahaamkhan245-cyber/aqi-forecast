import pandas as pd
import joblib

# =====================================================
# Load Trained Model
# =====================================================

model = joblib.load(
    "data/models/random_forest_aqi.pkl"
)

# =====================================================
# Load Weather Forecast
# =====================================================

weather = pd.read_csv(
    "data/processed/weather_forecast.csv"
)

weather["date"] = pd.to_datetime(weather["date"])

# =====================================================
# Load Air Pollution Forecast
# =====================================================

air = pd.read_csv(
    "data/processed/air_quality_forecast.csv"
)

air["date"] = pd.to_datetime(air["date"])

# =====================================================
# Merge Weather + Air Pollution
# =====================================================

forecast = pd.merge(
    weather,
    air,
    on="date",
    how="inner"
)

# =====================================================
# Load Historical Dataset
# =====================================================

history = pd.read_csv(
    "data/processed/final_features.csv"
)

history["date"] = pd.to_datetime(history["date"])

latest = history.iloc[-1]

# =====================================================
# Date Features
# =====================================================

forecast["year"] = forecast["date"].dt.year
forecast["month"] = forecast["date"].dt.month
forecast["day"] = forecast["date"].dt.day
forecast["day_of_week"] = forecast["date"].dt.dayofweek
forecast["day_of_year"] = forecast["date"].dt.dayofyear
forecast["weekend"] = (
    forecast["day_of_week"] >= 5
).astype(int)

# =====================================================
# Feature List
# =====================================================

features = [

    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "rain",

    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",

    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "weekend",

    "AQI_lag_1",
    "AQI_lag_2",
    "AQI_lag_3",

    "AQI_3day_mean",
    "AQI_7day_mean",

    "rain_flag",
    "temp_humidity"

]

# =====================================================
# Recursive AQI Forecast
# =====================================================

predictions = []

lag1 = latest["AQI"]
lag2 = latest["AQI_lag_1"]
lag3 = latest["AQI_lag_2"]

mean7 = latest["AQI_7day_mean"]

for i in range(len(forecast)):

    # Historical AQI Features

    forecast.loc[i, "AQI_lag_1"] = lag1
    forecast.loc[i, "AQI_lag_2"] = lag2
    forecast.loc[i, "AQI_lag_3"] = lag3

    forecast.loc[i, "AQI_3day_mean"] = (
        lag1 + lag2 + lag3
    ) / 3

    forecast.loc[i, "AQI_7day_mean"] = mean7

    # Engineered Features

    forecast.loc[i, "rain_flag"] = int(
        forecast.loc[i, "rain"] > 0
    )

    forecast.loc[i, "temp_humidity"] = (
        forecast.loc[i, "temperature"]
        *
        forecast.loc[i, "humidity"]
    )

    # Predict

    row = forecast.loc[[i], features]

    prediction = model.predict(row)[0]

    predictions.append(prediction)

    # Update lag values for next day

    lag3 = lag2
    lag2 = lag1
    lag1 = prediction

forecast["Predicted_AQI"] = predictions

# =====================================================
# AQI Category
# =====================================================

def get_category(aqi):

    if aqi <= 50:
        return "Good 🟢"

    elif aqi <= 100:
        return "Moderate 🟡"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups 🟠"

    elif aqi <= 200:
        return "Unhealthy 🔴"

    elif aqi <= 300:
        return "Very Unhealthy 🟣"

    else:
        return "Hazardous ⚫"

forecast["Category"] = forecast["Predicted_AQI"].apply(get_category)

# =====================================================
# Save Forecast
# =====================================================

forecast.to_csv(
    "data/processed/aqi_forecast.csv",
    index=False
)

# =====================================================
# Display Results
# =====================================================

print("\n" + "=" * 80)
print("                 3-DAY AQI FORECAST")
print("=" * 80)

print(
    forecast[
        [
            "date",
            "temperature",
            "humidity",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "Predicted_AQI",
            "Category"
        ]
    ]
)

print("\nAQI Forecast saved to:")
print("data/processed/aqi_forecast.csv")

print("\n✅ Recursive AQI Forecast Completed Successfully!")