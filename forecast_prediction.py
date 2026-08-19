import sys
import os
import pandas as pd
import joblib
import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD PRODUCTION MODEL FROM REGISTRY
# ══════════════════════════════════════════════════════════════════════════

try:

    registry = pd.read_csv(
        "data/registry/model_registry.csv"
    )

    prod_rows = registry[
        registry["status"] == "Production"
    ]

    if prod_rows.empty:
        print(
            "ERROR: No Production model in registry.",
            file=sys.stderr
        )
        sys.exit(1)

    production = prod_rows.iloc[-1]

    algorithm = production["algorithm"]
    model_file = production["model_file"]

    model_path = os.path.join(
        "data/models",
        model_file
    )

    if not os.path.exists(model_path):
        print(
            f"ERROR: Model file not found: {model_path}",
            file=sys.stderr
        )
        sys.exit(1)

    print(f"✅ Production model: {algorithm}")
    print(f"   Model file: {model_file}")


    # ══════════════════════════════════════════════════════════════════════
    # PYTORCH LSTM
    # ══════════════════════════════════════════════════════════════════════

    if algorithm == "PyTorch LSTM":

        import torch
        import torch.nn as nn

        print("   Loading PyTorch checkpoint...")

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        input_size = int(
            checkpoint["input_size"]
        )

        hidden1 = int(
            checkpoint["hidden1"]
        )

        hidden2 = int(
            checkpoint["hidden2"]
        )

        print(
            f"   LSTM input size : {input_size}"
        )
        print(
            f"   Hidden layers   : {hidden1} → {hidden2}"
        )


        # ────────────────────────────────────────────────────────────────
        # EXACT SAME ARCHITECTURE USED DURING TRAINING
        # ────────────────────────────────────────────────────────────────

        class AQI_LSTM(nn.Module):

            def __init__(
                self,
                input_size,
                hidden1=128,
                hidden2=64
            ):
                super().__init__()

                self.lstm1 = nn.LSTM(
                    input_size,
                    hidden1,
                    batch_first=True,
                    bidirectional=True
                )

                self.drop1 = nn.Dropout(0.2)

                self.lstm2 = nn.LSTM(
                    hidden1 * 2,
                    hidden2,
                    batch_first=True,
                    bidirectional=False
                )

                self.drop2 = nn.Dropout(0.2)

                self.fc1 = nn.Linear(
                    hidden2,
                    32
                )

                self.bn = nn.BatchNorm1d(32)

                self.relu = nn.ReLU()

                self.fc2 = nn.Linear(
                    32,
                    1
                )

            def forward(self, x):

                out, _ = self.lstm1(x)

                out = self.drop1(out)

                out, _ = self.lstm2(out)

                out = self.drop2(
                    out[:, -1, :]
                )

                out = self.relu(
                    self.bn(
                        self.fc1(out)
                    )
                )

                return self.fc2(out)


        # ────────────────────────────────────────────────────────────────
        # BUILD MODEL
        # ────────────────────────────────────────────────────────────────

        model = AQI_LSTM(
            input_size=input_size,
            hidden1=hidden1,
            hidden2=hidden2
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        model.eval()

        print("✅ PyTorch LSTM loaded successfully")


        # ────────────────────────────────────────────────────────────────
        # LOAD LSTM SCALER
        # ────────────────────────────────────────────────────────────────

        scaler_path = os.path.join(
            "data/models",
            "pytorch_scaler.pkl"
        )

        if not os.path.exists(scaler_path):

            print(
                f"ERROR: PyTorch scaler not found: {scaler_path}",
                file=sys.stderr
            )

            sys.exit(1)

        scaler = joblib.load(
            scaler_path
        )

        print(
            "✅ PyTorch feature scaler loaded"
        )


    # ══════════════════════════════════════════════════════════════════════
    # RANDOM FOREST / RIDGE
    # ══════════════════════════════════════════════════════════════════════

    else:

        model = joblib.load(
            model_path
        )

        print(
            f"✅ Sklearn model loaded: {model_file}"
        )


except Exception as e:

    print(
        f"ERROR loading model: {e}",
        file=sys.stderr
    )

    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# 2. LOAD WEATHER + AIR QUALITY FORECASTS
# ══════════════════════════════════════════════════════════════════════════

try:

    weather = pd.read_csv(
        "data/processed/weather_forecast.csv"
    )

    weather["date"] = pd.to_datetime(
        weather["date"]
    )


    air = pd.read_csv(
        "data/processed/air_quality_forecast.csv"
    )

    air["date"] = pd.to_datetime(
        air["date"]
    )

except Exception as e:

    print(
        f"ERROR loading forecast CSVs: {e}",
        file=sys.stderr
    )

    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# 3. MERGE WEATHER + AIR QUALITY
# ══════════════════════════════════════════════════════════════════════════

forecast = pd.merge(
    weather,
    air,
    on="date",
    how="inner"
)

if forecast.empty:

    print(
        "ERROR: Merge of weather + air quality produced 0 rows.",
        file=sys.stderr
    )

    print(
        f"  weather dates:     {weather['date'].tolist()}",
        file=sys.stderr
    )

    print(
        f"  air quality dates: {air['date'].tolist()}",
        file=sys.stderr
    )

    print(
        "Dates do not overlap — run forecast_weather.py "
        "and forecast_air_quality.py first.",
        file=sys.stderr
    )

    sys.exit(1)


print(
    f"✅ Merged {len(forecast)} forecast rows"
)


# ══════════════════════════════════════════════════════════════════════════
# 4. LOAD HISTORICAL FEATURES
# ══════════════════════════════════════════════════════════════════════════

try:

    history = pd.read_csv(
        "data/processed/final_features.csv"
    )

    history["date"] = pd.to_datetime(
        history["date"],
        utc=True,        # handles both tz-aware and tz-naive strings
        errors="coerce"  # bad rows become NaT instead of crashing
    ).dt.tz_localize(None)  # strip timezone so comparisons work cleanly

    history = history.sort_values(
        "date"
    ).reset_index(drop=True)

    latest = history.iloc[-1]

except Exception as e:

    print(
        f"ERROR loading historical features: {e}",
        file=sys.stderr
    )

    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# 5. DATE FEATURES
# ══════════════════════════════════════════════════════════════════════════

forecast["year"] = (
    forecast["date"].dt.year
)

forecast["month"] = (
    forecast["date"].dt.month
)

forecast["day"] = (
    forecast["date"].dt.day
)

forecast["day_of_week"] = (
    forecast["date"].dt.dayofweek
)

forecast["day_of_year"] = (
    forecast["date"].dt.dayofyear
)

forecast["weekend"] = (
    forecast["day_of_week"] >= 5
).astype(int)


# ══════════════════════════════════════════════════════════════════════════
# 6. FEATURES
# ══════════════════════════════════════════════════════════════════════════

FEATURES = [
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
    "temp_humidity",
]


# ══════════════════════════════════════════════════════════════════════════
# 7. CHECK FEATURE COUNT
# ══════════════════════════════════════════════════════════════════════════

if algorithm == "PyTorch LSTM":

    if len(FEATURES) != input_size:

        print(
            "\nERROR: Feature count mismatch!",
            file=sys.stderr
        )

        print(
            f"  Model expects : {input_size}",
            file=sys.stderr
        )

        print(
            f"  Forecast has  : {len(FEATURES)}",
            file=sys.stderr
        )

        print(
            "The training and prediction feature sets "
            "must contain the same number of features.",
            file=sys.stderr
        )

        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# 8. RECURSIVE AQI PREDICTION
# ══════════════════════════════════════════════════════════════════════════

lag1 = float(
    latest["AQI"]
)

lag2 = float(
    latest["AQI_lag_1"]
)

lag3 = float(
    latest["AQI_lag_2"]
)

mean7 = float(
    latest["AQI_7day_mean"]
)

predictions = []


for i in range(len(forecast)):

    # ────────────────────────────────────────────────────────────────────
    # Historical lag features
    # ────────────────────────────────────────────────────────────────────

    forecast.loc[i, "AQI_lag_1"] = lag1

    forecast.loc[i, "AQI_lag_2"] = lag2

    forecast.loc[i, "AQI_lag_3"] = lag3

    forecast.loc[i, "AQI_3day_mean"] = (
        lag1 + lag2 + lag3
    ) / 3

    forecast.loc[i, "AQI_7day_mean"] = mean7


    # ────────────────────────────────────────────────────────────────────
    # Engineered features
    # ────────────────────────────────────────────────────────────────────

    forecast.loc[i, "rain_flag"] = int(
        forecast.loc[i, "rain"] > 0
    )

    forecast.loc[i, "temp_humidity"] = (
        forecast.loc[i, "temperature"]
        *
        forecast.loc[i, "humidity"]
    )


    # ────────────────────────────────────────────────────────────────────
    # Prepare feature row
    # ────────────────────────────────────────────────────────────────────
    row = forecast.loc[[i], FEATURES]
    # ════════════════════════════════════════════════════════════════════
    # PYTORCH LSTM PREDICTION
    # ════════════════════════════════════════════════════════════════════

    if algorithm == "PyTorch LSTM":

        X_scaled = scaler.transform(
            row
        ).astype(
            np.float32
        )

        X_tensor = torch.tensor(
            X_scaled.reshape(
                1,
                1,
                input_size
            ),
            dtype=torch.float32
        )

        with torch.no_grad():

            pred = float(
                model(
                    X_tensor
                )
                .cpu()
                .numpy()
                .flatten()[0]
            )


    # ════════════════════════════════════════════════════════════════════
    # RIDGE REGRESSION
    # ════════════════════════════════════════════════════════════════════

    elif algorithm == "Ridge Regression":

        ridge_model = model["model"]

        ridge_scaler = model["scaler"]

        X_scaled = ridge_scaler.transform(
            row
        )

        pred = float(
            ridge_model.predict(
                X_scaled
            )[0]
        )


    # ════════════════════════════════════════════════════════════════════
    # RANDOM FOREST
    # ════════════════════════════════════════════════════════════════════

    else:

        pred = float(
            model.predict(
                row
            )[0]
        )


    # ────────────────────────────────────────────────────────────────────
    # Store prediction
    # ────────────────────────────────────────────────────────────────────

    predictions.append(
        pred
    )


    # ────────────────────────────────────────────────────────────────────
    # Recursive update
    # ────────────────────────────────────────────────────────────────────

    lag3 = lag2

    lag2 = lag1

    lag1 = pred


# ══════════════════════════════════════════════════════════════════════════
# 9. AQI CATEGORIES
# ══════════════════════════════════════════════════════════════════════════

forecast["Predicted_AQI"] = predictions


def get_category(aqi):

    if aqi <= 50:
        return "Good 🟢"

    if aqi <= 100:
        return "Moderate 🟡"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups 🟠"

    if aqi <= 200:
        return "Unhealthy 🔴"

    if aqi <= 300:
        return "Very Unhealthy 🟣"

    return "Hazardous ⚫"


forecast["Category"] = (
    forecast["Predicted_AQI"]
    .apply(get_category)
)


# ══════════════════════════════════════════════════════════════════════════
# 10. SAVE FORECAST
# ══════════════════════════════════════════════════════════════════════════

forecast.to_csv(
    "data/processed/aqi_forecast.csv",
    index=False
)


# ══════════════════════════════════════════════════════════════════════════
# 11. DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)

print(
    "3-DAY AQI FORECAST — Defence Phase 7, Karachi"
)

print("=" * 60)

print(
    forecast[
        [
            "date",
            "temperature",
            "humidity",
            "pm2_5",
            "Predicted_AQI",
            "Category"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\n✅ Recursive AQI Forecast Completed Successfully!"
)