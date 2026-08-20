"""
api.py — Flask REST API for Pearls AQI Predictor
Exposes two endpoints:
  GET  /predict        → AQI prediction from latest feature row
  POST /predict        → AQI prediction from custom feature JSON body
  GET  /forecast       → 3-day forecast from aqi_forecast.csv
  GET  /health         → service health check
Run: python api.py
"""

import os
import json
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from utils.aqi import get_aqi_category

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH   = "data/models/random_forest_aqi.pkl"
FEATURES_PATH = "data/processed/final_features.csv"
FORECAST_PATH = "data/processed/aqi_forecast.csv"

# Load once at startup
try:
    model = joblib.load(MODEL_PATH)
    df    = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
except Exception as e:
    raise RuntimeError(f"Failed to load model or data: {e}")

FEATURE_COLS = [c for c in df.columns if c not in ("date", "AQI")]


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def _row_to_prediction(row: pd.Series) -> dict:
    X = pd.DataFrame([row[FEATURE_COLS]])
    pred = float(model.predict(X)[0])
    category, advice = get_aqi_category(pred)
    return {
        "predicted_aqi": round(pred, 2),
        "category": category,
        "advice": advice,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH,
        "records": len(df),
        "features": FEATURE_COLS,
    })


@app.route("/predict", methods=["GET"])
def predict_latest():
    """
    GET /predict
    Returns AQI prediction for the latest row in the feature dataset.
    """
    latest = df.iloc[-1]
    result = _row_to_prediction(latest)
    result["date"] = str(latest["date"].date())
    result["actual_aqi"] = float(latest["AQI"])
    result["error"] = round(abs(result["predicted_aqi"] - result["actual_aqi"]), 2)
    return jsonify(result)


@app.route("/predict", methods=["POST"])
def predict_custom():
    """
    POST /predict
    Body (JSON): supply any subset of feature values.
    Missing values are filled from the latest row.

    Example body:
    {
      "temperature": 32.5,
      "humidity": 78,
      "pm2_5": 45.0,
      "pm10": 90.0
    }
    """
    body = request.get_json(silent=True) or {}

    # Start from latest row as defaults
    base = df.iloc[-1].copy()

    # Override with user-supplied values
    unknown_keys = []
    for key, val in body.items():
        if key in FEATURE_COLS:
            base[key] = val
        else:
            unknown_keys.append(key)

    result = _row_to_prediction(base)
    result["overridden_features"] = list(body.keys())
    if unknown_keys:
        result["ignored_keys"] = unknown_keys

    return jsonify(result)


@app.route("/forecast", methods=["GET"])
def forecast():
    """
    GET /forecast
    Returns the 3-day AQI forecast from aqi_forecast.csv.
    Optional query param: ?days=3 (default 3, max 5)
    """
    days = min(int(request.args.get("days", 3)), 5)

    try:
        fc = pd.read_csv(FORECAST_PATH)
        fc["date"] = pd.to_datetime(fc["date"])
    except FileNotFoundError:
        return jsonify({"error": "Forecast file not found. Run refresh_data.py first."}), 404

    results = []
    for _, row in fc.head(days).iterrows():
        cat, adv = get_aqi_category(row["Predicted_AQI"])
        results.append({
            "date":          str(row["date"].date()),
            "predicted_aqi": round(float(row["Predicted_AQI"]), 2),
            "category":      cat,
            "advice":        adv,
        })

    return jsonify({
        "city":        "Karachi, Pakistan",
        "generated":   datetime.utcnow().isoformat() + "Z",
        "forecast":    results,
    })


@app.route("/features", methods=["GET"])
def feature_list():
    """GET /features — list all feature names the model uses."""
    return jsonify({"features": FEATURE_COLS, "count": len(FEATURE_COLS)})


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🌍 Pearls AQI API running on http://0.0.0.0:{port}")
    print("  GET  /health")
    print("  GET  /predict")
    print("  POST /predict   (JSON body with feature overrides)")
    print("  GET  /forecast")
    print("  GET  /features\n")
    app.run(host="0.0.0.0", port=port, debug=False)
