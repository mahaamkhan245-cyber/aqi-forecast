"""
train_model_comparison.py
Trains and compares three models:
  1. Random Forest (baseline, already in train_model.py)
  2. Ridge Regression
  3. TensorFlow LSTM (deep learning, suited for time-series AQI)

Saves the BEST model to data/models/random_forest_aqi.pkl (or best_model.pkl)
and logs all three to the model registry.

Run: python train_model_comparison.py
"""

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/final_features.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)   # keep chronological order

X = df.drop(columns=["date", "AQI"])
y = df["AQI"]

FEATURE_COLS = list(X.columns)

# Chronological split — do NOT shuffle for time-series
split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")
print(f"Features         : {len(FEATURE_COLS)}")

os.makedirs("data/models",   exist_ok=True)
os.makedirs("data/registry", exist_ok=True)

results = []   # collect metrics for all models


# ─────────────────────────────────────────────────────────────
# Helper: evaluate & record
# ─────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return mae, rmse, r2


# ─────────────────────────────────────────────────────────────
# Model 1: Random Forest
# ─────────────────────────────────────────────────────────────
print("\n[1/3] Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae, rf_rmse, rf_r2 = evaluate("Random Forest", y_test, rf_pred)

joblib.dump(rf, "data/models/random_forest_v_compare.pkl")
results.append({
    "algorithm": "Random Forest",
    "mae":  round(rf_mae,  4),
    "rmse": round(rf_rmse, 4),
    "r2":   round(rf_r2,   4),
    "model_file": "random_forest_v_compare.pkl",
})


# ─────────────────────────────────────────────────────────────
# Model 2: Ridge Regression (needs scaling)
# ─────────────────────────────────────────────────────────────
print("\n[2/3] Training Ridge Regression...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_train)
ridge_pred = ridge.predict(X_test_sc)
ridge_mae, ridge_rmse, ridge_r2 = evaluate("Ridge Regression", y_test, ridge_pred)

joblib.dump({"model": ridge, "scaler": scaler}, "data/models/ridge_model.pkl")
results.append({
    "algorithm": "Ridge Regression",
    "mae":  round(ridge_mae,  4),
    "rmse": round(ridge_rmse, 4),
    "r2":   round(ridge_r2,   4),
    "model_file": "ridge_model.pkl",
})


# ─────────────────────────────────────────────────────────────
# Model 3: TensorFlow LSTM
# ─────────────────────────────────────────────────────────────
print("\n[3/3] Training TensorFlow LSTM...")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    # Scale features
    scaler_lstm = StandardScaler()
    X_train_lstm = scaler_lstm.fit_transform(X_train)
    X_test_lstm  = scaler_lstm.transform(X_test)

    # Reshape for LSTM: (samples, timesteps=1, features)
    X_train_3d = X_train_lstm.reshape(X_train_lstm.shape[0], 1, X_train_lstm.shape[1])
    X_test_3d  = X_test_lstm.reshape(X_test_lstm.shape[0],  1, X_test_lstm.shape[1])

    # Build LSTM
    lstm_model = Sequential([
        LSTM(64, input_shape=(1, X_train_3d.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = lstm_model.fit(
        X_train_3d, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )

    lstm_pred = lstm_model.predict(X_test_3d, verbose=0).flatten()
    lstm_mae, lstm_rmse, lstm_r2 = evaluate("TensorFlow LSTM", y_test, lstm_pred)

    # Save LSTM model + scaler
    lstm_model.save("data/models/lstm_model.keras")
    joblib.dump(scaler_lstm, "data/models/lstm_scaler.pkl")

    results.append({
        "algorithm": "TensorFlow LSTM",
        "mae":  round(lstm_mae,  4),
        "rmse": round(lstm_rmse, 4),
        "r2":   round(lstm_r2,   4),
        "model_file": "lstm_model.keras",
    })

except ImportError:
    print("  TensorFlow not installed — skipping LSTM. Run: pip install tensorflow")
    lstm_r2 = -999   # ensure LSTM doesn't win if not installed


# ─────────────────────────────────────────────────────────────
# Pick best model by R²
# ─────────────────────────────────────────────────────────────
best = max(results, key=lambda x: x["r2"])

print(f"\n{'='*50}")
print(f"  BEST MODEL: {best['algorithm']}")
print(f"  R² = {best['r2']} | MAE = {best['mae']} | RMSE = {best['rmse']}")
print(f"{'='*50}")

# Copy best model as the production model used by app.py / h.py
if best["algorithm"] == "Random Forest":
    joblib.dump(rf, "data/models/random_forest_aqi.pkl")
    print("  → Saved best model as data/models/random_forest_aqi.pkl")
elif best["algorithm"] == "Ridge Regression":
    joblib.dump({"model": ridge, "scaler": scaler}, "data/models/best_model.pkl")
    print("  → Saved best model as data/models/best_model.pkl")
elif best["algorithm"] == "TensorFlow LSTM":
    lstm_model.save("data/models/best_model_lstm.keras")
    print("  → Saved best LSTM model as data/models/best_model_lstm.keras")


# ─────────────────────────────────────────────────────────────
# Update model registry
# ─────────────────────────────────────────────────────────────
registry_path = "data/registry/model_registry.csv"
if os.path.exists(registry_path):
    registry = pd.read_csv(registry_path)
    registry["status"] = "Archived"
else:
    registry = pd.DataFrame(columns=[
        "version", "model_name", "algorithm",
        "mae", "rmse", "r2",
        "train_date", "status", "model_file"
    ])

for i, res in enumerate(results):
    version = len(registry) + i + 1
    is_best = res["algorithm"] == best["algorithm"]
    new_row = pd.DataFrame([{
        "version":    version,
        "model_name": "AQI Predictor",
        "algorithm":  res["algorithm"],
        "mae":        res["mae"],
        "rmse":       res["rmse"],
        "r2":         res["r2"],
        "train_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status":     "Production" if is_best else "Challenger",
        "model_file": res["model_file"],
    }])
    registry = pd.concat([registry, new_row], ignore_index=True)

registry.to_csv(registry_path, index=False)

print("\nModel Registry (current):")
print(registry.to_string(index=False))
print("\n✅ Model comparison complete!")
