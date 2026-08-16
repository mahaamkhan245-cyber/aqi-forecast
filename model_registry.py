"""
register_models.py
Registers all trained models into the Hopsworks Model Registry.
Models: Random Forest, Ridge Regression, PyTorch LSTM
"""

import os
import sys
import json
import shutil
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT_NAME", "pearl_aqi")

REGISTRY_CSV        = "data/registry/model_registry.csv"
FEATURES_CSV        = "data/processed/final_features.csv"

MODELS = [
    {
        "name":        "random_forest_aqi",
        "version":     1,
        "file":        "data/models/random_forest_v8.pkl",
        "scaler":      None,
        "type":        "sklearn",
        "description": "Random Forest Regressor for AQI forecasting — Defence Phase 7, Karachi",
    },
    {
        "name":        "ridge_regression_aqi",
        "version":     1,
        "file":        "data/models/ridge_model.pkl",
        "scaler":      None,
        "type":        "sklearn",
        "description": "Ridge Regression for AQI forecasting — Defence Phase 7, Karachi",
    },
    {
        "name":        "pytorch_lstm_aqi",
        "version":     1,
        "file":        "data/models/pytorch_lstm.pt",
        "scaler":      "data/models/pytorch_scaler.pkl",
        "type":        "pytorch",
        "description": "Bidirectional PyTorch LSTM for AQI forecasting — Defence Phase 7, Karachi",
    },
]

print("=" * 60)
print("  Pearls AQI — Hopsworks Model Registry")
print("=" * 60)

# ============================================================
# VALIDATE FILES
# ============================================================

print("\n[1/3] Validating model files …")

for m in MODELS:
    if not os.path.exists(m["file"]):
        print(f"  ❌ Missing: {m['file']}")
        sys.exit(1)
    if m["scaler"] and not os.path.exists(m["scaler"]):
        print(f"  ❌ Missing scaler: {m['scaler']}")
        sys.exit(1)
    print(f"  ✅ {m['name']} — {m['file']}")

# ============================================================
# LOAD METRICS FROM REGISTRY CSV
# ============================================================

print("\n[2/3] Loading metrics from model registry …")

metrics_map = {}

try:
    reg = pd.read_csv(REGISTRY_CSV)

    # Map algorithm names to our model names
    algo_to_name = {
        "RandomForest":  "random_forest_aqi",
        "Random Forest": "random_forest_aqi",
        "Ridge":         "ridge_regression_aqi",
        "PyTorch LSTM":  "pytorch_lstm_aqi",
        "LSTM":          "pytorch_lstm_aqi",
    }

    for _, row in reg.iterrows():
        algo = str(row.get("algorithm", ""))
        for key, model_name in algo_to_name.items():
            if key.lower() in algo.lower():
                metrics_map[model_name] = {
                    "r2":   float(row.get("r2",   0)),
                    "mae":  float(row.get("mae",  0)),
                    "rmse": float(row.get("rmse", 0)),
                }
                break

    print(f"  ✅ Loaded metrics for {len(metrics_map)} models")
    for name, m in metrics_map.items():
        print(f"     {name}: R²={m['r2']:.3f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}")

except Exception as e:
    print(f"  ⚠️  Could not load registry CSV: {e}")
    print("      Using placeholder metrics.")
    for m in MODELS:
        metrics_map[m["name"]] = {"r2": 0.0, "mae": 0.0, "rmse": 0.0}

# ============================================================
# CONNECT TO HOPSWORKS
# ============================================================

print("\n[3/3] Connecting to Hopsworks …")

if not HOPSWORKS_API_KEY:
    print("  ❌ HOPSWORKS_API_KEY not set in .env")
    sys.exit(1)

try:
    import hopsworks

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )

    mr = project.get_model_registry()
    print("  ✅ Connected to Hopsworks Model Registry")

except ImportError as e:
    print(f"  ❌ Hopsworks import failed: {e}")
    sys.exit(1)

except Exception as e:
    print(f"  ❌ Hopsworks login failed: {e}")
    sys.exit(1)

# ============================================================
# REGISTER EACH MODEL
# ============================================================

print("\n" + "=" * 60)
print("  Registering Models")
print("=" * 60)

registered = []

for m in MODELS:

    print(f"\n  → Registering: {m['name']} …")

    # ── Create a temp directory to hold model artifacts ──────
    tmpdir = tempfile.mkdtemp()

    try:

        metrics = metrics_map.get(m["name"], {"r2": 0.0, "mae": 0.0, "rmse": 0.0})

        # ── Copy model file into temp dir ────────────────────
        model_filename = os.path.basename(m["file"])
        shutil.copy(m["file"], os.path.join(tmpdir, model_filename))

        # ── Copy scaler if exists (PyTorch LSTM) ─────────────
        if m["scaler"]:
            scaler_filename = os.path.basename(m["scaler"])
            shutil.copy(m["scaler"], os.path.join(tmpdir, scaler_filename))

        # ── Save model metadata JSON ──────────────────────────
        metadata = {
            "model_name":  m["name"],
            "model_type":  m["type"],
            "model_file":  model_filename,
            "scaler_file": os.path.basename(m["scaler"]) if m["scaler"] else None,
            "description": m["description"],
            "metrics": {
                "r2":   metrics["r2"],
                "mae":  metrics["mae"],
                "rmse": metrics["rmse"],
            },
            "features": [
                "temperature", "humidity", "pressure", "wind_speed", "rain",
                "pm2_5", "pm10", "ozone", "carbon_monoxide",
                "nitrogen_dioxide", "sulphur_dioxide",
                "year", "month", "day", "day_of_week", "day_of_year", "weekend",
                "AQI_lag_1", "AQI_lag_2", "AQI_lag_3",
                "AQI_3day_mean", "AQI_7day_mean",
                "rain_flag", "temp_humidity",
            ],
            "target":   "AQI",
            "location": "Defence Phase 7, Karachi",
        }

        with open(os.path.join(tmpdir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # ── Register in Hopsworks ─────────────────────────────
        hops_model = mr.python.create_model(
            name=m["name"],
            version=m["version"],
            metrics={
                "r2":   metrics["r2"],
                "mae":  metrics["mae"],
                "rmse": metrics["rmse"],
            },
            description=m["description"],
            input_example=None,
        )

        hops_model.save(tmpdir)

        print(f"     ✅ {m['name']} v{m['version']} registered")
        print(f"        R²={metrics['r2']:.3f}  MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}")

        registered.append(m["name"])

    except Exception as e:
        print(f"     ❌ Failed to register {m['name']}: {e}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)

if len(registered) == len(MODELS):
    print(f"  ✅ All {len(registered)} models registered successfully!")
else:
    print(f"  ⚠️  {len(registered)}/{len(MODELS)} models registered.")
    failed = [m["name"] for m in MODELS if m["name"] not in registered]
    for f in failed:
        print(f"     ❌ {f}")

print("\n  Check your Hopsworks dashboard:")
print("  https://app.hopsworks.ai → Model Registry")
print("=" * 60)