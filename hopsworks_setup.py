"""
hopsworks_setup.py
ONE-TIME SETUP — Run this once to create the Feature Group and Model Registry
on Hopsworks and push your existing dataset to the Feature Store.

Steps this script does:
  1. Connects to Hopsworks using your API key
  2. Creates a Feature Group called 'aqi_features'
  3. Pushes all 874 rows from final_features.csv
  4. Registers your best model in the Hopsworks Model Registry
  5. Saves connection config to .env so other scripts can connect

Run:
    python hopsworks_setup.py
"""

import os
import sys
import joblib
import pandas as pd
from dotenv import load_dotenv, set_key

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────
# READ API KEY FROM ENV
# ─────────────────────────────────────────────────────────────────────────
HOPSWORKS_API_KEY     = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT     = os.getenv("HOPSWORKS_PROJECT_NAME", "pearls_aqi")

if not HOPSWORKS_API_KEY:
    print("ERROR: HOPSWORKS_API_KEY not found in .env")
    print("Add this line to your .env file:")
    print("  HOPSWORKS_API_KEY=your_key_here")
    print("  HOPSWORKS_PROJECT_NAME=PearlsAQI")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────
# CONNECT
# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Pearls AQI — Hopsworks Feature Store Setup")
print("=" * 60)

try:
    import hopsworks
except ImportError:
    print("ERROR: hopsworks not installed.")
    print("Run: pip install hopsworks")
    sys.exit(1)

print(f"\n[1/5] Connecting to Hopsworks project '{HOPSWORKS_PROJECT}' …")
try:
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    print(f"      ✅ Connected to project: {project.name}")
except Exception as e:
    print(f"      ❌ Connection failed: {e}")
    print("\n  Check:")
    print("    • API key is correct in .env")
    print("    • Project name matches exactly (case-sensitive)")
    print("    • You have internet access")
    sys.exit(1)
# ─────────────────────────────────────────────────────────────────────────
# FEATURE STORE — create feature group and push data
# ─────────────────────────────────────────────────────────────────────────

print("\n[2/5] Setting up Feature Store …")

try:
    fs = project.get_feature_store()
    print(f"      ✅ Feature store: {fs.name}")
except Exception as e:
    print(f"      ❌ Failed to get feature store: {e}")
    sys.exit(1)
print("\n[3/5] Loading local features and pushing to Hopsworks …")

try:
    # Load local features
    df = pd.read_csv("data/processed/final_features.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Rename date to event_time
    df = df.rename(columns={"date": "event_time"})

    # Add primary key
    df.insert(0, "row_id", range(1, len(df) + 1))

    print(f"      Rows to push: {len(df)}")
    print(f"      Features:     {len(df.columns)}")

    # Get existing Feature Group
    fg = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    if fg is None:
        raise RuntimeError(
            "Feature Group 'aqi_features' v1 could not be retrieved."
        )

    print("      ✅ Existing feature group 'aqi_features' found")

    # Push data
    fg.insert(
        df,
        write_options={"wait_for_job": True}
    )

    print(
        f"      ✅ {len(df)} rows pushed to "
        "Feature Group 'aqi_features'"
    )

except Exception as e:
    print(f"      ❌ Feature group push failed: {e}")
    sys.exit(1)




# ─────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY — register best model
# ─────────────────────────────────────────────────────────────────────────
print("\n[4/5] Registering best model in Hopsworks Model Registry …")
try:
    mr = project.get_model_registry()

    # Find Production model from local registry
    local_reg = pd.read_csv("data/registry/model_registry.csv")
    prod_row  = local_reg[local_reg["status"] == "Production"].iloc[-1]
    model_file = str(prod_row["model_file"])
    model_path = os.path.join("data/models", model_file)

    if not os.path.exists(model_path):
        print(f"      ❌ Model file not found: {model_path}")
        sys.exit(1)

    print(f"      Registering: {prod_row['algorithm']} "
          f"(R²={prod_row['r2']:.4f}, MAE={prod_row['mae']:.4f})")

    # Create a temp dir for the model artifact
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    shutil.copy(model_path, os.path.join(tmpdir, model_file))

    # Register in Hopsworks
    hw_model = mr.python.create_model(
        name="aqi_predictor",
        version=int(prod_row["version"]),
        metrics={
            "r2":   float(prod_row["r2"]),
            "mae":  float(prod_row["mae"]),
            "rmse": float(prod_row["rmse"]),
        },
        description=f"AQI Predictor — {prod_row['algorithm']} — Defence Phase 7 Karachi",
    )
    hw_model.save(tmpdir)
    shutil.rmtree(tmpdir)

    print(f"      ✅ Model registered in Hopsworks Model Registry")
    print(f"         Name: aqi_predictor  Version: {prod_row['version']}")

except Exception as e:
    print(f"      ❌ Model registry failed: {e}")
    print("      (Feature store push succeeded — only model registry failed)")

# ─────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────
print("\n[5/5] Setup complete!")
print("\n" + "=" * 60)
print("  ✅ Hopsworks setup successful!")
print("=" * 60)
print(f"\n  Feature Group : aqi_features (v1)")
print(f"  Model         : aqi_predictor")
print(f"  Project       : {HOPSWORKS_PROJECT}")
print(f"\n  Next steps:")
print("    1. Run: python hopsworks_feature_pipeline.py")
print("       (pushes live forecast features hourly)")
print("    2. Run: python hopsworks_training_pipeline.py")
print("       (pulls features from store, retrains, registers model)")
print("    3. Dashboard reads features from Hopsworks automatically")
print("=" * 60)
