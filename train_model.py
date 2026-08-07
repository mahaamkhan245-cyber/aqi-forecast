import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("data/processed/final_features.csv")

# =====================================================
# Features & Target
# =====================================================

X = df.drop(columns=["date", "AQI"])
y = df["AQI"]

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# =====================================================
# Train Model
# =====================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# =====================================================
# Predictions
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# Evaluation
# =====================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 45)
print("MODEL EVALUATION")
print("=" * 45)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# =====================================================
# Create Required Folders
# =====================================================

os.makedirs("data/models", exist_ok=True)
os.makedirs("data/registry", exist_ok=True)

# =====================================================
# Model Registry
# =====================================================

registry_path = "data/registry/model_registry.csv"

if os.path.exists(registry_path):

    registry = pd.read_csv(registry_path)

else:

    registry = pd.DataFrame(columns=[
        "version",
        "model_name",
        "algorithm",
        "mae",
        "rmse",
        "r2",
        "train_date",
        "status",
        "model_file"
    ])

# Archive previous production model

if not registry.empty:
    registry["status"] = "Archived"

# Create version

version = len(registry) + 1

model_filename = f"random_forest_v{version}.pkl"

model_path = os.path.join(
    "data/models",
    model_filename
)

# =====================================================
# Save Versioned Model
# =====================================================

joblib.dump(model, model_path)

# Also save latest model (for current dashboard compatibility)

joblib.dump(
    model,
    "data/models/random_forest_aqi.pkl"
)

# =====================================================
# Update Registry
# =====================================================

new_row = pd.DataFrame([{

    "version": version,

    "model_name": "AQI Predictor",

    "algorithm": "Random Forest",

    "mae": round(mae, 2),

    "rmse": round(rmse, 2),

    "r2": round(r2, 3),

    "train_date": datetime.now().strftime("%Y-%m-%d %H:%M"),

    "status": "Production",

    "model_file": model_filename

}])

registry = pd.concat(
    [registry, new_row],
    ignore_index=True
)

registry.to_csv(
    registry_path,
    index=False
)

# =====================================================
# Summary
# =====================================================

print("\nModel Registry Updated Successfully!\n")

print(registry)

print("\nFeatures Used:")

print(list(X.columns))

print("\nLatest model saved as:")
print("data/models/random_forest_aqi.pkl")

print("\nVersioned model saved as:")
print(model_path)

print("\nTraining Completed Successfully!")