import joblib
import pandas as pd

# -----------------------------------
# Load Saved Model
# -----------------------------------

model = joblib.load("data/models/random_forest_aqi.pkl")

print("✅ Model loaded successfully!")

# -----------------------------------
# Load Feature Dataset
# -----------------------------------

df = pd.read_csv("data/processed/final_features.csv")

# -----------------------------------
# Prepare Features & Target
# -----------------------------------

X = df.drop(columns=["date", "AQI"])
y = df["AQI"]

print(f"\nDataset Shape: {df.shape}")

# -----------------------------------
# Select One Sample
# -----------------------------------

sample = X.iloc[[0]]

actual_aqi = y.iloc[0]

# -----------------------------------
# Predict
# -----------------------------------

predicted_aqi = model.predict(sample)[0]

# -----------------------------------
# Results
# -----------------------------------

print("\n" + "=" * 40)
print("AQI PREDICTION")
print("=" * 40)

print(f"Actual AQI     : {actual_aqi:.2f}")
print(f"Predicted AQI  : {predicted_aqi:.2f}")
print(f"Absolute Error : {abs(actual_aqi - predicted_aqi):.2f}")