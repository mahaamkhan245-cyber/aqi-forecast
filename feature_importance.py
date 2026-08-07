import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/final_features.csv")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("data/models/random_forest_aqi.pkl")

# -----------------------------
# Prepare Features
# -----------------------------
X = df.drop(columns=["date", "AQI"])

# -----------------------------
# Get Feature Importance
# -----------------------------
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort descending
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 8))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.gca().invert_yaxis()   # Highest importance at the top

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()
plt.savefig("data/eda/feature_importance.png")
plt.show()