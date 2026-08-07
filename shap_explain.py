import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Create EDA Folder
# -----------------------------
os.makedirs("data/eda", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/final_features.csv")

X = df.drop(columns=["date", "AQI"])

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("data/models/random_forest_aqi.pkl")

# -----------------------------
# Create SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")

shap_values = explainer(X)

print("Done!")
#SHAP Summary Plot

plt.figure()

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig("data/eda/shap_summary.png")

plt.show()

#Step 5 — SHAP Bar Plot


plt.figure()

shap.plots.bar(
    shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X.values,
        feature_names=X.columns
    ),
    show=False
)

plt.tight_layout()

plt.savefig("data/eda/shap_bar.png")

plt.show()
#waterfall
plt.figure(figsize=(8,6))

shap.plots.waterfall(
    shap_values[0],
    show=False
)

plt.tight_layout()
plt.savefig("data/eda/shap_waterfall.png")
plt.show()