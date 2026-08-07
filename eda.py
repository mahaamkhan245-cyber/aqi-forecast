import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Create EDA folder
# -----------------------------
os.makedirs("data/eda", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/historical_karachi_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())
#AQI Distribution
plt.figure(figsize=(8,5))

plt.hist(df["AQI"], bins=30)

plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("data/eda/aqi_distribution.png")

plt.show()
#AQI Trend Over Time

plt.figure(figsize=(12,5))

plt.plot(df["date"], df["AQI"])

plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")

plt.tight_layout()

plt.savefig("data/eda/aqi_trend.png")

plt.show()
#Temperature Trend
plt.figure(figsize=(12,5))

plt.plot(df["date"], df["temperature"])

plt.title("Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")

plt.tight_layout()

plt.savefig("data/eda/temperature_trend.png")

plt.show()
#PM2.5 vs AQI
plt.figure(figsize=(6,6))

plt.scatter(df["pm2_5"], df["AQI"])

plt.title("PM2.5 vs AQI")
plt.xlabel("PM2.5")
plt.ylabel("AQI")

plt.tight_layout()

plt.savefig("data/eda/pm25_vs_aqi.png")

plt.show()
#Correlation Matrix
correlation = df.drop(columns=["date"]).corr()

print("\nCorrelation Matrix:\n")
print(correlation)

plt.figure(figsize=(10,8))

plt.imshow(correlation, aspect="auto")

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("data/eda/correlation_heatmap.png")

plt.show()
# -----------------------------
# AQI Boxplot (Outlier Detection)
# -----------------------------
plt.figure(figsize=(8,5))

plt.boxplot(
    df["AQI"].dropna(),
    vert=True,
    patch_artist=True
)

plt.title("AQI Boxplot")
plt.ylabel("AQI")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig("data/eda/aqi_boxplot.png")

plt.show()