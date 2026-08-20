import pandas as pd

# Load merged dataset
df = pd.read_csv("data/processed/historical_karachi_dataset.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# -------------------------
# Time Features
# -------------------------

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear

# Weekend (Saturday=5, Sunday=6)
df["weekend"] = (df["day_of_week"] >= 5).astype(int)

# -------------------------
# Lag Features
# -------------------------

df["AQI_lag_1"] = df["AQI"].shift(1)
df["AQI_lag_2"] = df["AQI"].shift(2)
df["AQI_lag_3"] = df["AQI"].shift(3)

# -------------------------
# Rolling Features
# -------------------------

df["AQI_3day_mean"] = df["AQI"].rolling(3).mean()
df["AQI_7day_mean"] = df["AQI"].rolling(7).mean()

# -------------------------
# Weather Features
# -------------------------

df["rain_flag"] = (df["rain"] > 0).astype(int)
df["temp_humidity"] = df["temperature"] * df["humidity"]

# Remove rows with NaN created by lag/rolling features
df = df.dropna().reset_index(drop=True)

print(df.head())
print("\nShape:", df.shape)

# Save engineered dataset
df.to_csv(
    "data/processed/final_features.csv",
    index=False
)

print("✅ Feature engineering completed successfully!")