"""
merge_dataset.py
Merges daily weather + daily air quality into one clean dataset.
BUG FIX applied: dropna() now runs BEFORE saving to CSV so the output file
is clean, not after (as was done in the original).
"""

import pandas as pd

weather = pd.read_csv("data/processed/daily_weather.csv")
air     = pd.read_csv("data/processed/daily_air_quality.csv")

# Convert dates
weather["date"] = pd.to_datetime(weather["date"])
air["date"]     = pd.to_datetime(air["date"])

# Merge on date (inner join keeps only dates present in both)
merged_df = pd.merge(weather, air, on="date", how="inner")

print("Shape before cleaning:", merged_df.shape)
print("Missing AQI before clean:", merged_df["AQI"].isnull().sum())

# ─── BUG FIX: clean BEFORE saving ───────────────────────────
merged_df = merged_df.dropna(subset=["AQI"]).reset_index(drop=True)
# ─────────────────────────────────────────────────────────────

print("Shape after cleaning :", merged_df.shape)
print(merged_df.head())

merged_df.to_csv(
    "data/processed/historical_karachi_dataset.csv",
    index=False
)

print("\n✅ Historical Karachi dataset created successfully!")
print("Missing values per column:\n", merged_df.isnull().sum())
