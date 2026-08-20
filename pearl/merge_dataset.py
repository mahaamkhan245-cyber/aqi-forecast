import pandas as pd

weather = pd.read_csv("data/processed/daily_weather.csv")
air = pd.read_csv("data/processed/daily_air_quality.csv")

# Convert dates
weather["date"] = pd.to_datetime(weather["date"])
air["date"] = pd.to_datetime(air["date"])

# Merge on date
merged_df = pd.merge(
    weather,
    air,
    on="date",
    how="inner"
)

print(merged_df.head())
print("\nShape:", merged_df.shape)

merged_df.to_csv(
    "data/processed/historical_karachi_dataset.csv",
    index=False
)

print("✅ Historical Karachi dataset created successfully!")
# Remove rows where AQI is missing
merged_df = merged_df.dropna(subset=["AQI"])

# Reset index
merged_df = merged_df.reset_index(drop=True)

print("\nDataset after cleaning:")
print(merged_df.head())
print("Shape:", merged_df.shape)
print(merged_df.isnull().sum())