from api_client import health, get_forecast, get_features

print("\n=== HEALTH ===")
print(health())

print("\n=== FORECAST ===")
print(get_forecast())

print("\n=== FEATURES ===")
print(get_features())