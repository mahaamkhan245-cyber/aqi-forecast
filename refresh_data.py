import subprocess
import sys

print("=" * 70)
print("Refreshing Live AQI Data...")
print("=" * 70)

scripts = [

    "forecast_weather.py",

    "forecast_air_quality.py",

    "forecast_prediction.py"

]

for script in scripts:

    print(f"\nRunning {script}...\n")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:

        print(result.stdout)

    else:

        print(result.stderr)

print("\n")
print("=" * 70)
print("Live Forecast Updated Successfully!")
print("=" * 70)