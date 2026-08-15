import subprocess
import sys
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo


# ═══════════════════════════════════════════════════════════════════════════
# UTF-8 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Force this Python process to use UTF-8 for stdout/stderr.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace"
    )


# Make child Python processes use UTF-8 too.
ENV = os.environ.copy()
ENV["PYTHONIOENCODING"] = "utf-8"


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

STATUS_FILE = "data/processed/forecast_status.json"

scripts = [
    "forecast_weather.py",
    "forecast_air_quality.py",
    "forecast_prediction.py",
]


# Make sure the directory exists.
os.makedirs(
    os.path.dirname(STATUS_FILE),
    exist_ok=True
)


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Pearls AQI Predictor — Refreshing Live Forecast")
print("Location: Defence Phase 7, Karachi")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# SAVE FORECAST STATUS
# ═══════════════════════════════════════════════════════════════════════════

def save_status(success, message=""):
    """
    Save the exact time when the forecast pipeline finished.

    Timezone:
        Asia/Karachi
    """

    now = datetime.now()

    status = {
        "success": success,

        # Machine-readable timestamp
        "updated_at": now.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        # Dashboard-friendly date
        "display_date": now.strftime(
            "%d %B %Y"
        ),

        # Dashboard-friendly time
        "display_time": now.strftime(
            "%I:%M %p"
        ),

        # Day name
        "day": now.strftime(
            "%A"
        ),

        # Timezone
        "timezone": "Asia/Karachi",

        # Success/failure message
        "message": message,
    }

    with open(
        STATUS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            status,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"📅 Forecast timestamp: "
        f"{status['display_date']} "
        f"{status['display_time']} PKT"
    )


# ═══════════════════════════════════════════════════════════════════════════
# RUN FORECAST PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

failed = []


for script in scripts:

    print("\n" + "-" * 40)
    print(f"▶ Running {script}...")
    print("-" * 40)

    try:

        # IMPORTANT:
        # Run the CURRENT script from the list.
        #
        # DO NOT use "refresh_data.py" here.
        #
        result = subprocess.run(
            [sys.executable, script],

            capture_output=True,

            text=True,

            encoding="utf-8",

            errors="replace",

            env=ENV
        )


        # ───────────────────────────────────────────────────────────────
        # PRINT CHILD SCRIPT OUTPUT
        # ───────────────────────────────────────────────────────────────

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(
                result.stderr,
                file=sys.stderr
            )


        # ───────────────────────────────────────────────────────────────
        # CHECK EXIT CODE
        # ───────────────────────────────────────────────────────────────

        if result.returncode != 0:

            print(
                f"\n❌ {script} FAILED "
                f"(exit code {result.returncode})"
            )

            failed.append(script)

        else:

            print(
                f"\n✅ {script} completed successfully"
            )


    except Exception as e:

        print(
            f"\n❌ Could not run {script}: {e}",
            file=sys.stderr
        )

        failed.append(script)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE RESULT
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# FAILURE
# ═══════════════════════════════════════════════════════════════════════════

if failed:

    print(
        f"❌ Pipeline FAILED. "
        f"{len(failed)} script(s) had errors:"
    )

    for failed_script in failed:

        print(
            f"   • {failed_script}"
        )


    # Save failed refresh status.
    save_status(
        False,
        f"Failed scripts: {', '.join(failed)}"
    )


    print("=" * 60)

    # Important:
    # GitHub Actions / Streamlit can detect failure through exit code 1.
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# SUCCESS
# ═══════════════════════════════════════════════════════════════════════════

else:

    # Save successful refresh timestamp.
    save_status(
        True,
        "All forecast scripts completed successfully."
    )


    print(
        "✅ All forecast scripts completed successfully!"
    )

    print("=" * 60)

    # Successful pipeline.
    sys.exit(0)