def get_aqi_category(aqi):

    if aqi <= 50:
        return "🟢 Good", "Air quality is satisfactory."

    elif aqi <= 100:
        return "🟡 Moderate", "Air quality is acceptable."

    elif aqi <= 150:
        return (
            "🟠 Unhealthy for Sensitive Groups",
            "Sensitive people should limit prolonged outdoor activity."
        )

    elif aqi <= 200:
        return (
            "🔴 Unhealthy",
            "Everyone may begin to experience health effects."
        )

    elif aqi <= 300:
        return (
            "🟣 Very Unhealthy",
            "Health alert: everyone may experience serious effects."
        )

    else:
        return (
            "⚫ Hazardous",
            "Avoid all outdoor activities."
        )