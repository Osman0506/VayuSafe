# pipeline/weather_fetcher.py
"""
Fetches current weather parameters from OpenWeatherMap for each monitored city.

Why these parameters?
  - Wind speed:   High wind disperses dust → lower AQI; calm wind → dust accumulates
  - Humidity:     High humidity suppresses dust (droplets weigh particles down)
  - Temperature:  Affects atmospheric mixing layer height → dispersion capacity
  - Pressure:     Low pressure = poor ventilation → dust trapping

These features are critical inputs to the forecasting model because
construction dust spikes are strongly correlated with calm + dry + hot conditions.
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OWM_API_KEY, OWM_BASE_URL, CITIES, DATA_DIR

import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OWM] %(message)s")
log = logging.getLogger(__name__)


def fetch_weather_all_cities() -> pd.DataFrame:
    """
    Fetches current weather for all configured cities.

    Returns a DataFrame:
        timestamp | city | temp_c | humidity_pct | wind_speed_ms |
        wind_dir_deg | pressure_hpa | weather_desc | visibility_m
    """
    if not OWM_API_KEY:
        log.warning("No OWM_API_KEY — loading cached weather data.")
        return _load_weather_cache()

    rows = []
    for city, meta in CITIES.items():
        row = fetch_weather_city(city, meta["lat"], meta["lon"])
        if row:
            rows.append(row)

    if not rows:
        log.warning("No weather data fetched — loading cache.")
        return _load_weather_cache()

    df = pd.DataFrame(rows)
    _save_weather_cache(df)
    return df


def fetch_weather_city(city: str, lat: float, lon: float) -> dict | None:
    """
    Fetches weather for a single lat/lon point.
    Uses OWM /weather (current conditions) endpoint.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OWM_API_KEY,
        "units": "metric",   # °C, m/s
    }

    try:
        resp = requests.get(OWM_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        log.info(f"Weather fetched for {city}: {data['weather'][0]['description']}")
        return _parse_owm_response(city, data)

    except requests.RequestException as e:
        log.error(f"OWM fetch failed for {city}: {e}")
        return None


def _parse_owm_response(city: str, data: dict) -> dict:
    """
    Extracts the fields we need from the raw OWM JSON response.

    Raw OWM response structure:
    {
      "main": { "temp": 32.1, "humidity": 45, "pressure": 1003 },
      "wind": { "speed": 2.3, "deg": 180 },
      "visibility": 8000,
      "weather": [{ "description": "haze" }],
      "dt": 1717228800   ← Unix timestamp (UTC)
    }
    """
    main    = data.get("main", {})
    wind    = data.get("wind", {})
    weather = data.get("weather", [{}])[0]

    return {
        "timestamp":       datetime.fromtimestamp(data["dt"], tz=timezone.utc),
        "city":            city,
        "temp_c":          main.get("temp"),
        "feels_like_c":    main.get("feels_like"),
        "humidity_pct":    main.get("humidity"),
        "pressure_hpa":    main.get("pressure"),
        "wind_speed_ms":   wind.get("speed"),
        "wind_dir_deg":    wind.get("deg"),
        "wind_gust_ms":    wind.get("gust"),
        "visibility_m":    data.get("visibility"),
        "weather_desc":    weather.get("description", ""),
        "weather_main":    weather.get("main", ""),
        # Derived feature: dust risk score (calm + dry + hot = high risk)
        "dust_risk_score": _compute_dust_risk(
            wind_speed=wind.get("speed", 5),
            humidity=main.get("humidity", 50),
            temp=main.get("temp", 25),
        ),
    }


def _compute_dust_risk(wind_speed: float, humidity: float, temp: float) -> float:
    """
    Heuristic dust risk score (0–1) based on meteorological conditions.

    Logic (adapted from USEPA dust dispersion guidelines):
      - Low wind (< 2 m/s): dust stays suspended → higher risk
      - Low humidity (< 30%): no moisture suppression → higher risk
      - High temp (> 35°C):  dry soil + convective mixing → higher risk

    Score is a simple weighted average of these three factors.
    """
    # Wind: risk is inverse of wind speed (capped at 10 m/s)
    wind_risk = max(0, 1 - min(wind_speed, 10) / 10)

    # Humidity: risk decreases above 50% humidity
    hum_risk = max(0, 1 - min(humidity, 100) / 100)

    # Temperature: risk increases above 25°C
    temp_risk = min(max((temp - 25) / 20, 0), 1)

    # Weighted combination (wind matters most for construction dust)
    score = 0.5 * wind_risk + 0.35 * hum_risk + 0.15 * temp_risk
    return round(score, 3)


def _save_weather_cache(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    df.to_csv(os.path.join(DATA_DIR, f"weather_{ts}.csv"), index=False)
    df.to_csv(os.path.join(DATA_DIR, "weather_latest.csv"), index=False)
    log.info("Weather cache saved.")


def _load_weather_cache() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "weather_latest.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["timestamp"])
    log.warning("No weather cache found.")
    return pd.DataFrame()


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch_weather_all_cities()
    if df.empty:
        print("No weather data. Check OWM_API_KEY.")
    else:
        print(f"\n✅ Weather data for {df['city'].nunique()} cities:")
        print(df[["city", "temp_c", "humidity_pct", "wind_speed_ms",
                   "dust_risk_score", "weather_desc"]].to_string(index=False))