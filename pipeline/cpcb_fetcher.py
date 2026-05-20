# pipeline/cpcb_fetcher.py
"""
Fetches real-time and historical PM2.5/PM10 data from OpenAQ v3 API.

OpenAQ aggregates data directly from CPCB monitoring stations across India,
so the readings are identical to CPCB — just via a much more reliable endpoint.

API docs: https://docs.openaq.org/

Key endpoints used:
  GET /v3/locations              → discover station IDs for a city
  GET /v3/locations/{id}/latest  → latest reading per station
  GET /v3/sensors/{id}/hours     → hourly historical data (for model training)
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CITIES, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OpenAQ] %(message)s")
log = logging.getLogger(__name__)

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")
BASE_URL       = "https://api.openaq.org/v3"

def _get_headers():
    return {"X-API-Key": OPENAQ_API_KEY} if OPENAQ_API_KEY else {}

# Hardcoded OpenAQ location IDs for CPCB stations in our 3 cities.
# Run discover_station_ids() once to find/update these for your region.
CITY_STATION_IDS = {
    "Delhi":     [8173, 8174, 8175, 8176, 8177],
    "Mumbai":    [8190, 8191, 8192],
    "Hyderabad": [8323, 8324, 8325],
}


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_cpcb_live() -> pd.DataFrame:
    """
    Fetches the latest reading from every configured station.
    Returns: timestamp | city | station | PM2_5 | PM10 | AQI
    """
    if not OPENAQ_API_KEY:
        log.warning("No OPENAQ_API_KEY set — loading cached data.")
        return _load_cached("all_cities")

    rows = []
    for city, station_ids in CITY_STATION_IDS.items():
        for loc_id in station_ids:
            record = _fetch_latest_for_location(city, loc_id)
            if record:
                rows.append(record)

    if not rows:
        log.warning("No live data fetched — falling back to cache.")
        return _load_cached("all_cities")

    df = pd.DataFrame(rows)
    df["AQI"] = df["PM2_5"].apply(_pm25_to_aqi)
    _save_cache(df, "all_cities")
    return df


def fetch_historical(days: int = 7) -> pd.DataFrame:
    """
    Fetches hourly historical data for the past `days` days.
    This is what the Prophet model trains on — real patterns over time.
    Returns the same schema as fetch_cpcb_live() but many rows per station.
    """
    if not OPENAQ_API_KEY:
        log.warning("No OPENAQ_API_KEY set — loading cached historical data.")
        return _load_cached("historical")

    date_from = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_rows = []
    for city, station_ids in CITY_STATION_IDS.items():
        log.info(f"Fetching {days}-day history for {city}...")
        for loc_id in station_ids:
            rows = _fetch_hourly_history(city, loc_id, date_from, date_to)
            all_rows.extend(rows)

    if not all_rows:
        log.warning("No historical data returned — falling back to cache.")
        return _load_cached("historical")

    df = pd.DataFrame(all_rows)
    df["AQI"] = df["PM2_5"].apply(_pm25_to_aqi)
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    _save_cache(df, "historical")
    log.info(f"Historical fetch complete: {len(df)} rows.")
    return df


def discover_station_ids(city: str, limit: int = 20) -> pd.DataFrame:
    """
    Helper: searches OpenAQ for monitoring stations in a city.
    Run this ONCE manually to find the correct IDs for CITY_STATION_IDS above.

    Usage from terminal:
        python -c "
        from pipeline.cpcb_fetcher import discover_station_ids
        print(discover_station_ids('Delhi'))
        "
    """
    url    = f"{BASE_URL}/locations"
    params = {"country_id": "IN", "city": city, "limit": limit}
    try:
        resp = requests.get(url, headers=_get_headers(), params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        rows = []
        for r in results:
            rows.append({
                "id":      r["id"],
                "name":    r.get("name", ""),
                "city":    city,
                "lat":     r.get("coordinates", {}).get("latitude"),
                "lon":     r.get("coordinates", {}).get("longitude"),
                "sensors": [s["parameter"]["name"] for s in r.get("sensors", [])],
            })
        return pd.DataFrame(rows)
    except Exception as e:
        log.error(f"Discovery failed for {city}: {e}")
        return pd.DataFrame()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_latest_for_location(city: str, loc_id: int) -> dict | None:
    url = f"{BASE_URL}/locations/{loc_id}/latest"
    try:
        resp = requests.get(url, headers=_get_headers(), timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None

        record = {"city": city, "station": str(loc_id)}
        for sensor in results:
            param = sensor.get("parameter", {}).get("name", "").lower()
            value = sensor.get("value")
            ts    = sensor.get("datetime", {}).get("utc")
            if param == "pm25" and value is not None:
                record["PM2_5"]     = float(value)
                record["timestamp"] = pd.to_datetime(ts, utc=True)
            elif param == "pm10" and value is not None:
                record["PM10"] = float(value)

        if "PM2_5" not in record and "PM10" not in record:
            return None
        record.setdefault("PM2_5",     float("nan"))
        record.setdefault("PM10",      float("nan"))
        record.setdefault("timestamp", datetime.now(timezone.utc))
        return record
    except Exception as e:
        log.warning(f"Latest fetch failed for location {loc_id}: {e}")
        return None


def _fetch_hourly_history(city: str, loc_id: int, date_from: str, date_to: str) -> list:
    # Step 1: get sensors for this location
    try:
        resp = requests.get(f"{BASE_URL}/locations/{loc_id}", headers=_get_headers(), timeout=15)
        resp.raise_for_status()
        location_data = resp.json().get("results", [{}])[0]
        sensors       = location_data.get("sensors", [])
        station_name  = location_data.get("name", str(loc_id))
    except Exception as e:
        log.warning(f"Could not get sensors for location {loc_id}: {e}")
        return []

    pm25_series: dict = {}
    pm10_series: dict = {}

    for sensor in sensors:
        param     = sensor.get("parameter", {}).get("name", "").lower()
        sensor_id = sensor.get("id")
        if param not in ("pm25", "pm10") or not sensor_id:
            continue
        try:
            hrs = requests.get(
                f"{BASE_URL}/sensors/{sensor_id}/hours",
                headers=_get_headers(),
                params={"datetime_from": date_from, "datetime_to": date_to, "limit": 500},
                timeout=20,
            )
            hrs.raise_for_status()
            for entry in hrs.json().get("results", []):
                ts  = entry.get("period", {}).get("datetimeFrom", {}).get("utc")
                val = entry.get("summary", {}).get("mean")
                if ts and val is not None:
                    (pm25_series if param == "pm25" else pm10_series)[ts] = float(val)
        except Exception as e:
            log.warning(f"History fetch failed for sensor {sensor_id}: {e}")

    all_ts = set(pm25_series) | set(pm10_series)
    rows = []
    for ts in sorted(all_ts):
        rows.append({
            "timestamp": pd.to_datetime(ts, utc=True),
            "city":      city,
            "station":   station_name,
            "PM2_5":     pm25_series.get(ts, float("nan")),
            "PM10":      pm10_series.get(ts, float("nan")),
        })
    return rows


def _pm25_to_aqi(pm25: float) -> float:
    """CPCB National AQI breakpoints for PM2.5 (2014 standard)."""
    if pd.isna(pm25):
        return float("nan")
    breakpoints = [
        (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (250, 500, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo, 1)
    return 500.0


def _save_cache(df: pd.DataFrame, label: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    df.to_csv(os.path.join(DATA_DIR, f"cpcb_{label}_{ts}.csv"), index=False)
    df.to_csv(os.path.join(DATA_DIR, f"cpcb_{label}_latest.csv"), index=False)
    log.info(f"Cache saved → cpcb_{label}_latest.csv")


def _load_cached(label: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"cpcb_{label}_latest.csv")
    if os.path.exists(path):
        log.info(f"Loaded cache: {path}")
        return pd.read_csv(path, parse_dates=["timestamp"])
    log.warning(f"No cache at {path} — returning empty DataFrame.")
    return pd.DataFrame()


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Discovering stations ===")
    for city in ["Delhi", "Mumbai", "Hyderabad"]:
        stations = discover_station_ids(city, limit=10)
        if not stations.empty:
            print(f"\n{city}:")
            print(stations[["id", "name", "sensors"]].to_string(index=False))

    print("\n=== Live readings ===")
    live = fetch_cpcb_live()
    print(live if not live.empty else "No live data.")

    print("\n=== 7-day historical ===")
    hist = fetch_historical(days=7)
    print(f"Rows: {len(hist)}")