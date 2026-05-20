#!/usr/bin/env python3
# run_pipeline.py
"""
Orchestrator: runs the full VayuSafe data pipeline end-to-end.

Usage:
    python run_pipeline.py              # fetch live data
    python run_pipeline.py --demo       # run with simulated data (no API keys needed)

Schedule this with cron (every hour) for live monitoring:
    0 * * * * /path/to/venv/bin/python /path/to/vayusafe/run_pipeline.py
"""

import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PIPELINE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_demo_pipeline() -> pd.DataFrame:
    """
    Generates realistic synthetic data for demo/testing purposes.
    Simulates 7 days of hourly AQI + weather readings for 3 cities.

    WHY SYNTHETIC?
    CPCB API access can sometimes be slow to approve.
    This demo mode lets you build/test the full stack (model + dashboard + map)
    while waiting for API credentials.
    """
    log.info("Running in DEMO mode — generating synthetic data.")

    from config import CITIES, AQI_LIMITS

    cities = list(CITIES.keys())
    hours  = pd.date_range(
        end=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
        periods=7 * 24,   # 7 days of hourly data
        freq="h",
        tz="UTC",
    )

    # City-specific baseline PM2.5 levels (realistic for Indian cities)
    baselines = {"Delhi": 95, "Mumbai": 55, "Hyderabad": 65}

    rows = []
    for city in cities:
        base = baselines[city]
        np.random.seed(42 + cities.index(city))

        for i, ts in enumerate(hours):
            hour_of_day = ts.hour

            # Diurnal pattern: peaks at 8am and 8pm (traffic + construction activity)
            diurnal = (
                20 * np.sin(np.pi * (hour_of_day - 6) / 12) +
                10 * np.sin(2 * np.pi * hour_of_day / 24)
            )
            # Random noise + occasional spike (simulates dust event)
            noise   = np.random.normal(0, 8)
            spike   = 40 * (1 if np.random.random() < 0.05 else 0)

            pm25 = max(5, base + diurnal + noise + spike)
            pm10 = pm25 * np.random.uniform(1.8, 2.2)   # PM10 ~ 2× PM2.5
            aqi  = min(500, pm25 * 2.0)                  # rough AQI

            # Weather: vary by time of day
            temp     = 28 + 8 * np.sin(np.pi * (hour_of_day - 6) / 12) + np.random.normal(0, 1)
            humidity = 45 - 15 * np.sin(np.pi * (hour_of_day - 6) / 12) + np.random.normal(0, 3)
            wind     = max(0.5, np.random.exponential(2.5))

            rows.append({
                "ds":             ts,
                "city":           city,
                "PM2_5":          round(pm25, 1),
                "PM10":           round(pm10, 1),
                "AQI":            round(aqi, 1),
                "n_stations":     3,
                "temp_c":         round(temp, 1),
                "humidity_pct":   round(max(10, min(95, humidity)), 1),
                "wind_speed_ms":  round(wind, 2),
                "wind_dir_deg":   np.random.randint(0, 360),
                "pressure_hpa":   1008 + np.random.normal(0, 3),
                "dust_risk_score": round(max(0, 1 - wind / 10) * 0.5 +
                                         max(0, 1 - humidity / 100) * 0.35 +
                                         max(0, (temp - 25) / 20) * 0.15, 3),
                "weather_desc":   "haze" if pm25 > 80 else "clear sky",
            })

    df = pd.DataFrame(rows)

    # Add lag/roll features
    from pipeline.preprocessor import merge_and_prepare
    # Since data is already merged, add features manually
    for city in cities:
        mask = df["city"] == city
        s    = df.loc[mask].sort_values("ds")
        df.loc[s.index, "PM2_5_lag1h"]  = s["PM2_5"].shift(1).values
        df.loc[s.index, "PM10_lag1h"]   = s["PM10"].shift(1).values
        df.loc[s.index, "PM2_5_roll3h"] = s["PM2_5"].rolling(3, min_periods=1).mean().values
        df.loc[s.index, "PM2_5_roll24h"]= s["PM2_5"].rolling(24, min_periods=3).mean().values
        df.loc[s.index, "PM2_5_delta"]  = s["PM2_5"].diff(1).values

    # Alert flags
    pm25_limit = AQI_LIMITS["PM2.5_ug_m3"]
    pm10_limit = AQI_LIMITS["PM10_ug_m3"]
    from config import ALERT_BUFFER_PCT

    df["alert_pm25"] = df["PM2_5"].apply(
        lambda v: "RED" if v >= pm25_limit
        else ("AMBER" if v >= pm25_limit * ALERT_BUFFER_PCT else "GREEN")
    )
    df["alert_pm10"] = df["PM10"].apply(
        lambda v: "RED" if v >= pm10_limit
        else ("AMBER" if v >= pm10_limit * ALERT_BUFFER_PCT else "GREEN")
    )
    priority = {"GREEN": 0, "AMBER": 1, "RED": 2}
    df["alert"] = df.apply(
        lambda r: max([r["alert_pm25"], r["alert_pm10"]],
                      key=lambda x: priority[x]),
        axis=1,
    )

    from pipeline.preprocessor import save_merged
    save_merged(df)
    log.info(f"Demo data saved. Shape: {df.shape}")
    return df


def run_live_pipeline() -> pd.DataFrame:
    """Runs the full live pipeline: CPCB → OWM → merge → save."""
    from pipeline.cpcb_fetcher import fetch_cpcb_live
    from pipeline.weather_fetcher import fetch_weather_all_cities
    from pipeline.preprocessor import merge_and_prepare, save_merged

    log.info("Step 1/3: Fetching CPCB AQI data...")
    aqi_df = fetch_cpcb_live()

    log.info("Step 2/3: Fetching OpenWeatherMap data...")
    wx_df = fetch_weather_all_cities()

    log.info("Step 3/3: Merging and preprocessing...")
    merged = merge_and_prepare(aqi_df, wx_df)

    if not merged.empty:
        save_merged(merged)
        log.info(f"✅ Pipeline complete. {len(merged)} rows saved.")
    else:
        log.warning("Pipeline produced no data. Check API keys.")

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VayuSafe Data Pipeline")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data instead of live API calls")
    args = parser.parse_args()

    if args.demo:
        df = run_demo_pipeline()
    else:
        df = run_live_pipeline()

    if not df.empty:
        print("\n── Summary ─────────────────────────────────────────")
        print(f"  Rows: {len(df)} | Cities: {df['city'].nunique()}")
        print(f"  Date range: {df['ds'].min()} → {df['ds'].max()}")
        print(f"\n  Latest readings by city:")
        latest = df.groupby("city").last().reset_index()
        print(latest[["city", "PM2_5", "PM10", "AQI", "alert"]].to_string(index=False))