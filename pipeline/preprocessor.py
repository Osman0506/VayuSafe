# pipeline/preprocessor.py
"""
Merges CPCB AQI data + OpenWeatherMap weather into a unified,
model-ready time-series DataFrame.

Why a separate preprocessor?
  - CPCB data is station-level (multiple stations per city)
  - Weather data is city-level (one reading per city)
  - We need to aggregate stations → city level, then join on (city, timestamp)
  - We also generate lag features and rolling averages here, which Prophet
    can use as extra regressors and LSTM can use as input channels.

Output schema (one row per city per hour):
    ds          | city | PM2_5 | PM10 | AQI | temp_c | humidity_pct |
    wind_speed_ms | wind_dir_deg | dust_risk_score | PM2_5_lag1h |
    PM2_5_roll3h | PM10_roll3h | alert_flag
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import sys, os

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import AQI_LIMITS, ALERT_BUFFER_PCT, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PREP] %(message)s")
log = logging.getLogger(__name__)


def merge_and_prepare(
    aqi_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Returns a clean, feature-rich DataFrame
    suitable for both Prophet and LSTM training.

    Steps:
      1. Aggregate station-level AQI → city-level hourly means
      2. Floor timestamps to the hour for alignment
      3. Left-join weather onto AQI by (city, hour)
      4. Forward-fill small gaps (≤2 hours) in AQI readings
      5. Add lag features and rolling statistics
      6. Add alert flag column
      7. Drop rows with no PM2.5 or PM10
    """
    if aqi_df.empty:
        log.warning("Empty AQI DataFrame passed to merge_and_prepare.")
        return pd.DataFrame()

    # ── Step 1: City-level hourly aggregation ─────────────────────────────────
    aqi = aqi_df.copy()
    aqi["timestamp"] = pd.to_datetime(aqi["timestamp"])
    aqi["hour"] = aqi["timestamp"].dt.floor("h")

    city_hourly = (
        aqi.groupby(["hour", "city"])
        .agg(
            PM2_5=("PM2_5", "mean"),
            PM10=("PM10", "mean"),
            AQI=("AQI", "mean"),
            n_stations=("station", "nunique"),
        )
        .reset_index()
        .rename(columns={"hour": "ds"})
    )
    log.info(f"Aggregated to {len(city_hourly)} city-hour rows.")

    # ── Step 2: Prepare weather ───────────────────────────────────────────────
    if not weather_df.empty:
        wx = weather_df.copy()
        wx["timestamp"] = pd.to_datetime(wx["timestamp"])
        wx["ds"] = wx["timestamp"].dt.floor("h")

        wx_cols = [
            "ds", "city", "temp_c", "feels_like_c", "humidity_pct",
            "pressure_hpa", "wind_speed_ms", "wind_dir_deg",
            "wind_gust_ms", "visibility_m", "dust_risk_score",
            "weather_desc", "weather_main",
        ]
        wx = wx[[c for c in wx_cols if c in wx.columns]]

        # ── Step 3: Join ──────────────────────────────────────────────────────
        merged = city_hourly.merge(wx, on=["ds", "city"], how="left")
        log.info("Merged AQI + weather.")
    else:
        merged = city_hourly
        log.warning("No weather data — proceeding without weather features.")

    # ── Step 4: Forward-fill weather gaps ────────────────────────────────────
    weather_feature_cols = [
        "temp_c", "humidity_pct", "wind_speed_ms", "dust_risk_score",
        "pressure_hpa",
    ]
    for col in weather_feature_cols:
        if col in merged.columns:
            # Fill per city
            merged[col] = merged.groupby("city")[col].transform(
                lambda s: s.ffill(limit=2)
            )

    # ── Step 5: Lag and rolling features ─────────────────────────────────────
    # These help the model learn autocorrelation in AQI time series
    for city in merged["city"].unique():
        mask = merged["city"] == city
        city_data = merged.loc[mask].sort_values("ds")

        # 1-hour lag
        merged.loc[city_data.index, "PM2_5_lag1h"] = (
            city_data["PM2_5"].shift(1).values
        )
        merged.loc[city_data.index, "PM10_lag1h"] = (
            city_data["PM10"].shift(1).values
        )

        # 3-hour rolling mean
        merged.loc[city_data.index, "PM2_5_roll3h"] = (
            city_data["PM2_5"].rolling(3, min_periods=1).mean().values
        )
        merged.loc[city_data.index, "PM10_roll3h"] = (
            city_data["PM10"].rolling(3, min_periods=1).mean().values
        )

        # 24-hour rolling mean (daily baseline)
        merged.loc[city_data.index, "PM2_5_roll24h"] = (
            city_data["PM2_5"].rolling(24, min_periods=3).mean().values
        )

        # Rate of change (hour-over-hour delta)
        merged.loc[city_data.index, "PM2_5_delta"] = (
            city_data["PM2_5"].diff(1).values
        )

    # ── Step 6: Alert flag ────────────────────────────────────────────────────
    # Pre-alert: PM2.5 ≥ 85% of CPCB construction limit → amber alert
    # Breach:    PM2.5 ≥ CPCB limit → red alert
    pm25_limit = AQI_LIMITS["PM2.5_ug_m3"]
    pm10_limit = AQI_LIMITS["PM10_ug_m3"]

    merged["alert_pm25"] = merged["PM2_5"].apply(
        lambda v: "RED" if v >= pm25_limit
        else ("AMBER" if v >= pm25_limit * ALERT_BUFFER_PCT else "GREEN")
    )
    merged["alert_pm10"] = merged["PM10"].apply(
        lambda v: "RED" if v >= pm10_limit
        else ("AMBER" if v >= pm10_limit * ALERT_BUFFER_PCT else "GREEN")
    )
    # Overall alert = worst of PM2.5 / PM10 alerts
    priority = {"GREEN": 0, "AMBER": 1, "RED": 2}
    merged["alert"] = merged.apply(
        lambda r: max([r["alert_pm25"], r["alert_pm10"]],
                      key=lambda x: priority[x]),
        axis=1,
    )

    # ── Step 7: Drop rows with no target variables ────────────────────────────
    merged = merged.dropna(subset=["PM2_5", "PM10"])
    merged = merged.sort_values(["city", "ds"]).reset_index(drop=True)

    log.info(f"Preprocessing complete. Final shape: {merged.shape}")
    return merged


def prepare_prophet_input(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Formats the merged DataFrame into Prophet's required format for a single city:
        ds | y | [regressor cols]

    Prophet requires:
      - 'ds': datetime column
      - 'y': target variable (we use PM2.5)
      - any extra regressors must be added via model.add_regressor() before fitting
    """
    city_df = df[df["city"] == city].copy()
    city_df = city_df.sort_values("ds").reset_index(drop=True)

    prophet_df = city_df.rename(columns={"PM2_5": "y"})

    # Select cols Prophet can use
    keep = [
        "ds", "y", "PM10", "temp_c", "humidity_pct",
        "wind_speed_ms", "dust_risk_score",
        "PM2_5_lag1h", "PM2_5_roll3h", "PM2_5_delta",
    ]
    prophet_df = prophet_df[[c for c in keep if c in prophet_df.columns]]
    return prophet_df.dropna(subset=["y"])


def get_latest_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the most recent row per city — used for the dashboard's
    "current conditions" panel.
    """
    return (
        df.sort_values("ds")
        .groupby("city")
        .last()
        .reset_index()
    )


def save_merged(df: pd.DataFrame, tag: str = "merged"):
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    df.to_csv(os.path.join(DATA_DIR, f"{tag}_{ts}.csv"), index=False)
    df.to_csv(os.path.join(DATA_DIR, f"{tag}_latest.csv"), index=False)
    log.info(f"Saved merged data → {DATA_DIR}/{tag}_latest.csv")


def load_merged(tag: str = "merged") -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{tag}_latest.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["ds"])
    return pd.DataFrame()


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate minimal test data
    from pipeline.cpcb_fetcher import fetch_cpcb_live
    from pipeline.weather_fetcher import fetch_weather_all_cities

    aqi_df     = fetch_cpcb_live()
    weather_df = fetch_weather_all_cities()

    merged = merge_and_prepare(aqi_df, weather_df)
    save_merged(merged)

    if not merged.empty:
        print("\n✅ Merged DataFrame (last 10 rows):")
        print(merged.tail(10)[["ds", "city", "PM2_5", "PM10", "AQI",
                                "dust_risk_score", "alert"]].to_string(index=False))
        print(f"\nShape: {merged.shape}")
        print(f"\nAlert counts:\n{merged['alert'].value_counts()}")