# model/forecaster.py
"""
VayuSafe Forecasting Model — scikit-learn based PM2.5 predictor.

Features used:
  - Hour of day, day of week (captures diurnal + weekly patterns)
  - PM2.5 lag (1h, 3h rolling mean, 24h rolling mean, delta)
  - Weather: wind speed, humidity, temp, dust risk score
"""

import os
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    CITIES, AQI_LIMITS, ALERT_BUFFER_PCT,
    FORECAST_HOURS_AHEAD, MODEL_DIR
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [MODEL] %(message)s")

PM25_LIMIT = AQI_LIMITS["PM2.5_ug_m3"]

FEATURE_COLS = [
    "hour", "day_of_week", "month",
    "PM2_5_lag1h", "PM2_5_roll3h", "PM2_5_roll24h", "PM2_5_delta",
    "humidity_pct", "wind_speed_ms", "dust_risk_score", "temp_c",
]


# ── Public API ─────────────────────────────────────────────────────────────────

def train_all_cities(merged_df: pd.DataFrame) -> dict:
    models = {}
    for city in CITIES:
        city_df = merged_df[merged_df["city"] == city].copy()
        if len(city_df) < 24:
            log.warning(f"{city}: only {len(city_df)} rows — skipping.")
            continue
        log.info(f"Training model for {city} ({len(city_df)} rows)...")
        model = _train_model(city_df, city)
        models[city] = model
        _save_model(model, city)
        log.info(f"  ✅ {city} model trained.")
    return models


def forecast_city(model, city: str, latest_df: pd.DataFrame,
                  hours_ahead: int = FORECAST_HOURS_AHEAD) -> pd.DataFrame:
    city_df = latest_df[latest_df["city"] == city].sort_values("ds")
    if city_df.empty:
        return pd.DataFrame()

    last_row = city_df.iloc[-1]
    last_ds  = city_df["ds"].max()

    future_timestamps = pd.date_range(
        start=last_ds + pd.Timedelta(hours=1),
        periods=hours_ahead, freq="h",
    )

    rows = []
    prev_pm25 = last_row.get("PM2_5", 50.0) or 50.0
    roll3     = last_row.get("PM2_5_roll3h", prev_pm25) or prev_pm25
    roll24    = last_row.get("PM2_5_roll24h", prev_pm25) or prev_pm25

    for ts in future_timestamps:
        feat = {
            "hour":            ts.hour,
            "day_of_week":     ts.dayofweek,
            "month":           ts.month,
            "PM2_5_lag1h":     prev_pm25,
            "PM2_5_roll3h":    roll3,
            "PM2_5_roll24h":   roll24,
            "PM2_5_delta":     last_row.get("PM2_5_delta", 0.0) or 0.0,
            "humidity_pct":    last_row.get("humidity_pct", 50.0) or 50.0,
            "wind_speed_ms":   last_row.get("wind_speed_ms", 2.0) or 2.0,
            "dust_risk_score": last_row.get("dust_risk_score", 0.3) or 0.3,
            "temp_c":          last_row.get("temp_c", 28.0) or 28.0,
        }
        X = _build_feature_row(feat)
        yhat = float(model.predict(X)[0])
        yhat = max(0, yhat)

        # Simple uncertainty: ±15% of prediction
        margin = yhat * 0.15
        rows.append({
            "ds":          ts,
            "city":        city,
            "yhat":        round(yhat, 1),
            "yhat_lower":  round(max(0, yhat - margin), 1),
            "yhat_upper":  round(yhat + margin, 1),
            "alert":       _classify_alert(yhat + margin),
        })
        prev_pm25 = yhat

    return pd.DataFrame(rows)


def forecast_all_cities(models: dict, latest_df: pd.DataFrame,
                        hours_ahead: int = FORECAST_HOURS_AHEAD) -> pd.DataFrame:
    results = []
    for city, model in models.items():
        fc = forecast_city(model, city, latest_df, hours_ahead)
        if not fc.empty:
            results.append(fc)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def load_all_models() -> dict:
    models = {}
    for city in CITIES:
        m = _load_model(city)
        if m:
            models[city] = m
            log.info(f"Loaded model for {city}.")
    return models


def get_forecast_with_history(model, city: str, merged_df: pd.DataFrame,
                               hours_ahead: int = FORECAST_HOURS_AHEAD) -> pd.DataFrame:
    city_df = merged_df[merged_df["city"] == city].copy()
    recent  = city_df.sort_values("ds").tail(48)[["ds", "PM2_5"]].copy()
    recent["type"]  = "actual"
    recent["value"] = recent["PM2_5"]

    fc = forecast_city(model, city, city_df, hours_ahead)
    if fc.empty:
        return recent[["ds", "value", "type"]]

    fc["type"]  = "forecast"
    fc["value"] = fc["yhat"]

    return pd.concat([
        recent[["ds", "value", "type"]],
        fc[["ds", "value", "yhat_lower", "yhat_upper", "type", "alert"]],
    ], ignore_index=True)


# ── Internal ───────────────────────────────────────────────────────────────────

def _train_model(city_df: pd.DataFrame, city: str):
    from sklearn.ensemble import GradientBoostingRegressor

    df = city_df.sort_values("ds").copy()
    df["hour"]        = pd.to_datetime(df["ds"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["ds"]).dt.dayofweek
    df["month"]       = pd.to_datetime(df["ds"]).dt.month

    # Fill missing feature cols with medians
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)

    df = df.dropna(subset=["PM2_5"])
    X  = df[FEATURE_COLS].values
    y  = df["PM2_5"].values

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)
    return model


def _build_feature_row(feat: dict):
    return np.array([[feat.get(c, 0.0) for c in FEATURE_COLS]])


def _classify_alert(predicted_pm25: float) -> str:
    if pd.isna(predicted_pm25):
        return "GREEN"
    if predicted_pm25 >= PM25_LIMIT:
        return "RED"
    if predicted_pm25 >= PM25_LIMIT * ALERT_BUFFER_PCT:
        return "AMBER"
    return "GREEN"


def _save_model(model, city: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"gbr_{city.lower()}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _load_model(city: str):
    path = os.path.join(MODEL_DIR, f"gbr_{city.lower()}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from run_pipeline import run_demo_pipeline
    merged  = run_demo_pipeline()
    models  = train_all_cities(merged)
    fc      = forecast_all_cities(models, merged)
    print(fc[["city", "ds", "yhat", "alert"]].to_string(index=False))