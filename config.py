# config.py — VayuSafe Central Configuration
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
CPCB_API_KEY = os.getenv("CPCB_API_KEY", "")
OWM_API_KEY  = os.getenv("OWM_API_KEY", "")

# ── Cities to Monitor ─────────────────────────────────────────────────────────
# lat/lon used for OpenWeatherMap; station_ids used for CPCB lookup
CITIES = {
    "Delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        # CPCB station IDs for construction-heavy zones (Anand Vihar, RK Puram, Punjabi Bagh)
        "cpcb_stations": ["site_5076", "site_5072", "site_5168"],
        "display_name": "Delhi NCR",
    },
    "Mumbai": {
        "lat": 19.0760,
        "lon": 72.8777,
        # Bandra, Chembur, Worli
        "cpcb_stations": ["site_1270", "site_1272", "site_1278"],
        "display_name": "Mumbai",
    },
    "Hyderabad": {
        "lat": 17.3850,
        "lon": 78.4867,
        # Bollaram, ICRISAT, Pashamylaram
        "cpcb_stations": ["site_5531", "site_5533", "site_5534"],
        "display_name": "Hyderabad",
    },
}

# ── CPCB API endpoints ────────────────────────────────────────────────────────
# data.gov.in Open Government Data API for real-time AQI
CPCB_BASE_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
CPCB_PARAMS = {
    "api-key": CPCB_API_KEY,
    "format": "json",
    "limit": 500,           # rows per call
    "filters[country]": "India",
}

# OpenWeatherMap current weather
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# ── CPCB Construction Site AQI Limits ────────────────────────────────────────
# Source: CPCB Guidelines for Construction & Demolition Sites (2017)
AQI_LIMITS = {
    "PM2.5_ug_m3": 60,   # 24-hr average limit for construction sites (µg/m³)
    "PM10_ug_m3": 100,   # 24-hr average limit (µg/m³)
    "AQI_index": 200,    # "Very Poor" AQI threshold — triggers mandatory stoppage
}

# Alert buffer: predict breach this many % above limit before actual breach
ALERT_BUFFER_PCT = 0.85   # alert when predicted value ≥ 85% of limit

# ── Forecast settings ─────────────────────────────────────────────────────────
FORECAST_HOURS_AHEAD = 2       # predict this many hours into the future
HISTORY_DAYS_FOR_TRAINING = 7  # use last N days to train/update model

# ── File paths ────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "model/saved"