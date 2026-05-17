# 🌫️ VayuSafe
### Real-Time Construction Site Air Quality Monitoring & Prediction System for India

VayuSafe is an AI-powered environmental monitoring system designed to track, forecast, and visualize construction-site air pollution across Indian cities.

The system collects real-time AQI and weather data, predicts future dust spike events using machine learning, triggers alerts for unsafe conditions, and visualizes high-risk zones through an interactive dashboard and map interface.

---

# Features

## Real-Time AQI Monitoring
Fetches live and historical:
- PM2.5
- PM10
- AQI data

from:
- CPCB (Central Pollution Control Board)
- OpenWeatherMap API

Supported cities:
- Delhi
- Hyderabad
- Mumbai

---

## Weather-Aware Forecasting
Collects:
- Temperature
- Humidity
- Wind Speed

to improve pollution prediction accuracy.

---

## AQI Forecasting
Uses:
- Prophet OR LSTM models

to predict:
- PM2.5 spikes
- PM10 spikes
- AQI breach events

1–2 hours in advance.

---

## 🚨 Smart Alerts
Triggers alerts when predicted AQI exceeds CPCB construction-site safety thresholds.

Examples:
- High Dust Risk
- Severe AQI Warning
- Unsafe Work Conditions

---

## Interactive Dashboard
Built using Streamlit.

Displays:
- Live AQI metrics
- Forecast charts
- Risk indicators
- City-wise monitoring
- Alert status

---

## Interactive Risk Maps
Built with Folium.

Shows:
- Construction site locations
- AQI heat zones
- Pollution hotspots
- Forecasted risk regions

---

# Tech Stack

| Component | Technology |
|---|---|
| Backend | Python |
| Data Processing | Pandas |
| APIs | CPCB API, OpenWeatherMap |
| Forecasting | Prophet / TensorFlow LSTM |
| Dashboard | Streamlit |
| Mapping | Folium |
| Visualization | Plotly |
| Data Requests | Requests |

---

# Project Structure

```bash
vayusafe/
│
├── .env
├── requirements.txt
├── config.py
├── run_pipeline.py
│
├── pipeline/
│   ├── __init__.py
│   ├── cpcb_fetcher.py
│   ├── weather_fetcher.py
│   └── preprocessor.py
│
├── data/
│
├── model/
│   ├── __init__.py
│   ├── forecaster.py
│   └── saved/
│
├── dashboard/
│   ├── __init__.py
│   └── app.py
│
└── maps/
