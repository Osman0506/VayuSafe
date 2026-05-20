# dashboard/app.py
"""
VayuSafe — Real-time Construction Site Air Quality Dashboard
Run with: streamlit run dashboard/app.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import CITIES, AQI_LIMITS, FORECAST_HOURS_AHEAD
from run_pipeline import run_demo_pipeline, run_live_pipeline
from pipeline.preprocessor import load_merged, get_latest_conditions
from model.forecaster import (
    train_all_cities, forecast_all_cities,
    load_all_models, get_forecast_with_history,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VayuSafe | Construction AQI Monitor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark industrial aesthetic ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@700;900&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0d0f14;
    color: #e0e4ef;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Header */
.vayusafe-header {
    background: linear-gradient(135deg, #0d0f14 0%, #111827 50%, #0d1a2b 100%);
    border-bottom: 2px solid #1e3a5f;
    padding: 1.2rem 2rem;
    margin: -1.5rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.vayusafe-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    color: #ffffff;
    text-transform: uppercase;
}
.vayusafe-title span { color: #f97316; }
.vayusafe-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #6b7a99;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.1rem;
}
.live-dot {
    width: 10px; height: 10px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 2s infinite;
    display: inline-block;
    margin-right: 6px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(34,197,94,0); }
}

/* Alert banner */
.alert-red {
    background: linear-gradient(90deg, #7f1d1d, #991b1b);
    border-left: 5px solid #ef4444;
    border-radius: 6px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #fecaca;
    text-transform: uppercase;
}
.alert-amber {
    background: linear-gradient(90deg, #78350f, #92400e);
    border-left: 5px solid #f59e0b;
    border-radius: 6px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #fde68a;
    text-transform: uppercase;
}
.alert-green {
    background: linear-gradient(90deg, #064e3b, #065f46);
    border-left: 5px solid #10b981;
    border-radius: 6px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #a7f3d0;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.green::before  { background: #10b981; }
.metric-card.amber::before  { background: #f59e0b; }
.metric-card.red::before    { background: #ef4444; }

.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #6b7a99;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.4rem;
    font-weight: 900;
    line-height: 1;
    color: #f1f5f9;
}
.metric-unit {
    font-size: 0.85rem;
    font-weight: 300;
    color: #94a3b8;
    margin-left: 4px;
}
.metric-delta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    margin-top: 0.3rem;
}
.delta-up   { color: #f87171; }
.delta-down { color: #34d399; }
.delta-flat { color: #94a3b8; }

/* Section headers */
.section-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b7a99;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0c11;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

/* Plotly chart containers */
.js-plotly-plot { border-radius: 10px; }

/* Status timestamp */
.status-bar {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #374151;
    text-align: right;
    margin-top: -1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────
ALERT_COLORS = {"GREEN": "#10b981", "AMBER": "#f59e0b", "RED": "#ef4444"}
ALERT_ICONS  = {"GREEN": "✅", "AMBER": "⚠️", "RED": "🚨"}
PM25_LIMIT   = AQI_LIMITS["PM2.5_ug_m3"]
PM10_LIMIT   = AQI_LIMITS["PM10_ug_m3"]
CITY_COLORS  = {"Delhi": "#f97316", "Mumbai": "#38bdf8", "Hyderabad": "#a78bfa"}


# ── Data loading with caching ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)   # refresh every 60 min
def load_data(use_demo: bool) -> pd.DataFrame:
    if use_demo:
        return run_demo_pipeline()
    merged = load_merged()
    if merged.empty:
        return run_live_pipeline()
    return merged


@st.cache_resource(show_spinner=False)          # persist models across reruns
def load_or_train_models(use_demo: bool):
    models = load_all_models()
    if not models:
        with st.spinner("Training forecast models..."):
            merged = load_data(use_demo)
            models = train_all_cities(merged)
    return models


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Barlow Condensed",sans-serif;font-size:1.5rem;
                font-weight:900;color:#f97316;letter-spacing:0.1em;
                text-transform:uppercase;margin-bottom:0.2rem;'>
        🏗️ VayuSafe
    </div>
    <div style='font-family:"Share Tech Mono",monospace;font-size:0.65rem;
                color:#4b5563;letter-spacing:0.12em;margin-bottom:1.5rem;'>
        CONSTRUCTION AQI MONITOR
    </div>
    """, unsafe_allow_html=True)

    use_demo = st.toggle("Demo mode (synthetic data)", value=True,
                         help="Turn off to fetch live OpenAQ + OWM data")

    st.markdown('<div class="section-header">Cities</div>', unsafe_allow_html=True)
    selected_city = st.selectbox("Active city", list(CITIES.keys()), index=0)

    st.markdown('<div class="section-header">Forecast</div>', unsafe_allow_html=True)
    hours_ahead = st.slider("Hours ahead", 1, 4, FORECAST_HOURS_AHEAD)

    st.markdown('<div class="section-header">Limits (CPCB)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:"Share Tech Mono",monospace;font-size:0.72rem;color:#6b7a99;'>
        PM2.5 limit &nbsp;→ <span style='color:#f59e0b'>{PM25_LIMIT} µg/m³</span><br>
        PM10 limit &nbsp;&nbsp;→ <span style='color:#f59e0b'>{PM10_LIMIT} µg/m³</span><br>
        Source: CPCB C&D Guidelines 2017
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Controls</div>', unsafe_allow_html=True)
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    if st.button("🧠 Retrain models", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


# ── Load data & models ─────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    merged_df = load_data(use_demo)

with st.spinner("Loading forecast models..."):
    models = load_or_train_models(use_demo)

if merged_df.empty:
    st.error("No data available. Check API keys in .env or enable Demo mode.")
    st.stop()

latest = get_latest_conditions(merged_df)
forecasts = forecast_all_cities(models, merged_df, hours_ahead) if models else pd.DataFrame()


# ── Header ─────────────────────────────────────────────────────────────────────
mode_label = "DEMO MODE" if use_demo else "LIVE"
st.markdown(f"""
<div class="vayusafe-header">
    <div>
        <div class="vayusafe-title">Vayu<span>Safe</span></div>
        <div class="vayusafe-subtitle">
            <span class="live-dot"></span>{mode_label} &nbsp;|&nbsp;
            Construction Site Air Quality Monitor &nbsp;|&nbsp; India
        </div>
    </div>
</div>
<div class="status-bar">
    Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
</div>
""", unsafe_allow_html=True)


# ── Alert banners (top of page) ────────────────────────────────────────────────
if not forecasts.empty:
    red_cities   = forecasts[forecasts["alert"] == "RED"]["city"].unique()
    amber_cities = forecasts[forecasts["alert"] == "AMBER"]["city"].unique()

    if len(red_cities) > 0:
        cities_str = ", ".join(red_cities)
        st.markdown(f"""
        <div class="alert-red">
            🚨 BREACH PREDICTED — {cities_str}:
            Forecast PM2.5 will exceed {PM25_LIMIT} µg/m³ within {hours_ahead}h.
            STOP non-essential earthwork. Deploy water spraying immediately.
        </div>
        """, unsafe_allow_html=True)

    if len(amber_cities) > 0:
        cities_str = ", ".join(amber_cities)
        st.markdown(f"""
        <div class="alert-amber">
            ⚠️ APPROACHING LIMIT — {cities_str}:
            Predicted PM2.5 ≥ 85% of CPCB limit. Prepare dust suppression measures.
        </div>
        """, unsafe_allow_html=True)

    if len(red_cities) == 0 and len(amber_cities) == 0:
        st.markdown(f"""
        <div class="alert-green">
            ✅ All sites within safe limits — No breaches predicted in next {hours_ahead}h
        </div>
        """, unsafe_allow_html=True)


# ── City overview cards ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Current Conditions — All Cities</div>',
            unsafe_allow_html=True)

city_cols = st.columns(3)
for i, city in enumerate(CITIES.keys()):
    city_row = latest[latest["city"] == city]
    fc_row   = forecasts[forecasts["city"] == city] if not forecasts.empty else pd.DataFrame()

    pm25  = city_row["PM2_5"].values[0]  if not city_row.empty else None
    pm10  = city_row["PM10"].values[0]   if not city_row.empty else None
    aqi   = city_row["AQI"].values[0]    if not city_row.empty else None
    alert = city_row["alert"].values[0]  if not city_row.empty else "GREEN"
    delta = city_row["PM2_5_delta"].values[0] if (not city_row.empty and "PM2_5_delta" in city_row.columns) else 0

    fc_pm25  = fc_row["yhat"].values[0]  if not fc_row.empty else None
    fc_alert = fc_row["alert"].values[0] if not fc_row.empty else "GREEN"
    card_cls = alert.lower()

    def fmt(v): return f"{v:.1f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"
    def delta_html(d):
        if d is None or (isinstance(d, float) and np.isnan(d)): return ""
        if d > 1:   return f'<span class="delta-up">▲ {d:+.1f}</span>'
        if d < -1:  return f'<span class="delta-down">▼ {d:+.1f}</span>'
        return f'<span class="delta-flat">→ {d:+.1f}</span>'

    with city_cols[i]:
        st.markdown(f"""
        <div class="metric-card {card_cls}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.15rem;
                            font-weight:700;color:{CITY_COLORS[city]};letter-spacing:0.05em;">
                    {city.upper()}
                </div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;
                            color:{ALERT_COLORS[alert]};font-weight:700;">
                    {ALERT_ICONS[alert]} {alert}
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.8rem;">
                <div>
                    <div class="metric-label">PM2.5</div>
                    <div class="metric-value">{fmt(pm25)}<span class="metric-unit">µg/m³</span></div>
                    <div class="metric-delta">{delta_html(delta)}</div>
                </div>
                <div>
                    <div class="metric-label">PM10</div>
                    <div class="metric-value">{fmt(pm10)}<span class="metric-unit">µg/m³</span></div>
                </div>
                <div>
                    <div class="metric-label">AQI</div>
                    <div class="metric-value">{fmt(aqi)}</div>
                </div>
            </div>
            <div style="margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid #1e293b;
                        font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#6b7a99;">
                2h forecast: <span style="color:{ALERT_COLORS[fc_alert]};font-weight:700;">
                {fmt(fc_pm25)} µg/m³ ({fc_alert})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Forecast chart for selected city ──────────────────────────────────────────
st.markdown(f'<div class="section-header">PM2.5 Forecast — {selected_city}</div>',
            unsafe_allow_html=True)

if selected_city in models:
    combined = get_forecast_with_history(models[selected_city], selected_city,
                                         merged_df, hours_ahead)

    actual   = combined[combined["type"] == "actual"]
    forecast = combined[combined["type"] == "forecast"]

    fig = go.Figure()

    # Uncertainty band
    if not forecast.empty and "yhat_upper" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"].iloc[::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(249,115,22,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% confidence",
            hoverinfo="skip",
        ))

    # Actual readings
    fig.add_trace(go.Scatter(
        x=actual["ds"], y=actual["value"],
        mode="lines",
        name="Actual PM2.5",
        line=dict(color=CITY_COLORS[selected_city], width=2),
        hovertemplate="<b>%{x|%H:%M}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
    ))

    # Forecast line
    if not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["value"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#f97316", width=2.5, dash="dash"),
            marker=dict(size=8, color="#f97316", symbol="diamond"),
            hovertemplate="<b>%{x|%H:%M}</b><br>Predicted: %{y:.1f} µg/m³<extra></extra>",
        ))

    # CPCB limit line
    fig.add_hline(
        y=PM25_LIMIT, line_dash="dot", line_color="#ef4444", line_width=1.5,
        annotation_text=f"CPCB limit {PM25_LIMIT} µg/m³",
        annotation_font_color="#ef4444",
        annotation_font_size=11,
    )

    # Amber threshold
    fig.add_hline(
        y=PM25_LIMIT * 0.85, line_dash="dot", line_color="#f59e0b", line_width=1,
        annotation_text="Alert threshold",
        annotation_font_color="#f59e0b",
        annotation_font_size=10,
    )

    # Divider between actual and forecast
    if not forecast.empty:
        split_time = forecast["ds"].min()
        fig.add_vline(x=split_time, line_dash="dash", line_color="#374151", line_width=1)
        fig.add_annotation(
            x=split_time, y=1, yref="paper",
            text="NOW", showarrow=False,
            font=dict(color="#6b7a99", size=10, family="Share Tech Mono"),
            xshift=8, yshift=-15,
        )

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#9ca3af", family="Barlow"),
        legend=dict(
            bgcolor="#0d0f14", bordercolor="#1e293b", borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor="#1e293b", zerolinecolor="#1e293b",
            tickformat="%d %b\n%H:%M",
        ),
        yaxis=dict(
            gridcolor="#1e293b", zerolinecolor="#1e293b",
            title="PM2.5 (µg/m³)",
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=340,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"No trained model available for {selected_city}. Click 'Retrain models'.")


# ── Weather + AQI breakdown side by side ──────────────────────────────────────
col_wx, col_aqi = st.columns([1, 1])

with col_wx:
    st.markdown(f'<div class="section-header">Weather — {selected_city}</div>',
                unsafe_allow_html=True)
    wx_row = latest[latest["city"] == selected_city]

    if not wx_row.empty:
        r = wx_row.iloc[0]
        def wx_metric(label, val, unit=""):
            v = f"{val:.1f}{unit}" if (val is not None and not (isinstance(val, float) and np.isnan(val))) else "—"
            return f"""
            <div style="display:flex;justify-content:space-between;padding:0.5rem 0;
                        border-bottom:1px solid #1e293b;">
                <span style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                             color:#6b7a99;letter-spacing:0.08em;">{label}</span>
                <span style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;
                             font-weight:700;color:#e0e4ef;">{v}</span>
            </div>"""

        dust_risk = r.get("dust_risk_score", None)
        risk_color = "#ef4444" if (dust_risk and dust_risk > 0.6) else \
                     "#f59e0b" if (dust_risk and dust_risk > 0.35) else "#10b981"

        st.markdown(
            wx_metric("Temperature", r.get("temp_c"), "°C") +
            wx_metric("Humidity", r.get("humidity_pct"), "%") +
            wx_metric("Wind speed", r.get("wind_speed_ms"), " m/s") +
            wx_metric("Pressure", r.get("pressure_hpa"), " hPa") +
            f"""<div style="display:flex;justify-content:space-between;padding:0.5rem 0;">
                <span style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                             color:#6b7a99;letter-spacing:0.08em;">DUST RISK SCORE</span>
                <span style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;
                             font-weight:700;color:{risk_color};">
                    {f"{dust_risk:.2f}" if dust_risk else "—"} / 1.00
                </span></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("No weather data for this city.")

with col_aqi:
    st.markdown('<div class="section-header">PM2.5 Across Cities</div>',
                unsafe_allow_html=True)

    if not latest.empty:
        bar_fig = go.Figure()
        for _, row in latest.iterrows():
            city    = row["city"]
            pm25    = row.get("PM2_5", 0) or 0
            alert   = row.get("alert", "GREEN")
            bar_fig.add_trace(go.Bar(
                x=[city], y=[pm25],
                name=city,
                marker_color=ALERT_COLORS[alert],
                text=[f"{pm25:.1f}"],
                textposition="outside",
                textfont=dict(family="Barlow Condensed", size=13, color="#e0e4ef"),
            ))

        bar_fig.add_hline(
            y=PM25_LIMIT, line_dash="dot", line_color="#ef4444", line_width=1.5,
            annotation_text=f"Limit {PM25_LIMIT}",
            annotation_font_color="#ef4444",
        )
        bar_fig.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font=dict(color="#9ca3af", family="Barlow"),
            showlegend=False,
            yaxis=dict(gridcolor="#1e293b", title="PM2.5 (µg/m³)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            bargap=0.35,
        )
        st.plotly_chart(bar_fig, use_container_width=True)


# ── Historical trend — all cities ─────────────────────────────────────────────
st.markdown('<div class="section-header">48-Hour PM2.5 Trend — All Cities</div>',
            unsafe_allow_html=True)

recent_all = (
    merged_df.sort_values("ds")
    .groupby("city")
    .tail(48)
    .reset_index(drop=True)
)

if not recent_all.empty:
    trend_fig = go.Figure()
    for city in CITIES:
        city_data = recent_all[recent_all["city"] == city]
        if city_data.empty:
            continue
        trend_fig.add_trace(go.Scatter(
            x=city_data["ds"], y=city_data["PM2_5"],
            mode="lines", name=city,
            line=dict(color=CITY_COLORS[city], width=1.8),
            hovertemplate=f"<b>{city}</b><br>%{{x|%d %b %H:%M}}<br>PM2.5: %{{y:.1f}} µg/m³<extra></extra>",
        ))

    trend_fig.add_hline(
        y=PM25_LIMIT, line_dash="dot", line_color="#ef4444", line_width=1,
        annotation_text="CPCB limit",
        annotation_font_color="#ef4444", annotation_font_size=10,
    )
    trend_fig.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font=dict(color="#9ca3af", family="Barlow"),
        legend=dict(bgcolor="#0d0f14", bordercolor="#1e293b", borderwidth=1),
        xaxis=dict(gridcolor="#1e293b", tickformat="%d %b\n%H:%M"),
        yaxis=dict(gridcolor="#1e293b", title="PM2.5 (µg/m³)"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        hovermode="x unified",
    )
    st.plotly_chart(trend_fig, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #1e293b;
            font-family:'Share Tech Mono',monospace;font-size:0.65rem;
            color:#374151;text-align:center;letter-spacing:0.1em;">
    VAYUSAFE v1.0 &nbsp;|&nbsp; DATA: OPENAQ (CPCB) + OPENWEATHERMAP &nbsp;|&nbsp;
    LIMITS: CPCB C&D SITE GUIDELINES 2017 &nbsp;|&nbsp;
    MODEL: PROPHET (META) &nbsp;|&nbsp; FOR AUTHORIZED USE ONLY
</div>
""", unsafe_allow_html=True)