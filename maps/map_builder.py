# maps/map_builder.py
"""
VayuSafe — Interactive Folium Map
Displays construction site monitoring stations, live AQI levels,
risk zones, and forecast alerts on an interactive map of India.

Integrated into the Streamlit dashboard via streamlit-folium.
Can also be run standalone to export an HTML map file.

Usage:
    # Standalone export:
    python maps/map_builder.py --demo

    # In Streamlit (dashboard/app.py calls build_map()):
    from maps.map_builder import build_map
    folium_map = build_map(latest_df, forecasts_df)
"""

import os
import sys
import folium
import numpy as np
import pandas as pd
from pathlib import Path
from folium.plugins import HeatMap, MarkerCluster, MiniMap

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CITIES, AQI_LIMITS

PM25_LIMIT = AQI_LIMITS["PM2.5_ug_m3"]

# ── Simulated construction site locations ──────────────────────────────────────
# In production these would come from a sites database or GIS layer.
# Each dict: name, lat, lon, city, type, area_sqkm
CONSTRUCTION_SITES = [
    # Delhi
    {"name": "Dwarka Expressway Extension",    "city": "Delhi",     "lat": 28.5921, "lon": 77.0460, "type": "Highway",    "area_sqkm": 4.2},
    {"name": "Delhi-Meerut RRTS Corridor",     "city": "Delhi",     "lat": 28.6742, "lon": 77.2781, "type": "Rail",       "area_sqkm": 6.8},
    {"name": "Anand Vihar Bus Terminal",       "city": "Delhi",     "lat": 28.6469, "lon": 77.3164, "type": "Terminal",   "area_sqkm": 1.1},
    {"name": "Sarojini Nagar Redevelopment",   "city": "Delhi",     "lat": 28.5755, "lon": 77.1910, "type": "Residential","area_sqkm": 0.8},
    {"name": "IGI Airport T2 Expansion",       "city": "Delhi",     "lat": 28.5562, "lon": 77.0999, "type": "Airport",    "area_sqkm": 3.5},
    # Mumbai
    {"name": "Mumbai Coastal Road Phase 2",    "city": "Mumbai",    "lat": 19.0488, "lon": 72.8198, "type": "Highway",    "area_sqkm": 2.9},
    {"name": "Bandra-Versova Sea Link",        "city": "Mumbai",    "lat": 19.0728, "lon": 72.8347, "type": "Bridge",     "area_sqkm": 1.6},
    {"name": "Dharavi Redevelopment Zone A",   "city": "Mumbai",    "lat": 19.0422, "lon": 72.8530, "type": "Residential","area_sqkm": 2.2},
    {"name": "MTHL Approach Viaduct",          "city": "Mumbai",    "lat": 18.9758, "lon": 72.8777, "type": "Bridge",     "area_sqkm": 1.8},
    # Hyderabad
    {"name": "ORR Package 7 Extension",        "city": "Hyderabad", "lat": 17.4474, "lon": 78.3762, "type": "Highway",    "area_sqkm": 5.1},
    {"name": "Hyderabad Metro Phase 2",        "city": "Hyderabad", "lat": 17.3616, "lon": 78.4747, "type": "Rail",       "area_sqkm": 3.3},
    {"name": "Genome Valley Pharma Hub",       "city": "Hyderabad", "lat": 17.5434, "lon": 78.5236, "type": "Industrial", "area_sqkm": 4.7},
    {"name": "Narsingi Township",              "city": "Hyderabad", "lat": 17.3869, "lon": 78.3488, "type": "Residential","area_sqkm": 1.9},
]

# AQI monitoring station locations (approximate coords for known CPCB stations)
MONITORING_STATIONS = {
    "Delhi": [
        {"name": "Anand Vihar",   "lat": 28.6469, "lon": 77.3164},
        {"name": "RK Puram",      "lat": 28.5641, "lon": 77.1703},
        {"name": "Punjabi Bagh",  "lat": 28.6723, "lon": 77.1322},
    ],
    "Mumbai": [
        {"name": "Bandra",        "lat": 19.0596, "lon": 72.8295},
        {"name": "Chembur",       "lat": 19.0522, "lon": 72.8994},
        {"name": "Colaba",        "lat": 18.9067, "lon": 72.8147},
    ],
    "Hyderabad": [
        {"name": "Bollaram",      "lat": 17.4916, "lon": 78.3731},
        {"name": "ICRISAT",       "lat": 17.5268, "lon": 78.2738},
        {"name": "Pashamylaram",  "lat": 17.5072, "lon": 78.2983},
    ],
}

CITY_COLORS = {"Delhi": "#f97316", "Mumbai": "#38bdf8", "Hyderabad": "#a78bfa"}
ALERT_COLORS = {"GREEN": "#10b981", "AMBER": "#f59e0b", "RED": "#ef4444"}
ALERT_LABELS = {"GREEN": "Safe", "AMBER": "Approaching limit", "RED": "Breach predicted"}

SITE_TYPE_ICONS = {
    "Highway":    ("road",       "gray"),
    "Rail":       ("train",      "darkblue"),
    "Bridge":     ("exchange",   "cadetblue"),
    "Residential":("home",       "orange"),
    "Airport":    ("plane",      "darkred"),
    "Industrial": ("industry",   "purple"),
    "Terminal":   ("bus",        "darkgreen"),
}


def build_map(
    latest_df: pd.DataFrame = None,
    forecasts_df: pd.DataFrame = None,
    show_heatmap: bool = True,
    show_risk_zones: bool = True,
    show_stations: bool = True,
) -> folium.Map:
    """
    Builds and returns the interactive Folium map.

    Layers:
      1. Base map — dark CartoDB tiles (matches dashboard theme)
      2. AQI heatmap — PM2.5 intensity across monitoring points
      3. Risk zone circles — radius proportional to site area,
         color = current alert status
      4. Construction site markers — clustered, with popups
      5. Monitoring station markers — with live AQI readings
      6. Mini-map — orientation reference

    Args:
        latest_df:   Output of get_latest_conditions() — current AQI per city
        forecasts_df: Output of forecast_all_cities() — predicted alert per city
        show_heatmap:    Toggle heatmap layer
        show_risk_zones: Toggle risk zone circles
        show_stations:   Toggle monitoring station markers
    """
    # Centre on India
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles=None,
        prefer_canvas=True,
    )

    # ── Base tile layers ───────────────────────────────────────────────────────
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap © CARTO",
        name="Dark (default)",
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap © CARTO",
        name="Light",
        max_zoom=19,
    ).add_to(m)

    # ── Prepare per-city AQI data ──────────────────────────────────────────────
    city_aqi   = _extract_city_aqi(latest_df)
    city_alert = _extract_city_alert(forecasts_df)

    # ── Layer 1: AQI Heatmap ───────────────────────────────────────────────────
    if show_heatmap:
        heat_data = _build_heatmap_data(city_aqi)
        if heat_data:
            HeatMap(
                heat_data,
                name="PM2.5 Heatmap",
                min_opacity=0.3,
                max_zoom=10,
                radius=60,
                blur=40,
                gradient={
                    "0.0": "#00e400",   # Good
                    "0.3": "#ffff00",   # Satisfactory
                    "0.5": "#ff7e00",   # Moderate
                    "0.7": "#ff0000",   # Poor
                    "0.9": "#8f3f97",   # Very poor
                    "1.0": "#7e0023",   # Severe
                },
            ).add_to(m)

    # ── Layer 2: Risk zone circles ─────────────────────────────────────────────
    if show_risk_zones:
        risk_group = folium.FeatureGroup(name="Risk Zones", show=True)
        for site in CONSTRUCTION_SITES:
            city    = site["city"]
            alert   = city_alert.get(city, "GREEN")
            color   = ALERT_COLORS[alert]
            pm25    = city_aqi.get(city, {}).get("PM2_5", 0) or 0
            radius  = int(site["area_sqkm"] * 600)   # scale area → map radius

            folium.Circle(
                location=[site["lat"], site["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.15,
                weight=1.5,
                opacity=0.6,
                tooltip=folium.Tooltip(
                    f"<b>{site['name']}</b><br>"
                    f"City: {city}<br>"
                    f"Type: {site['type']}<br>"
                    f"Area: {site['area_sqkm']} km²<br>"
                    f"Current PM2.5: {pm25:.1f} µg/m³<br>"
                    f"Forecast status: <b style='color:{color}'>{ALERT_LABELS[alert]}</b>",
                    sticky=True,
                ),
            ).add_to(risk_group)
        risk_group.add_to(m)

    # ── Layer 3: Construction site markers ────────────────────────────────────
    site_cluster = MarkerCluster(name="Construction Sites", show=True)
    for site in CONSTRUCTION_SITES:
        city    = site["city"]
        alert   = city_alert.get(city, "GREEN")
        pm25    = city_aqi.get(city, {}).get("PM2_5", 0) or 0
        fc_pm25 = city_aqi.get(city, {}).get("forecast_pm25", None)
        icon_name, icon_color = SITE_TYPE_ICONS.get(site["type"], ("wrench", "gray"))

        # Override icon color with alert status
        if alert == "RED":
            icon_color = "red"
        elif alert == "AMBER":
            icon_color = "orange"

        popup_html = _site_popup_html(site, pm25, fc_pm25, alert)

        folium.Marker(
            location=[site["lat"], site["lon"]],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f" {site['name']} — {alert}",
            icon=folium.Icon(
                color=icon_color,
                icon=icon_name,
                prefix="fa",
            ),
        ).add_to(site_cluster)

    site_cluster.add_to(m)

    # ── Layer 4: Monitoring stations ───────────────────────────────────────────
    if show_stations:
        station_group = folium.FeatureGroup(name="AQI Monitoring Stations", show=True)
        for city, stations in MONITORING_STATIONS.items():
            pm25  = city_aqi.get(city, {}).get("PM2_5", None)
            alert = city_aqi.get(city, {}).get("alert", "GREEN")
            color = ALERT_COLORS.get(alert, "#10b981")

            for st in stations:
                folium.CircleMarker(
                    location=[st["lat"], st["lon"]],
                    radius=9,
                    color="#ffffff",
                    weight=1.5,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    tooltip=folium.Tooltip(
                        f"<b>📡 {st['name']} Station</b><br>"
                        f"City: {city}<br>"
                        f"PM2.5: {f'{pm25:.1f} µg/m³' if pm25 else 'No data'}<br>"
                        f"Status: <b style='color:{color}'>{alert}</b>",
                        sticky=True,
                    ),
                ).add_to(station_group)

        station_group.add_to(m)

    # ── City labels ────────────────────────────────────────────────────────────
    for city, meta in CITIES.items():
        pm25  = city_aqi.get(city, {}).get("PM2_5", None)
        alert = city_alert.get(city, "GREEN")
        color = ALERT_COLORS[alert]
        label = f"{city}: {f'{pm25:.0f} µg/m³' if pm25 else 'No data'}"

        folium.Marker(
            location=[meta["lat"] + 0.25, meta["lon"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
                    font-size: 13px;
                    font-weight: 700;
                    color: {color};
                    background: rgba(13,15,20,0.85);
                    border: 1px solid {color};
                    border-radius: 4px;
                    padding: 3px 8px;
                    white-space: nowrap;
                    letter-spacing: 0.05em;
                    text-transform: uppercase;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
                ">
                    {label}
                </div>""",
                icon_size=(200, 30),
                icon_anchor=(100, 15),
            ),
        ).add_to(m)

    # ── Legend ─────────────────────────────────────────────────────────────────
    _add_legend(m)

    # ── Mini-map ───────────────────────────────────────────────────────────────
    MiniMap(
        tile_layer="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        position="bottomleft",
        width=150, height=120,
        toggle_display=True,
    ).add_to(m)

    # ── Layer control ──────────────────────────────────────────────────────────
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m


def export_map_html(
    latest_df: pd.DataFrame = None,
    forecasts_df: pd.DataFrame = None,
    output_path: str = "maps/vayusafe_map.html",
) -> str:
    """Saves the map as a standalone HTML file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m = build_map(latest_df, forecasts_df)
    m.save(output_path)
    return output_path


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_city_aqi(latest_df: pd.DataFrame) -> dict:
    """Converts latest_df rows → {city: {PM2_5, PM10, AQI, alert}} dict."""
    result = {}
    if latest_df is None or latest_df.empty:
        # Use placeholder values so map still renders
        for city in CITIES:
            result[city] = {"PM2_5": None, "PM10": None, "AQI": None, "alert": "GREEN"}
        return result

    for _, row in latest_df.iterrows():
        city = row.get("city")
        if city:
            result[city] = {
                "PM2_5": row.get("PM2_5"),
                "PM10":  row.get("PM10"),
                "AQI":   row.get("AQI"),
                "alert": row.get("alert", "GREEN"),
            }
    return result


def _extract_city_alert(forecasts_df: pd.DataFrame) -> dict:
    """Gets the worst forecast alert per city."""
    result = {city: "GREEN" for city in CITIES}
    if forecasts_df is None or forecasts_df.empty:
        return result

    priority = {"GREEN": 0, "AMBER": 1, "RED": 2}
    for city in CITIES:
        city_fc = forecasts_df[forecasts_df["city"] == city]
        if not city_fc.empty:
            worst = max(city_fc["alert"].tolist(), key=lambda x: priority.get(x, 0))
            result[city] = worst
    return result


def _build_heatmap_data(city_aqi: dict) -> list:
    """
    Generates heatmap point data from monitoring station locations + AQI.
    Each point: [lat, lon, intensity (0–1)]
    We scatter points around each station to create a smoother heatmap.
    """
    heat_data = []
    for city, stations in MONITORING_STATIONS.items():
        pm25 = city_aqi.get(city, {}).get("PM2_5") or 0
        # Normalise PM2.5 to 0–1 (severe = 250+ µg/m³)
        intensity = min(pm25 / 250.0, 1.0)

        for station in stations:
            # Add the central point
            heat_data.append([station["lat"], station["lon"], intensity])
            # Add scattered points for visual spread
            np.random.seed(hash(station["name"]) % 999)
            for _ in range(6):
                jitter_lat = station["lat"] + np.random.uniform(-0.05, 0.05)
                jitter_lon = station["lon"] + np.random.uniform(-0.05, 0.05)
                heat_data.append([jitter_lat, jitter_lon, intensity * 0.6])

    return heat_data


def _site_popup_html(site: dict, pm25: float, fc_pm25, alert: str) -> str:
    """Generates styled HTML for construction site marker popups."""
    color = ALERT_COLORS[alert]
    pm25_str   = f"{pm25:.1f} µg/m³" if pm25 else "No data"
    fc_pm25_str = f"{fc_pm25:.1f} µg/m³" if fc_pm25 else "—"
    pct_of_limit = f"{(pm25 / PM25_LIMIT * 100):.0f}%" if pm25 else "—"

    return f"""
    <div style="font-family:'Barlow',Arial,sans-serif;width:260px;
                background:#111827;color:#e0e4ef;border-radius:8px;
                overflow:hidden;border:1px solid {color};">
        <div style="background:{color};padding:8px 12px;
                    font-weight:700;font-size:13px;letter-spacing:0.05em;
                    text-transform:uppercase;color:#000;">
            🏗️ {site['type']} Site
        </div>
        <div style="padding:10px 12px;">
            <div style="font-size:14px;font-weight:700;margin-bottom:6px;
                        color:#f1f5f9;">{site['name']}</div>
            <div style="font-size:11px;color:#94a3b8;margin-bottom:10px;">
                {site['city']} · {site['area_sqkm']} km²
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
                <tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:4px 0;color:#6b7a99;">Current PM2.5</td>
                    <td style="text-align:right;font-weight:600;">{pm25_str}</td>
                </tr>
                <tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:4px 0;color:#6b7a99;">% of CPCB limit</td>
                    <td style="text-align:right;font-weight:600;color:{color};">{pct_of_limit}</td>
                </tr>
                <tr style="border-bottom:1px solid #1e293b;">
                    <td style="padding:4px 0;color:#6b7a99;">2h Forecast PM2.5</td>
                    <td style="text-align:right;font-weight:600;">{fc_pm25_str}</td>
                </tr>
                <tr>
                    <td style="padding:4px 0;color:#6b7a99;">Alert status</td>
                    <td style="text-align:right;font-weight:700;color:{color};">
                        {ALERT_LABELS[alert]}
                    </td>
                </tr>
            </table>
        </div>
    </div>
    """


def _add_legend(m: folium.Map):
    """Injects a styled HTML legend into the map."""
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; right: 10px; z-index: 1000;
        font-family: 'Barlow', Arial, sans-serif;
        background: rgba(13,15,20,0.92);
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        min-width: 180px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    ">
        <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#6b7a99;margin-bottom:10px;">
            Alert Status
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="width:12px;height:12px;border-radius:50%;background:#10b981;"></div>
            <span style="font-size:12px;color:#e0e4ef;">GREEN — Safe</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="width:12px;height:12px;border-radius:50%;background:#f59e0b;"></div>
            <span style="font-size:12px;color:#e0e4ef;">AMBER — Approaching limit</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <div style="width:12px;height:12px;border-radius:50%;background:#ef4444;"></div>
            <span style="font-size:12px;color:#e0e4ef;">RED — Breach predicted</span>
        </div>
        <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#6b7a99;margin-bottom:8px;">
            Site Types
        </div>
        <div style="font-size:11px;color:#94a3b8;line-height:1.7;">
             Construction site marker<br>
             AQI monitoring station<br>
            Risk zone (radius = area)
        </div>
        <div style="margin-top:10px;padding-top:8px;border-top:1px solid #1e293b;
                    font-size:10px;color:#374151;letter-spacing:0.08em;">
            CPCB PM2.5 LIMIT: 60 µg/m³
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", default=True)
    args = parser.parse_args()

    print("Building VayuSafe map...")

    # Generate demo data for the map
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from run_pipeline import run_demo_pipeline
    from pipeline.preprocessor import get_latest_conditions
    from model.forecaster import train_all_cities, forecast_all_cities

    merged     = run_demo_pipeline()
    latest     = get_latest_conditions(merged)
    models     = train_all_cities(merged)
    forecasts  = forecast_all_cities(models, merged)

    output = export_map_html(latest, forecasts, "maps/vayusafe_map.html")
    print(f"Map saved → {output}")
    print("Open maps/vayusafe_map.html in your browser to view.")