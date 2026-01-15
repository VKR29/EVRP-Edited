from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import folium

LatLng = Tuple[float, float]


def render_ev_route_with_charging(
    *,
    origin: LatLng,
    destination: LatLng,
    route_legs: List[List[LatLng]],
    all_stations: List[Dict],
    chosen_stations: List[Dict],
    soc_track: List[Dict],
    out_html: str = "routes.html",
) -> str:
    center = ((origin[0] + destination[0]) / 2.0, (origin[1] + destination[1]) / 2.0)
    m = folium.Map(location=center, zoom_start=11)

    # -------------------------
    # Origin / Destination
    # -------------------------
    folium.Marker(
        origin,
        tooltip="Origin",
        icon=folium.Icon(color="blue", icon="play"),
    ).add_to(m)

    folium.Marker(
        destination,
        tooltip="Destination",
        icon=folium.Icon(color="blue", icon="stop"),
    ).add_to(m)

    # -------------------------
    # Route legs (no tooltip like "Leg1/Leg2")
    # -------------------------
    leg_colors = ["purple", "purple", "purple"]
    for i, leg in enumerate(route_legs):
        if not leg:
            continue
        folium.PolyLine(
            locations=leg,
            weight=6,
            opacity=0.85,
            color=leg_colors[i % len(leg_colors)],
        ).add_to(m)

    # -------------------------
    # Charging stations: ALL stations use charging icon (⚡)
    # chosen station uses different color
    # -------------------------
    chosen_ids = {c.get("station_id") for c in chosen_stations if c.get("station_id")}

    # Plot ALL stations with a lightning icon (same style)
    for st in all_stations:
        sid = st.get("station_id")
        lat = st.get("lat")
        lon = st.get("lon")
        if sid is None or lat is None or lon is None:
            continue

        is_chosen = sid in chosen_ids

        # Use different icon color for chosen vs other stations
        icon_color = "green" if is_chosen else "lightgray"

        # Minimal popup (no probability required)
        popup_lines = [f"<b>{sid}</b>"]
        if "distance_km" in st:
            popup_lines.append(f"distance_km: {float(st['distance_km']):.2f}")
        if "p_success" in st:
            # keep if present, but not required
            popup_lines.append(f"p_success: {float(st['p_success']):.3f}")

        folium.Marker(
            location=(float(lat), float(lon)),
            tooltip=f"⚡ {sid}",
            popup=folium.Popup("<br>".join(popup_lines), max_width=300),
            icon=folium.Icon(color=icon_color, icon="flash"),
        ).add_to(m)

    # -------------------------
    # SOC dots
    # Hover on dots shows SOC
    # -------------------------
    if soc_track and len(soc_track) > 2:
        # Make it more continuous: aim ~1200 dots max
        stride = max(1, len(soc_track) // 1200)

        for p in soc_track[::stride]:
            lat = float(p["lat"])
            lon = float(p["lon"])
            soc_pct = float(p["soc"]) * 100.0

            folium.CircleMarker(
                location=(lat, lon),
                radius=2.2,         # small dot
                color="black",
                fill=True,
                fill_opacity=0.85,
                opacity=0.85,
                tooltip=f"SOC: {soc_pct:.1f}%",
            ).add_to(m)

    m.save(out_html)
    return out_html