from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Tuple, Optional

import requests
import polyline

from src.models.energy import edge_energy_joules, joules_to_kwh


def _cache_path(cache_dir: str, name: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, name)


def _stable_hash_locations(latlngs: List[Tuple[float, float]]) -> str:
    parts = [f"{lat:.6f},{lng:.6f}" for lat, lng in latlngs]
    return str(abs(hash("|".join(parts))))


def _request_json_with_retries(
    url: str,
    params: dict,
    *,
    connect_timeout_s: int = 15,
    read_timeout_s: int = 90,
    max_attempts: int = 6,
    base_backoff_s: float = 1.0,
) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, params=params, timeout=(connect_timeout_s, read_timeout_s))
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            wait = base_backoff_s * (2 ** attempt) + random.random()
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def get_elevations_m(
    api_key: str,
    latlngs: List[Tuple[float, float]],
    cache_dir: str = "data/cache",
    batch_size: int = 50,
    sleep_s: float = 0.02,
) -> List[float]:
    if not latlngs:
        return []

    out: List[float] = []
    url = "https://maps.googleapis.com/maps/api/elevation/json"

    for i in range(0, len(latlngs), batch_size):
        batch = latlngs[i : i + batch_size]
        cache_key = _stable_hash_locations(batch)
        cache_file = _cache_path(cache_dir, f"elev_{cache_key}_{len(batch)}.json")

        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            locs = "|".join([f"{lat:.6f},{lng:.6f}" for lat, lng in batch])
            params = {"locations": locs, "key": api_key}
            data = _request_json_with_retries(url, params)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)

            time.sleep(sleep_s)

        status = data.get("status", "")
        if status != "OK":
            raise RuntimeError(f"Elevation API error: {status} {data.get('error_message','')}".strip())

        results = data.get("results", [])
        if len(results) != len(batch):
            raise RuntimeError(f"Elevation returned {len(results)} for {len(batch)} points (batch {i}).")

        out.extend([float(r["elevation"]) for r in results])

    return out


def google_fastest_route(
    api_key: str,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    mode: str = "driving",
    departure_time: str | int = "now",
) -> Dict:
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "mode": mode,
        "departure_time": departure_time,
        "traffic_model": "best_guess",
        "key": api_key,
    }

    data = _request_json_with_retries(url, params)

    if data.get("status") != "OK":
        raise RuntimeError(f"Directions error: {data.get('status')} {data.get('error_message','')}".strip())

    route = data["routes"][0]
    leg = route["legs"][0]

    poly = route["overview_polyline"]["points"]
    latlng = polyline.decode(poly)

    duration_s = leg.get("duration_in_traffic", leg.get("duration", {})).get("value")
    distance_m = leg.get("distance", {}).get("value")
    steps = leg.get("steps", [])

    if duration_s is None or distance_m is None:
        raise RuntimeError("Directions response missing duration or distance fields.")

    return {
        "path_latlng": latlng,
        "duration_s": int(duration_s),
        "distance_m": int(distance_m),
        "steps": steps,
        "raw": data,
    }


def energy_for_google_route_kwh(
    api_key: str,
    steps: List[dict],
    vehicle_mass_kg: float,
    energy_cfg: dict,
    cache_dir: str = "data/cache",
) -> float:
    if not steps:
        return 0.0

    # step endpoints for elevation
    pts: List[Tuple[float, float]] = []
    for st in steps:
        s = st.get("start_location")
        e = st.get("end_location")
        if not s or not e:
            continue
        pts.append((float(s["lat"]), float(s["lng"])))
        pts.append((float(e["lat"]), float(e["lng"])))

    if not pts:
        return 0.0

    elevs = get_elevations_m(api_key, pts, cache_dir=cache_dir)

    total_j = 0.0
    idx = 0
    for st in steps:
        s = st.get("start_location")
        e = st.get("end_location")
        if not s or not e:
            continue

        dist_m = float(st.get("distance", {}).get("value", 0.0))
        dur_s = float(st.get("duration", {}).get("value", 0.0))
        if dist_m <= 0:
            idx += 2
            continue

        speed_mps = max(dist_m / max(dur_s, 1.0), 1.0)

        e_start = elevs[idx]
        e_end = elevs[idx + 1]
        idx += 2

        delta_h = float(e_end - e_start)

        total_j += edge_energy_joules(
            mass_kg=vehicle_mass_kg,
            distance_m=dist_m,
            delta_h_m=delta_h,
            speed_mps=speed_mps,
            c_rr=energy_cfg["rolling_c_rr"],
            air_density=energy_cfg["air_density"],
            cd_a=energy_cfg["cd_a"],
            drivetrain_eff=energy_cfg["drivetrain_eff"],
            regen_eff=energy_cfg["regen_eff"],
        )

    return joules_to_kwh(total_j)


# src/routing/google_api.py (append)

from typing import Tuple, List, Dict
from datetime import datetime, timedelta
import polyline


def google_fastest_route_basic(
    api_key: str,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    departure_time: str | int = "now",
) -> Dict:
    """
    Thin wrapper around your existing google_fastest_route (if you already have it).
    Returns: path_latlng, duration_s, distance_m, steps
    """
    return google_fastest_route(
        api_key=api_key,
        origin=origin,
        destination=destination,
        departure_time=departure_time,
    )


def energy_profile_for_google_steps(
    api_key: str,
    steps: List[dict],
    vehicle_mass_kg: float,
    energy_cfg: dict,
    cache_dir: str = "data/cache",
) -> Dict:
    """
    Returns per-step energy (kWh), per-step time (s), and cumulative distance (m).
    This lets us find the point where SOC would drop below SOC_MIN and compute arrival time there.
    """
    if not steps:
        return {"step_energy_kwh": [], "step_time_s": [], "step_dist_m": [], "cum_time_s": [], "cum_dist_m": []}

    pts = []
    for st in steps:
        s = st.get("start_location")
        e = st.get("end_location")
        if not s or not e:
            continue
        pts.append((float(s["lat"]), float(s["lng"])))
        pts.append((float(e["lat"]), float(e["lng"])))

    elevs = get_elevations_m(api_key, pts, cache_dir=cache_dir)

    step_energy_kwh = []
    step_time_s = []
    step_dist_m = []
    cum_time_s = []
    cum_dist_m = []

    total_time = 0.0
    total_dist = 0.0

    idx = 0
    total_j = 0.0
    for st in steps:
        s = st.get("start_location")
        e = st.get("end_location")
        if not s or not e:
            continue

        dist_m = float(st.get("distance", {}).get("value", 0.0))
        dur_s = float(st.get("duration", {}).get("value", 0.0))
        if dist_m <= 0:
            idx += 2
            continue

        speed_mps = max(dist_m / max(dur_s, 1.0), 1.0)

        e_start = elevs[idx]
        e_end = elevs[idx + 1]
        idx += 2

        delta_h = float(e_end - e_start)

        j = edge_energy_joules(
            mass_kg=vehicle_mass_kg,
            distance_m=dist_m,
            delta_h_m=delta_h,
            speed_mps=speed_mps,
            c_rr=energy_cfg["rolling_c_rr"],
            air_density=energy_cfg["air_density"],
            cd_a=energy_cfg["cd_a"],
            drivetrain_eff=energy_cfg["drivetrain_eff"],
            regen_eff=energy_cfg["regen_eff"],
        )

        e_kwh = joules_to_kwh(j)

        total_time += dur_s
        total_dist += dist_m

        step_energy_kwh.append(e_kwh)
        step_time_s.append(dur_s)
        step_dist_m.append(dist_m)
        cum_time_s.append(total_time)
        cum_dist_m.append(total_dist)

    return {
        "step_energy_kwh": step_energy_kwh,
        "step_time_s": step_time_s,
        "step_dist_m": step_dist_m,
        "cum_time_s": cum_time_s,
        "cum_dist_m": cum_dist_m,
    }