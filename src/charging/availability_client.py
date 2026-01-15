from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import requests


@dataclass
class StationReco:
    station_id: str
    distance_km: float
    p_free_next: float
    p_success: float
    pmf_free_k: List[float]


def recommend(
    base_url: str,
    lat: float,
    lon: float,
    when: datetime,
    k: int = 3,
    radius_km: float = 10.0,
    lambda_decay: float = 0.05,
    timeout_s: int = 20,
) -> List[StationReco]:
    url = base_url.rstrip("/") + "/recommend"
    # backend expects "%Y-%m-%d %H:%M"
    time_str = when.strftime("%Y-%m-%d %H:%M")

    params = {
        "lat": lat,
        "lon": lon,
        "time": time_str,
        "k": int(k),
        "radius_km": float(radius_km),
        "lambda_decay": float(lambda_decay),
    }
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    out: List[StationReco] = []
    for it in data:
        out.append(
            StationReco(
                station_id=str(it["station_id"]),
                distance_km=float(it["distance_km"]),
                p_free_next=float(it["p_free_next"]),
                p_success=float(it["p_success"]),
                pmf_free_k=[float(x) for x in it.get("pmf_free_k", [])],
            )
        )
    return out


def stations_nearby(
    base_url: str,
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    timeout_s: int = 20,
) -> List[Dict]:
    url = base_url.rstrip("/") + "/stations/nearby"
    params = {"lat": lat, "lon": lon, "radius_km": float(radius_km)}
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()