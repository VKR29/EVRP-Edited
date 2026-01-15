from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
import math

from src.routing.google_api import get_elevations_m, edge_energy_joules, joules_to_kwh


LatLng = Tuple[float, float]


@dataclass
class StepEnergy:
    step_end_lat: float
    step_end_lon: float
    dt_s: float
    dist_m: float
    e_kwh: float


def compute_step_energies_from_google_steps(
    api_key: str,
    steps: List[dict],
    vehicle_mass_kg: float,
    energy_cfg: dict,
    cache_dir: str = "data/cache",
) -> List[StepEnergy]:
    """
    Uses per-step distance/duration (traffic-aware) + elevation delta to estimate energy per step.
    """
    if not steps:
        return []

    pts: List[LatLng] = []
    for st in steps:
        s = st.get("start_location")
        e = st.get("end_location")
        if not s or not e:
            continue
        pts.append((float(s["lat"]), float(s["lng"])))
        pts.append((float(e["lat"]), float(e["lng"])))

    elevs = get_elevations_m(api_key, pts, cache_dir=cache_dir)

    out: List[StepEnergy] = []
    idx = 0
    for st in steps:
        s = st.get("start_location")
        e = st.get("end_location")
        if not s or not e:
            continue

        dist_m = float(st.get("distance", {}).get("value", 0.0))
        dt_s = float(st.get("duration", {}).get("value", 0.0))
        if dist_m <= 0.0:
            idx += 2
            continue

        speed_mps = max(dist_m / max(dt_s, 1.0), 1.0)
        e_start = float(elevs[idx])
        e_end = float(elevs[idx + 1])
        idx += 2

        delta_h = e_end - e_start

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

        out.append(
            StepEnergy(
                step_end_lat=float(e["lat"]),
                step_end_lon=float(e["lng"]),
                dt_s=dt_s,
                dist_m=dist_m,
                e_kwh=e_kwh,
            )
        )
    return out


def build_soc_track(
    step_energies: List[StepEnergy],
    battery_kwh: float,
    soc_start: float,
) -> List[Dict]:
    """
    Returns list of {"lat","lon","soc","t_s"} points (end of each step).
    """
    track: List[Dict] = []
    soc = float(soc_start)
    t_s = 0.0

    # Add initial point if we can infer it (use first step end as next best)
    if step_energies:
        track.append(
            {"lat": step_energies[0].step_end_lat, "lon": step_energies[0].step_end_lon, "soc": soc, "t_s": 0.0}
        )

    for se in step_energies:
        t_s += float(se.dt_s)
        soc = max(0.0, soc - float(se.e_kwh) / max(battery_kwh, 1e-6))
        track.append({"lat": se.step_end_lat, "lon": se.step_end_lon, "soc": soc, "t_s": t_s})

    return track