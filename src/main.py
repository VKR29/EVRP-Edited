from datetime import datetime

from src.config import (
    GOOGLE_MAPS_API_KEY,
    ENERGY_MODEL,
    AVAILABILITY_API_BASE,
    SOC_START,
    SOC_MIN,
    CHARGE_TARGET_SOC,
    WAIT_PENALTY_MIN,
    P_SUCCESS_MIN,
    LOOKBACK_KM,
    SAMPLE_EVERY_KM,
    CHARGER_POWER_KW_DEFAULT,
)

from src.models.vehicle import ElectricVehicle
from src.planner.min_time_with_charging import plan_min_time_with_one_stop
from src.viz.map_viz import render_ev_route_with_charging

# src/main.py (append/modify)
import argparse
import time
from src.planner.alns_planner import alns_solve, greedy_initial_solution
from src.planner.integration_wrappers import get_travel_time_and_energy, predict_p_success_at, recommend_chargers_fn, candidate_finder

def run_alns_example():
    origin = (12.9716, 77.5946)      # sample lat/lon (Bengaluru)
    destination = (13.0827, 80.2707) # sample lat/lon (Chennai)
    initial_soc = 80
    vehicle_kwh = 40.0
    start_epoch = time.time()
    init = greedy_initial_solution(origin, destination, initial_soc, vehicle_kwh,
                                   get_travel_time_and_energy, recommend_chargers_fn, start_epoch)
    best_sol, best_val = alns_solve(init, start_epoch,
                                    get_travel_time_and_energy, predict_p_success_at,
                                    recommend_chargers_fn, candidate_finder,
                                    max_iters=2000, time_limit_seconds=30)
    print("Best expected time (s):", best_val)
    for i, stop in enumerate(best_sol.route_stops):
        print(i, stop.charger_id, stop.charge_target_soc, stop.lat, stop.lon)


def run():
    # Example coordinates (Bengaluru)
    origin = (12.9629, 77.5775)       # Bengaluru
    destination = (12.2958, 76.6394)  # Mysuru

    ev = ElectricVehicle()

    departure_dt = datetime.now()

    # === PLAN MIN EXPECTED TOTAL TIME (with charging if needed) ===
    plan = plan_min_time_with_one_stop(
        google_api_key=GOOGLE_MAPS_API_KEY,
        availability_base_url=AVAILABILITY_API_BASE,
        origin=origin,
        destination=destination,
        departure_dt=departure_dt,
        vehicle_mass_kg=ev.total_mass_kg,
        battery_kwh=ev.battery_kwh,
        soc_start=SOC_START,
        soc_min=SOC_MIN,
        charge_target_soc=CHARGE_TARGET_SOC,
        charger_power_kw_default=CHARGER_POWER_KW_DEFAULT,
        wait_penalty_min_=WAIT_PENALTY_MIN,
        p_success_min=P_SUCCESS_MIN,
        lookback_km=LOOKBACK_KM,
        sample_every_km=SAMPLE_EVERY_KM,
        radius_km=10.0,
        reco_k=3,
        lambda_decay=0.05,
        energy_cfg=ENERGY_MODEL,
    )

    # === PRINT SUMMARY ===
    print("\n=== MIN EXPECTED TOTAL TIME ROUTE ===")
    print(f"Total time:      {plan.total_time_min:.1f} min")
    print(f"Driving energy:  {plan.total_energy_kwh:.3f} kWh")

    if plan.chosen_stops:
        s = plan.chosen_stops[0]
        print("\nChosen charging stop:")
        print(f"  Station ID:    {s.station_id}")
        print(f"  Location:      ({s.lat:.6f}, {s.lon:.6f})")
        print(f"  p_success:     {s.p_success:.3f}")
        print(f"  Exp. wait:     {s.expected_wait_min:.1f} min")
        print(f"  Charge time:   {s.charge_min:.1f} min")
    else:
        print("\nNo charging stop needed (direct route feasible).")

    # === PREPARE DATA FOR MAP ===
    chosen = [
        {
            "station_id": s.station_id,
            "lat": s.lat,
            "lon": s.lon,
            "p_success": s.p_success,
            "expected_wait_min": s.expected_wait_min,
            "charge_min": s.charge_min,
        }
        for s in plan.chosen_stops
    ]

    out_html = render_ev_route_with_charging(
        origin=origin,
        destination=destination,
        route_legs=plan.route_legs,
        all_stations=plan.nearby_stations,
        chosen_stations=chosen,
        soc_track=plan.soc_track,
        out_html="routes.html",
    )

    print(f"\nMap saved: {out_html}  (open routes.html)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["single","alns"], default="alns")
    args = parser.parse_args()
    if args.planner == "alns":
        run_alns_example()
    else:
        # existing behavior
        pass
