# planner/alns_planner.py
# Starter ALNS for EVRP multi-stop, time-optimal planning.
# Adapt and plug into your repo: expects google_api and availability_client to be importable.
# Uses Python 3.8+. Keep this file under src/planner/ in your repo and import from src.main.

import random
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ---- CONFIG ----
OMEGA = (33, 9, 3, 1)   # (omega1, omega2, omega3, omega4)
LAMBDA = 0.8
INIT_TEMP = 100.0
COOLING = 0.9995
SOC_STEP = 5  # SOC discretization in percent
MIN_SOC_RESERVE = 10  # percent reserve
CANDIDATE_K = 8  # consider top-K candidate chargers when inserting

# Replace these imports with actual functions in your repo
# from src.routing.google_api import get_travel_time_and_energy
# from src.charging.availability_client import predict_p_success_at

# For now we assume the above functions exist with signatures:
# get_travel_time_and_energy(from_latlon, to_latlon) -> (drive_time_seconds, energy_wh)
# predict_p_success_at(charger_id, arrival_epoch_seconds) -> float in [0,1]

# ---- DATA CLASSES ----
@dataclass
class ChargerStop:
    charger_id: str
    lat: float
    lon: float
    arrival_time: float = 0.0
    charge_target_soc: int = 80  # percent

@dataclass
class RouteSolution:
    origin: Tuple[float,float]
    destination: Tuple[float,float]
    initial_soc: int
    vehicle_kwh: float
    route_stops: List[ChargerStop] = field(default_factory=list)
    # cached objective
    expected_time: Optional[float] = None

# ---- UTILITIES ----
def soc_after_driving(soc_percent: int, energy_needed_wh: float, vehicle_kwh: float) -> int:
    """Return SOC percent after consuming energy_needed_wh."""
    total_wh = vehicle_kwh * 1000.0
    soc_wh = soc_percent / 100.0 * total_wh
    soc_wh_after = max(0.0, soc_wh - energy_needed_wh)
    return int(round((soc_wh_after / total_wh) * 100.0))

def charge_time_seconds(soc_from: int, soc_to: int, vehicle_kwh: float):
    """
    Simple charging time model: assume constant average power until target.
    Replace this with vehicle's actual charging curve from models/vehicle.py
    """
    if soc_to <= soc_from:
        return 0.0
    # average charging power (kW) approximation
    avg_kw = 50.0  # placeholder; use vehicle + charger capabilities
    total_wh = vehicle_kwh * 1000.0
    wh = (soc_to - soc_from) / 100.0 * total_wh
    return wh / (avg_kw * 1000.0) * 3600.0

def p_to_expected_wait(p_success: float, base_wait_seconds=600.0):
    """
    Heuristic mapping from p_success -> expected wait.
    If p_success is close to 1 => small wait; if low, scale up.
    E[wait] = base_wait * (1 - p)/p  (keeps finite but grows when p->0)
    """
    eps = 1e-6
    p = max(min(p_success, 0.9999), eps)
    return base_wait_seconds * (1.0 - p) / p

# ---- EVALUATION (KEY) ----
def evaluate_solution(sol: RouteSolution, start_epoch: float,
                      get_travel_time_and_energy, predict_p_success_at) -> float:
    """
    Compute expected total time (seconds) for the route solution.
    - get_travel_time_and_energy: function(from_coord, to_coord) -> (drive_seconds, energy_wh)
    - predict_p_success_at: function(charger_id, arrival_epoch) -> p_success
    """
    t = 0.0
    soc = sol.initial_soc
    pos = sol.origin
    vehicle_kwh = sol.vehicle_kwh
    for stop in sol.route_stops:
        drive_time, energy_wh = get_travel_time_and_energy(pos, (stop.lat, stop.lon))
        t += drive_time
        soc = soc_after_driving(soc, energy_wh, vehicle_kwh)
        arrival_epoch = start_epoch + t
        p = predict_p_success_at(stop.charger_id, arrival_epoch)
        wait = p_to_expected_wait(p)
        t += wait
        # charging time
        charge_sec = charge_time_seconds(soc, stop.charge_target_soc, vehicle_kwh)
        t += charge_sec
        soc = stop.charge_target_soc
        pos = (stop.lat, stop.lon)
    # drive to destination
    drive_time, energy_wh = get_travel_time_and_energy(pos, sol.destination)
    t += drive_time
    soc = soc_after_driving(soc, energy_wh, vehicle_kwh)
    sol.expected_time = t
    return t

# ---- CONSTRUCTION (initial solution) ----
def greedy_initial_solution(origin, destination, initial_soc, vehicle_kwh,
                            get_travel_time_and_energy, recommend_chargers_fn, start_epoch):
    """
    Greedy: follow fastest route; whenever SOC would drop below reserve, insert best nearby charger.
    recommend_chargers_fn(location, k) -> list of candidate chargers with (charger_id, lat, lon)
    """
    sol = RouteSolution(origin=origin, destination=destination, initial_soc=initial_soc, vehicle_kwh=vehicle_kwh)
    pos = origin
    soc = initial_soc
    # naive approach: we attempt to go straight to destination and insert chargers when needed
    # More advanced: discretize route and check at critical points.
    # Here we do a loop: if destination reachable with reserve -> done, else insert nearest good charger.
    while True:
        drive_time, energy_wh = get_travel_time_and_energy(pos, destination)
        soc_after = soc_after_driving(soc, energy_wh, vehicle_kwh)
        if soc_after >= MIN_SOC_RESERVE:
            break
        # need a charger: find candidate chargers near the straight-line midpoint
        candidates = recommend_chargers_fn(pos, CANDIDATE_K)
        # score by small detour + high predicted p (use start_epoch+t approximation)
        best = None
        best_score = float('inf')
        for cid, lat, lon in candidates:
            dtime_to_c, _ = get_travel_time_and_energy(pos, (lat, lon))
            dtime_c_to_dest, _ = get_travel_time_and_energy((lat, lon), destination)
            # simple combined measure
            score = dtime_to_c + dtime_c_to_dest
            if score < best_score:
                best_score = score
                best = (cid, lat, lon)
        if best is None:
            # fail-safe: no chargers found
            break
        cid, lat, lon = best
        # choose a conservative charge target: enough to reach destination
        sol.route_stops.append(ChargerStop(charger_id=cid, lat=lat, lon=lon, charge_target_soc=80))
        # update pos & soc roughly as if we executed this
        drive_time_to_c, energy_to_c = get_travel_time_and_energy(pos, (lat, lon))
        soc = soc_after_driving(soc, energy_to_c, vehicle_kwh)
        soc = 80  # after charging (approx)
        pos = (lat, lon)
    return sol

# ---- DESTROY OPERATORS ----
def destroy_random(sol: RouteSolution, q: int):
    removed = []
    if not sol.route_stops:
        return removed
    q = min(q, len(sol.route_stops))
    idxs = random.sample(range(len(sol.route_stops)), q)
    idxs.sort(reverse=True)
    for i in idxs:
        removed.append(sol.route_stops.pop(i))
    return removed

def destroy_worst(sol: RouteSolution, q: int, eval_func, start_epoch, get_travel_time_and_energy, predict_p_success_at):
    """
    Remove stops with largest marginal expected-time contribution.
    We compute contribution by evaluating solution without each stop.
    """
    if q <= 0:
        return []
    contributions = []
    base_val = eval_func(sol, start_epoch, get_travel_time_and_energy, predict_p_success_at)
    for i, stop in enumerate(sol.route_stops):
        s_copy = RouteSolution(sol.origin, sol.destination, sol.initial_soc, sol.vehicle_kwh, sol.route_stops.copy())
        s_copy.route_stops.pop(i)
        val = eval_func(s_copy, start_epoch, get_travel_time_and_energy, predict_p_success_at)
        contributions.append((val - base_val, i))
    # pick largest deltas
    contributions.sort(reverse=True)
    removed = []
    for _, idx in contributions[:q]:
        removed.append(sol.route_stops.pop(idx))
    return removed

# ---- REPAIR OPERATORS ----
def repair_best_insertion(sol: RouteSolution, removed: List[ChargerStop],
                          candidate_finder, eval_func, start_epoch, get_travel_time_and_energy, predict_p_success_at):
    """
    Reinsert removed stops by trying best insertion positions and top candidate alternatives.
    candidate_finder(stop_location, k) -> list of (cid, lat, lon)
    """
    for r in removed:
        best_pos = None
        best_obj = float('inf')
        best_stop = None
        # try insertion at all positions
        for ins_idx in range(len(sol.route_stops) + 1):
            # consider a few nearby alternative chargers (including the original)
            candidates = candidate_finder((r.lat, r.lon), CANDIDATE_K)
            for cid, lat, lon in candidates:
                # try new stop with same charge target
                new_stop = ChargerStop(charger_id=cid, lat=lat, lon=lon, charge_target_soc=r.charge_target_soc)
                s_copy = RouteSolution(sol.origin, sol.destination, sol.initial_soc, sol.vehicle_kwh, sol.route_stops.copy())
                s_copy.route_stops.insert(ins_idx, new_stop)
                val = eval_func(s_copy, start_epoch, get_travel_time_and_energy, predict_p_success_at)
                if val < best_obj:
                    best_obj = val
                    best_pos = ins_idx
                    best_stop = new_stop
        if best_pos is None:
            # fallback: append
            sol.route_stops.append(r)
        else:
            sol.route_stops.insert(best_pos, best_stop)
    return sol

def repair_greedy(sol: RouteSolution, removed: List[ChargerStop], candidate_finder, eval_func, start_epoch,
                  get_travel_time_and_energy, predict_p_success_at):
    # simple: reinsert each removed at best position greedily (one-by-one)
    for r in removed:
        best_pos, best_stop, best_val = None, None, float('inf')
        for ins_idx in range(len(sol.route_stops) + 1):
            cand_list = candidate_finder((r.lat, r.lon), CANDIDATE_K)
            for cid, lat, lon in cand_list:
                new_stop = ChargerStop(charger_id=cid, lat=lat, lon=lon, charge_target_soc=r.charge_target_soc)
                s_copy = RouteSolution(sol.origin, sol.destination, sol.initial_soc, sol.vehicle_kwh, sol.route_stops.copy())
                s_copy.route_stops.insert(ins_idx, new_stop)
                val = eval_func(s_copy, start_epoch, get_travel_time_and_energy, predict_p_success_at)
                if val < best_val:
                    best_val = val
                    best_pos = ins_idx
                    best_stop = new_stop
        if best_pos is None:
            sol.route_stops.append(r)
        else:
            sol.route_stops.insert(best_pos, best_stop)
    return sol

# ---- ALNS MAIN LOOP ----
def alns_solve(initial_solution: RouteSolution, start_epoch: float,
               get_travel_time_and_energy, predict_p_success_at,
               recommend_chargers_fn, candidate_finder, max_iters=5000, time_limit_seconds=60):
    """
    initial_solution: constructed via greedy_initial_solution or given
    recommend_chargers_fn(location, k) -> list of (cid,lat,lon) used by constructor
    candidate_finder(location, k) -> list of candidate chargers used by repair
    """
    # set up neighborhood pools
    destroy_methods = [destroy_random, lambda s,q: destroy_worst(s,q, evaluate_solution, start_epoch, get_travel_time_and_energy, predict_p_success_at)]
    repair_methods = [repair_best_insertion, repair_greedy]
    # weights
    rho_minus = [1.0] * len(destroy_methods)
    rho_plus = [1.0] * len(repair_methods)
    best = initial_solution
    best_val = evaluate_solution(best, start_epoch, get_travel_time_and_energy, predict_p_success_at)
    current = RouteSolution(best.origin, best.destination, best.initial_soc, best.vehicle_kwh, best.route_stops.copy())
    current_val = best_val
    T = INIT_TEMP
    start_time = time.time()
    iter_ = 0
    while iter_ < max_iters and (time.time() - start_time) < time_limit_seconds:
        # pick destroy and repair by roulette wheel
        d_idx = roulette_select_index(rho_minus)
        r_idx = roulette_select_index(rho_plus)
        # choose q
        q = random.randint(1, max(1, math.ceil(0.15 * max(1, len(current.route_stops)))))
        # copy current
        cand = RouteSolution(current.origin, current.destination, current.initial_soc, current.vehicle_kwh, current.route_stops.copy())
        removed = destroy_methods[d_idx](cand, q)
        # repair
        cand = repair_methods[r_idx](cand, removed, candidate_finder, evaluate_solution, start_epoch, get_travel_time_and_energy, predict_p_success_at)
        cand_val = evaluate_solution(cand, start_epoch, get_travel_time_and_energy, predict_p_success_at)
        accepted = accept_solution(current_val, cand_val, T)
        # score update
        psi = score_delta(cand_val, current_val, best_val)
        # update adaptive weights
        rho_minus[d_idx] = LAMBDA * rho_minus[d_idx] + (1.0 - LAMBDA) * psi
        rho_plus[r_idx] = LAMBDA * rho_plus[r_idx] + (1.0 - LAMBDA) * psi
        if accepted:
            current = cand
            current_val = cand_val
        if cand_val < best_val:
            best = cand
            best_val = cand_val
        # cooling
        T *= COOLING
        iter_ += 1
    return best, best_val

# ---- HELPERS ----
def roulette_select_index(weights):
    s = sum(weights)
    if s <= 0:
        idx = random.randrange(len(weights))
        return idx
    pick = random.random() * s
    cur = 0.0
    for i, w in enumerate(weights):
        cur += w
        if pick <= cur:
            return i
    return len(weights) - 1

def accept_solution(current_val, cand_val, T):
    if cand_val < current_val:
        return True
    # simulated annealing criterion
    delta = cand_val - current_val
    p = math.exp(-delta / max(1e-6, T))
    return random.random() < p

def score_delta(cand_val, current_val, best_val):
    # produce psi according to rules: omega1 if new global best, omega2 if better than current, omega3 if accepted, else omega4
    if cand_val < best_val:
        return OMEGA[0]
    elif cand_val < current_val:
        return OMEGA[1]
    # we treat accepted case outside; caller should manage; for simplicity give omega3
    return OMEGA[2]

# ---- END OF FILE ----
