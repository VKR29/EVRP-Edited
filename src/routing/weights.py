import networkx as nx
from src.models.energy import edge_energy_joules

def attach_energy_weights(G: nx.MultiDiGraph, vehicle_mass_kg: float, energy_cfg: dict) -> None:
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0))
        dh = float(data.get("delta_h_m", 0.0))

        # Use OSM inferred speed as a proxy for typical speed (m/s) for aero term
        speed_kph = float(data.get("speed_kph", 30.0))
        speed_mps = max(speed_kph * 1000.0 / 3600.0, 1.0)

        e_j = edge_energy_joules(
            mass_kg=vehicle_mass_kg,
            distance_m=length_m,
            delta_h_m=dh,
            speed_mps=speed_mps,
            c_rr=energy_cfg["rolling_c_rr"],
            air_density=energy_cfg["air_density"],
            cd_a=energy_cfg["cd_a"],
            drivetrain_eff=energy_cfg["drivetrain_eff"],
            regen_eff=energy_cfg["regen_eff"],
        )
        data["energy_j"] = float(e_j)

def attach_time_weights(G: nx.MultiDiGraph) -> None:
    # Already has 'travel_time' from osmnx.add_edge_travel_times()
    # Ensure presence / fallback
    for _, _, _, data in G.edges(keys=True, data=True):
        if "travel_time" not in data:
            length_m = float(data.get("length", 0.0))
            speed_kph = float(data.get("speed_kph", 30.0))
            speed_mps = max(speed_kph * 1000.0 / 3600.0, 1.0)
            data["travel_time"] = length_m / speed_mps