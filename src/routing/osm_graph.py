from typing import Optional
import osmnx as ox
import networkx as nx

def build_drive_graph(place_query: str) -> nx.MultiDiGraph:
    G = ox.graph_from_place(place_query, network_type="drive")

    # In most OSMnx versions, edges already have 'length' in meters.
    # If not, we compute it using great-circle distance.
    # We'll defensively ensure it's there.
    missing_len = any(("length" not in data) for _, _, _, data in G.edges(keys=True, data=True))
    if missing_len:
        for u, v, k, data in G.edges(keys=True, data=True):
            uy, ux = G.nodes[u]["y"], G.nodes[u]["x"]
            vy, vx = G.nodes[v]["y"], G.nodes[v]["x"]
            data["length"] = ox.distance.great_circle_vec(uy, ux, vy, vx)

    # Add speeds + travel times (works across OSMnx versions)
    G = ox.add_edge_speeds(G)        # adds 'speed_kph'
    G = ox.add_edge_travel_times(G)  # adds 'travel_time' (seconds)

    return G

def nearest_node_by_edge(G, lat: float, lng: float) -> int:
    """
    Robust snapping: find nearest edge to (lat,lng) and return one endpoint node.
    Works well even when nearest_nodes behaves oddly.
    """
    # nearest_edges expects X=lon, Y=lat
    edges = ox.nearest_edges(G, X=[lng], Y=[lat])
    u, v, k = edges[0]
    return int(u)

def add_node_elevations(G: nx.MultiDiGraph, elevations_m: dict) -> None:
    for n, elev in elevations_m.items():
        G.nodes[n]["elevation"] = float(elev)

def add_edge_grades(G: nx.MultiDiGraph) -> None:
    for u, v, k, data in G.edges(keys=True, data=True):
        eu = G.nodes[u].get("elevation")
        ev = G.nodes[v].get("elevation")
        if eu is None or ev is None:
            data["delta_h_m"] = 0.0
            data["grade"] = 0.0
            data["grade_abs"] = 0.0
            continue

        dh = float(ev) - float(eu)
        d = float(data.get("length", 0.0))
        grade = (dh / d) if d > 1e-6 else 0.0
        data["delta_h_m"] = dh
        data["grade"] = grade
        data["grade_abs"] = abs(grade)