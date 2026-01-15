# src/routing/solvers.py
from __future__ import annotations

from typing import List, Tuple, Any, Dict
import networkx as nx


def shortest_path_nodes(G: nx.MultiDiGraph, source: int, target: int, weight: str) -> List[int]:
    """
    Returns a list of node IDs for the shortest path according to `weight`.
    Works for MultiDiGraph as long as edges have the weight attribute.
    """
    return nx.shortest_path(G, source=source, target=target, weight=weight)


def _best_edge_data_between(G: nx.MultiDiGraph, u: int, v: int, weight: str) -> Dict[str, Any]:
    """
    For MultiDiGraph there may be multiple parallel edges (u,v,key).
    Pick the edge data dict with the smallest weight value (and ensure weight exists).
    """
    ed = G.get_edge_data(u, v)
    if not ed:
        raise KeyError(f"No edge data between {u} -> {v}")

    best_key = None
    best_val = None
    best_data = None

    for k, data in ed.items():
        if weight not in data:
            continue
        val = float(data[weight])
        if best_val is None or val < best_val:
            best_val = val
            best_key = k
            best_data = data

    if best_data is None:
        # No edge had the weight attribute
        raise KeyError(f"No '{weight}' on any parallel edge between {u} -> {v}")

    return best_data


def path_edge_sum(G: nx.MultiDiGraph, path: List[int], weight: str) -> float:
    """
    Sum `weight` across the path, choosing the best (min-weight) parallel edge each step.
    Raises if any step has no such weight (so you can detect missing annotations).
    """
    if not path or len(path) < 2:
        return 0.0

    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = _best_edge_data_between(G, u, v, weight)
        total += float(data[weight])
    return total


def path_to_latlng(G: nx.MultiDiGraph, path: List[int]) -> List[Tuple[float, float]]:
    """
    Convert node path to lat/lng for mapping.
    IMPORTANT: after projection, nodes may not have lat/lng in x/y,
    but OSMnx keeps original lat/lng in 'x'/'y' only on unprojected graphs.

    However, OSMnx projected graphs still retain 'x'/'y' in projected units.
    Folium expects lat/lng degrees.

    Solution: use node attributes 'lon'/'lat' if present, otherwise fall back:
    - If graph has 'crs' projected, nodes often still have 'x'/'y' projected.
      In that case you should map using the unprojected graph OR reproject points back.

    For our pipeline, easiest is: use the unprojected graph for lat/lng output.
    But if you only have projected graph here, we try 'lon'/'lat' first.
    """
    latlng = []
    for n in path:
        node = G.nodes[n]
        if "lat" in node and "lon" in node:
            latlng.append((float(node["lat"]), float(node["lon"])))
        else:
            # Fallback: works only if graph is unprojected (x=lon, y=lat)
            latlng.append((float(node["y"]), float(node["x"])))
    return latlng