# src/planner/integration_wrappers.py
import time
from typing import Tuple, List
# Replace these imports with actual functions in your repo
# from src.routing.google_api import get_directions_energy  # adapt name
# from src.charging.availability_client import availability_client

def get_travel_time_and_energy(from_coord: Tuple[float,float], to_coord: Tuple[float,float]):
    """
    Wrapper that returns (drive_seconds, energy_wh) between two lat/lon pairs.
    Replace internal calls to whatever function you have for Google directions + energy model.
    """
    # Example: if your google_api has get_route_energy(from_coord, to_coord)
    drive_seconds, energy_wh = get_directions_energy(from_coord, to_coord)
    return drive_seconds, energy_wh

def predict_p_success_at(charger_id: str, arrival_epoch: float) -> float:
    """
    Wrapper to call the backend's recommend/predict endpoint.
    If your availability_client has a function like `predict(charger_id, timestamp)` use it.
    """
    # e.g., return availability_client.predict_p_success(charger_id, arrival_epoch)
    return availability_client.predict_p_success(charger_id, arrival_epoch)

def recommend_chargers_fn(location: Tuple[float,float], k: int):
    """
    Use backend recommend API or a spatial index to return nearby chargers.
    location: (lat, lon) or (tuple)
    Returns list of (charger_id, lat, lon)
    """
    # Example: availability_client.recommend_nearby(location, k)
    return availability_client.recommend_nearby(location, k)

# candidate_finder can reuse recommend_chargers_fn or be more specialized
candidate_finder = recommend_chargers_fn
