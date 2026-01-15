import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
if not GOOGLE_MAPS_API_KEY:
    raise RuntimeError("Missing GOOGLE_MAPS_API_KEY in environment/.env")

ENERGY_MODEL = {
    "regen_eff": 0.60,
    "rolling_c_rr": 0.010,
    "air_density": 1.225,
    "cd_a": 0.60,
    "drivetrain_eff": 0.90,
}

AVAILABILITY_API_BASE = "http://127.0.0.1:8000"
SOC_START = 0.20
SOC_MIN = 0.15
P_SUCCESS_MIN = 0.05      # relax to see behavior
LOOKBACK_KM = 40.0        # search earlier
SAMPLE_EVERY_KM = 1.0     # denser sampling
WAIT_PENALTY_MIN = 20.0     # expected wait if p_success=0
CHARGER_POWER_KW_DEFAULT = 50.0  # if you don't have per-station power yet
CHARGE_TARGET_SOC = 0.80         # charge up to this SOC (simple v1)