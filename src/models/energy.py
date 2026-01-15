import math

G = 9.80665

def edge_energy_joules(
    mass_kg: float,
    distance_m: float,
    delta_h_m: float,
    speed_mps: float,
    c_rr: float,
    air_density: float,
    cd_a: float,
    drivetrain_eff: float,
    regen_eff: float,
) -> float:
    e_roll = mass_kg * G * c_rr * distance_m
    e_aero = 0.5 * air_density * cd_a * (speed_mps ** 2) * distance_m
    e_grade = mass_kg * G * delta_h_m

    if e_grade >= 0:
        e_trac = (e_roll + e_aero + e_grade) / max(drivetrain_eff, 1e-6)
        return max(e_trac, 0.0)
    else:
        recovered = (-e_grade) * regen_eff
        e_trac = (e_roll + e_aero) / max(drivetrain_eff, 1e-6) - recovered
        return max(e_trac, 0.0)

def joules_to_kwh(j: float) -> float:
    return j / 3.6e6