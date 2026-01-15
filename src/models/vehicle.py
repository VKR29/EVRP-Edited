from dataclasses import dataclass

@dataclass
class ElectricVehicle:
    mass_kg: float = 1700.0
    battery_kwh: float = 50.0
    payload_kg: float = 100.0

    @property
    def total_mass_kg(self) -> float:
        return self.mass_kg + self.payload_kg