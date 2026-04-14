"""Energy dynamics for each satellite."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from config import CFG


@dataclass
class EnergyState:
    battery_kj: float
    harvested_kj: float = 0.0
    consumed_kj: float = 0.0


class EnergyModel:
    def __init__(self, n_satellites: int) -> None:
        init_kj = CFG.BATTERY_INIT_FRACTION * CFG.BATTERY_MAX_KJ
        self.states = [EnergyState(battery_kj=init_kj) for _ in range(n_satellites)]

    def solar_input_w(self, illuminated: bool, sun_angle_rad: float) -> float:
        if not illuminated:
            return 0.0
        return (
            CFG.SOLAR_PANEL_AREA_M2
            * CFG.SOLAR_EFFICIENCY
            * CFG.SOLAR_IRRADIANCE
            * max(float(np.cos(sun_angle_rad)), 0.0)
        )

    def tx_consumption_w(self, tx_power_w: float) -> float:
        # Payload baseline plus transmit-chain efficiency overhead.
        return CFG.BASE_PAYLOAD_W + CFG.TX_EFFICIENCY_FACTOR * max(tx_power_w, 0.0)

    def step(
        self,
        dt_s: float,
        illuminated_flags: Iterable[bool],
        sun_angles_rad: Iterable[float],
        tx_powers_w: Iterable[float],
    ) -> None:
        for i, (ill, ang, ptx) in enumerate(zip(illuminated_flags, sun_angles_rad, tx_powers_w)):
            harvest_kj = self.solar_input_w(ill, ang) * dt_s / 1000.0
            consume_kj = self.tx_consumption_w(ptx) * dt_s / 1000.0
            nxt = self.states[i].battery_kj + harvest_kj - consume_kj
            # Do not clamp to C_min; otherwise ESR becomes artificially 100%.
            self.states[i].battery_kj = float(np.clip(nxt, 0.0, CFG.BATTERY_MAX_KJ))
            self.states[i].harvested_kj += harvest_kj
            self.states[i].consumed_kj += consume_kj

    def soc(self) -> np.ndarray:
        vals = [st.battery_kj / CFG.BATTERY_MAX_KJ for st in self.states]
        return np.asarray(vals, dtype=float)
