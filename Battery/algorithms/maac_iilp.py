"""Energy-aware linear baseline."""

import numpy as np
from typing import Optional

from algorithms.base import RoutingAlgorithm
from config import CFG


class MaacIilp(RoutingAlgorithm):
    name = "MAAC-IILP"

    def allocate_power(
        self,
        demand_mbps: np.ndarray,
        soc: np.ndarray,
        distance_km: np.ndarray,
        prev_power_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        norm_d = demand_mbps / max(np.max(demand_mbps), 1e-9)
        distance_factor = np.clip(distance_km / np.percentile(distance_km, 90), 0.4, 1.5)
        energy_factor = 0.3 + 0.7 * soc
        p = CFG.P_MAX_W * norm_d * distance_factor * energy_factor
        return self._clip(p)
