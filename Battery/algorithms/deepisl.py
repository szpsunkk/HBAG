"""Neural-inspired smooth controller baseline."""

import numpy as np
from typing import Optional

from algorithms.base import RoutingAlgorithm
from config import CFG


class DeepIsl(RoutingAlgorithm):
    name = "DeepISL"

    def allocate_power(
        self,
        demand_mbps: np.ndarray,
        soc: np.ndarray,
        distance_km: np.ndarray,
        prev_power_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x = 1.2 * (demand_mbps / max(np.mean(demand_mbps), 1e-9))
        x -= 0.8 * (1.0 - soc)
        x += 0.3 * (distance_km / max(np.mean(distance_km), 1e-9))
        p = CFG.P_MAX_W / (1.0 + np.exp(-x))
        if prev_power_w is not None:
            p = 0.7 * prev_power_w + 0.3 * p
        return self._clip(p)
