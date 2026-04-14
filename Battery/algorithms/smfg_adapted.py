"""Simplified mean-field adapted baseline."""

import numpy as np
from typing import Optional

from algorithms.base import RoutingAlgorithm
from config import CFG


class SmfgAdapted(RoutingAlgorithm):
    name = "SMFG-adapted"

    def allocate_power(
        self,
        demand_mbps: np.ndarray,
        soc: np.ndarray,
        distance_km: np.ndarray,
        prev_power_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        global_load = np.mean(demand_mbps)
        local_gap = demand_mbps - global_load
        p = 0.5 * CFG.P_MAX_W + 0.8 * local_gap / max(np.std(demand_mbps), 1e-6)
        p *= 0.6 + 0.4 * soc
        if prev_power_w is not None:
            p = 0.5 * p + 0.5 * prev_power_w
        return self._clip(p)
