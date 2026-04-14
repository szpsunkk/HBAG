"""Static Pmax baseline."""

import numpy as np
from typing import Optional

from algorithms.base import RoutingAlgorithm
from config import CFG


class SatflowL(RoutingAlgorithm):
    name = "SatFlow-L"

    def allocate_power(
        self,
        demand_mbps: np.ndarray,
        soc: np.ndarray,
        distance_km: np.ndarray,
        prev_power_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        _ = (demand_mbps, soc, distance_km, prev_power_w)
        return np.full_like(demand_mbps, CFG.P_MAX_W, dtype=float)
