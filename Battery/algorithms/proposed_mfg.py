"""Proposed MFG extension."""

import numpy as np
from typing import Optional

from algorithms.base import RoutingAlgorithm
from config import CFG


class ProposedMFG(RoutingAlgorithm):
    name = "Proposed-MFG"

    def allocate_power(
        self,
        demand_mbps: np.ndarray,
        soc: np.ndarray,
        distance_km: np.ndarray,
        prev_power_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        soc_min = CFG.BATTERY_MIN_KJ / max(CFG.BATTERY_MAX_KJ, 1e-9)
        demand_norm = demand_mbps / max(np.max(demand_mbps), 1e-9)
        distance_term = np.clip(np.mean(distance_km) / np.maximum(distance_km, 1e-9), 0.7, 1.3)

        # Mean-field coupling: encourage higher transmission only for healthier-than-average nodes.
        mean_soc = float(np.mean(soc))
        soc_relative = np.clip(0.95 + 0.9 * (soc - mean_soc), 0.45, 1.3)

        # Softer penalty than PG when fleet-level SOC is healthy.
        fleet_guard = np.clip((mean_soc - soc_min) / max(1.0 - soc_min, 1e-9), 0.1, 1.0)
        local_guard = np.clip((soc - soc_min) / max(1.0 - soc_min, 1e-9), 0.06, 1.0)

        reserve_mode = np.clip((mean_soc - (soc_min + 0.16)) / max(1.0 - soc_min, 1e-9), 0.04, 1.0)
        p = (
            CFG.P_MAX_W
            * demand_norm
            * distance_term
            * soc_relative
            * (0.25 * fleet_guard + 0.75 * local_guard)
            * reserve_mode
        )
        if prev_power_w is not None:
            p = 0.55 * prev_power_w + 0.45 * p
        return self._clip(p)
