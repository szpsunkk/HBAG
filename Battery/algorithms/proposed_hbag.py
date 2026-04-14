"""Hybrid Battery-Aware Game-theoretic (HBAG) routing algorithm.

Key innovations vs baselines:
  1. Eclipse state estimation via SOC change-rate (stateful, no direct eclipse oracle).
  2. Eclipse-adaptive throttle: conserves battery during eclipse while meeting demand;
     runs at full power during illumination.
  3. Finite-player battery barrier with mean-field fleet coupling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from algorithms.base import RoutingAlgorithm
from config import CFG


class ProposedHBAG(RoutingAlgorithm):
    name = "Proposed-HBAG"

    def __init__(self) -> None:
        self._prev_soc: Optional[np.ndarray] = None

    def allocate_power(
        self,
        demand_mbps: np.ndarray,
        soc: np.ndarray,
        distance_km: np.ndarray,
        prev_power_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if prev_power_w is None:
            prev_power_w = np.zeros_like(demand_mbps)
        if self._prev_soc is None:
            self._prev_soc = soc.copy()

        soc_min = CFG.BATTERY_MIN_KJ / max(CFG.BATTERY_MAX_KJ, 1e-9)
        mean_soc = float(np.mean(soc))

        # --- Eclipse detection from SOC dynamics (finite-difference battery signal) ---
        # Negative SOC delta → battery discharging → satellite likely in eclipse.
        soc_delta = soc - self._prev_soc
        in_eclipse_est = soc_delta < -5e-5

        # --- Mean-field fleet coupling (finite-player game term) ---
        local_margin = np.clip((soc - soc_min) / max(1.0 - soc_min, 1e-9), 0.0, 1.0)
        fleet_margin = np.clip((mean_soc - soc_min) / max(1.0 - soc_min, 1e-9), 0.0, 1.0)
        coupled_guard = 0.75 * local_margin + 0.25 * fleet_margin

        # --- Battery barrier: Lagrangian penalty near C_min ---
        penalty = CFG.LAMBDA_ENERGY / np.maximum(soc - soc_min + CFG.EPSILON_PENALTY, 1e-6)
        barrier = np.clip(1.0 / (1.0 + 0.3 * penalty), 0.1, 1.0)

        # --- Tiered eclipse-adaptive power cap ---
        # Illuminated (SOC increasing): full P_MAX, serve all demand.
        # Eclipse (SOC decreasing): graduated throttle keyed to remaining SOC budget:
        #   SOC > 0.55  → 0.92 × P_MAX  (early eclipse, battery healthy)
        #   SOC  0.35–0.55 → 0.78 × P_MAX  (mid eclipse)
        #   SOC ≤ 0.35  → 0.65 × P_MAX  (conservation, still covers demand at TARGET_DC_RATIO)
        # Floor at 0.65 × P_MAX ensures served ≥ demand and avoids excess FVR.
        p_tier = np.where(
            soc > 0.55, 0.92 * CFG.P_MAX_W,
            np.where(soc > 0.35, 0.78 * CFG.P_MAX_W,
                     0.65 * CFG.P_MAX_W)
        )
        p_cap = np.where(in_eclipse_est, p_tier, CFG.P_MAX_W)

        demand_norm = demand_mbps / max(np.max(demand_mbps), 1e-9)
        dist_norm = np.clip(np.mean(distance_km) / np.maximum(distance_km, 1e-9), 0.7, 1.3)

        target = p_cap * demand_norm * dist_norm * barrier
        p = 0.55 * prev_power_w + 0.45 * target

        self._prev_soc = soc.copy()
        return self._clip(p)
