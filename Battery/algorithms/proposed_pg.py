"""Proposed potential-game strategy."""

import numpy as np
from typing import Optional

from algorithms.base import RoutingAlgorithm
from config import CFG


class ProposedPG(RoutingAlgorithm):
    name = "Proposed-PG"

    def compute_battery_penalty_coefficient(self, soc: float, soc_min: float) -> float:
        """
        计算电池惩罚系数：λ / (SOC - SOC_min + ε)
        当SOC接近SOC_min时惩罚系数增大，强制降低功率
        """
        gap = soc - soc_min + CFG.EPSILON_PENALTY
        if gap <= 0:
            return 100.0  # 极端惩罚
        return CFG.LAMBDA_ENERGY / gap

    def allocate_power(
        self,
        demand_mbps: np.ndarray,
        soc: np.ndarray,
        distance_km: np.ndarray,
        prev_power_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if prev_power_w is None:
            prev_power_w = np.zeros_like(demand_mbps)

        soc_min = CFG.BATTERY_MIN_KJ / max(CFG.BATTERY_MAX_KJ, 1e-9)
        demand_norm = demand_mbps / max(np.max(demand_mbps), 1e-9)
        dist_norm = np.clip(np.mean(distance_km) / np.maximum(distance_km, 1e-9), 0.7, 1.3)

        # 计算每个卫星的电池惩罚系数
        battery_coeffs = np.array([self.compute_battery_penalty_coefficient(s, soc_min) for s in soc])
        
        # 动态功率上限：根据电池状态调整最大功率
        # SOC越低，允许的最大功率越小
        dynamic_pmax = CFG.P_MAX_W * np.clip(0.35 + 0.65 * (soc - soc_min) / max(1.0 - soc_min, 0.1), 0.08, 1.0)

        # Hard guard near minimum battery to force safety-first behavior.
        emergency_mode = np.where(soc <= soc_min + 0.20, 0.06, 1.0)
        
        # 功率分配：考虑电池惩罚和紧急模式
        power_factor = np.clip(1.0 / (1.0 + 1.6 * battery_coeffs), 0.03, 1.0) * emergency_mode

        target = dynamic_pmax * demand_norm * dist_norm * power_factor
        p = 0.7 * prev_power_w + 0.3 * target
        return self._clip(p)