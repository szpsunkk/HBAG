"""Common algorithm interface."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from config import CFG


class RoutingAlgorithm(ABC):
    name: str = "base"

    @abstractmethod
    def allocate_power(
        self,
        demand_mbps: np.ndarray,
        soc: np.ndarray,
        distance_km: np.ndarray,
        prev_power_w: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def _clip(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p, 0.0, CFG.P_MAX_W)
