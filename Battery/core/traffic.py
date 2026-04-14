"""Traffic generator utilities."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from config import CFG


@dataclass
class TrafficSnapshot:
    offered_mbps: np.ndarray
    demand_level: str


class TrafficModel:
    def __init__(self, constellation_key: str, seed: int) -> None:
        self.constellation_key = constellation_key
        self.rng = np.random.default_rng(seed)
        self.level_ranges: Dict[str, tuple] = CFG.TRAFFIC_LEVELS[constellation_key]
        self.levels = list(self.level_ranges.keys())

    def sample(self, n_nodes: int, level: str = None) -> TrafficSnapshot:
        if level is None:
            level = self.levels[int(self.rng.integers(0, len(self.levels)))]
        elif level not in self.level_ranges:
            raise ValueError("Unsupported traffic level: %s" % level)
        lo, hi = self.level_ranges[level]
        offered = self.rng.uniform(lo, hi, size=n_nodes)
        return TrafficSnapshot(offered_mbps=offered, demand_level=level)
