"""Sanity checks for eclipse setup and SATFLOW policy."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.satflow_l import SatflowL
from config import CFG
from experiments.simulate import simulate_once


def main() -> None:
    r0 = simulate_once("starlink_a", SatflowL, seed=CFG.SEEDS[0], n_steps=90, traffic_level="high", eclipse_fraction=0.0)
    assert r0.n_eclipse_events == 0, "BUG: theta=0 should have zero eclipse events"

    sat = SatflowL()
    p = sat.allocate_power(
        demand_mbps=np.array([10.0, 20.0]),
        soc=np.array([0.2, 0.9]),
        distance_km=np.array([7000.0, 7200.0]),
    )
    assert np.allclose(p, CFG.P_MAX_W), "BUG: SATFLOW should use fixed P_MAX"

    print("All sanity checks passed!")


if __name__ == "__main__":
    main()
