"""Experiment 7: ablation study for major algorithmic components."""

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.base import RoutingAlgorithm
from algorithms.proposed_hbag import ProposedHBAG
from algorithms.satflow_l import SatflowL
from config import CFG
from experiments.simulate import batch_run


class PgNoDynamic(RoutingAlgorithm):
    """Ablation V2: fixed Pmax + battery-aware gate (no dynamic upper bound)."""

    name = "PG-no-dynamic"

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
        penalty = np.clip((soc - soc_min) / max(1.0 - soc_min, 1e-9), 0.05, 1.0)
        target = CFG.P_MAX_W * demand_norm * dist_norm * penalty
        return self._clip(0.7 * prev_power_w + 0.3 * target)


class PgNoPenalty(RoutingAlgorithm):
    """Ablation V1: dynamic upper bound idea without explicit battery penalty."""

    name = "PG-no-penalty"

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
        dynamic_pmax = CFG.P_MAX_W * np.clip(0.6 + 0.4 * (soc - soc_min) / max(1.0 - soc_min, 1e-9), 0.2, 1.0)
        target = dynamic_pmax * demand_norm * dist_norm
        return self._clip(0.7 * prev_power_w + 0.3 * target)


def run_experiment() -> dict:
    """Run V0-V4 ablation and report ESR/FVR/EE."""
    print("[Exp7] Ablation study")
    # Keep ablation scenario aligned with Exp1 for fair comparison.
    scenario = {
        "constellation": "starlink_a",
        "traffic_level": "high",
        "eclipse_fraction": 0.38,
        "n_steps": 360,
        "n_seeds": len(CFG.SEEDS),
    }
    algo_map = {
        "V0-SATFLOW": SatflowL,
        "V1-dynamic-only": PgNoPenalty,
        "V2-penalty-only": PgNoDynamic,
        "V3-HBAG-full": ProposedHBAG,
    }
    res = batch_run(
        scenario["constellation"],
        algo_map,
        CFG.SEEDS,
        n_steps=scenario["n_steps"],
        traffic_level=scenario["traffic_level"],
        eclipse_fraction=scenario["eclipse_fraction"],
    )
    out = {}
    for name, runs in res.items():
        out[name] = {
            "ESR": float(np.mean([r.esr for r in runs])),
            "FVR": float(np.mean([r.fvr for r in runs])),
            "EE": float(np.mean([r.ee for r in runs])),
        }
    out["_scenario"] = scenario
    print("[Exp7] Done.")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Exp7 ablation study.")
    parser.add_argument("--verify-config", action="store_true", help="Print Exp7 scenario and exit after run.")
    args = parser.parse_args()
    result = run_experiment()
    if args.verify_config:
        print("[Exp7] verify-config:", result.get("_scenario", {}))
