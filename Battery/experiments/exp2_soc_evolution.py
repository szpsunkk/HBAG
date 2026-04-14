"""Experiment 2: SOC evolution analysis (Figure 2)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.proposed_hbag import ProposedHBAG
from algorithms.satflow_l import SatflowL
from config import CFG
from experiments.simulate import batch_run
from experiments.run_fig2 import run_fig2


def run_experiment() -> dict:
    """Run SOC figure and return SOC statistics."""
    print("[Exp2] Battery SOC evolution")
    run_fig2()

    algo_map = {"SATFLOW": SatflowL, "Proposed (HBAG)": ProposedHBAG}
    # Use same single-orbit setup as Figure 2.
    n_steps = int(90 * 60 / CFG.DE_SECONDS)
    orig_sem = CFG.ECLIPSE_SEMANTICS
    object.__setattr__(CFG, "ECLIPSE_SEMANTICS", "global_block_time_fraction")
    try:
        res_map = batch_run(
            "starlink_a",
            algo_map,
            CFG.SEEDS[:5],
            n_steps=n_steps,
            traffic_level="high",
            eclipse_fraction=35.0 / 90.0,
        )
    finally:
        object.__setattr__(CFG, "ECLIPSE_SEMANTICS", orig_sem)
    stats = {}
    for name, runs in res_map.items():
        curves = np.vstack([r.soc_curve for r in runs])
        stats[name] = {
            "min_soc": float(np.min(curves)),
            "mean_final_soc": float(np.mean(curves[:, -1])),
        }
    print(f"[Exp2] SATFLOW min SOC: {stats['SATFLOW']['min_soc']*100:.2f}%")
    print(f"[Exp2] HBAG min SOC: {stats['Proposed (HBAG)']['min_soc']*100:.2f}%")
    cb_min_ratio = CFG.BATTERY_MIN_KJ / max(CFG.BATTERY_MAX_KJ, 1e-9)
    assert stats["Proposed (HBAG)"]["min_soc"] > cb_min_ratio, "HBAG depleted below C_min in Exp2"
    print("[Exp2] Done.")
    return stats


if __name__ == "__main__":
    run_experiment()
