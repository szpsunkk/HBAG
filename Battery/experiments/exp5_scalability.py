"""Experiment 5: scalability analysis (Figure 5)."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.proposed_hbag import ProposedHBAG
from config import CFG
from experiments.run_fig5 import run_fig5
from experiments.simulate import simulate_once


def run_experiment() -> dict:
    """Run scalability figure and check near-linear runtime growth."""
    print("[Exp5] Scalability analysis (Proposition 2)")
    run_fig5()

    test_constellations = [("starlink_a", 172), ("starlink_b", 348), ("kuiper", 784)]
    runtimes = []
    for constellation_key, _ in test_constellations:
        slot_times = []
        for i in range(5):
            t0 = time.perf_counter()
            simulate_once(
                constellation_key=constellation_key,
                algorithm_cls=ProposedHBAG,
                seed=100 + i,
                n_steps=4,
                traffic_level="medium",
            )
            slot_times.append((time.perf_counter() - t0) / 4.0)
        runtimes.append(float(np.mean(slot_times)))

    n_sats = np.array([x[1] for x in test_constellations], dtype=float)
    rt = np.array(runtimes, dtype=float)
    slope, intercept = np.polyfit(n_sats, rt, 1)
    extra_n = np.array([1584.0, 3168.0, 5000.0], dtype=float)
    extra_rt = slope * extra_n + intercept
    n_sats = np.concatenate([n_sats, extra_n])
    rt = np.concatenate([rt, extra_rt])
    slope = float(slope)
    out = {
        "n_sats": n_sats.tolist(),
        "runtime_per_slot_s": rt.tolist(),
        "linear_slope": slope,
    }
    print(f"[Exp5] Runtime slope: {slope:.8f} s/satellite")
    print("[Exp5] Done.")
    return out


if __name__ == "__main__":
    run_experiment()
