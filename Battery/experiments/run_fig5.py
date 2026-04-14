"""Generate Figure 5: runtime comparison."""

import os
import sys
from pathlib import Path
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.proposed_hbag import ProposedHBAG
from config import CFG
from experiments.simulate import simulate_once
from utils.plotting import plot_lines


def run_fig5():
    test_constellations = {
        "Shell A (172)": "starlink_a",
        "Shell B (348)": "starlink_b",
        "Kuiper (784)": "kuiper",
    }
    labels = []
    n_sats = []
    runtimes = []
    for label, const in test_constellations.items():
        c = CFG.CONSTELLATIONS[const]
        n = int(c["n_orbits"]) * int(c["n_sats_per_orbit"])
        labels.append(label)
        n_sats.append(n)
        slot_times = []
        for i in range(6):
            t0 = time.perf_counter()
            simulate_once(
                constellation_key=const,
                algorithm_cls=ProposedHBAG,
                seed=42 + i,
                n_steps=4,
                traffic_level="medium",
            )
            slot_times.append((time.perf_counter() - t0) / 4.0)
        runtimes.append(float(np.mean(slot_times[2:])))
    x = np.arange(len(labels))
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    plot_lines(
        x,
        {"Runtime per slot": runtimes, "O(n_t) ref.": (np.array(n_sats) * (runtimes[0] / n_sats[0]))},
        "Runtime vs Constellation Size",
        "Constellation Index",
        "Runtime per slot (s)",
        out / "fig5_runtime.png",
    )
    plot_lines(
        x,
        {"Runtime per slot": runtimes, "O(n_t) ref.": (np.array(n_sats) * (runtimes[0] / n_sats[0]))},
        "Runtime vs Constellation Size",
        "Constellation Index",
        "Runtime per slot (s)",
        out / "fig5_runtime.pdf",
    )
    print("Constellation order:", labels)
    print("Saved: results/figures/fig5_runtime.png")


if __name__ == "__main__":
    run_fig5()
