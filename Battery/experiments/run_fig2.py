"""Generate Figure 2: SOC evolution."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.deepisl import DeepIsl
from algorithms.proposed_hbag import ProposedHBAG
from algorithms.satflow_l import SatflowL
from config import CFG
from experiments.simulate import batch_run
from utils.plotting import plot_lines


def run_fig2():
    algo_map = {"SatFlow-L": SatflowL, "DeepISL": DeepIsl, "Proposed-HBAG": ProposedHBAG}
    # Single-orbit profile keeps Figure-2 aligned with diagnostics stats.
    n_steps = int(90 * 60 / CFG.DE_SECONDS)
    orig_sem = CFG.ECLIPSE_SEMANTICS
    object.__setattr__(CFG, "ECLIPSE_SEMANTICS", "global_block_time_fraction")
    try:
        res_map = batch_run(
            "starlink_a",
            algo_map,
            CFG.SEEDS[:3],
            n_steps=n_steps,
            traffic_level="high",
            eclipse_fraction=35.0 / 90.0,
        )
    finally:
        object.__setattr__(CFG, "ECLIPSE_SEMANTICS", orig_sem)

    y = {}
    for name, runs in res_map.items():
        curves = np.vstack([r.soc_curve for r in runs])
        y[name] = np.mean(curves, axis=0)

    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(next(iter(y.values())))) * CFG.DE_SECONDS / 60.0
    plot_lines(x, y, "SOC over Time", "Time (min)", "Mean SOC", out / "fig2_soc_evolution.png")
    plot_lines(x, y, "SOC over Time", "Time (min)", "Mean SOC", out / "fig2_soc_evolution.pdf")
    print("Saved: results/figures/fig2_soc_evolution.png")


if __name__ == "__main__":
    run_fig2()
