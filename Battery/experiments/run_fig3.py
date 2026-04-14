"""Generate Figure 3: ESR vs eclipse fraction."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.deepisl import DeepIsl
from algorithms.maac_iilp import MaacIilp
from algorithms.proposed_hbag import ProposedHBAG
from algorithms.satflow_l import SatflowL
from algorithms.smfg_adapted import SmfgAdapted
from config import CFG
from experiments.simulate import batch_run
from utils.plotting import plot_lines


def run_fig3():
    fractions = np.arange(0.0, 0.8, 0.1)
    algo_map = {
        "SATFLOW": SatflowL,
        "MAAC-IILP": MaacIilp,
        "DeepISL": DeepIsl,
        "SMFG-adapted": SmfgAdapted,
        "Proposed (HBAG)": ProposedHBAG,
    }
    y = {k: [] for k in algo_map}
    for frac in fractions:
        res_map = batch_run(
            "starlink_a",
            algo_map,
            seeds=CFG.SEEDS[:3],
            n_steps=60,
            traffic_level="medium",
            eclipse_fraction=float(frac),
        )
        for name, runs in res_map.items():
            y[name].append(np.mean([r.esr for r in runs]) * 100.0)

    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    plot_lines(
        fractions,
        y,
        "ESR vs Eclipse Fraction",
        "Eclipse Fraction",
        "ESR (%)",
        out / "fig3_esr_vs_theta.png",
    )
    plot_lines(
        fractions,
        y,
        "ESR vs Eclipse Fraction",
        "Eclipse Fraction",
        "ESR (%)",
        out / "fig3_esr_vs_theta.pdf",
    )
    print("Saved: results/figures/fig3_esr_vs_theta.png")


if __name__ == "__main__":
    run_fig3()
