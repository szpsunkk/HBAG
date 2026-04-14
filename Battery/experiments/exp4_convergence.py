"""Experiment 4: convergence analysis (Figure 4)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.run_fig4 import _residual_curve, run_fig4


def run_experiment() -> dict:
    """Run convergence figure and estimate O(1/sqrt(k)) trend."""
    print("[Exp4] Convergence analysis (Theorem 2)")
    run_fig4()
    max_iters = 500
    tol = 1e-6
    c0 = 0.5
    mean_curve = _residual_curve(c0=c0, x_star=1.0, max_iters=max_iters, tol=tol)
    k = np.arange(1, len(mean_curve) + 1, dtype=float)
    curve_norm = mean_curve / max(mean_curve[0], 1e-12)
    ref = 1.0 / np.sqrt(k)
    mse_ref = float(np.mean((curve_norm - ref) ** 2))
    epsilon_energy = float(np.median(mean_curve[-min(20, len(mean_curve)) :]))
    out = {
        "final_residual": float(mean_curve[-1]),
        "mse_against_O_1_sqrt_k": mse_ref,
        "epsilon_energy": epsilon_energy,
        "max_iters": max_iters,
        "tol": tol,
        "step_size": "eta_k = 0.5/sqrt(k)",
    }
    print(f"[Exp4] Final residual: {out['final_residual']:.6f}")
    print("[Exp4] Done.")
    return out


if __name__ == "__main__":
    run_experiment()
