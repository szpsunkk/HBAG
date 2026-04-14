"""Run Exp1~Exp5 in one command (improve.md style)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.exp1_main_comparison import run_experiment as run_exp1
from experiments.exp2_soc_evolution import run_experiment as run_exp2
from experiments.exp3_eclipse_sensitivity import run_experiment as run_exp3
from experiments.exp4_convergence import run_experiment as run_exp4
from experiments.exp5_scalability import run_experiment as run_exp5
from experiments.exp6_mfg_rate import run_experiment as run_exp6
from experiments.exp7_ablation import run_experiment as run_exp7
from experiments.energy_diagnostics import collect_energy_diagnostics
from experiments.plot_all_baselines import main as plot_all_baselines
from experiments.run_ieee_figures import run_ieee_figures
from experiments.sanity_check import main as run_sanity_check
from experiments.single_sat_orbit_test import collect_single_sat_orbit_diagnostics


def _build_diagnostics(all_results: dict) -> dict:
    exp1 = all_results.get("exp1", {})
    sat_runs = exp1.get("SatFlow-L", {})
    exp4 = all_results.get("exp4", {})
    exp6 = all_results.get("exp6", {})

    # These are printed by run_table1 diagnostics; keep same keys for downstream use.
    exp1_theta = sat_runs.get("theta_actual", None)
    exp1_ratio = sat_runs.get("demand_capacity_ratio", None)

    ieee_fig_dir = Path("results/figures/ieee")
    ieee_figures = sorted([p.name for p in ieee_fig_dir.glob("*") if p.is_file()]) if ieee_fig_dir.exists() else []
    diag = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_summary": {
            "experiments_included": sorted(list(all_results.keys())),
            "results_json": "results/all_experiment_results.json",
            "table1_json": "results/table1.json",
            "ieee_figures_dir": "results/figures/ieee",
            "ieee_figure_files": ieee_figures,
        },
        "table1_diagnostics": {
            "eclipse_fraction_theta": {
                "value": exp1_theta,
                "target": 0.38,
                "status": (
                    "ok" if isinstance(exp1_theta, (int, float)) and abs(exp1_theta - 0.38) <= 0.03 else "check"
                ),
            },
            "demand_capacity_ratio": {
                "value": exp1_ratio,
                "target_range": [0.3, 0.7],
                "status": (
                    "ok"
                    if isinstance(exp1_ratio, (int, float)) and 0.3 <= exp1_ratio <= 0.7
                    else "high"
                ),
            },
        },
        "energy_balance_diagnostics": collect_energy_diagnostics(),
        "single_satellite_orbit_test": collect_single_sat_orbit_diagnostics(),
        "experiment_diagnostics": {
            "exp3_delta_monotone_nondecreasing": all_results.get("exp3", {}).get(
                "delta_monotone_nondecreasing", None
            ),
            "exp4_final_residual": exp4.get("final_residual", None),
            "exp4_mse_against_O_1_sqrt_k": exp4.get("mse_against_O_1_sqrt_k", None),
            "exp4_step_size": exp4.get("step_size", None),
            "exp6_fitted_loglog_slope": exp6.get("fitted_loglog_slope", None),
            "exp6_target_slope": exp6.get("target_slope", None),
        },
        "all_experiment_results": all_results,
        "exp1_snapshot": exp1,
    }
    return diag


def main() -> None:
    print("=" * 70)
    print("Battery-Aware ISL: Full Experiment Suite")
    print("=" * 70)
    print("[CHECK] Sanity checks")
    run_sanity_check()

    all_results = {}
    steps = [
        ("exp1", "Table I main comparison", run_exp1),
        ("exp2", "Figure 2 SOC evolution", run_exp2),
        ("exp3", "Figure 3 eclipse sensitivity", run_exp3),
        ("exp4", "Figure 4 convergence", run_exp4),
        ("exp5", "Figure 5 scalability", run_exp5),
        ("exp6", "Finite-size strategy-gap rate", run_exp6),
        ("exp7", "Ablation study", run_exp7),
    ]

    for key, desc, fn in steps:
        print(f"\n[{key.upper()}] {desc}")
        t0 = time.time()
        try:
            all_results[key] = fn()
            print(f"[{key.upper()}] done in {time.time() - t0:.1f}s")
        except Exception as exc:  # pragma: no cover
            print(f"[{key.upper()}] failed: {exc}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "all_experiment_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n[IEEE] Replot publication-quality figures")
    run_ieee_figures()
    print("[IEEE] Plot all-baseline comparison (Exp-1)")
    plot_all_baselines()

    diagnostics = _build_diagnostics(all_results)
    interact_dir = Path("interact")
    interact_dir.mkdir(parents=True, exist_ok=True)
    with (interact_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    exp_map = {
        "exp1": ("Table I", "Main performance comparison"),
        "exp2": ("Figure 2", "Battery SOC evolution"),
        "exp3": ("Figure 3", "Eclipse sensitivity (Proposition 1)"),
        "exp4": ("Figure 4", "Convergence analysis (Theorem 2)"),
        "exp5": ("Figure 5", "Scalability (Proposition 2)"),
        "exp6": ("Exp-6", "Finite-size strategy-gap rate"),
        "exp7": ("Exp-7", "Ablation study"),
    }
    for exp_key, (figure, desc) in exp_map.items():
        status = "✓ DONE" if exp_key in all_results else "✗ FAILED"
        if exp_key == "exp3" and exp_key in all_results:
            ok = bool(all_results[exp_key].get("delta_monotone_nondecreasing", False))
            status = "✓ DONE" if ok else "✗ FAILED"
        print(f"  {figure:10s}: {desc:40s} [{status}]")
    print("=" * 70)
    print("Saved: results/all_experiment_results.json")
    print("Saved: interact/diagnostics.json")


if __name__ == "__main__":
    main()
