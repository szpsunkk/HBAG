"""Plot complete Exp-1 comparison for all six methods."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _load_exp1_snapshot() -> dict:
    results_path = Path("results/all_experiment_results.json")
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "exp1" in data:
            return data["exp1"]
    diag_path = Path("interact/diagnostics.json")
    if diag_path.exists():
        with diag_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "exp1_snapshot" in data:
            return data["exp1_snapshot"]
    raise FileNotFoundError("Missing exp1 snapshot. Run run_all_experiments.py first.")


def main() -> None:
    exp1 = _load_exp1_snapshot()
    methods = ["SatFlow-L", "MAAC-IILP", "DeepISL", "SMFG-adapted", "Proposed-HBAG"]
    labels = ["SATFLOW", "MAAC-IILP", "DeepISL", "SMFG", "HBAG"]
    esr = [exp1[m]["ESR"]["mean"] for m in methods]
    fvr = [exp1[m]["FVR"]["mean"] for m in methods]
    ee = [exp1[m]["EE_mbit_per_kj"]["mean"] for m in methods]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x = np.arange(len(methods))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    axes[0].bar(x, esr, color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_ylabel("ESR (%)")
    axes[0].set_title("(a) Energy Sustainability", fontsize=8)
    axes[0].set_ylim([max(0, min(esr) - 5), min(100, max(esr) + 5)])

    axes[1].bar(x, fvr, color=colors, edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("FVR (%)")
    axes[1].set_title("(b) Flow Violation Ratio", fontsize=8)
    axes[1].set_ylim([0, max(fvr) + 10])
    inset = axes[1].inset_axes([0.48, 0.35, 0.48, 0.58])
    nz_idx = np.arange(1, len(methods))
    inset.bar(nz_idx, np.array(fvr)[nz_idx], color=np.array(colors)[nz_idx], edgecolor="black", linewidth=0.3)
    inset.set_xticks(nz_idx)
    inset.set_xticklabels([labels[i] for i in nz_idx], rotation=35, ha="right", fontsize=5.2)
    inset.set_ylim([0, max(np.array(fvr)[nz_idx]) * 1.25 + 1e-6])
    inset.tick_params(direction="in", top=True, right=True, labelsize=5.2)
    inset.set_title("Non-SATFLOW zoom", fontsize=5.5)

    axes[2].bar(x, ee, color=colors, edgecolor="black", linewidth=0.4)
    axes[2].set_ylabel("EE (Mbit/kJ)")
    axes[2].set_title("(c) Energy Efficiency", fontsize=8)
    axes[2].set_ylim([0, max(ee) + 1.0])

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=6.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(alpha=0.25, linestyle=":")

    out = Path("results/figures/ieee")
    out.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.25, w_pad=0.6)
    plt.savefig(out / "fig_exp1_all_methods_ieee.pdf", bbox_inches="tight")
    plt.savefig(out / "fig_exp1_all_methods_ieee.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'fig_exp1_all_methods_ieee.pdf'}")
    print(f"Saved: {out / 'fig_exp1_all_methods_ieee.png'}")


if __name__ == "__main__":
    main()
