"""IEEE-style publication figures generated from experiment outputs."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.deepisl import DeepIsl
from algorithms.maac_iilp import MaacIilp
from algorithms.proposed_hbag import ProposedHBAG
from algorithms.satflow_l import SatflowL
from algorithms.smfg_adapted import SmfgAdapted
from config import CFG
from experiments.simulate import batch_run

warnings.filterwarnings("ignore")

rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "mathtext.fontset": "stix",
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.75",
        "legend.borderpad": 0.35,
        "legend.handlelength": 1.8,
        "legend.labelspacing": 0.25,
    }
)

COLORS = {
    "SatFlow-L": "#1f77b4",
    "MAAC-IILP": "#ff7f0e",
    "DeepISL": "#2ca02c",
    "SMFG-adapted": "#d62728",
    "Proposed-HBAG": "#9467bd",
    "ref": "#7f7f7f",
}


def _save(out_dir: Path, stem: str) -> None:
    plt.tight_layout(pad=0.3)
    plt.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / (stem + '.pdf')}")
    print(f"Saved: {out_dir / (stem + '.png')}")


def _load_all_results() -> dict:
    fp = Path("results/all_experiment_results.json")
    if not fp.exists():
        return {}
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def _plot_soc(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    algo_map = {
        "SatFlow-L": SatflowL,
        "MAAC-IILP": MaacIilp,
        "DeepISL": DeepIsl,
        "SMFG-adapted": SmfgAdapted,
        "Proposed-HBAG": ProposedHBAG,
    }
    # One-orbit profile with explicit eclipse pressure (~35 min eclipse in 90 min orbit).
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
    x = np.arange(n_steps) * CFG.DE_SECONDS / 60.0

    line_spec = {
        "SatFlow-L": ("-", 1.0),
        "MAAC-IILP": ("--", 1.0),
        "DeepISL": ("-.", 1.0),
        "SMFG-adapted": (":", 1.0),
        "Proposed-HBAG": ("-", 1.6),
    }
    hbag_final = None
    for name in ["SatFlow-L", "MAAC-IILP", "DeepISL", "SMFG-adapted", "Proposed-HBAG"]:
        curves = np.vstack([r.soc_curve for r in res_map[name]])
        y = np.mean(curves, axis=0) * 100.0
        y_std = np.std(curves, axis=0) * 100.0
        style, width = line_spec[name]
        ax.plot(x, y, linestyle=style, linewidth=width, color=COLORS[name], label=name, zorder=4)
        if "Proposed" in name:
            ax.fill_between(x, y - y_std, y + y_std, color=COLORS[name], alpha=0.10, linewidth=0)
            hbag_final = float(y[-1])

    cmin_pct = 100.0 * CFG.BATTERY_MIN_KJ / max(CFG.BATTERY_MAX_KJ, 1e-9)
    ax.axhline(cmin_pct, color="#c0392b", ls=":", lw=1.0, label=r"$C^B_{min}$", zorder=5)
    ax.text(2, cmin_pct + 1.2, r"$C^B_{min}$ threshold", fontsize=6.2, color="#c0392b")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Battery State-of-Charge (%)")
    ax.set_xlim(0, float(x[-1]))
    # Leave headroom above 100% so top curves/CI bands are not visually clipped.
    ax.set_ylim(max(0.0, cmin_pct - 5.0), 104.0)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticks([10, 20, 40, 60, 80, 100])
    ax.axvline(55.0, color="#7f8c8d", ls=":", lw=0.9, alpha=0.75)
    ax.text(55.7, max(12.0, cmin_pct + 2.0), "Eclipse start", fontsize=6.0, color="#7f8c8d")
    ax.axvline(90.0, color="#7f8c8d", ls=":", lw=0.9, alpha=0.75)
    if hbag_final is not None:
        ax.text(72.0, hbag_final + 2.0, "HBAG keeps SOC above Cmin", fontsize=6.0, color=COLORS["Proposed-HBAG"])
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="lower right", fontsize=5.9, ncol=2)
    _save(out_dir, "fig_soc_evolution_ieee")


def _plot_convergence(out_dir: Path, all_results: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    algo_map = {"Proposed-HBAG": ProposedHBAG}
    res_map = batch_run("starlink_a", algo_map, CFG.SEEDS[:3], n_steps=220)

    for name in ["Proposed-HBAG"]:
        curves = np.vstack([r.convergence_curve for r in res_map[name]])
        y = np.mean(curves, axis=0) + 1e-8
        max_show = min(200, len(y))
        x = np.arange(1, max_show + 1)
        y = y[:max_show]
        y_norm = y / max(y[0], 1e-12)
        ax.semilogy(x, y_norm, linestyle="-", linewidth=0.8, color=COLORS[name], alpha=0.35, label=f"{name} (raw)")
        if len(y_norm) >= 5:
            sm = np.convolve(y_norm, np.ones(5) / 5, mode="valid")
            ys = np.concatenate([np.full(4, sm[0]), sm])
            ax.semilogy(x, ys, linestyle="-", linewidth=1.7, color=COLORS[name], label=f"{name} (smoothed)")
            y0 = ys[0]
        else:
            y0 = y_norm[0]
        ref = y0 / np.sqrt(x)
        ax.semilogy(x, ref, "--", color=COLORS["ref"], linewidth=0.9, label=r"$\mathcal{O}(1/\sqrt{k})$")
        plateau = max(float(np.percentile(y_norm[-20:], 50)), 1e-5)
        ax.semilogy(x, np.full_like(x, plateau, dtype=float), ":", color="#8e44ad", linewidth=0.9, label="Convergence plateau")
    ax.set_xlabel(r"Iteration $k$")
    ax.set_ylabel("Normalized Residual")
    ax.set_xlim(1, 200)
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2, 3, 5], numticks=10))
    ax.grid(True, which="both", linestyle=":", alpha=0.3, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=6.5)
    _save(out_dir, "fig_convergence_ieee")


def _plot_esr_eclipse(out_dir: Path, all_results: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    exp3 = all_results.get("exp3", {})
    theta = np.array(exp3.get("theta", [0.0, 0.1, 0.2, 0.3, 0.38, 0.5, 0.6]), dtype=float)
    res_mean = {
        "SatFlow-L": np.array(exp3.get("SATFLOW", []), dtype=float) * 100.0,
        "MAAC-IILP": np.array(exp3.get("MAAC-IILP", []), dtype=float) * 100.0,
        "DeepISL": np.array(exp3.get("DeepISL", []), dtype=float) * 100.0,
        "SMFG-adapted": np.array(exp3.get("SMFG-adapted", []), dtype=float) * 100.0,
        "Proposed-HBAG": np.array(exp3.get("Proposed (HBAG)", []), dtype=float) * 100.0,
    }
    res_ci = {k: np.zeros_like(theta, dtype=float) for k in res_mean}

    for name, marker, style in [
        ("SatFlow-L", "o", "-"),
        ("MAAC-IILP", "s", "--"),
        ("DeepISL", "^", "-."),
        ("SMFG-adapted", "D", ":"),
        ("Proposed-HBAG", "v", "-"),
    ]:
        width = 1.7 if "Proposed" in name else 1.1
        size = 6 if "Proposed" in name else 4.5
        y = np.array(res_mean[name], dtype=float)
        yerr = np.array(res_ci[name], dtype=float)
        assert len(y) == len(theta), f"{name} length mismatch: {len(y)} vs {len(theta)}"
        ax.plot(theta, y, color=COLORS[name], marker=marker, linestyle=style, linewidth=width, markersize=size, label=name)
        ax.fill_between(theta, y - yerr, y + yerr, color=COLORS[name], alpha=0.13, linewidth=0)

    # As suggested in improve.md, explicitly show the widening gap PG - SATFLOW.
    gap = np.array(res_mean["Proposed-HBAG"]) - np.array(res_mean["SatFlow-L"])
    ax2 = ax.twinx()
    ax2.plot(theta, gap, color="#2c3e50", linestyle=":", marker="d", markersize=4, linewidth=1.1, label="Gap: HBAG - SATFLOW")
    ax2.set_ylabel("ESR Gap (%)", color="#2c3e50")
    ax2.tick_params(direction="in", colors="#2c3e50")

    ax.set_xlabel(r"Eclipse Fraction $\theta$")
    ax.set_ylabel("Energy Sustainability Rate (%)")
    ax.set_xlim(-0.02, 0.62)
    ax.set_xticks(theta)
    y_min = min(min(v) for v in res_mean.values())
    y_max = max(max(v) for v in res_mean.values())
    ax.set_ylim(max(0.0, y_min - 4.0), min(104.0, y_max + 6.0))
    ax.tick_params(direction="in", top=True, right=True)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower right", bbox_to_anchor=(1.0, 0.27), fontsize=5.7, ncol=2)
    _save(out_dir, "fig_esr_vs_eclipse_ieee")


def _plot_runtime(out_dir: Path, all_results: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    exp5 = all_results.get("exp5", {})
    n_sats = np.array(exp5.get("n_sats", [172, 348, 784]), dtype=float)
    rt = np.array(exp5.get("runtime_per_slot_s", [0.0016, 0.0030, 0.0065]), dtype=float) * 1000.0
    x = np.arange(len(n_sats))
    bars = ax.bar(x, rt, width=0.52, color=COLORS["Proposed-HBAG"], alpha=0.84, edgecolor="black", linewidth=0.5, zorder=3, label="Measured runtime")
    ref = rt[0] * n_sats / n_sats[0]
    ax.plot(x, ref, "k--o", linewidth=1.0, markersize=5, label=r"$\mathcal{O}(n_t)$ reference")
    for bar, val in zip(bars, rt):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.85),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in n_sats], fontsize=7)
    ax.set_xlabel("Number of Satellites")
    ax.set_ylabel("Runtime per slot (ms)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="upper left", fontsize=6.5)
    _save(out_dir, "fig_runtime_ieee")


def _plot_performance_bar(out_dir: Path, all_results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    exp1 = all_results.get("exp1", {})
    preferred = ["SatFlow-L", "MAAC-IILP", "DeepISL", "SMFG-adapted", "Proposed-HBAG"]
    methods = [m for m in preferred if m in exp1]
    labels_map = {
        "SatFlow-L": "SATFLOW",
        "MAAC-IILP": "MAAC-IILP",
        "DeepISL": "DeepISL",
        "SMFG-adapted": "SMFG",
        "Proposed-HBAG": "HBAG",
    }
    labels = [labels_map[m] for m in methods]
    if not methods:
        plt.close()
        return
    x = np.arange(len(methods))
    esr_m = np.array([exp1[m]["ESR"]["mean"] for m in methods], dtype=float)
    esr_ci = np.array([exp1[m]["ESR"]["ci"] for m in methods], dtype=float)
    ee_m = np.array([exp1[m]["EE_mbit_per_kj"]["mean"] for m in methods], dtype=float)
    ee_ci = np.array([exp1[m]["EE_mbit_per_kj"]["ci"] for m in methods], dtype=float)
    colors = [COLORS[m] for m in methods]

    b1 = axes[0].bar(x, esr_m, width=0.62, color=colors, alpha=0.86, edgecolor="black", linewidth=0.4)
    axes[0].errorbar(x, esr_m, yerr=esr_ci, fmt="none", color="black", capsize=2, linewidth=0.7, capthick=0.7)
    b2 = axes[1].bar(x, ee_m, width=0.62, color=colors, alpha=0.86, edgecolor="black", linewidth=0.4)
    axes[1].errorbar(x, ee_m, yerr=ee_ci, fmt="none", color="black", capsize=2, linewidth=0.7, capthick=0.7)

    for i in [4]:
        b1[i].set_hatch("//")
        b1[i].set_linewidth(1.0)
        b2[i].set_hatch("//")
        b2[i].set_linewidth(1.0)

    axes[0].set_title("(a) ESR", fontsize=8)
    axes[1].set_title("(b) EE", fontsize=8)
    axes[0].set_ylabel("ESR (%)")
    axes[1].set_ylabel("Energy Efficiency (Mbit/kJ)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.3, rotation=10, ha="right")
        ax.tick_params(direction="in", top=True, right=True)

    plt.tight_layout(pad=0.3, w_pad=0.7)
    plt.savefig(out_dir / "fig_performance_bar_ieee.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "fig_performance_bar_ieee.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'fig_performance_bar_ieee.pdf'}")
    print(f"Saved: {out_dir / 'fig_performance_bar_ieee.png'}")


def run_ieee_figures() -> None:
    out_dir = Path("results/figures/ieee")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = _load_all_results()
    print("Generating IEEE-quality figures from experiment outputs ...")
    _plot_soc(out_dir)
    _plot_convergence(out_dir, all_results)
    _plot_esr_eclipse(out_dir, all_results)
    _plot_runtime(out_dir, all_results)
    if "exp1" in all_results:
        _plot_performance_bar(out_dir, all_results)
    print("\nAll figures saved to results/figures/ieee/")


if __name__ == "__main__":
    run_ieee_figures()
