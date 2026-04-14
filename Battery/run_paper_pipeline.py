"""One-command pipeline for reproducible paper experiments and IEEE figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.run_ieee_figures import run_ieee_figures
from run_all_experiments import main as run_full_suite


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_summary(results: dict) -> None:
    exp1 = results.get("exp1", {})
    if not exp1:
        return

    ranking = sorted(
        ((name, vals["ESR"]["mean"]) for name, vals in exp1.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_esr = ranking[0]

    lines = [
        "# Experiment Summary",
        "",
        "## ESR Ranking (higher is better)",
    ]
    for i, (name, esr) in enumerate(ranking, start=1):
        lines.append(f"{i}. {name}: {esr:.3f}%")
    lines.extend(
        [
            "",
            f"Best ESR method: {best_name} ({best_esr:.3f}%)",
            "",
            "Note: optimality is guaranteed only within the tested algorithm set and parameter settings.",
        ]
    )

    out = Path("results/final_summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full paper pipeline.")
    parser.parse_args()

    print("=" * 72)
    print("Paper Pipeline: experiments + IEEE figures + summary")
    print("=" * 72)
    run_full_suite()
    run_ieee_figures()
    all_results = _load_json(Path("results/all_experiment_results.json"))
    _write_summary(all_results)
    print("=" * 72)
    print("Pipeline completed. Check `results/` for artifacts.")
    print("=" * 72)


if __name__ == "__main__":
    main()
