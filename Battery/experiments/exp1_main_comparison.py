"""Experiment 1: main performance comparison (Table I)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.run_table1 import run_table1


def run_experiment() -> dict:
    """Run Table I and return structured metrics."""
    print("[Exp1] Main performance comparison (Table I)")
    data = run_table1()
    print("[Exp1] Done.")
    return data


if __name__ == "__main__":
    run_experiment()
