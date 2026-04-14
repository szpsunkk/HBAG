from __future__ import annotations

"""Experiment metrics."""

import numpy as np


def esr(served_mbps: np.ndarray, demand_mbps: np.ndarray) -> float:
    return float(np.sum(served_mbps) / max(np.sum(demand_mbps), 1e-9))


def fvr(served_mbps: np.ndarray, demand_mbps: np.ndarray, threshold: float = 0.8) -> float:
    ratio = served_mbps / np.maximum(demand_mbps, 1e-9)
    # Violation ratio: fraction of samples below required service level.
    return float(np.mean(ratio < threshold))


def demand_unmet_ratio(served_mbps: np.ndarray, demand_mbps: np.ndarray) -> float:
    shortfall = np.maximum(demand_mbps - served_mbps, 0.0)
    return float(np.sum(shortfall) / max(np.sum(demand_mbps), 1e-9))


def fvr_battery_induced(
    served_mbps: np.ndarray,
    demand_mbps: np.ndarray,
    battery_depleted_mask: np.ndarray,
    threshold: float = 0.8,
) -> float:
    ratio = served_mbps / np.maximum(demand_mbps, 1e-9)
    violated = ratio < threshold
    induced = violated & battery_depleted_mask
    return float(np.mean(induced))


def ee(served_mbps: np.ndarray, energy_kj: float) -> float:
    return float(np.sum(served_mbps) / max(energy_kj, 1e-9))


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean, 0.0
    half = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return mean, half
