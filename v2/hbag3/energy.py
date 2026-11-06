"""Physical energy accounting and a backward reserve certificate.

No clipping at the safety floor is permitted. The energy update distinguishes
raw energy, physically stored energy, curtailed harvest, and unmet electrical
load. Reserve feasibility is conditional on the stated harvest/load bounds.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Battery:
    capacity_j: float = 400_000.0
    minimum_j: float = 40_000.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95

    def __post_init__(self):
        if not 0 <= self.minimum_j < self.capacity_j:
            raise ValueError("Require 0 <= minimum < capacity")
        if not (0 < self.charge_efficiency <= 1 and 0 < self.discharge_efficiency <= 1):
            raise ValueError("Charge and discharge efficiencies must lie in (0,1]")


@dataclass
class EnergyStep:
    before_j: np.ndarray
    harvest_w: np.ndarray
    base_w: np.ndarray
    terminal_w: np.ndarray
    delta_j: np.ndarray
    raw_after_j: np.ndarray
    after_j: np.ndarray
    spill_j: np.ndarray
    unserved_energy_j: np.ndarray
    safety_margin_j: np.ndarray


def energy_drift(harvest_w, load_w, battery: Battery):
    """Stored energy change per second, with one net DC-bus charge/discharge path."""
    surplus = np.asarray(harvest_w, dtype=float) - np.asarray(load_w, dtype=float)
    return battery.charge_efficiency * np.maximum(surplus, 0) - np.maximum(-surplus, 0) / battery.discharge_efficiency


def step(energy_j, harvest_w, base_w, terminal_w, dt_s: float, battery: Battery) -> EnergyStep:
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    e, h, b, p = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (energy_j, harvest_w, base_w, terminal_w)])
    if any(np.any(~np.isfinite(x)) for x in (e, h, b, p)) or any(np.any(x < 0) for x in (h, b, p)):
        raise ValueError("Inputs must be finite and powers nonnegative")
    delta = dt_s * energy_drift(h, b + p, battery)
    raw = e + delta
    after = np.clip(raw, 0.0, battery.capacity_j)
    return EnergyStep(e.copy(), h.copy(), b.copy(), p.copy(), delta, raw, after,
                      np.maximum(raw - battery.capacity_j, 0.0),
                      np.maximum(-raw, 0.0), raw - battery.minimum_j)


def load_budget(energy_j, target_next_j, harvest_lower_w, dt_s: float, battery: Battery):
    """Largest aggregate electrical load reaching target at the next slot.

    A negative budget is an infeasibility certificate, NOT a zero-safe budget.
    Targets above battery capacity must be handled as infeasible by the caller.
    """
    if np.any(np.asarray(dt_s) <= 0):
        raise ValueError("dt_s must be positive")
    e, target, h = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (energy_j, target_next_j, harvest_lower_w)])
    change = target - e
    return h - np.where(change >= 0.0, change / (battery.charge_efficiency * dt_s),
                        battery.discharge_efficiency * change / dt_s)


def reserve_envelope(harvest_lower_w, mandatory_upper_w, dt_s: float, battery: Battery, terminal_j=None):
    """Minimum stored energy at every slot to support a committed fallback.

    Arrays have shape (time, satellite). The fallback load includes base load,
    tracking idle load, and any explicitly promised minimum communication load.
    Values above capacity are deliberately retained to identify infeasibility.
    """
    h, l = np.broadcast_arrays(np.asarray(harvest_lower_w, dtype=float), np.asarray(mandatory_upper_w, dtype=float))
    if h.ndim == 1:
        h, l = h[:, None], l[:, None]
    if h.ndim != 2 or dt_s <= 0:
        raise ValueError("Expected time-by-satellite bounds and positive dt_s")
    out = np.empty((h.shape[0] + 1, h.shape[1]))
    out[-1] = battery.minimum_j if terminal_j is None else np.asarray(terminal_j, dtype=float)
    if np.any(out[-1] < battery.minimum_j):
        raise ValueError("Terminal reserve cannot be below the safety floor")
    increments = dt_s * energy_drift(h, l, battery)
    for t in range(h.shape[0] - 1, -1, -1):
        out[t] = np.maximum(battery.minimum_j, out[t + 1] - increments[t])
    return out


def periodic_reserve(harvest_lower_w, mandatory_upper_w, dt_s: float, battery: Battery):
    """Least periodic fallback reserve, with explicit energy/capacity feasibility.

    For a periodic lower-bound drift, the required reserve is the largest future
    cumulative deficit over one period. Nonnegative net drift is necessary for
    indefinite fallback operation. This certifies only the stated periodic bounds.
    """
    h, l = np.broadcast_arrays(np.asarray(harvest_lower_w, dtype=float), np.asarray(mandatory_upper_w, dtype=float))
    if h.ndim == 1:
        h, l = h[:, None], l[:, None]
    if h.ndim != 2 or dt_s <= 0:
        raise ValueError("Expected time-by-satellite arrays and positive dt_s")
    drift = dt_s * energy_drift(h, l, battery)
    period = drift.shape[0]
    cumulative = np.vstack([np.zeros((1, drift.shape[1])), np.cumsum(np.concatenate([drift, drift]), axis=0)])
    reserve = np.empty_like(drift)
    for t in range(period):
        deficit = cumulative[t] - np.min(cumulative[t:t + period + 1], axis=0)
        reserve[t] = battery.minimum_j + np.maximum(deficit, 0)
    feasible = (np.sum(drift, axis=0) >= -1e-8) & (np.max(reserve, axis=0) <= battery.capacity_j + 1e-8)
    return np.vstack([reserve, reserve[:1]]), feasible
