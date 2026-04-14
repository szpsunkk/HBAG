from __future__ import annotations

"""Shared simulator for all experiment scripts."""

from dataclasses import dataclass
import os
import sys
from typing import Dict, Iterable, Type

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.base import RoutingAlgorithm
from config import CFG
from core.communication import shannon_capacity_kbps
from core.constellation import Constellation
from core.energy_model import EnergyModel
from core.traffic import TrafficModel
from utils.metrics import demand_unmet_ratio, ee, fvr, fvr_battery_induced


@dataclass
class SimResult:
    esr: float
    fvr: float
    ee: float
    demand_unmet: float
    fvr_battery: float
    soc_curve: np.ndarray
    runtime_s: float
    convergence_curve: np.ndarray
    theta_actual: float
    demand_capacity_ratio: float
    n_eclipse_events: int


def simulate_once(
    constellation_key: str,
    algorithm_cls: Type[RoutingAlgorithm],
    seed: int,
    n_steps: int = 40,
    traffic_level: str = None,
    eclipse_fraction: float = 0.0,
) -> SimResult:
    import time

    c = CFG.CONSTELLATIONS[constellation_key]
    constellation = Constellation(
        int(c["n_orbits"]),
        int(c["n_sats_per_orbit"]),
        float(c["altitude_km"]),
        float(c["inclination_deg"]),
    )
    n_nodes = len(constellation.satellites)
    traffic = TrafficModel(constellation_key, seed=seed)
    energy = EnergyModel(n_nodes)
    algo = algorithm_cls()

    served_all, demand_all = [], []
    battery_ok_all = []
    battery_depleted_all = []
    soc_curve, conv = [], []
    theta_all = []
    ratio_all = []
    prev_p = np.zeros(n_nodes, dtype=float)
    eclipse_fraction = float(np.clip(eclipse_fraction, 0.0, 1.0))
    eclipse_events = 0

    orbit_slots = max(1, int(round(90 * 60 / max(CFG.DE_SECONDS, 1e-9))))
    sat_phase = np.arange(n_nodes, dtype=float) / max(n_nodes, 1)
    t0 = time.perf_counter()
    for s in range(n_steps):
        if CFG.ECLIPSE_SEMANTICS == "phase_shifted_time_fraction":
            eclipse_slots = int(round(eclipse_fraction * orbit_slots))
            local_slot = s % orbit_slots
            sat_slot = (local_slot + np.floor(sat_phase * orbit_slots).astype(int)) % orbit_slots
            in_eclipse_mask = sat_slot >= (orbit_slots - eclipse_slots)
        else:
            n_eclipse_slots = int(round(eclipse_fraction * n_steps))
            in_eclipse_mask = np.full(n_nodes, s >= (n_steps - n_eclipse_slots), dtype=bool)
        t = s * CFG.DE_SECONDS
        constellation.update(t)
        snap = traffic.sample(n_nodes, level=traffic_level)
        soc = energy.soc()
        distance = np.array([np.linalg.norm(sat.position_ecef) for sat in constellation.satellites])
        cap_upper_mbps = np.array([shannon_capacity_kbps(CFG.P_MAX_W, di) for di in distance]) / 1000.0
        raw_demand = snap.offered_mbps
        scale = (
            CFG.TARGET_DC_RATIO * float(np.sum(cap_upper_mbps)) / max(float(np.sum(raw_demand)), 1e-9)
        )
        offered_mbps = raw_demand * scale

        p = algo.allocate_power(offered_mbps, soc, distance, prev_power_w=prev_p)
        # Physical guard: hard cutoff only when battery is physically empty (applies to all algorithms equally).
        battery_kj_before = np.asarray([st.battery_kj for st in energy.states], dtype=float)
        battery_scale = np.where(battery_kj_before > 1e-6, 1.0, 0.0)
        p = p * battery_scale
        caps_kbps = np.array([shannon_capacity_kbps(pi, di) for pi, di in zip(p, distance)])
        served_mbps = np.minimum(offered_mbps, caps_kbps / 1000.0)

        illuminated = np.array([sat.illuminated for sat in constellation.satellites], dtype=bool)
        sun_angles = np.array([sat.sun_angle_rad for sat in constellation.satellites], dtype=float)
        # Theta controls eclipse time fraction over the whole simulation horizon.
        illuminated[:] = ~in_eclipse_mask
        sun_angles[:] = np.where(in_eclipse_mask, 0.0, np.deg2rad(60.0))
        eclipse_events += int(np.sum(in_eclipse_mask))
        theta_all.append(1.0 - float(np.mean(illuminated.astype(float))))
        total_demand = float(np.sum(offered_mbps))
        total_capacity = float(np.sum(cap_upper_mbps)) + 1e-9
        ratio_all.append(total_demand / total_capacity)
        energy.step(CFG.DE_SECONDS, illuminated, sun_angles, p)
        battery_kj = np.asarray([st.battery_kj for st in energy.states], dtype=float)
        if s == 0 and CFG.ENERGY_DIAGNOSTICS:
            p_h0 = energy.solar_input_w(bool(illuminated[0]), float(sun_angles[0]))
            p_c0 = energy.tx_consumption_w(float(p[0]))
            e_h0 = p_h0 * CFG.DE_SECONDS / 1000.0
            e_c0 = p_c0 * CFG.DE_SECONDS / 1000.0
            print("=== ENERGY MODEL DIAGNOSTICS ===")
            print(f"CB_min = {CFG.BATTERY_MIN_KJ:.1f} kJ")
            print(f"CB_max = {CFG.BATTERY_MAX_KJ:.1f} kJ")
            print(f"CB_min/CB_max = {CFG.BATTERY_MIN_KJ / max(CFG.BATTERY_MAX_KJ, 1e-9):.3f}")
            print("\nSlot 0:")
            print(f"  phi_0 = {int(bool(illuminated[0]))}")
            print(f"  P_harvest_0 = {p_h0:.2f} W")
            print(f"  P_consume_0 = {p_c0:.2f} W")
            print(f"  E_harvest_0 = {e_h0:.3f} kJ")
            print(f"  E_consume_0 = {e_c0:.3f} kJ")
            print(f"  Net_0 = {e_h0 - e_c0:.3f} kJ")
        battery_ok_all.append(battery_kj > CFG.BATTERY_MIN_KJ)
        battery_depleted_all.append(battery_kj <= CFG.BATTERY_MIN_KJ)
        soc_curve.append(float(np.mean(energy.soc())))
        conv.append(float(np.mean(np.abs(p - prev_p))))
        prev_p = p

        served_all.append(served_mbps)
        demand_all.append(offered_mbps)

    elapsed = time.perf_counter() - t0

    served = np.concatenate(served_all)
    demand = np.concatenate(demand_all)
    battery_ok = np.concatenate(battery_ok_all)
    battery_depleted = np.concatenate(battery_depleted_all)
    total_energy_kj = float(sum(st.consumed_kj for st in energy.states))
    return SimResult(
        # ESR is battery sustainability rate: fraction of (satellite, time) above C_min.
        esr=float(np.mean(battery_ok)),
        fvr=fvr(served, demand),
        ee=ee(served, total_energy_kj),
        demand_unmet=demand_unmet_ratio(served, demand),
        fvr_battery=fvr_battery_induced(served, demand, battery_depleted),
        soc_curve=np.asarray(soc_curve),
        runtime_s=elapsed,
        convergence_curve=np.asarray(conv),
        theta_actual=float(np.mean(theta_all)),
        demand_capacity_ratio=float(np.mean(ratio_all)),
        n_eclipse_events=int(eclipse_events),
    )


def batch_run(
    constellation_key: str,
    algorithm_map: Dict[str, Type[RoutingAlgorithm]],
    seeds: Iterable[int],
    n_steps: int = 40,
    traffic_level: str = None,
    eclipse_fraction: float = 0.0,
) -> Dict[str, list[SimResult]]:
    out: Dict[str, list[SimResult]] = {}
    for name, cls in algorithm_map.items():
        out[name] = [
            simulate_once(
                constellation_key=constellation_key,
                algorithm_cls=cls,
                seed=seed,
                n_steps=n_steps,
                traffic_level=traffic_level,
                eclipse_fraction=eclipse_fraction,
            )
            for seed in seeds
        ]
    return out
