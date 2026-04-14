"""Quick energy-balance diagnostics requested in improve.md."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import CFG


def collect_energy_diagnostics() -> dict:
    sigma = np.radians(60.0)
    p_harvest = CFG.SOLAR_EFFICIENCY * CFG.SOLAR_IRRADIANCE * CFG.SOLAR_PANEL_AREA_M2 * np.sin(sigma)
    p_consume_max = CFG.BASE_PAYLOAD_W + CFG.TX_EFFICIENCY_FACTOR * CFG.P_MAX_W
    d_e = CFG.DE_SECONDS

    e_h = p_harvest * d_e / 1000.0
    e_c = p_consume_max * d_e / 1000.0
    net = e_h - e_c
    denom = max(e_c - e_h, 1e-9)
    depletion_slots = None if net >= 0 else (CFG.BATTERY_MAX_KJ / denom)
    return {
        "p_harvest_w": float(p_harvest),
        "p_consume_max_w": float(p_consume_max),
        "e_harvest_per_slot_kj": float(e_h),
        "e_consume_per_slot_kj": float(e_c),
        "net_per_slot_kj": float(net),
        "cb_max_kj": float(CFG.BATTERY_MAX_KJ),
        "depletion_slots_worst_case": None if depletion_slots is None else float(depletion_slots),
        "depletion_status": "charging" if net >= 0 else "depleting",
    }


def main() -> None:
    data = collect_energy_diagnostics()

    print("=== ENERGY DIAGNOSTICS ===")
    print(f"P_harvest: {data['p_harvest_w']:.2f} W")
    print(f"P_consume_max: {data['p_consume_max_w']:.2f} W")
    print(f"E_harvest/slot: {data['e_harvest_per_slot_kj']:.3f} kJ")
    print(f"E_consume/slot: {data['e_consume_per_slot_kj']:.3f} kJ")
    print(f"Net/slot: {data['net_per_slot_kj']:.3f} kJ")
    print(f"CB_max: {data['cb_max_kj']:.1f} kJ")
    if data["depletion_slots_worst_case"] is not None:
        print(f"Depletion time: {data['depletion_slots_worst_case']:.1f} slots (worst-case)")
    else:
        print("Depletion time: INF (charging in illuminated condition)")


if __name__ == "__main__":
    main()
