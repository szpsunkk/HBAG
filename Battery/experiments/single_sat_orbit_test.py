"""Single-satellite one-orbit battery sanity test."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import CFG


def collect_single_sat_orbit_diagnostics() -> dict:
    cb_max = CFG.BATTERY_MAX_KJ
    cb_min = CFG.BATTERY_MIN_KJ
    cb = CFG.BATTERY_INIT_FRACTION * cb_max
    d_e = CFG.DE_SECONDS
    sigma = np.radians(60.0)
    samples = []

    for t_min in range(90):
        phi = 1.0 if t_min < 55 else 0.0
        p_h = phi * CFG.SOLAR_EFFICIENCY * CFG.SOLAR_IRRADIANCE * CFG.SOLAR_PANEL_AREA_M2 * np.sin(sigma)
        e_h = p_h * d_e / 1000.0

        # Worst-case full-power transmission.
        p_c = CFG.BASE_PAYLOAD_W + CFG.TX_EFFICIENCY_FACTOR * CFG.P_MAX_W
        e_c = p_c * d_e / 1000.0
        cb = float(np.clip(cb + e_h - e_c, 0.0, cb_max))

        if t_min in (0, 54, 55, 89):
            status = "OK" if cb > cb_min else "DEPLETED"
            samples.append(
                {
                    "t_min": int(t_min),
                    "phi": int(phi),
                    "e_h_kj": float(e_h),
                    "e_c_kj": float(e_c),
                    "cb_kj": float(cb),
                    "status": status,
                }
            )

    return {
        "cb_max_kj": float(cb_max),
        "cb_min_kj": float(cb_min),
        "cb_init_kj": float(CFG.BATTERY_INIT_FRACTION * cb_max),
        "samples": samples,
        "final_cb_kj": float(cb),
    }


def main() -> None:
    data = collect_single_sat_orbit_diagnostics()
    print("=== SINGLE SAT ORBIT TEST ===")
    print(
        f"CB_max={data['cb_max_kj']:.1f} kJ, CB_min={data['cb_min_kj']:.1f} kJ, "
        f"init={data['cb_init_kj']:.1f} kJ"
    )
    for row in data["samples"]:
        print(
            f"t={row['t_min']:2d} phi={row['phi']:.0f} "
            f"E_h={row['e_h_kj']:6.2f} E_c={row['e_c_kj']:6.2f} "
            f"CB={row['cb_kj']:7.1f} {row['status']}"
        )
    print(f"\nFinal CB = {data['final_cb_kj']:.1f} kJ")


if __name__ == "__main__":
    main()
