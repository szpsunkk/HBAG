"""Quick Exp3 verification for theta-to-eclipse propagation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algorithms.proposed_hbag import ProposedHBAG
from algorithms.satflow_l import SatflowL
from config import CFG
from experiments.simulate import batch_run


def main() -> None:
    thetas = [0.0, 0.1, 0.3, 0.6]
    for theta in thetas:
        m = batch_run(
            "starlink_a",
            {"SATFLOW": SatflowL, "HBAG": ProposedHBAG},
            [CFG.SEEDS[0]],
            n_steps=90,
            traffic_level="high",
            eclipse_fraction=theta,
        )
        r_sat = m["SATFLOW"][0]
        r_hbag = m["HBAG"][0]
        print(
            f"theta={theta:.1f}: "
            f"SATFLOW_eclipse_slots={r_sat.n_eclipse_events}, "
            f"HBAG_eclipse_slots={r_hbag.n_eclipse_events}, "
            f"SATFLOW_ESR={r_sat.esr:.3f}, "
            f"HBAG_ESR={r_hbag.esr:.3f}"
        )


if __name__ == "__main__":
    main()
