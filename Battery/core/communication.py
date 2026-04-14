"""Physical-layer inspired communication utility."""

import numpy as np

from config import CFG


def db_to_linear(db: float) -> float:
    return 10 ** (db / 10.0)


def free_space_path_loss_db(distance_km: float, freq_ghz: float) -> float:
    distance_km = max(distance_km, 1e-3)
    return 92.45 + 20 * np.log10(distance_km) + 20 * np.log10(freq_ghz)


def shannon_capacity_kbps(tx_power_w: float, distance_km: float) -> float:
    fspl_db = free_space_path_loss_db(distance_km, CFG.CARRIER_FREQ_GHZ)
    tx_dbw = 10 * np.log10(max(tx_power_w, 1e-9))
    pr_dbw = tx_dbw + CFG.ANTENNA_GAIN_DB - fspl_db
    pr_w = db_to_linear(pr_dbw)

    bw_hz = CFG.BANDWIDTH_MHZ * 1e6
    noise_w = 1.38e-23 * CFG.THERMAL_NOISE_K * bw_hz
    snr = pr_w / max(noise_w, 1e-18)
    cap_bps = bw_hz * np.log2(1 + max(snr, 0.0))
    return float(cap_bps / 1e3)
