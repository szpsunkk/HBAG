"""Global configuration for satellite routing experiments."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Config:
    # Constellations
    CONSTELLATIONS: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "starlink_a": {
                "n_orbits": 4,
                "n_sats_per_orbit": 43,
                "altitude_km": 560.0,
                "inclination_deg": 97.6,
                "name": "Starlink Shell A",
            },
            "starlink_b": {
                "n_orbits": 6,
                "n_sats_per_orbit": 58,
                "altitude_km": 560.0,
                "inclination_deg": 97.6,
                "name": "Starlink Shell B",
            },
            "kuiper": {
                "n_orbits": 28,
                "n_sats_per_orbit": 28,
                "altitude_km": 590.0,
                "inclination_deg": 33.0,
                "name": "Kuiper",
            },
        }
    )

    # Communication: Ka-band RF ISL per paper Table II (f, B, G, P_max).
    P_MAX_W: float = 100.0
    C_MIN_KBPS: float = 10.0
    BANDWIDTH_MHZ: float = 500.0
    CARRIER_FREQ_GHZ: float = 26.0
    THERMAL_NOISE_K: float = 318.0
    ANTENNA_GAIN_DB: float = 30.0
    SPEED_OF_LIGHT: float = 3e8

    # Energy
    # Tuned to avoid pathological "always depleted" behavior while keeping eclipse pressure.
    SOLAR_PANEL_AREA_M2: float = 1.2
    SOLAR_EFFICIENCY: float = 0.29
    SOLAR_IRRADIANCE: float = 1361.0
    BATTERY_MAX_KJ: float = 400.0
    BATTERY_MIN_KJ: float = 40.0
    BATTERY_INIT_FRACTION: float = 0.8
    BASE_PAYLOAD_W: float = 55.0
    TX_EFFICIENCY_FACTOR: float = 1.05

    # Time
    DE_SECONDS: float = 60.0
    DI_SECONDS: float = 1200.0
    TOTAL_PERIOD_S: float = 7200.0

    # Game-theory / optimization
    LAMBDA_ENERGY: float = 0.2
    EPSILON_PENALTY: float = 1e-3
    ECLIPSE_RESERVE_FACTOR: float = 0.7
    ECLIPSE_RESERVE_MARGIN: float = 0.10
    HBAG_LAMBDA_CANDIDATES: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30)
    HBAG_MARGIN_CANDIDATES: Tuple[float, ...] = (0.02, 0.05, 0.08, 0.12, 0.16)
    HBAG_TARGET_ESR: float = 0.80
    ALPHA_WEIGHT: float = 2.0 / 3.0
    BETA_WEIGHT: float = 1.0 / 3.0
    K_MAX: int = 300
    CONVERGENCE_TOL: float = 1e-3

    # Traffic
    TRAFFIC_LEVELS: Dict[str, Dict[str, Tuple[int, int]]] = field(
        default_factory=lambda: {
            "starlink_a": {"low": (10, 20), "medium": (20, 30), "high": (30, 40)},
            "starlink_b": {"low": (20, 40), "medium": (40, 60), "high": (60, 80)},
            "kuiper": {"low": (20, 30), "medium": (30, 40), "high": (40, 50)},
        }
    )
    FLOW_RATES_MBPS: List[float] = field(default_factory=lambda: [4.0, 6.0, 8.0])
    TARGET_DC_RATIO: float = 0.65
    # Eclipse semantics:
    # - "phase_shifted_time_fraction": each satellite has theta eclipse-time ratio with orbital phase shift.
    # - "global_block_time_fraction": all satellites enter eclipse simultaneously for theta tail slots.
    ECLIPSE_SEMANTICS: str = "phase_shifted_time_fraction"

    # Experiment repeats
    N_SEEDS: int = 10
    SEEDS: List[int] = field(default_factory=lambda: list(range(42, 52)))
    ENERGY_DIAGNOSTICS: bool = False


CFG = Config()
