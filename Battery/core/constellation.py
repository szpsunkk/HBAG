"""Simple Walker-like constellation model."""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Satellite:
    sat_id: int
    orbit_id: int
    pos_in_orbit: int
    altitude_km: float
    inclination_deg: float
    n_orbits: int
    n_sats_per_orbit: int

    def __post_init__(self) -> None:
        r_earth_km = 6371.0
        gm = 398600.4418  # km^3/s^2
        r = r_earth_km + self.altitude_km
        self.orbital_period_s = 2 * np.pi * np.sqrt((r**3) / gm)
        self.position_ecef = np.zeros(3, dtype=float)
        self.illuminated = True
        self.sun_angle_rad = 0.0

    def compute_position(self, time_s: float) -> np.ndarray:
        r_earth_km = 6371.0
        r = r_earth_km + self.altitude_km
        raan_0 = 2 * np.pi * self.orbit_id / max(self.n_orbits, 1)
        ma_0 = 2 * np.pi * self.pos_in_orbit / max(self.n_sats_per_orbit, 1)
        mean_motion = 2 * np.pi / self.orbital_period_s
        true_anomaly = ma_0 + mean_motion * time_s
        inc = np.deg2rad(self.inclination_deg)

        x_orb = r * np.cos(true_anomaly)
        y_orb = r * np.sin(true_anomaly)

        earth_rotation = 7.2921150e-5
        theta_earth = earth_rotation * time_s

        x_eci = x_orb * np.cos(raan_0) - y_orb * np.cos(inc) * np.sin(raan_0)
        y_eci = x_orb * np.sin(raan_0) + y_orb * np.cos(inc) * np.cos(raan_0)
        z_eci = y_orb * np.sin(inc)

        x_ecef = x_eci * np.cos(theta_earth) + y_eci * np.sin(theta_earth)
        y_ecef = -x_eci * np.sin(theta_earth) + y_eci * np.cos(theta_earth)
        z_ecef = z_eci

        self.position_ecef = np.array([x_ecef, y_ecef, z_ecef], dtype=float)
        return self.position_ecef

    def compute_illumination(self, time_s: float) -> None:
        earth_orbit_period = 365.25 * 24 * 3600
        sun_angle = 2 * np.pi * time_s / earth_orbit_period
        sun_dir = np.array([np.cos(sun_angle), np.sin(sun_angle), 0.0], dtype=float)

        sat_pos = self.position_ecef
        sat_dist = float(np.linalg.norm(sat_pos)) + 1e-12
        proj = float(np.dot(sat_pos, sun_dir))
        r_earth_km = 6371.0

        if proj < 0:
            perp_dist = np.sqrt(max(sat_dist**2 - proj**2, 0.0))
            self.illuminated = bool(perp_dist > r_earth_km)
        else:
            self.illuminated = True

        if self.illuminated:
            panel_normal = sat_pos / sat_dist
            self.sun_angle_rad = float(np.arccos(np.clip(np.dot(panel_normal, sun_dir), -1, 1)))
        else:
            self.sun_angle_rad = 0.0


class Constellation:
    def __init__(
        self,
        n_orbits: int,
        n_sats_per_orbit: int,
        altitude_km: float,
        inclination_deg: float,
    ) -> None:
        self.n_orbits = n_orbits
        self.n_sats_per_orbit = n_sats_per_orbit
        self.satellites: List[Satellite] = []

        sat_id = 0
        for o in range(n_orbits):
            for p in range(n_sats_per_orbit):
                self.satellites.append(
                    Satellite(
                        sat_id=sat_id,
                        orbit_id=o,
                        pos_in_orbit=p,
                        altitude_km=altitude_km,
                        inclination_deg=inclination_deg,
                        n_orbits=n_orbits,
                        n_sats_per_orbit=n_sats_per_orbit,
                    )
                )
                sat_id += 1

    def update(self, time_s: float) -> None:
        for sat in self.satellites:
            sat.compute_position(time_s)
            sat.compute_illumination(time_s)
