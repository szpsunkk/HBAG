"""Circular-orbit geometry, calibrated optical service, and simple path catalogues.

This is a synthetic Walker-like shell, not an operational constellation trace.
Optical service extrapolates a published design point; idle power and beam
divergence are declared modelling assumptions rather than hardware measurements.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

EARTH_KM = 6371.0
MU_KM3_S2 = 398600.4418
EARTH_RATE = 2 * np.pi / 86164.0905
RATE_BPS = 1e8


@dataclass(frozen=True)
class Shell:
    planes: int = 32
    per_plane: int = 32
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    # Full RAAN circle. A non-full circle must be described as a regional sector.
    raan_span_deg: float = 360.0

    @property
    def size(self):
        return self.planes * self.per_plane

    @property
    def period_s(self):
        return 2 * np.pi * np.sqrt((EARTH_KM + self.altitude_km) ** 3 / MU_KM3_S2)

    def positions(self, time_s):
        p, s = np.meshgrid(np.arange(self.planes), np.arange(self.per_plane), indexing="ij")
        node = np.deg2rad(self.raan_span_deg) * p / self.planes
        phase = 2 * np.pi * (s / self.per_plane + p / self.size + time_s / self.period_s)
        inc = np.deg2rad(self.inclination_deg)
        c, u = np.cos(phase), np.sin(phase)
        xyz = np.stack([np.cos(node)*c - np.sin(node)*u*np.cos(inc),
                        np.sin(node)*c + np.cos(node)*u*np.cos(inc), u*np.sin(inc)], axis=-1)
        return (EARTH_KM + self.altitude_km) * xyz.reshape(-1, 3)


@dataclass(frozen=True)
class OpticalTerminal:
    active_w: float = 35.0
    idle_w: float = 5.0  # Assumption: source does not report tracking-idle power.
    optical_w: float = 1.0
    wavelength_m: float = 1553e-9
    aperture_m: float = 0.02
    photons_per_bit: float = 500.0
    design_rate_bps: float = 1e8
    design_range_km: float = 1000.0
    maximum_range_km: float = 1500.0
    beam_half_angle_rad: float = 100e-6  # Assumption, used only to expose calibration.
    extra_loss_db: float = 0.0
    payload_availability: float = 1.0

    def __post_init__(self):
        if not 0 <= self.idle_w <= self.active_w or not 0 < self.payload_availability <= 1:
            raise ValueError("Invalid terminal electrical power or availability")

    @property
    def photon_j(self):
        return 6.62607015e-34 * 299792458.0 / self.wavelength_m

    @property
    def calibrated_efficiency(self):
        spot = self.design_range_km * 1000 * self.beam_half_angle_rad
        capture = 1 - np.exp(-2 * (self.aperture_m / 2 / spot) ** 2)
        return self.design_rate_bps * self.photons_per_bit * self.photon_j / (self.optical_w * capture)

    def capacity(self, distance_km):
        distance = np.asarray(distance_km, dtype=float)
        if np.any(distance <= 0):
            raise ValueError("Link distance must be positive")
        spot = distance * 1000 * self.beam_half_angle_rad
        capture = -np.expm1(-2 * (self.aperture_m / 2 / spot) ** 2)
        receive_w = self.optical_w * capture * self.calibrated_efficiency * 10**(-self.extra_loss_db/10)
        raw = receive_w / (self.photons_per_bit * self.photon_j)
        rate = self.payload_availability * np.minimum(raw, self.design_rate_bps)
        return np.where(distance <= self.maximum_range_km, rate, 0.)


def sunlit(positions, sun_direction=(1., 0., 0.)):
    """Cylindrical umbra for a fixed far-field Sun; no penumbra or attitude model."""
    sun = np.asarray(sun_direction, dtype=float)
    sun /= np.linalg.norm(sun)
    along = np.asarray(positions) @ sun
    radial2 = np.sum(np.asarray(positions)**2, axis=1) - along**2
    return (along >= 0) | (radial2 >= EARTH_KM**2)


def slot_harvest_lower(shell, time_s, dt_s, solar_w, sun_direction=(1., 0., 0.)):
    """Zero harvest if ANY part of this slot intersects the cylindrical umbra.

    Exact minimization of the sinusoidal Sun-axis coordinate avoids an invalid
    endpoint-only sunlight certificate at eclipse entry. Slots shorter than one
    orbital period are required. Solar output when fully lit is an assumption.
    """
    if not 0 < dt_s <= shell.period_s:
        raise ValueError("Slot must lie in (0, orbital period]")
    sun = np.asarray(sun_direction, dtype=float)
    sun = sun/np.linalg.norm(sun)
    a = shell.positions(time_s) @ sun
    b = shell.positions(time_s + shell.period_s/4) @ sun
    angle = 2*np.pi*dt_s/shell.period_s
    final = a*np.cos(angle)+b*np.sin(angle)
    minimum_phase = (np.arctan2(b, a)+np.pi) % (2*np.pi)
    low = np.minimum(a, final)
    low = np.where(minimum_phase <= angle + 1e-13, -np.hypot(a,b), low)
    threshold = -np.sqrt((EARTH_KM+shell.altitude_km)**2-EARTH_KM**2)
    return solar_w*(low >= threshold)


def line_of_sight(a, b):
    delta = b - a
    t = np.clip(-np.sum(a*delta, axis=-1)/np.sum(delta*delta, axis=-1), 0, 1)
    nearest = a + t[..., None]*delta
    return np.linalg.norm(nearest, axis=-1) > EARTH_KM


@dataclass
class Snapshot:
    positions: np.ndarray
    edges: np.ndarray
    distance_km: np.ndarray
    capacity_bps: np.ndarray
    degree: np.ndarray
    graph: sparse.csr_matrix


def snapshot(shell, time_s, terminal, duration_s=0.):
    """Two along-plane terminals and one matched terminal per adjacent plane.

    Adjacent planes use the cyclic index shift minimizing total squared distance.
    Every proposed undirected edge is then checked for range and Earth occultation.
    There are at most four active links per node. Acquisition is not optimized.
    """
    xyz = shell.positions(time_s)
    ids = np.arange(shell.size).reshape(shell.planes, shell.per_plane)
    pairs = [(ids.ravel(), np.roll(ids, -1, axis=1).ravel())]
    last = shell.planes if shell.raan_span_deg == 360 else shell.planes - 1
    for p in range(last):
        left, right = ids[p], ids[(p+1) % shell.planes]
        costs = [np.sum((xyz[left] - xyz[np.roll(right, k)])**2) for k in range(shell.per_plane)]
        pairs.append((left, np.roll(right, int(np.argmin(costs)))))
    edges = np.vstack([np.column_stack(v) for v in pairs])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    dist = np.linalg.norm(xyz[edges[:, 0]]-xyz[edges[:, 1]], axis=1)
    if duration_s:
        if not 0 < duration_s <= shell.period_s/2:
            raise ValueError("Geometry envelope requires at most half an orbit")
        quarter = shell.positions(time_s+shell.period_s/4)
        a = xyz[edges[:,0]]-xyz[edges[:,1]]
        b = quarter[edges[:,0]]-quarter[edges[:,1]]
        aa, bb = np.sum(a*a,axis=1), np.sum(b*b,axis=1)
        center, cosine = (aa+bb)/2, (aa-bb)/2
        sine = np.sum(a*b,axis=1)
        angle = 4*np.pi*duration_s/shell.period_s
        peak_phase = np.arctan2(sine,cosine) % (2*np.pi)
        maximum = np.maximum(aa,center+cosine*np.cos(angle)+sine*np.sin(angle))
        maximum = np.where(peak_phase <= angle+1e-13,center+np.hypot(cosine,sine),maximum)
        dist = np.sqrt(np.maximum(maximum,0))
    capacity = terminal.capacity(dist)
    good = (capacity > 0) & line_of_sight(xyz[edges[:, 0]], xyz[edges[:, 1]])
    edges, dist, capacity = edges[good], dist[good], capacity[good]
    degree = np.bincount(edges.ravel(), minlength=shell.size)
    assert np.max(degree, initial=0) <= 4
    # Path selection uses propagation distance; allocation uses shared capacities.
    graph = sparse.coo_matrix((np.tile(dist, 2), (edges.T.ravel(), edges[:, ::-1].T.ravel())),
                              shape=(shell.size, shell.size)).tocsr()
    return Snapshot(xyz, edges, dist, capacity, degree, graph)


def ground_positions(longitude_deg, latitude_deg, time_s):
    lon = np.deg2rad(longitude_deg) + EARTH_RATE*time_s
    lat = np.deg2rad(latitude_deg)
    return EARTH_KM*np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


def attach_ground(ground_xyz, satellite_xyz, minimum_elevation_deg=10.0):
    """Nearest visible satellite; -1 means the ground endpoint has no access."""
    ground = np.asarray(ground_xyz)
    # Chord range is monotone in elevation for this common-altitude shell.
    dot = ground @ satellite_xyz.T
    index = np.argmax(dot, axis=1)
    delta = satellite_xyz[index] - ground
    sine = np.sum(delta*ground, axis=1)/(np.linalg.norm(delta, axis=1)*EARTH_KM)
    return np.where(sine >= np.sin(np.deg2rad(minimum_elevation_deg)), index, -1)


@dataclass
class PathCatalogue:
    path_mask: np.ndarray
    paths: list
    link_activity: sparse.csr_matrix
    node_power_w: sparse.csr_matrix
    hop_count: np.ndarray


def path_catalogue(net, sources, destinations, terminal, count=3):
    """One distance-shortest path plus reproducible perturbed shortest paths.

    Alternatives use fixed node-pair hashes, not battery or future traffic data.
    Duplicates are masked. This is a finite catalogue, not an all-path optimum.
    """
    sources, destinations = np.asarray(sources), np.asarray(destinations)
    mask = np.zeros((len(sources), count), dtype=bool)
    paths = [[None]*count for _ in sources]
    hops = np.zeros_like(mask, dtype=int)
    valid = (sources >= 0) & (destinations >= 0)
    unique = np.unique(sources[valid])
    source_row = {int(v): i for i, v in enumerate(unique)}
    edge_index = {tuple(edge): i for i, edge in enumerate(net.edges)}
    rows, cols, values = [], [], []
    for k in range(count):
        # Hash yields stable [0,1) penalties without consuming the traffic RNG.
        a, b = net.edges.T
        perturb = ((a*73856093 + b*19349663 + k*83492791) % 104729)/104729.
        weights = net.distance_km * (1 if k == 0 else (1 + 2.0*perturb))
        graph = sparse.coo_matrix((np.tile(weights, 2), (net.edges.T.ravel(), net.edges[:, ::-1].T.ravel())),
                                   shape=net.graph.shape).tocsr()
        if not len(unique):
            break
        distances, pred = dijkstra(graph, directed=False, indices=unique, return_predecessors=True)
        for f in np.flatnonzero(valid):
            src, dst = int(sources[f]), int(destinations[f])
            if src == dst:  # Intrasatellite ground traffic is not ISL service.
                continue
            r = source_row[src]
            if not np.isfinite(distances[r, dst]):
                continue
            route = [dst]
            while route[-1] != src:
                route.append(int(pred[r, route[-1]]))
                if route[-1] < 0 or len(route) > len(net.positions):
                    raise RuntimeError("Invalid shortest-path predecessor chain")
            route = tuple(route[::-1])
            if route in paths[f]:
                continue
            paths[f][k], mask[f, k], hops[f, k] = route, True, len(route)-1
            for u, v in zip(route[:-1], route[1:]):
                e = edge_index[tuple(sorted((u, v)))]
                rows.append(e); cols.append(f*count+k); values.append(RATE_BPS/net.capacity_bps[e])
    activity = sparse.coo_matrix((values, (rows, cols)), shape=(len(net.edges), mask.size)).tocsr()
    endpoints = sparse.coo_matrix((np.ones(net.edges.size),
                                  (net.edges.T.ravel(), np.tile(np.arange(len(net.edges)), 2))),
                                  shape=(len(net.positions), len(net.edges))).tocsr()
    power = (terminal.active_w-terminal.idle_w)*(endpoints @ activity)
    return PathCatalogue(mask, paths, activity, power.tocsr(), hops)
