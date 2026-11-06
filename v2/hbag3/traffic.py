"""Public-city, synthetic OD traffic and a single reference load calibration."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .network import EARTH_KM, attach_ground, ground_positions


def load_cities(path):
    data = json.loads(Path(path).read_text())
    records = []
    for feature in data['features']:
        p = feature['properties']
        lon, lat = feature['geometry']['coordinates'][:2]
        if p['POP_MAX'] > 0:
            records.append((p['NAME'], lon, lat, p['POP_MAX']))
    return records


def sample_pairs(cities, net, flows, seed, minimum_separation_km=2500):
    """Population-weighted sampling among cities accessible at reference time.

    POP_MAX is a coarse Natural Earth attribute, not a 2026 population census.
    Subsequent access failures remain counted in the offered traffic denominator.
    """
    rng = np.random.default_rng(seed)
    lon = np.array([v[1] for v in cities]); lat = np.array([v[2] for v in cities])
    population = np.array([v[3] for v in cities], dtype=float)
    xyz = ground_positions(lon, lat, 0.)
    access = attach_ground(xyz, net.positions)
    weight = population*(access >= 0)
    weight /= weight.sum()
    pairs = []
    attempts = 0
    while len(pairs) < flows:
        a, b = rng.choice(len(cities), 2, replace=False, p=weight)
        attempts += 1
        if np.linalg.norm(xyz[a]-xyz[b]) >= minimum_separation_km:
            pairs.append((int(a), int(b)))
        if attempts > flows*10000:
            raise ValueError("Cannot sample the requested separated, accessible OD pairs")
    shape = rng.lognormal(0, .4, flows)
    shape /= shape.mean()
    return np.asarray(pairs), shape


def map_pairs(cities, pairs, satellite_xyz, time_s):
    chosen = np.unique(pairs)
    xyz = ground_positions(np.array([cities[i][1] for i in chosen]),
                           np.array([cities[i][2] for i in chosen]), time_s)
    mapping = dict(zip(chosen, attach_ground(xyz, satellite_xyz)))
    return np.array([mapping[a] for a in pairs[:,0]]), np.array([mapping[b] for b in pairs[:,1]])


def concurrent_reference(catalogue, shape):
    """Max common scale of the fixed OD shape under reference link capacities.

    Every policy uses this same once-computed scale. No per-slot renormalization
    and no sum-of-link-capacity denominator. Reference has no battery constraint.
    """
    if not np.all(catalogue.path_mask.any(axis=1)):
        raise ValueError("Reference contains unreachable commodities")
    flows, paths = catalogue.path_mask.shape
    aggregate = sparse.kron(sparse.eye(flows), np.ones((1,paths)), format='csr')
    link = sparse.hstack([catalogue.link_activity, sparse.csr_matrix((catalogue.link_activity.shape[0],1))])
    delivery = sparse.hstack([-aggregate, sparse.csr_matrix(np.asarray(shape)[:,None])])
    constraints = sparse.vstack([link, delivery]).tocsr()
    bounds = [(0,None) if v else (0,0) for v in catalogue.path_mask.ravel()]+[(0,None)]
    objective = np.zeros(flows*paths+1); objective[-1] = -1
    result = linprog(objective, A_ub=constraints,
                     b_ub=np.r_[np.ones(link.shape[0]), np.zeros(flows)], bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError('Reference concurrent flow failed: '+result.message)
    return float(result.x[-1])


def demand_trace(shape, scale, load, times_s, period_s, seed, kind, source_longitudes):
    """Equal per-OD horizon means; temporal total load is explicitly NOT fixed."""
    rng = np.random.default_rng(seed+7919)
    times = np.asarray(times_s)
    phase = rng.uniform(0, 2*np.pi, len(shape))
    smooth = 1 + .25*np.sin(2*np.pi*times[:,None]/period_s + phase)
    smooth *= rng.uniform(.9, 1.1, (len(times), len(shape)))
    if kind == 'smooth':
        modulation = smooth
    elif kind == 'bursty':
        # Shared bursts survive normalization; select a reproducible 20% of slots.
        spikes = np.zeros(len(times), dtype=bool)
        spikes[rng.choice(len(times), max(1,len(times)//5), replace=False)] = True
        modulation = smooth*np.where(spikes,3.4,.4)[:,None]
    elif kind == 'hotspot':
        center = (times/period_s*360) % 360 - 180
        delta = (np.asarray(source_longitudes)[None,:]-center[:,None]+180)%360-180
        modulation = smooth*(.35+6*np.exp(-.5*(delta/25)**2))
    else:
        raise ValueError('Unknown traffic trace '+kind)
    modulation /= modulation.mean(axis=0, keepdims=True)
    return load*scale*np.asarray(shape)[None,:]*modulation
