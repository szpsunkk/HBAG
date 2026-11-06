"""Auditable synthetic network simulation with explicit conditional safety.

Each slot is a fluid-service interval. All unserved bits are rejected in that
slot; there is no unreported queue. Geometry, traffic, and random seeds are
common across policies. Timing measures a single-process solver emulator.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy import sparse

from .energy import Battery, periodic_reserve, load_budget, step
from .network import (Shell, OpticalTerminal, RATE_BPS, snapshot, path_catalogue,
                      slot_harvest_lower)
from .traffic import load_cities, sample_pairs, map_pairs, concurrent_reference, demand_trace
from .optimization import AllocationProblem, solve_distributed, feasible_recovery


@dataclass(frozen=True)
class Experiment:
    implementation: str = 'diagonal-steps-presolve-v3'
    planes: int = 32
    per_plane: int = 32
    raan_span_deg: float = 360.0
    flows: int = 128
    paths: int = 3
    slots_per_orbit: int = 48
    orbits: int = 3
    load: float = .65
    traffic: str = 'bursty'
    seed: int = 2000
    solar_w: float = 240.0
    base_w: float = 55.0
    idle_w: float = 5.0
    active_w: float = 35.0
    extra_loss_db: float = 0.0
    payload_availability: float = 1.0
    initial_soc: float = .8
    initial_reserve_margin_j: float = -1.0
    alpha: float = .02
    soft_weight: float = .05
    epsilon: float = .05
    regularization: float = .05
    max_iterations: int = 1200
    tolerance: float = 1e-3

    def key(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Environment:
    config: Experiment
    shell: Shell
    terminal: OpticalTerminal
    battery: Battery
    times: np.ndarray
    dt_s: float
    harvest: np.ndarray
    reserve: np.ndarray
    certificate_feasible: np.ndarray
    demand: np.ndarray
    catalogues: list
    endpoint_pairs: list
    link_counts: np.ndarray
    city_pairs: np.ndarray
    city_names: list
    reference_scale: float
    build_time_s: float


def build_environment(config, cities_path, progress=None):
    start = time.perf_counter()
    c = config
    shell = Shell(c.planes,c.per_plane,raan_span_deg=c.raan_span_deg)
    terminal = OpticalTerminal(active_w=c.active_w,idle_w=c.idle_w,extra_loss_db=c.extra_loss_db,
                               payload_availability=c.payload_availability)
    battery = Battery()
    dt = shell.period_s/c.slots_per_orbit
    times = np.arange(c.slots_per_orbit*c.orbits)*dt
    nets = [snapshot(shell,t,terminal,duration_s=dt) for t in times[:c.slots_per_orbit]]
    harvest_period = np.array([slot_harvest_lower(shell,t,dt,c.solar_w) for t in times[:c.slots_per_orbit]])
    reserve_period, possible = periodic_reserve(harvest_period,c.base_w+4*c.idle_w,dt,battery)
    harvest = np.tile(harvest_period,(c.orbits,1))
    reserve = np.vstack([np.tile(reserve_period[:-1],(c.orbits,1)),reserve_period[-1:]])
    cities = load_cities(cities_path)
    pairs, shape = sample_pairs(cities,nets[0],c.flows,c.seed)
    sources,destinations = map_pairs(cities,pairs,nets[0].positions,0)
    # Calibration always uses three candidates, even when the evaluated route
    # restriction is one/five paths. Load therefore stays comparable across K.
    reference = path_catalogue(nets[0],sources,destinations,terminal,count=3)
    scale = concurrent_reference(reference,shape)
    demand = demand_trace(shape,scale,c.load,times,shell.period_s,c.seed,c.traffic,
                          [cities[i][1] for i in pairs[:,0]])
    catalogues, endpoints = [], []
    for i,t in enumerate(times):
        net = nets[i % c.slots_per_orbit]
        sources,destinations = map_pairs(cities,pairs,net.positions,t)
        catalogues.append(path_catalogue(net,sources,destinations,terminal,count=c.paths))
        endpoints.append((sources,destinations))
        if progress and i % c.slots_per_orbit == 0:
            progress('environment slot %d/%d'%(i,len(times)))
    return Environment(c,shell,terminal,battery,times,dt,harvest,reserve,possible,demand,
                       catalogues,endpoints,np.array([len(n.edges) for n in nets]),pairs,
                       [v[0] for v in cities],scale,time.perf_counter()-start)


POLICIES = {
    'unprotected': (False, False, False),
    'reserve_only': (True, False, False),
    'penalty_only': (False, True, False),
    'full': (True, True, False),
    'one_step': (True, True, True),
    'static_budget': (False, False, False),
}


def allocation_problem(env, index, energy, policy, dt_s=None, harvest=None, target=None, reserve_now=None):
    c, cat, battery = env.config, env.catalogues[index], env.battery
    hard, soft, one_step = POLICIES[policy]
    dt = env.dt_s if dt_s is None else dt_s
    h = env.harvest[index] if harvest is None else harvest
    desired = (np.full(env.shell.size,battery.minimum_j) if one_step else env.reserve[index+1]) if target is None else target
    hardware_increment = 4*(c.active_w-c.idle_w)
    power_scale = max(hardware_increment,1.)
    mandatory = c.base_w + 4*c.idle_w
    safe_budget = load_budget(energy,desired,h,dt,battery)-mandatory
    infeasible = safe_budget < -1e-7
    if hard:
        budgets = np.minimum(hardware_increment,np.maximum(safe_budget,0))
        # Conservative removal of sub-nanowatt roundoff at the reserve boundary.
        budgets[budgets < 1e-9] = 0.
    elif policy == 'static_budget':
        # A fixed per-node average-energy budget, not a SatFlow reproduction.
        mean_h = env.harvest.mean(axis=0)
        day_fraction = np.mean(env.harvest > 0,axis=0)
        # Exact net-drift-neutral constant load for binary solar/zero harvest.
        eta = battery.charge_efficiency*battery.discharge_efficiency
        neutral = eta*mean_h/(eta*day_fraction + (1-day_fraction))
        budgets = np.clip(neutral-mandatory,0,hardware_increment)
    else:
        budgets = np.full(env.shell.size,hardware_increment)
    current_reserve = env.reserve[index] if reserve_now is None else reserve_now
    margin = np.maximum(energy-current_reserve,0)/(battery.capacity_j-battery.minimum_j)
    nu = c.alpha + (c.soft_weight/(c.epsilon+margin) if soft else np.zeros_like(margin))
    cost = np.asarray(cat.node_power_w.T @ nu).reshape(cat.path_mask.shape)/power_scale
    K = sparse.vstack([cat.link_activity,cat.node_power_w/power_scale],format='csr')
    b = np.r_[np.ones(cat.link_activity.shape[0]),budgets/power_scale]
    problem = AllocationProblem(K,b,env.demand[index],cat.path_mask,cost,
                                np.ones(c.flows),np.full(c.flows,.1),c.regularization)
    return problem, infeasible, safe_budget


def run_policy(env, policy='reserve_only', output_path=None, initial_energy_j=None, improve_execution=False):
    c, battery = env.config, env.battery
    if policy not in POLICIES:
        raise ValueError(policy)
    slots, nodes = len(env.times), env.shell.size
    energy = (np.minimum(env.reserve[0]+c.initial_reserve_margin_j,battery.capacity_j)
              if c.initial_reserve_margin_j >= 0 else np.full(nodes,c.initial_soc*battery.capacity_j))
    if initial_energy_j is not None:
        energy=np.broadcast_to(np.asarray(initial_energy_j,dtype=float),(nodes,)).copy()
        if np.any((energy<0)|(energy>battery.capacity_j)):
            raise ValueError('Explicit initial energy must lie within physical capacity')
    initial_feasible = (energy >= env.reserve[0]-1e-7) & env.certificate_feasible
    states = np.empty((slots+1,nodes)); states[0] = energy
    terminal_power = np.empty((slots,nodes)); raw = np.empty((slots,nodes))
    unserved = np.empty((slots,nodes)); spill = np.empty((slots,nodes))
    allocations = np.empty((slots,c.flows,c.paths)); delivered = np.empty((slots,c.flows))
    ledger = []
    start = time.perf_counter()
    for t in range(slots):
        problem, infeasible, safe_budget = allocation_problem(env,t,energy,policy)
        result = solve_distributed(problem,max_iterations=c.max_iterations,tolerance=c.tolerance,check_every=20,
                                   improve_recovery=improve_execution)
        x = result.allocation
        # Even ablations cannot draw nonexistent physical battery energy. This
        # emergency hardware guard uses zero as its floor, NOT the safety floor.
        physical = load_budget(energy,0.,env.harvest[t],env.dt_s,battery)-(c.base_w+4*c.idle_w)
        actual_b = problem.budgets.copy()
        actual_b[-nodes:] = np.minimum(actual_b[-nodes:],np.maximum(physical,0)/max(4*(c.active_w-c.idle_w),1))
        executed_problem = replace(problem,budgets=actual_b)
        x = feasible_recovery(x,executed_problem)
        p = 4*c.idle_w + env.catalogues[t].node_power_w @ x.ravel()
        e = step(energy,env.harvest[t],c.base_w,p,env.dt_s,battery)
        # If mandatory loads alone exhaust the bus, reject communication at
        # affected nodes for the whole slot and retain unserved-energy evidence.
        deficit = e.unserved_energy_j > 1e-7
        if np.any(deficit):
            actual_b[-nodes:][deficit] = 0
            x = feasible_recovery(x,replace(problem,budgets=actual_b))
            p = 4*c.idle_w + env.catalogues[t].node_power_w @ x.ravel()
            e = step(energy,env.harvest[t],c.base_w,p,env.dt_s,battery)
        energy = e.after_j
        states[t+1],terminal_power[t],raw[t],unserved[t],spill[t] = energy,p,e.raw_after_j,e.unserved_energy_j,e.spill_j
        allocations[t],delivered[t] = x,x.sum(axis=1)
        power_scale = max(4*(c.active_w-c.idle_w),1.)
        active = (safe_budget < 4*(c.active_w-c.idle_w)-1e-7)
        source,destination = env.endpoint_pairs[t]
        ledger.append(dict(slot=t,time_s=float(env.times[t]),solver_s=result.runtime_s,
                           iterations=result.iterations,gap=result.relative_gap,converged=result.converged,
                           violation=executed_problem.violations(x),objective=problem.objective(x),
                           infeasible_nodes=int(infeasible.sum()),bound_active_nodes=int(active.sum()),
                           served_rate=float(x.sum()),offered_rate=float(env.demand[t].sum()),
                           access_unavailable_rate=float(env.demand[t][(source<0)|(destination<0)].sum()),
                           no_path_rate=float(env.demand[t][~env.catalogues[t].path_mask.any(axis=1)].sum()),
                           minimum_margin_j=float(e.raw_after_j.min()-battery.minimum_j),
                           nnz=int(problem.resource_matrix.nnz),
                           logical_scalars=int(2*problem.resource_matrix.nnz*result.iterations)))
    offer_bits = float(env.demand.sum()*RATE_BPS*env.dt_s)
    delivery_bits = float(delivered.sum()*RATE_BPS*env.dt_s)
    # Shortages in energy.step are stored-energy-equivalent joules. Convert to
    # unmet DC-bus joules, with baseline supplied before idle terminal loads.
    requested_terminal_j = float(terminal_power.sum()*env.dt_s)
    supplied_bus = np.maximum((c.base_w+terminal_power)*env.dt_s-battery.discharge_efficiency*unserved,0)
    supplied_base = np.minimum(c.base_w*env.dt_s,supplied_bus)
    terminal_j = float(np.sum(supplied_bus-supplied_base))
    system_j = float(np.sum(supplied_bus))
    unsafe = raw < battery.minimum_j-1e-6
    initially_unsafe = states[0] < battery.minimum_j-1e-6
    summary = dict(config=asdict(c),policy=policy,experiment_key=c.key(),metrics_version='actual-bus-energy-v1',
                   recovery='pathwise+residual_fill' if improve_execution else 'historical pathwise',
                   initial_state_rule='explicit input array' if initial_energy_j is not None else 'configuration SOC or reserve margin',
                   offered_bits=offer_bits,delivered_bits=delivery_bits,
                   delivery_fraction=delivery_bits/offer_bits if offer_bits else 1.,
                   fvr=1-delivery_bits/offer_bits if offer_bits else 0.,
                   ee_isl_bits_j=delivery_bits/terminal_j if terminal_j else 0.,
                   ee_system_bits_j=delivery_bits/system_j,
                   terminal_energy_j=terminal_j,system_energy_j=system_j,
                   requested_terminal_energy_j=requested_terminal_j,
                   esr=1-float(unsafe.mean()),all_safe=bool(not (unsafe.any() or initially_unsafe.any())),
                   esr_definition='safe slot-end satellite states / (slots * satellites); initial state excluded',
                   esr_denominator=int(slots*nodes),initially_unsafe_nodes=int(initially_unsafe.sum()),
                   initially_unsafe_node_ids=np.flatnonzero(initially_unsafe).tolist(),
                   min_margin_j=float(min(states[0].min(),raw.min())-battery.minimum_j),
                   unsafe_node_seconds=float(unsafe.sum()*env.dt_s),
                   initially_certified_nodes=int(initial_feasible.sum()),
                   periodic_feasible_nodes=int(env.certificate_feasible.sum()),
                   minimum_final_reserve_margin_j=float(np.min(states[-1]-env.reserve[-1])),
                   mandatory_unserved_j=float(unserved.sum()),
                   curtailed_harvest_j=float(spill.sum()),
                   mean_solver_s=float(np.mean([r['solver_s'] for r in ledger])),
                   p95_solver_s=float(np.percentile([r['solver_s'] for r in ledger],95)),
                   max_gap=float(max(r['gap'] for r in ledger)),
                   converged_fraction=float(np.mean([r['converged'] for r in ledger])),
                   max_executed_violation=float(max(r['violation'] for r in ledger)),
                   reference_scale=env.reference_scale,environment_build_s=env.build_time_s,
                   simulation_s=time.perf_counter()-start,dt_s=env.dt_s,
                   baseline_definition='Own stated optimization baseline; no external algorithm reproduction')
    if output_path:
        out = Path(output_path);out.mkdir(parents=True,exist_ok=True)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'slots.json').write_text(json.dumps(ledger,indent=2))
        np.savez_compressed(str(out/'raw.npz'),energy_j=states,raw_after_j=raw,harvest_w=env.harvest,
                            reserve_j=env.reserve,terminal_w=terminal_power,unserved_j=unserved,spill_j=spill,
                            offered=env.demand,delivered=delivered,allocation=allocations,
                            city_pairs=env.city_pairs,times_s=env.times)
    return summary,ledger
