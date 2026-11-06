"""R2 terminal state, handshake, capture and identical route fallback for all policies."""
from dataclasses import dataclass, replace
import networkx as nx
import numpy as np
from scipy import sparse
from .environment import Catalogue
from .model import POWER
from .energy import load_budget
from .pricing import OrbitalPrice
from .link_values import counterfactual_values,path_wake_values,recent_peak


def terminal_ports(edges, per_plane, planes):
    """Ports 0/1 = along-track +/-; ports 2/3 = adjacent-plane +/-."""
    ports = np.empty_like(edges)
    for i, (a, b) in enumerate(edges):
        pa, ia = divmod(int(a), per_plane)
        pb, ib = divmod(int(b), per_plane)
        if pa == pb:
            ports[i] = (0, 1) if (ib-ia) % per_plane == 1 else (1, 0)
        else:
            ports[i] = (2, 3) if (pb-pa) % planes == 1 else (3, 2)
    pairs = [(int(n), int(p)) for ns, ps in zip(edges, ports) for n, p in zip(ns, ps)]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Two geometric neighbours assigned to one physical terminal')
    return ports


@dataclass
class SwitchResult:
    on: np.ndarray
    ports: np.ndarray
    terminals_on: np.ndarray
    available_terminals: np.ndarray
    capture: np.ndarray
    newly_usable: np.ndarray
    fixed_w: np.ndarray
    raw_budget_w: np.ndarray
    weight: np.ndarray
    xi: np.ndarray
    infeasible: np.ndarray
    brownout: np.ndarray
    budget_forced: np.ndarray
    price_sleep: int
    sleep_events: int
    wake_events: int
    sleep_per_node: np.ndarray
    wake_per_node: np.ndarray
    a0_sleep: int
    rescue_wakes: int
    price_trace: dict
    counterfactual: np.ndarray
    support_value: np.ndarray
    wake_value: np.ndarray
    a2_nodes: np.ndarray
    zero_price_a2: int
    acquisition_active_s: np.ndarray = None
    a2_edges: np.ndarray = None


class SwitchController:
    def __init__(self, config, method):
        self.c, self.method = config, method
        self.price = OrbitalPrice(config,method)
        self.hard = self.price.hard
        self.soft = self.price.kind != 'zero'
        m = config.planes*config.per_plane
        self.states = np.zeros((m, 4), bool)
        self.ages = np.zeros((m, 4), np.int32)
        self.guard_asleep = np.zeros(m, bool)
        self.previous_edges = set()
        self.support = {}
        self.counterfactual = {}
        self.last_awake_support = {}
        self.utilization = {}
        self.low_counts = {}
        self.started = False
        self.delivered = np.zeros(config.flows)
        self.peak=None
        self.peak_factor=np.ones(config.flows)
        self.acquisition_remaining={}

    def decide(self, env, t, energy, budget_energy=None):
        c = self.c; b = env['battery']; cat = env['catalogues'][t]
        count = len(cat.edges)//2
        edges = cat.edges[:count]
        ports = terminal_ports(edges, c.per_plane, c.planes)
        a, z = edges.T; ap, zp = ports.T
        m = len(energy); dt = env['dt']
        from .network import Shell
        period=getattr(env['shell'],'period_s',Shell(c.planes,c.per_plane).period_s)
        self.peak,self.peak_factor=recent_peak(env['demand_bps'][t]/1e9,self.peak,dt,period)
        keys = [tuple(map(int, e)) for e in edges]
        available = np.zeros((m, 4), bool)
        available[a, ap] = True; available[z, zp] = True
        old = self.states.copy()
        # The common initial condition powers all geometrically available ports.
        on = np.ones(count, bool) if not self.started else old[a, ap] & old[z, zp]
        brownout = energy <= c.cut_soc*b.capacity_j
        raw_budget = load_budget(energy if budget_energy is None else budget_energy,
                                 env['reserve'][t+1], env['harvest'][t], dt, b)-c.base_w
        infeasible = (raw_budget < 4*c.sleep_w-1e-8) | (env['reserve'][t+1] > b.capacity_j+1e-6)
        weight, xi, price_trace = self.price.prepare(env,t,energy)
        support = np.array([self.support.get(k, 0.) for k in keys])
        cf = np.array([self.counterfactual.get(k,0.) for k in keys])
        age = np.minimum(self.ages[a, ap], self.ages[z, zp])
        was_on = on.copy()
        blocked = brownout[a] | brownout[z]
        on[blocked] = False
        a0_sleep = int(np.sum(old & (~available | brownout[:, None])))
        forced = np.zeros(m, bool)
        slept_now = blocked.copy()
        price_sleep = 0; rescue_wakes = 0
        a2_nodes=np.zeros(m,np.int32);zero_price_a2=0
        a2_edges=np.zeros(count,bool)
        wake=np.zeros(count)

        def fixed():
            return 4*c.sleep_w+(25.-c.sleep_w)*np.bincount(edges[on].ravel(), minlength=m)

        def switch_off(j):
            on[j] = False; slept_now[j] = True

        def may_wake(j, extra=0.):
            if blocked[j] or slept_now[j] or age[j] < c.minimum_hold_slots:
                return False
            if self.hard:
                if infeasible[a[j]] or infeasible[z[j]]:
                    return False
                slack = raw_budget-fixed()
                if min(slack[a[j]], slack[z[j]]) < 25.-c.sleep_w+extra-1e-8:
                    return False
            return True

        if self.method == 'V0':
            on = ~blocked
        elif self.method.startswith('Guard') or self.method == 'Static+Guard':
            soc = energy/b.capacity_j
            self.guard_asleep[soc < c.guard_soc] = True
            self.guard_asleep[soc >= c.guard_soc+.05] = False
            on = ~(blocked | self.guard_asleep[a] | self.guard_asleep[z])
        else:
            # A1: a hard violation can override the two-slot hold.
            if self.hard:
                forced = infeasible | (fixed() > raw_budget+1e-8)
                for j in np.flatnonzero(on & (infeasible[a] | infeasible[z])):
                    switch_off(j)
                q = fixed()
                for j in np.argsort(cf, kind='stable'):
                    if on[j] and (q[a[j]] > raw_budget[a[j]]+1e-8 or q[z[j]] > raw_budget[z[j]]+1e-8):
                        switch_off(j)
                        q[a[j]] -= 25.-c.sleep_w; q[z[j]] -= 25.-c.sleep_w
            # A2 / utilization sleep: count only actual preceding-slot observations.
            low = np.zeros(count, bool)
            pi=price_trace['pi']
            cost=(pi[a]+pi[z])*(25.-c.sleep_w)/POWER
            if self.soft:
                low = cf < c.kappa_off*cost
            elif self.method == 'Util-Sleep':
                low = np.array([self.utilization.get(k, 0.) < c.utilization_sleep_threshold for k in keys])
            if self.soft or self.method == 'Util-Sleep':
                eligible=np.zeros(count,bool)
                for j, k in enumerate(keys):
                    value = self.low_counts.get(k, 0)+1 if on[j] and k in self.previous_edges and low[j] else 0
                    self.low_counts[k] = value
                    eligible[j]=value>=2 and age[j]>=c.minimum_hold_slots
                ratios=np.divide(cf,cost,out=np.full(count,np.inf),where=cost>0)
                for j in np.argsort(ratios,kind='stable'):
                    if not eligible[j]:continue
                    if self.soft and (a2_nodes[a[j]] or a2_nodes[z[j]]):continue
                    switch_off(j)
                    if self.soft:
                        a2_edges[j]=True
                        price_sleep+=1;a2_nodes[a[j]]+=1;a2_nodes[z[j]]+=1
                        zero_price_a2+=int(pi[a[j]]==0 and pi[z[j]]==0)
                assert np.max(a2_nodes,initial=0)<=1 and zero_price_a2==0
            # A3: deterministic simultaneous-intent handshake, with sequential budget reservation.
            wakeable=(~on)&(~blocked)&(~slept_now)&(age>=c.minimum_hold_slots)
            wake=path_wake_values(cat,on,max(1-c.acquisition_s/dt,0),env['demand_bps'][t]/1e9,self.delivered,c.utility_s_bps/1e9,wakeable)
            for j in np.flatnonzero(~on):
                if not may_wake(j, c.wake_slack_w if self.hard else 0.):
                    continue
                if self.soft:
                    last_value = self.last_awake_support.get(keys[j],0.)
                    price_wake = cost[j] <= c.kappa_off/2*max(last_value,wake[j]) or (pi[a[j]]==0 and pi[z[j]]==0)
                    if not price_wake:continue
                if self.method == 'Util-Sleep' and age[j] < 5:
                    continue
                on[j] = True
            # A4: first check all K candidates and connectivity of the usable graph.
            # The K+1 route means an existing usable detour does not trigger needless wakeups.
            index = {k: j for j, k in enumerate(keys)}
            graph = nx.Graph(); graph.add_nodes_from(range(m))
            graph.add_edges_from(edges[on].tolist())
            for f, routes in enumerate(cat.paths):
                if env['demand_bps'][t, f] <= 0 or not any(routes):
                    continue
                s, d = int(cat.sources[f]), int(cat.destinations[f])
                if s < 0 or d < 0 or s == d or nx.has_path(graph, s, d):
                    continue
                route = next((p for p in routes if p), None)
                indices = [index[tuple(sorted((u, v)))] for u, v in zip(route[:-1], route[1:])]
                need = [j for j in indices if not on[j]]
                if not all(may_wake(j) for j in need):
                    continue
                if self.hard:
                    added = (25.-c.sleep_w)*np.bincount(edges[need].ravel(), minlength=m)
                    affected = added > 0
                    if np.any(fixed()[affected]+added[affected] > raw_budget[affected]+1e-8):
                        continue
                on[need] = True; rescue_wakes += len(need)
                graph.add_edges_from(edges[need].tolist())

        state = np.zeros_like(old)
        state[a[on], ap[on]] = True; state[z[on], zp[on]] = True
        assert np.array_equal(state[a, ap] & state[z, zp], on)
        assert np.array_equal(state[a, ap], state[z, zp])
        assert not np.any(state[~available]) and not np.any(state[brownout])
        if self.method == 'V0':
            assert np.array_equal(on, ~blocked), 'V0 non-A0 sleep'
        if self.hard:
            assert np.all(fixed()[~infeasible] <= raw_budget[~infeasible]+1e-6)
            assert not np.any(state[infeasible])
        new = np.array([bool(on[j] and k not in self.previous_edges) for j, k in enumerate(keys)])
        capture = np.where(on, np.where(new, max(1-c.acquisition_s/dt, 0.), 1.), 0.)
        acquisition_active=np.where(new,min(dt,c.acquisition_s),0.)
        if c.acquisition_carry:
            remaining=np.array([c.acquisition_s if new[j] else self.acquisition_remaining.get(k,0.)
                                for j,k in enumerate(keys)])
            capture=np.where(on,np.maximum(1-remaining/dt,0.),0.)
            acquisition_active=np.where(on,np.minimum(remaining,dt),0.)
            self.acquisition_remaining={k:max(float(remaining[j])-dt,0.)
                                        for j,k in enumerate(keys) if on[j]}
        sleeping = old & ~state
        waking = ~old & state
        self.ages = np.where(state == old, self.ages+1, 1)
        self.states = state; self.started = True
        self.previous_edges = {k for j, k in enumerate(keys) if on[j]}
        return SwitchResult(on, ports, state, available, capture, new, fixed(), raw_budget,
                            weight, xi, infeasible, brownout, forced, price_sleep,
                            int(sleeping.sum()), int(waking.sum()), sleeping.sum(axis=1),
                            waking.sum(axis=1), a0_sleep, rescue_wakes, price_trace,cf,support,wake,a2_nodes,zero_price_a2,
                            acquisition_active,a2_edges)

    def observe(self, cat, allocation, problem, switch, term=None):
        from .model import RATE
        count = len(cat.edges)//2
        derivative = problem.utility_weight/(problem.utility_scale+allocation.sum(axis=1))
        directed = cat.incidence @ (allocation*derivative[:, None]).ravel()
        rate = cat.incidence @ allocation.ravel()*RATE
        util = np.divide(rate, cat.link_capacity_bps, out=np.zeros_like(rate), where=cat.link_capacity_bps > 0)
        if term is None:
            from .model import Terminal
            term=Terminal(medium=self.c.medium,interference_db=self.c.interference_db)
        # Every policy using A1/A2 counterfactual values gets the same robust
        # rule. V0/Guard/Util do not use these values; keep their old diagnostic
        # definition so the unaffected V0 trajectory remains byte-identical.
        phi=self.peak_factor if self.hard or self.soft else None
        capture=np.tile(switch.capture,2) if self.c.rf_capture_power and term.medium=='rf' else None
        cf,sup=counterfactual_values(cat,allocation,switch.weight,term,problem,phi,capture=capture)
        self.counterfactual={tuple(map(int,e)):float(cf[j]) for j,e in enumerate(cat.edges[:count])}
        self.support = {tuple(map(int,e)):float(sup[j]) for j,e in enumerate(cat.edges[:count])}
        for j,e in enumerate(cat.edges[:count]):
            if switch.on[j]:self.last_awake_support[tuple(map(int,e))]=float(cf[j])
        self.delivered=allocation.sum(axis=1).copy()
        self.utilization = {tuple(map(int, e)): float(max(util[j], util[j+count])) for j, e in enumerate(cat.edges[:count])}


def usable_catalogue(geometric, switch, routing='distance'):
    """Mask geometric Yen paths; one Dijkstra route only when all K are blocked."""
    cat = geometric; count = len(cat.edges)//2; f, k = cat.mask.shape
    usable = np.tile(switch.on, 2)
    lookup = {tuple(map(int, e)): j for j, e in enumerate(cat.edges)}
    graph = None; paths = []; rows = []; cols = []
    mask = np.zeros((f, k+1), bool); hops = np.zeros((f, k+1), np.int32)
    for i, candidates in enumerate(cat.paths):
        routes = list(candidates)+[None]
        for j, route in enumerate(candidates):
            if route:
                ids = [lookup[u, v] for u, v in zip(route[:-1], route[1:])]
                mask[i, j] = bool(np.all(usable[ids]))
        s, d = int(cat.sources[i]), int(cat.destinations[i])
        if not mask[i].any() and s >= 0 and d >= 0 and s != d:
            if graph is None:
                graph = nx.Graph(); graph.add_nodes_from(range(cat.sender.shape[0]))
                for edge, dist, active in zip(cat.edges[:count], cat.distance_km[:count], switch.on):
                    if active:
                        graph.add_edge(int(edge[0]), int(edge[1]), weight=1. if routing == 'hops' else float(dist))
            try:
                routes[k] = tuple(nx.shortest_path(graph, s, d, weight='weight', method='dijkstra'))
                mask[i, k] = True
            except nx.NetworkXNoPath:
                pass
        for j, route in enumerate(routes):
            if route:
                hops[i, j] = len(route)-1
                for u, v in zip(route[:-1], route[1:]):
                    rows.append(lookup[u, v]); cols.append(i*(k+1)+j)
        paths.append(routes)
    incidence = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cat.edges), mask.size)).tocsr()
    return Catalogue(mask, paths, incidence, cat.sender, cat.link_capacity_bps*np.tile(switch.capture, 2),
                     cat.distance_km, cat.edges, cat.sources, cat.destinations, hops)
