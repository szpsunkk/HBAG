"""v2.1 link values from executed flow, alternative slack and current unmet demand."""
import numpy as np
from .model import RATE, POWER


def counterfactual_values(cat,x,weight,term,problem,phi=None,capture=None):
    """Exactly the registered per-used-path heuristic; no shared-slack optimization."""
    count=len(cat.edges)//2
    directed_rate=np.asarray(cat.incidence@x.ravel())*RATE
    slack=np.maximum(cat.link_capacity_bps-directed_rate,0.)/RATE
    derivative=problem.utility_weight/(problem.utility_scale+x.sum(axis=1))
    slope=term.slot_derivative(directed_rate,cat.distance_km,capture)*RATE/POWER
    directed_cost=weight[cat.edges[:,0]]*slope
    lookup={tuple(map(int,e)):j for j,e in enumerate(cat.edges)}
    values=np.zeros(count);support=np.zeros(count)
    for f,routes in enumerate(cat.paths):
        resource=[];edge_sets=[];path_slack=[];path_cost=[]
        for j,route in enumerate(routes):
            ids=[lookup[int(u),int(v)] for u,v in zip(route[:-1],route[1:])] if route else []
            resource.append(ids);edge_sets.append({i%count for i in ids})
            available=cat.mask[f,j] and ids and all(cat.link_capacity_bps[i]>0 for i in ids)
            path_slack.append(float(np.min(slack[ids])) if available else 0.)
            path_cost.append(float(np.sum(directed_cost[ids])) if ids else 0.)
        for p,amount in enumerate(x[f]):
            if amount<=0:continue
            scaled=float(amount)*(1. if phi is None else float(phi[f]))
            for edge in edge_sets[p]:
                support[edge]+=derivative[f]*amount
                alternatives=[j for j in range(len(routes)) if edge not in edge_sets[j] and cat.mask[f,j] and resource[j] and path_slack[j]>0]
                if alternatives:
                    best=min(alternatives,key=lambda j:(-path_slack[j],path_cost[j],j))
                    reroute=min(scaled,path_slack[best])
                    extra=max(path_cost[best]-path_cost[p],0.)
                else:reroute=0.;extra=0.
                values[edge]+=reroute*extra+(scaled-reroute)*derivative[f]
    return values,support


def wake_values(cat,on,capacity_factor,demand_solver,previous_delivered,utility_s):
    """One-edge counterfactual wake: all OTHER edges must already be ON.

    Directed resources are separate, as in the rate model. A commodity may
    claim a direction only once even when several of its candidates use it.
    """
    count=len(cat.edges)//2;lookup={tuple(map(int,e)):j for j,e in enumerate(cat.edges)}
    z=np.asarray(previous_delivered)
    unmet=np.maximum(demand_solver-z,0.)
    derivative=1./(1.+z/utility_s)
    claimants={}
    for f,routes in enumerate(cat.paths):
        if unmet[f]<=0:continue
        for route in routes:
            if not route:continue
            ids=[lookup[int(u),int(v)] for u,v in zip(route[:-1],route[1:])]
            off={i%count for i in ids if not on[i%count]}
            if len(off)!=1:continue
            edge=next(iter(off))
            for directed in (i for i in ids if i%count==edge):
                claimants.setdefault(directed,set()).add(f)
    result=np.zeros(count)
    for directed,flows in claimants.items():
        remaining=capacity_factor*cat.link_capacity_bps[directed]/RATE
        for f in sorted(flows,key=lambda f:(-derivative[f],f)):
            amount=min(float(unmet[f]),remaining)
            result[directed%count]+=derivative[f]*amount
            remaining-=amount
            if remaining<=0:break
    return result


def path_wake_values(cat,on,capacity_factor,demand_solver,previous_delivered,utility_s,wakeable):
    """One best wakeable path per commodity; share its value over sleeping edges.

    Capture capacity is reserved in the path's direction and cannot be reused
    by later commodities. On-edge capacity does not enter this upper-layer
    heuristic; the subsequent inner problem enforces all physical capacities.
    """
    count=len(cat.edges)//2;lookup={tuple(map(int,e)):j for j,e in enumerate(cat.edges)}
    z=np.asarray(previous_delivered);unmet=np.maximum(demand_solver-z,0.)
    derivative=1./(1.+z/utility_s)
    remaining=np.maximum(capacity_factor*np.asarray(cat.link_capacity_bps)/RATE,0.)
    result=np.zeros(count)
    for f in sorted(np.flatnonzero(unmet>0),key=lambda f:(-derivative[f],f)):
        best=None
        for route in cat.paths[f]:
            if not route:continue
            ids=[lookup[int(u),int(v)] for u,v in zip(route[:-1],route[1:])]
            off={i%count for i in ids if not on[i%count]}
            if not off or not all(wakeable[e] for e in off):continue
            offids=[i for i in ids if i%count in off]
            capacity=float(np.min(remaining[offids]))
            if capacity<=0:continue
            key=(len(off),-capacity)
            if best is None or key<best[0]:best=(key,off,offids,capacity)
        if best is None:continue
        _,off,offids,capacity=best;amount=min(float(unmet[f]),capacity)
        remaining[offids]-=amount
        for edge in off:result[edge]+=derivative[f]*amount/len(off)
    return result


def recent_peak(demand,previous,dt_s,period_s):
    demand=np.asarray(demand,float)
    peak=demand.copy() if previous is None else np.maximum(demand,np.exp(-dt_s/period_s)*previous)
    phi=np.divide(peak,demand,out=np.ones_like(demand),where=demand>0)
    return peak,phi
