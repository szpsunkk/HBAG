"""E14: controller values versus certified fixed-catalogue removal optima."""
from dataclasses import replace,asdict
import json
import time
import numpy as np
from .model import RATE,POWER,Config
from .environment import build_environment
from .control import SwitchController,usable_catalogue
from .allocation import allocate,protect_bus
from .energy import step
from .optimization import solve_centralized,feasible_recovery
from .link_values import counterfactual_values
from .runner import write_json
from .v24_runtime import ROOT,OUT,install


def greedy_values(cat,x,weight,term,problem,phi):
    """Same heuristic, reserving alternative directed slack per removed ISL.

    Flow/path order is deterministic (ascending indices); each candidate
    removal starts with independent residual capacity. Reserved capacity is
    shared across all displaced flows/paths for that removal only.
    """
    count=len(cat.edges)//2
    rate=np.asarray(cat.incidence@x.ravel())*RATE
    slack=np.maximum(cat.link_capacity_bps-rate,0.)/RATE
    derivative=problem.utility_weight/(problem.utility_scale+x.sum(axis=1))
    cost=weight[cat.edges[:,0]]*term.derivative(rate,cat.distance_km)*RATE/POWER
    lookup={tuple(map(int,e)):j for j,e in enumerate(cat.edges)}
    resources=[];edge_sets=[];costs=[]
    for routes in cat.paths:
        ids=[[lookup[u,v] for u,v in zip(r[:-1],r[1:])] if r else [] for r in routes]
        resources.append(ids);edge_sets.append([{j%count for j in p} for p in ids])
        costs.append([float(cost[p].sum()) if p else 0. for p in ids])
    result=np.zeros(count)
    for edge in range(count):
        remaining=slack.copy()
        for f in range(len(x)):
            for p,amount in enumerate(x[f]):
                if amount<=0 or edge not in edge_sets[f][p]:continue
                scaled=float(amount)*float(phi[f])
                choices=[]
                for j,ids in enumerate(resources[f]):
                    if ids and cat.mask[f,j] and edge not in edge_sets[f][j]:
                        capacity=float(remaining[ids].min())
                        if capacity>0:choices.append((j,capacity))
                if choices:
                    best,capacity=min(choices,key=lambda v:(-v[1],costs[f][v[0]],v[0]))
                    reroute=min(scaled,capacity)
                    remaining[resources[f][best]]-=reroute
                    extra=max(costs[f][best]-costs[f][p],0.)
                else:reroute=0.;extra=0.
                result[edge]+=reroute*extra+(scaled-reroute)*derivative[f]
    return result


def utility(p,x):
    return float(np.sum(p.utility_weight*np.log1p(x.sum(axis=1)/p.utility_scale)))


def optimum(p):
    r=solve_centralized(p,tolerance=1e-9)
    if not r['reference_target_met'] or r['degraded']:
        from .central_reference_r2 import retain_failure
        retain_failure(p,1e-9,r['numerical_attempts'])
        raise ArithmeticError('E14 Clarabel did not certify the requested 1e-9 tolerance')
    x=feasible_recovery(r['allocation'],p)
    value=p.objective(x)
    return dict(phi=value,utility=utility(p,x),gap=r['certificate_gap'],
                error=max(0.,r['dual_upper_bound']-value),iterations=r['iterations'])


def removed_problem(p,cat,edges):
    count=len(cat.edges)//2
    budget=p.budgets.copy()
    for j in edges:budget[j]=budget[j+count]=0.
    return replace(p,budgets=budget)


def run(seed):
    install()
    frozen=json.loads((ROOT/'results_r2/E11_validation/frozen.json').read_text())
    c=Config(seed=seed,planes=8,per_plane=8,flows=32,orbits=1,rho_energy=1.4,dt_override_s=60.,
             load=frozen['load'],capacity_j=frozen['capacity_by_rho']['1.4'],
             **frozen['selected_by_rho']['1.4']['HBAG']['parameters'])
    out=OUT/'E14'/('seed_%d'%seed)
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    start=time.perf_counter();env=build_environment(c);ctl=SwitchController(c,'HBAG')
    energy=np.full(env['shell'].size,c.initial_soc*c.capacity_j)
    rows=[];actual=[];joint=[];previous=None;solves=0

    def loss(snapshot,indices):
        nonlocal solves
        indices=tuple(sorted(indices))
        if indices not in snapshot['cache']:
            r=optimum(removed_problem(snapshot['problem'],snapshot['cat'],indices));solves+=1
            base=snapshot['base'];delta=base['phi']-r['phi']
            error=base['error']+r['error']+1e-12
            if delta < -error:raise ArithmeticError('Removal improved the certified optimum beyond error bounds')
            snapshot['cache'][indices]=dict(dPhi=delta,utility_loss=base['utility']-r['utility'],
                numerical_error_bound=error,removed_phi=r['phi'],removed_gap=r['gap'])
        return snapshot['cache'][indices]

    for t in range(len(env['times'])):
        sw=ctl.decide(env,t,energy)
        # A2 consumes the preceding observe() values. Match actual removals to
        # that exact value-generating slot/problem, never to a changed budget.
        if previous is not None:
            keys=[tuple(map(int,e)) for e in env['catalogues'][t].edges[:len(sw.on)]]
            lookup=previous['lookup']
            slept=[lookup[keys[j]] for j in np.flatnonzero(sw.a2_edges) if keys[j] in lookup]
            individual=[]
            for j in slept:
                item=dict(seed=seed,value_slot=t-1,decision_slot=t,edge=list(previous['keys'][j]),
                          predicted=float(previous['cf'][j]),greedy=float(previous['greedy'][j]),
                          **loss(previous,[j]))
                actual.append(item);individual.append(item['dPhi'])
            if len(slept)>=2:
                joint.append(dict(seed=seed,value_slot=t-1,decision_slot=t,count=len(slept),
                    sum_individual_loss=float(sum(individual)),predicted_sum=float(previous['cf'][slept].sum()),
                    greedy_sum=float(previous['greedy'][slept].sum()),**loss(previous,slept)))
        cat=usable_catalogue(env['catalogues'][t],sw,c.routing)
        x,p,meta,info=allocate(env,t,cat,sw,ctl)
        x,q,_=protect_bus(env,t,energy,cat,sw,x,p,meta)
        ctl.observe(cat,x,p,sw,env['terminal'])
        count=len(sw.on);keys=[tuple(map(int,e)) for e in cat.edges[:count]]
        cf,_=counterfactual_values(cat,x,sw.weight,env['terminal'],p,ctl.peak_factor)
        # This assertion binds the diagnostic to the actual controller values.
        np.testing.assert_array_equal(cf,[ctl.counterfactual[k] for k in keys])
        greedy=greedy_values(cat,x,sw.weight,env['terminal'],p,ctl.peak_factor)
        peak_problem=replace(p,demand=p.demand*ctl.peak_factor)
        base=optimum(peak_problem);solves+=1
        snapshot=dict(cat=cat,problem=peak_problem,base=base,cf=cf,greedy=greedy,
                      keys=keys,lookup={k:j for j,k in enumerate(keys)},cache={})
        rates=np.asarray(cat.incidence@x.ravel())
        if np.any(env['harvest'][t]==0):
            for j in np.flatnonzero(sw.on & ((rates[:count]+rates[count:])>0)):
                rows.append(dict(seed=seed,slot=t,edge=list(keys[j]),
                    endpoint_eclipsed=bool(np.any(env['harvest'][t,list(keys[j])]==0)),
                    predicted=float(cf[j]),greedy=float(greedy[j]),base_phi=base['phi'],base_gap=base['gap'],
                    **loss(snapshot,[int(j)])))
        previous=snapshot
        energy=step(energy,env['harvest'][t],c.base_w,q,env['dt'],env['battery']).after_j
        ctl.price.commit(q,t)
        if t%4==0:write_json(out/'status.json',dict(slot=t+1,slots=len(env['times']),reference_solves=solves,elapsed_s=time.perf_counter()-start))
    write_json(out/'links.json',rows);write_json(out/'actual_a2.json',actual);write_json(out/'joint_a2.json',joint)
    result=dict(seed=seed,config=asdict(c),slots=len(env['times']),link_samples=len(rows),actual_a2_samples=len(actual),
                joint_samples=len(joint),reference_solves=solves,runtime_s=time.perf_counter()-start)
    write_json(out/'summary.json',result)
    return result
