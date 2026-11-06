"""Event-accurate delayed execution with <=1 s battery integration."""
from pathlib import Path
from dataclasses import asdict,replace
import heapq,time,json
import numpy as np
from .model import RATE,POWER
from .environment import build_environment,sun_vector
from .runner import write_json,source_hashes
from .solvers import solve,node_power
from .energy import step,load_budget,energy_drift
from .network import slot_harvest_lower
from .optimization import AllocationProblem,feasible_recovery


def run_delayed(c,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    start=time.perf_counter();env=build_environment(c);n=len(env['times']);m=env['shell'].size;b=env['battery'];term=env['terminal'];dt=env['dt'];end=n*dt
    energy=np.full(m,c.initial_soc*b.capacity_j);queue=[];active=None;t=0.;next_slot=0;records=[];observations={};bounds=[];last_derivative={};total=0.;offered=0.;energy_used=0.;unsafe_seconds=np.zeros(m);minimum=energy.copy();max_depth=0.
    while t<end-1e-8:
        if next_slot<n and t>=env['times'][next_slot]-1e-8:
            idx=next_slot;x,p,meta,info=solve(env,idx,energy,'HBAG')
            heapq.heappush(queue,(t+c.delay_s,idx,x));observations[idx]=(t,energy.copy());last_derivative[idx]=[np.zeros(m),np.zeros(m)]
            next_slot+=1
        while queue and queue[0][0]<=t+1e-8:
            ready,idx,x=heapq.heappop(queue);active=(idx,x)
            begun,old=observations[idx];drift,net=last_derivative.pop(idx)
            actual=np.abs(energy-old)
            bounds.append(dict(slot=idx,delay_s=c.delay_s,measured_max_j=float(actual.max()),efficiency_correct_bound_j=float((c.delay_s*drift).max()),raw_net_bound_j=float((c.delay_s*net).max()),max_efficiency_bound_violation_j=float(np.max(actual-c.delay_s*drift,initial=0)),max_raw_net_bound_violation_j=float(np.max(actual-c.delay_s*net,initial=0))))
        idx=min(int((t+1e-7)/dt),n-1);cat=env['catalogues'][idx];x=np.zeros(cat.mask.shape)
        if active:
            old_idx,old_x=active;oldcat=env['catalogues'][old_idx]
            for f in range(c.flows):
                routes={route:j for j,route in enumerate(cat.paths[f]) if route is not None}
                for j,route in enumerate(oldcat.paths[f]):
                    if route in routes:x[f,routes[route]]+=old_x[f,j]
        event=min(t+c.substep_s,end,(env['times'][next_slot] if next_slot<n else end),(queue[0][0] if queue else end))
        delta=event-t
        if delta<=1e-9:raise RuntimeError('Nonadvancing delayed event clock')
        h=slot_harvest_lower(env['shell'],t,delta,c.solar_w,sun_vector(c.beta_deg))
        link=cat.incidence.multiply((RATE/cat.link_capacity_bps)[:,None])
        inc=cat.sender@__import__('scipy').sparse.diags(term.derivative(np.zeros(len(cat.edges)),cat.distance_km)*RATE)@cat.incidence
        physical=np.maximum(load_budget(energy,0.,h,delta,b)-c.base_w-4*term.idle_w,0.)
        # Only empty-bus and current link/demand guards. Reserve is intentionally NOT refreshed at execution.
        p=AllocationProblem(__import__('scipy').sparse.vstack([link,inc/POWER]),np.r_[np.ones(len(cat.edges)),physical/POWER],env['demand_bps'][idx]/RATE,cat.mask,np.zeros(cat.mask.shape),np.ones(c.flows),np.ones(c.flows),c.rho)
        x=feasible_recovery(x,p);q=node_power(cat,term,x,np.full(m,4*term.idle_w));record=step(energy,h,c.base_w,q,delta,b)
        drift=np.abs(energy_drift(h,c.base_w+q,b));net=np.abs(h-c.base_w-q)
        for k in last_derivative:
            last_derivative[k][0]=np.maximum(last_derivative[k][0],drift);last_derivative[k][1]=np.maximum(last_derivative[k][1],net)
        total+=float(x.sum()*RATE*delta);offered+=float(env['demand_bps'][idx].sum()*delta);energy_used+=float(q.sum()*delta)
        energy=record.after_j;minimum=np.minimum(minimum,energy);unsafe_seconds+=(energy<b.minimum_j-1e-6)*delta;max_depth=max(max_depth,float(np.maximum(b.minimum_j-energy,0).max()))
        records.append((event,energy.copy(),q.copy(),h.copy()));t=event
    np.savez_compressed(out/'raw.npz',times_s=np.array([r[0] for r in records]),energy_j=np.array([r[1] for r in records]),terminal_w=np.array([r[2] for r in records]),harvest_w=np.array([r[3] for r in records]))
    summary=dict(config=asdict(c),seed=c.seed,method='HBAG',delivered_bits=total,offered_bits=offered,fvr=1-total/offered,esr=1-float(unsafe_seconds.sum()/(end*m)),min_soc=float(minimum.min()/b.capacity_j),violation_depth_j=max_depth,unsafe_node_seconds=float(unsafe_seconds.sum()),terminal_energy_j=energy_used,dt_s=dt,delay_s=c.delay_s,runtime_s=time.perf_counter()-start,bound_max_excess_j=max(v['max_efficiency_bound_violation_j'] for v in bounds),raw_net_bound_max_excess_j=max(v['max_raw_net_bound_violation_j'] for v in bounds),note='Old paths are dropped if not present in the current topology/OD catalogue. Integration splits at fractional completion events. Efficiency-correct drift bound reported alongside requested raw net-power bound.')
    write_json(out/'summary.json',summary);write_json(out/'delay_bounds.json',bounds);write_json(out/'manifest.json',dict(config=asdict(c),source_sha256=source_hashes(),environment_cache=env['cache_key']))
    return summary
