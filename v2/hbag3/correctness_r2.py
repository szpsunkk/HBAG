"""E1-prime: actual 8x8 controller and allocator under randomized bounded physics."""
from dataclasses import replace,asdict
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from .model import Config,Terminal,RATE
from .network import Shell,snapshot
from .environment import ROOT,catalogue
from .energy import reserve_envelope,step
from .runner import seed_guard,save_npz,write_json,source_hashes,fingerprint
from .control import SwitchController,usable_catalogue
from .allocation import allocate,protect_bus


def trial_batch(args):
    seed,capacity,parameters=args
    c=Config(seed=seed,planes=8,per_plane=8,flows=32,orbits=1,capacity_j=capacity,**parameters)
    seed_guard(c,'test');rng=np.random.default_rng(seed);shell=Shell(8,8);term=Terminal()
    base=ROOT/'results_r2/E1'/('seed_%d'%seed)
    if (base/'summary.json').exists():
        import json
        prior=json.loads((base/'summary.json').read_text())
        if prior.get('source_fingerprint')!=fingerprint():raise RuntimeError('E1 source mismatch')
        return prior
    net=snapshot(shell,0,term);sources=np.arange(32);dest=(sources+2)%64
    cat=catalogue(net,sources,dest,4,'hybrid',node_dark=np.zeros(64,bool))
    rows=[];start=time.perf_counter();initials=[];lowers=[];harvests=[];bases=[];demands=[];trajectories=[];powers=[];switch_states=[]
    for trial in range(500):
        lower=rng.choice([0.,250.,500.],size=(8,64),p=[.5,.15,.35])
        reserve=reserve_envelope(lower,63.,60.,c.battery)
        initial=reserve[0]+rng.uniform(0,1,64)*(capacity-reserve[0])
        demand=rng.uniform(0,5e9,(4,32))
        actual_h=lower[:4]+rng.uniform(0,1,(4,64))*(500-lower[:4])
        actual_base=rng.uniform(30,55,(4,64))
        initials.append(initial.copy());lowers.append(lower);harvests.append(actual_h);bases.append(actual_base);demands.append(demand)
        env=dict(config=c,shell=shell,terminal=term,battery=c.battery,dt=60.,
                 harvest=lower[:4],harvest_extended=lower,reserve=reserve[:5],catalogues=[cat]*4,demand_bps=demand)
        for method in ['V1','HBAG']:
            ctl=SwitchController(c,method);e=initial.copy();minimum=1e100;constraint=0.;shadow_gap=0.
            states=[];qs=[];switches=[]
            for t in range(4):
                sw=ctl.decide(env,t,e);active=usable_catalogue(cat,sw)
                x,p,meta,info=allocate(env,t,active,sw,ctl)
                shadow_gap=max(shadow_gap,info.get('shadow_relative_gap',0.))
                x,q,v=protect_bus(env,t,e,active,sw,x,p,meta)
                e=step(e,actual_h[t],actual_base[t],q,60,c.battery).after_j
                states.append(e.copy());qs.append(q.copy());switches.append(sw.terminals_on.copy())
                ctl.observe(active,x,p,sw,term);ctl.price.commit(q,t)
                minimum=min(minimum,float((e-reserve[t+1]).min()));constraint=max(constraint,v)
                if minimum < -1e-6:raise AssertionError('Recursive reserve failure seed=%d trial=%d method=%s'%(seed,trial,method))
            rows.append([trial,0 if method=='V1' else 1,minimum,float(e.min()/capacity),constraint,shadow_gap])
            trajectories.append(states);powers.append(qs);switch_states.append(switches)
    save_npz(base/'trials.npz',metrics=np.array(rows),energy_after_j=np.asarray(trajectories),terminal_w=np.asarray(powers),terminals_on=np.asarray(switch_states))
    save_npz(base/'inputs.npz',initial_energy_j=np.asarray(initials),harvest_lower_w=np.asarray(lowers),
             actual_harvest_w=np.asarray(harvests),actual_base_w=np.asarray(bases),demand_bps=np.asarray(demands))
    result=dict(seed=seed,initial_trials=500,method_trials=1000,steps_per_trial=4,
                min_reserve_margin_j=float(np.min(np.array(rows)[:,2])),max_constraint_violation=float(np.max(np.array(rows)[:,4])),runtime_s=time.perf_counter()-start,
                config=asdict(c),source_sha256=source_hashes(),source_fingerprint=fingerprint(),phase='test',complete=True,
                actual_trial_inputs='inputs.npz overrides nominal initial_soc and harvest/base fields; 4 executed 60s steps, 8-step lower-bound reserve')
    write_json(base/'summary.json',result);return result


def run_correctness(frozen,workers):
    done=ROOT/'results_r2/E1/summary.json'
    if done.exists():
        import json
        return json.loads(done.read_text())
    parameters=frozen['selected_by_rho']['1.4']['HBAG']['parameters'];capacity=frozen['capacity_by_rho']['1.4']
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(trial_batch,[(s,capacity,parameters) for s in range(4000,4020)]))
    # Explicit counterexample: E starts at the safety floor below eclipse reserve.
    battery=Config(capacity_j=capacity).battery;e0=np.array([battery.minimum_j])
    after=step(e0,0.,55.,8.,60.,battery).after_j
    if not after[0]<battery.minimum_j:raise AssertionError('Expected initial-below-reserve counterexample')
    result=dict(complete=True,independent_initial_trials=sum(r['initial_trials'] for r in rows),
                method_trials=sum(r['method_trials'] for r in rows),methods=['V1','HBAG'],min_reserve_margin_j=min(r['min_reserve_margin_j'] for r in rows),
                counterexample=dict(initial_j=float(e0[0]),next_j=float(after[0]),minimum_j=battery.minimum_j,drop_below_floor_j=float(battery.minimum_j-after[0])),
                samples=rows,preflight='hbag3/test_gate.py and hbag3/test_v21.py cover handshake/capture/fixed power/V0/infeasible behavior')
    write_json(done,result);return result
