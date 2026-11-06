"""Declared experiment matrix. Every reported stochastic cell uses all 20 test seeds."""
from dataclasses import replace,asdict
from itertools import product
from pathlib import Path
import csv,json,time,os,sys,resource
import numpy as np
from .model import Config,RATE,POWER
from .environment import ROOT,build_environment
from .runner import run_policy,write_json,source_hashes
from .solvers import make_problem,solve_dual,solve_admm,kkt_residual
from .optimization import solve_centralized,solve_distributed,primal_prox

SEEDS=list(range(2000,2020));VALIDATION=list(range(1000,1010))
MAIN_METHODS=['V0','V1','V2','HBAG','Static+Guard','Centralized-opt','DualDecomp','ADMM','MPC-H']
CORE=['V0','V1','V2','HBAG','Static+Guard']

def matrix(experiment):
    cells=[]
    def add(label,methods=CORE,seeds=SEEDS,**kw):
        for seed in seeds:cells.append(dict(experiment=experiment,label=label,config=asdict(Config(seed=seed,**kw)),methods=methods))
    if experiment=='E2':add('main',MAIN_METHODS)
    elif experiment=='E4':
        # Optical source is the exact E2 run, linked in the report; RF keeps identical absolute demand.
        for leakage in [-300.,-20.,-10.,-5.,0.]:add('rf_IN_'+str(leakage),medium='rf',interference_db=leakage)
    elif experiment=='E5':
        for load,traffic in product([.3,.5,.65,.8,.95,1.1],['steady','mmpp','hotspot3','hotspot5','eclipse']):add('load_%s_%s'%(load,traffic),load=load,traffic=traffic)
    elif experiment=='E6':
        for k,routing in product([1,3,5],['hops','distance','energy']):add('K%d_%s'%(k,routing),paths=k,routing=routing)
    elif experiment=='E7':
        for beta,soc in product([0.,30.,60.],[.4,.6,.8]):add('beta_%s_soc_%s'%(beta,soc),orbits=10,beta_deg=beta,initial_soc=soc)
        for theta in [.4,.5,.6]:add('artificial_theta_'+str(theta),orbits=10,artificial_eclipse=theta)
        for preset in ['moderate','abundant']:add('energy_'+preset,orbits=10,preset=preset)
    elif experiment=='E8':
        for dt,delay in product([5.,15.,30.,60.],[0.,.1,1.,5.,15.]):add('dt_%s_delay_%s'%(dt,delay),['HBAG'],planes=8,per_plane=8,flows=16,orbits=1,dt_override_s=dt,delay_s=delay,substep_s=1.)
    elif experiment=='E9':add('frozen_and_online',['HBAG','DualDecomp','ADMM'],planes=8,per_plane=8,flows=16,orbits=1,max_iterations=10000)
    elif experiment=='E10':
        for p,s in [(16,16),(24,24),(32,32),(22,72),(50,100)]:add('M_'+str(p*s),['HBAG'],planes=p,per_plane=s,flows=max(48,p*s//12),orbits=1,slots_per_orbit=96)
    elif experiment=='E11':
        for alpha,lam,eps,rho in product([.01,.04],[.02,.1],[.02,.1],[.02,.1]):add('a%s_l%s_e%s_r%s'%(alpha,lam,eps,rho),['HBAG'],seeds=VALIDATION,alpha=alpha,lam=lam,epsilon=eps,rho=rho)
    else:raise ValueError(experiment)
    return cells


def run_cell(cell):
    c=Config(**cell['config']);exp=cell['experiment'];base=ROOT/'results'/exp/cell['label']/('seed_'+str(c.seed))
    if exp=='E8':
        from .temporal import run_delayed
        return [run_delayed(c,base/'HBAG')]
    if exp=='E9':return convergence(c,base)
    if exp=='E10':return benchmark(c,base)
    env=build_environment(c);output=[]
    for method in cell['methods']:
        output.append(run_policy(c,method,base/method,env))
    return output


def convergence(c,base):
    base.mkdir(parents=True,exist_ok=True);done=base/'convergence.json'
    if done.exists():return json.loads(done.read_text())
    env=build_environment(c);energy=np.full(env['shell'].size,c.initial_soc*c.battery.capacity_j);rows=[];witness={}
    # Three orbit phases, independent central optimum at every frozen snapshot.
    for t in [0,len(env['times'])//3,2*len(env['times'])//3]:
        p,meta=make_problem(env,t,energy,'HBAG');central=solve_centralized(p,tolerance=1e-10);witness['central_%d'%t]=central['allocation']*RATE
        for name,array in [('A_data',p.resource_matrix.data),('A_indices',p.resource_matrix.indices),('A_indptr',p.resource_matrix.indptr),('A_shape',np.array(p.resource_matrix.shape)),('budget',p.budgets),('demand',p.demand),('mask',p.path_mask),('cost',p.linear_cost)]:witness['%s_%d'%(name,t)]=array
        for method in ['HBAG','DualDecomp','ADMM']:
            if method=='HBAG':
                r=solve_distributed(p,max_iterations=c.max_iterations,tolerance=0.,check_every=20);history=r.history;x=r.allocation
            else:
                x,info=(solve_dual if method=='DualDecomp' else solve_admm)(p,c.max_iterations,0.,True);history=info['history']
            for h in history:
                val=h.get('primal_objective',h.get('objective'))
                rows.append(dict(seed=c.seed,slot=t,method=method,iteration=h['iteration'],central_objective=central['objective'],objective=val,objective_gap=max(0.,central['objective']-val),raw_violation=h['raw_violation'],kkt_residual=h.get('kkt_residual',h.get('kkt'))))
            witness['%s_%d'%(method,t)]=x*RATE
    # Online tracking recomputes the independent optimum at each moving physical state.
    from .energy import step
    from .solvers import node_power
    online=[]
    for t in range(len(env['times'])):
        p,meta=make_problem(env,t,energy,'HBAG');central=solve_centralized(p);r=solve_distributed(p,max_iterations=200,tolerance=c.tolerance)
        online.append(dict(slot=t,central_objective=central['objective'],executed_objective=p.objective(r.allocation),objective_gap=max(0.,central['objective']-p.objective(r.allocation)),rate_error_bps=float(np.linalg.norm((r.allocation-central['allocation'])*RATE)),kkt=kkt_residual(p,r.last_iterate,r.dual)))
        energy=step(energy,env['harvest'][t],c.base_w,node_power(env['catalogues'][t],env['terminal'],r.allocation,meta['idle_w']),env['dt'],env['battery']).after_j
    np.savez_compressed(base/'allocations.npz',**witness)
    write_json(base/'convergence.json',rows);write_json(base/'tracking.json',online);write_json(base/'manifest.json',dict(config=asdict(c),central_solver='Clarabel; independent frozen optimum',environment_cache=env['cache_key'],source_sha256=source_hashes()))
    return rows


def benchmark(c,base):
    import psutil
    base.mkdir(parents=True,exist_ok=True)
    if (base/'benchmark.json').exists():return json.loads((base/'benchmark.json').read_text())
    indices=[0,c.slots_per_orbit//4,c.slots_per_orbit//2,3*c.slots_per_orbit//4]
    build_start=time.perf_counter();env=build_environment(c,snapshot_indices=indices);build_s=time.perf_counter()-build_start;rows=[]
    before_affinity=os.sched_getaffinity(0)
    # Actual measurements with one logical CPU and one BLAS thread. No extrapolation.
    os.sched_setaffinity(0,{0})
    try:
        for t in indices:
            start=time.perf_counter();p,meta=make_problem(env,t,np.full(env['shell'].size,c.initial_soc*c.battery.capacity_j),'HBAG');build_problem_s=time.perf_counter()-start
            result=solve_distributed(p,max_iterations=c.max_iterations,tolerance=c.tolerance);cat=env['catalogues'][t]
            owner=np.r_[cat.edges[:,0],np.arange(env['shell'].size)]
            messages=np.bincount(owner,weights=2*np.diff(p.resource_matrix.indptr)*result.iterations,minlength=env['shell'].size)
            # Time actual per-source commodity proximal work; no distributed-network latency is asserted.
            local=[]
            for node in np.unique(cat.sources[cat.sources>=0]):
                selected=cat.sources==node
                from .optimization import AllocationProblem
                sub=AllocationProblem(sparse_empty(selected.sum()*c.paths),np.empty(0),p.demand[selected],p.path_mask[selected],p.linear_cost[selected],p.utility_weight[selected],p.utility_scale[selected],p.regularization)
                trial=np.zeros(sub.shape);tic=time.perf_counter()
                for _ in range(100):primal_prox(trial,1.,sub)
                local.append((time.perf_counter()-tic)/100)
            rows.append(dict(seed=c.seed,nodes=env['shell'].size,slot=t,flows=c.flows,paths=c.paths,build_s=build_s,problem_s=build_problem_s,solver_s=result.runtime_s,wall_slot_s=result.runtime_s+build_problem_s,iterations=result.iterations,gap=result.relative_gap,logical_messages_per_satellite_mean=float(messages.mean()),logical_messages_per_satellite_max=float(messages.max()),logical_payload_bytes_per_satellite_mean=float(8*messages.mean()),rss_bytes=psutil.Process().memory_info().rss,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,local_subproblem_mean_s=float(np.mean(local)),local_subproblem_max_s=float(max(local)),cpu_affinity=[0],blas_threads=1,measurement='x86 CPU proxy; logical scalar payload excludes protocol headers; no ARM measurement'))
            np.savez_compressed(base/('snapshot_%d.npz'%t),allocation_bps=result.allocation*RATE,messages_per_satellite=messages,local_subproblem_seconds=local)
    finally:os.sched_setaffinity(0,before_affinity)
    write_json(base/'benchmark.json',rows);write_json(base/'manifest.json',dict(config=asdict(c),environment_cache=env['cache_key'],platform=os.uname().machine,source_sha256=source_hashes(),dt_s=env['dt'],snapshot_indices=indices))
    return rows


def sparse_empty(n):
    from scipy import sparse
    return sparse.csr_matrix((0,n))
