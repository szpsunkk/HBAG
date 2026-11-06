"""P2: newly trained R2 PPO+V1 and a disclosed small fractional-switch oracle."""
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor
import json,time,hashlib
import numpy as np
from .environment import ROOT,build_environment
from .model import Config,RATE,POWER
from .control import SwitchController,usable_catalogue
from .allocation import problem_for_switch,protect_bus
from .optimization import solve_distributed
from .energy import step
from .runner import seed_guard,disk_guard,write_json,save_npz,fingerprint,source_hashes,run_policy
from .storage import shared_inputs,storage_tier,register_routes
from .ppo import network,observation,LEVELS


def config(frozen,seed,small=False,orbits=3):
    return Config(seed=seed,rho_energy=1.4,capacity_j=frozen['capacity_by_rho']['1.4'],load=frozen['load'],orbits=orbits,
                  planes=8 if small else 24,per_plane=8 if small else 24,flows=32 if small else frozen['flows'])


def ppo_action(env,t,energy,ctl,fraction):
    sw=ctl.decide(env,t,energy);cat=usable_catalogue(env['catalogues'][t],sw,env['config'].routing)
    p,meta=problem_for_switch(env,t,cat,sw,True)
    admits=np.where(cat.sources>=0,fraction[np.maximum(cat.sources,0)],0.)
    p=replace(p,demand=p.demand*admits)
    solved=solve_distributed(p,max_iterations=env['config'].max_iterations,tolerance=env['config'].tolerance)
    x,q,v=protect_bus(env,t,energy,cat,sw,solved.allocation,p,meta)
    ctl.observe(cat,x,p,sw,env['terminal']);ctl.price.commit(q,t)
    return x,q,cat,sw,solved


def train_ppo(frozen,steps=4096):
    import torch
    root=ROOT/'results_r2/E13/training';root.mkdir(parents=True,exist_ok=True)
    if (root/'model.pt').exists() and (root/'training.json').exists():return
    torch.set_num_threads(1);torch.manual_seed(1000);rng=np.random.default_rng(1000)
    model=network();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    start=time.perf_counter();envs={};frames=[];rewards=[];episode=0;t=0;env=None
    for k in range(steps):
        disk_guard()
        if env is None or t>=len(env['times']):
            seed=1000+episode%10;c=config(frozen,seed,small=True);seed_guard(c,'validation')
            if seed not in envs:envs[seed]=build_environment(c,progress=lambda r:write_json(root/'environment_status.json',dict(r,seed=seed)))
            env=envs[seed];e=np.full(env['shell'].size,c.initial_soc*c.capacity_j);ctl=SwitchController(c,'V1');t=0;episode+=1
        obs=observation(env,t,e)
        with torch.no_grad():
            logits,value=model(torch.from_numpy(obs));dist=torch.distributions.Categorical(logits=logits);actions=dist.sample();logp=dist.log_prob(actions)
        x,q,cat,sw,solved=ppo_action(env,t,e,ctl,LEVELS[actions.numpy()])
        rec=step(e,env['harvest'][t],c.base_w,q,env['dt'],env['battery'])
        served=np.bincount(np.maximum(cat.sources,0),weights=x.sum(axis=1),minlength=env['shell'].size)
        reward=served-.01*q/100.-10*(rec.after_j<env['battery'].minimum_j-1e-6)
        frames.append((obs,actions.numpy(),logp.numpy(),value.numpy(),reward.astype(np.float32),t==len(env['times'])-1))
        rewards.append(float(reward.mean()));e=rec.after_j;t+=1
        if len(frames)==128 or k==steps-1:
            with torch.no_grad():_,nv=model(torch.from_numpy(observation(env,min(t,len(env['times'])-1),e)));last=nv.numpy()
            advantage=np.zeros((len(frames),env['shell'].size),np.float32);gae=np.zeros(env['shell'].size,np.float32)
            for j in range(len(frames)-1,-1,-1):
                _,_,_,v,r,done=frames[j];cont=0. if done else 1.
                delta=r+.99*last*cont-v;gae=delta+.99*.95*cont*gae;advantage[j]=gae;last=v
            observations=torch.from_numpy(np.concatenate([f[0] for f in frames]));actions=torch.from_numpy(np.concatenate([f[1] for f in frames]))
            oldlog=torch.from_numpy(np.concatenate([f[2] for f in frames]));values=np.stack([f[3] for f in frames])
            returns=torch.from_numpy((advantage+values).ravel());adv=torch.from_numpy(advantage.ravel());adv=(adv-adv.mean())/(adv.std()+1e-8)
            for epoch in range(4):
                for index in np.array_split(rng.permutation(len(actions)),16):
                    logits,v=model(observations[index]);dist=torch.distributions.Categorical(logits=logits)
                    ratio=torch.exp(dist.log_prob(actions[index])-oldlog[index]);surrogate=torch.minimum(ratio*adv[index],torch.clamp(ratio,.8,1.2)*adv[index])
                    loss=-surrogate.mean()+.5*((v-returns[index])**2).mean()-.01*dist.entropy().mean()
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            frames=[];write_json(root/'status.json',dict(completed_steps=k+1,budget=steps,elapsed_s=time.perf_counter()-start))
    torch.save(dict(state_dict=model.state_dict(),steps=steps),root/'model.pt')
    save_npz(root/'training_raw.npz',reward=np.array(rewards))
    write_json(root/'training.json',dict(complete=True,training_env_steps=steps,nodes=64,validation_seeds=list(range(1000,1010)),
        training_satellite_transitions=steps*64,test_seeds_used_for_training=[],device='CPU',network='4-32-32 tanh shared actor-critic, 5 admission fractions',
        optimizer='Adam lr=3e-4',ppo=dict(clip=.2,gamma=.99,gae=.95,rollout=128,epochs=4),
        routing_safety_layer='R2 V1 terminal switching, CP routing and hard budget; hybrid, no external RL reproduction',
        runtime_s=time.perf_counter()-start,source_sha256=source_hashes(),optional_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def evaluate_ppo(args):
    seed,frozen=args;c=config(frozen,seed);seed_guard(c,'test')
    cell=ROOT/'results_r2/E13/main'/('seed_%d'%seed);out=cell/'PPO+V1'
    if (out/'summary.json').exists():return
    import torch
    torch.set_num_threads(1);model=network();model.load_state_dict(torch.load(ROOT/'results_r2/E13/training/model.pt',map_location='cpu')['state_dict']);model.eval()
    env=build_environment(c);shared_inputs(env,cell,storage_tier(seed,'test'));ctl=SwitchController(c,'V1')
    e=np.full(env['shell'].size,c.initial_soc*c.capacity_j);energies=[e.copy()];powers=[];allocations=[];states=[];cats=[];start=time.perf_counter()
    for t in range(len(env['times'])):
        disk_guard()
        with torch.no_grad():logits,_=model(torch.from_numpy(observation(env,t,e)));fraction=torch.softmax(logits,dim=-1).numpy()@LEVELS
        x,q,cat,sw,solved=ppo_action(env,t,e,ctl,fraction)
        e=step(e,env['harvest'][t],c.base_w,q,env['dt'],env['battery']).after_j
        energies.append(e.copy());powers.append(q);allocations.append(x*RATE);states.append(sw.terminals_on);cats.append(cat)
    e=np.asarray(energies);q=np.asarray(powers);x=np.asarray(allocations);dt=env['dt'];delivered=float(x.sum()*dt);offered=float(env['demand_bps'].sum()*dt)
    raw=dict(energy_j=e,terminal_w=q,terminals_on=np.asarray(states),delivered_bps=x.sum(axis=2))
    if 4000<=seed<=4004:raw.update(allocation_bps=x,executed_path_ids=register_routes(cell,cats),link_rate_bps=np.concatenate([cat.incidence@a.ravel() for cat,a in zip(cats,x)]))
    save_npz(out/'raw.npz',**raw)
    write_json(out/'summary.json',dict(complete=True,config=asdict(c),phase='test',method='PPO+V1',seed=seed,storage_tier=storage_tier(seed,'test'),
        delivered_bits=delivered,offered_bits=offered,fvr=1-delivered/offered,esr=float(np.mean(e[1:]>=env['battery'].minimum_j-1e-6)),
        min_soc=float(e.min()/c.capacity_j),violation_depth_j=float(np.maximum(env['battery'].minimum_j-e,0).max()),terminal_energy_j=float(q.sum()*dt),
        system_energy_j=float((q+c.base_w).sum()*dt),dt_s=dt,slots=len(x),runtime_s=time.perf_counter()-start,source_fingerprint=fingerprint(),source_sha256=source_hashes()))


def fractional_mpc(env,t,energy,previous,horizon,return_plan=False):
    """Oracle convex relaxation; a shared undirected a in [0,1], no hold constraints."""
    import cvxpy as cp
    from scipy import sparse
    c=env['config'];m=env['shell'].size;dt=env['dt'];b=env['battery'];term=env['terminal']
    horizon=min(horizon,len(env['times'])-t);E=cp.Variable((horizon+1,m));xs=[];acts=[];power=[]
    constraints=[E[0]==energy/1e6,E>=b.minimum_j/1e6,E<=b.capacity_j/1e6];objective=0
    previous_expr={k:float(v) for k,v in previous.items()};elapsed=time.perf_counter()
    for k in range(horizon):
        cat=env['catalogues'][t+k];count=len(cat.edges)//2;x=cp.Variable(cat.mask.shape,nonneg=True);a=cp.Variable(count)
        change=cp.Variable(count,nonneg=True);keys=[tuple(e) for e in cat.edges[:count]]
        old=cp.hstack([previous_expr.get(key,0.) for key in keys])
        constraints+=[a>=0,a<=1,change>=a-old,change<=a]
        # Linear acquisition relaxation and the same full optical capacity.
        rates=cat.incidence@cp.reshape(x,(cat.mask.size,),order='C')
        effective=a-min(c.acquisition_s/dt,1.)*change
        constraints+=[rates<=cp.multiply(cat.link_capacity_bps/RATE,cp.hstack([effective,effective])),cp.sum(x,axis=1)<=env['demand_bps'][t+k]/RATE]
        disabled=np.flatnonzero(~cat.mask.ravel())
        if len(disabled):constraints.append(cp.reshape(x,(cat.mask.size,),order='C')[disabled]==0)
        inc=cat.sender@sparse.diags(term.derivative(0.,cat.distance_km)*RATE)@cat.incidence
        degree=cat.sender[:,:count]+cat.sender[:,count:]
        q=4*c.sleep_w+(25-c.sleep_w)*(degree@a)+inc@cp.reshape(x,(cat.mask.size,),order='C')
        constraints+=[inc@cp.reshape(x,(cat.mask.size,),order='C')<=4*(60-25)]
        net=env['harvest'][t+k]-c.base_w-q
        constraints+=[E[k+1]<=E[k]+dt*b.charge_efficiency*net/1e6,E[k+1]<=E[k]+dt/b.discharge_efficiency*net/1e6]
        scale=c.utility_s_bps/RATE
        objective+=cp.sum(scale*cp.log1p(cp.sum(x,axis=1)/scale))-c.alpha*cp.sum(inc@cp.reshape(x,(cat.mask.size,),order='C'))/POWER-c.rho/2*cp.sum_squares(x)
        xs.append(x);acts.append(a);power.append(q);previous_expr={key:a[j] for j,key in enumerate(keys)}
    constraints+=[E[-1]>=env['reserve'][t+horizon]/1e6]
    problem=cp.Problem(cp.Maximize(objective),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-8,max_iter=300)
    if xs[0].value is None:raise RuntimeError('Fractional MPC failed: '+str(problem.status))
    max_violation=max(float(np.max(np.abs(con.violation()),initial=0)) for con in constraints)
    if max_violation>1e-5:raise RuntimeError('Fractional MPC constraint residual exceeds 1e-5 in scaled coordinates')
    info=dict(
        runtime_s=time.perf_counter()-elapsed,status=problem.status,horizon=horizon,objective=float(problem.value),max_scaled_violation=max_violation,
        predicted_energy_j=np.asarray(E.value)*1e6)
    if return_plan:
        return [np.maximum(x.value,0) for x in xs],[np.clip(a.value,0,1) for a in acts],[np.asarray(q.value) for q in power],info
    return np.maximum(xs[0].value,0),np.clip(acts[0].value,0,1),np.asarray(power[0].value),info


def evaluate_mpc(args):
    seed,frozen=args[:2];options=args[2] if len(args)>2 else {};phase=options.get('phase','test')
    c=config(frozen,seed,small=True,orbits=1);seed_guard(c,phase)
    cell=ROOT/options.get('output_root','results_r2/MPC/small')/('seed_%d'%seed);out=cell/'MPC-H-relaxed'
    if (out/'summary.json').exists() and (cell/'HBAG/summary.json').exists():return
    env=build_environment(c);N=c.slots_per_orbit;evaluation=c
    shared_inputs(env,cell,storage_tier(seed,phase))
    from .matrix_v21 import policy_config
    hbag_config,_=policy_config(evaluation,'HBAG',frozen)
    run_policy(hbag_config,'HBAG',cell/'HBAG',env=env,phase=phase)
    from .compact_r2 import compact_run,compact_slots
    compact_run(cell/'HBAG');compact_slots(cell/'HBAG')
    if (out/'summary.json').exists():return
    e=np.full(64,c.initial_soc*c.capacity_j)
    first=env['catalogues'][0];previous={tuple(edge):0. for edge in first.edges[:len(first.edges)//2]}
    states=[e.copy()];powers=[];allocations=[];fractions=[];ledger=[];start=time.perf_counter()
    write_json(out/'status.json',dict(phase='offline_solve',horizon=N,solver_calls=1))
    disk_guard();planned_x,planned_a,planned_q,info=fractional_mpc(env,0,e,previous,N,return_plan=True)
    predicted=info.pop('predicted_energy_j');write_json(out/'offline_solve.json',dict(info,solver_calls=1,solve_at_slot=0))
    for t,(x,a,q) in enumerate(zip(planned_x,planned_a,planned_q)):
        disk_guard();rec=step(e,env['harvest'][t],c.base_w,q,env['dt'],env['battery'])
        residual=float(np.max(np.abs(rec.after_j-(e+rec.delta_j-rec.spill_j+rec.unserved_energy_j))))
        e=rec.after_j;states.append(e.copy());powers.append(q);allocations.append(x*RATE);fractions.append(a)
        ledger.append(dict(slot=t,ledger_residual_j=residual,spill_j=float(rec.spill_j.sum()),unserved_j=float(rec.unserved_energy_j.sum())))
        write_json(out/'status.json',dict(phase='execute_offline_plan',slot=t+1,slots=N,solver_calls=1,elapsed_s=time.perf_counter()-start))
    e=np.asarray(states);q=np.asarray(powers);x=np.asarray(allocations);dt=env['dt'];delivered=float(x.sum()*dt);offered=float(env['demand_bps'][:len(x)].sum()*dt)
    raw=dict(energy_j=e,terminal_w=q,delivered_bps=x.sum(axis=2),predicted_energy_j=predicted)
    if storage_tier(seed,phase)=='full':raw.update(allocation_bps=x,fractional_switch=np.asarray(fractions),executed_path_ids=register_routes(cell,env['catalogues'][:len(x)]))
    save_npz(out/'raw.npz',**raw);write_json(out/'slots.json',ledger)
    write_json(out/'summary.json',dict(complete=True,config=asdict(evaluation),phase=phase,seed=seed,method='MPC-H-relaxed',evaluated_orbits=1,forecast_environment_orbits=1,
        delivered_bits=delivered,offered_bits=offered,fvr=1-delivered/offered,esr=float(np.mean(e[1:]>=env['battery'].minimum_j-1e-6)),min_soc=float(e.min()/c.capacity_j),
        violation_depth_j=float(np.maximum(env['battery'].minimum_j-e,0).max()),terminal_energy_j=float(q.sum()*dt),system_energy_j=float((q+c.base_w).sum()*dt),
        dt_s=dt,runtime_s=time.perf_counter()-start,source_fingerprint=fingerprint(),source_sha256=source_hashes(),
        solver_calls=1,offline_solve_s=info['runtime_s'],max_scaled_violation=info['max_scaled_violation'],
        max_ledger_residual_j=max(row['ledger_residual_j'] for row in ledger),
        limitations='64 satellites only; t=0 single offline H=96 solve with true demand; same demand as HBAG; shared fractional switches, relaxed acquisition, no minimum hold, fixed K4 candidates; optimistic reference, not a physical binary policy or a formal throughput upper bound'))


def run_optional(frozen,workers):
    from .gate import log
    from .analysis_r2 import summaries
    done=ROOT/'results_r2/P2_DONE.json'
    if done.exists():return
    train_ppo(frozen)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(evaluate_ppo,[(s,frozen) for s in range(4000,4020)]):disk_guard()
    import os
    for seed in range(4000,4020):
        parent=ROOT/'results_r2/E13/main'/('seed_%d'%seed)
        for method in ['V0','V1','HBAG']:
            p=parent/method;original=ROOT/'results_r2/E2/rho_1.4'/('seed_%d'%seed)/method
            if not p.exists():p.symlink_to(os.path.relpath(original,parent),target_is_directory=True)
    summaries('E13');log('R2 E13′完成：64星验证场景重新训练4096步，576星20个新测试种子评估PPO+V1。')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(evaluate_mpc,[(s,frozen) for s in range(4000,4020)]):disk_guard()
    summaries('MPC');write_json(done,dict(complete=True,source_fingerprint=fingerprint(),time=time.time()))
    log('R2 P2 DONE')
