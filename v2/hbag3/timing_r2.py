"""R2 delayed actions, Appendix A early measurement, and measured sparse snapshots."""
from dataclasses import replace,asdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,time,resource,hashlib
import numpy as np
from .model import Config,RATE
from .environment import ROOT,build_environment,sun_vector
from .network import slot_harvest_lower
from .control import SwitchController,usable_catalogue,terminal_ports
from .allocation import allocate,protect_bus,problem_for_switch
from .energy import step,energy_drift
from .runner import seed_guard,disk_guard,save_npz,write_json,source_hashes,fingerprint
from .storage import storage_tier,shared_inputs,register_routes
from .matrix_v21 import matrix,policy_config
from .plan_r2 import FULL_SEEDS


def measured_state(times,states,when):
    # The prescribed delays and dt are integer seconds. Initial state is held
    # before t=0; no fabricated prehistory or extra orbit is simulated.
    i=int(np.searchsorted(times,max(float(when),0.),side='right')-1)
    return states[max(i,0)].copy()


def apply_action(env,slot,energy,action,h,delta):
    """Only current topology, demand and physical bus guards; no reserve refresh."""
    if action is None:
        # Common initial physical state is ON on every geometric terminal. Until
        # the first delayed command arrives, it carries no allocated traffic.
        initial=SwitchController(env['config'],'V0').decide(env,slot,energy)
        initial_cat=usable_catalogue(env['catalogues'][slot],initial,env['config'].routing)
        action=(initial,initial_cat,np.zeros(initial_cat.mask.shape))
    sw,oldcat,oldx=action;c=env['config'];cat=env['catalogues'][slot]
    count=len(cat.edges)//2;edges=cat.edges[:count];ports=terminal_ports(edges,c.per_plane,c.planes)
    oldmap={tuple(e):(sw.on[j],sw.capture[j]) for j,e in enumerate(oldcat.edges[:len(oldcat.edges)//2])}
    cut=energy<=c.cut_soc*c.capacity_j
    on=np.array([oldmap.get(tuple(e),(False,0.))[0] and not cut[e].any() for e in edges])
    capture=np.array([oldmap.get(tuple(e),(False,0.))[1] if on[j] else 0. for j,e in enumerate(edges)])
    states=np.zeros((len(energy),4),bool);available=np.zeros_like(states)
    a,b=edges.T;ap,bp=ports.T
    available[a,ap]=True;available[b,bp]=True
    states[a[on],ap[on]]=True;states[b[on],bp[on]]=True
    fixed=4*c.sleep_w+(25-c.sleep_w)*states.sum(axis=1)
    physical=replace(sw,on=on,capture=capture,ports=ports,terminals_on=states,available_terminals=available,
                     fixed_w=fixed,brownout=cut,newly_usable=np.zeros(count,bool))
    active=usable_catalogue(cat,physical,c.routing);x=np.zeros(active.mask.shape)
    for f,routes in enumerate(active.paths):
        targets={r:j for j,r in enumerate(routes) if r is not None}
        for j,route in enumerate(oldcat.paths[f]):
            if route in targets:x[f,targets[route]]+=oldx[f,j]
    harvest=env['harvest'].copy();harvest[slot]=h
    local=dict(env,dt=delta,harvest=harvest)
    p,meta=problem_for_switch(local,slot,active,physical,False)
    x,q,v=protect_bus(local,slot,energy,active,physical,x,p,meta)
    if v>1e-7:raise ArithmeticError('Delayed execution physical recovery failed')
    return (active,x,p,physical),q


def run_temporal(c,method,out,env=None,phase='test'):
    seed_guard(c,phase);disk_guard();out=Path(out)
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    start=time.perf_counter();env=build_environment(c) if env is None else dict(env,config=c)
    b=env['battery'];m=env['shell'].size;n=len(env['times']);dt=env['dt'];end=n*dt
    if dt<1 or c.delay_s>dt:raise ValueError('Registered temporal grid requires 1s<=dt and delay<=dt')
    tier=storage_tier(c.seed,phase);shared_inputs(env,out.parent,tier)
    ctl=SwitchController(c,method);margin=method=='HBAG+margin'
    D=max(b.charge_efficiency*c.solar_w,(c.base_w+4*env['terminal'].max_electrical_w)/b.discharge_efficiency)
    e=np.full(m,c.initial_soc*c.capacity_j);history_t=[0.];history_e=[e.copy()];drifts=[]
    grid_e=[e.copy()];grid_q=[];grid_z=[];grid_on=[];grid_mu=[];grid_pace=[];grid_budget=[]
    grid_fraction=[];grid_surplus=[];grid_remaining=[]
    price_diagnostics={key:[] for key in ['p_star_w','p_full_w','p_star_computed','reset_realistic_fill','reset_sustainable_slack']}
    grid_h=[];grid_delta=[];grid_spill=[];grid_unserved=[];integration_q=[];integration_h=[]
    active=None;pending=None;last_actual=None;previous_q=None;offered=0.;delivered=0.;terminal=0.;system=0.
    unsafe=np.zeros(m);minimum=e.copy();bounds=[];rows=[];full_cats=[];full_x=[];full_link=[];full_on=[];full_times=[]
    def activation_bound(item,at):
        _,_,observed,when,decision_slot=item
        actual=np.abs(e-observed);window=[v for tt,v in drifts if tt>when+1e-9]
        peak=np.max(window,axis=0) if window else np.zeros(m)
        bounds.append(dict(slot=decision_slot,measurement_s=when,execution_s=at,measured_max_j=float(actual.max()),
                           uniform_bound_j=c.delay_s*D,efficiency_correct_bound_j=float(c.delay_s*peak.max()),
                           uniform_excess_j=float(np.maximum(actual-c.delay_s*D,0).max())))
    for slot in range(n):
        disk_guard();t0=float(env['times'][slot]);stop=t0+dt
        if previous_q is not None:
            if last_actual is not None:
                cat,x,p,sw=last_actual;ctl.observe(cat,x,p,sw,env['terminal'])
            ctl.price.commit(previous_q,slot-1)
        if pending is not None and pending[0]<=t0+1e-9:
            activation_bound(pending,t0)
            active=pending[1];pending=None
        measurement=measured_state(history_t,history_e,t0-c.delay_s) if margin else e.copy()
        conservative=measurement-c.delay_s*D if margin else None
        sw=ctl.decide(env,slot,measurement,budget_energy=conservative)
        cat=usable_catalogue(env['catalogues'][slot],sw,c.routing)
        x,p,meta,info=allocate(env,slot,cat,sw,ctl)
        planned=(sw,cat,x)
        if margin or c.delay_s==0:active=planned
        else:pending=(t0+c.delay_s,planned,measurement.copy(),t0,slot)
        grid_mu.append(sw.price_trace['mu'].copy());grid_pace.append(sw.price_trace['pace_w'].copy());grid_budget.append(sw.raw_budget_w.copy())
        grid_fraction.append(sw.price_trace['pace_weight_fraction'].copy())
        grid_surplus.append(sw.price_trace['z_j'].copy());grid_remaining.append(sw.price_trace['remaining_slots'].copy())
        if ctl.price.kind=='dual':
            for key in price_diagnostics:price_diagnostics[key].append(sw.price_trace[key].copy())
        if margin:
            actual=np.abs(e-measurement)
            window=[v for tt,v in drifts if tt>max(t0-c.delay_s,0.)-1e-9]
            peak=np.max(window,axis=0) if window else np.zeros(m)
            bounds.append(dict(slot=slot,measurement_s=t0-c.delay_s,execution_s=t0,measured_max_j=float(actual.max()),
                               uniform_bound_j=c.delay_s*D,efficiency_correct_bound_j=float(c.delay_s*peak.max()),
                               uniform_excess_j=float(np.maximum(actual-c.delay_s*D,0).max())))
        q_sum=np.zeros(m);z_sum=np.zeros(c.flows);on_sum=np.zeros((m,4));t=t0;slot_min=float(e.min());max_ledger=0.
        h_sum=np.zeros(m);delta_sum=np.zeros(m);spill_sum=np.zeros(m);unserved_sum=np.zeros(m)
        while t<stop-1e-9:
            if pending is not None and pending[0]<=t+1e-9:
                activation_bound(pending,t)
                active=pending[1];pending=None
            next_event=min(t+1.,stop,pending[0] if pending is not None else stop);delta=next_event-t
            if delta<=0:raise RuntimeError('Nonadvancing temporal clock')
            h=slot_harvest_lower(env['shell'],t,delta,c.solar_w,sun_vector(c.beta_deg))
            executed,q=apply_action(env,slot,e,active,h,delta)
            if executed is None:z=np.zeros(c.flows);on=np.zeros((m,4),bool)
            else:
                actcat,actx,actp,actsw=executed;z=actx.sum(axis=1)*RATE;on=actsw.terminals_on;last_actual=executed
                if tier=='full':
                    full_cats.append(actcat);full_x.append(actx*RATE);full_link.append(actcat.incidence@actx.ravel()*RATE)
                    full_on.append(np.tile(actsw.on,2));full_times.append((t,next_event,slot))
            rec=step(e,h,c.base_w,q,delta,b)
            h_sum+=h*delta;delta_sum+=rec.delta_j;spill_sum+=rec.spill_j;unserved_sum+=rec.unserved_energy_j
            if tier=='full':integration_q.append(q.copy());integration_h.append(h.copy())
            max_ledger=max(max_ledger,float(np.max(np.abs(rec.after_j-(e+rec.delta_j-rec.spill_j+rec.unserved_energy_j)))))
            drifts.append((next_event,np.abs(energy_drift(h,c.base_w+q,b))))
            e=rec.after_j;history_t.append(next_event);history_e.append(e.copy());minimum=np.minimum(minimum,e)
            unsafe+=(e<b.minimum_j-1e-6)*delta;slot_min=min(slot_min,float(e.min()))
            q_sum+=q*delta;z_sum+=z*delta;on_sum+=on*delta
            offered+=float(env['demand_bps'][slot].sum()*delta);delivered+=float(z.sum()*delta)
            bus=np.maximum((c.base_w+q)*delta-b.discharge_efficiency*rec.unserved_energy_j,0.)
            terminal+=float((bus-np.minimum(c.base_w*delta,bus)).sum());system+=float(bus.sum());t=next_event
        if not margin and c.delay_s==0:
            bounds.append(dict(slot=slot,measurement_s=t0,execution_s=t0,measured_max_j=0.,uniform_bound_j=0.,efficiency_correct_bound_j=0.,uniform_excess_j=0.))
        previous_q=q_sum/dt;grid_e.append(e.copy());grid_q.append(previous_q);grid_z.append(z_sum/dt);grid_on.append(on_sum/dt)
        grid_h.append(h_sum/dt);grid_delta.append(delta_sum);grid_spill.append(spill_sum);grid_unserved.append(unserved_sum)
        rows.append(dict(slot=slot,min_energy_j=slot_min,delivered_bits=float(z_sum.sum()),terminal_energy_j=float(q_sum.sum()),
                         max_ledger_residual_j=max_ledger,iterations=info['iterations'],relative_gap=info.get('gap')))
        if slot%8==0:write_json(out/'status.json',dict(slot=slot+1,slots=n,elapsed_s=time.perf_counter()-start))
    if pending is not None and pending[0]<=end+1e-9:activation_bound(pending,end)
    arrays=dict(energy_j=np.asarray(grid_e),terminal_w=np.asarray(grid_q),delivered_bps=np.asarray(grid_z),
                on_time_fraction=np.asarray(grid_on),mu=np.asarray(grid_mu),pace_w=np.asarray(grid_pace),raw_budget_w=np.asarray(grid_budget),
                pace_weight_fraction=np.asarray(grid_fraction),discretionary_energy_j=np.asarray(grid_surplus),
                remaining_window_slots=np.asarray(grid_remaining),
                unsafe_seconds_per_node=unsafe,min_energy_j_per_node=minimum,
                actual_harvest_mean_w=np.asarray(grid_h),integrated_drift_j=np.asarray(grid_delta),
                integrated_spill_j=np.asarray(grid_spill),integrated_unserved_j=np.asarray(grid_unserved))
    if ctl.price.kind=='dual':arrays.update({key:np.asarray(value) for key,value in price_diagnostics.items()})
    if tier=='full' and full_cats:
        arrays.update(segment_times_s=np.asarray(full_times),allocation_bps=np.asarray(full_x),
                      executed_path_ids=register_routes(out.parent,full_cats),link_rate_bps=np.concatenate(full_link),link_on=np.concatenate(full_on),
                      segment_link_offsets=np.r_[0,np.cumsum([len(a) for a in full_link])],
                      integration_times_s=np.asarray(history_t),integration_energy_j=np.asarray(history_e),
                      integration_terminal_w=np.asarray(integration_q),integration_harvest_w=np.asarray(integration_h))
    save_npz(out/'raw.npz',**arrays);write_json(out/'slots.json',rows);write_json(out/'delay_bounds.json',bounds)
    result=dict(complete=True,config=asdict(c),phase=phase,seed=c.seed,method=method,storage_tier=tier,
        delivered_bits=delivered,offered_bits=offered,fvr=1-delivered/offered,terminal_energy_j=terminal,system_energy_j=system,
        ee_isl_bits_j=delivered/terminal,ee_sys_bits_j=delivered/system,
        esr=1-float(unsafe.sum()/(end*m)),min_soc=float(minimum.min()/c.capacity_j),
        violation_depth_j=float(np.maximum(b.minimum_j-minimum,0).max()),violation_duration_node_s=float(unsafe.sum()),
        dt_s=dt,delay_s=c.delay_s,slots=n,D_w=D,bound_measured_max_j=max(r['measured_max_j'] for r in bounds),
        bound_uniform_j=c.delay_s*D,bound_max_excess_j=max(r['uniform_excess_j'] for r in bounds),
        max_ledger_residual_j=max(r['max_ledger_residual_j'] for r in rows),runtime_s=time.perf_counter()-start,
        source_fingerprint=fingerprint(),source_sha256=source_hashes(),temporal_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing='Measure t-tau, execute t, reserve aligned' if margin else 'Measure t, execute t+tau, no reserve refresh',
        initial_history='E(t<0)=E(0); delta=tau*D still charged',
        physical='<=1s integration, fresh geometric/demand and empty-bus guards only; outdated routes absent from current catalogue dropped')
    write_json(out/'summary.json',result)
    from .compact_r2 import compact_slots
    compact_slots(out)
    return result


def temporal_cell(args):
    cell,frozen=args;base=Config(**cell['config']);seed_guard(base,'test')
    out=ROOT/'results_r2/E8'/cell['label']/('seed_%d'%base.seed)
    if all((out/m/'summary.json').exists() for m in cell['methods']):return
    env=build_environment(base,progress=lambda r:write_json(out/'environment_status.json',r))
    for label in cell['methods']:
        c,method=policy_config(base,label,frozen);run_temporal(c,method,out/label,env)


def scaling_cell(args):
    cell,frozen=args;base=Config(**cell['config']);c,_=policy_config(base,'HBAG',frozen)
    phase=cell.get('phase','test');seed_guard(c,phase)
    out=ROOT/cell.get('output_root','results_r2/E10')/cell['label']/('seed_%d'%c.seed)/'HBAG'
    if (out/'summary.json').exists():return
    from threadpoolctl import threadpool_limits,threadpool_info
    slots=[24,48] if c.planes*c.per_plane==5000 else [0,24,48,72];start=time.perf_counter()
    with threadpool_limits(limits=1):
        env=build_environment(c,snapshot_indices=slots,progress=lambda r:write_json(out/'status.json',r))
        from .forecast_r22 import precompute_price_catalogues
        precomputed=precompute_price_catalogues(env,slots,progress=lambda r:write_json(out/'status.json',r))
        write_json(out/'ephemeris_precomputation.json',precomputed)
        records=[];arrays={};cats=[]
        for t in slots:
            ctl=SwitchController(c,'HBAG');e=np.full(env['shell'].size,c.initial_soc*c.capacity_j)
            tick=time.perf_counter();sw=ctl.decide(env,t,e);after_decide=time.perf_counter()
            cat=usable_catalogue(env['catalogues'][t],sw,c.routing);after_catalogue=time.perf_counter()
            x,p,meta,info=allocate(env,t,cat,sw,ctl);after_allocate=time.perf_counter()
            x,q,v=protect_bus(env,t,e,cat,sw,x,p,meta);after_protect=time.perf_counter()
            elapsed=time.perf_counter()-tick;messages=2*p.resource_matrix.nnz*info['iterations'];m=env['shell'].size
            records.append(dict(slot=t,wall_s=elapsed,iterations=info['iterations'],relative_gap=info.get('gap'),
                scalar_messages=messages,intent_bits=4*m,messages_per_satellite=messages/m+4,
                bytes_per_satellite=(8*messages+4*m/8)/m,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                constraint_violation=v,shadow_s=info.get('shadow_runtime_s',0.),
                decide_s=after_decide-tick,usable_catalogue_s=after_catalogue-after_decide,
                allocate_s=after_allocate-after_catalogue,protect_bus_s=after_protect-after_allocate,
                algorithm_wall_s=max(0.,elapsed-info.get('shadow_runtime_s',0.))))
            arrays['energy_j_%d'%t]=e;arrays['terminal_w_%d'%t]=q;arrays['delivered_bps_%d'%t]=x.sum(axis=1)*RATE
            arrays['demand_bps_%d'%t]=env['demand_bps'][t];arrays['reserve_j_%d'%t]=env['reserve'][t]
            for key,field in [('mu','mu'),('pace_w','pace_w'),('pace_weight_fraction','pace_weight_fraction'),
                              ('discretionary_energy_j','z_j'),('remaining_window_slots','remaining_slots'),
                              ('p_star_w','p_star_w'),('p_full_w','p_full_w'),('p_star_computed','p_star_computed'),
                              ('price_reset_a','reset_realistic_fill'),('price_reset_b','reset_sustainable_slack')]:
                arrays[key+'_%d'%t]=sw.price_trace[field].copy()
            if c.seed in FULL_SEEDS or phase=='validation':
                cats.append(cat);arrays['allocation_bps_%d'%t]=x*RATE;arrays['directed_edges_%d'%t]=cat.edges
        if cats:arrays['executed_path_ids']=register_routes(out.parent,cats)
        save_npz(out/'raw.npz',**arrays);write_json(out/'snapshots.json',records)
        import os
        children=Path('/proc/%d/task/%d/children'%(os.getppid(),os.getppid()))
        concurrent=len(children.read_text().split()) if children.exists() else None
        write_json(out/'summary.json',dict(complete=True,seed=c.seed,method='HBAG',phase=phase,config=asdict(c),
            mean_slot_s=float(np.mean([r['algorithm_wall_s'] for r in records])),
            mean_total_slot_s=float(np.mean([r['wall_s'] for r in records])),mean_iterations=float(np.mean([r['iterations'] for r in records])),
            mean_messages_per_satellite=float(np.mean([r['messages_per_satellite'] for r in records])),
            mean_bytes_per_satellite=float(np.mean([r['bytes_per_satellite'] for r in records])),
            peak_rss_bytes=max(r['peak_rss_bytes'] for r in records),runtime_s=time.perf_counter()-start,environment_build_s=env['build_s'],
            mean_decide_s=float(np.mean([r['decide_s'] for r in records])),
            mean_usable_catalogue_s=float(np.mean([r['usable_catalogue_s'] for r in records])),
            mean_allocate_s=float(np.mean([r['allocate_s'] for r in records])),
            mean_protect_bus_s=float(np.mean([r['protect_bus_s'] for r in records])),
            path_precomputation_s=precomputed['path_total_s'],path_precomputation_mean_slot_s=precomputed['path_mean_s_per_slot'],
            path_precomputation_mean_satellite_slot_s=precomputed['path_mean_s_per_satellite_slot'],
            precomputed_slots=precomputed['constructed_slots'],snapshot_slots=slots,
            configured_parallel_workers=cell.get('parallel_workers',1),concurrent_worker_processes=concurrent,
            online_timing_excludes='Public geometric candidate-path and sparse incidence precomputation; reported separately',
            threadpools=threadpool_info(),snapshot_policy='Independent cold starts at SOC .95; physical grid stays 96 slots/orbit',
            source_fingerprint=fingerprint(),source_sha256=source_hashes()))


def run_temporal_matrix(frozen,workers):
    from .gate import log
    from .analysis_r2 import summaries
    cells=matrix('E8',frozen);write_json(ROOT/'configs_r2/E8.json',cells)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(temporal_cell,[(c,frozen) for c in cells]):disk_guard()
    summaries('E8');log('R2 E8′与附录A完成：全部20个独立测试种子，1s子步与提前测量补偿。')


def run_scaling_matrix(frozen,workers):
    from .gate import log
    from .analysis_r2 import summaries
    workers=min(workers,max(1,physical_core_count()//2))
    cells=[dict(c,parallel_workers=workers) for c in matrix('E10',frozen)];write_json(ROOT/'configs_r2/E10.json',cells)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(scaling_cell,[(c,frozen) for c in cells]):disk_guard()
    summaries('E10');log('R2 v2.3 E10′完成：256–1584星20种子×4快照，5000星5种子×2快照；星历路径预计算单独报告。')


def physical_core_count():
    import os
    cores=set()
    for cpu in os.sched_getaffinity(0):
        root=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
        if (root/'core_id').exists():cores.add(((root/'physical_package_id').read_text().strip(),(root/'core_id').read_text().strip()))
    return len(cores) or 1
