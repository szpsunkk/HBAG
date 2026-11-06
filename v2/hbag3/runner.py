"""R2 auditable execution: validation gate, shared inputs, float32 raw arrays."""
from dataclasses import asdict
from pathlib import Path
import hashlib, json, os, platform, shutil, sys, time
import numpy as np
from .model import RATE
from .energy import step, reserve_envelope
from .environment import ROOT, build_environment
from .control import SwitchController, usable_catalogue
from .allocation import allocate, protect_bus
from .solvers import objective
from .storage import storage_tier,shared_inputs,register_routes
from .plan_r2 import VERSION, FROZEN, TEST_SEEDS, RF_TEST_SEEDS

CORE = ('plan_r2.py','reserve_pricing_r23.py','forecast_r22.py','diagnostics_r22.py','model.py','energy.py','network.py','traffic.py','environment.py',
        'control.py','pricing.py','link_values.py','storage.py','allocation.py','optimization.py','central_reference_r2.py','solvers.py','runner.py',
        'gate.py','matrix_v21.py','pipeline_r2.py','timing_r2.py','optional_r2.py','ppo.py','protocol_r231.py')


def source_hashes():
    return {'hbag3/'+name: hashlib.sha256((ROOT/'hbag3'/name).read_bytes()).hexdigest() for name in CORE}


def fingerprint():
    return hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False,
                              default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))+'\n')
    temp.replace(path)


def disk_guard():
    free = shutil.disk_usage(ROOT).free
    if free < 25_000_000_000:
        raise RuntimeError('DISK_PAUSE: %.3f GB free < 25 GB; preserve all R1/R2 artifacts' % (free/1e9))
    frozen=ROOT/FROZEN
    if frozen.exists() and frozen.read_bytes() != (ROOT/'EXPERIMENT_PLAN_R2.md').read_bytes():
        raise RuntimeError('PLAN_CHANGED: live plan differs from frozen '+VERSION+'; stop before mixing definitions')
    return free


def seed_guard(config, phase, experiment=None, method=None):
    if phase == 'validation':
        if config.seed not in range(1000, 1010):
            raise ValueError('G0/calibration/selection permit validation seeds 1000–1009 only')
    elif phase == 'test':
        if config.seed in RF_TEST_SEEDS:
            from .protocol_r231 import guard_rf_test
            guard_rf_test(config,experiment,method)
            return
        if VERSION=='v2.3.1':
            raise RuntimeError('Approved v2.3 test results are read-only; v2.3.1 permits only E4-double-prime seeds 5000–5019')
        gate = ROOT/'results_r2/G0_prime/gate.json'
        selection=ROOT/'results_r2/E11_validation/frozen.json'
        if any(not p.exists() for p in [gate,selection]):raise RuntimeError('Test seeds locked until BOTH G0-prime and E11-prime complete')
        for path in [gate,selection]:
            record=json.loads(path.read_text())
            if record.get('state')!='PASS' or record.get('plan_version')!=VERSION or record.get('source_fingerprint')!=fingerprint():
                raise RuntimeError('Test seed lock: gate/selection not PASS for current '+VERSION+' source')
        if config.seed not in TEST_SEEDS:
            raise ValueError('R2 v2.3 test seeds must be 4000–4019')
    else:
        raise ValueError('An explicit validation/test phase is required')


def save_npz(path, **arrays):
    disk_guard()
    values = {k: np.asarray(v, dtype=np.float32) if np.asarray(v).dtype.kind == 'f' else np.asarray(v) for k,v in arrays.items()}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz')
    np.savez_compressed(temp, **values); temp.replace(path)


def pack_routes(catalogues):
    nodes = []; offsets = [0]; slot_offsets = [0]
    for cat in catalogues:
        for routes in cat.paths:
            for route in routes:
                nodes.extend(route or ()); offsets.append(len(nodes))
        slot_offsets.append(len(offsets)-1)
    return dict(path_nodes=np.asarray(nodes,np.int32), path_offsets=np.asarray(offsets,np.int32),
                slot_path_offsets=np.asarray(slot_offsets,np.int32))


def save_inputs(env, cell):
    cell = Path(cell); cell.mkdir(parents=True, exist_ok=True)
    config = asdict(env['config'])
    for key in ('alpha','lam','epsilon','rho','guard_soc','kappa_off','xi_wake','max_iterations','tolerance','dual_eta','dual_warm','lyapunov_lambda','lyapunov_theta'):
        config.pop(key, None)
    meta = dict(config=config, source_sha256=source_hashes(), dt_s=env['dt'],
                reference_scale_bps=env['reference_scale_bps'], environment_key=env['cache_key'])
    if (cell/'inputs.npz').exists():
        prior = json.loads((cell/'inputs.json').read_text())
        if prior['config'] != config or prior['source_sha256'] != meta['source_sha256']:
            raise RuntimeError('Shared input mismatch: '+str(cell))
        return
    cats=env['catalogues']; sizes=np.array([len(x.edges) for x in cats],np.int32)
    arrays=dict(demand_bps=env['demand_bps'],harvest_w=env['harvest'],reserve_j=env['reserve'],
                times_s=env['times'],city_pairs=env['pairs'],plane_beta_deg=env['plane_beta_deg'],
                shape=env['shape'],source_lon=env['source_lon'],sources=np.array([x.sources for x in cats],np.int32),
                destinations=np.array([x.destinations for x in cats],np.int32),
                link_offsets=np.r_[np.int32(0),np.cumsum(sizes,dtype=np.int32)],
                directed_edges=np.concatenate([x.edges for x in cats]).astype(np.int32),
                capacity_bps=np.concatenate([x.link_capacity_bps for x in cats]),
                distance_km=np.concatenate([x.distance_km for x in cats]))
    arrays.update(pack_routes(cats))
    save_npz(cell/'inputs.npz',**arrays); write_json(cell/'inputs.json',meta)


def with_battery(env, config):
    reserve=reserve_envelope(env['harvest_extended'],config.base_w+4*config.sleep_w,env['dt'],config.battery)
    return dict(env,config=config,battery=config.battery,reserve=reserve[:len(env['times'])+1],reserve_extended=reserve,
                certified=reserve.max(axis=0)<=config.capacity_j+1e-6)


def eclipse_windows(harvest, discharge_j):
    rows=[]
    for m in range(harvest.shape[1]):
        dark=harvest[:,m]==0
        boundaries=np.diff(np.r_[False,dark,False].astype(np.int8))
        for lo,hi in zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)):
            if lo>0 and hi<len(dark):
                rows.append(dict(satellite=int(m),start_slot=int(lo),end_slot_exclusive=int(hi),
                                 battery_draw_j=float(discharge_j[lo:hi,m].sum())))
    return rows


def acquisition_energy_j(switch,dt,config):
    # Preserve the old arithmetic exactly when acquisition fits in one slot.
    if not config.acquisition_carry or dt>=config.acquisition_s:
        return float(switch.newly_usable.sum()*2*25.*min(dt,config.acquisition_s))
    return float(np.sum(switch.acquisition_active_s)*2*25.)


def run_policy(config, method, out, env=None, phase='validation',storage=None):
    out=Path(out).resolve()
    if ROOT/'results_r2' not in out.parents:
        raise ValueError('R2 results must remain under results_r2/')
    experiment=out.relative_to(ROOT/'results_r2').parts[0]
    seed_guard(config,phase,experiment=experiment,method=method); disk_guard()
    out.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes(); stamp=fingerprint()
    tier=storage_tier(config.seed,phase,storage,experiment=experiment)
    from .allocation import shadow_enabled
    do_shadow=shadow_enabled(experiment,phase,method,config.rho_energy)
    if (out/'summary.json').exists():
        prior=json.loads((out/'summary.json').read_text())
        if prior['config']==asdict(config) and prior['source_fingerprint']==stamp and prior.get('storage_tier')==tier:
            return prior
        raise RuntimeError('Completed run has different source/config: '+str(out))
    start=time.perf_counter()
    progress=lambda row: None if tier=='selection' else write_json(out/'status.json',dict(row,seed=config.seed,method=method))
    if env is None:
        env=build_environment(config,progress=progress)
    env=dict(env,config=config)
    shared_inputs(env,out.parent,tier)
    c=config;b=env['battery'];n=len(env['times']);m=env['shell'].size;dt=env['dt']
    controller=SwitchController(c,method)
    energy=np.full(m,c.initial_soc*b.capacity_j)
    states=np.empty((n+1,m));states[0]=energy
    raw=np.empty((n,m));terminal=np.empty_like(raw);fixed=np.empty_like(raw)
    spill=np.empty_like(raw);unserved=np.empty_like(raw);discharge=np.empty_like(raw)
    safe=np.zeros((n,m),bool);brownout=np.zeros_like(safe);forced=np.zeros_like(safe)
    switches=np.zeros((n,m,4),bool);available_ports=np.zeros_like(switches)
    allocation=np.zeros((n,c.flows,c.paths+1));budgets=np.empty_like(raw)
    mu=np.empty_like(raw);price_pi=np.empty_like(raw);pace=np.empty_like(raw);surplus=np.empty_like(raw)
    remaining=np.empty((n,m),np.int32);node_dual=np.zeros_like(raw)
    eclipse_thirds=np.empty((n,m),np.int8);spill_price_reset=np.zeros((n,m),bool)
    eclipse_sixths=np.empty((n,m),np.int8);pace_fraction=np.empty_like(raw)
    p_star=np.full((n,m),np.nan);p_full=np.full((n,m),np.nan)
    p_star_computed=np.zeros((n,m),bool);reset_fill=np.zeros((n,m),bool);reset_slack=np.zeros((n,m),bool)
    from .diagnostics_r22 import WindowForecasts
    forecasts=WindowForecasts(env) if experiment=='E2' else None
    actual_node_bits=np.zeros_like(raw) if forecasts is not None else None
    access_demand=np.zeros_like(raw);relay_bits=np.zeros_like(raw)
    ledger=[];executed_cats=[];link_rates=[];link_util=[];nominal_util=[]
    on_flags=[];capture_factors=[];new_flags=[];max_bidir=[]
    sleep_counts=np.zeros(m,np.int64);wake_counts=np.zeros(m,np.int64)
    cf_distributions={'sun':[],'eclipse':[]};sup_distributions={'sun':[],'eclipse':[]}
    cf_series=[];sup_series=[];wake_series=[];disconnected=np.zeros((n,c.flows),bool)
    eclipse_terminal_util=[];hop_bit_sum=0.;distance_bit_sum=0.;a2_counts=np.zeros((n,m),np.int32)
    run_manifest=dict(config=asdict(c),method=method,phase=phase,seed=c.seed,
               source_sha256=hashes,source_fingerprint=stamp,python=sys.version,platform=platform.platform(),
               start_unix=time.time(),pid=os.getpid(),inputs='../inputs.npz',
               units='J,W,s,bit/s; solver rate/1e9 and incremental power/100',raw_float_dtype='float32',
               metric_precision='float64 before storage; ESR tolerance 1e-6 J',storage_tier=tier)
    if tier!='selection':write_json(out/'manifest.json',run_manifest)
    for t in range(n):
        disk_guard();tick=time.perf_counter()
        switch=controller.decide(env,t,energy)
        cat=usable_catalogue(env['catalogues'][t],switch,c.routing)
        x,problem,meta,info=allocate(env,t,cat,switch,controller,shadow=do_shadow)
        x,q,violation=protect_bus(env,t,energy,cat,switch,x,problem,meta)
        rec=step(energy,env['harvest'][t],c.base_w,q,dt,b)
        controller.observe(cat,x,problem,switch,env['terminal'])
        controller.price.commit(q,t)
        trace=switch.price_trace
        count=len(cat.edges)//2
        current_cf=np.array([controller.counterfactual[tuple(map(int,e))] for e in cat.edges[:count]])
        current_sup=np.array([controller.support[tuple(map(int,e))] for e in cat.edges[:count]])
        cf_series.append(np.tile(current_cf,2));sup_series.append(np.tile(current_sup,2));wake_series.append(np.tile(switch.wake_value,2))
        side_dark=env['harvest'][t,cat.edges[:,0]]==0
        for name,mask in [('sun',~side_dark),('eclipse',side_dark)]:
            cf_distributions[name].append(np.tile(current_cf,2)[mask]);sup_distributions[name].append(np.tile(current_sup,2)[mask])
        a2_counts[t]=switch.a2_nodes
        disconnected[t]=(~cat.mask.any(axis=1)) & (env['demand_bps'][t]>0)
        hop_bit_sum+=float(np.sum(x*cat.hops)*RATE*dt)
        distance_bit_sum+=float(cat.distance_km@(cat.incidence@x.ravel())*RATE*dt)
        mu[t]=trace['mu'];price_pi[t]=trace['pi'];pace[t]=trace['pace_w'];surplus[t]=trace['z_j']
        remaining[t]=trace['remaining_slots'];eclipse_thirds[t]=trace['eclipse_third'];spill_price_reset[t]=trace['spill_reset']
        eclipse_sixths[t]=trace['eclipse_sixth'];pace_fraction[t]=trace['pace_weight_fraction']
        if controller.price.kind=='dual':
            p_star[t]=trace['p_star_w'];p_full[t]=trace['p_full_w'];p_star_computed[t]=trace['p_star_computed']
            reset_fill[t]=trace['reset_realistic_fill'];reset_slack[t]=trace['reset_sustainable_slack']
        if forecasts is not None:forecasts.observe_start(t,trace)
        if 'node_dual' in info:node_dual[t]=info['node_dual']
        # Source attachment demand; destination traffic is not counted twice.
        attached=cat.sources>=0
        np.add.at(access_demand[t],cat.sources[attached],env['demand_bps'][t,attached])
        for f,routes in enumerate(cat.paths):
            for j,route in enumerate(routes):
                if forecasts is not None and route and x[f,j]>0:
                    np.add.at(actual_node_bits[t],np.asarray(route),x[f,j]*RATE*dt)
                if route and len(route)>2 and x[f,j]>0:
                    np.add.at(relay_bits[t],np.asarray(route[1:-1]),x[f,j]*RATE*dt)
        energy=rec.after_j;states[t+1]=energy
        raw[t]=rec.raw_after_j;terminal[t]=q;fixed[t]=switch.fixed_w
        spill[t]=rec.spill_j;unserved[t]=rec.unserved_energy_j
        discharge[t]=np.maximum(-rec.delta_j,0.)
        allocation[t]=x*RATE;safe[t]=switch.infeasible if controller.hard else False
        brownout[t]=switch.brownout;forced[t]=switch.budget_forced
        switches[t]=switch.terminals_on;available_ports[t]=switch.available_terminals
        budgets[t]=switch.raw_budget_w
        sleep_counts+=switch.sleep_per_node;wake_counts+=switch.wake_per_node
        rate=cat.incidence@x.ravel()*RATE
        util=np.divide(rate,cat.link_capacity_bps,out=np.zeros_like(rate),where=cat.link_capacity_bps>0)
        nominal=rate/env['catalogues'][t].link_capacity_bps
        # Each installed terminal contributes one outgoing direction; absent neighbours have zero utilization.
        terminal_util=np.zeros((m,4))
        edges=cat.edges[:count];ports=switch.ports
        terminal_util[edges[:,0],ports[:,0]]=util[:count]
        terminal_util[edges[:,1],ports[:,1]]=util[count:]
        eclipse_terminal_util.append(terminal_util[env['harvest'][t]==0].ravel())
        directed_on=np.tile(switch.on,2)
        link_rates.append(rate);link_util.append(util[directed_on]);nominal_util.append(nominal[directed_on])
        max_bidir.append(np.maximum(util[:len(switch.on)],util[len(switch.on):])[switch.on])
        on_flags.append(directed_on);capture_factors.append(np.tile(switch.capture,2));new_flags.append(np.tile(switch.newly_usable,2))
        executed_cats.append(cat)
        residual=np.max(np.abs(rec.after_j-(rec.before_j+rec.delta_j-rec.spill_j+rec.unserved_energy_j)))
        identity=np.max(np.abs(q-(switch.fixed_w+cat.sender@env['terminal'].slot_increment(rate,cat.distance_km,meta.get('capture')))))
        row=dict(slot=t,time_s=float(env['times'][t]),iterations=int(info['iterations']),
                 solver_s=float(info['runtime_s']),slot_s=time.perf_counter()-tick,
                 relative_gap=float(info['gap']) if info.get('gap') is not None else None,
                 objective=float(objective(problem,x,cat,env['terminal'],meta)),solver_status=info.get('status'),
                 constraint_violation=float(violation),ledger_residual_j=float(residual),power_residual_w=float(identity),
                 min_reserve_margin_j=float((energy-env['reserve'][t+1]).min()),
                 unsafe_nodes=int(np.sum(energy<b.minimum_j-1e-6)),
                 mean_on_utilization=float(util[directed_on].mean()) if directed_on.any() else 0.,
                 on_directed_links=int(directed_on.sum()),delivered_bps=float(x.sum()*RATE),
                 terminal_energy_j=float(q.sum()*dt),variable_energy_j=float((q-switch.fixed_w).sum()*dt),
                 acquisition_energy_j=acquisition_energy_j(switch,dt,c),
                 sleep_terminal_events=switch.sleep_events,wake_terminal_events=switch.wake_events,
                 a0_sleep_terminal_events=switch.a0_sleep,price_sleep_link_events=switch.price_sleep,
                 zero_price_a2_events=switch.zero_price_a2,max_a2_per_node=int(np.max(switch.a2_nodes,initial=0)),
                 disconnected_commodity_slots=int(disconnected[t].sum()),
                 disconnected_demand_bps=float(env['demand_bps'][t,disconnected[t]].sum()),
                 delivered_weighted_hops=float(np.sum(x*cat.hops)/x.sum()) if x.sum() else 0.,
                 delivered_propagation_delay_ms=float(cat.distance_km@(cat.incidence@x.ravel())/x.sum()/299792.458*1000) if x.sum() else 0.,
                 mean_counterfactual_value=float(current_cf.mean()) if len(current_cf) else 0.,
                 rescue_wake_link_events=switch.rescue_wakes,budget_forced_nodes=int(switch.budget_forced.sum()),
                 intent_bits=4*m,logical_float64_scalars=int(2*problem.resource_matrix.nnz*info['iterations']))
        row.update({k:v for k,v in info.items() if k.startswith(('shadow_','central_')) or k=='problem_fingerprint'})
        if controller.price.kind=='dual':
            row.update(price_reset_a_nodes=int(reset_fill[t].sum()),price_reset_b_nodes=int(reset_slack[t].sum()),
                       price_p_star_evaluated_nodes=int(p_star_computed[t].sum()))
        ledger.append(row)
        if t%8==0 or t+1==n:
            progress(dict(phase='simulation',slot=t+1,slots=n,elapsed_s=time.perf_counter()-start))
    delivered=float(allocation.sum()*dt);offered=float(env['demand_bps'].sum()*dt)
    bus=np.maximum((c.base_w+terminal)*dt-b.discharge_efficiency*unserved,0.)
    base=np.minimum(c.base_w*dt,bus);isl=float((bus-base).sum());system=float(bus.sum())
    unsafe=states[1:]<b.minimum_j-1e-6;eligible=states[0]>=env['reserve'][0]-1e-6
    util=np.concatenate(link_util);nom=np.concatenate(nominal_util);undirected=np.concatenate(max_bidir)
    windows=eclipse_windows(env['harvest'],discharge);draws=np.array([x['battery_draw_j'] for x in windows])
    end=np.minimum(np.rint(np.arange(1,c.orbits+1)*env['shell'].period_s/dt).astype(int),n);starts=np.r_[0,end[:-1]]
    holes=(~switches.any(axis=2)) & (access_demand>0)
    dark=env['harvest']==0
    valid_gaps=[x['relative_gap'] for x in ledger if x['relative_gap'] is not None]
    shadow_gaps=[x['shadow_certificate_gap'] for x in ledger if x.get('shadow_certificate_gap') is not None]
    shadow_statuses=[x['shadow_solver_status'] for x in ledger if x.get('shadow_solver_status') is not None]
    shadow_objective_gaps=[x['shadow_relative_gap'] for x in ledger if x.get('shadow_relative_gap') is not None]
    e_util=np.concatenate(eclipse_terminal_util)
    def distribution(parts):
        a=np.concatenate(parts)
        return dict(count=len(a),mean=float(a.mean()) if len(a) else 0.,quantiles=np.percentile(a,[0,25,50,75,90,95,99,100]).tolist() if len(a) else [0.]*8)
    summary=dict(complete=True,config=asdict(c),method=method,phase=phase,seed=c.seed,source_fingerprint=stamp,
        delivered_bits=delivered,offered_bits=offered,fvr=1-delivered/offered if offered else 0.,
        terminal_energy_j=isl,system_energy_j=system,variable_energy_j=float(((terminal-fixed)*dt).sum()),
        variable_energy_fraction=float(((terminal-fixed)*dt).sum()/isl) if isl else 0.,
        ee_isl_bits_j=delivered/isl if isl else 0.,ee_sys_bits_j=delivered/system if system else 0.,
        esr=1-float(unsafe.mean()),esr_denominator=int(unsafe.size),unsafe_node_slots=int(unsafe.sum()),
        initially_certifiable_nodes=int(eligible.sum()),certifiable_esr=1-float(unsafe[:,eligible].mean()) if eligible.any() else None,
        min_soc=float(states.min()/b.capacity_j),violation_depth_j=float(np.maximum(b.minimum_j-states,0).max()),
        violation_duration_node_s=float(unsafe.sum()*dt),safe_mode_node_slots=int(safe.sum()),
        budget_forced_node_slots=int(forced.sum()),brownout_node_slots=int(brownout.sum()),
        brownout_events=int(np.sum(brownout&~np.vstack([np.zeros((1,m),bool),brownout[:-1]]))),
        terminal_on_fraction=float(switches.sum()/available_ports.sum()) if available_ports.any() else 0.,
        sleep_terminal_events=int(sleep_counts.sum()),wake_terminal_events=int(wake_counts.sum()),
        flips_per_satellite_per_orbit=float((sleep_counts+wake_counts).sum()/m/c.orbits),
        price_sleep_link_events=sum(x['price_sleep_link_events'] for x in ledger),
        acquisition_energy_j=sum(x['acquisition_energy_j'] for x in ledger),
        on_link_utilization_mean=float(util.mean()) if len(util) else 0.,
        on_link_utilization_p95=float(np.percentile(util,95)) if len(util) else 0.,
        on_link_fraction_above_1pct=float(np.mean(util>.01)) if len(util) else 0.,
        on_link_observations=int(len(util)),on_link_nominal_utilization_mean=float(nom.mean()) if len(nom) else 0.,
        on_link_bidirectional_max_mean=float(undirected.mean()) if len(undirected) else 0.,
        max_link_utilization=float(util.max(initial=0)),hottest_satellite_w=float(terminal.max()),
        eclipse_window_count=int(len(draws)),eclipse_draw_mean_j=float(draws.mean()) if len(draws) else None,
        eclipse_draw_p90_j=float(np.percentile(draws,90)) if len(draws) else None,
        orbit_max_dod=[float(1-states[lo:hi+1].min()/b.capacity_j) for lo,hi in zip(starts,end)],
        orbit_end_min_soc=(states[end].min(axis=1)/b.capacity_j).tolist(),orbit_end_mean_soc=(states[end].mean(axis=1)/b.capacity_j).tolist(),
        max_ledger_residual_j=max(x['ledger_residual_j'] for x in ledger),max_power_residual_w=max(x['power_residual_w'] for x in ledger),
        max_constraint_violation=max(x['constraint_violation'] for x in ledger),
        mean_iterations=float(np.mean([x['iterations'] for x in ledger])),mean_solver_s=float(np.mean([x['solver_s'] for x in ledger])),
        mean_slot_s=float(np.mean([x['slot_s'] for x in ledger])),
        mean_relative_gap=float(np.mean(valid_gaps)) if valid_gaps else None,
        mean_shadow_relative_gap=float(np.mean(shadow_objective_gaps)) if shadow_objective_gaps else None,
        shadow_evaluation_enabled=do_shadow,shadow_success_slots=len(shadow_objective_gaps),
        shadow_failure_slots=sum(x.get('shadow_failed',False) for x in ledger),
        central_degraded_slots=sum(x.get('central_degraded',False) for x in ledger),
        mean_shadow_certificate_gap=float(np.mean(shadow_gaps)) if shadow_gaps else None,
        max_shadow_certificate_gap=max(shadow_gaps,default=None),
        max_shadow_numerical_attempts=max((row.get('shadow_numerical_attempts',0) for row in ledger),default=0),
        shadow_reference_target_miss_slots=sum(row.get('shadow_reference_target_met') is False for row in ledger),
        shadow_solver_status_counts={status:shadow_statuses.count(status) for status in sorted(set(shadow_statuses))},
        environment_build_s=env['build_s'],runtime_s=time.perf_counter()-start,slots=n,dt_s=dt,reference_scale_bps=env['reference_scale_bps'],
        hole_node_slots=int(holes.sum()),hole_access_demand_fraction=float(access_demand[holes].sum()/access_demand.sum()) if access_demand.sum() else 0.,
        eclipse_mean_on_terminals_by_third=[float(switches.sum(axis=2)[eclipse_thirds==j].mean()) if np.any(eclipse_thirds==j) else None for j in range(3)],
        eclipse_mean_on_terminals_by_sixth=[float(switches.sum(axis=2)[eclipse_sixths==j].mean()) if np.any(eclipse_sixths==j) else None for j in range(6)],
        relay_bit_hops_sun=float(relay_bits[~dark].sum()),relay_bit_hops_eclipse=float(relay_bits[dark].sum()),
        relay_sun_fraction=float(relay_bits[~dark].sum()/relay_bits.sum()) if relay_bits.sum() else None,
        spill_stored_j=float(spill.sum()),spill_harvest_equivalent_j=float(spill.sum()/b.charge_efficiency),
        spill_harvest_fraction=float(spill.sum()/b.charge_efficiency/(env['harvest'].sum()*dt)) if env['harvest'].sum() else 0.,
        certificate_dual_positive_fraction=float(np.mean(node_dual>0)) if controller.hard else 0.,
        certificate_dual_above_1e_minus9_fraction=float(np.mean(node_dual>1e-9)) if controller.hard else 0.,
        delivered_offered_ratio=delivered/offered if offered else 1.,
        reference_r1_bps=env['reference_r1_bps'],reference_new_over_r1=env['reference_scale_bps']/env['reference_r1_bps'],
        delivered_weighted_hops=hop_bit_sum/delivered if delivered else 0.,
        delivered_propagation_delay_ms=distance_bit_sum/delivered/299792.458*1000 if delivered else 0.,
        disconnected_commodity_slots=int(disconnected.sum()),disconnected_demand_fraction=float(env['demand_bps'][disconnected].sum()*dt/offered) if offered else 0.,
        zero_price_a2_events=sum(row['zero_price_a2_events'] for row in ledger),max_a2_per_node=int(a2_counts.max()),
        counterfactual_distribution={key:distribution(value) for key,value in cf_distributions.items()},
        support_distribution={key:distribution(value) for key,value in sup_distributions.items()},
        eclipse_terminal_observations=len(e_util),eclipse_terminal_idle_fraction=float(np.mean(e_util<.001)) if len(e_util) else None,
        eclipse_terminal_above_1pct=float(np.mean(e_util>.01)) if len(e_util) else None,
        eclipse_terminal_above_10pct=float(np.mean(e_util>.1)) if len(e_util) else None,
        storage_tier=tier,manifest=run_manifest)
    sun=~dark
    on_count=switches.sum(axis=2);available_count=available_ports.sum(axis=2)
    def sun_on(lo,hi):
        mask=sun[lo:hi];den=available_count[lo:hi][mask].sum()
        return float(on_count[lo:hi][mask].sum()/den) if den else None
    summary.update(sun_terminal_on_fraction=sun_on(0,n),
        orbit_sun_terminal_on_fraction=[sun_on(lo,hi) for lo,hi in zip(starts,end)],
        orbit_spill_harvest_fraction=[float(spill[lo:hi].sum()/b.charge_efficiency/(env['harvest'][lo:hi].sum()*dt))
            if env['harvest'][lo:hi].sum() else 0. for lo,hi in zip(starts,end)],
        certificate_dual_reporting_threshold=1e-6 if method=='HBAG-central' else 0.,
        certificate_dual_reported_fraction=float(np.mean(node_dual>(1e-6 if method=='HBAG-central' else 0.))) if controller.hard else 0.)
    if method=='HBAG-central':
        summary.pop('certificate_dual_positive_fraction')
        summary.pop('certificate_dual_above_1e_minus9_fraction')
    if controller.price.kind=='dual':
        summary['price_reset_by_illumination']={name:dict(node_slots=int(mask.sum()),
            a_count=int(reset_fill[mask].sum()),b_count=int(reset_slack[mask].sum()),
            a_fraction=float(reset_fill[mask].mean()) if mask.any() else None,
            b_fraction=float(reset_slack[mask].mean()) if mask.any() else None,
            either_fraction=float((reset_fill[mask]|reset_slack[mask]).mean()) if mask.any() else None,
            b_evaluated_node_slots=int(p_star_computed[mask].sum())) for name,mask in [('sun',sun),('eclipse',dark)]}
    arrays=dict(energy_j=states,raw_after_j=raw,terminal_w=terminal,fixed_terminal_w=fixed,spill_j=spill,
        unserved_j=unserved,battery_discharge_j=discharge,allocation_bps=allocation,delivered_bps=allocation.sum(axis=2),
        safe_mode=safe,brownout=brownout,budget_forced=forced,terminals_on=switches,geometric_terminals=available_ports,
        raw_budget_w=budgets,link_rate_bps=np.concatenate(link_rates),link_on=np.concatenate(on_flags),
        link_capture=np.concatenate(capture_factors),link_newly_usable=np.concatenate(new_flags),
        sleep_events_per_node=sleep_counts,wake_events_per_node=wake_counts,
        mu=mu,price_pi=price_pi,pace_w=pace,discretionary_energy_j=surplus,remaining_window_slots=remaining,
        node_dual=node_dual,eclipse_third=eclipse_thirds,eclipse_sixth=eclipse_sixths,
        pace_weight_fraction=pace_fraction,spill_price_reset=spill_price_reset,
        access_demand_bps=access_demand,hole_node_slot=holes,relay_bit_hops=relay_bits,
        disconnected_commodity=disconnected,a2_count_per_node=a2_counts,
        counterfactual_value=np.concatenate(cf_series),support_value=np.concatenate(sup_series),wake_value=np.concatenate(wake_series))
    if controller.price.kind=='dual':
        arrays.update(p_star_w=p_star,p_full_w=p_full,p_star_computed=p_star_computed,
                      price_reset_a=reset_fill,price_reset_b=reset_slack)
    if forecasts is not None:
        correlations,diagnostics=forecasts.finish(actual_node_bits);summary.update(diagnostics)
        arrays.update(predicted_window_load_bps=forecasts.predicted,actual_relay_access_bits=actual_node_bits)
        write_json(out/'potential_load_windows.json',correlations)
    if tier=='full':
        arrays['executed_path_ids']=register_routes(out.parent,executed_cats)
    elif tier=='node':
        for key in ['allocation_bps','link_rate_bps','link_on','link_capture','link_newly_usable','counterfactual_value','support_value','wake_value']:
            arrays.pop(key)
    if tier!='selection':
        save_npz(out/'raw.npz',**arrays);write_json(out/'eclipse_windows.json',windows)
    write_json(out/'slots.json',ledger);write_json(out/'summary.json',summary)
    progress(dict(phase='complete',slots=n,elapsed_s=time.perf_counter()-start))
    return summary
