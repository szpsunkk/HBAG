"""Protocol v2.2: causal load pacing, robust values, path wakes and solver isolation."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from hbag3.test_gate import tiny
from hbag3.test_v21 import all_on_two_routes
from hbag3.model import Config,RATE
from hbag3.pricing import OrbitalPrice
from hbag3.forecast_r22 import potential_incidence,predicted_loads,pacing_fraction
from hbag3.link_values import recent_peak,counterfactual_values,path_wake_values
from hbag3.control import SwitchController,usable_catalogue
from hbag3.allocation import allocate,shadow_enabled


def forecast_fixture():
    c=Config(planes=1,per_plane=4,flows=1,paths=1,capacity_j=1e6)
    env=tiny(c,slots=2);first=replace(env['catalogues'][0],paths=[[(0,1,2)]])
    second=replace(first,paths=[[(0,3,2)]])
    env['catalogues']=[first,second]
    env['harvest_extended']=np.r_[np.zeros((2,4)),np.full((1,4),500.)]
    return env


def test_potential_load_counts_all_endpoints_nonempty_paths_and_nodes_once():
    cat=SimpleNamespace(paths=[[(0,1,2),(0,3,2),None],[(1,2),None,None]])
    a=potential_incidence(cat,4)
    np.testing.assert_allclose(a.toarray(),[[1,0],[.5,1],[1,1],[.5,0]])
    np.testing.assert_allclose(a@np.array([2.,3.]),[2,4,5,1])


@pytest.mark.parametrize('method',['HBAG','V2','HBAG-central','HBAG+margin'])
def test_shaped_eclipse_pace_is_causal_and_shared_by_dual_methods(method):
    env=forecast_fixture();energy=np.full(4,700600.)
    trace=OrbitalPrice(env['config'],method).prepare(env,0,energy)[2]
    np.testing.assert_allclose(trace['pace_weight_fraction'],[.5,1.05/1.1,.5,.05/1.1])
    np.testing.assert_allclose(trace['pace_w'],8+.95*600/60*trace['pace_weight_fraction'])
    env['demand_bps'][1]=1e99
    again=OrbitalPrice(env['config'],method).prepare(env,0,energy)[2]
    np.testing.assert_array_equal(trace['pace_w'],again['pace_w'])


def test_forecast_horizon_repeats_last_geometry_and_uniform_fallback():
    env=forecast_fixture();loads=predicted_loads(env,0,4)
    np.testing.assert_array_equal(loads[1],loads[2]);np.testing.assert_array_equal(loads[2],loads[3])
    n=np.array([2,2,3,3]);dark=np.ones(4,bool)
    np.testing.assert_array_equal(pacing_fraction(np.zeros((3,4)),n,dark),1./n)
    nonfinite=np.full((3,4),np.inf)
    np.testing.assert_array_equal(pacing_fraction(nonfinite,n,dark),1./n)


def test_sunlight_fraction_stays_uniform_and_pace_uses_v23_reserve_power():
    env=forecast_fixture();env['harvest'][0]=500.;env['harvest_extended'][0]=500.
    trace=OrbitalPrice(env['config'],'HBAG').prepare(env,0,np.full(4,700600.))[2]
    np.testing.assert_allclose(trace['pace_weight_fraction'],1./trace['remaining_slots'])
    np.testing.assert_allclose(trace['pace_w'],8+trace['p_star_w'])


def test_recent_peak_uses_physical_orbital_time_and_no_artificial_cap():
    peak,phi=recent_peak(np.array([10.,0.]),None,60.,600.)
    np.testing.assert_array_equal(phi,[1.,1.])
    peak,phi=recent_peak(np.array([1.,0.]),peak,60.,600.)
    assert peak[0]==pytest.approx(10*np.exp(-.1)) and phi[1]==1
    slow=10.
    for _ in range(4):slow=recent_peak(np.array([0.]),np.array([slow]),15.,600.)[0][0]
    assert slow==pytest.approx(peak[0])
    _,phi=recent_peak(np.array([1e-30]),np.array([10.]),60.,600.)
    assert phi[0]>1e20


def test_robust_counterfactual_changes_rerouting_amount_only():
    env,ctl,sw,cat,p=all_on_two_routes();x=np.array([[1.,0.,0.]])
    cf,sup=counterfactual_values(cat,x,np.ones(4),env['terminal'],p)
    robust,same_sup=counterfactual_values(cat,x,np.ones(4),env['terminal'],p,np.array([10.]))
    np.testing.assert_array_equal(sup,same_sup);assert np.max(cf)<1e-12
    assert np.max(robust)>1. and np.all(robust>=cf)


def wake_fixture(paths,capacities):
    edges=np.array([[0,1],[1,2],[0,3],[2,3]])
    return SimpleNamespace(paths=paths,edges=np.r_[edges,edges[:,::-1]],link_capacity_bps=np.tile(capacities,2)*RATE)


def test_path_wake_crosses_multiple_sleeping_edges_and_excludes_unwakeable():
    cat=wake_fixture([[(0,1,2)]],np.array([4.,4.,4.,4.]))
    on=np.array([False,False,True,True]);wakeable=~on
    v=path_wake_values(cat,on,.5,np.array([3.]),np.array([0.]),1.,wakeable)
    np.testing.assert_allclose(v,[1.,1.,0.,0.])
    wakeable[1]=False
    np.testing.assert_array_equal(path_wake_values(cat,on,.5,np.array([3.]),np.array([0.]),1.,wakeable),0.)


def test_path_wake_reserves_directional_capacity_and_uses_marginal_priority():
    cat=wake_fixture([[(0,1,2)],[(0,1,2)],[(2,1,0)]],np.array([4.,4.,4.,4.]))
    on=np.array([False,False,True,True]);z=np.array([1.,0.,0.])
    v=path_wake_values(cat,on,.5,np.array([4.,1.,2.]),z,1.,~on)
    # Forward: high marginal flow gets 1 bit, then marginal .5 gets 1 bit;
    # reverse has an independent 2 bits. Share total value equally on two edges.
    np.testing.assert_allclose(v,[1.75,1.75,0.,0.])


def test_path_wake_prefers_fewer_sleepers_then_larger_available_capacity():
    cat=wake_fixture([[(0,1,2),(0,3,2)]],np.array([1.,8.,8.,8.]))
    on=np.array([False,True,False,True])
    v=path_wake_values(cat,on,1.,np.array([3.]),np.array([0.]),1.,~on)
    np.testing.assert_allclose(v,[0.,0.,3.,0.])
    on[3]=False
    v=path_wake_values(cat,on,1.,np.array([3.]),np.array([0.]),1.,~on)
    np.testing.assert_allclose(v,[1.,0.,0.,0.])


@pytest.mark.parametrize('experiment,phase,method,rho,expected',[
    ('E2','test','HBAG',1.4,True),('E2','test','HBAG',1.2,False),
    ('E2','test','HBAG',1.6,False),('E2','test','V1',1.4,False),
    ('E2','test','HBAG-central',1.4,False),('E11_validation','validation','HBAG',1.4,False),
    ('G0_prime','validation','HBAG',1.4,False),('E4','test','HBAG',1.4,False),
    ('E8','test','HBAG+margin',1.4,False),('MPC','test','HBAG',1.4,False)])
def test_shadow_scope(experiment,phase,method,rho,expected):
    assert shadow_enabled(experiment,phase,method,rho) is expected


def test_shadow_failure_leaves_executed_policy_unchanged(monkeypatch):
    import hbag3.allocation as module
    c=Config(planes=1,per_plane=4,flows=1,paths=1,rho_energy=1.4,capacity_j=1e6)
    env=tiny(c);ctl=SwitchController(c,'HBAG');sw=ctl.decide(env,0,np.full(4,950000.))
    cat=usable_catalogue(env['catalogues'][0],sw)
    expected=allocate(env,0,cat,sw,ctl)[0];events=[]
    def failure(*args,**kwargs):raise RuntimeError('injected Clarabel failure')
    monkeypatch.setattr(module,'solve_centralized',failure)
    monkeypatch.setattr(module,'reference_event',lambda *args:events.append(args) or 'fixture')
    actual=allocate(env,0,cat,sw,ctl,shadow=True)
    np.testing.assert_array_equal(actual[0],expected)
    assert actual[3]['shadow_failed'] and actual[3]['shadow_relative_gap'] is None and len(events)==1


def test_central_all_failures_without_iterate_execute_explicit_feasible_zero(monkeypatch):
    import cvxpy as cp
    import hbag3.central_reference_r2 as reference
    from hbag3.test_reference_r2 import problem
    def failure(*args,**kwargs):raise cp.error.SolverError('injected failure without finite iterate')
    monkeypatch.setattr(cp.Problem,'solve',failure)
    monkeypatch.setattr(reference,'retain_failure',lambda *args:None)
    p=problem(.4);answer=reference.solve(p,allow_degraded=True)
    assert answer['degraded'] and answer['status']=='no_finite_iterate_zero_action'
    assert p.violations(answer['allocation'])==0 and not answer['allocation'].any()
    assert len(answer['numerical_attempts'])==4
    with pytest.raises(RuntimeError):reference.solve(p)


def test_central_failed_attempt_uses_smallest_recovered_certificate(monkeypatch):
    import cvxpy as cp
    import hbag3.central_reference_r2 as reference
    from hbag3.optimization import feasible_recovery
    from hbag3.test_reference_r2 import problem
    calls=[]
    def failure(*args,**kwargs):
        calls.append(kwargs)
        raise cp.error.SolverError('injected numerical failure with finite iterate')
    values=[.05,.6,2.,.1]
    def last_iterate(model,chain,inverse,variable,constraints):
        return (np.full(variable.shape,values[len(calls)-1]),
                [np.zeros(c.shape) for c in constraints],'InsufficientProgress',len(calls))
    monkeypatch.setattr(cp.Problem,'solve',failure)
    monkeypatch.setattr(reference,'failed_iterate',last_iterate)
    monkeypatch.setattr(reference,'retain_failure',lambda *args:None)
    p=problem(.4);answer=reference.solve(p,allow_degraded=True)
    attempts=answer['numerical_attempts'];gaps=[a['last_iterate_gap'] for a in attempts]
    chosen=int(np.argmin(gaps))
    assert len(set(gaps))>1 and answer['degraded']
    assert answer['certificate_gap']==min(gaps)
    assert answer['numerical_settings']==attempts[chosen]['settings']
    assert answer['iterations']==chosen+1
    assert p.violations(feasible_recovery(answer['allocation'],p))<1e-7


def test_clarabel_last_iterate_maps_back_after_rejected_status(monkeypatch):
    import cvxpy as cp
    from types import SimpleNamespace
    import hbag3.central_reference_r2 as reference
    from hbag3.test_reference_r2 import problem
    original=cp.Problem.solve
    def rejected(model,**settings):
        original(model,**settings)
        raw=model._solver_cache['CLARABEL']
        model._solver_cache['CLARABEL']=SimpleNamespace(status='NumericalError',x=raw.x,z=raw.z,
            obj_val=raw.obj_val,solve_time=raw.solve_time,iterations=raw.iterations)
        raise cp.error.SolverError('injected rejected status')
    monkeypatch.setattr(cp.Problem,'solve',rejected)
    monkeypatch.setattr(reference,'retain_failure',lambda *args:None)
    p=problem(.4);answer=reference.solve(p,allow_degraded=True)
    assert answer['degraded'] and answer['status']=='NumericalError'
    assert answer['certificate_gap']<1e-8 and np.any(answer['allocation']>0)
    assert len(answer['numerical_attempts'])==4


def test_shadow_io_failure_is_not_misreported_as_solver_failure(monkeypatch):
    import hbag3.allocation as module
    c=Config(planes=1,per_plane=4,flows=1,paths=1,rho_energy=1.4,capacity_j=1e6)
    env=tiny(c);ctl=SwitchController(c,'HBAG');sw=ctl.decide(env,0,np.full(4,950000.))
    cat=usable_catalogue(env['catalogues'][0],sw)
    def failure(*args,**kwargs):raise OSError('injected failed evidence write')
    monkeypatch.setattr(module,'solve_centralized',failure)
    with pytest.raises(OSError,match='evidence write'):allocate(env,0,cat,sw,ctl,shadow=True)


def test_reference_event_keeps_all_failed_attempt_evidence(monkeypatch,tmp_path):
    import json
    import hbag3.environment as environment
    from hbag3.allocation import reference_event
    from hbag3.test_reference_r2 import problem
    monkeypatch.setattr(environment,'ROOT',tmp_path)
    p=problem(.4);failure=tmp_path/'results_r2/audit/reference_failures'/p.fingerprint()/'failure.json'
    failure.parent.mkdir(parents=True);failure.write_text(json.dumps({'attempts':[1,2,3,4]}))
    before=failure.read_bytes()
    sw=SimpleNamespace(on=np.ones(1,bool),capture=np.ones(1),weight=np.ones(2),raw_budget_w=np.ones(2),price_trace={})
    reference_event({'config':Config()},0,p,sw,'shadow',{'error':'all attempts failed'})
    assert failure.read_bytes()==before


def test_test_seed_lock_requires_both_current_v22_records(monkeypatch,tmp_path):
    import json
    import hbag3.runner as runner
    VERSION='v2.3' # Exercise the retained legacy gate without enabling old reruns.
    monkeypatch.setattr(runner,'VERSION',VERSION)
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'fingerprint',lambda:'current')
    config=Config(seed=4000)
    paths=[tmp_path/'results_r2/G0_prime/gate.json',tmp_path/'results_r2/E11_validation/frozen.json']
    current=dict(state='PASS',plan_version=VERSION,source_fingerprint='current')
    for path in paths:
        with pytest.raises(RuntimeError):runner.seed_guard(config,'test')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(current))
    runner.seed_guard(config,'test')
    for bad in [dict(plan_version='v2.1'),dict(source_fingerprint='old'),dict(state='STOP')]:
        paths[1].write_text(json.dumps(dict(current,**bad)))
        with pytest.raises(RuntimeError):runner.seed_guard(config,'test')
    paths[1].write_text(json.dumps(current))
    with pytest.raises(ValueError):runner.seed_guard(Config(seed=1000),'test')


def test_eclipse_sixths_and_window_correlation():
    from hbag3.diagnostics_r22 import WindowForecasts
    env=forecast_fixture();env['dt']=1.;d=WindowForecasts(env)
    trace=OrbitalPrice(env['config'],'HBAG').prepare(env,0,np.full(4,700600.))[2]
    d.observe_start(0,trace)
    records,summary=d.finish(d.predicted.copy())
    valid=[r for r in records if r['correlation'] is not None]
    assert len(valid)==2 and all(r['correlation']==pytest.approx(1.) for r in valid)
    assert summary['potential_load_correlation_valid_windows']==2
    env=tiny(slots=6);price=OrbitalPrice(env['config'],'V0')
    assert [int(price.prepare(env,t,np.full(4,950000.))[2]['eclipse_sixth'][0]) for t in range(6)]==list(range(6))
