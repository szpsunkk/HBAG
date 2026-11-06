"""v2.3 reserve pacing, gate isolation, offline execution and timing boundaries."""
from dataclasses import replace,asdict
import json
import numpy as np
import pytest
from hbag3.test_gate import tiny
from hbag3.model import Config
from hbag3.energy import reserve_envelope,step
from hbag3.pricing import OrbitalPrice
from hbag3.reserve_pricing_r23 import eclipse_power,sustainable_sun_power


def physical_fixture(harvest,capacity=1e6):
    h=np.asarray(harvest,float)
    if h.ndim==1:h=np.repeat(h[:,None],4,axis=1)
    env=tiny(Config(planes=1,per_plane=4,flows=1,paths=1,capacity_j=capacity),slots=len(h))
    env['harvest']=h;env['harvest_extended']=h
    env['reserve']=reserve_envelope(h,63.,env['dt'],env['battery'])
    env['reserve_extended']=env['reserve'];env['times']=np.arange(len(h))*env['dt']
    return env


def test_eclipse_closed_form_matches_independent_backward_bisection():
    env=physical_fixture([0.]*30+[500.]);b=env['battery'];n=np.array([5,10,20,30])
    # Each independently chosen end retains its baseline reserve at that end.
    z=np.array([0.,10000.,80000.,1e6]);energy=env['reserve'][0]+z
    energy=np.minimum(energy,b.capacity_j)
    closed=eclipse_power(energy-env['reserve'][0],n,60.,b.discharge_efficiency,232.)
    binary=sustainable_sun_power(env,0,energy,n,np.arange(4))
    np.testing.assert_allclose(closed,binary,rtol=0,atol=232./2**14+1e-7)


def test_sun_certificate_checks_intermediate_capacity_and_infeasible_zero():
    env=physical_fixture([500.]*2+[0.]*40+[500.],capacity=100000.)
    answer=sustainable_sun_power(env,0,np.full(4,100000.),np.full(4,42),np.arange(4))
    np.testing.assert_array_equal(answer,0.)
    env=physical_fixture([500.]*4)
    np.testing.assert_array_equal(sustainable_sun_power(env,0,np.full(4,1e6),np.full(4,4),np.arange(4)),232.)


def test_condition_a_exact_boundary_and_no_sun_bisection_when_satisfied(monkeypatch):
    import hbag3.reserve_pricing_r23 as module
    env=physical_fixture([275.,0.,500.]);threshold=1e6-60*.95*(275.-55.-8.-100.)
    seen=[]
    def lower(env,t,energy,end,nodes):
        seen.extend(nodes.tolist());return np.full(len(nodes),99.)
    monkeypatch.setattr(module,'sustainable_sun_power',lower)
    energy=np.array([threshold,threshold+1e-4,threshold-1e-4,threshold-100.])
    price=OrbitalPrice(env['config'],'HBAG');price.mu[:]=2
    trace=price.prepare(env,0,energy)[2]
    np.testing.assert_array_equal(trace['reset_realistic_fill'],[True,True,False,False])
    assert seen==[2,3] and np.isnan(trace['p_star_w'][:2]).all()
    price.commit(np.full(4,230.),0)
    np.testing.assert_array_equal(price.mu[:2],0.)
    assert (price.mu[2:]>2).all()


def test_condition_b_eclipse_boundary_and_no_binary_solver(monkeypatch):
    import hbag3.reserve_pricing_r23 as module
    env=tiny(slots=1);env['dt']=.95
    def no_binary(env,t,energy,end,nodes):
        assert len(nodes)==0
        return np.empty(0)
    monkeypatch.setattr(module,'sustainable_sun_power',no_binary)
    price=OrbitalPrice(env['config'],'HBAG')
    trace=price.prepare(env,0,np.array([700100.,700100.0001,700099.9999,700000.]))[2]
    np.testing.assert_array_equal(trace['reset_sustainable_slack'],[True,True,False,False])
    price.commit(np.full(4,200.),0)
    np.testing.assert_array_equal(price.mu[:2],0.)
    assert (price.mu[2:]>0).all()


def test_full_power_ema_only_updates_from_all_four_on_load():
    env=physical_fixture([0.,500.]);price=OrbitalPrice(env['config'],'HBAG')
    price.prepare(env,0,np.full(4,710000.))
    price.commit(np.array([100.,99.999,120.,110.]),0)
    np.testing.assert_allclose(price.var_ema,[6.4,8.,10.4,8.4])


def test_sun_pace_is_reserve_power_and_not_current_surplus_only():
    env=physical_fixture([275.]*4+[0.]*8+[500.]);energy=env['reserve'][0]+10000.
    trace=OrbitalPrice(env['config'],'HBAG').prepare(env,0,energy)[2]
    assert not trace['reset_realistic_fill'].any()
    np.testing.assert_array_equal(trace['pace_w'],8.+trace['p_star_w'])
    assert np.any(trace['pace_w']>8.+.95*10000/(trace['remaining_slots']*60))


def test_new_test_seed_range_needs_both_current_gates(monkeypatch,tmp_path):
    import hbag3.runner as runner
    VERSION='v2.3' # Historical gate behavior; v2.3.1 keeps approved runs read-only.
    monkeypatch.setattr(runner,'VERSION',VERSION)
    from hbag3.storage import storage_tier
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'fingerprint',lambda:'current')
    gate=tmp_path/'results_r2/G0_prime/gate.json';frozen=tmp_path/'results_r2/E11_validation/frozen.json'
    for path in [gate,frozen]:
        with pytest.raises(RuntimeError):runner.seed_guard(Config(seed=4000),'test')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(state='PASS',source_fingerprint='current',plan_version=VERSION)))
    for seed in [4000,4019]:runner.seed_guard(Config(seed=seed),'test')
    for seed in [1000,3000,3019,3999,4020]:
        with pytest.raises(ValueError):runner.seed_guard(Config(seed=seed),'test')
    assert storage_tier(4004,'test')=='full' and storage_tier(4005,'test')=='node'


def test_declared_v23_grid_and_e10_seed_counts():
    from hbag3.gate import selection_grid
    from hbag3.matrix_v21 import matrix
    frozen=dict(load=.65,flows=288,capacity_by_rho={'1.4':1e6})
    assert len(selection_grid())==24 and {r['method'] for r in selection_grid()}=={'HBAG'}
    e5=matrix('E5',frozen);assert len(e5)==200 and all('V1' in c['methods'] for c in e5)
    e10=matrix('E10',frozen);assert len(e10)==85
    assert [c['config']['seed'] for c in e10 if c['label']=='M_5000']==list(range(4000,4005))


def test_offline_plan_executes_all_slots_after_a_single_solve(monkeypatch):
    import cvxpy as cp
    from hbag3.optional_r2 import fractional_mpc
    env=physical_fixture([0.,0.,500.,500.]);calls=[];original=cp.Problem.solve
    def counted(model,**settings):
        calls.append(settings);return original(model,**settings)
    monkeypatch.setattr(cp.Problem,'solve',counted)
    e=np.full(4,950000.);xs,acts,powers,info=fractional_mpc(env,0,e,{},4,return_plan=True)
    assert len(calls)==1 and len(xs)==len(acts)==len(powers)==4
    for t,(x,a,q) in enumerate(zip(xs,acts,powers)):
        rec=step(e,env['harvest'][t],55.,q,60.,env['battery']);e=rec.after_j
        assert e.min()>=env['battery'].minimum_j-1e-6
        assert np.all(e>=info['predicted_energy_j'][t+1]-10.)
    assert len(calls)==1


def test_e10_forecast_never_builds_paths_in_online_interval(monkeypatch):
    import hbag3.environment as geometry
    import hbag3.network as network
    from hbag3.forecast_r22 import precompute_price_catalogues,predicted_loads
    env=physical_fixture([0.,0.,500.,500.]);cat=env['catalogues'][0]
    env['config']=replace(env['config'],routing='hybrid',paths=4)
    env.update(geometric_sources=np.zeros((4,1),int),geometric_destinations=np.full((4,1),2),
               catalogue_build_s={0:.1},catalogues=[cat,None,None,None])
    monkeypatch.setattr(network,'snapshot',lambda *args:None)
    monkeypatch.setattr(geometry,'catalogue',lambda *args,**kwargs:cat)
    measured=precompute_price_catalogues(env,[0,2])
    assert measured['constructed_slots']==4
    def forbidden(*args,**kwargs):raise AssertionError('Path construction during online timing')
    monkeypatch.setattr(geometry,'catalogue',forbidden)
    assert predicted_loads(env,0,4).shape==(4,4)
    assert predicted_loads(env,2,10).shape==(10,4)


def test_esr_display_never_rounds_a_violation_to_full_safety():
    from hbag3.analysis_r2 import esr_percent
    assert esr_percent(1.)=='100.000'
    assert esr_percent(.9999999)=='<100.000'
    assert esr_percent(.99998)=='99.998'
