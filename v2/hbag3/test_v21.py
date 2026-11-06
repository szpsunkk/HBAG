"""v2.1 behavior checks: reference/MMPP, counterfactuals, switches and storage."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from hbag3.environment import city_mmpp_states,demand_trace,reference_scale,catalogue
from hbag3.model import Config,Terminal,RATE
from hbag3.control import SwitchController,usable_catalogue
from hbag3.allocation import problem_for_switch
from hbag3.link_values import counterfactual_values,wake_values
from hbag3.storage import storage_tier,register_routes
from hbag3.runner import seed_guard
from hbag3.test_gate import tiny


def test_city_mmpp_shared_within_city_independent_across_cities():
    times=np.arange(2000)*60.
    states=city_mmpp_states(1000,[11,29,11,37],times)
    assert np.array_equal(states[:,0],states[:,2])
    assert not np.array_equal(states[:,0],states[:,1])
    assert abs(np.corrcoef(states[:,0],states[:,1])[0,1])<.2
    assert np.array_equal(city_mmpp_states(1000,[37,11],times),states[:,[3,0]])


def test_mmpp_total_normalization_and_common_absolute_demand():
    times=np.arange(96)*60.;shape=np.array([.5,1.,1.5]);source=np.zeros((96,3),int);h=np.zeros((96,4))
    c=Config(seed=1000)
    first,states=demand_trace(c,shape,2e9,times,np.zeros(3),source,h,[3,3,5],True)
    assert np.array_equal(states[:,0],states[:,1])
    assert first.mean(axis=0).sum()==pytest.approx(.65*2e9*shape.sum())
    for alternate in [replace(c,medium='rf'),replace(c,routing='distance',paths=1)]:
        np.testing.assert_array_equal(first,demand_trace(alternate,shape,2e9,times,np.zeros(3),source,h,[3,3,5]))


def all_on_two_routes():
    c=Config(planes=1,per_plane=4,flows=1,paths=2,capacity_j=1e6)
    env=tiny(c)
    ctl=SwitchController(c,'V1');sw=ctl.decide(env,0,np.full(4,950000.))
    cat=usable_catalogue(env['catalogues'][0],sw)
    p,_=problem_for_switch(env,0,cat,sw,True)
    return env,ctl,sw,cat,p


def test_counterfactual_free_detour_has_zero_loss():
    env,ctl,sw,cat,p=all_on_two_routes();x=np.array([[1.,0.,0.]])
    cf,sup=counterfactual_values(cat,x,np.ones(4),env['terminal'],p)
    assert np.max(cf)<1e-12
    assert sup.sum()>0
    # Remove the available alternate: the full old support value is then lost.
    cat.mask[0,1]=False
    cf,sup=counterfactual_values(cat,x,np.ones(4),env['terminal'],p)
    np.testing.assert_allclose(cf,sup)


def test_counterfactual_detour_cost_and_shared_slack_formula():
    env,ctl,sw,cat,p=all_on_two_routes();x=np.array([[1.,0.,0.]])
    first=cat.paths[0][0];second=cat.paths[0][1]
    weights=np.ones(4);weights[second[1]]=10.
    cf,sup=counterfactual_values(cat,x,weights,env['terminal'],p)
    assert cf.max()>0
    used=set(tuple(sorted(e)) for e in zip(first[:-1],first[1:]))
    edge_map={tuple(e):j for j,e in enumerate(cat.edges[:4])}
    a=cf[edge_map[next(iter(used))]]
    expected=(10.-1.)*env['terminal'].derivative(0.,1000.)*RATE/100.
    assert a==pytest.approx(expected)


def test_wake_value_serves_unmet_demand_once_per_direction():
    env,ctl,sw,cat,p=all_on_two_routes()
    geometric=env['catalogues'][0];on=np.ones(4,bool)
    route=geometric.paths[0][0];edge=tuple(sorted(route[:2]))
    j=list(map(tuple,geometric.edges[:4])).index(edge);on[j]=False
    values=wake_values(geometric,on,2/3,np.array([4.]),np.array([1.]),1.)
    assert values[j]==pytest.approx(1.5)
    assert np.count_nonzero(values)==1
    on[:]=False
    np.testing.assert_array_equal(wake_values(geometric,on,1.,np.array([4.]),np.array([1.]),1.),0.)


def test_zero_scarcity_has_no_a2_sleep_or_chatter():
    c=Config(planes=1,per_plane=4,flows=1,paths=1,capacity_j=1e6)
    env=tiny(c,demand=0.,slots=8);env['harvest'][:]=500.
    ctl=SwitchController(c,'HBAG');e=np.full(4,c.capacity_j)
    for t in range(8):
        sw=ctl.decide(env,t,e)
        assert sw.on.all() and sw.price_sleep==0 and sw.zero_price_a2==0
        assert np.all(sw.price_trace['pi']==0)
        ctl.price.commit(sw.fixed_w,t)


def test_a2_at_most_one_incident_edge_per_node():
    c=Config(planes=1,per_plane=4,flows=1,paths=1,capacity_j=1e6,lam=.5)
    env=tiny(c,demand=0.);ctl=SwitchController(c,'HBAG-xi')
    ctl.started=True;ctl.states[:]=True;ctl.ages[:]=4
    keys=list(map(tuple,env['catalogues'][0].edges[:4]));ctl.previous_edges=set(keys)
    ctl.low_counts={k:1 for k in keys};ctl.counterfactual={k:0. for k in keys}
    sw=ctl.decide(env,0,np.full(4,950000.))
    assert sw.price_sleep==2
    assert sw.a2_nodes.max()==1


def test_storage_and_independent_gate_selection_locks(tmp_path,monkeypatch):
    import hbag3.runner as runner
    assert storage_tier(1000,'validation','selection')=='selection'
    assert storage_tier(4000,'test')=='full'
    assert storage_tier(4019,'test')=='node'
    with pytest.raises(ValueError):storage_tier(4019,'test','full')
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'fingerprint',lambda:'test-source-fingerprint')
    gate=tmp_path/'results_r2/G0_prime/gate.json';gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps({'state':'PASS','plan_version':'v2.1'}))
    with pytest.raises(RuntimeError):seed_guard(Config(seed=4000),'test')
    frozen=tmp_path/'results_r2/E11_validation/frozen.json';frozen.parent.mkdir(parents=True)
    frozen.write_text(json.dumps({'state':'RUNNING','plan_version':'v2.1'}))
    with pytest.raises(RuntimeError):seed_guard(Config(seed=4000),'test')


def test_route_nodes_saved_only_once_in_shared_inputs(tmp_path):
    env,ctl,sw,cat,p=all_on_two_routes()
    first=register_routes(tmp_path,[cat,cat])
    second=register_routes(tmp_path,[cat])
    np.testing.assert_array_equal(first[0],second[0])
    with np.load(tmp_path/'inputs/paths.npz') as data:
        assert len(data['path_offsets'])==3  # only two distinct paths
    assert len(list(tmp_path.rglob('paths.npz')))==1


@pytest.mark.parametrize('util,delivery,load,expected',[
    (.049,.95,.65,'STOP_UTILIZATION'),(.051,.89,.65,'RETRY_LOAD_0P5'),
    (.049,.89,.65,'STOP_UTILIZATION'),(.051,.89,.5,'STOP_DELIVERY'),
    (.05,.90,.65,'PASS'),(.06,.91,.5,'PASS')])
def test_only_registered_gate_fallback(util,delivery,load,expected):
    from hbag3.gate import calibration_decision
    assert calibration_decision(util,delivery,load)==expected


def test_full_and_selection_run_storage(tmp_path,monkeypatch):
    import hbag3.runner as runner
    from hbag3.runner import run_policy
    env,ctl,sw,cat,p=all_on_two_routes()
    c=env['config'];n=len(env['catalogues'])
    env.update(times=np.arange(n)*60.,pairs=np.array([[10,20]]),plane_beta_deg=np.array([0.]),
               shape=np.array([1.]),source_lon=np.array([0.]),mmpp_states=np.zeros((n,1),bool),
               reference_scale_bps=1e9,reference_r1_bps=.5e9,build_s=0.,cache_key='synthetic_test',
               harvest_extended=env['harvest'],certified=np.ones(4,bool))
    env['shell'].period_s=60*n
    original_sources=runner.source_hashes()
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'source_hashes',lambda:original_sources)
    result=run_policy(c,'V0',tmp_path/'results_r2/small/V0',env=env,storage='full')
    assert result['delivered_weighted_hops']==pytest.approx(2.)
    assert result['reference_new_over_r1']==2.
    with np.load(tmp_path/'results_r2/small/V0/raw.npz') as data:
        assert 'path_nodes' not in data and 'executed_path_ids' in data
    assert (tmp_path/'results_r2/small/inputs/paths.npz').exists()
    result=run_policy(c,'HBAG',tmp_path/'results_r2/select/HBAG',env=env,storage='selection')
    assert {p.name for p in (tmp_path/'results_r2/select/HBAG').iterdir()}=={'summary.json','slots.json'}
    assert not (tmp_path/'results_r2/select/inputs.npz').exists()
    assert result['zero_price_a2_events']==0 and result['mean_shadow_relative_gap'] is None
    assert not result['shadow_evaluation_enabled'] and result['shadow_success_slots']==0
