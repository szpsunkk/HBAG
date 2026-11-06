"""Appendix gates and statistics; no real test-seed simulations or mail."""
from dataclasses import replace,asdict
import hashlib,json
import numpy as np
import pytest
from hbag3.model import Config
from hbag3.protocol_r231 import rf_grid,select_rf
from hbag3.analysis_r231 import paired_ratio,interval,orbit_deliveries
from hbag3.figures_r231 import select_satellite
from hbag3.plan_r2 import VERSION,SHA256,RF_EXPERIMENT


def grid_rows():
    return [dict(seed=s,grid_label=p['label'],esr=1.,delivered_bits=100.,terminal_energy_j=50.)
            for p in rf_grid() for s in range(1000,1010)]


def test_rf_selection_uses_only_complete_33_point_validation_grid_and_safety():
    points=rf_grid();assert [sum(p['method']==m for p in points) for m in ['HBAG','Lyapunov','Static+Guard']]==[24,6,3]
    rows=grid_rows();bad=points[0]['label'];better=points[1]['label']
    for r in rows:
        if r['grid_label']==bad:r['delivered_bits']=1e6
        if r['grid_label']==bad and r['seed']==1009:r['esr']=.999999
        if r['grid_label']==better:r['terminal_energy_j']=1.
    selected,table=select_rf(rows)
    assert selected['HBAG']['label']==better
    assert not next(r for r in table if r['label']==bad)['eligible']
    with pytest.raises(ValueError):select_rf(rows[:-1])
    with pytest.raises(ValueError):select_rf(rows+[rows[0]])
    altered=[dict(r,esr=.99) if r['grid_label'].startswith('Lyapunov') else r for r in rows]
    selected,_=select_rf(altered);assert 'Lyapunov' not in selected


@pytest.fixture
def rf_gate(tmp_path,monkeypatch):
    import hbag3.protocol_r231 as protocol
    import hbag3.runner as runner
    monkeypatch.setattr(protocol,'ROOT',tmp_path);monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'fingerprint',lambda:'unit-source')
    selected={m:next(p for p in rf_grid() if p['method']==m) for m in ['HBAG','Lyapunov','Static+Guard']}
    prior=tmp_path/'results_r2/E11_validation/frozen.json';prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps(dict(load=.65,flows=288,capacity_by_rho={'1.4':783756.3498013499})))
    row=dict(state='PASS',plan_version=VERSION,plan_sha256=SHA256,source_fingerprint='unit-source')
    frozen=tmp_path/'results_r2/E11_RF_v231/frozen.json';frozen.parent.mkdir(parents=True)
    frozen.write_text(json.dumps(dict(row,selected=selected,test_seeds=list(range(5000,5020)))))
    audit=tmp_path/'results_r2/v231/audit';audit.mkdir(parents=True)
    (audit/'unchanged_bitwise.json').write_text(json.dumps(row))
    (audit/'launch_ready.json').write_text(json.dumps(dict(row,frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())))
    return runner,protocol,tmp_path,json.loads(frozen.read_text())


def test_fresh_seeds_are_exclusive_to_frozen_rf_experiment(rf_gate):
    runner,protocol,root,frozen=rf_gate
    for seed in [5000,5019]:
        c,method=protocol.selected_config(seed,-10.,'HBAG',frozen)
        runner.seed_guard(c,'test',RF_EXPERIMENT,method)
        for experiment in [None,'E2','E4','E5']:
            with pytest.raises(ValueError):runner.seed_guard(c,'test',experiment,method)
        for changed in [replace(c,medium='optical'),replace(c,orbits=2),replace(c,dual_eta=99),replace(c,capacity_j=1e6)]:
            with pytest.raises(ValueError):runner.seed_guard(changed,'test',RF_EXPERIMENT,method)
    with pytest.raises(RuntimeError):runner.seed_guard(Config(seed=4000),'test')
    with pytest.raises(ValueError):runner.seed_guard(Config(seed=5000),'validation')


@pytest.mark.parametrize('name',['unchanged_bitwise.json','launch_ready.json'])
def test_current_rf_lock_requires_both_preflight_and_budget(rf_gate,name):
    runner,protocol,root,frozen=rf_gate;c,method=protocol.selected_config(5000,-10.,'HBAG',frozen)
    (root/'results_r2/v231/audit'/name).unlink()
    with pytest.raises(RuntimeError):runner.seed_guard(c,'test',RF_EXPERIMENT,method)


def test_rf_lock_rejects_postfreeze_parameter_change(rf_gate):
    runner,protocol,root,frozen=rf_gate;c,method=protocol.selected_config(5000,-10.,'HBAG',frozen)
    p=root/'results_r2/E11_RF_v231/frozen.json';r=json.loads(p.read_text());r['selected']['HBAG']['parameters']['dual_eta']=.3;p.write_text(json.dumps(r))
    c=replace(c,dual_eta=.3)
    with pytest.raises(RuntimeError):runner.seed_guard(c,'test',RF_EXPERIMENT,method)


def test_rf_storage_tier_never_leaks_to_another_experiment():
    from hbag3.storage import storage_tier
    assert storage_tier(5004,'test',experiment=RF_EXPERIMENT)=='full'
    assert storage_tier(5005,'test',experiment=RF_EXPERIMENT)=='node'
    with pytest.raises(ValueError):storage_tier(5000,'test',experiment='E2')
    with pytest.raises(ValueError):storage_tier(5005,'test','full',experiment=RF_EXPERIMENT)


def test_ratio_is_mean_of_paired_ratios_and_undefined_is_not_zero():
    row,values=paired_ratio([3,5],[1,1],[2,9],[4000,4001]);assert row['mean']==1.25
    row,values=paired_ratio([2,2,1],[1,2,3],[3,2,5],[4000,4001,4002])
    assert row['n']==2 and row['zero_denominator_seeds']==1 and row['negative_denominator_seeds']==1
    assert row['mean']==.5 and values[1]['ratio'] is None and values[2]['ratio']==-1
    row,_=paired_ratio([1,2],[1,2],[1,3],[4000,4001]);assert row['n']==0 and row['mean'] is None and row['ci95_low'] is None
    row,_=paired_ratio([1e-12],[0],[1],[4000]);assert row['mean']==1e12


def test_paired_bootstrap_respects_seed_as_unit():
    a=np.arange(20.)*1000;b=a+7
    row=interval(b-a);assert row['mean']==row['ci95_low']==row['ci95_high']==7


def test_orbits_sum_existing_slots_and_reject_missing_data(tmp_path):
    s=dict(config=dict(slots_per_orbit=2,orbits=2),dt_s=2.,delivered_bits=20.)
    (tmp_path/'summary.json').write_text(json.dumps(s))
    rows=[dict(slot=i,delivered_bps=i+1) for i in range(4)]
    (tmp_path/'slots.json').write_text(json.dumps(rows))
    np.testing.assert_array_equal(orbit_deliveries(tmp_path),[6,14])
    (tmp_path/'slots.json').write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError):orbit_deliveries(tmp_path)


def test_representative_rule_is_median_first_complete_entry_not_best_outcome():
    dark=np.array([[0,1,0],[1,0,1],[1,1,1],[0,1,0],[1,0,1]],bool)
    z=np.zeros_like(dark,float);z[1]=[5,0,3];z[2,1]=1
    chosen,ranked,rank=select_satellite({'discretionary_energy_j':z},{'harvest_w':np.where(dark,0,500)})
    assert chosen['satellite']==2 and chosen['entry_slot']==1 and rank==1 and len(ranked)==3
    z[1,0]=3;chosen,_,_=select_satellite({'discretionary_energy_j':z},{'harvest_w':np.where(dark,0,500)})
    assert chosen['satellite']==0
