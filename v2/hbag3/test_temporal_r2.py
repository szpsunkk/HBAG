"""Check timing alignment, delayed boundary events, and Appendix A safety."""
from dataclasses import replace
import numpy as np
import pytest
from hbag3.model import Config
from hbag3.test_gate import tiny
from hbag3.energy import reserve_envelope
from hbag3.timing_r2 import run_temporal


def setup_env(monkeypatch,tmp_path,delay):
    import hbag3.timing_r2 as timing
    c=Config(seed=1000,planes=1,per_plane=4,flows=1,paths=1,orbits=1,capacity_j=1e6,
             initial_soc=.74,dt_override_s=15.,delay_s=delay,acquisition_s=0.)
    env=tiny(c,slots=4);env.update(dt=15.,times=np.arange(4)*15.,harvest_extended=np.zeros((8,4)))
    env['reserve']=reserve_envelope(np.zeros((8,4)),63.,15.,c.battery)[:5]
    monkeypatch.setattr(timing,'shared_inputs',lambda *args:None)
    monkeypatch.setattr(timing,'storage_tier',lambda *args:'node')
    monkeypatch.setattr(timing,'slot_harvest_lower',lambda *args:np.zeros(4))
    return c,env


def test_zero_delay_margin_equals_uncompensated(monkeypatch,tmp_path):
    c,env=setup_env(monkeypatch,tmp_path,0.)
    a=run_temporal(c,'HBAG',tmp_path/'a',env,phase='validation')
    b=run_temporal(c,'HBAG+margin',tmp_path/'b',env,phase='validation')
    for key in ['delivered_bits','min_soc','esr','terminal_energy_j']:
        assert a[key]==pytest.approx(b[key],rel=1e-12,abs=1e-10)
    assert b['bound_measured_max_j']==0.


@pytest.mark.parametrize('delay',[1.,5.,15.])
def test_delay_boundary_and_conservative_certificate(monkeypatch,tmp_path,delay):
    c,env=setup_env(monkeypatch,tmp_path,delay)
    a=run_temporal(c,'HBAG',tmp_path/'late',env,phase='validation')
    b=run_temporal(c,'HBAG+margin',tmp_path/'early',env,phase='validation')
    assert a['bound_max_excess_j']<1e-6 and b['bound_max_excess_j']<1e-6
    assert b['esr']==1. and b['violation_depth_j']==0.
    assert b['D_w']==475.
    import json
    bounds=json.loads((tmp_path/'late/delay_bounds.json').read_text())
    assert len(bounds)==4
    assert all(r['execution_s']-r['measurement_s']==delay for r in bounds)
    assert a['max_ledger_residual_j']<1e-6
    with np.load(tmp_path/'early/raw.npz') as data:
        assert data['pace_weight_fraction'].shape==(4,4)
        assert np.all(np.isfinite(data['pace_weight_fraction']))
        assert np.all(data['remaining_window_slots']>=1)
        e=data['energy_j'].astype(float)
        ledger=e[:-1]+data['integrated_drift_j']-data['integrated_spill_j']+data['integrated_unserved_j']
        bound=2*np.max(np.spacing(data['energy_j']))
        assert np.max(np.abs(e[1:]-ledger))<=bound
