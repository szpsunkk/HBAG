"""Small validation fixtures for the actual R2 P2 optimization layers."""
import numpy as np
import pytest
from hbag3.test_gate import tiny
from hbag3.control import SwitchController
from hbag3.optional_r2 import ppo_action,fractional_mpc
from hbag3.energy import step


def test_ppo_admission_zero_blocks_service_but_keeps_physical_sleep_power():
    env=tiny();c=env['config'];env['times']=np.arange(4)*60.
    ctl=SwitchController(c,'V1');energy=np.full(4,.95*c.capacity_j)
    x,q,cat,sw,solved=ppo_action(env,0,energy,ctl,np.zeros(4))
    assert np.all(x==0)
    np.testing.assert_allclose(q,sw.fixed_w)
    assert np.all(q>=4*c.sleep_w)


def test_fractional_oracle_uses_shared_switches_and_feasible_energy():
    env=tiny();c=env['config'];env['times']=np.arange(4)*60.
    energy=np.full(4,.95*c.capacity_j)
    previous={tuple(e):1. for e in env['catalogues'][0].edges[:4]}
    x,a,q,info=fractional_mpc(env,0,energy,previous,2)
    assert info['horizon']==2 and len(a)==4
    assert np.all((a>=0)&(a<=1)) and x.sum()<=1.+1e-7
    after=step(energy,env['harvest'][0],c.base_w,q,60.,env['battery']).after_j
    assert after.min()>=env['battery'].minimum_j-1e-6
    assert np.all(q>=4*c.sleep_w-1e-5)
    cat=env['catalogues'][0]
    rates=cat.incidence@x.ravel()*1e9
    assert np.all(rates<=cat.link_capacity_bps*np.tile(a,2)+1e3)
