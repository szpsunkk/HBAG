from dataclasses import replace
import numpy as np
import pytest
from hbag3.model import Config, Terminal, RATE
from hbag3.test_gate import tiny
from hbag3.control import SwitchController, usable_catalogue
from hbag3.allocation import problem_for_switch, protect_bus
from hbag3.solvers import solve_rf, solve_rf_central, node_power, recover_rf, objective
from hbag3.runner import acquisition_energy_j


@pytest.mark.parametrize('dt,expected',[(5.,[0.,0.,0.,0.,1.]),(15.,[0.,2/3,1.,1.,1.]),(60.,[2/3,1.,1.,1.,1.])])
def test_acquisition_duration_and_energy(dt,expected):
    c=Config(planes=1,per_plane=4,flows=1,paths=1,capacity_j=1e6)
    env=tiny(c,slots=6);env['dt']=dt;ctl=SwitchController(c,'V0');e=np.full(4,950000.)
    total=0.
    for t,value in enumerate(expected):
        sw=ctl.decide(env,t,e)
        np.testing.assert_allclose(sw.capture,value,rtol=0,atol=1e-15)
        total+=acquisition_energy_j(sw,dt,c)
    assert total==4*2*25*20
    e[:]=0.;ctl.decide(env,5,e)
    e[:]=950000.;sw=ctl.decide(env,5,e)
    np.testing.assert_allclose(sw.capture,max(1-20/dt,0))


def test_rf_perspective_and_optical_exactness():
    r=np.array([0.,1e8,4e8]);d=np.full(3,1000.);capture=np.array([0.,.25,1.])
    term=Terminal(medium='rf')
    expected=np.array([0.,.25*term.increment(4e8,1000.),term.increment(4e8,1000.)])
    np.testing.assert_array_equal(term.slot_increment(r,d,capture),expected)
    assert expected[1]>term.increment(r,d)[1]
    optical=Terminal()
    assert optical.slot_increment(r,d,capture).tobytes()==optical.increment(r,d).tobytes()
    assert optical.slot_derivative(r,d,capture).tobytes()==optical.derivative(r,d).tobytes()


@pytest.mark.parametrize('capture',[0.,.25,1.])
def test_rf_budget_solver_and_ledger(capture):
    c=Config(planes=1,per_plane=4,flows=1,paths=1,capacity_j=1e6,medium='rf')
    env=tiny(c,demand=5e9);term=Terminal(medium='rf');env['terminal']=term
    geo=replace(env['catalogues'][0],link_capacity_bps=term.capacity(env['catalogues'][0].distance_km))
    env['catalogues']=[geo]*4;ctl=SwitchController(c,'V1');e=np.full(4,950000.)
    sw=ctl.decide(env,0,e);sw.capture[:]=capture
    cat=usable_catalogue(geo,sw);p,meta=problem_for_switch(env,0,cat,sw,True)
    meta['incremental_w'][:]=2.
    x,info=solve_rf(p,cat,term,meta,800)
    xc,_=solve_rf_central(p,cat,term,meta)
    assert objective(p,xc,cat,term,meta)-objective(p,x,cat,term,meta)<.005
    for xx in (x,xc,recover_rf(np.full(p.shape,5.),p,cat,term,meta['incremental_w'],meta['capture'])):
        assert p.violations(xx)<1e-8
        rate=cat.incidence@xx.ravel()*RATE
        expected=term.slot_increment(rate,cat.distance_km,np.tile(sw.capture,2))
        np.testing.assert_allclose(node_power(cat,term,xx,sw.fixed_w,meta['capture']),sw.fixed_w+cat.sender@expected)
        assert np.max(cat.sender@expected)<=2.+1e-6
    if capture==0:assert not x.any()


def test_legacy_switch_reproduces_one_slot_acquisition():
    c=Config(planes=1,per_plane=4,flows=1,paths=1,acquisition_carry=False)
    env=tiny(c);env['dt']=5.;ctl=SwitchController(c,'V0');e=np.full(4,.95*c.capacity_j)
    assert not ctl.decide(env,0,e).capture.any()
    assert ctl.decide(env,1,e).capture.all()


def test_sleep_greedy_shared_bottleneck_and_exact_removal():
    from scipy import sparse
    from hbag3.environment import Catalogue
    from hbag3.optimization import AllocationProblem
    from hbag3.link_values import counterfactual_values
    from hbag3.v24_sleep import greedy_values,removed_problem,optimum
    edges=np.array([[0,1],[1,3],[0,2],[2,3],[1,0],[3,1],[2,0],[3,2]])
    paths=[[(0,1,3),(0,2,3)]]*2
    incidence=sparse.csr_matrix(np.array([[1,0,1,0],[1,0,1,0],[0,1,0,1],[0,1,0,1]]+[[0,0,0,0]]*4,float))
    sender=sparse.csr_matrix((np.ones(8),(edges[:,0],np.arange(8))),shape=(4,8))
    capacity=np.array([2,2,1,1,2,2,1,1],float)*RATE
    cat=Catalogue(np.ones((2,2),bool),paths,incidence,sender,capacity,np.ones(8)*1000,
                  edges,np.array([0,0]),np.array([3,3]),np.ones((2,2),int)*2)
    p=AllocationProblem(incidence,capacity/RATE,np.ones(2),cat.mask,np.zeros((2,2)),np.ones(2),np.ones(2),.02)
    x=np.array([[1.,0.],[1.,0.]])
    cf,_=counterfactual_values(cat,x,np.zeros(4),Terminal(),p,np.ones(2))
    greedy=greedy_values(cat,x,np.zeros(4),Terminal(),p,np.ones(2))
    assert cf[0]==0. and greedy[0]==.5
    full=optimum(p);removed=optimum(removed_problem(p,cat,[0]))
    assert full['phi']>removed['phi']+.5
    assert full['gap']<=1e-9 and removed['gap']<=1e-9
