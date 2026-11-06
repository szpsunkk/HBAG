"""Pre-G0 implementation checks; these are not the later E1' 10^4 trials."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from hbag3.model import Config, Terminal
from hbag3.energy import step, load_budget, reserve_envelope
from hbag3.environment import catalogue
from hbag3.network import Snapshot, Shell, snapshot
from hbag3.control import SwitchController, usable_catalogue, terminal_ports
from hbag3.allocation import allocate, protect_bus
from hbag3.runner import eclipse_windows, seed_guard, save_npz
from hbag3.pricing import OrbitalPrice


def tiny(config=None, demand=1e9, slots=4):
    c=config or Config(planes=1,per_plane=4,flows=1,paths=1,orbits=1,capacity_j=1e6)
    edges=np.array([[0,1],[0,3],[1,2],[2,3]])
    d=np.full(4,1000.);term=Terminal()
    graph=sparse.coo_matrix((np.tile(d,2),(edges.T.ravel(),edges[:,::-1].T.ravel())),shape=(4,4)).tocsr()
    net=Snapshot(np.zeros((4,3)),edges,d,term.capacity(d),np.full(4,2),graph)
    cat=catalogue(net,[0],[2],c.paths)
    return dict(config=c,shell=SimpleNamespace(size=4),terminal=term,battery=c.battery,dt=60.,
                harvest=np.zeros((slots,4)),reserve=np.full((slots+1,4),.7*c.capacity_j),
                catalogues=[cat]*slots,demand_bps=np.full((slots,1),demand))


def test_v0_handshake_acquisition_and_sleep_power():
    env=tiny();c=env['config'];ctl=SwitchController(c,'V0');e=np.full(4,.95*c.capacity_j)
    first=ctl.decide(env,0,e)
    np.testing.assert_allclose(first.capture,2/3)
    np.testing.assert_allclose(first.fixed_w,54.)
    second=ctl.decide(env,1,e)
    np.testing.assert_allclose(second.capture,1.)
    assert second.sleep_events==0 and second.on.all()
    e[1]=.1*c.capacity_j
    cut=ctl.decide(env,2,e)
    assert not cut.terminals_on[1].any()
    assert not cut.on[0] and not cut.on[2]
    assert cut.fixed_w[1]==8.


def test_capture_longer_than_slot_and_energy_ledger():
    c=Config(planes=1,per_plane=4,flows=1,paths=1,capacity_j=1e6,acquisition_s=60.)
    env=tiny(c);e=np.full(4,.95*c.capacity_j);ctl=SwitchController(c,'V0')
    s=ctl.decide(env,0,e);cat=usable_catalogue(env['catalogues'][0],s)
    x,p,meta,info=allocate(env,0,cat,s,ctl)
    x,q,v=protect_bus(env,0,e,cat,s,x,p,meta)
    assert not np.any(x) and not np.any(s.capture)
    np.testing.assert_allclose(q,54.)
    rec=step(e,0,55,q,60,c.battery)
    np.testing.assert_allclose(e-rec.after_j,(55+54)*60/.95)


@pytest.mark.parametrize('method',['V1','HBAG'])
def test_hard_fixed_budget_and_infeasible_sleep(method):
    env=tiny();c=env['config'];ctl=SwitchController(c,method)
    # Exactly 20 W ISL allowance: even one ON port (31 W total) is unaffordable.
    e=np.full(4,.7*c.capacity_j+(55+20)*60/.95)
    s=ctl.decide(env,0,e)
    assert not s.on.any() and s.budget_forced.all()
    np.testing.assert_allclose(s.fixed_w,8.)
    cat=usable_catalogue(env['catalogues'][0],s)
    x,p,meta,info=allocate(env,0,cat,s,ctl)
    x,q,v=protect_bus(env,0,e,cat,s,x,p,meta)
    assert np.all(step(e,0,55,q,60,c.battery).after_j>=.7*c.capacity_j-1e-6)
    e=np.full(4,.7*c.capacity_j)
    s=ctl.decide(env,1,e)
    assert s.infeasible.all() and not s.terminals_on.any()


def test_common_k_plus_one_detour():
    env=tiny();ctl=SwitchController(env['config'],'V0')
    s=ctl.decide(env,0,np.full(4,950000.));original=env['catalogues'][0]
    route=original.paths[0][0]
    edge_to_block=tuple(sorted(route[:2]))
    j=list(map(tuple,original.edges[:4])).index(edge_to_block)
    s.on[j]=False;s.capture[j]=0.
    cat=usable_catalogue(original,s)
    assert not cat.mask[0,0] and cat.mask[0,1]
    assert edge_to_block not in [tuple(sorted(e)) for e in zip(cat.paths[0][1][:-1],cat.paths[0][1][1:])]


def test_complete_eclipse_windows_exclude_boundary_fragments():
    h=np.array([[0,500],[0,0],[500,0],[0,500],[0,0],[500,0]],float)
    rows=eclipse_windows(h,np.ones_like(h)*10)
    assert [(x['satellite'],x['start_slot'],x['end_slot_exclusive'],x['battery_draw_j']) for x in rows]==[(0,3,5,20.),(1,1,3,20.)]


def test_reserve_and_budget_inverse_random_physics():
    rng=np.random.default_rng(1000);b=Config(capacity_j=1e6).battery
    h=rng.uniform(0,500,(100,64));base=rng.uniform(30,55,(100,64))
    s=reserve_envelope(h,base+8,60,b)
    e=s[0]+rng.uniform(0,10000,64)
    for t in range(100):
        q=load_budget(e,s[t+1],h[t],60,b)-base[t]
        assert np.min(q)>=8-1e-6
        action=8+rng.uniform(0,1,64)*np.minimum(np.maximum(q-8,0),232.)
        e=step(e,h[t],base[t],action,60,b).after_j
        assert np.min(e-s[t+1])>=-1e-6


def test_seed_lock_and_float32_storage(tmp_path):
    with pytest.raises(ValueError):seed_guard(Config(seed=3000),'validation')
    with pytest.raises(RuntimeError):seed_guard(Config(seed=3000),'test')
    save_npz(tmp_path/'a.npz',energy=np.array([1.,2.]),state=np.array([True,False]),indices=np.array([1,2],np.int32))
    raw=np.load(tmp_path/'a.npz')
    assert raw['energy'].dtype==np.float32 and raw['state'].dtype==bool and raw['indices'].dtype==np.int32


def test_four_distinct_ports_on_real_geometry():
    shell=Shell(24,24)
    for t in [0,shell.period_s/4,shell.period_s/2]:
        net=snapshot(shell,t,Terminal())
        ports=terminal_ports(net.edges,24,24)
        assert ports.shape==net.edges.shape and np.all((ports>=0)&(ports<4))


def test_v2_hop_order_and_distinct_sun_path():
    import networkx as nx
    edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    dist=np.array([100.,150.,500.,50.,150.,50.]);term=Terminal()
    graph=sparse.coo_matrix((np.tile(dist,2),(edges.T.ravel(),edges[:,::-1].T.ravel())),shape=(4,4)).tocsr()
    net=Snapshot(np.zeros((4,3)),edges,dist,term.capacity(dist),np.full(4,3),graph)
    dark=np.array([False,True,False,False])
    cat=catalogue(net,[0],[3],4,'hybrid',node_dark=dark)
    assert cat.paths[0][:3]==[(0,3),(0,2,3),(0,1,3)]
    assert len(set(cat.paths[0]))==4
    g=nx.Graph()
    for (a,b),d in zip(edges,dist):g.add_edge(int(a),int(b),weight=1+1.5*(int(dark[a])+int(dark[b])))
    all_routes=list(nx.all_simple_paths(g,0,3))
    cost=lambda p:sum(g[a][b]['weight'] for a,b in zip(p[:-1],p[1:]))
    optimum=min(cost(p) for p in all_routes if tuple(p) not in cat.paths[0][:3])
    assert cost(cat.paths[0][3])==optimum
    # R1 diagnostic uses independent distance K=3; it is no longer the load denominator.
    assert catalogue(net,[0],[3],3,'distance').paths[0][0]==(0,2,3)


def test_dual_window_pacing_update_and_warm_start():
    c=Config(planes=1,per_plane=4,flows=1,paths=1,dual_eta=.1,dual_warm='prev',capacity_j=1e6)
    env=tiny(c,slots=6)
    env['harvest'][:]=np.array([0,0,500,0,0,500])[:,None]
    env['harvest_extended']=env['harvest']
    price=OrbitalPrice(c,'HBAG');e=np.full(4,700600.)
    w,xi,trace=price.prepare(env,0,e)
    np.testing.assert_allclose(w,.01);np.testing.assert_array_equal(trace['remaining_slots'],2)
    np.testing.assert_allclose(trace['pace_w'],8+.95*600/120)
    price.commit(np.full(4,60.),0)
    expected=.1*(60-(8+.95*600/120))/100
    np.testing.assert_allclose(price.mu,expected)
    w,xi,trace=price.prepare(env,1,e);price.commit(np.full(4,60.),1)
    last=price.mu.copy();np.testing.assert_allclose(price.last_window_end,last)
    price.prepare(env,2,e);price.commit(np.full(4,60.),2)
    w,xi,trace=price.prepare(env,3,e)
    np.testing.assert_allclose(trace['mu'],last)


def test_spill_reset_and_separate_price_formulas():
    c=Config(planes=1,per_plane=4,flows=1,paths=1,capacity_j=1e6)
    env=tiny(c,slots=12);env['harvest'][:10]=500.
    dual=OrbitalPrice(c,'HBAG');dual.mu[:]=2.
    w,xi,trace=dual.prepare(env,0,np.full(4,990000.))
    assert trace['spill_reset'].all();np.testing.assert_allclose(w,.01)
    dual.commit(np.full(4,240.),0);np.testing.assert_allclose(dual.mu,0.)
    low=np.full(4,730000.)
    xi_price=OrbitalPrice(c,'HBAG-xi').prepare(env,0,low)[0]
    lyap_price=OrbitalPrice(c,'Lyapunov').prepare(env,0,low)[0]
    np.testing.assert_allclose(xi_price,.01+c.lam/(.1+.1))
    np.testing.assert_allclose(lyap_price,.01+c.lyapunov_lambda*(c.lyapunov_theta-.1))
    np.testing.assert_allclose(OrbitalPrice(c,'V1').prepare(env,0,low)[0],.01)


def test_v2_a3_retains_pre_sleep_support():
    c=Config(planes=1,per_plane=4,flows=1,paths=1,capacity_j=1e6,lam=.5)
    env=tiny(c,demand=0.);ctl=SwitchController(c,'HBAG-xi')
    ctl.started=True;ctl.ages[:]=2
    keys=[tuple(map(int,e)) for e in env['catalogues'][0].edges[:4]]
    ctl.last_awake_support={k:20. for k in keys}
    s=ctl.decide(env,0,np.full(4,950000.))
    assert s.on.all()
    ctl=SwitchController(c,'HBAG-xi');ctl.started=True;ctl.ages[:]=2
    s=ctl.decide(env,0,np.full(4,950000.))
    assert not s.on.any()
