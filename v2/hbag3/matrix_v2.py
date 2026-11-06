"""Declared v2 experiment matrix. Listing it never executes or inspects test data."""
from itertools import product
from .environment import ROOT
from .runner import write_json


def declared_matrix():
    main=['V0','Guard(.75)','Guard(.80)','Guard(.90)','Util-Sleep','Lyapunov','V1','V2','HBAG-xi','HBAG-Lyap','HBAG','HBAG-central']
    common=['V0','Guard*','Util-Sleep','Lyapunov','V1','HBAG']
    rows=[]
    def add(priority,experiment,cells,methods,orbits=3,notes=''):
        rows.append(dict(priority=priority,experiment=experiment,cells=cells,methods=methods,
                         seeds=list(range(3000,3020)),orbits=orbits,notes=notes,state='locked_until_G0_PASS'))
    add('P0','E1',[],['V1','HBAG'],notes='10^4 randomized 8×8 trials, recursive certificate and counterexample; not replaced by pre-G0 unit checks')
    add('P0','E2_main',[dict(rho_energy=1.2)],main)
    add('P0','E2_energy',[dict(rho_energy=r) for r in [.6,1.,1.4]],common)
    add('P0','E3_extra',[dict(rho_energy=1.)],['V2','HBAG-xi','HBAG-Lyap'],notes='Reuse E2 for all other paired cells; 10^4 bootstrap, two-sided Wilcoxon, global Holm')
    add('P0','E9_snapshots',[dict(planes=8,per_plane=8,flows=32,load=1.2),dict(planes=24,per_plane=24,flows='G0',load=.65)],['HBAG'],notes='Three slots with greatest count of <1% node-budget slack, tie earlier slot; CP/Dual/ADMM vs Clarabel 1e-10')
    add('P0','E9_iterations',[dict(max_iterations=i,rho_energy=1.2) for i in [10,20,50,100,200]],['HBAG'],notes='Reuse identical 200-iteration E2 runs')
    add('P1','E4',[dict(medium='rf',interference_db=i,rho_energy=1.2) for i in [-300.,-10.,0.]],['V0','Guard*','Lyapunov','V1','HBAG'],notes='Optical Emax, shared absolute demand; report RF effective rho')
    add('P1','E5',[dict(load=l,traffic=t) for l,t in product([.5,.8,1.1],['mmpp','hotspot5','eclipse'])]+[dict(load=.65,traffic='steady')],['V0','Guard*','Util-Sleep','Lyapunov','HBAG'])
    add('P1','E6',[dict(routing=r,paths=k) for r,k in [('hybrid',4),('hops',3),('distance',3),('distance',1),('hops',5)]],['V0','V1','HBAG'],notes='Reuse identical default routing E2 cells')
    add('P1','E7',[dict(solar_w=h,beta_deg=b,initial_soc=.95,rho_energy=1.) for h,b in product([500.,275.],[0.,60.])]+[dict(solar_w=500.,beta_deg=0.,initial_soc=.8,rho_energy=1.)],['V0','Guard*','Lyapunov','V1','HBAG'],orbits=6)
    add('P1','E8_delay',[dict(delay_s=t,dt_override_s=d,planes=8,per_plane=8,substep_s=1.) for t,d in product([0.,1.,5.,15.],[15.,60.])],['V0','HBAG','HBAG+margin'],orbits=1,notes='Distinct delayed versus early-measure/on-time-execute schedules, delta=tau D, D=475W; price uses uncompensated Ehat')
    add('P1','E8_acquisition',[dict(acquisition_s=a,dt_override_s=60.,planes=8,per_plane=8,substep_s=1.) for a in [0.,10.,30.,60.]],['V0','HBAG'],orbits=1)
    add('P1','E10',[dict(planes=p,per_plane=q,flows=p*q//2) for p,q in [(16,16),(24,24),(32,32),(22,72),(50,100)]],['HBAG'],orbits=1,notes='Four measured snapshots; one core and one BLAS thread, no extrapolated timings')
    add('P2','E13',[],['PPO admission + certificate routing'],notes='Retrain CPU hybrid in R2; declare actual training budget')
    add('P2','MPC',[dict(planes=8,per_plane=8)],['fractional-switch oracle MPC'],notes='Optimistic reference on small graph only')
    return rows


if __name__=='__main__':
    write_json(ROOT/'configs_r2/matrix_v2.json',declared_matrix())
    print('Wrote configuration declarations only; no experiments launched and no test data read.')
