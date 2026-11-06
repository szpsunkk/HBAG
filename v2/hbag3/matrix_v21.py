"""Declared R2 v2.1 matrix and frozen per-rho policy configuration."""
from dataclasses import asdict,replace
from itertools import product
from .model import Config
from .environment import ROOT
from .runner import write_json

SEEDS=list(range(4000,4020))
MAIN=['V0','Guard_0.75','Guard_0.8','Guard_0.9','Util-Sleep','Lyapunov','V1','V2','HBAG-xi','HBAG-Lyap','HBAG']
COMMON=['V0','Guard*','Util-Sleep','Lyapunov','V1','HBAG']


def policy_config(base,method,frozen):
    rho=base.rho_energy if base.rho_energy in [1.2,1.4,1.6] else 1.2
    selected=frozen['selected_by_rho'][str(rho)]
    if method.startswith('Guard_'):
        return replace(base,guard_soc=float(method.split('_')[1])),'Static+Guard'
    key='Static+Guard' if method=='Guard*' else ('HBAG' if method in ('V2','HBAG-central','HBAG+margin') else method)
    params=selected.get(key,{}).get('parameters',{})
    return replace(base,**params),('Static+Guard' if method=='Guard*' else method)


def matrix(experiment,frozen):
    cells=[]
    def add(label,methods,**fields):
        rho=fields.pop('rho_energy',1.4)
        for seed in (SEEDS[:5] if experiment=='E10' and fields.get('planes',0)*fields.get('per_plane',0)==5000 else SEEDS):
            c=Config(seed=seed,load=frozen['load'],flows=frozen['flows'],rho_energy=rho,
                     capacity_j=frozen['capacity_by_rho'][str(rho)])
            c=replace(c,**fields)
            cells.append(dict(experiment=experiment,label=label,config=asdict(c),methods=methods))
    if experiment=='E2':
        for rho in [1.2,1.4,1.6]:add('rho_'+str(rho),MAIN+(['HBAG-central'] if rho==1.4 else []),rho_energy=rho)
        for rho in [.6,1.]:add('rho_'+str(rho),COMMON,rho_energy=rho)
    elif experiment=='E4':
        for interference in [-300.,-10.,0.]:add('rf_IN_'+str(interference),['V0','Guard*','Lyapunov','V1','HBAG'],medium='rf',interference_db=interference)
    elif experiment=='E5':
        for load,traffic in product([.5,.8,1.1],['mmpp','hotspot5','eclipse']):add('load_%g_%s'%(load,traffic),['V0','Guard*','Util-Sleep','Lyapunov','V1','HBAG'],load=load,traffic=traffic)
        add('load_0.65_steady',['V0','Guard*','Util-Sleep','Lyapunov','V1','HBAG'],load=.65,traffic='steady')
    elif experiment=='E6':
        for routing,k in [('hybrid',4),('hops',3),('distance',3),('distance',1),('hops',5)]:
            if routing=='hybrid':continue # E2 exact common cell is linked, not duplicated.
            add('K%d_%s'%(k,routing),['V0','V1','HBAG'],paths=k,routing=routing)
    elif experiment=='E7':
        for solar,beta in product([500.,275.],[0.,60.]):
            add('H_%g_beta_%g_soc_0.95'%(solar,beta),['V0','Guard*','Lyapunov','V1','HBAG'],solar_w=solar,beta_deg=beta,orbits=6,rho_energy=1.)
        add('H_500_beta_0_soc_0.8',['V0','Guard*','Lyapunov','V1','HBAG'],solar_w=500.,beta_deg=0.,initial_soc=.8,orbits=6,rho_energy=1.)
    elif experiment=='E9_iterations':
        for iterations in [10,20,50,100]:add('iterations_%d'%iterations,['HBAG'],max_iterations=iterations)
    elif experiment=='E8':
        for dt,delay in product([15.,60.],[0.,1.,5.,15.]):
            add('dt_%g_delay_%g'%(dt,delay),['V0','HBAG','HBAG+margin'],planes=8,per_plane=8,flows=32,orbits=1,dt_override_s=dt,delay_s=delay,substep_s=1.)
        for acq in [0.,10.,30.,60.]:
            add('acquisition_%g'%acq,['V0','HBAG'],planes=8,per_plane=8,flows=32,orbits=1,dt_override_s=60.,acquisition_s=acq,substep_s=1.)
    elif experiment=='E10':
        for p,q in [(16,16),(24,24),(32,32),(22,72),(50,100)]:
            add('M_%d'%(p*q),['HBAG'],planes=p,per_plane=q,flows=p*q//2,orbits=1)
    else:raise ValueError(experiment)
    return cells


def declared_counts():
    # Gate-independent declaration; this does not instantiate or run test environments.
    return dict(E2=dict(cells=100,method_runs=920,orbits=3),
        E9_iterations=dict(cells=80,method_runs=80,orbits=3),E4=dict(cells=60,method_runs=300,orbits=3),
        E5=dict(cells=200,method_runs=1200,orbits=3),E6=dict(cells=80,method_runs=240,orbits=3),
        E7=dict(cells=100,method_runs=500,orbits=6),E8=dict(cells=240,method_runs=640,orbits=1),
        E10=dict(cells=85,method_runs=85,snapshots=330),E11=dict(method_runs=720+90,orbits=2,storage='selection'))


if __name__=='__main__':
    write_json(ROOT/'configs_r2/matrix_v21_counts.json',declared_counts())
    print('Wrote declarations only; test seeds remain locked.')
