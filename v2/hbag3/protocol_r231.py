"""Appendix B experiment identity and RF-only seed gate; no policy algorithms."""
from dataclasses import asdict,replace
import hashlib,json
import numpy as np
from .environment import ROOT
from .model import Config
from .plan_r2 import VERSION,SHA256,RF_TEST_SEEDS,RF_EXPERIMENT

VALIDATION_SEEDS=tuple(range(1000,1010))
IN_LEVELS=(-300.,-10.,0.)
METHODS=('V0','Guard*','Lyapunov','V1','HBAG')
RF_SELECTION='E11_RF_v231'


def prior_frozen():
    return json.loads((ROOT/'results_r2/E11_validation/frozen.json').read_text())


def base_config(seed,orbits=2,interference=-10.):
    frozen=prior_frozen()
    return Config(seed=seed,orbits=orbits,load=frozen['load'],flows=frozen['flows'],
                  medium='rf',interference_db=interference,rho_energy=1.4,
                  capacity_j=frozen['capacity_by_rho']['1.4'])


def rf_grid():
    from .gate import validation_grid
    return [r for r in validation_grid() if r['method'] in ('HBAG','Lyapunov','Static+Guard')]


def select_rf(rows):
    """Require every validation seed safe; no target-driven search or fallback."""
    table=[];selected={}
    for point in rf_grid():
        group=sorted([r for r in rows if r['grid_label']==point['label']],key=lambda r:r['seed'])
        if [r['seed'] for r in group]!=list(VALIDATION_SEEDS):raise ValueError('Incomplete RF grid point '+point['label'])
        table.append(dict(point,n=10,eligible=all(r['esr']==1 for r in group),
                          min_esr=min(r['esr'] for r in group),mean_esr=float(np.mean([r['esr'] for r in group])),
                          mean_delivered_bits=float(np.mean([r['delivered_bits'] for r in group])),
                          mean_terminal_energy_j=float(np.mean([r['terminal_energy_j'] for r in group]))))
    for method in ('HBAG','Lyapunov','Static+Guard'):
        eligible=[r for r in table if r['method']==method and r['eligible']]
        if eligible:selected[method]=min(eligible,key=lambda r:(-r['mean_delivered_bits'],r['mean_terminal_energy_j']))
    return selected,table


def selected_config(seed,interference,method,frozen):
    actual='Static+Guard' if method=='Guard*' else method
    params=frozen['selected'].get(actual,{}).get('parameters',{})
    return replace(base_config(seed,3,interference),**params),actual


def guard_rf_test(config,experiment,method):
    from .runner import fingerprint
    if experiment!=RF_EXPERIMENT:raise ValueError('Fresh seeds 5000–5019 are exclusive to E4-double-prime')
    if config.seed not in RF_TEST_SEEDS:raise ValueError('Unexpected RF test seed')
    if method not in ('V0','Static+Guard','Lyapunov','V1','HBAG'):raise ValueError('Unregistered E4-double-prime method')
    stamp=fingerprint();records={}
    for name in ['v231/audit/unchanged_bitwise.json',RF_SELECTION+'/frozen.json','v231/audit/launch_ready.json']:
        p=ROOT/'results_r2'/name
        if not p.exists():raise RuntimeError('RF test seeds locked: missing '+name)
        row=json.loads(p.read_text());records[name]=row
        if row.get('state')!='PASS' or row.get('plan_version')!=VERSION or row.get('source_fingerprint')!=stamp or row.get('plan_sha256')!=SHA256:
            raise RuntimeError('RF test seeds locked: stale or failed '+name)
    frozen=records[RF_SELECTION+'/frozen.json']
    if frozen.get('test_seeds')!=list(RF_TEST_SEEDS):raise RuntimeError('Frozen RF seed range mismatch')
    expected,actual=selected_config(config.seed,config.interference_db,method,frozen)
    if config.interference_db not in IN_LEVELS or asdict(config)!=asdict(expected):
        raise ValueError('E4-double-prime config differs from frozen RF protocol')
    launch=records['v231/audit/launch_ready.json']
    digest=hashlib.sha256((ROOT/'results_r2'/RF_SELECTION/'frozen.json').read_bytes()).hexdigest()
    if launch.get('frozen_sha256')!=digest:raise RuntimeError('RF launch budget is not bound to frozen parameters')
