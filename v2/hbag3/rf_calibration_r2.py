"""RF's own nominal energy tightness, without retuning the optical battery."""
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from .model import Config,Terminal
from .environment import ROOT,build_environment
from .runner import run_policy,write_json


def job(args):
    seed,frozen=args;c=Config(seed=seed,orbits=1,load=frozen['load'],capacity_j=100e6)
    env=build_environment(c);rows=[]
    for interference in [-300.,-10.,0.]:
        chosen=replace(c,medium='rf',interference_db=interference)
        term=Terminal(medium='rf',interference_db=interference)
        # Geometry, common candidate paths, optical reference and exogenous
        # demand are identical; only the evaluated link medium changes.
        cats=[replace(cat,link_capacity_bps=term.capacity(cat.distance_km)) for cat in env['catalogues']]
        rf=dict(env,config=chosen,terminal=term,catalogues=cats)
        out=ROOT/'results_r2/RF_nominal_validation'/('IN_'+str(interference))/('seed_%d'%seed)/'V0'
        rows.append(run_policy(chosen,'V0',out,env=rf,phase='validation',storage='node'))
    return rows


def run_rf_calibration(frozen,workers):
    done=ROOT/'results_r2/RF_nominal_validation/calibration.json'
    if done.exists():return json.loads(done.read_text())
    with ProcessPoolExecutor(max_workers=workers) as pool:groups=list(pool.map(job,[(s,frozen) for s in range(1000,1010)]))
    runs=[r for group in groups for r in group];rows=[]
    for interference in [-300.,-10.,0.]:
        selected=[r for r in runs if r['config']['interference_db']==interference]
        windows=[]
        for r in selected:
            out=done.parent/('IN_'+str(interference))/('seed_%d'%r['seed'])/'V0'
            windows.extend(x['battery_draw_j'] for x in json.loads((out/'eclipse_windows.json').read_text()))
        mean=float(np.mean(windows));cap=frozen['capacity_by_rho']['1.4']
        rows.append(dict(interference_db=interference,validation_seeds=list(range(1000,1010)),
            window_count=len(windows),nominal_eclipse_draw_mean_j=mean,nominal_eclipse_draw_p90_j=float(np.percentile(windows,90)),
            common_optical_capacity_j=cap,rf_rho_energy=mean/(.3*cap)))
    result=dict(complete=True,selection=False,definition='V0 at 100 MJ, complete eclipse windows; no RF battery/load retuning',rows=rows)
    write_json(done,result);write_json(ROOT/'tables_r2/E4_RF_nominal_tightness.json',rows)
    return result
