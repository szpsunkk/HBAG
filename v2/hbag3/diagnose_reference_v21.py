"""Validation-only numerical corpus: execute the unchanged CP policy and audit shadows.

No simulated performance result is reported here. A failed shadow cannot affect
the executed allocation, so its failure is recorded and the same CP trajectory
continues to expose later numerical cases. Exact float64 snapshots are numerical
regression fixtures, not formal experiment arrays.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,traceback
import numpy as np
from .model import Config
from .environment import ROOT,build_environment
from .runner import seed_guard,write_json,source_hashes,disk_guard
from .control import SwitchController,usable_catalogue
from .allocation import problem_for_switch,protect_bus
from .optimization import solve_distributed,solve_centralized
from .energy import step


def snapshot(problem,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,matrix_data=problem.resource_matrix.data,
        matrix_indices=problem.resource_matrix.indices,matrix_indptr=problem.resource_matrix.indptr,
        matrix_shape=problem.resource_matrix.shape,budgets=problem.budgets,demand=problem.demand,
        path_mask=problem.path_mask,linear_cost=problem.linear_cost,utility_weight=problem.utility_weight,
        utility_scale=problem.utility_scale,regularization=problem.regularization)


def replay(args):
    seed,rho,cal=args
    c=Config(seed=seed,orbits=1,rho_energy=rho,capacity_j=cal['capacity_by_rho'][str(rho)],load=cal['load'])
    seed_guard(c,'validation');disk_guard()
    out=ROOT/'results_r2/audit/reference_corpus_v21'/('seed_%d_rho_%g'%(seed,rho))
    env=build_environment(c,progress=lambda row:write_json(out/'status.json',row))
    ctl=SwitchController(c,'HBAG');energy=np.full(env['shell'].size,c.initial_soc*c.capacity_j);rows=[]
    for t in range(len(env['times'])):
        switch=ctl.decide(env,t,energy);cat=usable_catalogue(env['catalogues'][t],switch,c.routing)
        problem,meta=problem_for_switch(env,t,cat,switch,True)
        cp=solve_distributed(problem,max_iterations=c.max_iterations,tolerance=c.tolerance,check_every=20)
        row=dict(seed=seed,rho_energy=rho,slot=t,problem_fingerprint=problem.fingerprint())
        try:
            answer=solve_centralized(problem,tolerance=1e-9)
            row.update(status=answer['status'],certificate_gap=answer['certificate_gap'])
            capture=answer['certificate_gap']>1e-8
        except Exception:
            row.update(status='ERROR',traceback=traceback.format_exc());capture=True
        if capture:
            snapshot(problem,out/('slot_%03d.npz'%t));write_json(out/('slot_%03d.json'%t),row)
        rows.append(row)
        x,q,v=protect_bus(env,t,energy,cat,switch,cp.allocation,problem,meta)
        energy=step(energy,env['harvest'][t],c.base_w,q,env['dt'],c.battery).after_j
        ctl.observe(cat,x,problem,switch,env['terminal']);ctl.price.commit(q,t)
        write_json(out/'status.json',dict(phase='numerical_replay',slot=t+1,slots=len(env['times'])))
    record=dict(phase='validation_diagnostic',seed=seed,rho_energy=rho,complete=True,rows=rows,
        source_sha256=source_hashes(),performance_comparison=False,test_seeds_executed=[])
    write_json(out/'replay.json',record)
    return dict(seed=seed,rho_energy=rho,failures=[r['slot'] for r in rows if r['status']=='ERROR'],
                max_gap=max((r['certificate_gap'] for r in rows if 'certificate_gap' in r),default=0.))


def main():
    cal=json.loads((ROOT/'results_r2/G0_prime/calibration_load_0p65.json').read_text())
    with ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(replay,[(1007,1.4,cal),(1002,1.6,cal),(1003,1.6,cal)]):print(json.dumps(row),flush=True)


if __name__=='__main__':main()
