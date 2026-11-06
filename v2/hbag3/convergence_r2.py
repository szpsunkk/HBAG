"""E9-prime selects the three tightest actual HBAG closed-loop snapshots."""
from dataclasses import replace,asdict
import numpy as np
from .model import Config,POWER
from .environment import ROOT,build_environment
from .matrix_v21 import policy_config
from .runner import seed_guard,write_json,save_npz,source_hashes
from .control import SwitchController,usable_catalogue
from .allocation import allocate,protect_bus
from .energy import step
from .optimization import solve_centralized,solve_distributed,feasible_recovery
from .solvers import solve_dual,solve_admm


def convergence_cell(args):
    seed,small,frozen=args
    rho=1.2 if small else 1.4
    base=Config(seed=seed,rho_energy=rho,capacity_j=frozen['capacity_by_rho'][str(rho)],flows=32 if small else frozen['flows'],
                planes=8 if small else 24,per_plane=8 if small else 24,load=1.2 if small else frozen['load'],orbits=3)
    c,_=policy_config(base,'HBAG',frozen);seed_guard(c,'test')
    out=ROOT/'results_r2/E9'/('M64' if small else 'M576')/('seed_%d'%seed)
    if (out/'summary.json').exists():
        import json
        return json.loads((out/'summary.json').read_text())
    env=build_environment(c,progress=lambda row:write_json(out/'status.json',row))
    ctl=SwitchController(c,'HBAG');e=np.full(env['shell'].size,c.initial_soc*c.capacity_j);best=[]
    for t in range(len(env['times'])):
        sw=ctl.decide(env,t,e);cat=usable_catalogue(env['catalogues'][t],sw)
        x,p,meta,info=allocate(env,t,cat,sw,ctl);x,q,v=protect_bus(env,t,e,cat,sw,x,p,meta)
        slack=np.divide(sw.raw_budget_w-q,np.maximum(sw.raw_budget_w,1e-9))
        tight=int(np.sum((slack<.01)&~sw.infeasible))
        best.append(dict(slot=t,tight_nodes=tight,problem=p,x=x.copy(),energy=e.copy(),power=q.copy(),slack=slack.copy()))
        best=sorted(best,key=lambda s:(-s['tight_nodes'],s['slot']))[:3]
        e=step(e,env['harvest'][t],c.base_w,q,env['dt'],c.battery).after_j
        ctl.observe(cat,x,p,sw,env['terminal']);ctl.price.commit(q,t)
    histories=[];snapshots=[];rates=[]
    for chosen in best:
        t=chosen['slot'];p=chosen['problem'];central=solve_centralized(p,tolerance=1e-10)
        reference=feasible_recovery(central['allocation'],p);value=p.objective(reference)
        slack=p.budgets-p.resource_matrix@reference.ravel();nl=len(p.budgets)-env['shell'].size
        snapshots.append(dict(slot=t,tight_nodes=chosen['tight_nodes'],central_objective=value,
            active_link_fraction=float(np.mean(slack[:nl]<1e-7)),active_node_fraction=float(np.mean(slack[nl:]<1e-7)),
            central_gap=central['certificate_gap'],problem_sha256=p.fingerprint()))
        arrays=dict(matrix_data=p.resource_matrix.data,matrix_indices=p.resource_matrix.indices,matrix_indptr=p.resource_matrix.indptr,
                    matrix_shape=np.array(p.resource_matrix.shape),budgets=p.budgets,demand=p.demand,mask=p.path_mask,linear_cost=p.linear_cost,
                    utility_weight=p.utility_weight,utility_scale=p.utility_scale,regularization=np.array(p.regularization),
                    central_allocation=reference,executed_allocation=chosen['x'],energy_j=chosen['energy'],budget_relative_slack=chosen['slack'])
        save_npz(out/('snapshot_%d.npz'%t),**arrays)
        for method in ['HBAG','DualDecomp','ADMM']:
            if method=='HBAG':
                solved=solve_distributed(p,max_iterations=10000,tolerance=0.,check_every=20);history=solved.history
            else:
                x,info=(solve_dual if method=='DualDecomp' else solve_admm)(p,10000,0.,True);history=info['history']
            series=[]
            for h in history:
                objective=h.get('primal_objective',h.get('objective'));gap=max(value-objective,0)/max(abs(value),1.)
                row=dict(seed=seed,slot=t,method=method,iteration=h['iteration'],relative_objective_gap=gap,
                         recovered_objective=objective,raw_violation=h['raw_violation'],kkt_residual=h.get('kkt_residual',h.get('kkt')),
                         logical_scalar_messages=2*p.resource_matrix.nnz*h['iteration'],switch_intent_bits=4*env['shell'].size)
                histories.append(row);series.append(row)
            reference_floor=max(1e-10,central['certificate_gap'])
            valid=[r for r in series if r['relative_objective_gap']>reference_floor and r['iteration']>=20]
            if len(valid)>=5:
                it=np.array([r['iteration'] for r in valid]);gap=np.array([r['relative_objective_gap'] for r in valid])
                rates.append(dict(slot=t,method=method,loglog_slope=float(np.polyfit(np.log(it),np.log(gap),1)[0]),
                                  geometric_ratio_per_iteration=float(np.exp(np.polyfit(it,np.log(gap),1)[0])),points=len(valid),reference_uncertainty_floor=reference_floor))
            else:rates.append(dict(slot=t,method=method,loglog_slope=None,geometric_ratio_per_iteration=None,points=len(valid),reference_uncertainty_floor=reference_floor,reason='Insufficient points above reference certificate uncertainty; do not invent a rate'))
    write_json(out/'convergence.json',histories)
    result=dict(complete=True,config=asdict(c),snapshots=snapshots,empirical_rates=rates,source_sha256=source_hashes(),
                selection_rule='3 largest count of feasible nodes with (Qbar-Q)/Qbar<1%; ties earliest slot',iterations=10000,
                snapshot_precision='float32 archive; central and convergence computed on the exact float64 frozen problem in memory')
    write_json(out/'summary.json',result);return result
