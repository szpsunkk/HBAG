"""R2 v2.1: G0-prime and E11-prime independently gate all test-seed execution."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
import argparse,datetime,itertools,json,time,traceback,csv
import numpy as np
from .environment import ROOT,build_environment
from .model import Config
from .plan_r2 import VERSION,TEST_SEEDS
from .runner import run_policy,write_json,disk_guard,with_battery,fingerprint,source_hashes

SEEDS=list(range(1000,1010))
RHOS=[.6,1.,1.2,1.4,1.6]
MAIN_RHOS=[1.2,1.4,1.6]
G0=ROOT/'results_r2/G0_prime'
E11=ROOT/'results_r2/E11_validation'


def now():return datetime.datetime.now().astimezone().isoformat()


def log(message):
    print(now()+' '+message,flush=True)
    with (ROOT/'PROGRESS.md').open('a') as f:f.write('\n- '+now()+'：'+message+'\n')


def issue(message):
    with (ROOT/'QUJESTIONS_R2').open('a') as f:f.write('\n- '+now()+'：'+message+'\n')
    log('QUJESTIONS_R2：'+message)


def label(value):return ('%g'%value).replace('.','p')


def map_jobs(function,jobs,workers):
    if not 1<=workers<=10:raise ValueError('workers must be 1–10')
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(function,j):j for j in jobs}
        for future in as_completed(pending):
            rows.append(future.result());disk_guard()
            print(now()+' completed '+repr(pending[future]),flush=True)
    return rows


def validation_grid():
    grid=[]
    for eta,warm,kappa in itertools.product([.03,.1,.3,1.],['zero','prev'],[.5,1.,2.]):
        grid.append(dict(method='HBAG',parameters=dict(dual_eta=eta,dual_warm=warm,kappa_off=kappa)))
    for lam,kappa in itertools.product([.02,.1,.5],[.5,1.,2.]):
        grid.append(dict(method='HBAG-xi',parameters=dict(lam=lam,kappa_off=kappa)))
    for method in ['Lyapunov','HBAG-Lyap']:
        for lam,theta in itertools.product([.1,.5,2.],[.3,.6]):
            grid.append(dict(method=method,parameters=dict(lyapunov_lambda=lam,lyapunov_theta=theta,kappa_off=1.)))
    for theta in [.75,.8,.9]:grid.append(dict(method='Static+Guard',parameters=dict(guard_soc=theta)))
    for theta in [.01,.05,.1]:grid.append(dict(method='Util-Sleep',parameters=dict(utilization_sleep_threshold=theta)))
    for i,row in enumerate(grid):row['label']='%s_%02d'%(row['method'],i)
    return grid


def selection_grid():
    return [row for row in validation_grid() if row['method']=='HBAG']


def previous_selection():
    transition=json.loads((ROOT/'results_r2/audit/v23_transition.json').read_text())
    return json.loads((ROOT/transition['prior_frozen_parameters']).read_text())


def diagnostic_job(args):
    """Development-only abundance and degraded-array checks; never selection."""
    load,capacities,seed,kind=args
    base=Config(seed=seed,load=load,orbits=3 if kind=='abundance' else 6,
                rho_energy=.6 if kind=='abundance' else 1.,
                capacity_j=capacities['0.6' if kind=='abundance' else '1.0'],
                solar_w=500. if kind=='abundance' else 275.,beta_deg=0. if kind=='abundance' else 60.)
    cell=G0/'diagnostics'/kind/('seed_%d'%seed)
    env=build_environment(base,progress=lambda row:write_json(cell/'environment_status.json',row))
    parameters=previous_selection()['selected_by_rho']['1.2']['HBAG']['parameters']
    return [run_policy(replace(base,**(parameters if method=='HBAG' else {})),method,cell/method,
                       env=env,phase='validation',storage='node') for method in ['V1','HBAG']]


def calibration_job(args):
    load,seed=args;c=Config(seed=seed,flows=288,load=load,orbits=1,capacity_j=100e6)
    return run_policy(c,'V0',G0/('calibration_load_'+label(load))/('seed_%d'%seed)/'V0')


def calibration_report(load):
    rows=[];draws=[];util=[]
    for seed in SEEDS:
        cell=G0/('calibration_load_'+label(load))/('seed_%d'%seed)
        row=json.loads((cell/'V0/summary.json').read_text());rows.append(row)
        draws.extend(x['battery_draw_j'] for x in json.loads((cell/'V0/eclipse_windows.json').read_text()))
        with np.load(cell/'inputs.npz') as inp,np.load(cell/'V0/raw.npz') as raw:
            cap=inp['capacity_bps'].astype(float)*raw['link_capture'].astype(float)
            rate=raw['link_rate_bps'].astype(float)
            util.append(np.divide(rate,cap,out=np.zeros_like(rate),where=cap>0)[raw['link_on']])
    util=np.concatenate(util);draws=np.asarray(draws)
    observations=sum(x['on_link_observations'] for x in rows)
    eobs=sum(x['eclipse_terminal_observations'] for x in rows)
    result=dict(load=load,flows=288,seeds=SEEDS,source_fingerprint=fingerprint(),plan_version=VERSION,
        on_link_utilization_mean=sum(x['on_link_utilization_mean']*x['on_link_observations'] for x in rows)/observations,
        on_link_utilization_p95=float(np.percentile(util,95)),on_link_fraction_above_1pct=float(np.mean(util>.01)),
        delivered_offered_ratio=sum(x['delivered_bits'] for x in rows)/sum(x['offered_bits'] for x in rows),
        variable_energy_fraction=sum(x['variable_energy_j'] for x in rows)/sum(x['terminal_energy_j'] for x in rows),
        eclipse_draw_mean_j=float(draws.mean()),eclipse_draw_p90_j=float(np.percentile(draws,90)),window_count=len(draws),
        capacity_by_rho={str(rho):float(draws.mean()/(.3*rho)) for rho in RHOS},
        mean_run_s=float(np.mean([x['runtime_s'] for x in rows])),
        max_ledger_residual_j=max(x['max_ledger_residual_j'] for x in rows),per_seed=rows)
    for name in ['eclipse_terminal_idle_fraction','eclipse_terminal_above_1pct','eclipse_terminal_above_10pct']:
        result[name]=sum(x[name]*x['eclipse_terminal_observations'] for x in rows)/eobs
    write_json(G0/('calibration_load_'+label(load)+'.json'),result)
    return result


def calibration_decision(util,delivery,load):
    if util<.05:return 'STOP_UTILIZATION'
    if delivery>=.90:return 'PASS'
    return 'RETRY_LOAD_0P5' if load==.65 else 'STOP_DELIVERY'


def stress_job(args):
    load,capacities,seed=args;base=Config(seed=seed,load=load,orbits=1)
    env=build_environment(base,progress=lambda row:write_json(G0/'stress'/('environment_%d.json'%seed),row))
    rows=[]
    for rho in [1.,1.2,1.4,1.6]:
        c=replace(base,capacity_j=capacities[str(rho)],rho_energy=rho)
        e=with_battery(env,c);cell=G0/'stress'/('rho_'+label(rho))/('seed_%d'%seed)
        for method in ['V0','Util-Sleep']:rows.append(run_policy(c,method,cell/method,env=e,storage='node'))
    return rows


def functional_job(args):
    load,capacities,seed=args;base=Config(seed=seed,load=load,orbits=1)
    env=build_environment(base,progress=lambda row:write_json(G0/'functional'/('environment_%d.json'%seed),row))
    rows=[]
    for rho in MAIN_RHOS:
        c=replace(base,capacity_j=capacities[str(rho)],rho_energy=rho)
        e=with_battery(env,c);cell=G0/'functional'/('rho_'+label(rho))/('seed_%d'%seed)
        for method in ['V1','HBAG']:rows.append(run_policy(c,method,cell/method,env=e,storage='node'))
    return rows


def selection_job(args):
    """One process constructs one two-orbit environment, reused by all 3x51 points."""
    load,capacities,seed=args;base=Config(seed=seed,load=load,orbits=2)
    env=build_environment(base,progress=lambda row:write_json(E11/('environment_%d.json'%seed),row))
    rows=[]
    for rho in MAIN_RHOS:
        c=replace(base,capacity_j=capacities[str(rho)],rho_energy=rho)
        e=with_battery(env,c);cell=E11/('rho_'+label(rho))/('seed_%d'%seed)
        for i,item in enumerate(selection_grid()):
            chosen=replace(c,**item['parameters'])
            result=run_policy(chosen,item['method'],cell/item['label'],env=e,storage='selection')
            rows.append(dict(result,grid_label=item['label']))
            write_json(E11/('status_%d.json'%seed),dict(seed=seed,rho_energy=rho,grid_completed=i+1,grid_total=24,method=item['method']))
    return rows


def select_parameters(rows,grid=None):
    """One rho at a time. Safe on every seed, then throughput, then energy."""
    candidates=[]
    grid=validation_grid() if grid is None else grid
    for item in grid:
        group=[r for r in rows if r.get('grid_label')==item['label']]
        if len(group)!=10:raise RuntimeError('Incomplete validation grid '+item['label'])
        candidates.append(dict(item,n=10,eligible=all(r['esr']==1 for r in group),
            mean_esr=float(np.mean([r['esr'] for r in group])),min_esr=min(r['esr'] for r in group),
            mean_delivered_bits=float(np.mean([r['delivered_bits'] for r in group])),
            mean_terminal_energy_j=float(np.mean([r['terminal_energy_j'] for r in group]))))
    selected={}
    for method in dict.fromkeys(item['method'] for item in grid):
        choices=[r for r in candidates if r['method']==method];eligible=[r for r in choices if r['eligible']]
        order=(lambda r:(-r['mean_delivered_bits'],r['mean_terminal_energy_j'])) if eligible else (lambda r:(-r['mean_esr'],-r['mean_delivered_bits'],r['mean_terminal_energy_j']))
        selected[method]=sorted(eligible or choices,key=order)[0]
    return selected,candidates


def abundance_job(args):
    load,capacities,selected,seed=args
    base=Config(seed=seed,load=load,orbits=2,capacity_j=capacities['0.6'],rho_energy=.6)
    env=build_environment(base,progress=lambda row:write_json(E11/'abundance'/('environment_%d.json'%seed),row))
    cell=E11/'abundance'/('seed_%d'%seed);rows=[]
    for method in ['V0','V1']+list(selected)+['V2']:
        parameters=selected['HBAG']['parameters'] if method=='V2' else selected.get(method,{}).get('parameters',{})
        c=replace(base,**parameters);rows.append(run_policy(c,method,cell/method,env=env,storage='selection'))
    return rows


def gate_text(state,calibrations,detail=None):
    lines=['# R2 v2.3：G0′ / E11′','', '**状态：'+state+'。** 测试种子4000–4019在G0′与E11′均为PASS前保持锁定。','',
      '共同路径K=4，参考容量采用同一公共候选集；按源城市独立MMPP。流数固定288，仅交付率不达标时允许load=0.5。','',
      '| load | ON利用率均值 | P95 | >1% | Delivered/Offered | 增量能耗占比 | 日食取能均值 J | P90 J |',
      '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in calibrations:
        lines.append('|%g|%.4f%%|%.4f%%|%.4f%%|%.4f%%|%.4f%%|%.6f|%.6f|'%(c['load'],100*c['on_link_utilization_mean'],100*c['on_link_utilization_p95'],100*c['on_link_fraction_above_1pct'],100*c['delivered_offered_ratio'],100*c['variable_energy_fraction'],c['eclipse_draw_mean_j'],c['eclipse_draw_p90_j']))
    if calibrations:
        c=calibrations[-1]
        lines+=['','| 种子 | λ*_new (Gbit/s) | λ*_R1 (Gbit/s) | 比值 | 交付率 |','|---:|---:|---:|---:|---:|']
        for r in c['per_seed']:
            lines.append('|%d|%.9f|%.9f|%.6f|%.4f%%|'%(r['seed'],r['reference_scale_bps']/1e9,r['reference_r1_bps']/1e9,r['reference_new_over_r1'],100*r['delivered_offered_ratio']))
        lines+=['','日食星终端×隙：空闲<0.1%%为%.4f%%，>1%%为%.4f%%，>10%%为%.4f%%。完整日食窗口数%d。'%(100*c['eclipse_terminal_idle_fraction'],100*c['eclipse_terminal_above_1pct'],100*c['eclipse_terminal_above_10pct'],c['window_count']),
            '', '| ρ_E | Emax (J) |','|---:|---:|']
        lines+=['|%s|%.6f|'%(r,e) for r,e in c['capacity_by_rho'].items()]
    stress=G0/'stress.json'
    if stress.exists():
        rows=json.loads(stress.read_text());lines+=['','| ρ_E | 方法 | 平均ESR | 最小ESR |','|---:|---|---:|---:|']
        for rho,method in itertools.product([1.,1.2,1.4,1.6],['V0','Util-Sleep']):
            group=[r for r in rows if r['config']['rho_energy']==rho and r['method']==method]
            lines.append('|%g|%s|%.6f%%|%.6f%%|'%(rho,method,100*np.mean([r['esr'] for r in group]),100*min(r['esr'] for r in group)))
    if detail:lines+=['',detail]
    (ROOT/'GATE_R2.md').write_text('\n'.join(lines)+'\n')


def stop_gate(reason,calibrations):
    state=dict(state='STOP',plan_version=VERSION,reason=reason,time=now(),test_seeds_executed=[],source_fingerprint=fingerprint())
    write_json(G0/'gate.json',state);gate_text('G0′ STOP',calibrations,reason)
    issue(reason+' 未开始E11′或正式测试；不更改未授权的模型或参数。')
    return state


def save_grid_tables(rows):
    write_json(ROOT/'tables_r2/E11_validation_grid.json',rows)
    with (ROOT/'tables_r2/E11_validation_grid.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow(dict(row,parameters=json.dumps(row['parameters'],sort_keys=True)))
    tex=['% v2.3 HBAG 3 x 24 x 10; unchanged-method v2.2 grids explicitly marked reused in JSON/CSV','\\begin{tabular}{rlrrr}','\\toprule',
         '$\\rho_E$ & Configuration & ESR & Delivered (Tbit) & Terminal (MJ) \\\\','\\midrule']
    for r in rows:
        tex.append('%g & %s & %.6f & %.6f & %.6f \\\\'%(r['rho_energy'],r['label'].replace('_','\\_'),r['mean_esr'],r['mean_delivered_bits']/1e12,r['mean_terminal_energy_j']/1e6))
    tex+=['\\bottomrule','\\end{tabular}'];(ROOT/'tables_r2/E11_validation_grid.tex').write_text('\n'.join(tex)+'\n')


def execute(workers=9):
    G0.mkdir(parents=True,exist_ok=True);disk_guard();calibrations=[]
    write_json(ROOT/'configs_r2/G0_prime.json',dict(plan_version=VERSION,validation_seeds=SEEDS,
        test_seeds=list(TEST_SEEDS),workers=workers,loads=[.65,.5],flows=288,
        configuration=asdict(Config(orbits=1)),source_sha256=source_hashes(),selection_grid=selection_grid(),selection_rhos=MAIN_RHOS))
    existing=G0/'gate.json'
    if existing.exists():
        state=json.loads(existing.read_text())
        if state['source_fingerprint']!=fingerprint():raise RuntimeError('Existing G0-prime code differs; archive explicitly before rerun')
        calibrations=[json.loads(p.read_text()) for p in sorted(G0.glob('calibration_load_*.json'),reverse=True)]
        if state['state']=='STOP':return state
        cal=calibrations[-1]
    else:
        import hashlib
        transition=json.loads((ROOT/'results_r2/audit/v23_transition.json').read_text())
        for name,entry in transition['reused_calibration_and_stress'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=entry['sha256']:
                raise RuntimeError('Reused v2.1 evidence changed: '+name)
        verification=ROOT/'results_r2/audit/v23_unchanged_bitwise.json'
        if not verification.exists():raise RuntimeError('Unchanged-method byte verification required before v2.3 G0 functional rerun')
        audit=json.loads(verification.read_text())
        if not audit['passed'] or audit['source_fingerprint']!=fingerprint():
            raise RuntimeError('Unchanged-method equivalence audit does not match current v2.3 source')
        calibrations=[json.loads(p.read_text()) for p in sorted(G0.glob('calibration_load_*.json'),reverse=True)]
        cal=calibrations[-1]
        if calibration_decision(cal['on_link_utilization_mean'],cal['delivered_offered_ratio'],cal['load'])!='PASS':
            raise RuntimeError('Cannot reuse a failed load calibration')
        stress=json.loads((G0/'stress.json').read_text())
        main=[r for r in stress if r['method']=='V0' and r['config']['rho_energy']==1.2]
        if len(main)!=10 or not all(r['esr']<1 for r in main):raise RuntimeError('Cannot reuse a failed stress check')
        log('v2.3沿用既有标定、紧张度及E_max；七个未改方法逐比特核验通过。重跑三档×10验证种子×V1/HBAG功能检验，无影子求解。')
        functional=map_jobs(functional_job,[(cal['load'],cal['capacity_by_rho'],seed) for seed in SEEDS],workers)
        rows=[r for group in functional for r in group];write_json(G0/'functional.json',rows)
        if any(r['certifiable_esr']!=1 or r['max_ledger_residual_j']>1e-6 or r['max_constraint_violation']>1e-7 or r['zero_price_a2_events']!=0 or r['max_a2_per_node']>1 for r in rows):
            return stop_gate('v2.3 V1/HBAG功能检验未通过；保留全部证据，不进入选参或测试。',calibrations)
        if any(r['shadow_evaluation_enabled'] or r['shadow_success_slots'] or r['shadow_failure_slots'] for r in rows):
            raise RuntimeError('Unexpected shadow execution during G0')
        differences=[]
        for rho in MAIN_RHOS:
            paired=[]
            for seed in SEEDS:
                v1=next(r for r in rows if r['seed']==seed and r['method']=='V1' and r['config']['rho_energy']==rho)
                hbag=next(r for r in rows if r['seed']==seed and r['method']=='HBAG' and r['config']['rho_energy']==rho)
                paired.append(dict(seed=seed,difference_bits=hbag['delivered_bits']-v1['delivered_bits'],
                    relative_difference=hbag['delivered_bits']/v1['delivered_bits']-1))
            differences.append(dict(rho_energy=rho,paired=paired,mean_difference_bits=float(np.mean([p['difference_bits'] for p in paired])),
                mean_relative_difference=float(np.mean([p['relative_difference'] for p in paired]))))
        write_json(G0/'validation_development_comparison.json',dict(plan_version=VERSION,development_only=True,
            include_in_manuscript=False,source_fingerprint=fingerprint(),rows=differences))
        diagnostic=map_jobs(diagnostic_job,[(cal['load'],cal['capacity_by_rho'],seed,kind) for kind in ['abundance','degraded_solar'] for seed in SEEDS],workers)
        write_json(G0/'diagnostics.json',dict(development_only=True,include_in_manuscript=False,participates_in_selection=False,source_fingerprint=fingerprint(),rows=[r for group in diagnostic for r in group]))
        state=dict(state='PASS',plan_version=VERSION,source_fingerprint=fingerprint(),time=now(),load=cal['load'],flows=288,
            capacity_by_rho=cal['capacity_by_rho'],test_seeds_executed=[],reused_calibration_source=cal['source_fingerprint'],
            reuse_audit='results_r2/audit/v23_transition.json',unchanged_bitwise_audit=str(verification.relative_to(ROOT)))
        write_json(existing,state);gate_text('v2.3功能检验PASS，E11′待完成，测试仍锁定',calibrations)
        log('v2.3 G0′功能检验PASS；重新开始HBAG E11′，24点网格不变；其余方法参数沿用v2.2，无影子求解，测试种子仍锁定。')
        from .storage_budget import projection
        projection()
        from .report_r2 import main as report
        report()
    frozen=E11/'frozen.json'
    if frozen.exists():
        final=json.loads(frozen.read_text())
        if final['source_fingerprint']!=fingerprint():raise RuntimeError('Frozen E11-prime belongs to different source')
        return final
    selection=map_jobs(selection_job,[(cal['load'],cal['capacity_by_rho'],s) for s in SEEDS],workers)
    rows=[r for group in selection for r in group];write_json(E11/'all_summaries.json',rows)
    selected=previous_selection()['selected_by_rho'];tables=[]
    for rho in MAIN_RHOS:
        chosen,grid=select_parameters([r for r in rows if r['config']['rho_energy']==rho],selection_grid())
        selected[str(rho)].update(chosen);tables.extend(dict(row,rho_energy=rho,source_plan_version=VERSION,reused=False) for row in grid)
    transition=json.loads((ROOT/'results_r2/audit/v23_transition.json').read_text())
    old_grid=json.loads((ROOT/transition['archive']/'tables_r2/E11_validation_grid.json').read_text())
    tables.extend(dict(row,source_plan_version='v2.2',reused=True) for row in old_grid if row['method']!='HBAG')
    save_grid_tables(tables)
    abundance=map_jobs(abundance_job,[(cal['load'],cal['capacity_by_rho'],selected['1.2'],s) for s in SEEDS],workers)
    write_json(E11/'abundance.json',[r for group in abundance for r in group])
    final=dict(state='PASS',plan_version=VERSION,source_fingerprint=fingerprint(),time=now(),load=cal['load'],flows=288,
               capacity_by_rho=cal['capacity_by_rho'],selected_by_rho=selected,validation_seeds=SEEDS,test_seeds=list(TEST_SEEDS),test_seeds_executed=[],
               selection_scope='HBAG 24 points x 3 rhos x 10 validation seeds; all other parameters reused from verified v2.2',reused_selection=transition['prior_frozen_parameters'])
    write_json(frozen,final);gate_text('G0′ PASS / E11′ PASS',calibrations,'冻结参数：\n\n```json\n'+json.dumps(final,indent=2)+'\n```')
    log('G0′与E11′均PASS，三档参数冻结。开始按P0→P1→P2自动继续。')
    return final


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=9);args=parser.parse_args()
    try:print(json.dumps(execute(args.workers),ensure_ascii=False,indent=2),flush=True)
    except Exception:
        detail=traceback.format_exc();write_json(G0/'error.json',dict(time=now(),traceback=detail));issue(detail);raise


if __name__=='__main__':main()
