"""Seed-level bootstrap summaries and pre-registered paired effects; no slot pseudoreplication."""
from pathlib import Path
import json,csv
import numpy as np
from .environment import ROOT
from .runner import write_json
from .statistics import paired,holm
from .plan_r2 import TEST_SEEDS


METRICS=['delivered_bits','fvr','ee_isl_bits_j','ee_sys_bits_j','esr','min_soc','violation_depth_j',
    'terminal_energy_j','system_energy_j','terminal_on_fraction','sleep_terminal_events','wake_terminal_events',
    'flips_per_satellite_per_orbit','budget_forced_node_slots','price_sleep_link_events','acquisition_energy_j',
    'violation_duration_node_s','brownout_events','on_link_utilization_mean','on_link_utilization_p95',
    'on_link_fraction_above_1pct','variable_energy_fraction','delivered_weighted_hops','delivered_propagation_delay_ms',
    'disconnected_commodity_slots','disconnected_demand_fraction','zero_price_a2_events','hole_node_slots',
    'hole_access_demand_fraction','relay_sun_fraction','spill_harvest_fraction','certificate_dual_positive_fraction',
    'mean_iterations','mean_slot_s','mean_relative_gap','mean_shadow_relative_gap']
METRICS+=['safe_mode_node_slots','brownout_node_slots','max_link_utilization','hottest_satellite_w',
    'mean_messages_per_satellite','mean_bytes_per_satellite','peak_rss_bytes','bound_measured_max_j','bound_uniform_j','bound_max_excess_j',
    'mean_shadow_certificate_gap','max_shadow_certificate_gap','max_shadow_numerical_attempts','shadow_reference_target_miss_slots']
METRICS+=['shadow_failure_slots','shadow_success_slots','central_degraded_slots',
    'potential_load_correlation_mean','potential_load_correlation_valid_windows','potential_load_correlation_total_windows']
METRICS+=['sun_terminal_on_fraction','certificate_dual_reported_fraction','certificate_dual_reporting_threshold',
    'mean_decide_s','mean_usable_catalogue_s','mean_allocate_s','mean_protect_bus_s',
    'path_precomputation_s','path_precomputation_mean_slot_s','path_precomputation_mean_satellite_slot_s',
    'precomputed_slots','configured_parallel_workers','concurrent_worker_processes','offline_solve_s','solver_calls']
EFFECT_METRICS=['delivered_bits','esr','min_soc','violation_depth_j','terminal_energy_j','terminal_on_fraction']


def summaries(experiment):
    root=ROOT/'results_r2'/experiment;groups={}
    for p in root.glob('*/seed_*/*/summary.json'):
        row=json.loads(p.read_text())
        for key in ['orbit_max_dod','orbit_end_min_soc','orbit_end_mean_soc','eclipse_mean_on_terminals_by_third','eclipse_mean_on_terminals_by_sixth',
                    'orbit_sun_terminal_on_fraction','orbit_spill_harvest_fraction']:
            for i,v in enumerate(row.get(key,[])):row[key+'_'+str(i+1)]=v
        for side,data in row.get('price_reset_by_illumination',{}).items():
            for reason,value in data.items():row['price_reset_'+side+'_'+reason]=value
        for key in ['counterfactual_distribution','support_distribution']:
            for side,data in row.get(key,{}).items():
                row[key+'_'+side+'_mean']=data['mean']
                for quantile,value in zip([0,25,50,75,90,95,99,100],data['quantiles']):row[key+'_'+side+'_p'+str(quantile)]=value
        groups.setdefault((p.parents[2].name,p.parent.name),[]).append(row)
    table=[]
    for (cell,method),runs in sorted(groups.items()):
        runs=sorted(runs,key=lambda r:r['seed'])
        expected=list(TEST_SEEDS[:5] if experiment=='E10' and cell=='M_5000' else TEST_SEEDS)
        if [r['seed'] for r in runs]!=expected:
            raise RuntimeError('Incomplete independent seed set for '+cell+'/'+method)
        n=len(expected);rng=np.random.default_rng(1729);index=rng.integers(0,n,size=(10000,n))
        extra=[k for k,v in runs[0].items() if k.startswith(('orbit_','price_reset_','eclipse_mean_on_terminals_by_third_','eclipse_mean_on_terminals_by_sixth_','counterfactual_distribution_','support_distribution_')) and isinstance(v,(int,float))]
        for metric in METRICS+extra:
            if any(r.get(metric) is None for r in runs):continue
            values=np.array([r[metric] for r in runs],float);boot=values[index].mean(axis=1)
            table.append(dict(cell=cell,method=method,metric=metric,n=n,mean=float(values.mean()),
                              ci95_low=float(np.percentile(boot,2.5)),ci95_high=float(np.percentile(boot,97.5))))
    write_json(ROOT/('tables_r2/'+experiment+'_summary.json'),table)
    if table:
        with (ROOT/('tables_r2/'+experiment+'_summary.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        tex=['% Automatically generated seed bootstrap intervals','\\begin{tabular}{lllrrr}','\\toprule',
             'Cell & Method & Metric & Mean & 95\\% low & 95\\% high \\\\','\\midrule']
        for r in table:
            if experiment=='E8' and r['metric']=='esr':
                values=[esr_percent(r[k]).replace('<',r'$<$') for k in ['mean','ci95_low','ci95_high']]
                tex.append('%s & %s & ESR (\\%%) & %s & %s & %s \\\\'%(r['cell'].replace('_','\\_'),r['method'],*values))
            else:tex.append('%s & %s & %s & %.6g & %.6g & %.6g \\\\'%(r['cell'].replace('_','\\_'),r['method'].replace('_','\\_'),r['metric'].replace('_','\\_'),r['mean'],r['ci95_low'],r['ci95_high']))
        tex+=['\\bottomrule','\\end{tabular}'];(ROOT/('tables_r2/'+experiment+'_summary.tex')).write_text('\n'.join(tex)+'\n')
    if experiment in ('E5','MPC'):paired_experiment(experiment,groups)
    return table


def esr_percent(value):
    text='%.3f'%(100*value)
    return '<100.000' if value<1 and text=='100.000' else text


def paired_experiment(experiment,groups):
    left,right=('HBAG','V1') if experiment=='E5' else ('MPC-H-relaxed','HBAG')
    rows=[]
    for cell in sorted({key[0] for key in groups}):
        a=sorted(groups[(cell,left)],key=lambda r:r['seed']);b=sorted(groups[(cell,right)],key=lambda r:r['seed'])
        if [r['seed'] for r in a]!=list(TEST_SEEDS) or [r['seed'] for r in b]!=list(TEST_SEEDS):raise RuntimeError('Unpaired new test seeds')
        for metric in ['delivered_bits','esr']:
            rows.append(dict(cell=cell,contrast=left+'−'+right,metric=metric,**paired([r[metric] for r in a],[r[metric] for r in b])))
    for row,p in zip(rows,holm([r['p_wilcoxon'] for r in rows])):row['p_holm']=float(p)
    write_json(ROOT/('tables_r2/'+experiment+'_paired.json'),rows)
    with (ROOT/('tables_r2/'+experiment+'_paired.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def ablation():
    effects=[]
    for rho in [1.2,1.4,1.6]:
        root=ROOT/'results_r2/E2'/('rho_'+str(rho));runs={}
        for method in ['V0','V1','V2','HBAG','HBAG-xi','HBAG-Lyap']:
            runs[method]=[json.loads((root/('seed_%d'%s)/method/'summary.json').read_text()) for s in range(4000,4020)]
        for metric in EFFECT_METRICS:
            vectors={k:np.array([r[metric] for r in rows]) for k,rows in runs.items()}
            comparisons=[('V1−V0',vectors['V1'],vectors['V0']),('V2−V0',vectors['V2'],vectors['V0']),
                ('HBAG−V1',vectors['HBAG'],vectors['V1']),('HBAG−V2',vectors['HBAG'],vectors['V2']),
                ('interaction',vectors['HBAG']-vectors['V1']-vectors['V2']+vectors['V0'],np.zeros(20)),
                ('HBAG−HBAG-xi',vectors['HBAG'],vectors['HBAG-xi']),('HBAG−HBAG-Lyap',vectors['HBAG'],vectors['HBAG-Lyap']),
                ('HBAG-xi−V1',vectors['HBAG-xi'],vectors['V1']),('HBAG-Lyap−V1',vectors['HBAG-Lyap'],vectors['V1']),
                ('HBAG-Lyap−HBAG-xi',vectors['HBAG-Lyap'],vectors['HBAG-xi'])]
            for name,a,b in comparisons:effects.append(dict(rho_energy=rho,metric=metric,contrast=name,**paired(a,b)))
    adjusted=holm([r['p_wilcoxon'] for r in effects])
    for r,p in zip(effects,adjusted):r['p_holm']=float(p)
    write_json(ROOT/'tables_r2/E3_ablation.json',effects)
    with (ROOT/'tables_r2/E3_ablation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)
    tex=['% One Holm family over all three tightness levels, outcomes and contrasts',
         '\\begin{tabular}{rllrrrr}','\\toprule',
         '$\\rho_E$ & Outcome & Contrast & Difference & 95\\% low & 95\\% high & Holm $p$ \\\\','\\midrule']
    for r in effects:
        tex.append('%g & %s & %s & %.6g & %.6g & %.6g & %.6g \\\\'%(r['rho_energy'],r['metric'].replace('_','\\_'),r['contrast'].replace('−','$-$'),r['mean_difference'],r['ci95_low'],r['ci95_high'],r['p_holm']))
    tex+=['\\bottomrule','\\end{tabular}'];(ROOT/'tables_r2/E3_ablation.tex').write_text('\n'.join(tex)+'\n')
    write_json(ROOT/'results_r2/E3/summary.json',dict(complete=True,n=20,bootstrap=10000,holm_family_size=len(effects),effects=effects,
                source='Existing E2 trajectories only; no additional simulation or parameter selection'))
    return effects


def convergence_summary():
    """Average the three selected snapshots within each seed before inference."""
    table=[];rates=[];rng=np.random.default_rng(1729)
    weights=np.array([np.bincount(r,minlength=20)/20. for r in rng.integers(0,20,(10000,20))])
    for size in ['M64','M576']:
        histories=[];estimates=[]
        for seed in range(4000,4020):
            out=ROOT/'results_r2/E9'/size/('seed_%d'%seed)
            histories.append(json.loads((out/'convergence.json').read_text()))
            estimates.append(json.loads((out/'summary.json').read_text())['empirical_rates'])
        for method in ['HBAG','DualDecomp','ADMM']:
            for metric in ['loglog_slope','geometric_ratio_per_iteration']:
                per_seed=[]
                for group in estimates:
                    valid=[r[metric] for r in group if r['method']==method and r[metric] is not None]
                    per_seed.append(float(np.mean(valid)) if valid else None)
                if all(v is not None for v in per_seed):
                    v=np.array(per_seed);boot=weights@v
                    rates.append(dict(size=size,method=method,metric=metric,n=20,mean=float(v.mean()),ci95_low=float(np.percentile(boot,2.5)),ci95_high=float(np.percentile(boot,97.5))))
                else:rates.append(dict(size=size,method=method,metric=metric,n=sum(v is not None for v in per_seed),mean=None,reason='Insufficient nonzero gaps for all 20 seeds; no complete-sample rate claimed'))
            iterations=sorted({r['iteration'] for rows in histories for r in rows if r['method']==method})
            for metric in ['relative_objective_gap','raw_violation','recovered_objective','kkt_residual','logical_scalar_messages']:
                values=[]
                for group in histories:
                    by_slot={}
                    for r in group:
                        if r['method']==method:by_slot.setdefault(r['slot'],[]).append(r)
                    snapshots=[]
                    for rows in by_slot.values():
                        rows=sorted(rows,key=lambda r:r['iteration']);it=np.array([r['iteration'] for r in rows])
                        snapshots.append([rows[max(0,int(np.searchsorted(it,i,side='right')-1))][metric] for i in iterations])
                    values.append(np.mean(snapshots,axis=0))
                values=np.array(values);boot=weights@values
                mean=values.mean(axis=0);lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
                table.extend(dict(size=size,method=method,metric=metric,iteration=int(it),n=20,mean=float(a),ci95_low=float(l),ci95_high=float(h)) for it,a,l,h in zip(iterations,mean,lo,hi))
    write_json(ROOT/'tables_r2/E9_convergence.json',table);write_json(ROOT/'tables_r2/E9_rates.json',rates)
    for name,rows in [('E9_convergence',table),('E9_rates',rates)]:
        fields=sorted({k for r in rows for k in r})
        with (ROOT/('tables_r2/'+name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return table,rates
