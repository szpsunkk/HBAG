"""Automatic publication rows and transparent coverage for the v24 pipeline."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon,spearmanr,kendalltau
from .runner import write_json
from .statistics import holm
from .v24_runtime import ROOT,OUT,TABLES


def read(path):return json.loads(Path(path).read_text())


def ci(values):
    a=np.asarray(values,float)
    if not len(a):return dict(mean=None,ci95_low=None,ci95_high=None)
    indices=np.random.default_rng(1729).integers(0,len(a),(10000,len(a)))
    low,high=np.percentile(a[indices].mean(axis=1),[2.5,97.5])
    return dict(mean=float(a.mean()),ci95_low=float(low),ci95_high=float(high))


def table(name,rows):
    TABLES.mkdir(parents=True,exist_ok=True)
    write_json(TABLES/(name+'.json'),rows)
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (TABLES/(name+'.csv')).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,keys);writer.writeheader();writer.writerows(rows)
    def tex(v):
        if v is None:return '--'
        return str(v).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    lines=[r'\begin{tabular}{'+'l'*len(keys)+'}', ' & '.join(tex(k) for k in keys)+r' \\',r'\hline']
    lines+=[' & '.join(tex(row.get(k)) for k in keys)+r' \\' for row in rows]
    lines+=[r'\end{tabular}']
    (TABLES/(name+'.tex')).write_text('\n'.join(lines)+'\n')


def rf_rows():
    rows=[];paired=[];per_seed=[]
    for stage,seeds in [('E4_prime',list(range(4000,4020))),('E4_double_prime',list(range(5000,5020)))]:
        for level in (-300.,-10.,0.):
            data={}
            for method in ('V0','Static+Guard','Lyapunov','V1','HBAG'):
                group=[]
                for seed in seeds:
                    path=OUT/stage/('IN_%g'%level)/('seed_%d'%seed)/method/'summary.json'
                    if path.exists():group.append(read(path))
                data[method]=group
                if len(group)!=20:
                    rows.append(dict(experiment=stage,interference_db=level,method=method,n=len(group),status='unavailable/incomplete'))
                    continue
                stat=ci([g['delivered_bits']/1e12 for g in group])
                rows.append(dict(experiment=stage,interference_db=level,method=method,n=20,status='complete',
                    delivered_tbit=stat['mean'],ci95_low=stat['ci95_low'],ci95_high=stat['ci95_high'],
                    esr=float(np.mean([g['esr'] for g in group])),min_seed_esr=min(g['esr'] for g in group)))
                per_seed.extend(dict(experiment=stage,interference_db=level,method=method,seed=g['seed'],
                    delivered_tbit=g['delivered_bits']/1e12,esr=g['esr']) for g in group)
            if len(data['HBAG'])==len(data['V1'])==20:
                h=np.array([g['delivered_bits'] for g in data['HBAG']]);v=np.array([g['delivered_bits'] for g in data['V1']])
                percent=100*(h-v)/v;stat=ci(percent)
                p=1. if np.all(h==v) else float(wilcoxon(h-v,alternative='two-sided',zero_method='wilcox',method='auto').pvalue)
                indices=np.random.default_rng(1729).integers(0,20,(10000,20))
                ratio=100*(h[indices].mean(axis=1)/v[indices].mean(axis=1)-1)
                paired.append(dict(experiment=stage,interference_db=level,n=20,paired_percent=stat['mean'],
                    ci95_low=stat['ci95_low'],ci95_high=stat['ci95_high'],p_wilcoxon=p,
                    ratio_of_means_percent=float(100*(h.mean()/v.mean()-1)),
                    ratio_ci95_low=float(np.percentile(ratio,2.5)),ratio_ci95_high=float(np.percentile(ratio,97.5))))
    # A predeclared family of six HBAG-vs-V1 tests, three per RF experiment.
    corrected=holm([p['p_wilcoxon'] for p in paired]+[1.]*(6-len(paired)))
    for row,p in zip(paired,corrected):row['p_holm']=float(p)
    table('RF_summary',rows);table('RF_paired',paired);table('RF_per_seed',per_seed)
    return rows,paired


def delay_rows():
    rows=[];checks=[]
    for folder in sorted((OUT/'E8').glob('*')):
        groups={}
        for path in sorted(folder.glob('seed_*/*/summary.json')):
            r=read(path);groups.setdefault(r['method'],[]).append(r)
            check=path.parent/'regression.json'
            if check.exists():checks.append(read(check))
        for method,group in groups.items():
            if len(group)!=20:continue
            changes=[]
            for g in group:
                zero=OUT/'E8'/('dt_%g_delay_0'%g['dt_s'])/('seed_%d'%g['seed'])/method/'summary.json'
                # Acquisition-sensitivity rows use the matching zero-delay
                # reference only when acquisition_s matches (the main table is 20s).
                if zero.exists() and read(zero)['config']['acquisition_s']==g['config']['acquisition_s']:
                    changes.append(100*(g['delivered_bits']/read(zero)['delivered_bits']-1))
            d=ci(changes);energy=ci([g['bound_measured_max_j'] for g in group])
            rows.append(dict(cell=folder.name,dt_s=group[0]['dt_s'],delay_s=group[0]['delay_s'],method=method,n=20,
                delivered_change_percent=d['mean'],change_ci95_low=d['ci95_low'],change_ci95_high=d['ci95_high'],
                esr=float(np.mean([g['esr'] for g in group])),min_seed_esr=min(g['esr'] for g in group),
                measured_energy_change_mean_j=energy['mean'],energy_ci95_low=energy['ci95_low'],energy_ci95_high=energy['ci95_high'],
                measured_energy_change_max_j=max(g['bound_measured_max_j'] for g in group)))
    table('delay_summary',rows);table('delay_dt15',[r for r in rows if r['dt_s']==15.]);table('delay_regression',checks)
    return rows,checks


def distribution(values):
    a=np.asarray(values,float)
    if not len(a):return dict(n=0)
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std()),
                **{'q%g'%q:float(np.percentile(a,q)) for q in [0,1,5,25,50,75,95,99,100]})


def correlation_stats(rows,variant):
    if not rows:return dict(n=0)
    predicted=np.array([r[variant] for r in rows]);true=np.array([r['dPhi'] for r in rows])
    error=np.array([r['numerical_error_bound'] for r in rows]);resolved=true>error
    rank_ok=len(rows)>1 and np.ptp(predicted)>0 and np.ptp(true)>0
    ratio=predicted[resolved]/true[resolved]
    return dict(n=len(rows),spearman=float(spearmanr(predicted,true).correlation) if rank_ok else None,
        kendall=float(kendalltau(predicted,true).correlation) if rank_ok else None,
        predicted_mean=float(predicted.mean()),true_loss_mean=float(true.mean()),
        ratio=distribution(ratio),resolved_positive_losses=int(resolved.sum()),
        numerically_unresolved_or_zero_losses=int((~resolved).sum()),negative_within_error_count=int((true<0).sum()),
        under_half_share=float(np.mean(predicted[resolved]<.5*true[resolved])) if resolved.any() else None,
        under_half_share_all=float(np.mean(predicted<.5*true)),
        utility_only_loss=distribution([r['utility_loss'] for r in rows]))


def sleep_rows():
    rows=[];actual=[];joint=[]
    for folder in sorted((OUT/'E14').glob('seed_*')):
        for name,target in [('links.json',rows),('actual_a2.json',actual),('joint_a2.json',joint)]:
            path=folder/name
            if path.exists():target.extend(read(path))
    result={}
    for variant in ('predicted','greedy'):
        result[variant]=dict(all=correlation_stats(rows,variant),
            eclipsed_endpoint=correlation_stats([r for r in rows if r['endpoint_eclipsed']],variant),
            actual_a2=correlation_stats(actual,variant),
            by_seed={str(s):correlation_stats([r for r in rows if r['seed']==s],variant) for s in range(4000,4010)})
    result['joint']=dict(n=len(joint),sum_individual=distribution([r['sum_individual_loss'] for r in joint]),
        joint_loss=distribution([r['dPhi'] for r in joint]),
        joint_minus_sum=distribution([r['dPhi']-r['sum_individual_loss'] for r in joint]),
        joint_over_sum=distribution([r['dPhi']/r['sum_individual_loss'] for r in joint if r['sum_individual_loss']>r['numerical_error_bound']]),
        predicted_sum=distribution([r['predicted_sum'] for r in joint]),greedy_sum=distribution([r['greedy_sum'] for r in joint]))
    write_json(TABLES/'sleep_diagnostic.json',result)
    table('sleep_links',rows);table('sleep_actual_a2',actual);table('sleep_joint_a2',joint)
    return result


def f(value,digits=4):return 'unavailable' if value is None else ('%.*g'%(digits,value))


def report(failure=None):
    TABLES.mkdir(parents=True,exist_ok=True)
    rf,paired=rf_rows();delay,checks=delay_rows();sleep=sleep_rows()
    regression=read(OUT/'regression/check.json') if (OUT/'regression/check.json').exists() else {}
    frozen=read(OUT/'E11_RF/frozen.json') if (OUT/'E11_RF/frozen.json').exists() else {}
    timing=read(OUT/'E10_fallback.json') if (OUT/'E10_fallback.json').exists() else {}
    lines=['# HBAG R2 v2.4 results','',
        'Offline execution of BRIEF_v24.md. Existing manuscript and archived evidence are unchanged. '
        'All new scientific outputs are under results_r2/v24 and tables_r2/v24.','']
    if failure:lines+=['**Pipeline stopped:** '+failure['error'],'No incomplete cell is presented as a completed estimate.','']
    lines+=['## Model changes and regression','',
        'RF links use slot-average incremental power kappa q(R/kappa) in the distributed and central objectives, '
        'power constraints, recovery, marginal costs, and energy ledger. Optical arithmetic is unchanged. '
        'Each on ISL retains its remaining acquisition time across slots, resets on a new wake, and consumes '
        'acquisition idle energy in every affected slot. Config flags rf_capture_power and acquisition_carry '
        'default to true and can independently restore the prior behavior.','',
        'E2 optical regression: '+('PASS' if regression.get('passed') else 'not passed/pending')+'. '+regression.get('comparison',''),'']
    for r in regression.get('rows',[]):
        lines.append('- Seed %d, %s: %s, %d slots, dt = %.15g s.'%(r['seed'],r['method'],'bit-identical scientific summaries and ledgers' if r['passed'] else 'FAILED',r['slots'],r['dt_s']))
    lines+=['','## RF re-selection','',
        'The exact 33-point v2.3.1 grid uses validation seeds 1000–1009, two orbits, I/N = −10 dB, '
        'and the unchanged optical battery. Eligibility requires ESR = 1 for every seed; ties use maximal '
        'mean delivered traffic, minimal mean terminal energy, then original grid order. Fresh seeds '
        '5000–5019 are locked until E11_RF/frozen.json exists.','']
    for method,row in frozen.get('selected',{}).items():lines.append('- %s: `%s`.'%(method,json.dumps(row['parameters'],sort_keys=True)))
    if frozen.get('unselected_methods'):
        lines+=['','No eligible RF point for: '+', '.join(frozen['unselected_methods'])+'. '
                'Those RF-selected rows are unavailable; no ineligible or optical parameters are substituted.']
    lines+=['','## RF table rows','',
        'Delivered is Tbit, with a 95% percentile bootstrap CI (10,000 seed resamples, bootstrap seed 1729). '
        'ESR is the mean of the 20 seed values. E4′ uses optical-selected parameters on seeds 4000–4019; '
        'E4″ uses RF-selected parameters on seeds 5000–5019. Each run spans three orbits.','',
        '| Experiment | I/N dB | Method | Delivered Tbit [95% CI] | ESR |','|---|---:|---|---|---:|']
    for r in rf:
        value=(f(r['delivered_tbit'],7)+' ['+f(r['ci95_low'],7)+', '+f(r['ci95_high'],7)+']') if r['status']=='complete' else r['status']
        lines.append('| %s | %s | %s | %s | %s |'%(r['experiment'],r['interference_db'],r['method'],value,f(r.get('esr'),10)))
    lines+=['','Paired difference is the mean of 100(HBAG − V1)/V1 across matched seeds. '
            'The CI resamples matched seed pairs. Wilcoxon tests use delivered-bit differences, two-sided, '
            'with Holm correction across the six planned RF comparisons. Ratio-of-means sensitivity estimates '
            'are also retained in RF_paired.csv.','',
            '| Experiment | I/N dB | Paired difference % [95% CI] | Holm p |','|---|---:|---|---:|']
    for r in paired:lines.append('| %s | %g | %s [%s, %s] | %s |'%(r['experiment'],r['interference_db'],f(r['paired_percent']),f(r['ci95_low']),f(r['ci95_high']),f(r['p_holm'])))
    lines+=['','## Delay table (dt = 15 s)','',
        'Change is relative to zero delay of the same method and seed. The measured energy change is '
        'the mean across seeds of each run’s maximum measured absolute energy change during the delay; '
        'the maximum across seeds is also shown. The 64-satellite shell, 32 flows, 1 s integration, and '
        'published seeds are unchanged.','',
        '| Delay s | Method | Delivered change % [95% CI] | ESR | Measured change, mean / max J |',
        '|---:|---|---|---:|---:|']
    for r in delay:
        if r['dt_s']==15.:lines.append('| %g | %s | %s [%s, %s] | %s | %s / %s |'%(r['delay_s'],r['method'],f(r['delivered_change_percent']),f(r['change_ci95_low']),f(r['change_ci95_high']),f(r['esr'],10),f(r['measured_energy_change_mean_j'],7),f(r['measured_energy_change_max_j'],7)))
    lines+=['','Published dt = 60 s regression: %d/%d completed comparisons are bit-identical in all scientific summary fields. '
            'This includes the acquisition-duration sensitivity cells. No published dt = 5 s cell exists '
            'in the frozen matrix or current E8 results; no new 5 s cell is substituted.'%(sum(r['passed'] for r in checks),len(checks)),
            'The corrected 5 s acquisition sequence is covered by the physical regression tests.','',
            '## Sleep-value diagnostic (E14)','',
            'Ten seeds (4000–4009), one orbit, 60 s slots, 64 satellites, 32 OD pairs, optical-selected HBAG parameters at rho_E = 1.4. '
            'Every on ISL with positive executed flow is sampled whenever any endpoint in the shell is eclipsed; '
            'the subset with an eclipsed endpoint on that ISL is also saved. Controller values use the actual '
            'executed allocation and recent-peak scaling. Both exact optima use the same prices, node budgets, '
            'candidate catalogue and peak-scaled demand. Removal zeros both directed capacities; it does not '
            'release fixed-power budgets or add new routes. Phi includes utility, incremental power price, and '
            'quadratic regularization. Utility-only losses are saved separately.','',
            'Clarabel uses tolerance 1e-9 and must meet its original-problem primal/dual certificate. '
            'Ratios exclude losses no larger than the summed certified numerical error bounds; counts and '
            'the literal under-half share over all samples are saved. Rank correlations are descriptive '
            'pooled correlations; dependent links/slots are not treated as independent replicates.','',
            '| Value | Samples | Spearman | Kendall | Ratio median [5%, 95%] | Share below half true loss |',
            '|---|---:|---:|---:|---|---:|']
    for variant in ('predicted','greedy'):
        r=sleep[variant]['all'];ratio=r.get('ratio',{})
        lines.append('| %s | %d | %s | %s | %s [%s, %s] | %s |'%(variant,r['n'],f(r.get('spearman')),f(r.get('kendall')),f(ratio.get('q50')),f(ratio.get('q5')),f(ratio.get('q95')),f(r.get('under_half_share'))))
    lines+=['','A2 uses the preceding slot’s observed value. Actual sleeps are matched by physical edge to that '
            'value-generating slot and its unchanged reference problem (value_slot and decision_slot are saved). '
            'Thus predicted and true losses refer to the same prices and budgets. Zero-flow actual sleeps are '
            'included in this separate cohort.','']
    for variant in ('predicted','greedy'):
        r=sleep[variant]['actual_a2'];lines.append('- %s: %d actual A2 sleeps; mean predicted loss %s; mean true loss %s.'%(variant,r['n'],f(r.get('predicted_mean'),7),f(r.get('true_loss_mean'),7)))
    j=sleep['joint'];lines+=['','For %d slots with at least two A2 sleeps, mean sum of individual losses = %s; '
        'mean joint-removal loss = %s; mean joint minus individual sum = %s.'%(j['n'],f(j['sum_individual'].get('mean'),7),f(j['joint_loss'].get('mean'),7),f(j['joint_minus_sum'].get('mean'),7)),
        'The greedy diagnostic reserves alternative directed residual capacity across displaced flows, in '
        'ascending flow/path order, independently for each hypothetical removed edge. It is never passed '
        'to the controller. Full ratio distributions, utility losses, per-seed correlations, actual-sleep '
        'statistics and joint-removal records for both predictors are in tables_r2/v24.','',
        '## Fallback-path timing','']
    if timing:
        lines += [timing['evidence'],timing['limitation'],'',
                  '| Satellites | Snapshots | Online mean s | Usable catalogue mean s (included) |','|---:|---:|---:|---:|']
        for r in timing['rows']:lines.append('| %d | %d | %s | %s |'%(r['M'],r['snapshots'],f(r['mean_online_s'],7),f(r['mean_usable_catalogue_s'],7)))
        table('fallback_timing',timing['rows'])
    else:lines+=['Pending.']
    lines+=['','## Reproducibility','',
        'See results_r2/v24/REPORT.md, code.patch, launch.json, progress.json and DONE.json. '
        'DONE.json is written only after report/table generation. A failed pipeline writes FAILED.json '
        'and a partial report. The source snapshot under pristine/hbag3 is the unmodified baseline. '
        'No network or external notification is used.']
    (ROOT/'RESULTS_R2_v24.md').write_text('\n'.join(lines)+'\n')
