"""Seed-paired appendix statistics, reading approved trajectories without simulation."""
import csv,json
from pathlib import Path
import numpy as np
from .environment import ROOT
from .runner import write_json
from .compact_r2 import read_slots
from .audit_r231 import identity,sha
from .plan_r2 import TEST_SEEDS,RF_TEST_SEEDS,RF_EXPERIMENT

TABLES=ROOT/'tables_r2/v231'
BOOTSTRAP_SEED=2311729


def interval(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(n=0,mean=None,ci95_low=None,ci95_high=None)
    if not np.isfinite(a).all():raise ValueError('Nonfinite bootstrap observation')
    rng=np.random.default_rng(BOOTSTRAP_SEED);index=rng.integers(0,len(a),size=(10000,len(a)))
    lo,hi=np.percentile(a[index].mean(axis=1),[2.5,97.5])
    return dict(n=len(a),mean=float(a.mean()),ci95_low=float(lo),ci95_high=float(hi))


def paired_ratio(v0,v1,hbag,seeds):
    v0,v1,hbag=(np.asarray(a,dtype=float) for a in (v0,v1,hbag))
    if any(len(a)!=len(seeds) for a in [v0,v1,hbag]):raise ValueError('Unpaired observations')
    numerator=hbag-v1;denominator=v0-v1;valid=denominator!=0
    ratio=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=valid)
    per_seed=[dict(seed=int(s),numerator_bits=float(a),denominator_bits=float(b),ratio=float(c) if ok else None,
                   status='defined' if ok else 'undefined_zero_denominator') for s,a,b,c,ok in zip(seeds,numerator,denominator,ratio,valid)]
    result=dict(total_seeds=len(seeds),zero_denominator_seeds=int((~valid).sum()),
                negative_denominator_seeds=int((denominator<0).sum()),
                minimum_absolute_nonzero_denominator_bits=float(np.abs(denominator[valid]).min()) if valid.any() else None,
                estimand='Mean of per-seed paired ratios; conditional on nonzero denominator when undefined seeds exist',
                **interval(ratio[valid]))
    return result,per_seed


def tex_escape(value):
    return str(value).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('−','$-$')


def write_table(name,rows):
    TABLES.mkdir(parents=True,exist_ok=True);write_json(TABLES/(name+'.json'),rows)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (TABLES/(name+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    columns=(['label','method','eligible','min_esr','mean_delivered_bits','mean_terminal_energy_j']
             if 'mean_delivered_bits' in keys else
             ['cell','seed','orbit','numerator_bits','denominator_bits','ratio','status','HBAG_bits','V1_bits','difference_bits']
             if 'seed' in keys else
             ['cell','protocol','method','metric','orbit','n','zero_denominator_seeds','mean','ci95_low','ci95_high'])
    visible=[k for k in columns if k in keys]
    if not visible:visible=keys[:8]
    lines=['% Generated from recorded paired seeds; 10,000 bootstrap resamples',r'\begin{tabular}{'+'l'*len(visible)+'}',r'\toprule',
           ' & '.join(tex_escape(k) for k in visible)+r' \\',r'\midrule']
    for r in rows:
        values=[]
        for k in visible:
            v=r.get(k);values.append('NA' if v is None else ('%.8g'%v if isinstance(v,float) else tex_escape(v)))
        lines.append(' & '.join(values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(TABLES/(name+'.tex')).write_text('\n'.join(lines)+'\n')


def orbit_deliveries(path):
    summary=json.loads((Path(path)/'summary.json').read_text());slots=read_slots(path)
    per_orbit=summary['config']['slots_per_orbit'];orbits=summary['config']['orbits'];values=np.zeros(orbits)
    if len(slots)!=orbits*per_orbit or [r['slot'] for r in slots]!=list(range(len(slots))):raise ValueError('Missing orbit slots')
    for row in slots:values[row['slot']//per_orbit]+=row['delivered_bps']*summary['dt_s']
    if not np.isclose(values.sum(),summary['delivered_bits'],rtol=2e-14,atol=1.):raise ArithmeticError('Orbit sum disagrees with original delivered total')
    return values


def existing_statistics():
    sources=[];counts={}
    for experiment in ['E2','E5','E6','E7']:
        ratios=[];per_seed=[];differences=[];orbit_rows=[];orbit_seed=[]
        for cell in sorted((ROOT/'results_r2'/experiment).iterdir()):
            if not cell.is_dir() or not (cell/'seed_4000/HBAG/summary.json').exists():continue
            rows={m:[] for m in ['V0','V1','HBAG']}
            for seed in TEST_SEEDS:
                for method in rows:
                    p=cell/('seed_%d'%seed)/method/'summary.json';row=json.loads(p.read_text())
                    if row['seed']!=seed:raise ValueError('Seed pairing mismatch')
                    rows[method].append(row['delivered_bits']);sources.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
            result,individual=paired_ratio(rows['V0'],rows['V1'],rows['HBAG'],TEST_SEEDS)
            ratios.append(dict(cell=cell.name,metric='(HBAG-V1)/(V0-V1)',**result))
            per_seed.extend(dict(cell=cell.name,**r) for r in individual)
            differences.append(dict(cell=cell.name,metric='HBAG-V1 delivered bits',**interval(np.array(rows['HBAG'])-rows['V1'])))
            if experiment in ['E2','E7']:
                delta=[]
                for seed in TEST_SEEDS:
                    a=orbit_deliveries(cell/('seed_%d'%seed)/'HBAG');b=orbit_deliveries(cell/('seed_%d'%seed)/'V1');delta.append(a-b)
                    orbit_seed.extend(dict(cell=cell.name,seed=seed,orbit=i+1,HBAG_bits=float(x),V1_bits=float(y),difference_bits=float(x-y)) for i,(x,y) in enumerate(zip(a,b)))
                values=np.asarray(delta)
                orbit_rows.extend(dict(cell=cell.name,orbit=i+1,metric='HBAG-V1 delivered bits',**interval(values[:,i])) for i in range(values.shape[1]))
        write_table(experiment+'_paired_ratio',ratios);write_table(experiment+'_paired_ratio_per_seed',per_seed)
        write_table(experiment+'_paired_difference',differences)
        if orbit_rows:write_table(experiment+'_orbit_difference',orbit_rows);write_table(experiment+'_orbit_per_seed',orbit_seed)
        counts[experiment]=len(ratios)
    write_json(ROOT/'results_r2/v231/audit/existing_statistics.json',identity(state='PASS',bootstrap=10000,
        bootstrap_seed=BOOTSTRAP_SEED,source='Approved v2.3 existing data only; no new simulations',cells=counts,sources=sources))
    render_report()


def rf_statistics():
    from .analysis_r2 import METRICS
    rows=[];paired_rows=[]
    for version,folder,seeds in [('E4-prime optical-selected','E4',TEST_SEEDS),('E4-double-prime RF-selected',RF_EXPERIMENT,RF_TEST_SEEDS)]:
        for interference in [-300.,-10.,0.]:
            label='rf_IN_'+str(interference);root=ROOT/'results_r2'/folder/label;data={}
            for method in ['V0','Guard*','Lyapunov','V1','HBAG']:
                values=[]
                for seed in seeds:
                    row=json.loads((root/('seed_%d'%seed)/method/'summary.json').read_text())
                    if row['seed']!=seed:raise ValueError('RF seed mismatch')
                    values.append(row)
                data[method]=values
                for metric in METRICS:
                    if any(r.get(metric) is None for r in values):continue
                    rows.append(dict(cell=label,protocol=version,method=method,metric=metric,**interval([r[metric] for r in values])))
            for metric in ['delivered_bits','esr']:
                difference=np.array([r[metric] for r in data['HBAG']])-np.array([r[metric] for r in data['V1']])
                paired_rows.append(dict(cell=label,protocol=version,metric='HBAG-V1 '+metric,**interval(difference)))
    write_table('E4_side_by_side',rows);write_table('E4_within_protocol_paired',paired_rows)
    write_json(ROOT/'results_r2/v231/audit/rf_statistics.json',identity(state='PASS',bootstrap=10000,
        pairing='Within each protocol only; independent 4000–4019 and 5000–5019 seed sets',groups=30))
    render_report()


def fmt(row,scale=1.):
    if row['mean'] is None:return 'NA'
    return '%.6g [%.6g, %.6g]'%tuple(row[k]*scale for k in ['mean','ci95_low','ci95_high'])


def render_report():
    lines=['# HBAG-TMC R2 v2.3.1 追加结果','',
           'v2.3 结果已通过用户审阅，保持原样；本报告追加 RF 专属选参/独立测试和指定统计图表。方法行为未改变。',
           '原报告：RESULTS_R2.md。E4′与E4″使用不同独立种子，只在各自协议内作配对。',
           '全部 CI 为 10,000 次按种子重采样的 95% percentile bootstrap，随机种子 2311729。']
    frozen=ROOT/'results_r2/E11_RF_v231/frozen.json'
    if frozen.exists():
        r=json.loads(frozen.read_text());lines += ['','## E11-RF','', '状态：'+r['state'],'','```json',json.dumps(r.get('selected',{}),ensure_ascii=False,indent=2),'```']
    else:lines+=['','E11-RF 尚未冻结；E4″种子保持锁定。']
    for exp in ['E2','E5','E6','E7']:
        path=TABLES/(exp+'_paired_ratio.json')
        if not path.exists():continue
        lines += ['','## '+exp+' 配对比值','',
                  '每种子先计算 (HBAG−V1)/(V0−V1)，再求均值；分母为零记 NA，不加 epsilon、不裁剪。部分未定义时为有效配对上的条件统计。',
                  '', '| 单元 | 有效 n / 20 | 零分母数 | 均值 [95% CI] |','|---|---:|---:|---|']
        for r in json.loads(path.read_text()):lines.append('|%s|%d|%d|%s|'%(r['cell'],r['n'],r['zero_denominator_seeds'],fmt(r)))
        orbit=TABLES/(exp+'_orbit_difference.json')
        if orbit.exists():
            lines += ['','逐轨 HBAG−V1 交付差（Tbit）；各轨之和已核验与原总交付量一致。','', '| 单元 | 轨道 | 均值 [95% CI] |','|---|---:|---|']
            for r in json.loads(orbit.read_text()):lines.append('|%s|%d|%s|'%(r['cell'],r['orbit'],fmt(r,1e-12)))
    path=TABLES/'E4_side_by_side.json'
    if path.exists():
        lines += ['','## E4′ / E4″ 并列报告','',
                  'E4′：4000–4019，原光学选参。E4″：5000–5019，RF -10 dB 验证选参后同一参数应用三档。组间不是配对试验，不能把两组样本差异直接归因为调参。',
                  '', '| 协议 | RF 单元 | 方法 | Delivered Tbit [95% CI] |','|---|---|---|---|']
        for r in json.loads(path.read_text()):
            if r['metric']=='delivered_bits':lines.append('|%s|%s|%s|%s|'%(r['protocol'],r['cell'],r['method'],fmt(r,1e-12)))
        lines += ['','组内 HBAG−V1 配对差：','', '| 协议 | 单元 | 指标 | 均值 [95% CI] |','|---|---|---|---|']
        for r in json.loads((TABLES/'E4_within_protocol_paired.json').read_text()):lines.append('|%s|%s|%s|%s|'%(r['protocol'],r['cell'],r['metric'],fmt(r,1e-12 if 'bits' in r['metric'] else 1.)))
    else:lines += ['','E4″ 尚未完成；本报告不填入未运行的结果。']
    lines += ['','## 图与来源','', '新图保存在 figures_r2/v231/，旧图保持原样；选择规则见现行方案附录 B。',
              '新增交付差为负、比值为负或未定义、RF 新参数无优势的结果均原样保留。完整逐种子值、其他指标和 TeX 表见 tables_r2/v231/。']
    (ROOT/'RESULTS_R2_v231.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':existing_statistics()
