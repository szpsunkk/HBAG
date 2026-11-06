"""Append experiment results from saved summaries; never enter numerical results by hand."""
import json,datetime,math
from pathlib import Path
import numpy as np
from .environment import ROOT
from .runner import write_json


def extend_results():
    path=ROOT/'RESULTS_R2.md';lines=[path.read_text().split('\n<!-- GENERATED FORMAL RESULTS -->')[0],
                                '\n<!-- GENERATED FORMAL RESULTS -->']
    e1=ROOT/'results_r2/E1/summary.json'
    if e1.exists():
        s=json.loads(e1.read_text());lines+=['\n## E1′ 正确性','',
            '%d个独立随机初态，HBAG/V1共%d个方法试验，每次4步；最小储备余量 %.9g J。E(0)<S(0)反例越过安全下限 %.9g J。'%(s['independent_initial_trials'],s['method_trials'],s['min_reserve_margin_j'],s['counterexample']['drop_below_floor_j']),
            '原始逐试验数据：`results_r2/E1/seed_*/trials.npz`；握手、捕获、休眠与不可行状态测试记录见预检日志。']
    settings={'E2':'576星，3轨，5档ρ_E，20个配对测试种子。',
      'E4':'光学公共参考需求与电池容量；RF I/N三档，ρ_E=1.4；optical为E2同配置数据引用。',
      'E5':'ρ_E=1.4；3档load×3类非平稳业务与steady@0.65。',
      'E6':'ρ_E=1.4，5种公共候选路径集；默认hybrid引用E2。',
      'E7':'ρ_E=1.0、6轨，H×β四格与SOC0=0.80附加格。',
      'E8':'64星、1s子步，状态过期执行与附录A提前测量补偿；τ_acq独立扫描。',
      'E9_iterations':'ρ_E=1.4，10/20/50/100/200内层迭代；200引用E2。',
      'E10':'256–1584星20种子×4个冷启动快照；5000星5种子×2快照。星历公共路径预计算单列，在线decide/usable catalogue/allocate/protect_bus分项实测，单线程。',
      'E13':'64星验证场景重训4096环境步，576星新测试种子评估；PPO admission + V1开关/路由/安全混合策略。',
      'MPC':'64星分数开关，t=0单次离线H=96全轨求解并按计划执行；oracle需求与HBAG完全相同。乐观参考，不是形式化上界。'}
    for name,setting in settings.items():
        table=ROOT/('tables_r2/'+name+'_summary.json')
        if not table.exists():continue
        rows=json.loads(table.read_text());groups={}
        for r in rows:groups.setdefault((r['cell'],r['method']),{})[r['metric']]=r
        lines+=['\n## '+name+'′','',setting,'',
                '| cell | 方法 | Delivered Tbit [95% CI] | ESR % [95% CI] | 最低SOC [95% CI] |','|---|---|---|---|---|']
        def show(metrics,key,scale):
            if key not in metrics:return '—'
            r=metrics[key]
            if name=='E8' and key=='esr':
                from .analysis_r2 import esr_percent
                return '%s [%s, %s]'%tuple(esr_percent(r[k]) for k in ['mean','ci95_low','ci95_high'])
            return '%.5g [%.5g, %.5g]'%(r['mean']*scale,r['ci95_low']*scale,r['ci95_high']*scale)
        for (cell,method),metrics in sorted(groups.items()):lines.append('|%s|%s|%s|%s|%s|'%(cell,method,show(metrics,'delivered_bits',1e-12),show(metrics,'esr',100),show(metrics,'min_soc',1)))
        lines+=['','全部指标（含机制诊断）与10,000次按种子bootstrap：`tables_r2/%s_summary.{json,csv,tex}`。除预注册E10/5000星为5个种子外，每组20个独立新测试种子；槽/卫星不作为独立重复。'%name,
                '原始数据：`results_r2/%s/*/seed_*/*/`。完整数组可用 `hbag3.compact_r2.read_raw(path)` 读取；时间序列可用 `read_slots(path)` 读取。'%name]
        adverse=[]
        for (cell,method),metrics in groups.items():
            if method!='HBAG':continue
            for baseline in ['V0','V1','Util-Sleep','Lyapunov','Guard*','Guard_0.75','Guard_0.8','Guard_0.9','PPO+V1','MPC-H-relaxed']:
                other=groups.get((cell,baseline),{})
                if 'delivered_bits' in metrics and 'delivered_bits' in other:
                    delta=metrics['delivered_bits']['mean']-other['delivered_bits']['mean']
                    if delta<0:adverse.append('%s 相对 %s 的平均交付量差为 %.6g Tbit'%(cell,baseline,delta/1e12))
        lines+=['','对HBAG不利的结果：'+('；'.join(adverse)+'。这些是描述性均值差；显著性依据E3配对检验。' if adverse else '本表所有基线数字完整保留；不从未检验的均值差推断显著性。')]
        if name=='E4':
            nominal=ROOT/'tables_r2/E4_RF_nominal_tightness.json'
            if nominal.exists():
                rows=json.loads(nominal.read_text())
                lines+=['','RF自身名义紧张度（100MJ V0、独立验证种子1000–1009完整日食窗口均值/共同0.30Emax）：'+json.dumps({str(r['interference_db']):r['rf_rho_energy'] for r in rows},ensure_ascii=False)+'。不改变光学标定的电池或共同绝对需求。']
        if name=='E7':lines+=['','逐轨最低/末SOC、日照终端开启比例、溢出采能占比、跨轨变化及brown-out见完整指标表；低SOC或退化采能的不可行状态不被删除。']
        if name=='E2':lines+=['','价格归零按日照/日食及条件(a)/(b)分别统计；(b)只在(a)不成立处计算，另列被检验的星×时隙数。HBAG-central的活跃率按ν>1e-6，一阶方法按ν>0，阈值列明，不并作同一判据比较。']
        if name in ('E5','MPC'):
            paired_path=ROOT/('tables_r2/'+name+'_paired.json')
            if paired_path.exists():
                lines+=['','| cell | 配对对比 | Delivered差 Tbit [95% CI] |','|---|---|---|']
                for r in json.loads(paired_path.read_text()):
                    if r['metric']=='delivered_bits':lines.append('|%s|%s|%.6g [%.6g, %.6g]|'%(r['cell'],r['contrast'],r['mean_difference']/1e12,r['ci95_low']/1e12,r['ci95_high']/1e12))
                lines+=['','同一新种子内配对；全部ESR与交付差、bootstrap及检验见`tables_r2/%s_paired.json`。'%name]
    e3=ROOT/'tables_r2/E3_ablation.json'
    if e3.exists():
        rows=json.loads(e3.read_text());lines+=['\n## E3′ 配对消融','',
            '直接复用三档E2数据；共%d个检验统一做Holm，双侧Wilcoxon、10,000次配对bootstrap，含交互项。全部效应方向/区间：`tables_r2/E3_ablation.{json,csv,tex}`。'%len(rows)]
    e9=ROOT/'tables_r2/E9_rates.json'
    if e9.exists():lines+=['\n## E9′ 冻结快照收敛','',
        '每种子按预算余量<1%的节点最多的三个时隙选取，平手取较早时隙。先对同种子的三个快照求均值，再跨20个种子计算95% CI。达到数值精度后不强行拟合速率。`tables_r2/E9_convergence.*`、`E9_rates.*`；图为`figures_r2/E9_*.pdf`。']
    figures=sorted((ROOT/'figures_r2').glob('*.pdf'))
    if figures:lines+=['\n## 图与来源','']+['- [%s](figures_r2/%s)，逐图数据来源：`figures_r2/%s.sources.json`。'%(p.name,p.name,p.stem) for p in figures]
    path.write_text('\n'.join(lines)+'\n')


def extend_gate():
    p=ROOT/'GATE_R2.md'
    if not p.exists():return
    text=p.read_text().split('\n<!-- EXTRA GATE AUDIT -->')[0];lines=[text,'\n<!-- EXTRA GATE AUDIT -->']
    f=ROOT/'results_r2/G0_prime/functional.json'
    if f.exists():
        rows=json.loads(f.read_text());lines+=['\n## 功能检验','',
            '| ρ_E | 方法 | 验证运行数 | 可证初态最小ESR | 最大账本残差 J | 最大约束残差 | 双零价格A2次数 | 最大每星A2次数 |',
            '|---:|---|---:|---:|---:|---:|---:|---:|']
        for rho in [1.2,1.4,1.6]:
            for method in ['V1','HBAG']:
                group=[r for r in rows if r['method']==method and r['config']['rho_energy']==rho]
                if group:lines.append('|%g|%s|%d|%.8f|%.9g|%.9g|%d|%d|'%(rho,method,len(group),min(r['certifiable_esr'] for r in group),max(r['max_ledger_residual_j'] for r in group),max(r['max_constraint_violation'] for r in group),sum(r['zero_price_a2_events'] for r in group),max(r['max_a2_per_node'] for r in group)))
        lines+=['','v2.3功能检验不做影子求解；影子只在正式E2′/ρ_E=1.4/HBAG运行，失败计数单独保存。']
    comparison=ROOT/'results_r2/G0_prime/validation_development_comparison.json'
    if comparison.exists():
        development=json.loads(comparison.read_text())
        lines+=['\n## 验证集开发记录（不进入论文）','','| ρ_E | 平均HBAG−V1交付差 Tbit | 平均配对相对差 |','|---:|---:|---:|']
        for row in development['rows']:lines.append('|%g|%.9g|%.6f%%|'%(row['rho_energy'],row['mean_difference_bits']/1e12,100*row['mean_relative_difference']))
    diagnostics=ROOT/'results_r2/G0_prime/diagnostics.json'
    if diagnostics.exists():
        rows=json.loads(diagnostics.read_text())['rows']
        lines+=['\n## v2.3附加验证诊断（不参与选参，不进入论文）','',
                '| 单元 | 方法 | n | 平均交付Tbit | 最小ESR | 日照ON比例 | 溢出采能比例 | 价格休眠次数 |','|---|---|---:|---:|---:|---:|---:|---:|']
        for rho,name in [(.6,'rho=0.6，3轨'),(1.,'H=275W，beta=60°，6轨')]:
            for method in ['V1','HBAG']:
                group=[r for r in rows if r['config']['rho_energy']==rho and r['method']==method]
                lines.append('|%s|%s|%d|%.9g|%.8f|%.6f|%.6f|%d|'%(name,method,len(group),
                    np.mean([r['delivered_bits'] for r in group])/1e12,min(r['esr'] for r in group),
                    np.mean([r['sun_terminal_on_fraction'] for r in group]),np.mean([r['spill_harvest_fraction'] for r in group]),
                    sum(r['price_sleep_link_events'] for r in group)))
    f=ROOT/'results_r2/E11_validation/abundance.json'
    if f.exists():
        rows=json.loads(f.read_text());baseline={r['seed']:r['delivered_bits'] for r in rows if r['method']=='V0'}
        lines+=['\n## ρ_E=0.6宽裕诊断（不参与选参）','',
            '| 方法 | 平均ESR | 平均交付量Tbit | 相对V0配对交付差Tbit |','|---|---:|---:|---:|']
        for method in sorted({r['method'] for r in rows}):
            group=[r for r in rows if r['method']==method]
            lines.append('|%s|%.8f|%.8f|%.8f|'%(method,np.mean([r['esr'] for r in group]),np.mean([r['delivered_bits'] for r in group])/1e12,np.mean([r['delivered_bits']-baseline[r['seed']] for r in group])/1e12))
    p.write_text('\n'.join(lines)+'\n')


def extend_runtime(workers=10):
    from .runner import fingerprint
    from .plan_r2 import VERSION
    from .gate import selection_grid,label
    calfiles=list((ROOT/'results_r2/G0_prime').glob('calibration_load_*.json'))
    if not calfiles:return
    cal=json.loads(sorted(calfiles)[-1].read_text());env_one=float(np.mean([r['environment_build_s'] for r in cal['per_seed']]))
    stamp=fingerprint();by_method={};sources={}
    for folder in ['G0_prime/stress','G0_prime/functional','E11_validation','audit/storage_sample_v23']:
        for p in (ROOT/'results_r2'/folder).rglob('summary.json'):
            row=json.loads(p.read_text())
            reusable=row['method'] in ('V0','Util-Sleep') and folder=='G0_prime/stress'
            current=row.get('source_fingerprint')==stamp
            if row.get('phase')=='validation' and (current or reusable) and not row.get('shadow_evaluation_enabled',False):
                by_method.setdefault(row['method'],[]).append(row['runtime_s']/row['slots'])
                sources.setdefault(row['method'],[]).append(str(p.relative_to(ROOT)))
    if 'HBAG' not in by_method:return
    seconds={k:float(np.percentile(v,75)) for k,v in by_method.items()}
    fallback=max(seconds.values());grid=selection_grid();per_seed=2*env_one
    for rho in [1.2,1.4,1.6]:per_seed+=192*(sum(seconds.get(r['method'],fallback) for r in grid))
    abundant=2*env_one+192*sum(seconds.get(m,fallback) for m in ['V0','V1','HBAG','V2','HBAG-xi','HBAG-Lyap','Lyapunov','Static+Guard','Util-Sleep'])
    e11=math.ceil(10/workers)*(per_seed+abundant)
    current=len(list((ROOT/'results_r2/E11_validation').glob('rho_*/seed_*/*/summary.json')))
    remaining=[]
    for seed in range(1000,1010):
        total=abundant
        for rho in [1.2,1.4,1.6]:
            base=ROOT/'results_r2/E11_validation'/('rho_'+label(rho))/('seed_%d'%seed)
            for name,method in [(item['label'],item['method']) for item in grid]:
                if not (base/name/'summary.json').exists():total+=192*seconds.get(method,fallback)
        if not (ROOT/'results_r2/E11_validation'/('environment_%d.json'%seed)).exists():total+=2*env_one
        remaining.append(total)
    remaining_hours=max(remaining)*math.ceil(10/workers)/3600
    frozen_path=ROOT/'results_r2/E11_validation/frozen.json'
    frozen=json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    selection_complete=(frozen.get('state')=='PASS' and frozen.get('plan_version')==VERSION
                        and frozen.get('source_fingerprint')==stamp)
    abundance_completed=len(list((ROOT/'results_r2/E11_validation/abundance').glob('seed_*/*/summary.json')))
    if selection_complete:remaining_hours=0.
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),plan_version=VERSION,source_fingerprint=stamp,
        workers=workers,environment_one_orbit_s=env_one,policy_seconds_per_slot_p75=seconds,
        policy_samples={m:len(v) for m,v in by_method.items()},measurement_sources=sources,
        unmeasured_policy_proxy_seconds_per_slot=fallback,shadow_enabled=False,
        E11_full_stage_hours=e11/3600,E11_remaining_upper_point_hours=remaining_hours,
        E11_method_runs_completed=current,E11_method_runs_total=720,E11_abundance_runs=90,
        E11_abundance_runs_completed=abundance_completed,E11_complete=selection_complete,
        E11_remaining_range_hours=[remaining_hours*.8,remaining_hours*1.6],
        interpretation='Current v2.3 validation with shadow disabled; V0/Util stress timings explicitly reused. Unmeasured policies use current maximum as provisional proxy. Remaining includes the full currently executing run and abundance check; not a deadline.')
    write_json(ROOT/'results_r2/audit/runtime_projection.json',result)
    path=ROOT/'RUNTIME_ESTIMATE_R2.md'
    base=path.read_text().split('\n<!-- LIVE RUNTIME -->')[0] if path.exists() else '# R2 v2.3 时间估计\n'
    lines=[base,'\n<!-- LIVE RUNTIME -->','\n## E11′无影子实测外推','',
        ('E11′已通过并冻结参数：选参%d/720、宽裕诊断%d/90，E11′剩余时间为0。'%(current,abundance_completed) if selection_complete else
         '按%d并行、每种子一次环境构建与三档24点内存复用，整个E11′阶段点估计%.2f小时；当前完成%d/720，另有90个宽裕诊断。剩余粗估%.2f–%.2f小时。'%(workers,e11/3600,current,remaining_hours*.8,remaining_hours*1.6)),
        'HBAG与V1只使用当前v2.3、无影子验证耗时；旧v2.1含影子耗时不参与本估算。未实测的方法暂以当前最慢方法作排期代理，网格推进后自动更新。测试仍须功能检验与E11′双PASS。',
        '', '```json',json.dumps({k:v for k,v in result.items() if k!='measurement_sources'},indent=2,ensure_ascii=False),'```']
    path.write_text('\n'.join(lines)+'\n')
