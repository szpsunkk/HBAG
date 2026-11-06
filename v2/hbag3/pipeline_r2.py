"""Sequential R2 priorities with seed locks and completed-cell checks."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import json,time,traceback
from .environment import ROOT,build_environment
from .model import Config
from .runner import run_policy,seed_guard,write_json,disk_guard,fingerprint
from .matrix_v21 import matrix,policy_config
from .gate import log,issue


def run_cell(args):
    cell,frozen=args;c=Config(**cell['config']);seed_guard(c,'test')
    out=ROOT/'results_r2'/cell['experiment']/cell['label']/('seed_%d'%c.seed)
    expected=[out/m/'summary.json' for m in cell['methods']]
    if all(p.exists() for p in expected):
        rows=[json.loads(p.read_text()) for p in expected]
        if all(r['source_fingerprint']==fingerprint() for r in rows):
            from .compact_r2 import compact_run,compact_slots
            for method in cell['methods']:compact_run(out/method);compact_slots(out/method)
            return rows
        raise RuntimeError('Cell output source mismatch')
    env=build_environment(c,progress=lambda row:write_json(out/'environment_status.json',row))
    rows=[]
    for label in cell['methods']:
        chosen,method=policy_config(c,label,frozen)
        rows.append(run_policy(chosen,method,out/label,env=env,phase='test'))
        from .compact_r2 import compact_run,compact_slots
        compact_run(out/label);compact_slots(out/label)
    write_json(out/'complete.json',dict(complete=True,methods=cell['methods'],source_fingerprint=fingerprint()))
    return rows


def run_matrix(experiment,frozen,workers):
    cells=matrix(experiment,frozen);write_json(ROOT/('configs_r2/'+experiment+'.json'),cells)
    log('R2 '+experiment+'启动：%d个cell/seed，环境每cell只构建一次。'%len(cells))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_cell,(cell,frozen)):cell for cell in cells}
        complete=0
        for future in as_completed(pending):
            future.result();complete+=1;disk_guard()
            write_json(ROOT/('results_r2/'+experiment+'/progress.json'),dict(complete_cells=complete,total_cells=len(cells)))
    from .analysis_r2 import summaries
    link_common_cells(experiment,frozen)
    from .paired_inputs_r2 import verify
    verify(experiment)
    summaries(experiment)
    log('R2 '+experiment+'全部20测试种子完成，原始结果与bootstrap表已生成。')


def link_common_cells(experiment,frozen):
    """Exact already-completed E2 cells are references, without duplicating inputs."""
    import os
    labels={'E6':('K4_hybrid',['V0','V1','HBAG']),
            'E9_iterations':('iterations_200',['HBAG']),
            'E4':('optical',['V0','Guard*','Lyapunov','V1','HBAG'])}
    if experiment not in labels:return
    label,methods=labels[experiment]
    for seed in range(4000,4020):
        parent=ROOT/'results_r2'/experiment/label/('seed_%d'%seed);parent.mkdir(parents=True,exist_ok=True)
        original=ROOT/'results_r2/E2/rho_1.4'/('seed_%d'%seed)
        for method in methods:
            target=method
            if method=='Guard*':target='Guard_'+str(frozen['selected_by_rho']['1.4']['Static+Guard']['parameters']['guard_soc'])
            path=parent/method
            if not path.exists():path.symlink_to(os.path.relpath(original/target,parent),target_is_directory=True)
        for name in ['inputs.npz','inputs.json','inputs']:
            if (original/name).exists() and not (parent/name).exists():
                (parent/name).symlink_to(os.path.relpath(original/name,parent),target_is_directory=(original/name).is_dir())
        write_json(parent/'reuse.json',dict(source=str(original.relative_to(ROOT)),methods=methods,reason='Exact common configuration, parameters and seed'))


def run_priorities(priority,workers,frozen):
    if not 1<=workers<=10:raise ValueError('workers must be 1–10')
    seed_guard(Config(seed=4000),'test');disk_guard()
    from .storage_budget import ensure_storage_budget
    ensure_storage_budget(frozen)
    write_json(ROOT/'results_r2/pipeline_status.json',dict(state='running',priority=priority,start_unix=time.time()))
    try:
        # Passing --priority P1/P2 includes unfinished prerequisite priorities.
        from .correctness_r2 import run_correctness
        from .convergence_r2 import convergence_cell
        from .analysis_r2 import ablation
        done=ROOT/'results_r2/P0_DONE.json'
        if not done.exists():
            run_correctness(frozen,workers)
            run_matrix('E2',frozen,workers);ablation()
            jobs=[(seed,small,frozen) for small in [True,False] for seed in range(4000,4020)]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(convergence_cell,jobs):pass
            from .analysis_r2 import convergence_summary
            convergence_summary()
            run_matrix('E9_iterations',frozen,workers)
            from .figures_r2 import make_main_figures,convergence_figures
            make_main_figures(frozen);convergence_figures()
            write_json(done,dict(complete=True,source_fingerprint=fingerprint(),time=time.time()))
            log('R2 P0 DONE')
            from .report_r2 import main as report
            report()
        if priority in ('P1','P2','ALL'):
            done=ROOT/'results_r2/P1_DONE.json'
            if not done.exists():
                from .rf_calibration_r2 import run_rf_calibration
                run_rf_calibration(frozen,workers)
                for experiment in ['E4','E5','E6','E7']:run_matrix(experiment,frozen,workers)
                from .timing_r2 import run_temporal_matrix,run_scaling_matrix
                run_temporal_matrix(frozen,workers);run_scaling_matrix(frozen,workers)
                write_json(done,dict(complete=True,source_fingerprint=fingerprint(),time=time.time()))
                log('R2 P1 DONE')
                from .report_r2 import main as report
                report()
        if priority in ('P2','ALL'):
            from .optional_r2 import run_optional
            run_optional(frozen,workers)
        from .report_r2 import main as report
        report();write_json(ROOT/'results_r2/pipeline_status.json',dict(state='complete',priority=priority,end_unix=time.time()))
    except Exception:
        detail=traceback.format_exc();write_json(ROOT/'results_r2/pipeline_error.json',dict(traceback=detail,time=time.time()))
        issue('R2正式流水线未完成；保留所有数据，不写未完成的DONE。'+detail);raise
