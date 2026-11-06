#!/usr/bin/env python3
"""Portable release entry point; scientific modules are preserved byte for byte."""
import argparse
import collections
import csv
from dataclasses import asdict, replace
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
ACTIVE_EXPERIMENT = None


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def records(path):
    with gzip.open(str(path), 'rt') as stream:
        return [json.loads(line) for line in stream]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    expected = read(PACKAGE / 'frozen/scientific_source_sha256.json')
    wrong = [name for name, digest in expected.items() if file_hash(PACKAGE / name) != digest]
    if wrong:
        raise RuntimeError('Scientific source differs from the release: ' + ', '.join(wrong))
    return len(expected)


def verify_bundle():
    entries=(PACKAGE/'SHA256SUMS').read_text().splitlines()
    wrong=[]
    for entry in entries:
        digest,name=entry.split('  ',1)
        path=PACKAGE/name
        if not path.is_file() or file_hash(path)!=digest:wrong.append(name)
    if wrong:raise RuntimeError('Release checksum mismatch: '+', '.join(wrong))
    return len(entries)


def init_workspace(work):
    """Create a separate writable tree; refuse mismatched sources on resumption."""
    verify_source()
    work = Path(work).expanduser().resolve()
    if work == PACKAGE or work in PACKAGE.parents:
        raise ValueError('Choose a separate output directory, not the source directory or its parent')
    work.mkdir(parents=True, exist_ok=True)
    for source in list((PACKAGE / 'hbag3').glob('*.py')) + [PACKAGE / 'base_rev3/sources/natural_earth_cities.geojson']:
        target = work / source.relative_to(PACKAGE)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_hash(target) != file_hash(source):
                raise RuntimeError('Existing workspace has different code/input: ' + str(target))
        else:
            shutil.copyfile(source, target)
    for source, dest in [('frozen/optical.json','results_r2/E11_validation/frozen.json'),
                         ('frozen/rf.json','results_r2/v24/E11_RF/frozen.json'),
                         ('frozen/ppo/model.pt','results_r2/E13/training/model.pt'),
                         ('frozen/ppo/training.json','results_r2/E13/training/training.json')]:
        target=work/dest;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_hash(target)!=file_hash(PACKAGE/source):
            raise RuntimeError('Frozen parameters/checkpoint differ: '+str(target))
        if not target.exists():shutil.copyfile(PACKAGE/source,target)
    stamp=dict(release='2.0.0',science='hbag3 v2.4',source=file_hash(PACKAGE/'frozen/scientific_source_sha256.json'),
               optical=file_hash(PACKAGE/'frozen/optical.json'),rf=file_hash(PACKAGE/'frozen/rf.json'))
    marker=work/'REPRODUCTION.json'
    if marker.exists() and read(marker)!=stamp:raise RuntimeError('Workspace release identity differs')
    write(marker,stamp)
    return work


def configure(work, experiment, min_free_gb=1.):
    """Only output paths, disk threshold and the completed-study seed gate differ."""
    global ACTIVE_EXPERIMENT
    ACTIVE_EXPERIMENT = experiment
    work=Path(work).resolve()
    import hbag3.environment as environment
    environment.ROOT=work
    modules=[importlib.import_module('hbag3.'+name) for name in
             ['runner','timing_r2','convergence_r2','optional_r2','correctness_r2','v24_sleep',
              'protocol_r231','analysis_r2','analysis_r231','gate','matrix_v21','v24_runtime']]
    rf=read(PACKAGE/'frozen/rf.json')

    def guard(config,phase,experiment=None,method=None):
        if phase=='validation' and 1000<=config.seed<=1009:return
        if phase=='test' and 4000<=config.seed<=4019 and ACTIVE_EXPERIMENT!='E4_RF':return
        if phase=='test' and 5000<=config.seed<=5019 and ACTIVE_EXPERIMENT=='E4_RF':
            if config.medium!='rf' or config.interference_db not in (-300.,-10.,0.):
                raise ValueError('5000–5019 are reserved for RF at the registered interference levels')
            if method in rf['selected']:
                if any(getattr(config,k)!=v for k,v in rf['selected'][method]['parameters'].items()):
                    raise ValueError('RF parameters differ from the frozen validation selection')
            return
        raise ValueError('Unregistered seed/phase/experiment: %s/%s/%s' % (config.seed,phase,ACTIVE_EXPERIMENT))

    def disk():
        free=shutil.disk_usage(work).free
        if free<min_free_gb*1e9:
            raise RuntimeError('Insufficient free storage: %.2f GB < %.2f GB' % (free/1e9,min_free_gb))
        return free

    def storage(seed,phase,requested=None,experiment=None):
        if requested=='selection':return 'selection'
        # RF and short-slot delay use the published v2.4 node archive tier.
        if ACTIVE_EXPERIMENT in ('E4_optical','E4_RF','E8'):
            return 'selection' if phase=='validation' else 'node'
        from hbag3.storage import storage_tier
        return storage_tier(seed,phase,requested,experiment=experiment)

    # Update imported aliases too, without changing any controller or solver.
    for name,module in list(sys.modules.items()):
        if not name.startswith('hbag3.'):continue
        if hasattr(module,'ROOT'):module.ROOT=work
        if hasattr(module,'seed_guard'):module.seed_guard=guard
        if hasattr(module,'disk_guard'):module.disk_guard=disk
        if name in ('hbag3.runner','hbag3.timing_r2'):
            module.storage_tier=storage
    sleep=importlib.import_module('hbag3.v24_sleep')
    sleep.OUT=work/'results_r2/v24'
    sleep.install=lambda:None  # the release already provides an isolated output tree
    disk()


def cells_for(experiment):
    from hbag3.model import Config
    if experiment in ('E4_optical','E4_RF'):
        optical=read(PACKAGE/'frozen/optical.json')
        seeds=range(5000,5020) if experiment=='E4_RF' else range(4000,4020)
        return [dict(experiment=experiment,label='IN_%g'%level,
                     config=asdict(Config(seed=seed,orbits=3,medium='rf',interference_db=level,
                        rho_energy=1.4,capacity_j=optical['capacity_by_rho']['1.4'],
                        load=optical['load'],flows=optical['flows'])),
                     methods=['V0','Static+Guard','Lyapunov','V1','HBAG'])
                for level in (-300.,-10.,0.) for seed in seeds]
    if experiment in ('E9','MPC','E13','E14'):
        seeds=range(4000,4010) if experiment=='E14' else range(4000,4020)
        return [dict(experiment=experiment,label=label,seed=seed,methods=[])
                for label in (['M64','M576'] if experiment=='E9' else ['main']) for seed in seeds]
    return read(PACKAGE/'frozen/matrices'/('%s.json'%experiment))


def run_job(job):
    work,experiment,cell,minimum=job
    configure(work,experiment,minimum)
    from hbag3.model import Config
    from hbag3.environment import build_environment
    from hbag3.runner import run_policy
    from hbag3.matrix_v21 import policy_config
    from hbag3.compact_r2 import compact_run,compact_slots
    frozen=read(PACKAGE/'frozen/optical.json')
    work=Path(work)
    if experiment=='E9':
        from hbag3.convergence_r2 import convergence_cell
        return convergence_cell((cell['seed'],cell['label']=='M64',frozen))
    if experiment=='MPC':
        from hbag3.optional_r2 import evaluate_mpc
        return evaluate_mpc((cell['seed'],frozen))
    if experiment=='E13':
        from hbag3.optional_r2 import evaluate_ppo
        return evaluate_ppo((cell['seed'],frozen))
    if experiment=='E14':
        from hbag3.v24_sleep import run
        return run(cell['seed'])
    if experiment=='E10':
        from hbag3.timing_r2 import scaling_cell
        return scaling_cell((dict(cell,parallel_workers=1),frozen))
    config=Config(**cell['config'])
    folder=work/'results_r2'/experiment/cell['label']/('seed_%d'%config.seed)
    env=build_environment(config,progress=lambda row:write(folder/'environment_status.json',row))
    rows=[]
    for label in cell['methods']:
        if experiment=='E4_RF':
            actual='Static+Guard' if label=='Guard*' else label
            rf=read(PACKAGE/'frozen/rf.json')
            selected=replace(config,**rf['selected'].get(actual,{}).get('parameters',{}))
        else:selected,actual=policy_config(config,label,frozen)
        if experiment=='E8':
            from hbag3.timing_r2 import run_temporal
            summary=run_temporal(selected,actual,folder/label,env,phase='test')
        else:summary=run_policy(selected,actual,folder/label,env,phase='test')
        compact_run(folder/label);compact_slots(folder/label)
        rows.append(dict(method=label,seed=config.seed,delivered_bits=summary['delivered_bits'],esr=summary['esr']))
    return rows


def run_matrix(args):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    work=init_workspace(args.output)
    order=['E2','E9','E9_iterations','E4_optical','E4_RF','E5','E6','E7','E8','E10','MPC','E13','E14']
    experiments=order if args.experiment=='ALL' else [args.experiment]
    for experiment in experiments:
        cells=cells_for(experiment)
        if args.seeds:
            cells=[c for c in cells if c.get('seed',c.get('config',{}).get('seed')) in args.seeds]
        if args.cells:cells=[c for c in cells if c['label'] in args.cells]
        if args.methods:
            if experiment in ('E9','E10','MPC','E13','E14'):
                raise ValueError('--methods applies to policy-comparison matrices only')
            unknown=set(args.methods)-{m for c in cells for m in c['methods']}
            if unknown:raise ValueError('Unknown method labels: '+repr(sorted(unknown)))
            cells=[dict(c,methods=[m for m in c['methods'] if m in args.methods]) for c in cells]
            cells=[c for c in cells if c['methods']]
        if not cells:raise ValueError('No registered cells match the selection for '+experiment)
        workers=1 if experiment=='E10' else args.workers
        jobs=[(str(work),experiment,c,args.min_free_gb) for c in cells]
        write(work/'release_matrices'/(experiment+'.json'),cells)
        print('%s: %d cells, %d worker(s)'%(experiment,len(cells),workers),flush=True)
        done=0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(run_job,j):j[2] for j in jobs}
            for future in as_completed(pending):
                future.result();done+=1
                cell=pending[future]
                print('%s %d/%d %s seed=%s'%(experiment,done,len(cells),cell['label'],cell.get('seed',cell.get('config',{}).get('seed'))),flush=True)
                write(work/'progress.json',dict(experiment=experiment,completed_cells=done,total_cells=len(cells),state='running'))
    write(work/'progress.json',dict(state='complete',experiments=experiments))


def smoke(args):
    work=init_workspace(args.output);configure(work,'smoke',.05)
    from hbag3.model import Config
    from hbag3.environment import build_environment
    from hbag3.runner import run_policy
    from hbag3.matrix_v21 import policy_config
    frozen=read(PACKAGE/'frozen/optical.json')
    config=Config(seed=1000,planes=8,per_plane=8,flows=16,orbits=1,slots_per_orbit=24,
                  rho_energy=1.4,capacity_j=frozen['capacity_by_rho']['1.4'],max_iterations=50)
    env=build_environment(config);result=[]
    for method in ('V0','V1','HBAG'):
        c,actual=policy_config(config,method,frozen)
        row=run_policy(c,actual,work/'results_r2/smoke/main/seed_1000'/method,env,storage='selection')
        if not row['complete'] or row['max_ledger_residual_j']>1e-6:raise ArithmeticError('Smoke ledger failure')
        if method in ('V1','HBAG') and row['esr']!=1.:raise ArithmeticError('Smoke certificate failure')
        result.append(dict(method=method,delivered_bits=row['delivered_bits'],esr=row['esr']))
    write(work/'smoke.json',dict(passed=True,not_a_paper_result=True,results=result))
    print(json.dumps(result,indent=2))


def regression(args):
    work=init_workspace(args.output);configure(work,'regression',1.)
    from hbag3.model import Config
    from hbag3.environment import build_environment
    from hbag3.runner import run_policy
    from hbag3.v24_runtime import compare
    refs=[r for r in records(PACKAGE/'reference/validation.jsonl.gz') if r['seed']==args.seed]
    if len(refs)!=2:raise RuntimeError('Missing validation reference')
    folder=work/'results_r2/regression'/('seed_%d'%args.seed)
    print('Building the full-scale validation environment (several minutes)...',flush=True)
    env=build_environment(Config(**refs[0]['config']),progress=lambda row:write(folder/'environment_status.json',row));checks=[]
    for old in refs:
        print('Validation seed %d: %s'%(args.seed,old['method']),flush=True)
        current=run_policy(Config(**old['config']),old['method'],
            folder/old['method'],env,storage='selection')
        differences=compare(old,current)
        checks.append(dict(seed=args.seed,method=old['method'],exact_scientific_match=not differences,
                           differing_fields=differences,delivered_bits=current['delivered_bits']))
    write(work/'regression.json',checks)
    print(json.dumps(checks,indent=2))
    if any(not c['exact_scientific_match'] for c in checks):
        raise ArithmeticError('Validation regression differs; inspect regression.json. Different numerical libraries can change floating-point results.')


def rf_selection_job(job):
    work,seed=job;configure(work,'RF_validation',1.)
    from hbag3.model import Config
    from hbag3.environment import build_environment
    from hbag3.runner import run_policy
    protocol=read(PACKAGE/'frozen/rf_protocol.json')
    base=replace(Config(**protocol['config']),seed=seed);env=build_environment(base)
    for point in protocol['grid']:
        config=replace(base,**point['parameters'])
        run_policy(config,point['method'],Path(work)/'results_r2/RF_validation'/('seed_%d'%seed)/point['label'],env,storage='selection')


def selection(args):
    from concurrent.futures import ProcessPoolExecutor
    work=init_workspace(args.output)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for seed,_ in zip(range(1000,1010),pool.map(rf_selection_job,[(str(work),s) for s in range(1000,1010)])):
            print('RF validation seed %d complete'%seed,flush=True)
    from hbag3.protocol_r231 import select_rf
    protocol=read(PACKAGE/'frozen/rf_protocol.json');rows=[]
    for seed in protocol['seeds']:
        for point in protocol['grid']:
            rows.append(dict(read(work/'results_r2/RF_validation'/('seed_%d'%seed)/point['label']/'summary.json'),grid_label=point['label']))
    chosen,table=select_rf(rows)
    # The final manuscript discloses the original common fallback for Lyapunov.
    for method in ('HBAG','Static+Guard','Lyapunov'):
        if method not in chosen:
            group=[(i,r) for i,r in enumerate(table) if r['method']==method]
            chosen[method]=min(group,key=lambda t:(-t[1]['mean_esr'],-t[1]['mean_delivered_bits'],t[1]['mean_terminal_energy_j'],t[0]))[1]
    expected=read(PACKAGE/'frozen/rf.json')['selected']
    checks={m:chosen[m]['parameters']==expected[m]['parameters'] for m in chosen}
    write(work/'rf_selection.json',dict(selected=chosen,grid=table,matches_published_parameters=checks))
    print(json.dumps(checks,indent=2))
    if not all(checks.values()):
        raise ArithmeticError('RF replay differs from the frozen selection; inspect rf_selection.json')


def verify(args):
    import numpy as np
    bundle_count=verify_bundle()
    source_count=verify_source()
    catalogue=read(PACKAGE/'reference/catalogue.json')
    data=records(PACKAGE/'reference/per_seed.jsonl.gz')
    assert len(data)==catalogue['summary_total']
    groups=collections.defaultdict(list)
    for row in data:groups[row['experiment'],row['cell'],row['method_label']].append(row['summary'])
    checked=0
    for experiment in ('E2','E5','E6','E7','E9_iterations','E10','E13','MPC'):
        with (PACKAGE/'reference/tables'/(experiment+'_summary.csv')).open() as stream:
            for expected in csv.DictReader(stream):
                if expected['metric'] not in ('delivered_bits','esr','terminal_energy_j','min_soc','mean_slot_s'):continue
                rows=groups[experiment,expected['cell'],expected['method']]
                assert len(rows)==int(expected['n']),(experiment,expected['cell'],expected['method'])
                values=np.asarray([r[expected['metric']] for r in rows])
                assert np.isclose(values.mean(),float(expected['mean']),rtol=1e-12,atol=1e-9)
                checked+=1
    for experiment,seeds in [('E4_optical',range(4000,4020)),('E4_RF',range(5000,5020))]:
        rows=[r for r in data if r['experiment']==experiment]
        assert len(rows)==300
        for level in ('IN_-300','IN_-10','IN_0'):
            for method in ('V0','V1','HBAG','Static+Guard','Lyapunov'):
                assert sorted(r['seed'] for r in groups[experiment,level,method])==list(seeds)
    # These corrected tables supersede the original E4/E8 summaries.
    from hbag3.v24_report import ci
    corrected_checks=0
    with (PACKAGE/'reference/tables/v24/RF_summary.csv').open() as stream:
        for expected in csv.DictReader(stream):
            experiment={'E4_prime':'E4_optical','E4_double_prime':'E4_RF'}[expected['experiment']]
            rows=sorted(groups[experiment,'IN_%g'%float(expected['interference_db']),expected['method']],key=lambda r:r['seed'])
            current=ci([r['delivered_bits']/1e12 for r in rows])
            for old,new in [('delivered_tbit','mean'),('ci95_low','ci95_low'),('ci95_high','ci95_high')]:
                assert np.isclose(float(expected[old]),current[new],rtol=1e-12,atol=1e-10)
            assert np.isclose(float(expected['esr']),np.mean([r['esr'] for r in rows]),rtol=1e-12)
            assert float(expected['min_seed_esr'])==min(r['esr'] for r in rows)
            corrected_checks+=1
    with (PACKAGE/'reference/tables/v24/delay_summary.csv').open() as stream:
        for expected in csv.DictReader(stream):
            rows=sorted(groups['E8',expected['cell'],expected['method']],key=lambda r:r['seed'])
            assert len(rows)==int(expected['n'])==20
            energy=ci([r['bound_measured_max_j'] for r in rows])
            assert np.isclose(float(expected['measured_energy_change_mean_j']),energy['mean'],rtol=1e-12)
            assert np.isclose(float(expected['esr']),np.mean([r['esr'] for r in rows]),rtol=1e-12)
            if expected['delivered_change_percent']:
                baseline={r['seed']:r for r in groups['E8','dt_%g_delay_0'%float(expected['dt_s']),expected['method']]}
                changes=ci([100*(r['delivered_bits']/baseline[r['seed']]['delivered_bits']-1) for r in rows])
                for old,new in [('delivered_change_percent','mean'),('change_ci95_low','ci95_low'),('change_ci95_high','ci95_high')]:
                    assert np.isclose(float(expected[old]),changes[new],rtol=1e-12,atol=1e-12)
            corrected_checks+=1
    from hbag3.analysis_r231 import paired_ratio
    paired_checks=0
    for experiment in ('E2','E5','E6','E7'):
        with (PACKAGE/'reference/tables/v231'/(experiment+'_paired_ratio.csv')).open() as stream:
            for old in csv.DictReader(stream):
                values=[]
                for method in ('V0','V1','HBAG'):
                    runs=sorted(groups[experiment,old['cell'],method],key=lambda r:r['seed'])
                    assert [r['seed'] for r in runs]==list(range(4000,4020))
                    values.append([r['delivered_bits'] for r in runs])
                current,_=paired_ratio(*values,list(range(4000,4020)))
                for key in ('mean','ci95_low','ci95_high'):
                    if current[key] is None:assert old[key]==''
                    else:assert np.isclose(current[key],float(old[key]),rtol=1e-12,atol=1e-14)
                paired_checks+=1
    result=dict(passed=True,verified_package_files=bundle_count,unaltered_scientific_files=source_count,archived_runs=len(data),
                matched_table_means=checked,matched_seed_paired_bootstrap_cells=paired_checks,
                matched_corrected_rf_and_delay_rows=corrected_checks,
                rf_grid_fallback_disclosed=not read(PACKAGE/'frozen/rf.json')['selected']['Lyapunov']['eligible'])
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('smoke','regression','select-rf'):
        p=sub.add_parser(name);p.add_argument('--output',type=Path,default=PACKAGE/'outputs'/name)
        if name=='regression':p.add_argument('--seed',type=int,choices=[1000,1001],default=1000)
        if name=='select-rf':p.add_argument('--workers',type=int,default=4)
    sub.add_parser('verify')
    p=sub.add_parser('run');p.add_argument('--experiment',required=True,choices=['ALL','E2','E4_optical','E4_RF','E5','E6','E7','E8','E9','E9_iterations','E10','MPC','E13','E14'])
    p.add_argument('--output',type=Path,default=PACKAGE/'outputs/paper')
    p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--cells',nargs='+');p.add_argument('--methods',nargs='+')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--min-free-gb',type=float,default=25.)
    args=parser.parse_args()
    if hasattr(args,'workers') and not 1<=args.workers<=16:parser.error('--workers must be between 1 and 16')
    {'smoke':smoke,'regression':regression,'select-rf':selection,'verify':verify,'run':run_matrix}[args.command](args)


if __name__=='__main__':main()
