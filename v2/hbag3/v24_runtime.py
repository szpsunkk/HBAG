"""BRIEF_v24 execution boundary: local output isolation and fresh-seed lock."""
import importlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_r2/v24'
TABLES=ROOT/'tables_r2/v24'


def seed_guard(config,phase,experiment=None,method=None):
    if phase=='validation' and config.seed in range(1000,1010):return
    if phase=='test' and config.seed in range(4000,4020):return
    if phase=='test' and config.seed in range(5000,5020):
        frozen=OUT/'E11_RF/frozen.json'
        if not frozen.exists():raise RuntimeError('Fresh seeds locked until RF selection is frozen')
        data=json.loads(frozen.read_text())
        if not data.get('selection_frozen'):raise RuntimeError('RF selection is not frozen')
        if method is not None and method in ('HBAG','Lyapunov','Static+Guard'):
            if method not in data['selected']:raise RuntimeError('No eligible RF parameters for '+method)
            for k,v in data['selected'][method]['parameters'].items():
                if getattr(config,k)!=v:raise RuntimeError('RF configuration differs from frozen selection')
        if config.medium!='rf' or config.interference_db not in (-300.,-10.,0.):
            raise RuntimeError('Fresh seeds reserved for registered RF tests')
        return
    raise ValueError('Unregistered v24 seed/phase')


def storage_tier(seed,phase,requested=None,experiment=None):
    # Keep float64 summaries/ledgers and float32 node traces for every test seed.
    return 'selection' if phase=='validation' else 'node'


def install():
    sys.dont_write_bytecode=True
    from . import runner, timing_r2, central_reference_r2
    runner.seed_guard=timing_r2.seed_guard=seed_guard
    runner.storage_tier=timing_r2.storage_tier=storage_tier
    def failure(problem,tolerance,attempts):
        from .diagnose_reference_v21 import snapshot
        out=OUT/'reference_failures'/problem.fingerprint()
        snapshot(problem,out/'exact_problem_float64.npz')
        runner.write_json(out/'failure.json',dict(tolerance=tolerance,attempts=attempts))
    central_reference_r2.retain_failure=failure
    # The audit hook catches accidental writes by legacy helpers, including
    # archive compaction and failure paths. Reads of all prior evidence remain allowed.
    allowed=(OUT.resolve(),TABLES.resolve())
    final=(ROOT/'RESULTS_R2_v24.md').resolve()
    def check(path):
        if isinstance(path,int) or path is None:return
        p=Path(os.fsdecode(path)).resolve()
        if p==final or str(p).startswith(str(final)+'.'):return
        if any(p==a or a in p.parents for a in allowed):return
        if p in (Path('/dev/null'),):return
        raise PermissionError('v24 output boundary rejected write: '+str(p))
    def audit(event,args):
        if event=='open':
            path,mode,flags=args
            if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):check(path)
        elif event=='os.mkdir':
            if not Path(os.fsdecode(args[0])).is_dir():check(args[0])
        elif event in ('os.remove','os.rmdir'):check(args[0])
        elif event in ('os.rename','os.link','os.symlink'):
            check(args[0]);check(args[1])
        elif event.startswith('socket.'):
            raise PermissionError('BRIEF_v24 prohibits network access')
    sys.addaudithook(audit)


NONSCIENTIFIC={'config','manifest','source_sha256','source_fingerprint','temporal_source_sha256',
               'runtime_s','environment_build_s','mean_solver_s','mean_slot_s','storage_tier'}


def science(summary):
    return {k:v for k,v in summary.items() if k not in NONSCIENTIFIC}


def compare(a,b):
    a=science(a);b=science(b)
    # JSON serialization preserves each float's exact round-trip value.
    return [k for k in sorted(set(a)|set(b)) if json.dumps(a.get(k),sort_keys=True)!=json.dumps(b.get(k),sort_keys=True)]
