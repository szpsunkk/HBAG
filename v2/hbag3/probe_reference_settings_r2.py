"""Compare Clarabel numerical settings on retained validation failure fixtures."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from .environment import ROOT
from .optimization import AllocationProblem
from .runner import write_json


def load(path):
    with np.load(path) as z:
        matrix=sparse.csr_matrix((z['matrix_data'],z['matrix_indices'],z['matrix_indptr']),shape=tuple(z['matrix_shape']))
        return AllocationProblem(matrix,z['budgets'],z['demand'],z['path_mask'],z['linear_cost'],z['utility_weight'],z['utility_scale'],float(z['regularization']))


def main():
    archive=ROOT/'results_r2/audit/v21_step_archive.json'
    original=(ROOT/json.loads(archive.read_text())['code']/'central_reference_r2.py') if archive.exists() else ROOT/'hbag3/central_reference_r2.py'
    source=original.read_text()
    source=source.replace('def solve(problem,tolerance=1e-9):','def solve(problem,tolerance=1e-9,settings=None):')
    source=source.replace('max_iter=1000)','max_iter=1000,**(settings or {}))')
    namespace={'__name__':'hbag3._numerical_probe','__package__':'hbag3'}
    exec(compile(source,'<independent numerical settings probe>','exec'),namespace)
    options=[{},dict(max_step_fraction=.95),dict(max_step_fraction=.9),
        dict(static_regularization_constant=1e-10),
        dict(max_step_fraction=.9,static_regularization_constant=1e-10),
        dict(equilibrate_max_iter=50,equilibrate_min_scaling=1e-10,equilibrate_max_scaling=1e10)]
    corpus=ROOT/'results_r2/audit/reference_corpus_v21';rows=[]
    for path in sorted(corpus.glob('*/slot_*.npz')):
        p=load(path)
        for settings in options:
            row=dict(path=str(path.relative_to(ROOT)),settings=settings,problem_fingerprint=p.fingerprint())
            try:
                answer=namespace['solve'](p,settings=settings)
                row.update(status=answer['status'],gap=answer['certificate_gap'],iterations=answer['iterations'],runtime_s=answer['runtime_s'])
            except Exception as error:row.update(status='ERROR',error=str(error))
            rows.append(row);print(json.dumps(row),flush=True)
            write_json(corpus/'settings_probe.json',rows)


if __name__=='__main__':main()
