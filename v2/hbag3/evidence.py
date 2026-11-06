"""Portable problem and certificate witnesses using NumPy's non-pickle format."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from .optimization import AllocationProblem


def save_witness(path,problem,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    K=problem.resource_matrix
    np.savez_compressed(str(path),indptr=K.indptr,indices=K.indices,data=K.data,K_shape=K.shape,
                        budgets=problem.budgets,demand=problem.demand,path_mask=problem.path_mask,
                        linear_cost=problem.linear_cost,utility_weight=problem.utility_weight,
                        utility_scale=problem.utility_scale,regularization=problem.regularization,
                        allocation=result.allocation,dual=result.dual)
    path.with_suffix('.json').write_text(json.dumps(result.certificate,indent=2))


def load_problem(path):
    with np.load(str(path),allow_pickle=False) as a:
        K=sparse.csr_matrix((a['data'],a['indices'],a['indptr']),shape=tuple(a['K_shape']))
        return AllocationProblem(K,a['budgets'],a['demand'],a['path_mask'],a['linear_cost'],
                                 a['utility_weight'],a['utility_scale'],float(a['regularization']))
