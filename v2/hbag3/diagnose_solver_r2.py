"""Reproduce the recorded validation-only Clarabel failure and retain its exact problem."""
import json,traceback
import numpy as np
from .environment import ROOT
from .model import Config
from .runner import run_policy,write_json


def main():
    import hbag3.allocation as module
    original=module.solve_centralized;counter=[0]
    out=ROOT/'results_r2/audit/clarabel_failure'
    def captured(problem,**kwargs):
        slot=counter[0];counter[0]+=1
        try:return original(problem,**kwargs)
        except Exception:
            out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/'exact_problem_float64.npz',matrix_data=problem.resource_matrix.data,
                matrix_indices=problem.resource_matrix.indices,matrix_indptr=problem.resource_matrix.indptr,
                matrix_shape=problem.resource_matrix.shape,budgets=problem.budgets,demand=problem.demand,
                path_mask=problem.path_mask,linear_cost=problem.linear_cost,utility_weight=problem.utility_weight,
                utility_scale=problem.utility_scale,regularization=problem.regularization)
            write_json(out/'failure.json',dict(seed=1005,rho_energy=1.4,slot=slot,problem_fingerprint=problem.fingerprint(),
                exact_float64_for_numerical_regression=True,traceback=traceback.format_exc()))
            raise
    module.solve_centralized=captured
    cal=json.loads((ROOT/'results_r2/G0_prime/calibration_load_0p65.json').read_text())
    c=Config(seed=1005,orbits=1,capacity_j=cal['capacity_by_rho']['1.4'],rho_energy=1.4,load=cal['load'])
    run_policy(c,'HBAG',out/'replay/seed_1005/HBAG',phase='validation',storage='node')


if __name__=='__main__':main()
