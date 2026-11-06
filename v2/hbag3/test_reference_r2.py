"""Independent conic regression: zero resources, positive small budgets and exact saved failure."""
from pathlib import Path
import numpy as np
from scipy import sparse
import pytest
from hbag3.optimization import AllocationProblem
from hbag3.central_reference_r2 import solve


def problem(budget):
    return AllocationProblem(sparse.csr_matrix([[1.,0.,1.,0.],[0.,1.,0.,1.],[.2,.1,.2,.1]]),
        np.array([budget,3.,2.]),np.array([2.,2.]),np.ones((2,2),bool),np.full((2,2),.01),
        np.ones(2),np.ones(2),.02)


def dense_reference(p):
    import cvxpy as cp
    x=cp.Variable(p.shape,nonneg=True);z=cp.sum(x,axis=1)
    value=cp.sum(cp.multiply(p.utility_weight,cp.log1p(cp.multiply(1/p.utility_scale,z))))-cp.sum(cp.multiply(p.linear_cost,x))-.5*p.regularization*cp.sum_squares(x)
    constraints=[z<=p.demand,p.resource_matrix@cp.reshape(x,(x.size,),order='C')<=p.budgets]
    if np.any(~p.path_mask):constraints.append(x[~p.path_mask]==0)
    model=cp.Problem(cp.Maximize(value),constraints)
    model.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=1000)
    return np.asarray(x.value)


@pytest.mark.parametrize('budget',[0.,1e-12,1e-8,.4,2.])
def test_exact_reduction_preserves_optimum_and_original_dual_certificate(budget):
    p=problem(budget);old=dense_reference(p);new=solve(p,tolerance=1e-9)
    assert p.objective(new['allocation'])==pytest.approx(p.objective(old),abs=1e-6)
    assert p.violations(new['allocation'])<1e-7
    assert new['certificate_gap']<1e-6
    if budget==0:assert np.all(new['allocation'][:,0]==0)
    else:assert new['fixed_zero_variables']==0


def test_all_zero_resources_is_an_exact_singleton_with_valid_dual():
    p=problem(0.);p.budgets[:]=0.
    answer=solve(p)
    assert not answer['allocation'].any() and answer['certificate_gap']==0.
    assert answer['dual_upper_bound']==0.


def load_regression(folder):
    path=Path(__file__).resolve().parents[1]/'results_r2/audit'/folder/'exact_problem_float64.npz'
    if not path.exists():pytest.skip('Exact validation failure has not yet been captured')
    with np.load(path) as z:
        matrix=sparse.csr_matrix((z['matrix_data'],z['matrix_indices'],z['matrix_indptr']),shape=tuple(z['matrix_shape']))
        return AllocationProblem(matrix,z['budgets'],z['demand'],z['path_mask'],z['linear_cost'],z['utility_weight'],z['utility_scale'],float(z['regularization']))


@pytest.mark.parametrize('folder',['clarabel_failure','clarabel_failure_2'])
def test_original_validation_failure_is_solved_without_tolerance_change(folder):
    p=load_regression(folder);answer=solve(p,tolerance=1e-9)
    assert p.violations(answer['allocation'])<1e-7
    assert answer['certificate_gap']<1e-8


CORPUS=sorted((Path(__file__).resolve().parents[1]/'results_r2/audit/reference_corpus_v21').glob('*/slot_*.npz'))


@pytest.mark.parametrize('path',CORPUS,ids=lambda p:p.parent.name+'_'+p.stem)
@pytest.mark.parametrize('tolerance',[1e-9,1e-10])
def test_captured_numerical_breakdowns_at_shadow_and_e9_precision(path,tolerance):
    from hbag3.probe_reference_settings_r2 import load
    p=load(path);answer=solve(p,tolerance=tolerance)
    assert p.violations(answer['allocation'])<1e-7
    assert answer['certificate_gap']<10*tolerance
    assert all(a['requested_tolerance']==tolerance for a in answer['numerical_attempts'])


def test_failed_interior_point_attempt_retries_same_problem_and_tolerance(monkeypatch):
    import cvxpy as cp
    original=cp.Problem.solve;calls=[]
    def interrupted(model,**settings):
        calls.append(settings)
        if len(calls)==1:raise cp.error.SolverError('Injected numerical breakdown')
        return original(model,**settings)
    monkeypatch.setattr(cp.Problem,'solve',interrupted)
    p=problem(.4);answer=solve(p,tolerance=1e-9)
    assert len(calls)>=2 and answer['numerical_attempts'][0]['status']=='ERROR'
    assert answer['certificate_gap']<1e-8 and p.violations(answer['allocation'])<1e-7
    assert all(c['solver']=='CLARABEL' and c['tol_gap_abs']==c['tol_gap_rel']==c['tol_feas']==1e-9 for c in calls)
