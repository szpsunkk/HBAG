"""Equivalent Clarabel formulation in dimensionless bounded path coordinates."""
import time
import numpy as np
from scipy import sparse

# Fixed numerical settings only: same Clarabel, objective, constraints and
# requested tolerances in every attempt. A shorter interior-point step avoids
# the numerical breakdowns retained in the validation regression corpus.
NUMERICAL_ATTEMPTS = (
    dict(max_step_fraction=.90),
    dict(max_step_fraction=.95),
    dict(max_step_fraction=.90,static_regularization_constant=1e-10),
    dict(max_step_fraction=.99,static_regularization_constant=1e-10),
)


def retain_failure(problem,tolerance,attempts):
    from .environment import ROOT
    from .runner import write_json
    from .diagnose_reference_v21 import snapshot
    out=ROOT/'results_r2/audit/reference_failures'/problem.fingerprint()
    snapshot(problem,out/'exact_problem_float64.npz')
    write_json(out/'failure.json',dict(problem_fingerprint=problem.fingerprint(),
        tolerance=tolerance,attempts=attempts,solver='CLARABEL',
        purpose='Exact numerical regression fixture, not a performance observation'))


def failed_iterate(model,chain,inverse_data,variable,constraints):
    """Recover finite last iterates even when CVXPY rejects a solver status.

    CVXPY normally discards NumericalError/InsufficientProgress solutions. Its
    inverse reductions still map these coordinates exactly. The temporary
    status below only enables that coordinate map; the ACTUAL solver status is
    returned and always marked degraded, never presented as an optimal solve.
    """
    from types import SimpleNamespace
    raw=model._solver_cache.get('CLARABEL')
    if raw is None or not np.all(np.isfinite(raw.x)) or not np.all(np.isfinite(raw.z)):return None
    coordinates=SimpleNamespace(status='AlmostSolved',x=raw.x,z=raw.z,obj_val=raw.obj_val,
        solve_time=raw.solve_time,iterations=raw.iterations)
    restored=chain.invert(coordinates,inverse_data)
    return (restored.primal_vars[variable.id],
            [restored.dual_vars.get(c.id) for c in constraints],str(raw.status),int(raw.iterations))


def solve(problem,tolerance=1e-9,allow_degraded=False):
    from .optimization import feasible_recovery,dual_upper_bound
    import cvxpy as cp
    start=time.perf_counter();K=problem.resource_matrix.tocsr();f,k=problem.shape
    # All resource coefficients are nonnegative. A nonnegative variable using
    # a positive coefficient in an exactly zero-budget row is exactly zero.
    # No numerical tolerance is used for this feasibility-preserving reduction.
    if np.any(K.data<0):raise ValueError('Zero-budget elimination requires nonnegative resources')
    zero=problem.budgets==0
    positive=K.copy();positive.data=(positive.data>0).astype(float);positive.eliminate_zeros()
    fixed_by_zero=np.asarray(positive[zero].sum(axis=0)).ravel()>0
    allowed=problem.path_mask.ravel()&np.repeat(problem.demand>0,k)&~fixed_by_zero
    columns=np.flatnonzero(allowed);y=np.zeros(len(problem.budgets));solution=np.zeros(problem.shape)
    # Restore dual prices of removed zero-budget rows. These prices make every
    # removed otherwise-allowed path unattractive in the ORIGINAL Lagrangian,
    # hence its original primal/dual certificate remains meaningful.
    marginal=np.repeat(problem.utility_weight/problem.utility_scale,k)-problem.linear_cost.ravel()
    for row in np.flatnonzero(zero):
        lo,hi=K.indptr[row:row+2];ids=K.indices[lo:hi];data=K.data[lo:hi]
        use=(data>0)&problem.path_mask.ravel()[ids]
        if np.any(use):y[row]=float(np.max(np.maximum(marginal[ids[use]],0)/data[use],initial=0))
    attempts=[];best=None;last_iterate_best=None
    if len(columns):
        reduced=K[:,columns];rows=np.asarray(reduced.getnnz(axis=1)>0)&~zero
        upper=problem.demand[columns//k].copy()
        entries=reduced.tocoo();nonzero=entries.data>0
        np.minimum.at(upper,entries.col[nonzero],problem.budgets[entries.row[nonzero]]/entries.data[nonzero])
        if np.any(upper<=0):raise ArithmeticError('Nonpositive bound survived exact zero elimination')
        normalized=sparse.diags(1/problem.budgets[rows])@reduced[rows]@sparse.diags(upper)
        x=cp.Variable(len(columns),nonneg=True)
        group=sparse.coo_matrix((upper,(columns//k,np.arange(len(columns)))),shape=(f,len(columns))).tocsr()
        used_flows=np.asarray(group.getnnz(axis=1)>0);q=group[used_flows]@x
        objective=cp.sum(cp.multiply(problem.utility_weight[used_flows],cp.log1p(cp.multiply(1/problem.utility_scale[used_flows],q))))
        objective-=(problem.linear_cost.ravel()[columns]*upper)@x+.5*problem.regularization*cp.sum(cp.multiply(upper**2,cp.square(x)))
        constraints=[cp.multiply(1/problem.demand[used_flows],q)<=1.]
        if rows.any():constraints.append(normalized@x<=1.)
        model=cp.Problem(cp.Maximize(objective),constraints)
        _,chain,inverse_data=model.get_problem_data('CLARABEL')
        zero_dual=y.copy()
        def candidate(values,duals,status,iterations,settings,degraded):
            if values is None or not np.all(np.isfinite(values)):return None
            solution=np.zeros(problem.shape);solution.ravel()[columns]=upper*np.maximum(values,0)
            y=zero_dual.copy()
            if rows.any() and duals[1] is not None and np.all(np.isfinite(duals[1])):
                y[rows]=np.maximum(duals[1],0)/problem.budgets[rows]
            feasible=feasible_recovery(solution,problem);bound=dual_upper_bound(y,problem)
            objective=problem.objective(feasible)
            if not np.isfinite(bound) or bound<objective-1e-7*max(1.,abs(objective)):return None
            gap=max(0.,bound-objective)/max(1.,abs(objective))
            return dict(allocation=solution,objective=problem.objective(solution),max_violation=problem.violations(solution),
                status=status,iterations=iterations,dual=y,dual_upper_bound=bound,
                certificate_gap=gap,numerical_settings=settings,degraded=degraded)
        for settings in NUMERICAL_ATTEMPTS:
            attempt=dict(settings=settings,requested_tolerance=tolerance)
            model._solver_cache.pop('CLARABEL',None)
            try:
                value=model.solve(solver='CLARABEL',tol_gap_abs=tolerance,tol_gap_rel=tolerance,
                    tol_feas=tolerance,max_iter=1000,warm_start=False,**settings)
                if model.status not in ('optimal','optimal_inaccurate') or x.value is None:
                    raise RuntimeError('Clarabel status: '+str(model.status))
                result=candidate(x.value,[c.dual_value for c in constraints],model.status,
                    int(model.solver_stats.num_iters or 0),settings,False)
                if result is None:
                    raise ArithmeticError('Invalid original-problem Clarabel dual certificate')
                attempt.update(status=model.status,certificate_gap=result['certificate_gap'],iterations=result['iterations'])
                if best is None or result['certificate_gap']<best['certificate_gap']:best=result
            except (cp.error.SolverError,RuntimeError,ArithmeticError) as error:
                attempt.update(status='ERROR',error=str(error))
                try:
                    raw=failed_iterate(model,chain,inverse_data,x,constraints)
                    result=candidate(*raw,settings,True) if raw is not None else None
                    if result is not None:
                        attempt.update(last_iterate_status=result['status'],last_iterate_gap=result['certificate_gap'])
                        if last_iterate_best is None or result['certificate_gap']<last_iterate_best['certificate_gap']:
                            last_iterate_best=result
                except (ValueError,KeyError,TypeError,ArithmeticError) as mapping_error:
                    attempt['last_iterate_mapping_error']=str(mapping_error)
            attempts.append(attempt)
            if best is not None and best['certificate_gap']<=tolerance:break
        if best is None:
            retain_failure(problem,tolerance,attempts)
            if not allow_degraded:
                raise RuntimeError('Independent Clarabel failed all fixed numerical attempts at unchanged tolerance; exact problem retained under results_r2/audit/reference_failures')
            best=last_iterate_best
            if best is None:
                # No finite solver iterate exists to recover. Zero rates are
                # feasible after A1, so use the explicit conservative safe action.
                solution=np.zeros(problem.shape);dual=np.zeros(len(problem.budgets))
                bound=dual_upper_bound(dual,problem)
                best=dict(allocation=solution,objective=0.,max_violation=0.,status='no_finite_iterate_zero_action',
                    iterations=0,dual=dual,dual_upper_bound=bound,certificate_gap=max(bound,0.),
                    numerical_settings={},degraded=True)
    else:
        best=dict(allocation=solution,objective=problem.objective(solution),max_violation=0.,status='optimal',
            iterations=0,dual=y,dual_upper_bound=dual_upper_bound(y,problem),certificate_gap=0.,numerical_settings={},degraded=False)
    best.update(runtime_s=time.perf_counter()-start,solver='CLARABEL',fallback_reason=None,
        numerical_attempts=attempts,reference_target_met=best['certificate_gap']<=tolerance,
        fixed_zero_variables=int(problem.path_mask.size-len(columns)),exact_zero_budget_rows=int(zero.sum()),
        formulation='Exact fixed-zero elimination; x_j=upper_j*z_j, resource and demand rows normalized by original positive bounds; original dual units reconstructed')
    return best
