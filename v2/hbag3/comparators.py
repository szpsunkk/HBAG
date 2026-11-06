"""Explicit comparison mechanisms; none inherits CEAR's competitive guarantee."""
import time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from .optimization import feasible_recovery


def maximum_service(problem):
    start=time.perf_counter()
    groups=sparse.kron(sparse.eye(problem.shape[0]),np.ones((1,problem.shape[1])),format='csr')
    result=linprog(-np.ones(problem.path_mask.size),
                   A_ub=sparse.vstack([problem.resource_matrix,groups]),
                   b_ub=np.r_[problem.budgets,problem.demand],
                   bounds=[(0,None if v else 0) for v in problem.path_mask.ravel()],method='highs')
    if not result.success:raise RuntimeError(result.message)
    x=feasible_recovery(result.x.reshape(problem.shape),problem)
    return x,dict(runtime_s=time.perf_counter()-start,status=result.message)


def cear_slot_adapter(problem, chunks=16, mu=8., order_seed=0):
    """One-slot, divisible-demand adapter of CEAR exponential resource pricing.

    Changes relative to Wang et al. ICDCS 2025 Algorithm 1:
    split demand into chunks; use incremental J as valuation; apply equal
    reserve-derived resource bounds; restrict to the supplied path catalogue;
    use normalized load/budget utilization for both resource classes. No future
    request reservation, deficit propagation, or competitive theorem is claimed.
    Order and mu are declared before evaluation; no future or optimum is used.
    """
    if chunks<1 or mu<=1:raise ValueError('Positive chunks and mu>1 required')
    start=time.perf_counter();K=problem.resource_matrix.tocsc()
    x=np.zeros(problem.shape);load=np.zeros_like(problem.budgets)
    order=np.random.default_rng(order_seed).permutation(problem.shape[0])
    accepted=0
    for _ in range(chunks):
        for f in order:
            delta=problem.demand[f]/chunks
            if delta<=0:continue
            best=None
            utilization=np.divide(load,problem.budgets,out=np.zeros_like(load),where=problem.budgets>0)
            prices=np.expm1(np.log(mu)*np.minimum(utilization,1.))
            for p in np.flatnonzero(problem.path_mask[f]):
                col=f*problem.shape[1]+p;lo,hi=K.indptr[col:col+2]
                rows,values=K.indices[lo:hi],K.data[lo:hi]
                use=values*delta
                if np.any(load[rows]+use>problem.budgets[rows]+1e-12):continue
                price=float(prices[rows]@use)
                valuation=(problem.utility_weight[f]*np.log1p(delta/(problem.utility_scale[f]+x[f].sum()))
                           -problem.linear_cost[f,p]*delta-problem.regularization*(x[f,p]*delta+.5*delta**2))
                if valuation>=price and (best is None or price<best[0]):best=(price,p,rows,use)
            if best is not None:
                _,p,rows,use=best;x[f,p]+=delta;load[rows]+=use;accepted+=1
    return x,dict(runtime_s=time.perf_counter()-start,accepted_chunks=accepted,chunks=chunks,mu=mu,order_seed=order_seed)
