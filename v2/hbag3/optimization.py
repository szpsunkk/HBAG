"""Convex admission/routing problem and an actual primal-dual solver.

Path rates use a declared reference bit rate. Resource rows represent directed
link rates and normalized satellite electrical power. All constraints in this
module are inequalities; flow conservation is exact by the path construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np
from scipy import sparse


@dataclass
class AllocationProblem:
    resource_matrix: sparse.csr_matrix
    budgets: np.ndarray
    demand: np.ndarray
    path_mask: np.ndarray
    linear_cost: np.ndarray
    utility_weight: np.ndarray
    utility_scale: np.ndarray
    regularization: float = 0.05

    def __post_init__(self):
        self.resource_matrix = sparse.csr_matrix(self.resource_matrix, dtype=float).copy()
        self.resource_matrix.sum_duplicates()
        self.resource_matrix.eliminate_zeros()
        self.resource_matrix.sort_indices()
        self.budgets = np.asarray(self.budgets, dtype=float)
        self.demand = np.asarray(self.demand, dtype=float)
        self.path_mask = np.asarray(self.path_mask, dtype=bool).copy()
        self.linear_cost = np.asarray(self.linear_cost, dtype=float).reshape(self.path_mask.shape)
        self.utility_weight = np.broadcast_to(np.asarray(self.utility_weight, dtype=float), self.demand.shape).copy()
        self.utility_scale = np.broadcast_to(np.asarray(self.utility_scale, dtype=float), self.demand.shape).copy()
        if self.path_mask.shape[0] != self.demand.size or self.resource_matrix.shape != (self.budgets.size, self.path_mask.size):
            raise ValueError("Inconsistent problem dimensions")
        if self.regularization <= 0 or np.any(self.utility_scale <= 0):
            raise ValueError("Positive regularization and utility scale required")
        if np.any(~np.isfinite(self.resource_matrix.data)):
            raise ValueError("Nonfinite resource coefficients")
        if np.any(self.resource_matrix.data < 0) or np.any(self.demand < 0) or np.any(self.utility_weight < 0):
            raise ValueError("Nonnegative load coefficients, demands and utility weights required")
        if np.any(self.budgets < -1e-9):
            raise ValueError("Negative residual electrical budget: mandatory-load infeasibility")
        self.budgets = np.maximum(self.budgets, 0)
        if any(np.any(~np.isfinite(x)) for x in (self.budgets, self.demand, self.linear_cost, self.utility_weight, self.utility_scale)):
            raise ValueError("Nonfinite problem data")
        # K>=0 makes every path touching an exactly zero resource identically
        # zero in EVERY feasible solution. Eliminate those variables before
        # computing the norm and local dual bound. This is exact presolve, not
        # early convergence or deletion of inconvenient constraints.
        zero_rows = self.budgets == 0
        if np.any(zero_rows):
            blocked = np.asarray(self.resource_matrix[zero_rows].getnnz(axis=0)).ravel() > 0
            self.path_mask.ravel()[blocked] = False
        self.path_mask[self.demand == 0] = False
        self.resource_matrix = self.resource_matrix.multiply(self.path_mask.ravel()[None,:]).tocsr()
        self.resource_matrix.eliminate_zeros()

    @property
    def shape(self):
        return self.path_mask.shape

    def objective(self, x):
        x = np.asarray(x).reshape(self.shape)
        q = x.sum(axis=1)
        return float(np.sum(self.utility_weight * np.log1p(q / self.utility_scale))
                     - np.sum(self.linear_cost * x) - 0.5 * self.regularization * np.sum(x * x))

    def fingerprint(self):
        """SHA-256 of canonical mathematical inputs (portable little-endian bytes)."""
        digest = hashlib.sha256(b'hbag-allocation-v1')
        arrays = [self.resource_matrix.indptr, self.resource_matrix.indices,
                  self.resource_matrix.data, self.budgets, self.demand,
                  self.path_mask, self.linear_cost, self.utility_weight,
                  self.utility_scale, np.array([self.regularization])]
        for index, a in enumerate(arrays):
            a = np.asarray(a, dtype='<i8' if index in (0, 1, 5) else '<f8')
            digest.update(json.dumps(list(a.shape), separators=(',', ':')).encode())
            digest.update(a.tobytes(order='C'))
        return digest.hexdigest()

    def violations(self, x):
        x = np.asarray(x).reshape(self.shape)
        return max(0., float(np.max(self.resource_matrix @ x.ravel() - self.budgets, initial=0)),
                   float(np.max(x.sum(axis=1) - self.demand, initial=0)),
                   float(np.max(-x, initial=0)), float(np.max(np.abs(x[~self.path_mask]), initial=0)))


def _simplex_projection(v, cap, mask, equality=False):
    """Projection on x>=0, sum x<=cap separately for each commodity."""
    z = np.where(mask, np.maximum(v, 0), 0)
    needs = (np.abs(z.sum(axis=1) - cap) > 1e-14) if equality else (z.sum(axis=1) > cap)
    if not np.any(needs):
        return z
    selected = np.where(mask[needs], v[needs], -1e100)
    order = np.sort(selected, axis=1)[:, ::-1]
    cumulative = np.cumsum(order, axis=1)
    count = np.arange(1, order.shape[1] + 1)[None, :]
    theta = (cumulative - cap[needs, None]) / count
    admissible = order > theta
    rho = np.maximum(admissible.sum(axis=1) - 1, 0)
    threshold = theta[np.arange(rho.size), rho]
    z[needs] = np.where(mask[needs], np.maximum(selected - threshold[:, None], 0), 0)
    return z


def _log_quadratic_solution(a, b, scale, demand, mask):
    """Exact active-set water filling for separable log utility plus quadratic.

    The unconstrained total q satisfies q=sum_i [a_i+b/(scale+q)]_+.
    For each possible active set its quadratic equation is solved explicitly.
    If q exceeds demand, the log term is constant and simplex projection applies.
    """
    a = np.asarray(a).reshape(mask.shape)
    order = np.sort(np.where(mask, a, -1e100), axis=1)[:, ::-1]
    valid_count = mask.sum(axis=1)
    q = np.zeros(mask.shape[0])
    active_total = np.zeros_like(q)
    for j in range(mask.shape[1]):
        active_total += np.where(j < valid_count, order[:, j], 0)
        n = j + 1
        discriminant = np.sqrt((active_total + scale) ** 2 + 4 * n * b)
        trial = 0.5 * (active_total - scale + discriminant)
        shift = b / np.maximum(scale + trial, 1e-300)
        is_active = (j < valid_count) & (trial > 0) & (order[:, j] + shift > -1e-12)
        if j + 1 < mask.shape[1]:
            is_active &= ((j + 1 >= valid_count) | (order[:, j + 1] + shift <= 1e-12))
        q = np.where(is_active, trial, q)
    x = np.where(mask, np.maximum(a + (b / (scale + q))[:, None], 0), 0)
    # Degenerate all-nonpositive optimum, including zero utility weights.
    x = np.where((valid_count > 0)[:, None], x, 0)
    constrained = q > demand + 1e-12
    if np.any(constrained):
        x[constrained] = _simplex_projection(a[constrained], demand[constrained], mask[constrained], equality=True)
    return _simplex_projection(x, demand, mask)


def primal_prox(v, step_size, problem: AllocationProblem):
    step_size = np.broadcast_to(np.asarray(step_size),problem.demand.shape)
    denominator = 1 + step_size * problem.regularization
    a = (np.asarray(v).reshape(problem.shape) - step_size[:,None] * problem.linear_cost) / denominator[:,None]
    b = step_size * problem.utility_weight / denominator
    return _log_quadratic_solution(a, b, problem.utility_scale, problem.demand, problem.path_mask)


def lagrangian_minimizer(dual, problem):
    c = problem.linear_cost + (problem.resource_matrix.T @ dual).reshape(problem.shape)
    return _log_quadratic_solution(-c / problem.regularization,
                                  problem.utility_weight / problem.regularization,
                                  problem.utility_scale, problem.demand, problem.path_mask)


def dual_upper_bound(dual, problem):
    """A valid upper bound on the maximum feasible utility for any dual>=0."""
    dual = np.maximum(np.asarray(dual), 0)
    x = lagrangian_minimizer(dual, problem)
    return problem.objective(x) - float(dual @ (problem.resource_matrix @ x.ravel() - problem.budgets))


def feasible_recovery(x, problem):
    """One resource-ratio exchange makes every path allocation safe.

    Nonnegative resource coefficients are essential. Each path takes the minimum
    attenuation broadcast by all resources that it uses. Recovery never creates
    demand, reroutes partial packets, or clips battery energy.
    """
    x = _simplex_projection(np.asarray(x).reshape(problem.shape), problem.demand, problem.path_mask)
    load = problem.resource_matrix @ x.ravel()
    ratios = np.minimum(1., np.divide(problem.budgets, load, out=np.ones_like(load), where=load > 0))
    edges = problem.resource_matrix.tocoo()
    factors = np.ones(x.size)
    np.minimum.at(factors, edges.col, ratios[edges.row])
    return x * factors.reshape(problem.shape)


def global_recovery(x, problem):
    """Demand projection followed by one common attenuation factor."""
    x = _simplex_projection(np.asarray(x).reshape(problem.shape), problem.demand, problem.path_mask)
    load = problem.resource_matrix @ x.ravel()
    positive = load > 0
    factor = min(1., float(np.min(problem.budgets[positive] / load[positive], initial=1.)))
    return x * factor


def residual_fill(x, problem, sweeps=2, time_limit_s=None):
    """Feasible coordinate increments maximizing J along each chosen coordinate.

    This cannot decrease J or consume more than the remaining resource/demand
    budgets. It is a local monotone improvement, without a global quality bound.
    Input must already be feasible; construction time counts toward the deadline.
    """
    start = time.perf_counter()
    x = np.array(x, dtype=float).reshape(problem.shape).copy()
    if problem.violations(x) > 1e-8:
        raise ValueError('Residual fill requires a feasible starting action')
    matrix = problem.resource_matrix.tocsc()
    slack = np.maximum(problem.budgets - matrix @ x.ravel(), 0.)
    totals = x.sum(axis=1)
    updates = 0
    for _ in range(sweeps):
        gradient = (problem.utility_weight / (problem.utility_scale + totals))[:, None] - problem.linear_cost - problem.regularization*x
        order = np.argsort(-gradient.ravel(), kind='stable')
        changed = False
        for col in order:
            if time_limit_s is not None and time.perf_counter()-start >= time_limit_s:
                return x, dict(runtime_s=time.perf_counter()-start, updates=updates, deadline_reached=True)
            f, p = divmod(int(col), problem.shape[1])
            if not problem.path_mask[f,p]:
                continue
            a = problem.utility_scale[f]+totals[f]
            b = problem.linear_cost[f,p]+problem.regularization*x[f,p]
            if problem.utility_weight[f] / a - b <= 1e-12:
                continue
            root = (np.sqrt((b-problem.regularization*a)**2+4*problem.regularization*problem.utility_weight[f])
                    - b - problem.regularization*a)/(2*problem.regularization)
            lo, hi = matrix.indptr[col:col+2]
            rows, values = matrix.indices[lo:hi], matrix.data[lo:hi]
            cap = np.min(slack[rows]/values, initial=np.inf)
            delta = max(0., min(root, cap, problem.demand[f]-totals[f]))
            if delta <= 1e-14:
                continue
            # Evaluate the exact objective increment to guard rounding at a root.
            gain = problem.utility_weight[f]*np.log1p(delta/a)-b*delta-.5*problem.regularization*delta**2
            if gain < 0:
                continue
            x[f,p] += delta
            totals[f] += delta
            slack[rows] = np.maximum(slack[rows]-values*delta, 0.)
            updates += 1
            changed = True
        if not changed:
            break
    return x, dict(runtime_s=time.perf_counter()-start, updates=updates, deadline_reached=False)


@dataclass
class SolverResult:
    allocation: np.ndarray
    last_iterate: np.ndarray
    averaged_iterate: np.ndarray
    dual: np.ndarray
    iterations: int
    runtime_s: float
    relative_gap: float
    max_violation: float
    history: list
    norm_bound: float
    converged: bool
    certificate: dict = None


def solve_distributed(problem, max_iterations=4000, tolerance=1e-5, check_every=20, warm_start=None, diagonal_steps=True,
                      time_limit_s=None, improve_recovery=False):
    """Chambolle-Pock primal-dual iteration with local commodity proximal steps.

    This vectorized implementation emulates resource/commodity messages in one
    process. Its wall time is NOT a measured satellite-network deployment time.
    """
    start = time.perf_counter()
    if max_iterations < 1 or check_every < 1 or (time_limit_s is not None and time_limit_s <= 0):
        raise ValueError('Positive iteration/check limits and wall-clock budget required')
    K = problem.resource_matrix
    problem_hash = problem.fingerprint()
    row_sums = np.asarray(np.abs(K).sum(axis=1)).ravel()
    col_sums = np.asarray(np.abs(K).sum(axis=0)).ravel()
    row_norm = row_sums.max(initial=0.)
    col_norm = col_sums.max(initial=0.)
    norm_bound = float(np.sqrt(row_norm * col_norm))
    eta = 0.95 / max(norm_bound, 1e-8)
    if norm_bound == 0:
        x = lagrangian_minimizer(np.zeros(K.shape[0]), problem)
        value = problem.objective(x)
        certificate = dict(problem_sha256=problem.fingerprint(), dual_upper_bound=value,
                           feasible_objective=value, denominator=max(1.,abs(value)), relative_gap=0.)
        return SolverResult(x, x, x, np.zeros(K.shape[0]), 0, time.perf_counter()-start, 0., 0., [], 0., True, certificate)
    if diagonal_steps:
        # Group-constant primal steps retain the exact commodity proximal map.
        # Schur normalization gives ||S^.5 K T^.5||_2 <= .95 < 1.
        group_sum = col_sums.reshape(problem.shape).max(axis=1)
        primal_step = .95 / np.where(group_sum > 0,group_sum,1.)
        dual_step = .95 / np.where(row_sums > 0,row_sums,1.)
    else:
        primal_step = np.full(problem.shape[0],eta)
        dual_step = np.full(K.shape[0],eta)
    x = np.zeros(problem.shape) if warm_start is None else _simplex_projection(np.asarray(warm_start).reshape(problem.shape), problem.demand, problem.path_mask)
    xbar = x.copy()
    y = np.zeros(K.shape[0])
    average = np.zeros_like(x)
    history = []
    converged = False
    best = feasible_recovery(x, problem)
    best_objective = problem.objective(best)
    upper = dual_upper_bound(y, problem)
    best_dual = y.copy()
    recovery_s = 0.
    # A continuously published feasible checkpoint. If the final check crosses
    # a wall-clock deadline, the last checkpoint available by that deadline is
    # returned. Wrapper/serialization latency is reported separately by runtime.
    published = (best.copy(),best_dual.copy(),upper,best_objective,0.,0)
    for iteration in range(1, max_iterations + 1):
        y = np.maximum(y + dual_step * (K @ xbar.ravel() - problem.budgets), 0)
        new = primal_prox(x - primal_step[:,None] * (K.T @ y).reshape(problem.shape), primal_step, problem)
        xbar = 2 * new - x
        x = new
        average += (x - average) / iteration
        expired = time_limit_s is not None and time.perf_counter()-start >= time_limit_s
        if iteration == 1 or iteration % check_every == 0 or iteration == max_iterations or expired:
            recover_start = time.perf_counter()
            for candidate in (x, average):
                recovered = feasible_recovery(candidate, problem)
                value = problem.objective(recovered)
                if value >= best_objective:
                    best, best_objective = recovered, value
            recovery_s += time.perf_counter()-recover_start
            trial_upper = dual_upper_bound(y, problem)
            if trial_upper < upper:
                upper, best_dual = trial_upper, y.copy()
            if upper < best_objective - 1e-7 * max(1., abs(best_objective)):
                raise ArithmeticError("Invalid primal/dual certificate: upper bound below feasible value")
            gap = max(0., upper - best_objective)
            relative = gap / max(1., abs(best_objective))
            history.append({"iteration": iteration, "runtime_s": time.perf_counter()-start,
                            "recovery_s": recovery_s, "primal_objective": best_objective,
                            "dual_upper_bound": upper, "absolute_gap": gap, "relative_gap": relative,
                            "raw_violation": problem.violations(x), "average_violation": problem.violations(average),
                            "executed_violation": problem.violations(best),
                            "kkt_residual": max(problem.violations(x), float(np.max(np.abs(x-primal_prox(x-(K.T@y).reshape(problem.shape),1.,problem)),initial=0)), float(np.max(np.abs(y*(K@x.ravel()-problem.budgets)),initial=0)))})
            available_s = time.perf_counter()-start
            history[-1]['available_s'] = available_s
            history[-1]['delivery_rate'] = float(best.sum())
            if time_limit_s is None or available_s <= time_limit_s:
                published = (best.copy(),best_dual.copy(),upper,best_objective,available_s,iteration)
            if relative <= tolerance:
                converged = True
                break
            if expired:
                break
    if time_limit_s is not None:
        best,best_dual,upper,best_objective,available_s,published_iteration = published
        relative = max(0.,upper-best_objective)/max(1.,abs(best_objective))
        converged = relative <= tolerance
    else:
        available_s=time.perf_counter()-start
        published_iteration=iteration
    if improve_recovery:
        remaining = None if time_limit_s is None else max(0.,time_limit_s-(time.perf_counter()-start))
        improved, detail = residual_fill(best, problem, time_limit_s=remaining)
        recovery_s += detail['runtime_s']
        if time_limit_s is None or time.perf_counter()-start <= time_limit_s:
            best=improved
            best_objective = problem.objective(best)
            relative = max(0.,upper-best_objective)/max(1.,abs(best_objective))
            converged = relative <= tolerance
            available_s=time.perf_counter()-start
    certificate = dict(problem_sha256=problem_hash, dual_upper_bound=upper,
                       feasible_objective=best_objective, denominator=max(1.,abs(best_objective)),
                       relative_gap=relative, recovery_s=recovery_s, requested_time_limit_s=time_limit_s,
                       published_available_s=available_s,published_iteration=published_iteration)
    return SolverResult(best, x, average, best_dual, iteration, time.perf_counter() - start,
                        relative, problem.violations(best), history, norm_bound, converged, certificate)


def solve_centralized(problem, tolerance=1e-9,allow_degraded=False):
    """Independent Clarabel reference after exact fixed-zero-variable elimination."""
    from .central_reference_r2 import solve
    return solve(problem,tolerance,allow_degraded=allow_degraded)
