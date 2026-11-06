"""R2 conditional rate problem; fixed terminal energy is outside the objective."""
from dataclasses import replace
import numpy as np
from scipy import sparse
from .model import RATE, POWER
from .energy import load_budget
from .optimization import AllocationProblem, solve_distributed, solve_centralized, feasible_recovery
from .solvers import node_power, recover_rf, solve_rf, solve_rf_central, objective


def problem_for_switch(env, t, cat, switch, hard):
    c = env['config']; term = env['terminal']; m = env['shell'].size
    hardware = 4*(term.max_electrical_w-term.idle_w)
    increment = np.minimum(np.maximum(switch.raw_budget_w-switch.fixed_w, 0), hardware) if hard else np.full(m, hardware)
    if hard:
        increment[switch.infeasible] = 0.
    link = sparse.diags(RATE/env['catalogues'][t].link_capacity_bps) @ cat.incidence
    link_bounds = np.tile(switch.capture, 2)
    shape = cat.mask.shape
    if term.medium == 'optical':
        node = cat.sender @ sparse.diags(term.derivative(0., cat.distance_km)*RATE) @ cat.incidence
        matrix = sparse.vstack([link, node/POWER], format='csr')
        bounds = np.r_[link_bounds, increment/POWER]
        costs = np.asarray(node.T @ switch.weight).reshape(shape)/POWER
    else:
        matrix, bounds, costs = link, link_bounds, np.zeros(shape)
    p = AllocationProblem(matrix, bounds, env['demand_bps'][t]/RATE, cat.mask, costs,
                          np.full(shape[0], c.utility_s_bps/RATE),
                          np.full(shape[0], c.utility_s_bps/RATE), c.rho)
    meta = dict(incremental_w=increment, idle_w=switch.fixed_w, weight=switch.weight,
                infeasible=switch.infeasible, raw_budget_w=switch.raw_budget_w, hard=hard,
                capture=link_bounds if c.rf_capture_power and term.medium=='rf' else None)
    return p, meta


def shadow_enabled(experiment,phase,method,rho_energy):
    return experiment=='E2' and phase=='test' and method=='HBAG' and rho_energy==1.4


def reference_event(env,t,p,switch,kind,detail):
    from .environment import ROOT
    from .runner import write_json
    from .central_reference_r2 import retain_failure
    failure=ROOT/'results_r2/audit/reference_failures'/p.fingerprint()/'failure.json'
    # The strict solver already retained every failed attempt. Do not overwrite
    # that evidence with the shadow caller's shorter exception description.
    if detail.get('numerical_attempts') or not failure.exists():
        retain_failure(p,1e-9,detail.get('numerical_attempts',[]))
    out=ROOT/'results_r2/audit/reference_events'/kind/('seed_%d'%env['config'].seed)/('slot_%03d_%s.json'%(t,p.fingerprint()))
    write_json(out,dict(kind=kind,seed=env['config'].seed,rho_energy=env['config'].rho_energy,slot=t,
        problem_fingerprint=p.fingerprint(),on=switch.on,capture=switch.capture,weight=switch.weight,
        raw_budget_w=switch.raw_budget_w,price_trace=switch.price_trace,detail=detail))
    import datetime
    with (ROOT/'QUJESTIONS_R2').open('a') as log:
        log.write('\n- %s：Clarabel %s，验证/测试种子%d，rho=%g，时隙%d；按v2.2记录并继续。证据：%s\n'%(
            datetime.datetime.now().astimezone().isoformat(),kind,env['config'].seed,
            env['config'].rho_energy,t,str(out.relative_to(ROOT))))
    return str(out.relative_to(ROOT))


def allocate(env, t, cat, switch, controller,shadow=False):
    c = env['config']; term = env['terminal']
    p, meta = problem_for_switch(env, t, cat, switch, controller.hard)
    if term.medium == 'rf':
        x, info = (solve_rf_central(p, cat, term, meta) if controller.method == 'HBAG-central'
                   else solve_rf(p, cat, term, meta, c.max_iterations))
    elif controller.method == 'HBAG-central':
        r = solve_centralized(p, tolerance=1e-9,allow_degraded=True)
        x = feasible_recovery(r['allocation'], p)
        info = dict(runtime_s=r['runtime_s'], iterations=r['iterations'], gap=r['certificate_gap'], status=r['status'])
        info['node_dual'] = r['dual'][-env['shell'].size:]
        info['central_degraded'] = r['degraded']
        info['central_reference_target_met'] = r['reference_target_met']
        if r['degraded']:
            info['central_failure_event']=reference_event(env,t,p,switch,'central',
                dict(status=r['status'],certificate_gap=r['certificate_gap'],numerical_attempts=r['numerical_attempts']))
    else:
        r = solve_distributed(p, max_iterations=c.max_iterations, tolerance=c.tolerance, check_every=20)
        x = r.allocation
        info = dict(runtime_s=r.runtime_s, iterations=r.iterations, gap=r.relative_gap, status='certificate' if r.converged else 'iteration_limit')
        info['node_dual'] = r.dual[-env['shell'].size:]
    info['objective_before_protection'] = objective(p, x, cat, term, meta)
    if shadow:
        if controller.method!='HBAG' or term.medium!='optical' or c.rho_energy!=1.4:
            raise ValueError('Shadow reference is only declared for optical E2 HBAG at rho1.4')
        import time
        started=time.perf_counter()
        try:
            shadow = solve_centralized(p, tolerance=1e-9)
            sx = feasible_recovery(shadow['allocation'], p)
            shadow_value = p.objective(sx)
            info['shadow_runtime_s'] = shadow['runtime_s']
            info['shadow_solver_status'] = shadow['status']
            info['shadow_certificate_gap'] = shadow['certificate_gap']
            info['shadow_iterations'] = shadow['iterations']
            info['shadow_numerical_attempts'] = len(shadow['numerical_attempts'])
            info['shadow_reference_target_met'] = shadow['reference_target_met']
            info['shadow_failed']=False
            info['shadow_objective'] = shadow_value
            info['shadow_objective_gap'] = shadow_value-info['objective_before_protection']
            info['shadow_relative_gap'] = info['shadow_objective_gap']/max(1., abs(shadow_value))
        except (RuntimeError,ArithmeticError) as error:
            info.update(shadow_failed=True,shadow_solver_status='failed',shadow_runtime_s=time.perf_counter()-started,
                shadow_objective=None,shadow_objective_gap=None,shadow_relative_gap=None,shadow_certificate_gap=None)
            info['shadow_failure_event']=reference_event(env,t,p,switch,'shadow',dict(error=str(error)))
        info['problem_fingerprint'] = p.fingerprint()
    return x, p, meta, info


def protect_bus(env, t, energy, cat, switch, x, p, meta):
    """Identical physical protection; never clips stored energy at C_min."""
    c = env['config']; term = env['terminal']; b = env['battery']; m = len(energy)
    available = load_budget(energy, 0., env['harvest'][t], env['dt'], b)-c.base_w-switch.fixed_w
    actual = np.minimum(meta['incremental_w'], np.maximum(available, 0.))
    if term.medium == 'optical':
        bounds = p.budgets.copy(); bounds[-m:] = np.minimum(bounds[-m:], actual/POWER)
        physical = replace(p, budgets=bounds)
        x = feasible_recovery(x, physical)
        violation = physical.violations(x)
    else:
        x = recover_rf(x, p, cat, term, actual,meta.get('capture'))
        violation = max(p.violations(x), float(np.max(node_power(cat, term, x, switch.fixed_w,meta.get('capture'))-switch.fixed_w-actual, initial=0))/POWER)
    return x, node_power(cat, term, x, switch.fixed_w,meta.get('capture')), violation
