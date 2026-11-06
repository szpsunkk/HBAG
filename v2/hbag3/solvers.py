"""Independent central solver, price decomposition, ADMM, and nonlinear RF PDHG."""
from dataclasses import replace
import time
import numpy as np
from scipy import sparse
from scipy.special import lambertw
from .model import RATE,POWER
from .optimization import (AllocationProblem,solve_distributed,solve_centralized,feasible_recovery,
    primal_prox,lagrangian_minimizer,dual_upper_bound,_simplex_projection)


def make_problem(env,t,energy,method):
    c=env['config'];cat=env['catalogues'][t];term=env['terminal'];b=env['battery'];m=env['shell'].size
    from .energy import load_budget
    hard=method in ('HBAG','V1','Centralized-opt','DualDecomp','ADMM','MPC-H','PPO-shared')
    soft=method in ('HBAG','V2','Centralized-opt','DualDecomp','ADMM','MPC-H')
    raw_budget=load_budget(energy,env['reserve'][t+1],env['harvest'][t],env['dt'],b)-c.base_w
    idle=np.full(m,4*term.idle_w)
    off=(energy<b.minimum_j-1e-6) if not hard else np.zeros(m,bool)
    if method=='Static+Guard': off|=energy/b.capacity_j<c.guard_soc
    idle[off]=0.
    incremental=np.maximum(raw_budget-idle,0.) if hard else np.full(m,4*(term.max_electrical_w-term.idle_w))
    incremental=np.minimum(incremental,4*(term.max_electrical_w-term.idle_w))
    incremental[off]=0.
    infeasible=(raw_budget<idle-1e-6)|(env['reserve'][t+1]>b.capacity_j+1e-6)
    if hard: incremental[infeasible]=0.
    margin=np.maximum((energy-env['reserve'][t])/(b.capacity_j-b.minimum_j),0.)
    weight=c.alpha+(c.lam/(margin+c.epsilon) if soft else np.zeros(m))
    shape=cat.mask.shape
    link=sparse.diags(RATE/cat.link_capacity_bps)@cat.incidence
    if term.medium=='optical':
        node=cat.sender@sparse.diags(term.derivative(np.zeros(len(cat.edges)),cat.distance_km)*RATE)@cat.incidence
        K=sparse.vstack([link,node/POWER],format='csr')
        cost=np.asarray(node.T@weight).reshape(shape)/POWER
        p=AllocationProblem(K,np.r_[np.ones(len(cat.edges)),incremental/POWER],env['demand_bps'][t]/RATE,cat.mask,cost,np.full(shape[0],c.utility_s_bps/RATE),np.full(shape[0],c.utility_s_bps/RATE),c.rho)
    else:
        # Link-only problem supplies commodity proximal operator; power is treated exactly by RF node prox.
        p=AllocationProblem(link,np.ones(len(cat.edges)),env['demand_bps'][t]/RATE,cat.mask,np.zeros(shape),np.full(shape[0],c.utility_s_bps/RATE),np.full(shape[0],c.utility_s_bps/RATE),c.rho)
        used=cat.sender@cat.incidence
        blocked=np.asarray(used[incremental<=1e-9].getnnz(axis=0)).ravel()>0
        p.path_mask.ravel()[blocked]=False
    return p,dict(incremental_w=incremental,idle_w=idle,weight=weight,infeasible=infeasible,raw_budget_w=raw_budget,off=off,hard=hard)


def node_power(cat,term,x,idle,capture=None):
    r=np.asarray(cat.incidence@x.ravel())*RATE
    return idle+cat.sender@term.slot_increment(r,cat.distance_km,capture)


def recover_rf(x,p,cat,term,budget,capture=None):
    x=feasible_recovery(x,p)
    r=np.asarray(cat.incidence@x.ravel())*RATE
    load=cat.sender@term.slot_increment(r,cat.distance_km,capture)
    # Convex q(0)=0: q(a*r)<=a*q(r). The minimum incident-node ratio is conservative and exact-safe.
    ratio=np.minimum(1.,np.divide(budget,load,out=np.ones_like(load),where=load>0))
    use=(cat.sender@cat.incidence).tocoo();factor=np.ones(x.size)
    np.minimum.at(factor,use.col,ratio[use.row])
    return x*factor.reshape(x.shape)


def objective(p,x,cat=None,term=None,meta=None):
    value=p.objective(x)
    if term is not None and term.medium=='rf': value-=float(meta['weight']@(node_power(cat,term,x,np.zeros(len(meta['weight'])),meta.get('capture'))))/POWER
    return value


def solve_dual(p,iterations,tolerance=1e-3,record=False):
    start=time.perf_counter();K=p.resource_matrix;row=np.asarray(K.sum(axis=1)).ravel();col=np.asarray(K.sum(axis=0)).ravel()
    eta=.95*p.regularization/max(float(row.max(initial=0)*col.max(initial=0)),1e-12)
    y=np.zeros(K.shape[0]);best=np.zeros(p.shape);value=p.objective(best);history=[]
    for it in range(1,iterations+1):
        x=lagrangian_minimizer(y,p)
        y=np.maximum(y+eta*(K@x.ravel()-p.budgets),0.)
        if it%20==0 or it in (1,iterations):
            z=feasible_recovery(x,p);v=p.objective(z)
            if v>=value:best=z;value=v
            gap=(dual_upper_bound(y,p)-value)/max(1,abs(value))
            history.append(dict(iteration=it,objective=value,raw_violation=p.violations(x),kkt=kkt_residual(p,x,y),gap=gap))
            if gap<=tolerance:break
    return best,dict(runtime_s=time.perf_counter()-start,iterations=it,history=history,dual=y,gap=gap)


def solve_admm(p,iterations,tolerance=1e-3,record=False):
    """Linearized two-block ADMM: v=Kx, v<=b, exact commodity proximal step."""
    start=time.perf_counter();K=p.resource_matrix;row=np.asarray(K.sum(axis=1)).ravel();col=np.asarray(K.sum(axis=0)).ravel()
    eta=.9/max(float(row.max(initial=0)*col.max(initial=0)),1.)
    x=np.zeros(p.shape);v=np.zeros(K.shape[0]);u=np.zeros_like(v);best=x.copy();value=p.objective(best);history=[]
    for it in range(1,iterations+1):
        x=primal_prox(x-eta*(K.T@(K@x.ravel()-v+u)).reshape(p.shape),eta,p)
        v=np.minimum(K@x.ravel()+u,p.budgets)
        u+=K@x.ravel()-v
        if it%20==0 or it in (1,iterations):
            z=feasible_recovery(x,p);val=p.objective(z)
            if val>=value:best=z;value=val
            y=np.maximum(u,0);gap=(dual_upper_bound(y,p)-value)/max(1,abs(value))
            history.append(dict(iteration=it,objective=value,raw_violation=p.violations(x),kkt=kkt_residual(p,x,y),gap=gap))
            if gap<=tolerance:break
    return best,dict(runtime_s=time.perf_counter()-start,iterations=it,history=history,dual=np.maximum(u,0),gap=gap)


def kkt_residual(p,x,y):
    # Fixed-point stationarity for the commodity feasible set, dual complementarity and primal feasibility.
    station=float(np.max(np.abs(x-primal_prox(x-(p.resource_matrix.T@y).reshape(p.shape),1.,p)),initial=0))
    complement=float(np.max(np.abs(y*(p.resource_matrix@x.ravel()-p.budgets)),initial=0))
    return max(station,complement,p.violations(x),float(np.max(-y,initial=0)))


def solve_rf(p,cat,term,meta,iterations):
    start=time.perf_counter();A=cat.incidence.multiply(p.path_mask.ravel()[None,:]).tocsr();A.eliminate_zeros()
    rows=np.asarray(A.sum(axis=1)).ravel();cols=np.asarray(A.sum(axis=0)).ravel()
    eta=.95/np.sqrt(max(float(rows.max(initial=0)*cols.max(initial=0)),1.))
    sender=cat.edges[:,0];cap=cat.link_capacity_bps/RATE
    a=np.log(2)*RATE/term.rf_bandwidth_hz;coef=term.kappa(cat.distance_km)/term.rf_eta_pa/POWER
    if meta.get('capture') is not None:
        capture=np.asarray(meta['capture'])
        a=a/np.where(capture>0,capture,1.)
        coef=coef*capture
    budget=meta['incremental_w']/POWER;w=meta['weight'];y=np.zeros(len(cat.edges));x=np.zeros(p.shape);xbar=x.copy();best=x.copy();bestval=objective(p,x,cat,term,meta)
    def prox(v,nu):
        factor=(w[sender]+nu[sender])/eta*coef*a*a
        # log-space stable principal Lambert W; v is bounded before exponent to avoid overflow.
        arg=np.maximum(factor,1e-300)*np.exp(np.minimum(a*v,600))
        return np.clip(v-np.real(lambertw(arg))/a,0,cap)
    for it in range(1,iterations+1):
        z=y+eta*(A@xbar.ravel());v=z/eta
        lo=np.zeros_like(budget);hi=np.ones_like(budget)
        r=prox(v,lo);load=cat.sender@(coef*np.expm1(a*r));active=load>budget+1e-10
        for _ in range(24):
            rr=prox(v,hi);bad=(cat.sender@(coef*np.expm1(a*rr)))>budget+1e-10
            if not np.any(bad&active):break
            hi[bad&active]*=4
        for _ in range(24):
            mid=(lo+hi)/2;rr=prox(v,mid);bad=(cat.sender@(coef*np.expm1(a*rr)))>budget
            lo=np.where(active&bad,mid,lo);hi=np.where(active&~bad,mid,hi)
        r=prox(v,np.where(active,hi,0));y=z-eta*r
        new=primal_prox(x-eta*(A.T@y).reshape(p.shape),eta,p);xbar=2*new-x;x=new
        if it%20==0 or it in (1,iterations):
            zsafe=recover_rf(x,p,cat,term,meta['incremental_w'],meta.get('capture'));val=objective(p,zsafe,cat,term,meta)
            if val>bestval:best=zsafe;bestval=val
    return best,dict(runtime_s=time.perf_counter()-start,iterations=it,gap=None,history=[],dual=y)


def solve_rf_central(p,cat,term,meta):
    import cvxpy as cp
    start=time.perf_counter();x=cp.Variable(p.shape,nonneg=True);r=cat.incidence@cp.reshape(x,(x.size,),order='C')*RATE
    capture=np.ones(len(cat.edges)) if meta.get('capture') is None else np.asarray(meta['capture'])
    exponent=np.log(2)/term.rf_bandwidth_hz/np.where(capture>0,capture,1.)
    power=cp.multiply(capture*term.kappa(cat.distance_km)/term.rf_eta_pa,cp.exp(cp.multiply(exponent,r))-1)
    q=cat.sender@power
    value=cp.sum(cp.multiply(p.utility_weight,cp.log1p(cp.multiply(1/p.utility_scale,cp.sum(x,axis=1)))))-.5*p.regularization*cp.sum_squares(x)-meta['weight']@q/POWER
    con=[cp.sum(x,axis=1)<=p.demand,r<=cat.link_capacity_bps,q<=meta['incremental_w']]
    if (~p.path_mask).any():con.append(x[~p.path_mask]==0)
    mod=cp.Problem(cp.Maximize(value),con);mod.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_feas=1e-8,tol_gap_rel=1e-8,max_iter=500)
    if x.value is None:raise RuntimeError('RF central '+mod.status)
    z=recover_rf(np.maximum(x.value,0),p,cat,term,meta['incremental_w'],meta.get('capture'))
    return z,dict(runtime_s=time.perf_counter()-start,iterations=mod.solver_stats.num_iters,status=mod.status,gap=None,history=[])


def solve(env,t,energy,method):
    p,meta=make_problem(env,t,energy,method);c=env['config'];cat=env['catalogues'][t];term=env['terminal']
    if term.medium=='rf':
        if method=='Centralized-opt': x,info=solve_rf_central(p,cat,term,meta)
        else:x,info=solve_rf(p,cat,term,meta,c.max_iterations)
    elif method=='Centralized-opt':
        result=solve_centralized(p);x=feasible_recovery(result['allocation'],p)
        info=dict(runtime_s=result['runtime_s'],iterations=result['iterations'],status=result['status'],gap=result['certificate_gap'],history=[])
    elif method=='DualDecomp':x,info=solve_dual(p,c.max_iterations,c.tolerance)
    elif method=='ADMM':x,info=solve_admm(p,c.max_iterations,c.tolerance)
    elif method=='MPC-H':x,info=solve_mpc(env,t,energy,p,meta)
    else:
        if method=='PPO-shared':
            from .ppo import fractions
            policy_fraction=fractions(env,t,energy)
            admissions=np.where(cat.sources>=0,policy_fraction[np.maximum(cat.sources,0)],0.)
            p=replace(p,demand=p.demand*admissions)
        result=solve_distributed(p,max_iterations=c.max_iterations,tolerance=c.tolerance,check_every=20)
        x=result.allocation;info=dict(runtime_s=result.runtime_s,iterations=result.iterations,gap=result.relative_gap,history=result.history)
    info['objective']=objective(p,x,cat,term,meta)
    return x,p,meta,info


def solve_mpc(env,t,energy,p,meta,allow_separable=True):
    """Rolling convex horizon through a full next orbit; conditional forecast benchmark.

    Exact same optical routing model at each prediction slot. Energy inequalities
    are a conservative hypograph of the piecewise-affine battery drift; arbitrary
    predicted curtailment is allowed. Only the first slot is executed.
    """
    import cvxpy as cp
    start=time.perf_counter();c=env['config'];m=env['shell'].size;b=env['battery'];dt=env['dt'];term=env['terminal']
    horizon=min(int(np.ceil(env['shell'].period_s/dt)),len(env['times'])-t)
    # A mathematical fast path, not a shorter horizon: if the independent
    # per-slot optima satisfy every intertemporal battery inequality, they also
    # solve the coupled horizon problem. Check the whole horizon explicitly.
    # This is especially useful when the mandatory reserve is the main load.
    if allow_separable:
        from .energy import step
        predicted=[np.asarray(energy).copy()];plans=[];values=[];gaps=[];iterations=0
        for j in range(horizon):
            cat=env['catalogues'][t+j]
            node=cat.sender@sparse.diags(term.derivative(np.zeros(len(cat.edges)),cat.distance_km)*RATE)@cat.incidence
            link=sparse.diags(RATE/cat.link_capacity_bps)@cat.incidence
            independent=AllocationProblem(sparse.vstack([link,node/POWER],format='csr'),np.r_[np.ones(len(cat.edges)),np.full(m,4*(term.max_electrical_w-term.idle_w)/POWER)],env['demand_bps'][t+j]/RATE,cat.mask,np.asarray(node.T@meta['weight']).reshape(cat.mask.shape)/POWER,np.full(c.flows,c.utility_s_bps/RATE),np.full(c.flows,c.utility_s_bps/RATE),c.rho)
            optimum=solve_centralized(independent,tolerance=1e-9)
            iterations+=optimum['iterations']
            action=feasible_recovery(optimum['allocation'],independent)
            q=4*term.idle_w+node@action.ravel()
            after=step(predicted[-1],env['harvest'][t+j],c.base_w,q,dt,b).after_j
            if np.any(after<env['reserve'][t+j+1]-1e-6):break
            predicted.append(after);plans.append(action);values.append(independent.objective(action)-float(meta['weight'].sum()*4*term.idle_w/POWER));gaps.append(optimum['certificate_gap'])
        if len(plans)==horizon:
            return feasible_recovery(plans[0],p),dict(runtime_s=time.perf_counter()-start,iterations=iterations,status='optimal_separable_horizon_certificate',gap=max(gaps),history=[],horizon=horizon,forecast_objective=float(sum(values)),predicted_energy_j=np.asarray(predicted),forecast_allocation_bps=np.asarray(plans)*RATE,forecast='All independent slot optima verified against the entire rolling battery horizon; same MPC problem, no horizon reduction.')
    # At the end of the run reserve enforces survival beyond the available forecast.
    e=cp.Variable((horizon+1,m));con=[e[0]==energy/1e6,e>=b.minimum_j/1e6,e<=b.capacity_j/1e6];value=0.;xs=[]
    for j in range(horizon):
        cat=env['catalogues'][t+j];x=cp.Variable(cat.mask.shape,nonneg=True);xs.append(x);flat=cp.reshape(x,(x.size,),order='C')
        node=cat.sender@sparse.diags(term.derivative(np.zeros(len(cat.edges)),cat.distance_km)*RATE)@cat.incidence
        power=4*term.idle_w+node@flat;surplus=env['harvest'][t+j]-c.base_w-power
        con.extend([cp.sum(x,axis=1)<=env['demand_bps'][t+j]/RATE,cat.incidence@flat<=cat.link_capacity_bps/RATE,
            e[j+1]<=e[j]+dt*b.charge_efficiency*surplus/1e6,e[j+1]<=e[j]+dt/b.discharge_efficiency*surplus/1e6,
            e[j+1]>=env['reserve'][t+j+1]/1e6,power<=4*term.max_electrical_w])
        if (~cat.mask).any():con.append(x[~cat.mask]==0)
        # Current known price held across the prediction horizon; oracle demand is labelled explicitly.
        value+=(c.utility_s_bps/RATE)*cp.sum(cp.log1p(cp.sum(x,axis=1)/(c.utility_s_bps/RATE)))-meta['weight']@power/POWER-.5*c.rho*cp.sum_squares(x)
    mod=cp.Problem(cp.Maximize(value),con)
    mod.solve(solver='CLARABEL',tol_gap_abs=1e-6,tol_gap_rel=1e-6,tol_feas=1e-7,max_iter=200)
    if xs[0].value is None:raise RuntimeError('MPC infeasible: '+mod.status)
    x=feasible_recovery(np.maximum(xs[0].value,0),p)
    return x,dict(runtime_s=time.perf_counter()-start,iterations=mod.solver_stats.num_iters,status=mod.status,gap=None,history=[],horizon=horizon,forecast_objective=float(mod.value),predicted_energy_j=np.asarray(e.value)*1e6,forecast_allocation_bps=np.array([np.maximum(v.value,0) for v in xs])*RATE,forecast='oracle realized demand; optimistic reference, no optimality upper-bound claim for Delivered')
