"""v2.3 constant excess-power certificate; eclipse evaluation is closed form."""
import numpy as np
from .energy import energy_drift, reserve_envelope


def reserve_for_price(env):
    if 'reserve_extended' in env:
        return env['reserve_extended']
    harvest=env.get('harvest_extended',env['harvest'])
    if len(env['reserve'])>=len(harvest)+1:
        return env['reserve']
    c=env['config']
    return reserve_envelope(harvest,c.base_w+4*c.sleep_w,env['dt'],env['battery'])


def sustainable_sun_power(env,t,energy,end,nodes):
    """14 bisections, only for sunlit nodes failing reset condition (a).

    The backward certificate checks *every* intermediate reserve, including
    the terminal reserve; initial energy alone cannot cover an oversized
    eclipse reserve behind a sunny charging interval.
    """
    nodes=np.asarray(nodes,dtype=int)
    if not len(nodes):return np.empty(0)
    c=env['config'];b=env['battery'];dt=env['dt']
    harvest=env.get('harvest_extended',env['harvest'])
    reserve=reserve_for_price(env)
    end=np.minimum(np.asarray(end,dtype=int)[nodes],min(len(harvest),len(reserve)-1))
    cap=4*(env['terminal'].max_electrical_w-c.sleep_w)
    initial=np.asarray(energy)[nodes]
    def feasible(power):
        need=reserve[end,nodes].astype(float).copy()
        valid=need<=b.capacity_j+1e-6
        for tau in range(int(end.max())-1,t-1,-1):
            active=tau<end
            need=np.where(active,np.maximum(b.minimum_j,need-dt*energy_drift(
                harvest[tau,nodes],c.base_w+4*c.sleep_w+power,b)),need)
            valid &= ~active | (need<=b.capacity_j+1e-6)
        return valid & (initial>=need-1e-6)
    lo=np.zeros(len(nodes));hi=np.full(len(nodes),cap)
    feasible_zero=feasible(lo);feasible_cap=feasible(hi)
    for _ in range(14):
        mid=(lo+hi)*.5;ok=feasible(mid)
        lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
    return np.where(feasible_cap,cap,np.where(feasible_zero,lo,0.))


def eclipse_power(z,remaining,dt,efficiency,cap):
    return np.clip(efficiency*np.maximum(z,0.)/(np.maximum(remaining,1)*dt),0.,cap)
