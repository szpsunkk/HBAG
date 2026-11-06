"""R2 v2 orbital-window dual pacing and the two separately tuned price baselines."""
import numpy as np


METHODS={
    'V0':(False,'zero'), 'Static+Guard':(False,'zero'), 'Util-Sleep':(False,'zero'),
    'V1':(True,'zero'), 'V2':(False,'dual'), 'HBAG':(True,'dual'),
    'HBAG-central':(True,'dual'), 'HBAG+margin':(True,'dual'),
    'HBAG-xi':(True,'xi'), 'HBAG-ξ':(True,'xi'),
    'Lyapunov':(False,'lyapunov'), 'HBAG-Lyap':(True,'lyapunov'),
}


class OrbitalPrice:
    def __init__(self,config,method):
        self.c=config
        self.hard,self.kind=METHODS[method]
        m=config.planes*config.per_plane
        self.mu=np.zeros(m)
        self.last_window_end=np.zeros(m)
        self.previous_dark=np.zeros(m,bool)
        self.ready=False
        self.last=None
        self.var_ema=np.full(m,8.)
        self.idle_w=None

    def prepare(self,env,t,energy):
        c=self.c;b=env['battery'];dt=env['dt'];m=len(energy)
        if not self.ready:
            h=env.get('harvest_extended',env['harvest'])
            dark=h==0;length=len(h)
            self.dark_remaining=np.zeros_like(h,dtype=np.int32)
            self.next_dark=np.full_like(h,length,dtype=np.int32)
            remain=np.zeros(m,np.int32);nearest=np.full(m,length,np.int32)
            for j in range(length-1,-1,-1):
                remain=np.where(dark[j],remain+1,0)
                nearest=np.where(dark[j],j,nearest)
                self.dark_remaining[j]=remain;self.next_dark[j]=nearest
            self.dark_elapsed=np.zeros_like(h,dtype=np.int32)
            elapsed=np.zeros(m,np.int32)
            for row in env.get('harvest_prior',[]):
                elapsed=np.where(row==0,elapsed+1,0)
            for j in range(length):
                elapsed=np.where(dark[j],elapsed+1,0);self.dark_elapsed[j]=elapsed
            self.length=length;self.ready=True
        dark=env['harvest'][t]==0
        entered=dark & ~self.previous_dark
        if self.kind=='dual':
            self.mu[entered]=self.last_window_end[entered] if c.dual_warm=='prev' else 0.
        next_dark=self.next_dark[t]
        has_eclipse=next_dark<self.length
        next_index=np.minimum(next_dark,self.length-1)
        end=next_dark+self.dark_remaining[next_index,np.arange(m)]
        n=np.where(dark,self.dark_remaining[t],np.maximum(end-t,1)).astype(np.int32)
        # The extended ephemeris contains the next eclipse in the main/G0 scene.
        # At a no-eclipse season, the finite extended horizon is recorded explicitly.
        n[~has_eclipse]=self.length-t
        time_to_eclipse=np.maximum(next_dark-t,0)*dt
        maximum_load=c.base_w+4*env['terminal'].max_electrical_w
        full_sun=(~dark) & (energy+time_to_eclipse*b.charge_efficiency*(env['harvest'][t]-maximum_load)>=b.capacity_j)
        z=np.maximum(energy-env['reserve'][t],0.)
        xi=z/(b.capacity_j-b.minimum_j)
        eta=np.where(dark,b.discharge_efficiency,b.charge_efficiency)
        pace=4*c.sleep_w+eta*z/(np.maximum(n,1)*dt)
        fraction=1./np.maximum(n,1)
        if self.kind=='dual' and dark.any():
            from .forecast_r22 import predicted_loads,pacing_fraction
            loads=predicted_loads(env,t,int(np.max(n[dark])))
            fraction=pacing_fraction(loads,n,dark)
            pace[dark]=4*c.sleep_w+eta[dark]*z[dark]*fraction[dark]/dt
        if self.kind=='dual':
            from .reserve_pricing_r23 import eclipse_power,sustainable_sun_power
            self.idle_w=env['terminal'].idle_w
            p_full=4*(self.idle_w-c.sleep_w)+self.var_ema
            fill=(~dark) & (energy+time_to_eclipse*b.charge_efficiency*(
                env['harvest'][t]-c.base_w-4*c.sleep_w-p_full)>=b.capacity_j)
            # NaN explicitly means not evaluated: (a) already proves reset.
            # Condition (b) is evaluated only where (a) is false.
            p_star=np.full(m,np.nan)
            p_star[dark]=eclipse_power(z[dark],n[dark],dt,b.discharge_efficiency,
                                      4*(env['terminal'].max_electrical_w-c.sleep_w))
            sun_nodes=np.flatnonzero((~dark)&(~fill))
            p_star[sun_nodes]=sustainable_sun_power(env,t,energy,t+n,sun_nodes)
            slack=(~fill)&(p_star>=p_full)
            full_sun=fill|slack
            pace[sun_nodes]=4*c.sleep_w+p_star[sun_nodes]
            self.mu[full_sun]=0.
            pi=self.mu.copy()
        elif self.kind=='xi':pi=c.lam/(xi+c.epsilon)
        elif self.kind=='lyapunov':pi=c.lyapunov_lambda*np.maximum(c.lyapunov_theta-xi,0.)
        else:pi=np.zeros(m)
        total=self.dark_elapsed[t]+self.dark_remaining[t]-1
        progress=np.divide(self.dark_elapsed[t]-1,np.maximum(total,1))
        thirds=np.where(dark,np.minimum(2,np.floor(progress*3).astype(np.int32)),-1)
        sixths=np.where(dark,np.minimum(5,np.floor(progress*6).astype(np.int32)),-1)
        self.last=dict(mu=self.mu.copy(),pi=pi,pace_w=pace,z_j=z,remaining_slots=n,dark=dark,
                       spill_reset=full_sun,window_has_eclipse=has_eclipse,eclipse_third=thirds,
                       eclipse_sixth=sixths,pace_weight_fraction=fraction)
        if self.kind=='dual':
            self.last.update(p_star_w=p_star,p_full_w=p_full,p_star_computed=~fill,
                             reset_realistic_fill=fill,reset_sustainable_slack=slack)
        self.previous_dark=dark.copy()
        return c.alpha+pi,xi,self.last

    def commit(self,q_w,t):
        if self.kind!='dual':return
        trace=self.last
        all_on=q_w>=4*self.idle_w
        self.var_ema[all_on]=.8*self.var_ema[all_on]+.2*(q_w[all_on]-4*self.idle_w)
        update=~trace['spill_reset']
        self.mu[update]=np.maximum(self.mu[update]+self.c.dual_eta*(q_w[update]-trace['pace_w'][update])/100.,0.)
        last=trace['dark'] & (self.dark_remaining[t]==1)
        self.last_window_end[last]=self.mu[last]
