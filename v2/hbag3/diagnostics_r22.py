"""Eclipse-window validation of causal potential-load forecasts."""
import numpy as np
from .forecast_r22 import predicted_loads


class WindowForecasts:
    def __init__(self,env):
        self.env=env;self.predicted=np.zeros_like(env['harvest'],dtype=float)
        self.previous=np.zeros(env['shell'].size,bool);self.windows=[]

    def observe_start(self,t,trace):
        dark=trace['dark'];entered=dark&~self.previous
        if entered.any():
            remaining=trace['remaining_slots'];loads=predicted_loads(self.env,t,int(remaining[entered].max()))
            for node in np.flatnonzero(entered):
                stop=min(t+int(remaining[node]),len(self.predicted))
                self.predicted[t:stop,node]=loads[:stop-t,node]
                self.windows.append(dict(satellite=int(node),start_slot=t,end_slot_exclusive=stop,
                    left_censored=bool(t==0),right_censored=bool(t+int(remaining[node])>len(self.predicted))))
        self.previous=dark.copy()

    def finish(self,actual_bits):
        records=[]
        for window in self.windows:
            lo=window['start_slot'];hi=window['end_slot_exclusive'];node=window['satellite']
            x=self.predicted[lo:hi,node]*self.env['dt'];y=actual_bits[lo:hi,node]
            finite=np.isfinite(x).all() and np.isfinite(y).all()
            valid=len(x)>=2 and finite and np.ptp(x)>0 and np.ptp(y)>0
            records.append(dict(window,slots=len(x),correlation=float(np.corrcoef(x,y)[0,1]) if valid else None,
                undefined_reason=None if valid else 'fewer_than_two_or_nonfinite_or_constant_trace',
                predicted_bits=float(x.sum()),actual_bits=float(y.sum())))
        correlations=[r['correlation'] for r in records if r['correlation'] is not None]
        return records,dict(potential_load_correlation_mean=float(np.mean(correlations)) if correlations else None,
            potential_load_correlation_valid_windows=len(correlations),potential_load_correlation_total_windows=len(records),
            potential_load_correlation_definition='Pearson over slots of each satellite eclipse window; forecast frozen at its first observed slot using current demand; actual delivered bits counted once at each path node including source and destination. Constant/short windows have undefined correlation; censored windows flagged.')
