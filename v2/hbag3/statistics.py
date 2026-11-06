"""Seed, not satellite or slot, is the experimental unit."""
import numpy as np
from scipy.stats import wilcoxon

def paired(a,b,seed=1729,repeats=10000):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)!=len(b) or len(a)<20:raise ValueError('At least 20 matched complete runs required')
    delta=a-b;rng=np.random.default_rng(seed);boot=delta[rng.integers(0,len(delta),(repeats,len(delta)))].mean(axis=1)
    p=1. if np.all(delta==0) else float(wilcoxon(delta,alternative='two-sided',zero_method='wilcox',method='auto').pvalue)
    return dict(n=len(a),mean_difference=float(delta.mean()),ci95_low=float(np.percentile(boot,2.5)),ci95_high=float(np.percentile(boot,97.5)),p_wilcoxon=p,bootstrap_replicates=repeats)

def holm(values):
    p=np.asarray(values);order=np.argsort(p);answer=np.empty(len(p));adjusted=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))));answer[order]=np.minimum(adjusted,1);return answer
