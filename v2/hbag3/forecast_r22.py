"""Ephemeris-only candidate incidence and causal persistence load forecasts."""
import numpy as np
from scipy import sparse


def precompute_price_catalogues(env,slots,progress=None):
    """Ephemeris-only public paths, outside E10's online timing interval.

    Cover all price windows (including sun-to-next-eclipse-end), with the last
    evaluated geometry repeated beyond the simulation as specified in S2.2a.
    Sparse incidence multiplication by current demand remains online.
    """
    import time
    from .environment import catalogue
    from .network import snapshot
    from .model import Terminal
    h=env.get('harvest_extended',env['harvest']);length=len(h);m=h.shape[1]
    remaining=np.zeros_like(h,dtype=np.int32);next_dark=np.full_like(remaining,length)
    n=np.zeros(m,np.int32);nearest=np.full(m,length,np.int32)
    for t in range(length-1,-1,-1):
        dark=h[t]==0;n=np.where(dark,n+1,0);nearest=np.where(dark,t,nearest)
        remaining[t]=n;next_dark[t]=nearest
    required=set(map(int,slots));last=len(env['catalogues'])-1
    for t in slots:
        nd=next_dark[t];end=nd+remaining[np.minimum(nd,length-1),np.arange(m)]
        end=np.where(nd<length,end,length)
        required.update(min(k,last) for k in range(t,int(end.max())))
    timings=dict(env.get('catalogue_build_s',{}));matrix_s=0.;start=time.perf_counter()
    for index in sorted(required):
        if env['catalogues'][index] is None:
            tick=time.perf_counter();net=snapshot(env['shell'],env['times'][index],Terminal())
            env['catalogues'][index]=catalogue(net,env['geometric_sources'][index],env['geometric_destinations'][index],
                4,'hybrid',node_dark=env['harvest'][index]==0)
            timings[index]=time.perf_counter()-tick
        tick=time.perf_counter();forecast_matrix(env,index);matrix_s+=time.perf_counter()-tick
        if progress:progress(dict(phase='ephemeris_path_precomputation',completed=len(env['potential_incidence_cache']),
                                 total=len(required),elapsed_s=time.perf_counter()-start))
    total=sum(timings[k] for k in required)
    return dict(label='所有方法共用的星历预计算',slots=sorted(required),constructed_slots=len(required),
                path_total_s=total,path_mean_s_per_slot=total/len(required),
                path_mean_s_per_satellite_slot=total/len(required)/m,
                incidence_precomputation_s=matrix_s,extra_wall_s=time.perf_counter()-start,
                path_s_by_slot={str(k):timings[k] for k in sorted(required)})


def potential_incidence(cat,nodes):
    rows=[];columns=[];values=[]
    for f,paths in enumerate(cat.paths):
        valid=[p for p in paths if p]
        counts={}
        for path in valid:
            for node in set(path):counts[node]=counts.get(node,0)+1
        for node,count in counts.items():
            rows.append(int(node));columns.append(f);values.append(count/len(valid))
    return sparse.csr_matrix((values,(rows,columns)),shape=(nodes,len(cat.paths)))


def forecast_matrix(env,index):
    """Cache only sparse geometry probabilities in the cell's shared memory.

    Main runs already contain all K4 catalogues. Path variants and sparse E10
    snapshots construct the public K4 forecast from the same known ephemeris;
    no future realized demand, battery state or executed route enters it.
    """
    index=min(index,len(env['catalogues'])-1)
    cache=env.setdefault('potential_incidence_cache',{})
    if index in cache:return cache[index]
    cat=env['catalogues'][index];c=env['config']
    if cat is None or c.routing!='hybrid' or c.paths!=4:
        # Synthetic unit fixtures have no full ephemeris; their supplied paths
        # are the complete geometric catalogue by construction.
        if 'geometric_sources' not in env:
            if cat is None:raise ValueError('Missing geometric forecast inputs')
        else:
            from .environment import catalogue
            from .network import snapshot
            from .model import Terminal
            net=snapshot(env['shell'],env['times'][index],Terminal())
            cat=catalogue(net,env['geometric_sources'][index],env['geometric_destinations'][index],
                4,'hybrid',node_dark=env['harvest'][index]==0)
    cache[index]=potential_incidence(cat,env['shell'].size)
    return cache[index]


def predicted_loads(env,t,horizon):
    demand=np.asarray(env['demand_bps'][t],float)
    return np.asarray([forecast_matrix(env,t+k)@demand for k in range(max(int(horizon),1))])


def pacing_fraction(loads,remaining,dark):
    """Zero/nonfinite total weight falls back to EXACTLY the uniform fraction."""
    fraction=1./np.maximum(remaining,1)
    for node in np.flatnonzero(dark):
        values=loads[:max(int(remaining[node]),1),node]
        weight=values+.1*np.mean(values)
        total=weight.sum()
        if np.isfinite(total) and total>0:fraction[node]=weight[0]/total
    return fraction
