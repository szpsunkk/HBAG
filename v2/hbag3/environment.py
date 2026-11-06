"""Common, policy-independent demand and strict Yen k-shortest route catalogues."""
from dataclasses import asdict,dataclass,replace
from pathlib import Path
from itertools import islice
import hashlib,json,pickle,time,os
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.optimize import linprog
from .model import Config,Terminal,RATE,POWER
from .network import Shell,snapshot,ground_positions,attach_ground,slot_harvest_lower
from .energy import periodic_reserve
from .traffic import load_cities

ROOT=Path(__file__).resolve().parents[1]
@dataclass
class Catalogue:
    mask:np.ndarray
    paths:list
    incidence:sparse.csr_matrix
    sender:sparse.csr_matrix
    link_capacity_bps:np.ndarray
    distance_km:np.ndarray
    edges:np.ndarray
    sources:np.ndarray
    destinations:np.ndarray
    hops:np.ndarray


def catalogue(net,sources,destinations,k=3,routing='distance',node_energy=None,node_dark=None):
    graph=nx.Graph()
    graph.add_nodes_from(range(len(net.positions)))
    # A whole simple path's distance perturbation is <1, so hop count is primary.
    tie_scale=len(net.positions)*float(np.max(net.distance_km,initial=1.))+1.
    if routing=='hybrid' and (k!=4 or node_dark is None):
        raise ValueError('v2 hybrid catalogue requires K=4 and the common ephemeris eclipse state')
    for (a,b),d in zip(net.edges,net.distance_km):
        weight=1.+float(d)/tie_scale if routing in ('hops','hybrid') else float(d)
        if routing=='energy': weight*=1.+2.*(1.-min(node_energy[a],node_energy[b]))
        solar=1.+1.5*(int(node_dark[a])+int(node_dark[b])) if node_dark is not None else 1.
        graph.add_edge(int(a),int(b),weight=weight,solar=solar)
    directed=np.vstack([net.edges,net.edges[:,::-1]])
    index={tuple(edge):i for i,edge in enumerate(directed)}
    rows=[];cols=[];paths=[];mask=np.zeros((len(sources),k),bool);hops=np.zeros_like(mask,int)
    memo={}
    for f,(s,d) in enumerate(zip(sources,destinations)):
        key=(int(s),int(d))
        if key not in memo:
            candidates=[]
            if s>=0 and d>=0 and s!=d:
                try:
                    candidates=list(islice(nx.shortest_simple_paths(graph,int(s),int(d),weight='weight'),3 if routing=='hybrid' else k))
                    if routing=='hybrid':
                        for solar_path in nx.shortest_simple_paths(graph,int(s),int(d),weight='solar'):
                            if solar_path not in candidates:
                                candidates.append(solar_path)
                                break
                except (nx.NetworkXNoPath,nx.NodeNotFound): pass
            memo[key]=candidates
        routes=memo[key]
        paths.append([tuple(p) for p in routes]+[None]*(k-len(routes)))
        for j,path in enumerate(routes):
            mask[f,j]=True;hops[f,j]=len(path)-1
            for a,b in zip(path[:-1],path[1:]): rows.append(index[a,b]);cols.append(f*k+j)
    incidence=sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(directed),mask.size)).tocsr()
    sender=sparse.coo_matrix((np.ones(len(directed)),(directed[:,0],np.arange(len(directed)))),shape=(len(net.positions),len(directed))).tocsr()
    return Catalogue(mask,paths,incidence,sender,np.tile(net.capacity_bps,2),np.tile(net.distance_km,2),directed,np.asarray(sources),np.asarray(destinations),hops)


def reference_scale(cat,shape):
    f,k=cat.mask.shape
    group=sparse.kron(sparse.eye(f),np.ones((1,k)),format='csr')
    link=sparse.diags(RATE/cat.link_capacity_bps)@cat.incidence
    A=sparse.vstack([sparse.hstack([link,sparse.csr_matrix((link.shape[0],1))]),sparse.hstack([-group,sparse.csr_matrix(shape[:,None])])],format='csr')
    out=linprog(np.r_[np.zeros(cat.mask.size),-1.],A_ub=A,b_ub=np.r_[np.ones(link.shape[0]),np.zeros(f)],bounds=[(0,None if x else 0.) for x in cat.mask.ravel()]+[(0,None)],method='highs')
    if not out.success or out.x[-1]<=0: raise ValueError('Reference disconnected or infeasible: '+out.message)
    return float(out.x[-1])*RATE


def sun_vector(beta):
    i=np.deg2rad(53.);b=np.deg2rad(beta)
    return np.array([np.cos(b),-np.sin(b)*np.sin(i),np.sin(b)*np.cos(i)])


def plane_betas(shell,sun):
    r=2*np.pi*np.arange(shell.planes)/shell.planes;i=np.deg2rad(shell.inclination_deg)
    normals=np.column_stack([np.sin(r)*np.sin(i),-np.cos(r)*np.sin(i),np.full(len(r),np.cos(i))])
    return np.rad2deg(np.arcsin(np.clip(normals@sun,-1,1)))


def city_mmpp_states(seed,city_ids,times):
    """Independent source-city chains; adding/reordering OD flows does not change a city's chain."""
    city_ids=np.asarray(city_ids,dtype=np.int64)
    unique,inverse=np.unique(city_ids,return_inverse=True)
    dt=np.median(np.diff(times)) if len(times)>1 else 60.
    states=np.zeros((len(times),len(unique)),bool)
    for j,city in enumerate(unique):
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),7919,int(city)]))
        state=False
        for t in range(len(times)):
            if rng.random()<1-np.exp(-dt/(300. if state else 1200.)):state=not state
            states[t,j]=state
    return states[:,inverse]


def demand_trace(config,shape,scale,times,source_lon,sources,harvest,source_city_ids=None,return_states=False):
    rng=np.random.default_rng(config.seed+7919)
    local_hour=(12+times[:,None]/3600+source_lon[None,:]/15)%24
    diurnal=.55+.45*(1+np.cos(2*np.pi*(local_hour-20)/24))/2
    factors=np.ones((len(times),len(shape)))
    states=np.zeros_like(factors,dtype=bool)
    if config.traffic in ('mmpp','bursty'):
        if source_city_ids is None:raise ValueError('v2.1 MMPP requires source-city identifiers')
        states=city_mmpp_states(config.seed,source_city_ids,times)
        factors=np.where(states,3.4,.4)
        factors*=rng.gamma(20.,1/20.,factors.shape)
    elif config.traffic.startswith('hotspot'):
        multiplier=float(config.traffic[-1]);region=(source_lon>=-30)&(source_lon<=60)
        active=(times>=times[-1]/2-900)&(times<times[-1]/2+900)
        factors[np.ix_(active,region)]=multiplier
    elif config.traffic=='eclipse':
        for t,src in enumerate(sources):
            good=src>=0
            factors[t,good]=np.where(harvest[t,src[good]]==0,5.,.05)
    elif config.traffic!='steady': raise ValueError(config.traffic)
    modulation=diurnal*factors
    # Preserve whole-horizon TOTAL mean across traffic types, retaining geographic concentration.
    modulation/=np.sum(modulation*shape[None,:])/len(times)/shape.sum()
    demand=config.load*scale*shape[None,:]*modulation
    return (demand,states) if return_states else demand


def cache_signature(config):
    # Cache key includes every environment implementation source; policy choices excluded.
    fields=asdict(config)
    for k in ('alpha','lam','epsilon','rho','max_iterations','tolerance','guard_soc','delay_s','substep_s','initial_soc','kappa_off','xi_wake','dual_eta','dual_warm','lyapunov_lambda','lyapunov_theta'): fields.pop(k,None)
    if config.routing=='energy': fields['initial_soc']=config.initial_soc
    data=json.dumps(fields,sort_keys=True).encode()
    for f in ['environment.py','model.py','network.py','energy.py']: data+=(ROOT/'hbag3'/f).read_bytes()
    return hashlib.sha256(data).hexdigest()[:20]


def build_environment(config,use_cache=False,snapshot_indices=None,progress=None):
    if use_cache:
        raise ValueError('R2 uses in-memory environments; R1 disk caches are read-only')
    c=config;key=cache_signature(c)
    if snapshot_indices is not None:key+='_'+hashlib.sha256(str(snapshot_indices).encode()).hexdigest()[:8]
    cache=ROOT/'results_r2'/'transient_cache'/('env_'+key+'.pkl')
    if use_cache and cache.exists():
        with cache.open('rb') as f: env=pickle.load(f)
        env['config']=c;return env
    start=time.perf_counter();shell=Shell(c.planes,c.per_plane);terminal=Terminal(medium=c.medium,interference_db=c.interference_db)
    dt=c.dt_override_s or shell.period_s/c.slots_per_orbit
    slots=int(np.ceil(c.orbits*shell.period_s/dt));times=np.arange(slots)*dt
    # Reserve is computed beyond the evaluation horizon by two orbits, never truncated at the last test slot.
    fulltimes=np.arange(slots+int(np.ceil(2*shell.period_s/dt))+1)*dt
    sun=sun_vector(c.beta_deg)
    h=np.array([slot_harvest_lower(shell,t,dt,c.solar_w,sun) for t in fulltimes])
    prior_times=np.arange(-int(np.ceil(shell.period_s/dt)),0)*dt
    prior_h=np.array([slot_harvest_lower(shell,t,dt,c.solar_w,sun) for t in prior_times])
    if c.heterogeneous:
        alternate=np.array([slot_harvest_lower(shell,t,dt,c.solar_w,sun_vector(60.)) for t in fulltimes])
        h[:,shell.size//2:]=alternate[:,shell.size//2:]
    if c.artificial_eclipse:
        phase=(np.arange(shell.size)/shell.size+fulltimes[:,None]/shell.period_s)%1
        h[phase>=1-c.artificial_eclipse]=0.
    from .energy import reserve_envelope
    reserve=reserve_envelope(h,c.base_w+4*c.sleep_w,dt,c.battery)
    certified=(reserve.max(axis=0)<=c.battery.capacity_j+1e-6)
    geometry_fields={k:asdict(c)[k] for k in ('seed','planes','per_plane','orbits','slots_per_orbit','flows','paths','routing','medium','interference_db','dt_override_s','demand_mode')}
    geometry_fields['snapshot_indices']=snapshot_indices
    if c.routing=='energy':
        geometry_fields.update({k:asdict(c)[k] for k in ('beta_deg','initial_soc','preset','base_w','solar_w','artificial_eclipse','heterogeneous')})
    geometry_key=hashlib.sha256(json.dumps(geometry_fields,sort_keys=True).encode()+Path(__file__).read_bytes()+(ROOT/'hbag3/model.py').read_bytes()+(ROOT/'hbag3/network.py').read_bytes()).hexdigest()[:20]
    geometry_cache=ROOT/'results_r2'/'transient_cache'/('geometry_'+geometry_key+'.pkl')
    if use_cache and geometry_cache.exists():
        with geometry_cache.open('rb') as f: geometry=pickle.load(f)
        catalogues,pairs,source_lon,shape,scale,all_sources=geometry
    else:
        firstnet=snapshot(shell,0,terminal)
        rng=np.random.default_rng(c.seed)
        cities=load_cities(ROOT/'base_rev3/sources/natural_earth_cities.geojson')
        lon=np.array([v[1] for v in cities]);lat=np.array([v[2] for v in cities]);population=np.array([v[3] for v in cities],float)
        # Purchasing power / penetration multipliers are declared synthetic assumptions, not measured data.
        penetration=np.where(abs(lat)>25,1.,.65)
        weight=population*penetration;weight/=weight.sum()
        pairs=[];source_lon=[];shape=[]
        if c.demand_mode=='symmetric':
            sources=np.arange(c.flows)%shell.size;dest=(sources+2)%shell.size
            pairs=np.column_stack([sources,dest]);source_lon=np.zeros(c.flows);shape=np.ones(c.flows)
        else:
            ground=ground_positions(lon,lat,0);access=attach_ground(ground,firstnet.positions)
            graph=nx.from_scipy_sparse_matrix(firstnet.graph)
            labels={v:i for i,comp in enumerate(nx.connected_components(graph)) for v in comp}
            attempts=0
            while len(pairs)<c.flows:
                a,b=rng.choice(len(cities),2,replace=False,p=weight);attempts+=1
                if access[a]>=0 and access[b]>=0 and access[a]!=access[b] and labels[int(access[a])]==labels[int(access[b])] and np.linalg.norm(ground[a]-ground[b])>=2500:
                    pairs.append([a,b]);source_lon.append(lon[a]);shape.append(rng.lognormal(0,.4))
                if attempts>100000: raise ValueError('No reachable reference OD pairs')
            pairs=np.asarray(pairs);source_lon=np.asarray(source_lon);shape=np.asarray(shape)
            # First 10% of OD streams represent synthetic aviation corridors between sampled hubs.
            aviation=np.arange(c.flows)<max(1,c.flows//10);shape[aviation]*=.25
        shape/=shape.mean()
        catalogues=[];all_sources=[];all_destinations=[];catalogue_build_s={}
        for j,t in enumerate(times):
            net=firstnet if j==0 else snapshot(shell,t,terminal)
            if c.demand_mode=='symmetric': src,dst=pairs.T
            else:
                unique=np.unique(pairs);access=attach_ground(ground_positions(lon[unique],lat[unique],t),net.positions)
                mapping=dict(zip(unique,access));src=np.array([mapping[a] for a in pairs[:,0]]);dst=np.array([mapping[b] for b in pairs[:,1]])
            route_energy=np.clip((c.initial_soc*c.battery.capacity_j-reserve[j])/(c.battery.capacity_j-c.battery.minimum_j),0,1)
            route_start=time.perf_counter()
            cat=catalogue(net,src,dst,c.paths,c.routing,route_energy,h[j]==0) if snapshot_indices is None or j in snapshot_indices else None
            if cat is not None:catalogue_build_s[j]=time.perf_counter()-route_start
            catalogues.append(cat);all_sources.append(src);all_destinations.append(dst)
            if progress is not None and (j%8==0 or j+1==len(times)):
                progress(dict(phase='environment',slot=j+1,slots=len(times),elapsed_s=time.perf_counter()-start))
        # v2.1 common reference: optical hybrid K=4, independent of evaluated medium/path variant.
        refnet=snapshot(shell,0,Terminal())
        refcat=catalogue(refnet,all_sources[0],all_destinations[0],4,'hybrid',node_dark=h[0]==0)
        scale=reference_scale(refcat,shape)
        r1cat=catalogue(refnet,all_sources[0],all_destinations[0],3,'distance')
        r1_scale=reference_scale(r1cat,shape)
        if use_cache:
            geometry_cache.parent.mkdir(exist_ok=True)
            temp=geometry_cache.with_suffix('.'+str(os.getpid())+'.tmp')
            with temp.open('wb') as f:pickle.dump((catalogues,pairs,source_lon,shape,scale,all_sources),f,protocol=4)
            temp.replace(geometry_cache)
    demand,mmpp_states=demand_trace(c,shape,scale,times,source_lon,all_sources,h,np.asarray(pairs)[:,0],True)
    env=dict(config=c,shell=shell,terminal=terminal,battery=c.battery,times=times,dt=dt,harvest=h[:slots],reserve=reserve[:slots+1],reserve_extended=reserve,harvest_extended=h,harvest_prior=prior_h,certified=certified,catalogues=catalogues,demand_bps=demand,mmpp_states=mmpp_states,shape=shape,reference_scale_bps=scale,reference_r1_bps=r1_scale,pairs=pairs,source_lon=source_lon,plane_beta_deg=plane_betas(shell,sun),build_s=time.perf_counter()-start,cache_key=key)
    env['geometric_sources']=np.asarray(all_sources)
    env['geometric_destinations']=np.asarray(all_destinations)
    env['potential_incidence_cache']={}
    env['catalogue_build_s']=catalogue_build_s
    if use_cache:
        cache.parent.mkdir(exist_ok=True);temp=cache.with_suffix('.'+str(os.getpid())+'.tmp')
        with temp.open('wb') as f:pickle.dump(env,f,protocol=4)
        temp.replace(cache)
    return env
