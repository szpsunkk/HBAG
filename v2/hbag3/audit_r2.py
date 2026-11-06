"""Independent checks of serialized R2 physical ledgers and executed flow paths."""
from pathlib import Path
import json, hashlib
import numpy as np
from .environment import ROOT
from .runner import write_json


def audit_run(out):
    out=Path(out)
    summary=json.loads((out/'summary.json').read_text())
    c=summary['config'];dt=summary['dt_s'];m=c['planes']*c['per_plane'];n=summary['slots']
    with np.load(out.parent/'inputs.npz') as inp_archive,np.load(out/'raw.npz') as raw_archive:
        inp={k:inp_archive[k] for k in inp_archive.files}
        raw={k:raw_archive[k] for k in raw_archive.files}
        with np.load(out.parent/'inputs/paths.npz') as path_archive:
            path_data={k:path_archive[k] for k in path_archive.files}
        wrong_dtypes={k:str(raw[k].dtype) for k in raw if raw[k].dtype.kind=='f' and raw[k].dtype!=np.float32}
        wrong_dtypes.update({'inputs/'+k:str(inp[k].dtype) for k in inp if inp[k].dtype.kind=='f' and inp[k].dtype!=np.float32})
        states=raw['energy_j'].astype(float);q=raw['terminal_w'].astype(float);h=inp['harvest_w'].astype(float)
        net=h-c['base_w']-q
        delta=dt*(.95*np.maximum(net,0)-np.maximum(-net,0)/.95)
        reconstructed=np.clip(states[:-1]+delta,0,c['capacity_j'])
        error=np.abs(states[1:]-reconstructed)
        rounding=.5*np.spacing(raw['energy_j'][:-1]).astype(float)+.5*np.spacing(raw['energy_j'][1:]).astype(float)
        rounding+=dt/.95*.5*np.spacing(raw['terminal_w']).astype(float)+1e-7
        fixed=4*c['sleep_w']+(25.-c['sleep_w'])*raw['terminals_on'].sum(axis=2)
        fixed_error=float(np.max(np.abs(fixed-raw['fixed_terminal_w'])))
        max_rate_error=0.;max_disabled_rate=0.;max_demand_violation=0.;max_cap_violation=0.
        missing_links=0;one_sided=0;v0_non_a0_sleep=0
        relay=np.zeros((n,m));access=np.zeros((n,m));reconstructed_util=[]
        for t in range(n):
            lo,hi=map(int,inp['link_offsets'][t:t+2]);edges=inp['directed_edges'][lo:hi]
            lookup={tuple(map(int,e)):j for j,e in enumerate(edges)}
            rate=np.zeros(hi-lo);allocation=raw['allocation_bps'][t].astype(float)
            source=inp['sources'][t];attached=source>=0
            np.add.at(access[t],source[attached],inp['demand_bps'][t,attached].astype(float))
            for f in range(c['flows']):
                for p in range(c['paths']+1):
                    j=int(raw['executed_path_ids'][t,f,p])
                    if j>=0:
                        a,b=map(int,path_data['path_offsets'][j:j+2]);path=path_data['path_nodes'][a:b]
                    else:path=[]
                    amount=allocation[f,p]
                    for u,v in zip(path[:-1],path[1:]):
                        if (int(u),int(v)) not in lookup:missing_links+=1
                        else:rate[lookup[int(u),int(v)]]+=amount
                    if len(path)>2:np.add.at(relay[t],path[1:-1],amount*dt)
            saved=raw['link_rate_bps'][lo:hi].astype(float)
            max_rate_error=max(max_rate_error,float(np.max(np.abs(rate-saved)/np.maximum(saved,1.))))
            on=raw['link_on'][lo:hi];cap=inp['capacity_bps'][lo:hi].astype(float)*raw['link_capture'][lo:hi].astype(float)
            max_disabled_rate=max(max_disabled_rate,float(np.max(saved[~on],initial=0)))
            max_demand_violation=max(max_demand_violation,float(np.max((allocation.sum(axis=1)-inp['demand_bps'][t])/np.maximum(inp['demand_bps'][t],1),initial=0)))
            max_cap_violation=max(max_cap_violation,float(np.max((saved-cap)/np.maximum(cap,1),initial=0)))
            reconstructed_util.extend(np.divide(saved,cap,out=np.zeros_like(saved),where=cap>0)[on])
            for j,(u,v) in enumerate(edges[:len(edges)//2]):
                up,ui=divmod(int(u),c['per_plane']);vp,vi=divmod(int(v),c['per_plane'])
                if up==vp:ports=(0,1) if (vi-ui)%c['per_plane']==1 else (1,0)
                else:ports=(2,3) if (vp-up)%c['planes']==1 else (3,2)
                left=raw['terminals_on'][t,u,ports[0]];right=raw['terminals_on'][t,v,ports[1]]
                one_sided+=int(left!=right or on[j]!=(left and right))
                if summary['method']=='V0' and states[t,u]>c['cut_soc']*c['capacity_j'] and states[t,v]>c['cut_soc']*c['capacity_j']:
                    v0_non_a0_sleep+=int(not on[j])
        relay_error=float(np.max(np.abs(relay-raw['relay_bit_hops'])/np.maximum(relay,1)))
        holes=(~raw['terminals_on'].any(axis=2))&(access>0)
        result=dict(seed=c['seed'],method=summary['method'],slots=n,raw_float32_only=not wrong_dtypes,
            dtype_errors=wrong_dtypes,energy_reconstruction_max_abs_j=float(error.max()),
            energy_float32_rounding_bound_max_j=float(rounding.max()),energy_within_rounding_bound=bool(np.all(error<=rounding)),
            fixed_terminal_power_max_error_w=fixed_error,path_to_link_max_relative_error=max_rate_error,
            disabled_link_max_rate_bps=max_disabled_rate,missing_path_links=missing_links,one_sided_handshakes=one_sided,
            v0_non_a0_sleep=v0_non_a0_sleep,demand_violation_relative=max_demand_violation,capacity_violation_relative=max_cap_violation,
            relay_bit_hops_max_relative_error=relay_error,hole_node_slots=int(holes.sum()),
            on_link_utilization_mean_reconstructed=float(np.mean(reconstructed_util)),
            gate_mean_error=abs(float(np.mean(reconstructed_util))-summary['on_link_utilization_mean']))
        result['passed']=(not wrong_dtypes and result['energy_within_rounding_bound'] and fixed_error<1e-5
                          and max_rate_error<2e-6 and max_disabled_rate==0 and missing_links==0 and one_sided==0
                          and v0_non_a0_sleep==0 and max_demand_violation<2e-6 and max_cap_violation<2e-6
                          and relay_error<2e-6 and int(holes.sum())==summary['hole_node_slots']
                          and result['gate_mean_error']<1e-7)
    write_json(out/'independent_audit.json',result)
    return result


def main():
    rows=[]
    for p in sorted((ROOT/'results_r2/G0_prime').glob('calibration_*/seed_*/V0/summary.json')):
        cached=p.parent/'independent_audit.json'
        rows.append(json.loads(cached.read_text()) if cached.exists() else audit_run(p.parent))
    result=dict(runs=len(rows),all_passed=all(r['passed'] for r in rows),results=rows)
    write_json(ROOT/'results_r2/audit/g0_prime_independent.json',result)
    print(json.dumps(dict(runs=len(rows),all_passed=result['all_passed'],failed=[r['seed'] for r in rows if not r['passed']]),indent=2))


if __name__=='__main__':main()
