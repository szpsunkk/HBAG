"""Lossless float32 bit-prediction archive for R2; decode is byte-exact, not recomputation."""
from io import BytesIO
from pathlib import Path
import hashlib,json,lzma,os,zipfile
import numpy as np


ORDER=['allocation_bps','executed_path_ids','terminals_on','delivered_bps','relay_bit_hops','actual_relay_access_bits','terminal_w','energy_j','mu','remaining_window_slots',
       'discretionary_energy_j','pace_weight_fraction','p_star_computed','pace_w','p_star_w','p_full_w','raw_budget_w','fixed_terminal_w',
       'raw_after_j','spill_j','unserved_j','battery_discharge_j','price_pi','access_demand_bps','link_rate_bps','support_value']


def predicted(name,arrays,inputs,summary):
    c=summary['config'];dt=summary['dt_s'];bcap=c['capacity_j']
    e=arrays.get('energy_j');q=arrays.get('terminal_w');h=inputs.get('harvest_w');reserve=inputs.get('reserve_j')
    if name=='discretionary_energy_j':return np.maximum(e[:-1].astype(float)-reserve[:-1],0.)
    if name=='pace_w':
        eta=np.where(h==0,.95,.95)
        if 'pace_weight_fraction' in arrays:
            value=4*c['sleep_w']+eta*arrays['discretionary_energy_j'].astype(float)*arrays['pace_weight_fraction']/dt
            if 'p_star_computed' in arrays:
                # Sun targets already lie on the 14-bisection grid. Predicting
                # them with unrelated continuous surplus destroys that entropy.
                # Zero prediction preserves their original bits; no rounding.
                value=np.where((h!=0)&arrays['p_star_computed'],0.,value)
            return value
        return 4*c['sleep_w']+eta*arrays['discretionary_energy_j'].astype(float)/(np.maximum(arrays['remaining_window_slots'],1)*dt)
    if name=='p_star_w':
        # Stored XOR residuals, not recomputation, retain every original bit.
        uniform=np.clip(.95*arrays['discretionary_energy_j'].astype(float)/
                        (np.maximum(arrays['remaining_window_slots'],1)*dt),0,4*(60-c['sleep_w']))
        value=np.where(h==0,uniform,arrays['pace_w'].astype(float)-4*c['sleep_w'])
        return np.where(arrays['p_star_computed'],value,np.nan)
    if name=='p_full_w':
        value=np.empty_like(q,dtype=float);variable=np.full(q.shape[1],8.)
        for t,power in enumerate(q.astype(float)):
            value[t]=4*(25-c['sleep_w'])+variable
            on=power>=100.;variable[on]=.8*variable[on]+.2*(power[on]-100.)
        return value
    if name=='raw_budget_w':
        change=reserve[1:].astype(float)-e[:-1]
        return h-np.where(change>=0,change/(.95*dt),.95*change/dt)-c['base_w']
    if name=='fixed_terminal_w':return 4*c['sleep_w']+(25.-c['sleep_w'])*arrays['terminals_on'].sum(axis=2)
    if name in ('raw_after_j','battery_discharge_j'):
        net=h.astype(float)-c['base_w']-q
        delta=dt*(.95*np.maximum(net,0)-np.maximum(-net,0)/.95)
        return e[:-1]+delta if name=='raw_after_j' else np.maximum(-delta,0.)
    if name=='spill_j':return np.maximum(arrays['raw_after_j'].astype(float)-bcap,0.)
    if name=='unserved_j':return np.maximum(-arrays['raw_after_j'].astype(float),0.)
    if name=='price_pi':
        if summary['method'] in ['HBAG','V2','HBAG-central','HBAG+margin']:return arrays['mu']
        xi=np.maximum(e[:-1].astype(float)-reserve[:-1],0)/(c['dod_max']*bcap)
        if summary['method'] in ['HBAG-xi','HBAG-ξ']:return c['lam']/(xi+c['epsilon'])
        if summary['method'] in ['Lyapunov','HBAG-Lyap']:return c['lyapunov_lambda']*np.maximum(c['lyapunov_theta']-xi,0)
        return np.zeros_like(e[:-1])
    if name=='access_demand_bps':
        result=np.zeros_like(e[:-1]);src=inputs['sources'];demand=inputs['demand_bps']
        for t in range(len(result)):
            valid=src[t]>=0;np.add.at(result[t],src[t,valid],demand[t,valid])
        return result
    if name=='delivered_bps':
        if 'allocation_bps' in arrays:return arrays['allocation_bps'].astype(float).sum(axis=2)
        return inputs.get('demand_bps')
    if name=='actual_relay_access_bits' and 'relay_bit_hops' in arrays:
        result=arrays['relay_bit_hops'].astype(float).copy()
        for t,rate in enumerate(arrays['delivered_bps']):
            for key in ['sources','destinations']:
                nodes=inputs[key][t];valid=nodes>=0
                np.add.at(result[t],nodes[valid],rate[valid].astype(float)*dt)
        return result
    if name=='relay_bit_hops' and 'executed_path_ids' in arrays:
        result=np.zeros((len(arrays['allocation_bps']),c['planes']*c['per_plane']))
        for t,x in enumerate(arrays['allocation_bps']):
            for f,ids in enumerate(arrays['executed_path_ids'][t]):
                for j,path_id in enumerate(ids):
                    if path_id<0 or x[f,j]==0:continue
                    a,b=inputs['path_offsets'][path_id:path_id+2];nodes=inputs['path_nodes'][a:b]
                    np.add.at(result[t],nodes[1:-1],float(x[f,j])*dt)
        return result
    if name in ('link_rate_bps','support_value') and 'executed_path_ids' in arrays:
        results=[]
        for t,x in enumerate(arrays['allocation_bps']):
            lo,hi=inputs['link_offsets'][t:t+2];edges=inputs['directed_edges'][lo:hi]
            lookup={tuple(e):j for j,e in enumerate(edges)};result=np.zeros(len(edges));count=len(edges)//2
            derivative=1./(1.+x.astype(float).sum(axis=1)/c['utility_s_bps'])
            for f,ids in enumerate(arrays['executed_path_ids'][t]):
                for p,path_id in enumerate(ids):
                    if path_id<0 or x[f,p]==0:continue
                    a,b=inputs['path_offsets'][path_id:path_id+2];path=inputs['path_nodes'][a:b]
                    value=float(x[f,p]) if name=='link_rate_bps' else float(x[f,p])/1e9*derivative[f]
                    for u,v in zip(path[:-1],path[1:]):result[lookup[u,v]]+=value
            if name=='support_value':result=np.tile(result[:count]+result[count:],2)
            results.append(result)
        return np.concatenate(results)
    return None


def encode(arrays,inputs,summary):
    keys=[k for k in ORDER if k in arrays]+sorted(set(arrays)-set(ORDER))
    metadata=[];chunks=[];decoded={}
    for key in keys:
        value=np.ascontiguousarray(arrays[key]);predictor=predicted(key,decoded,inputs,summary) if value.dtype==np.float32 else None
        recurrence=(key=='energy_j' and value.dtype==np.float32 and 'terminal_w' in decoded
                    and 'harvest_w' in inputs and value.shape[0]==decoded['terminal_w'].shape[0]+1)
        if recurrence:
            net=inputs['harvest_w'].astype(float)-summary['config']['base_w']-decoded['terminal_w']
            delta=summary['dt_s']*(.95*np.maximum(net,0)-np.maximum(-net,0)/.95)
            base=np.clip(value[:-1].astype(float)+delta,0,summary['config']['capacity_j']).astype(np.float32)
            encoded=value.view(np.uint32).copy();encoded[1:]^=base.view(np.uint32);mode='energy_recurrence_xor'
        elif predictor is not None:
            base=np.ascontiguousarray(predictor,dtype=np.float32)
            encoded=np.bitwise_xor(value.view(np.uint32),base.view(np.uint32));mode='predict_xor'
        elif value.dtype==np.float32 and value.ndim>=2:
            bits=value.view(np.uint32);encoded=bits.copy();encoded[1:]=bits[1:]^bits[:-1];mode='time_xor'
        else:encoded=value;mode='raw'
        # Byte planes place low-entropy exponent/high bytes together.
        bytes_=np.ascontiguousarray(encoded).view(np.uint8).reshape(-1,value.dtype.itemsize).T.tobytes()
        metadata.append(dict(key=key,dtype=value.dtype.str,shape=list(value.shape),mode=mode,length=len(bytes_),sha256=hashlib.sha256(value.tobytes()).hexdigest()))
        chunks.append(bytes_);decoded[key]=value
    header=json.dumps(metadata,separators=(',',':')).encode()
    return len(header).to_bytes(8,'little')+header+lzma.compress(b''.join(chunks),preset=6)


def decode(blob,inputs,summary):
    hlen=int.from_bytes(blob[:8],'little');meta=json.loads(blob[8:8+hlen]);data=lzma.decompress(blob[8+hlen:]);offset=0;arrays={}
    for item in meta:
        dtype=np.dtype(item['dtype']);length=item['length'];shape=tuple(item['shape'])
        plane=np.frombuffer(data[offset:offset+length],dtype=np.uint8);offset+=length
        contiguous=plane.reshape(dtype.itemsize,-1).T.copy()
        encoded=contiguous.reshape(-1).view(dtype).reshape(shape)
        if item['mode']=='energy_recurrence_xor':
            net=inputs['harvest_w'].astype(float)-summary['config']['base_w']-arrays['terminal_w']
            delta=summary['dt_s']*(.95*np.maximum(net,0)-np.maximum(-net,0)/.95)
            value=encoded.copy();bits=encoded.view(np.uint32)
            for t in range(len(delta)):
                base=np.clip(value[t].astype(float)+delta[t],0,summary['config']['capacity_j']).astype(np.float32)
                value[t+1]=(bits[t+1]^base.view(np.uint32)).view(np.float32)
        elif item['mode']=='predict_xor':
            base=np.ascontiguousarray(predicted(item['key'],arrays,inputs,summary),dtype=np.float32)
            value=(encoded.view(np.uint32)^base.view(np.uint32)).view(np.float32)
        elif item['mode']=='time_xor':value=np.bitwise_xor.accumulate(encoded.view(np.uint32),axis=0).view(np.float32)
        else:value=encoded.copy()
        if hashlib.sha256(value.tobytes()).hexdigest()!=item['sha256']:raise ArithmeticError('Lossless archive decode hash mismatch '+item['key'])
        arrays[item['key']]=value
    return arrays


def inputs_for(out):
    out=Path(out).resolve()
    with np.load(out.parent/'inputs.npz') as data:inputs={k:data[k] for k in data.files}
    paths=out.parent/'inputs/paths.npz'
    if paths.exists():
        with np.load(paths) as data:inputs.update({k:data[k] for k in data.files})
    return inputs


def read_raw(out):
    out=Path(out).resolve()
    if (out/'raw.npz').exists():
        with np.load(out/'raw.npz') as data:return {k:data[k] for k in data.files}
    summary=json.loads((out/'summary.json').read_text())
    return decode((out/'raw.r2z').read_bytes(),inputs_for(out),summary)


def compact_run(out):
    out=Path(out);path=out/'raw.npz'
    for name in ['potential_load_windows.json','eclipse_windows.json']:compact_json(out/name)
    if not path.exists():return None
    summary=json.loads((out/'summary.json').read_text());inputs=inputs_for(out)
    with np.load(path) as data:arrays={k:data[k] for k in data.files}
    blob=encode(arrays,inputs,summary);recovered=decode(blob,inputs,summary)
    if any(arrays[k].dtype!=recovered[k].dtype or arrays[k].shape!=recovered[k].shape or arrays[k].tobytes()!=recovered[k].tobytes() for k in arrays):raise ArithmeticError('Archive is not byte-identical')
    target=out/'raw.r2z';temp=out/'raw.r2z.tmp';temp.write_bytes(blob);temp.replace(target)
    from .runner import write_json
    result=dict(codec='R2 float32 predictor XOR + temporal XOR + byte planes + LZMA preset 6, lossless',
                input_bytes=path.stat().st_size,archived_bytes=len(blob),array_count=len(arrays),verified_byte_exact=True,
                decoder='hbag3.compact_r2.read_raw',original_npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(out/'storage.json',result);path.unlink()
    return result


def compact_slots(out):
    compact_json(Path(out)/'slots.json')


def compact_json(path):
    path=Path(path)
    if not path.exists():return
    original=path.read_bytes();packed=lzma.compress(original,preset=3)
    if lzma.decompress(packed)!=original:raise ArithmeticError('Scalar sequence archive mismatch')
    path.with_suffix('.json.xz').write_bytes(packed);path.unlink()


def read_slots(out):
    return read_json_archive(Path(out)/'slots.json')


def read_json_archive(p):
    p=Path(p)
    return json.loads(p.read_text() if p.exists() else lzma.decompress(p.with_suffix('.json.xz').read_bytes()))
