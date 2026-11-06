"""v2.1 storage tiers and one deduplicated route registry per cell/seed."""
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .plan_r2 import FULL_SEEDS,TEST_SEEDS,RF_FULL_SEEDS,RF_TEST_SEEDS,RF_EXPERIMENT

POLICY_FIELDS=('alpha','lam','epsilon','rho','guard_soc','kappa_off','xi_wake','max_iterations',
               'tolerance','dual_eta','dual_warm','lyapunov_lambda','lyapunov_theta','utilization_sleep_threshold')


def storage_tier(seed,phase,requested=None,experiment=None):
    if requested=='selection':
        if phase!='validation':raise ValueError('Selection storage is only for validation')
        return 'selection'
    if phase=='test':
        if seed in RF_TEST_SEEDS:
            if experiment!=RF_EXPERIMENT:raise ValueError('5000–5019 storage belongs exclusively to E4-double-prime')
            tier='full' if seed in RF_FULL_SEEDS else 'node'
            if requested not in (None,tier):raise ValueError('RF test storage tier is fixed before outcomes')
            return tier
        if seed not in TEST_SEEDS:raise ValueError('Formal storage requires v2.3 test seeds 4000–4019')
        tier='full' if seed in FULL_SEEDS else 'node'
        if requested not in (None,tier):raise ValueError('Test storage tier is fixed by seed, never by outcome')
        return tier
    return requested or 'full'


def register_routes(cell,catalogues):
    from .runner import save_npz
    path=Path(cell)/'inputs'/'paths.npz'
    lookup={};nodes=[];offsets=[0]
    if path.exists():
        with np.load(path) as old:
            nodes=old['path_nodes'].tolist();offsets=old['path_offsets'].tolist()
        lookup={tuple(nodes[a:b]):i for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:]))}
    ids=[]
    for cat in catalogues:
        slot=np.full(cat.mask.shape,-1,np.int32)
        for f,paths in enumerate(cat.paths):
            for j,route in enumerate(paths):
                if not route:continue
                key=tuple(map(int,route))
                if key not in lookup:
                    lookup[key]=len(offsets)-1;nodes.extend(key);offsets.append(len(nodes))
                slot[f,j]=lookup[key]
        ids.append(slot)
    save_npz(path,path_nodes=np.asarray(nodes,np.int32),path_offsets=np.asarray(offsets,np.int32))
    return np.asarray(ids,np.int32)


def shared_inputs(env,cell,tier):
    from .runner import save_npz,write_json,source_hashes
    if tier=='selection':return
    cell=Path(cell);cell.mkdir(parents=True,exist_ok=True)
    config=asdict(env['config'])
    for key in POLICY_FIELDS:config.pop(key,None)
    meta=dict(config=config,source_sha256=source_hashes(),storage_tier=tier,dt_s=env['dt'],
              reference_scale_bps=env['reference_scale_bps'],reference_r1_bps=env['reference_r1_bps'])
    if (cell/'inputs.npz').exists():
        import json
        prior=json.loads((cell/'inputs.json').read_text())
        if prior!=meta:raise RuntimeError('Shared cell inputs mismatch: '+str(cell))
        return
    cats=env['catalogues']
    arrays=dict(demand_bps=env['demand_bps'],harvest_w=env['harvest'],reserve_j=env['reserve'],
        times_s=env['times'],city_pairs=env['pairs'],plane_beta_deg=env['plane_beta_deg'],
        shape=env['shape'],source_lon=env['source_lon'],mmpp_states=env['mmpp_states'],
        sources=np.asarray([cat.sources for cat in cats],np.int32),destinations=np.asarray([cat.destinations for cat in cats],np.int32))
    if tier=='full':
        sizes=np.array([len(cat.edges) for cat in cats],np.int32)
        arrays.update(link_offsets=np.r_[np.int32(0),np.cumsum(sizes,dtype=np.int32)],
            directed_edges=np.concatenate([cat.edges for cat in cats]).astype(np.int32),
            capacity_bps=np.concatenate([cat.link_capacity_bps for cat in cats]),
            distance_km=np.concatenate([cat.distance_km for cat in cats]),geometric_path_ids=register_routes(cell,cats))
    save_npz(cell/'inputs.npz',**arrays);write_json(cell/'inputs.json',meta)
