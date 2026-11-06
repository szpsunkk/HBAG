"""Measure validation artifacts and conservatively project the declared test matrix."""
from pathlib import Path
from io import BytesIO
import json
import lzma
import numpy as np
from .environment import ROOT
from .runner import write_json,fingerprint
from .plan_r2 import VERSION
from .matrix_v21 import declared_counts
from .compact_r2 import encode,decode,inputs_for,read_raw,read_slots

DETAIL_FIELDS={'allocation_bps','link_rate_bps','link_on','link_capture','link_newly_usable',
               'counterfactual_value','support_value','wake_value','executed_path_ids'}
INPUT_DETAIL={'link_offsets','directed_edges','capacity_bps','distance_km','geometric_path_ids','path_nodes','path_offsets'}
E2_FIELDS={'predicted_window_load_bps','actual_relay_access_bits'}
METHOD_COUNTS={
 'E2':{'V0':100,'Static+Guard':220,'Util-Sleep':100,'Lyapunov':100,'V1':100,'HBAG':100,'V2':60,'HBAG-xi':60,'HBAG-Lyap':60,'HBAG-central':20},
 'E9_iterations':{'HBAG':80},'E4':dict.fromkeys(['V0','Static+Guard','Lyapunov','V1','HBAG'],60),
 'E5':dict.fromkeys(['V0','Static+Guard','Util-Sleep','Lyapunov','V1','HBAG'],200),
 'E6':dict.fromkeys(['V0','V1','HBAG'],80),'E7':dict.fromkeys(['V0','Static+Guard','Lyapunov','V1','HBAG'],100),
 'E8':{'V0':240,'HBAG':240,'HBAG+margin':160},'E10':{'HBAG':85}}


def compressed_size(arrays):
    buffer=BytesIO();np.savez_compressed(buffer,**arrays);return buffer.tell()


def projection():
    rows=[];stamp=fingerprint()
    paths=list((ROOT/'results_r2/G0_prime').glob('calibration_*/seed_*/V0/summary.json'))
    paths+=list((ROOT/'results_r2/G0_prime/functional').glob('rho_*/seed_*/*/summary.json'))
    paths+=list((ROOT/'results_r2/audit/storage_sample_v23').glob('seed_*/*/summary.json'))
    for p in sorted(paths):
        s=json.loads(p.read_text());out=p.parent
        if s['source_fingerprint']!=stamp and s['method']!='V0':continue
        all_arrays=read_raw(out);inputs=inputs_for(out);orbits=s['config']['orbits']
        arrays={k:v for k,v in all_arrays.items() if k not in E2_FIELDS}
        has_e2=E2_FIELDS.issubset(all_arrays)
        windows=out/'potential_load_windows.json'
        window_bytes=(len(lzma.compress(windows.read_bytes(),preset=3)) if windows.exists() else
                      windows.with_suffix('.json.xz').stat().st_size if windows.with_suffix('.json.xz').exists() else 0)
        scalar_bytes=len(lzma.compress(json.dumps(read_slots(out),indent=2).encode(),preset=3))+p.stat().st_size
        scalar_bytes+=sum(q.stat().st_size for q in out.glob('*.json') if q.name not in ('slots.json','summary.json','potential_load_windows.json','eclipse_windows.json'))
        eclipse=out/'eclipse_windows.json'
        scalar_bytes+=(len(lzma.compress(eclipse.read_bytes(),preset=3)) if eclipse.exists() else
                       eclipse.with_suffix('.json.xz').stat().st_size if eclipse.with_suffix('.json.xz').exists() else 0)
        node={k:v for k,v in arrays.items() if k not in DETAIL_FIELDS}
        blob=encode(node,inputs,s);restored=decode(blob,inputs,s)
        if any(v.tobytes()!=restored[k].tobytes() for k,v in node.items()):raise ArithmeticError('Node archive roundtrip failed')
        row=dict(path=str(out.relative_to(ROOT)),seed=s['seed'],method=s['method'],orbits=orbits,
                 source_fingerprint=s['source_fingerprint'],
                 node_bytes=(len(blob)+scalar_bytes)/orbits,
                 node_inputs_bytes=compressed_size({k:v for k,v in inputs.items() if k not in INPUT_DETAIL})/orbits,
                 verified_byte_exact=True)
        if 'allocation_bps' in arrays:
            blob=encode(arrays,inputs,s);restored=decode(blob,inputs,s)
            if any(v.tobytes()!=restored[k].tobytes() for k,v in arrays.items()):raise ArithmeticError('Full archive roundtrip failed')
            row.update(full_bytes=(len(blob)+scalar_bytes)/orbits,
                       full_inputs_bytes=((out.parent/'inputs.npz').stat().st_size+(out.parent/'inputs/paths.npz').stat().st_size)/orbits)
        if has_e2:
            for tier,values in [('node',{k:v for k,v in all_arrays.items() if k not in DETAIL_FIELDS}),('full',all_arrays)]:
                blob=encode(values,inputs,s);restored=decode(blob,inputs,s)
                if any(v.tobytes()!=restored[k].tobytes() for k,v in values.items()):raise ArithmeticError('E2 diagnostics archive roundtrip failed')
                row['e2_'+tier+'_bytes']=(len(blob)+scalar_bytes+window_bytes)/orbits
        rows.append(row)
    if not rows:raise RuntimeError('No real validation output for storage projection')
    node=max(r['node_bytes'] for r in rows)
    full=max(r['full_bytes'] for r in rows if 'full_bytes' in r)
    full=max(full,node+max(r['full_bytes']-r['node_bytes'] for r in rows if 'full_bytes' in r))
    input_full=max(r['full_inputs_bytes'] for r in rows if 'full_inputs_bytes' in r)
    input_node=max(r['node_inputs_bytes'] for r in rows)
    e2_full_extra=max((max(0.,r['e2_full_bytes']-r['full_bytes']) for r in rows if 'e2_full_bytes' in r),default=0.)
    e2_node_extra=max((max(0.,r['e2_node_bytes']-r['node_bytes']) for r in rows if 'e2_node_bytes' in r),default=0.)
    method_sizes={}
    for method in {r['method'] for r in rows}:
        group=[r for r in rows if r['method']==method]
        measured_node=max(r['node_bytes'] for r in group)
        details=[r for r in group if 'full_bytes' in r]
        measured_full=max([r['full_bytes'] for r in details],default=full)
        measured_full=max(measured_full,measured_node+max([r['full_bytes']-r['node_bytes'] for r in details],default=full-node))
        method_sizes[method]={'full':measured_full,'node':measured_node,'validation_samples':len(group)}
    total=0;parts=[]
    for experiment,count in declared_counts().items():
        if experiment=='E11':continue
        factor=count.get('orbits',1)
        # E8 uses 64 rather than 576 stars and more time steps at dt15; overestimate
        # its size at a full main-size one-orbit node trace rather than scale down.
        # E10 sparse snapshots are bounded by 5000/576 times four/96 snapshots.
        if experiment=='E10':factor=(5000/576)*(4/96)
        counts=METHOD_COUNTS[experiment]
        if sum(counts.values())!=count['method_runs']:raise AssertionError('Storage matrix count mismatch')
        output_bytes=0.
        for method,number in counts.items():
            # V2/central/margin have no independent pre-test full trace: use the
            # worst measured policy, rather than assuming they compress as HBAG.
            sizes=method_sizes.get(method,{'full':full,'node':node})
            output_bytes+=number*factor*(.25*sizes['full']+.75*sizes['node'])
            if experiment=='E2':output_bytes+=number*factor*(.25*e2_full_extra+.75*e2_node_extra)
        byte_count=output_bytes+count['cells']*factor*(.25*input_full+.75*input_node)
        parts.append(dict(experiment=experiment,projected_bytes=byte_count));total+=byte_count
    # Additional E9 frozen problems/history, E1 and optional small-model artifacts.
    allowance=500_000_000
    total+=allowance
    record=dict(plan_version=VERSION,source_fingerprint=stamp,
        e2_diagnostics_measured=any('e2_full_bytes' in r for r in rows),
        e2_full_extra_one_orbit_bytes=e2_full_extra,e2_node_extra_one_orbit_bytes=e2_node_extra,
        measured_full_one_orbit_bytes=full,measured_node_one_orbit_bytes=node,
        measured_shared_full_inputs_one_orbit_bytes=input_full,measured_shared_node_inputs_one_orbit_bytes=input_node,
        measured_samples=rows,per_method_max_one_orbit=method_sizes,parts=parts,extra_allowance_bytes=allowance,
        projected_test_bytes=total,projected_test_GB=total/1e9,target_GB=10.,
        meets_target=total<=10_000_000_000,method='Per-policy maximum measured byte-exact archives; unmeasured policies use global maximum. 25% detailed/75% node outputs AND inputs, exact method/orbit/matrix counts; +0.5GB E1/E9/P2 allowance. This is a projection, monitored against actual disk usage.')
    write_json(ROOT/'results_r2/audit/storage_projection.json',record);return record


def ensure_storage_budget(frozen):
    record=projection()
    from .report_r2 import runtime_report
    runtime_report()
    from .runtime_v23 import refresh
    runtime=refresh()
    if not record['e2_diagnostics_measured']:raise RuntimeError('STORAGE_PAUSE: v2.3 E2 diagnostics must be measured before test execution')
    if not record['meets_target']:
        raise RuntimeError('STORAGE_PAUSE: projected formal results %.3fGB exceeds declared 10GB target; refine lossless storage before starting tests'%record['projected_test_GB'])
    if runtime is None or runtime['unmeasured']:
        raise RuntimeError('TIMING_PAUSE: complete declared-size validation pilots and one-off MPC timing before formal execution')
    return record


if __name__=='__main__':print(json.dumps(projection(),indent=2))
