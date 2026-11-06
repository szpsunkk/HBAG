"""One validation seed proves unchanged V0 physics against the retained v2.1 run."""
from pathlib import Path
import json,hashlib
import numpy as np
from .environment import ROOT
from .model import Config
from .runner import run_policy,write_json,fingerprint


def compare_arrays(old,new):
    checked=[];errors=[]
    with np.load(old) as a,np.load(new) as b:
        for key in a.files:
            if key not in b or a[key].dtype!=b[key].dtype or a[key].shape!=b[key].shape or a[key].tobytes()!=b[key].tobytes():
                errors.append(key)
            else:checked.append(dict(field=key,sha256=hashlib.sha256(a[key].tobytes()).hexdigest()))
    return dict(old=str(old.relative_to(ROOT)),new=str(new.relative_to(ROOT)),checked=checked,differences=errors)


def main(env=None):
    old=ROOT/'results_r2/G0_prime/calibration_load_0p65/seed_1000/V0'
    prior=json.loads((old/'summary.json').read_text());c=Config(**prior['config'])
    out=ROOT/'results_r2/audit/v22_v0_bitwise/seed_1000/V0'
    current=run_policy(c,'V0',out,phase='validation',storage='full',env=env)
    arrays=[compare_arrays(old/'raw.npz',out/'raw.npz'),
            compare_arrays(old.parent/'inputs.npz',out.parent/'inputs.npz'),
            compare_arrays(old.parent/'inputs/paths.npz',out.parent/'inputs/paths.npz')]
    timing={'runtime_s','environment_build_s','mean_solver_s','mean_slot_s'}
    metadata={'source_fingerprint','manifest'}
    differences=[key for key,value in prior.items() if key not in timing|metadata and current.get(key)!=value]
    oldslots=json.loads((old/'slots.json').read_text());newslots=json.loads((out/'slots.json').read_text())
    slot_differences=[]
    for a,b in zip(oldslots,newslots):
        slot_differences.extend([dict(slot=a['slot'],field=k) for k,v in a.items() if k not in ('solver_s','slot_s') and b.get(k)!=v])
    result=dict(passed=not any(a['differences'] for a in arrays) and not differences and not slot_differences and len(oldslots)==len(newslots),
        seed=c.seed,plan_version='v2.2',source_fingerprint=fingerprint(),prior_source_fingerprint=prior['source_fingerprint'],
        array_comparisons=arrays,scientific_summary_differences=differences,slot_differences=slot_differences,
        exclusions=sorted(timing|metadata),new_diagnostic_fields=sorted(set(current)-set(prior)),test_seeds_executed=[])
    write_json(ROOT/'results_r2/audit/v22_v0_bitwise.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='array_comparisons'},ensure_ascii=False,indent=2))
    if not result['passed']:raise AssertionError('V0 v2.1/v2.2 comparison failed')


if __name__=='__main__':main()
