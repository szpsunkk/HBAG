"""Verify common absolute demand across media and candidate path variants."""
import hashlib,json
import numpy as np
from .environment import ROOT
from .runner import write_json


def verify(experiment):
    if experiment not in ('E4','E6'):return
    rows=[]
    for p in sorted((ROOT/'results_r2'/experiment).glob('*/seed_*/inputs.npz')):
        seed=p.parent.name
        reference=ROOT/'results_r2/E2/rho_1.4'/seed/'inputs.npz'
        with np.load(p) as actual,np.load(reference) as common:
            checks={key:bool(np.array_equal(actual[key],common[key])) for key in ['demand_bps','harvest_w','city_pairs','mmpp_states','shape','sources','destinations']}
            digest=hashlib.sha256(actual['demand_bps'].tobytes()).hexdigest()
        rows.append(dict(input=str(p.relative_to(ROOT)),reference=str(reference.relative_to(ROOT)),checks=checks,demand_sha256=digest))
    result=dict(experiment=experiment,checked_cells=len(rows),passed=bool(rows) and all(all(r['checks'].values()) for r in rows),cells=rows)
    write_json(ROOT/('results_r2/audit/'+experiment+'_common_inputs.json'),result)
    if not result['passed']:raise AssertionError(experiment+' common demand/environment audit failed')
