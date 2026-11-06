"""Actions carry commodity/route identities and a finite validity interval."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Candidate:
    rates: np.ndarray
    paths: list
    commodity_ids: list
    valid_until_s: float


def remap_candidate(candidate, paths, commodity_ids, now_s):
    """No positional reuse after a catalogue rebuild; expiry rejects everything."""
    width=len(paths[0]) if paths else 0
    out=np.zeros((len(paths),width))
    if now_s >= candidate.valid_until_s:
        return out,dict(expired=True,mapped_paths=0,dropped_rate=float(candidate.rates.sum()))
    old={key:i for i,key in enumerate(candidate.commodity_ids)}
    mapped=0
    for f,key in enumerate(commodity_ids):
        if key not in old:continue
        g=old[key];routes={tuple(p):j for j,p in enumerate(candidate.paths[g]) if p is not None}
        for j,p in enumerate(paths[f]):
            if p is not None and tuple(p) in routes:
                out[f,j]=candidate.rates[g,routes[tuple(p)]];mapped+=1
    return out,dict(expired=False,mapped_paths=mapped,dropped_rate=float(candidate.rates.sum()-out.sum()))
