#!/usr/bin/env python3
"""Recalculate transparent, seed-level summaries from reference or rerun results."""
import argparse
import collections
import csv
from pathlib import Path

import reproduce
import numpy as np
from hbag3.analysis_r231 import paired_ratio
from hbag3.v24_report import ci


def load_runs(work=None):
    if work is None:
        return reproduce.records(reproduce.PACKAGE/'reference/per_seed.jsonl.gz')
    root=Path(work).expanduser().resolve()/'results_r2'
    rows=[]
    for experiment in ('E2','E4_optical','E4_RF','E5','E6','E7','E8','E9_iterations','E10','MPC','E13'):
        for path in sorted((root/experiment).glob('*/seed_*/*/summary.json')):
            summary=reproduce.read(path)
            if summary.get('complete'):
                rows.append(dict(experiment=experiment,cell=path.parents[2].name,
                                 method_label=path.parent.name,summary=summary,
                                 source_path=str(path.relative_to(root))))
    # Common cells are the same scientific runs; reuse them, without rerunning.
    index={(r['experiment'],r['cell'],r['method_label'],r['summary']['seed']) for r in rows}
    for row in list(rows):
        if (row['experiment'],row['cell'])!=('E2','rho_1.4'):
            continue
        targets=[]
        if row['method_label'] in ('V0','V1','HBAG'):
            targets.extend([('E6','K4_hybrid'),('E13','main')])
        if row['method_label']=='HBAG':targets.append(('E9_iterations','iterations_200'))
        for experiment,cell in targets:
            key=(experiment,cell,row['method_label'],row['summary']['seed'])
            if key not in index:
                rows.append(dict(row,experiment=experiment,cell=cell));index.add(key)
    return rows


def save_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(rows)


def build(rows):
    groups=collections.defaultdict(dict)
    for row in rows:
        key=(row['experiment'],row['cell'],row['method_label'])
        seed=row['summary']['seed']
        if seed in groups[key]:raise ValueError('Duplicate seed in '+repr(key))
        groups[key][seed]=row['summary']
    summary=[];paired=[]
    metrics=('delivered_bits','esr','min_soc','terminal_energy_j','system_energy_j',
             'mean_slot_s','path_precomputation_s','peak_rss_bytes')
    for (experiment,cell,method),runs in sorted(groups.items()):
        expected=5 if (experiment,cell)==('E10','M_5000') else 20
        ordered=[runs[s] for s in sorted(runs)]
        complete=len(ordered)==expected
        for metric in metrics:
            if not all(metric in r for r in ordered):continue
            values=[r[metric] for r in ordered]
            stat=ci(values) if complete else dict(mean=float(np.mean(values)),ci95_low=None,ci95_high=None)
            summary.append(dict(experiment=experiment,cell=cell,method=method,metric=metric,
                                n=len(ordered),expected_n=expected,complete=complete,**stat))
    for experiment,cell in sorted({key[:2] for key in groups}):
        h=groups.get((experiment,cell,'HBAG'),{});v=groups.get((experiment,cell,'V1'),{})
        if len(h)!=20 or set(h)!=set(v):continue
        seeds=sorted(h)
        hv=np.array([h[s]['delivered_bits'] for s in seeds]);vv=np.array([v[s]['delivered_bits'] for s in seeds])
        for name,values in [('HBAG_minus_V1_bits',hv-vv),('HBAG_vs_V1_percent',100*(hv/vv-1))]:
            paired.append(dict(experiment=experiment,cell=cell,metric=name,n=20,**ci(values)))
        v0=groups.get((experiment,cell,'V0'),{})
        if set(v0)==set(h):
            stat,_=paired_ratio([v0[s]['delivered_bits'] for s in seeds],vv,hv,seeds)
            paired.append(dict(experiment=experiment,cell=cell,metric='(HBAG-V1)/(V0-V1)',**stat))
    return summary,paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,help='Output root passed to reproduce.py; omit for bundled evidence')
    parser.add_argument('--output',type=Path,default=reproduce.PACKAGE/'outputs/report')
    args=parser.parse_args()
    out=args.output.expanduser().resolve()
    if out==reproduce.PACKAGE or out in reproduce.PACKAGE.parents or (reproduce.PACKAGE/'reference') in (out,*out.parents):
        raise ValueError('Use a separate report output directory')
    rows=load_runs(args.results)
    if not rows:raise ValueError('No completed runs found')
    summary,paired=build(rows)
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'summary.csv',summary);save_csv(out/'paired.csv',paired)
    coverage=dict(sorted(collections.Counter(r['experiment'] for r in rows).items()))
    reproduce.write(out/'coverage.json',dict(runs=coverage,source='rerun' if args.results else 'bundled reference',
        bootstrap_replicates=10000,bootstrap_seed=1729,
        note='Seed is the experimental unit. Incomplete groups have no confidence interval. Common E2 cells are reused.'))
    print('Wrote summary.csv, paired.csv and coverage.json to '+str(out))


if __name__=='__main__':main()
