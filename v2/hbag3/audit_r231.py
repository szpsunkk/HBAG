"""Read-only evidence checks for the approved v2.3 corpus."""
from pathlib import Path
import datetime,hashlib,json,os
from .environment import ROOT
from .runner import write_json,fingerprint
from .plan_r2 import VERSION,SHA256


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1048576),b''):digest.update(part)
    return digest.hexdigest()


def identity(**fields):
    return dict(time=datetime.datetime.now().astimezone().isoformat(),plan_version=VERSION,
                plan_sha256=SHA256,source_fingerprint=fingerprint(),**fields)


def check_methods():
    transition=json.loads((ROOT/'results_r2/v231/audit/transition.json').read_text())
    old=ROOT/transition['archive']
    names=['reserve_pricing_r23.py','forecast_r22.py','diagnostics_r22.py','model.py','energy.py','network.py',
           'traffic.py','environment.py','control.py','pricing.py','link_values.py','allocation.py','optimization.py',
           'central_reference_r2.py','solvers.py','gate.py','matrix_v21.py','timing_r2.py','optional_r2.py','ppo.py']
    rows=[dict(path='hbag3/'+name,old=sha(old/'hbag3'/name),current=sha(ROOT/'hbag3'/name)) for name in names]
    record=identity(state='PASS' if all(r['old']==r['current'] for r in rows) else 'FAIL',files=rows)
    write_json(ROOT/'results_r2/v231/audit/method_files.json',record)
    if record['state']!='PASS':raise AssertionError('A policy implementation changed')
    return record


def verify_preservation():
    manifest_path=ROOT/'results_r2/v231/audit/v23_preservation_manifest.json'
    manifest=json.loads(manifest_path.read_text());differences=[]
    for row in manifest['files']:
        path=ROOT/row['path']
        if 'symlink' in row:
            if not path.is_symlink() or os.readlink(path)!=row['symlink']:differences.append(row['path'])
        elif not path.is_file() or path.stat().st_size!=row['size'] or sha(path)!=row['sha256']:
            differences.append(row['path'])
    record=identity(state='PASS' if not differences else 'FAIL',checked=manifest['file_count'],
                    manifest_sha256=sha(manifest_path),differences=differences)
    write_json(ROOT/'results_r2/v231/audit/v23_preservation_check.json',record)
    if differences:raise AssertionError('Approved v2.3 artifacts changed: '+repr(differences[:10]))
    return record


def compare_arrays(a,b):
    differences=[]
    for key in sorted(set(a)|set(b)):
        if key not in a or key not in b or a[key].dtype!=b[key].dtype or a[key].shape!=b[key].shape or a[key].tobytes()!=b[key].tobytes():
            differences.append(key)
    return dict(passed=not differences,array_count=len(a),differences=differences)
