import json
from dataclasses import replace
import pytest
import reproduce
from hbag3.model import Config


def test_release_inventory_preserves_sources_and_seed_ranges():
    assert reproduce.verify_source()>20
    cells=reproduce.cells_for('E4_RF')
    assert len(cells)==60
    assert {c['config']['seed'] for c in cells}==set(range(5000,5020))
    assert {c['config']['seed'] for c in reproduce.cells_for('E2')}==set(range(4000,4020))


def test_separate_workspace_refuses_changed_code(tmp_path):
    work=reproduce.init_workspace(tmp_path/'output')
    assert reproduce.init_workspace(work)==work
    (work/'hbag3/model.py').write_text('changed\n')
    with pytest.raises(RuntimeError,match='different code'):
        reproduce.init_workspace(work)
    with pytest.raises(ValueError):reproduce.init_workspace(reproduce.PACKAGE)


def test_rf_fallback_retains_unsafe_qualification():
    rf=reproduce.read(reproduce.PACKAGE/'frozen/rf.json')
    assert rf['selected']['HBAG']['eligible']
    assert not rf['selected']['Lyapunov']['eligible']
    assert rf['selected']['Lyapunov']['min_esr']<1


def test_portable_seed_guard_is_scoped_and_checks_parameters(tmp_path,monkeypatch):
    # Install in a child process so original protocol tests retain their own gates.
    import subprocess,sys
    script='''
from pathlib import Path
from dataclasses import replace
import reproduce
from hbag3.model import Config
work=reproduce.init_workspace(Path(__import__('sys').argv[1]))
reproduce.configure(work,'E4_RF',0)
from hbag3.runner import seed_guard
rf=reproduce.read(reproduce.PACKAGE/'frozen/rf.json')
c=Config(seed=5000,medium='rf',interference_db=-10.,**rf['selected']['HBAG']['parameters'])
seed_guard(c,'test',method='HBAG')
for bad in [replace(c,seed=3000),replace(c,medium='optical'),replace(c,dual_eta=99.)]:
    try: seed_guard(bad,'test',method='HBAG')
    except ValueError: pass
    else: raise AssertionError('Invalid RF experiment accepted')
reproduce.configure(work,'E2',0)
try: seed_guard(c,'test',method='HBAG')
except ValueError: pass
else: raise AssertionError('RF test range leaked to E2')
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path/'run')],cwd=str(reproduce.PACKAGE),check=True)
