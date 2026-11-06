"""Check declared grids and the common selection rule without simulation/test seeds."""
from collections import Counter
from dataclasses import asdict, replace
from hbag3.model import Config
from hbag3.gate import validation_grid, select_parameters


def test_exact_method_grids_and_no_v2_retuning():
    grid=validation_grid()
    assert Counter(r['method'] for r in grid)=={'HBAG':24,'HBAG-xi':9,'Lyapunov':6,'HBAG-Lyap':6,'Static+Guard':3,'Util-Sleep':3}
    assert len({r['label'] for r in grid})==51
    assert not any('xi_wake' in r['parameters'] for r in grid)
    assert {r['parameters']['dual_warm'] for r in grid if r['method']=='HBAG'}=={'zero','prev'}


def test_unified_selection_uses_safety_then_delivery_then_energy(tmp_path,monkeypatch):
    import hbag3.gate as gate
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    monkeypatch.setattr(gate,'G0',tmp_path/'G0')
    (tmp_path/'tables_r2').mkdir()
    rows=[];grid=validation_grid()
    for item in grid:
        for seed in range(1000,1010):
            index=int(item['label'].rsplit('_',1)[1])
            rows.append(dict(grid_label=item['label'],method=item['method'],seed=seed,
                esr=1. if index%3 else .99,
                delivered_bits=100.+index,terminal_energy_j=200.-index))
    selected,_=select_parameters(rows)
    for method,result in selected.items():
        eligible=[x for x in grid if x['method']==method and int(x['label'].rsplit('_',1)[1])%3]
        assert result['label']==eligible[-1]['label']
    # None safe: maximize mean ESR before delivery, including for Guard.
    for row in rows:
        index=int(row['grid_label'].rsplit('_',1)[1])
        row['esr']=.95 if index%2 else .96
    selected,_=select_parameters(rows)
    for method,result in selected.items():
        choices=[x for x in grid if x['method']==method and not int(x['label'].rsplit('_',1)[1])%2]
        assert result['label']==choices[-1]['label']
