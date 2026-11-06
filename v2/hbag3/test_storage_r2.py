"""Archival regression checks protect the scientific evidence byte for byte."""
import json
import numpy as np
import pytest
from hbag3.compact_r2 import encode,decode,compact_slots,read_slots


def test_archive_preserves_bits_shape_dtype_and_detects_corruption():
    rng=np.random.default_rng(1000)
    # Include negative zero, infinities and NaN bit patterns, not just equal values.
    bits=rng.integers(0,2**32,size=(8,7),dtype=np.uint32)
    bits[0,:4]=[0,0x80000000,0x7f800000,0x7fc00001]
    arrays={'misc':bits.view(np.float32),'flags':rng.random((8,7))>.5,
            'index':np.arange(56,dtype=np.int32).reshape(8,7),'empty':np.empty((0,7),np.float32)}
    summary={'config':{'capacity_j':1e6},'dt_s':60.}
    blob=encode(arrays,{},summary);result=decode(blob,{},summary)
    assert set(result)==set(arrays)
    for k,v in arrays.items():
        assert result[k].dtype==v.dtype and result[k].shape==v.shape
        assert result[k].tobytes()==v.tobytes()
    damaged=bytearray(blob);damaged[-10]^=1
    with pytest.raises(Exception):decode(bytes(damaged),{},summary)


def test_scalar_archive_preserves_original_bytes(tmp_path):
    rows=[{'slot':0,'margin_j':-0.0,'gap':None},{'slot':1,'margin_j':1.2345678901234567}]
    path=tmp_path/'slots.json';path.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    before=path.read_bytes();compact_slots(tmp_path)
    import lzma
    assert lzma.decompress((tmp_path/'slots.json.xz').read_bytes())==before
    assert read_slots(tmp_path)==rows and not path.exists()


def test_window_archives_roundtrip_without_touching_completed_arrays(tmp_path):
    from hbag3.compact_r2 import compact_run,read_json_archive
    import lzma
    rows=[{'satellite':0,'correlation':None,'left_censored':True},{'satellite':1,'correlation':-.823456789}]
    original=(json.dumps(rows,indent=2)+'\n').encode()
    for name in ['potential_load_windows.json','eclipse_windows.json']:(tmp_path/name).write_bytes(original)
    compact_run(tmp_path)
    compact_run(tmp_path)
    for name in ['potential_load_windows.json','eclipse_windows.json']:
        path=tmp_path/name
        assert read_json_archive(path)==rows
        assert lzma.decompress(path.with_suffix('.json.xz').read_bytes())==original


def test_energy_predictor_retains_rounding_residuals_and_supply_clipping():
    rng=np.random.default_rng(1000);n,m=50,8
    q=rng.uniform(8,200,(n,m));h=rng.choice([0.,500.],(n,m));e=np.empty((n+1,m));e[0]=9500.
    net=h-55.-q
    for t in range(n):e[t+1]=np.clip(e[t]+60*(.95*np.maximum(net[t],0)-np.maximum(-net[t],0)/.95),0,10000.)
    arrays={'energy_j':e.astype(np.float32),'terminal_w':q.astype(np.float32)}
    inputs={'harvest_w':h.astype(np.float32)};summary={'config':{'capacity_j':10000.,'base_w':55.},'dt_s':60.}
    blob=encode(arrays,inputs,summary);decoded=decode(blob,inputs,summary)
    for k,v in arrays.items():assert decoded[k].tobytes()==v.tobytes()
    hlen=int.from_bytes(blob[:8],'little');header=json.loads(blob[8:8+hlen])
    assert next(r for r in header if r['key']=='energy_j')['mode']=='energy_recurrence_xor'


def test_nonuniform_pacing_archive_roundtrips_exactly():
    rng=np.random.default_rng(1000);n,m=9,4
    e=rng.uniform(700000,950000,(n+1,m));reserve=np.full((n+1,m),700000.)
    surplus=e[:-1]-reserve[:-1];fraction=rng.uniform(.001,.8,(n,m))
    arrays={k:np.asarray(v,dtype=np.float32) for k,v in dict(energy_j=e,
        discretionary_energy_j=surplus,pace_weight_fraction=fraction,pace_w=8+.95*surplus*fraction/60.).items()}
    inputs=dict(harvest_w=np.zeros((n,m)),reserve_j=reserve)
    summary=dict(config=dict(capacity_j=1e6,sleep_w=2.),dt_s=60.)
    blob=encode(arrays,inputs,summary);restored=decode(blob,inputs,summary)
    for key,value in arrays.items():assert restored[key].tobytes()==value.tobytes()


def test_v23_price_diagnostics_preserve_nan_flags_and_rounding_bits():
    rng=np.random.default_rng(1000);n,m=8,4
    q=rng.uniform(80,150,(n,m));z=rng.uniform(0,15000,(n,m));remaining=np.full((n,m),4)
    h=rng.choice([0.,275.],(n,m));computed=rng.random((n,m))>.3
    star=np.where(h==0,.95*z/(remaining*60),rng.integers(0,16000,(n,m))*232/16384)
    star=np.where(computed,star,np.nan);pace=np.where(h==0,8+.95*z/60,8+star)
    full=np.empty((n,m));var=np.full(m,8.)
    for t in range(n):
        full[t]=92+var;on=q[t]>=100;var[on]=.8*var[on]+.2*(q[t,on]-100)
    arrays={key:np.asarray(value,np.float32) for key,value in dict(terminal_w=q,discretionary_energy_j=z,
        pace_w=pace,p_star_w=star,p_full_w=full,pace_weight_fraction=np.ones((n,m))).items()}
    arrays.update(p_star_computed=computed,remaining_window_slots=remaining.astype(np.int32))
    # The old surplus predictor requires energy/reserve; supply consistent inputs.
    arrays['energy_j']=np.r_[z+700000.,np.full((1,m),700000.)].astype(np.float32)
    inputs=dict(harvest_w=h,reserve_j=np.full((n+1,m),700000.))
    summary=dict(config=dict(capacity_j=1e6,sleep_w=2.,base_w=55.),dt_s=60.)
    blob=encode(arrays,inputs,summary);decoded=decode(blob,inputs,summary)
    for key,value in arrays.items():assert decoded[key].tobytes()==value.tobytes()
