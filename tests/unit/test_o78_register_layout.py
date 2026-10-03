"""The v85 layout candidate must preserve math, charge preparation and retain fallback."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_o78_register_layout_codegen import generated_header,generated_driver,NEW_A,NEW_B
from benchmark_o78_register_layout import register_order_indices,packed_reference,timing_contract


def test_layout_permutation_is_bijective_and_roundtrips_every_nibble():
    rng=np.random.default_rng(20261003)
    for weight in (False,True):
        order=register_order_indices(weight)
        assert sorted(order.tolist())==list(range(2048))
        shape=(32,64,64) if weight else (2,32,64,64)
        raw=rng.integers(0,256,shape,dtype=np.uint8)
        packed=packed_reference(raw,weight)
        assert packed.nbytes==raw.nbytes
        recovered=np.stack((packed&15,packed>>4),axis=-1).reshape(1 if weight else 2,32,4,2048)
        recovered=recovered[...,np.argsort(order)]
        back=(recovered[...,::2]|(recovered[...,1::2]<<4)).reshape(shape)
        assert np.array_equal(back,raw)


def test_integer_math_pipeline_and_epilogue_are_unchanged():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    old,new=eight_header(source),generated_header(source)
    assert NEW_A in new and NEW_B in new and 'ld.shared.v4.b32' in new
    assert 'sizeof(Storage)==34304' in new
    start='      // Eight independent chains:'
    assert new[new.index(start):].replace('o78_register_layout_experiment','o78_eight_chain_experiment')==old[old.index(start):]
    assert new.count('cp.async.wait_group 0')==old.count('cp.async.wait_group 0')==1
    assert new.count('__syncthreads()')==old.count('__syncthreads()')==1
    with pytest.raises(ValueError):generated_header(source.replace('s.low[slot]+la(row,col)','s.low[slot]'))


def test_timer_is_the_original_v73_body_with_explicit_repack_cost():
    source=(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    new=generated_driver(source)
    marker='extern "C" int roof_o78_row_fused_benchmark('
    body=new[new.index('extern "C" int roof_o78_register_layout_benchmark('):]
    assert body.replace('roof_o78_register_layout_benchmark(', 'roof_o78_row_fused_benchmark(').replace(
        'RegisterLayoutOnline x(', 'RowFusedOnline x(')==source[source.index(marker):]
    assert 'RowFusedOnline::weight(true)' in new and 'RowFusedOnline::activation(true)' in new
    assert new.count('o78_register_pack::pack<')==2
    c=timing_contract('cold',100)
    assert c['preparation_launches_by_candidate']['1']=={'weight':2,'activation':3}
    assert c['total_timing']=='single_execution_cuda_event'
    assert c['stage_timing_inner_repeats']['weight_conversion']==100
    assert 'weight_preparation_launches' not in c
    wrapper=(ROOT/'csrc/sm80/roof_o78_register_layout_probe.cu').read_text()
    assert 'if(flag>1u) return;' in wrapper and 'if(flag==1u)' in wrapper
    assert 'true,true,6,false,false,true,false,2>' in wrapper
    assert 'a+size_t(m)*k,w+size_t(n)*(k/2)' in wrapper
