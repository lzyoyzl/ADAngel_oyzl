from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_operand_stream_codegen import generated_header, worth_runtime
from probe_o78_eight_chain_codegen import generated_header as eight_header


def test_same_global_pipeline_factors_and_epilogue():
    old=eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    new=generated_header().replace('o78_operand_stream_experiment','o78_eight_chain_experiment')
    assert old.split('__device__ __forceinline__ void body')[0]==new.split('__device__ __forceinline__ void body')[0]
    assert old.split('// Reuse the 64 INT32')[1]==new.split('// Reuse the 64 INT32')[1]
    for s in ('slot=group%2','cp.async.wait_group 0','__syncthreads();','prefetch(s,1-slot,group+1'):
        assert s in new
    assert 's.activation_factors[slot][cute::get<0>(coord)]*' in new
    assert 's.weight_factors[slot][cute::get<1>(coord)]' in new
    assert 'acc(vi,mi,full_ni)+=partial(vi,ni)*coefficient' in new


def test_atom_view_not_guessed_lane_map_or_integer_conversion():
    text=generated_header()
    assert 'cute::get<0>(coords(cute::_0{},mi,nb*cute::_4{}))/16' in text
    assert 'cute::recast<cutlass::uint4b_t>(ar)' in text
    assert 'partial(i)*=16' in text
    assert text.count('cute::gemm(')==4
    assert text.count('ac_high.partition_S(atom_tile_a(')==2
    assert text.count('ac_low.partition_S(atom_tile_a(')==2
    assert 'a0=thr.partition_fragment_A' not in text
    assert 'cute::_4{},cute::_4{}' in text
    wrapper=(ROOT/'csrc/sm80/roof_o78_operand_stream_probe.cu').read_text()
    assert '__launch_bounds__(128,3)' in wrapper and 'flag==1u' in wrapper
    assert 'full_sample_args()' in (ROOT/'scripts/benchmark_o78_operand_stream.py').read_text()


def test_exhaustive_int8_nibbles_and_group_math():
    for a in range(-128,128):
        for w in range(-8,8):
            assert ((a>>4)*w)*16+(a&15)*w==a*w
    rng=np.random.default_rng(94)
    a=rng.integers(-128,128,(64,128),dtype=np.int64)
    w=rng.integers(-8,8,(128,128),dtype=np.int64)
    p=((a[:,:64]>>4)@w[:,:64].T+(a[:,64:]>>4)@w[:,64:].T)*16
    p+=(a[:,:64]&15)@w[:,:64].T;p+=(a[:,64:]&15)@w[:,64:].T
    assert np.array_equal(p,a@w.T)
    assert len({(vi,mi,nb*4+ni) for nb in range(2) for mi in range(2) for ni in range(4) for vi in range(4)})==64


def test_gate_requires_real_four_CTA_register_budget_and_known_extra_LDSM():
    counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':24}
    live=dict(allocated_gpr=128,loops=[dict(kind='integer',opcode_counts=counts)])
    assert worth_runtime(live)
    assert not worth_runtime(dict(live,allocated_gpr=129))
    for change in ({'LDL.LU':1},{'STL.64':1},{'LDSM.16.M88.4':32},{'IMMA.16864.S4.S4':16}):
        assert not worth_runtime(dict(live,loops=[dict(kind='integer',opcode_counts={**counts,**change})]))


def test_timing_contract_keeps_online_conversion_costs():
    from benchmark_o78_operand_stream import timing_contract
    for mode in ('conversion_only','compute_only','cold','steady_state'):
        r=timing_contract(mode,100)
        assert r['preparation_implementation']=='row_fused_conversion_factor_metadata'
        assert not r['new_preparation_or_layout']
        assert r['stage_timing_inner_repeats']['total']==(100 if mode=='conversion_only' else 1)
