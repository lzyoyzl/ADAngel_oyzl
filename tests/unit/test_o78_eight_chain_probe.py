from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o78_eight_chain_codegen import generated_header, START, END


def test_payload_pipeline_group_scale_and_epilogue_unchanged():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    out=generated_header(source).replace('o78_eight_chain_experiment','o78_fullk_integer_experiment')
    assert out.split('// Reuse the 64 INT32',1)[1]==source.split('// Reuse the 64 INT32',1)[1]
    prefix_end='    // Preserve N64 operand reuse;'
    assert out.split(prefix_end,1)[0].split('#pragma once',1)[1]==source.split('    // Same N64 stream',1)[0].split('#pragma once',1)[1]
    assert source.count('cute::copy(')==out.count('cute::copy(')==6
    assert source.count('cute::gemm(')==out.count('cute::gemm(')==4
    assert 'cute::_4{},cute::_2{},cute::_4{}' in out
    assert 'partial(i)*=16' in out and 'partial(vi,mi,ni)*coefficient' in out
    assert 'const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*' in out
    assert 'shared_storage.partial' not in out
    for marker in (START,END):
        with pytest.raises(ValueError):generated_header(source.replace(marker,'changed'))


def test_all_scalar_codes_and_g128_intermediate_bounds():
    for a in range(-128,128):
        lo=a&15; hi=a//16
        for w in range(-8,8):
            assert lo*w+16*hi*w==a*w
    assert 128*8*8==8192
    assert 16*(128*8*8)+128*15*8==146432 < 2**31
    # Eight independent atoms have distinct (M,N) coordinates, each4 values.
    assert len({(vi,mi,ni) for vi in range(4) for mi in range(2) for ni in range(4)})==32


def test_reordered_integer_dot_matches_original_for_negatives_and_extrema():
    rng=np.random.default_rng(20261003)
    for extreme in (False,True):
        a=rng.integers(-128,128,(32,128),dtype=np.int64)
        w=rng.integers(-8,8,(32,128),dtype=np.int64)
        if extreme:a.fill(-128);w.fill(-8)
        high0=np.sum((a[:,:64]//16)*w[:,:64],axis=1)
        high1=np.sum((a[:,64:]//16)*w[:,64:],axis=1)
        low0=np.sum((a[:,:64]&15)*w[:,:64],axis=1)
        low1=np.sum((a[:,64:]&15)*w[:,64:],axis=1)
        merged=(high0+high1)*16+low0+low1
        assert np.array_equal(merged,np.sum(a*w,axis=1))
        assert np.all(np.abs((high0+high1)*16)<=131072)


def test_mma_accumulator_count_restricts_to_exact_integer_entry():
    from benchmark_o78_eight_chain_probe import merged_mma_counts
    lines=['Function : target']
    for i in range(64):
        kind='S4' if i<32 else 'U4'
        c='RZ' if i<16 else 'R8'
        lines.append(f'/*{16*i:04x}*/ IMMA.16864.{kind}.S4 R8, R0.ROW, R4.COL, {c} ;')
    lines+=['Function : other','/*0000*/ IMMA.16864.U4.S4 R8, R0.ROW, R4.COL, RZ ;']
    live=dict(loops=[dict(kind='integer',begin_pc='0x0',end_pc='0x3f0')])
    assert merged_mma_counts('\n'.join(lines),live,'target')==dict(u4_total=32,u4_zero_c=0,s4_total=32,s4_zero_c=16)
    with pytest.raises(ValueError):merged_mma_counts('\n'.join(lines),live,'other')


def test_timing_contract_preserves_same_online_preparation_and_four_modes():
    from benchmark_o78_eight_chain_probe import timing_contract
    for mode in ('conversion_only','compute_only','cold','steady_state'):
        result=timing_contract(mode,100)
        assert result['preparation_implementation']=='row_fused_conversion_factor_metadata'
        assert not result['gemm_cufunction_identical_between_policies']
        assert result['weight_cached']==(mode in ('compute_only','steady_state'))
        assert result['stage_timing_inner_repeats']['total']==(100 if mode=='conversion_only' else 1)
