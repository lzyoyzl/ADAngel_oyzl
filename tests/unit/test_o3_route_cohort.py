"""CPU algebra/source contracts only, not CUDA mapping or numerical proof."""
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o3_route_cohort_codegen import LIMITS


def test_signed_int8_split_and_modular_fullk_routes():
    a=np.arange(-128,128,dtype=np.int64)
    assert np.array_equal(a,(a&15)+16*(a>>4))
    rng=np.random.default_rng(137)
    for case in range(128):
        a=rng.integers(-128,128,(32,128),dtype=np.int64)
        w=rng.integers(-8,8,(32,128),dtype=np.int64)
        factor=1<<rng.integers(0,8,32,dtype=np.int64)
        assert factor.sum()<=16383  # Original worst-case INT32 full-K guard.
        h=np.sum((a>>4)*w,axis=1);l=np.sum((a&15)*w,axis=1)
        expected=int(np.sum((h*16+l)*factor))
        high=int(np.sum(h*factor))&0xffffffff
        low=int(np.sum(l*factor))&0xffffffff
        merged=(16*high+low)&0xffffffff
        signed=merged if merged<2**31 else merged-2**32
        assert signed==expected


def test_barriers_shared_epilogue_and_original_guard_remain_explicit():
    source=(ROOT/'csrc/sm80/o3_route_cohort_candidate.cuh').read_text()
    wrapper=(ROOT/'csrc/sm80/roof_o3_route_cohort_probe.cu').read_text()
    assert source.count('__syncthreads()')==3
    assert source.index('for(int group=0;group<32;')<source.index('uint32_t* low_sum')
    assert '16u*acc(i)' in source and 'uint32_t(partial' in source
    assert 'cute::_32{},cute::_64{}' in source
    assert '__launch_bounds__(256,2)' in wrapper
    assert 'if(flag&6u) return' in wrapper and 'if(flag&1u)' in wrapper
    assert 'if(threadIdx.x>=128) return' in wrapper
    assert LIMITS['max_allocated_gpr']==128 and LIMITS['max_hot_local_instructions']==4
