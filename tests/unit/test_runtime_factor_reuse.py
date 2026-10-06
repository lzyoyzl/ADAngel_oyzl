from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from inspect_runtime_factor_reuse import factor_observation,summarize


def test_uniform_vectors_and_trivial_units():
    f=np.full((32,64),2,dtype=np.int32);st=np.zeros(64,dtype=np.uint32)
    x=factor_observation(f,st)
    assert x['warp_panels']==64 and x['warp_same4_fraction']==1
    assert x['same4_mean_saved_coefficient_imad']==48
    f.fill(1);assert factor_observation(f,st)['same4_mean_saved_coefficient_imad']==64
    f.fill(0);assert factor_observation(f,st)['same4_mean_saved_coefficient_imad']==64


def test_per_lane_equality_is_not_SIMD_instruction_savings():
    f=np.full((1,64),2,dtype=np.int32);st=np.zeros(64,dtype=np.uint32)
    f[0,[8,24]]=3
    x=factor_observation(f,st)
    assert x['per_lane_same4_fraction']==7/8
    assert x['warp_same4_fraction']==0
    assert x['same4_mean_saved_coefficient_imad']==0


def test_vector_equivalence_is_not_one_common_scalar():
    f=np.tile(np.arange(2,10,dtype=np.int32),8).reshape(1,64)
    x=factor_observation(f,np.zeros(64,dtype=np.uint32))
    assert x['warp_same4_fraction']==1 and x['same4_mean_saved_coefficient_imad']==48
    f=np.repeat([2,3,2,3,4,5,4,5],8).reshape(1,64).astype(np.int32)
    assert factor_observation(f,np.zeros(64,dtype=np.uint32))['ideal_any_pattern_mean_saved_coefficient_imad']==0


def test_actual_M16_warp_layout_repeats_at_M32_not_M16():
    f=np.repeat([2,2,3,3,2,2,3,3],8).reshape(1,64).astype(np.int32)
    x=factor_observation(f,np.zeros(64,dtype=np.uint32))
    assert x['quad_offsets']==[0,8,32,40]
    assert x['warp_same4_fraction']==1
    f[0,32]=7
    assert factor_observation(f,np.zeros(64,dtype=np.uint32))['warp_same4_fraction']==.5


def test_unrepresentable_rows_excluded_without_hiding_cost():
    f=np.ones((32,64),dtype=np.int32);st=np.zeros(64,dtype=np.uint32);st[0]=1
    x=factor_observation(f,st)
    assert x['excluded_warp_panels']==32 and x['representable_warp_panels']==32
    with pytest.raises(ValueError):factor_observation(f,np.ones(64,dtype=np.uint32))
    with pytest.raises(ValueError):factor_observation(f.astype(np.uint32),st)


def test_full24_gate_is_not_performance_acceptance():
    x=factor_observation(np.ones((32,64),dtype=np.int32),np.zeros(64,dtype=np.uint32))
    rows=[dict(sample_id=str(s),variant=v,statistics=x) for s in range(24) for v in ('o7','o8')]
    result=summarize(rows)
    assert all(r['raw_opportunity_gate'] for r in result['variants'])
    assert not any(result[k] for k in ('new_MSE_measured','new_GEMM_measured','new_kernel_implemented','production_default_changed'))
    with pytest.raises(ValueError):summarize(rows[:-1])
