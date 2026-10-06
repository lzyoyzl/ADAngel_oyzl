import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from inspect_scale_profile_buckets import activation_profiles,constant_weights,summarize


def test_profile_proportionality_and_complete_tiles_not_per_lane_coincidence():
    profile=np.arange(1,33,dtype=np.int32)
    f=profile[:,None]*np.arange(1,129,dtype=np.int32)[None,:]
    st=np.zeros(128,dtype=np.uint32)
    r=activation_profiles(f,st)
    assert r['unique_profiles']==1 and r['largest_profile_rows']==128
    assert r['covered_rows']==128 and r['optimally_bucketed_tiles']==2
    assert r['old_homogeneous_tiles']==2
    assert r['optimistic_loop_instruction_reduction_percent']==pytest.approx(64/383*100)
    st[0]=1
    r=activation_profiles(f,st)
    assert r['covered_rows']==64 and r['excluded_rows']==1


def test_bucketing_adds_opportunity_but_partial_tails_are_not_counted():
    p=np.ones(32,dtype=np.int32);q=p.copy();q[0]=2
    f=np.stack([p if i%2 else q for i in range(128)],axis=1)
    st=np.zeros(128,dtype=np.uint32)
    r=activation_profiles(f,st)
    assert r['old_homogeneous_tiles']==0 and r['optimally_bucketed_tiles']==2
    f[:,0]=p
    r=activation_profiles(f,st)
    assert r['covered_rows']==64 and r['largest_profile_rows']==65
    f.fill(0)
    assert activation_profiles(f,st)['covered_rows']==0


def test_constant_weight_classes_need_all32_groups_and_whole_N128_tile():
    c=np.full((256,32),127,dtype=np.uint8);c[::2,1]=128
    r=constant_weights(c)
    assert r['K_constant_columns']==128 and r['old_homogeneous_tiles']==0
    assert r['optimally_bucketed_tiles']==1 and r['covered_fraction']==.5
    c[1,31]=126
    assert constant_weights(c)['covered_columns']==0
    c.fill(0)
    assert constant_weights(c)['covered_columns']==0
    c[0,0]=255
    with pytest.raises(ValueError):constant_weights(c)


def test_factor_math_common_profile_folding_is_exact_integer_arithmetic():
    a=[3,6,12];w=[7,5,2];p=[11,-13,29]
    for gcd in (1,5,19):
        before=sum(x*y*z*gcd for x,y,z in zip(a,w,p))
        after=sum(x*y*z for x,y,z in zip(a,w,p))*gcd
        assert before==after


def test_summary_requires_all24_and_optimistic_gate_not_a_speedup_claim():
    rows=[dict(sample_id=f's{i}',variant=v,statistics=dict(covered_fraction=.5,
          optimistic_loop_instruction_reduction_percent=6.0,
          old_homogeneous_tiles=0,optimally_bucketed_tiles=32)) for i in range(24) for v in ('o3','o7','o8')]
    r=summarize(rows)
    assert all(v['opportunity_gate'] for v in r['variants'])
    assert not r['new_kernel_implemented'] and not r['new_GEMM_measured'] and not r['new_MSE_measured']
    rows[0]['statistics']['optimistic_loop_instruction_reduction_percent']=0
    assert summarize(rows)['variants'][0]['mean_optimistic_loop_work_reduction_percent']==5.75
    with pytest.raises(ValueError):summarize(rows[:-1])
    duplicate=copy.deepcopy(rows);duplicate[-1]=duplicate[0]
    with pytest.raises(ValueError):summarize(duplicate)


def test_invalid_shapes_dtypes_and_guard_cannot_create_fake_opportunity():
    f=np.ones((32,64),dtype=np.int32);st=np.zeros(64,dtype=np.uint32)
    for bad in (f.astype(np.int64),f[:31],f[:,:63],-f):
        with pytest.raises(ValueError):activation_profiles(bad,st)
    st[0]=3
    with pytest.raises(ValueError):activation_profiles(f,st)
