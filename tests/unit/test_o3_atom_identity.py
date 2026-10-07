from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from inspect_o3_atom_identity import (atom_statistics,observe,summarize,
    activation_statistics,observe_activation,summarize_activation)


def test_different_column_anchors_can_still_be_all_unit():
    c=np.broadcast_to((np.arange(128)%5+120).astype(np.uint8)[:,None],(128,32)).copy()
    r=atom_statistics(c)
    assert r['identity_panels']==512 and r['identity_panel_fraction']==1
    assert r['optimistic_instruction_work_reduction_percent']==pytest.approx(64/323*100)


def test_not_lane_coincidence_and_not_K_constant_bucketing():
    c=np.full((128,32),127,dtype=np.uint8);c[:,1]=128
    r=atom_statistics(c)
    assert r['identity_panel_fraction']==31/32
    assert r['per_group_identity_panels'][1]==0
    c[0,2]=129
    r=atom_statistics(c)
    assert r['identity_panels']==16*31-1
    assert r['per_group_identity_panels'][2]==15


def test_original_dtype_shape_and_normal_scale_contract():
    c=np.full((128,32),127,dtype=np.uint8)
    for bad in (c.astype(np.int16),c[:127],c[:,:31],c[:0]):
        with pytest.raises(ValueError):atom_statistics(bad)
    for value in (0,255):
        c[0,0]=value
        with pytest.raises(ValueError):atom_statistics(c)


def test_all24_original_SHA_snapshots_and_stop_below_instruction_budget():
    rows=observe();r=summarize(rows)
    assert len(rows)==24 and r['mean_identity_panel_fraction']==pytest.approx(0.1408360799153646)
    assert r['optimistic_loop_instruction_work_reduction_percent']==pytest.approx(2.7905600973942213)
    assert not r['opportunity_gate']
    assert not any(r[k] for k in ('candidate_GEMM_implemented','candidate_GPU_launched',
                                 'new_performance_result','new_MSE_result','production_default_changed'))
    with pytest.raises(ValueError):summarize(rows[:-1])
    with pytest.raises(ValueError):summarize(rows[:-1]+rows[:1])


def test_host_mapping_probe_never_creates_a_GPU_launch_or_context():
    source=(ROOT/'tests/cuda/validate_o3_atom_columns.cu').read_text()
    assert 'o3_row_scale_epilogue_experiment::O3AmpereConfig' in source
    assert 'partition_C(identity)' in source and '(warp/2)*8+ni*16+j' in source
    assert 'covered.size()==8192' in source
    for token in ('<<<','cudaMalloc','cudaSetDevice','cuLaunchKernel','__global__'):
        assert token not in source


def test_unit_product_necessary_condition_includes_zero_weight_factors():
    for a in range(16):
        for w in range(16):
            if a*w==1:assert a==w==1
    f=np.ones((32,64),dtype=np.int32);st=np.zeros(64,dtype=np.uint32)
    r=activation_statistics(f,st)
    assert r['necessary_A_unit_fraction_upper_bound']==1
    f[0,0]=0
    assert activation_statistics(f,st)['necessary_A_unit_panels']==127
    f.fill(1);st[0]=1
    assert activation_statistics(f,st)['necessary_A_unit_panels']==96


def test_full24_O78_upper_bounds_not_complete_weight_coverage():
    rows=observe_activation();r=summarize_activation(rows)
    assert len(rows)==48
    assert r[0]['mean_A_only_unit_fraction_upper_bound']==pytest.approx(0.016825358072916668)
    assert r[0]['optimistic_loop_instruction_work_reduction_percent_upper_bound']==pytest.approx(0.5623096170583116)
    assert r[1]['mean_A_only_unit_fraction_upper_bound']==0
    assert not any(v['opportunity_gate'] for v in r)
    with pytest.raises(ValueError):summarize_activation(rows[:-1])
