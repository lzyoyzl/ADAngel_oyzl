"""Independent integer/data contracts, not GPU MSE or speed evidence."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from analyze_sparse_q4_feasibility import paired_decomposition,VALID_METADATA
from inspect_activation_high_sparsity import digits,high_statistics,model,summarize,CORRECTION_BUDGET_MS


@pytest.mark.parametrize('variant,lo,hi',[('o3',-128,127),('o7',-112,112),('o8',-30,30)])
def test_all_source_integer_values_exact_and_no_unsupported_O3_balanced(variant,lo,hi):
    values=np.arange(lo,hi+1,dtype=np.int16).astype(np.int8)
    q=np.tile(np.resize(values,512),(2,1))
    for policy in ('ordinary',) if variant=='o3' else ('ordinary','balanced'):
        low,high=digits(q,variant,policy)
        np.testing.assert_array_equal(q.astype(np.int16),low.astype(np.int16)+16*high)
        main,residual,metadata,_=paired_decomposition(high)
        np.testing.assert_array_equal(q.astype(np.int16),low.astype(np.int16)+16*(main.astype(np.int16)+residual))
        assert set(metadata.ravel())<=VALID_METADATA
    if variant=='o3':
        with pytest.raises(ValueError):digits(q,variant,'balanced')


def test_G128_integer_dot_with_exact_sparse_residual_preserves_scale_semantics():
    rng=np.random.default_rng(20261007)
    a=rng.integers(-30,31,(7,256),dtype=np.int8)
    w=rng.integers(-7,8,(5,256),dtype=np.int8).astype(np.int64)
    low,high=digits(a,'o8','balanced');main,residual,_,_=paired_decomposition(high)
    af=rng.integers(1,4,(7,2));wf=rng.integers(1,8,(5,2))
    direct=np.zeros((7,5),dtype=np.int64);split=direct.copy()
    for g in range(2):
        k=slice(g*128,(g+1)*128);coeff=af[:,g,None]*wf[:,g]
        direct+=(a[:,k].astype(np.int64)@w[:,k].T)*coeff
        partial=low[:,k].astype(np.int64)@w[:,k].T
        partial+=16*(main[:,k].astype(np.int64)@w[:,k].T+residual[:,k].astype(np.int64)@w[:,k].T)
        split+=partial*coeff
    np.testing.assert_array_equal(direct,split)


def test_actual_int4_pair_constraint_not_optimistic_arbitrary_two_of_four():
    q=np.tile(np.array([16,0,16,0,16,0,0,0],dtype=np.int8),(1,16))
    s=high_statistics(q,'o8','ordinary')
    assert s['minimum_residual_nonzeros']==16 # Three active pairs: retain only two.
    assert s['minimum_residual_fraction']==.125
    assert s['representation_exact'] and s['no_pruning']


def test_favourable_correction_model_not_measured_or_additive_bound():
    zero=model(4096**2,0,4096)
    dense=model(4096**2,4096**2//2,4096)
    assert zero['data_gate_passed'] and not dense['data_gate_passed']
    assert zero['max_ideal_MMA_capacity_saving_ms']==CORRECTION_BUDGET_MS
    assert dense['mma_can_overlap_so_no_addition_to_mma_floor']
    assert not dense['measured_speedup'] and dense['gate_is_investment_filter_not_impossibility_proof']
    with pytest.raises(ValueError):summarize([])
