from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
from benchmark_o7_factor_table import Driver, table_coverage, timing_contract, summarize


def test_invalid_source_rejected_before_torch_or_gpu_call():
    driver = Driver.__new__(Driver)
    driver.handles = {0: None, 1: None, 2: None}
    case = SimpleNamespace(oracle={'status_flat': np.array([0, 2], dtype=np.uint32)})
    with pytest.raises(ValueError, match='invalid source'):
        driver.run(case, 1, 'compute_only', 0, 1, 2)


@pytest.mark.parametrize('policy,mode', [(3, 'compute_only'), (1, 'invalid')])
def test_invalid_request_rejected_before_case_access(policy, mode):
    driver = Driver.__new__(Driver)
    driver.handles = {0: None, 1: None, 2: None}
    with pytest.raises(ValueError, match='policy/mode'):
        driver.run(None, policy, mode, 0, 1, 2)


def test_exact_cta_table_selection_not_whole_matrix_fallback():
    factors = np.ones((32, 128), dtype=np.int32)
    flags = np.array([[0, 1], [0, 2]], dtype=np.uint32)
    got = table_coverage(factors, flags)
    assert (got['table_ctas'], got['fp32_fallback_ctas'], got['invalid_ctas']) == (2, 1, 1)
    factors[31, 100] = 1024
    got = table_coverage(factors, flags)
    assert got['table_ctas'] == 1 and got['original_integer_fallback_ctas'] == 1
    factors[31, 100] = 3
    assert table_coverage(factors, flags) == got
    factors[31, 100] = 512
    assert table_coverage(factors, flags)['table_ctas'] == 2


@pytest.mark.parametrize('mode', ['conversion_only', 'compute_only', 'cold', 'steady_state'])
def test_timing_keeps_same_preparation_and_charges_guard(mode):
    value = timing_contract(mode, 100)
    assert value['guard_and_table_generation_in_gemm']
    assert value['preparation_identical_between_policies']
    assert not value['gemm_cufunction_identical_between_policies']
    assert value['preparation_implementation'] == 'row_fused_conversion_factor_metadata'


def test_summary_preserves_bad_cv_and_rejects_missing_pairs():
    rows = []
    for sample in ('a', 'b'):
        for ri in range(3):
            for policy in (0, 1):
                rows.append(dict(sample_id=sample, variant='o7', mode='compute_only', round=ri, candidate=policy,
                    bitwise_equal_v67=True, MSE_regression_passed=True, metadata_exact=True,
                    summary=dict(median_ms=2 if policy else 1, cv_percent=5),
                    stage_summaries={'gemm': dict(cv_percent=5)}, mse_vs_paired_fp16=.01))
    got = summarize(rows, ('compute_only',))
    assert got[1]['paired_speedup'] == .5 and got[1]['selected_cv_failed_records'] == 6
    assert got[1]['median_mse'] == got[1]['mean_mse'] == .01
    with pytest.raises(ValueError):
        summarize(rows[:-1], ('compute_only',))
