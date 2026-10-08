from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from benchmark_o78_best_combo import route, timing_contract, make_driver
import benchmark_o7_conversion_combo as o7
import benchmark_o8_hif4_swar as o8


@pytest.mark.parametrize('variant,expected', [
    ('o7', ('roof_o78_mx8_swar_benchmark', 0)),
    ('o8', ('roof_o78_hif4_swar_benchmark', 1)),
])
def test_both_policies_use_exact_best_conversion(variant, expected):
    assert route(variant, 0) == route(variant, 1) == expected
    assert issubclass(make_driver(variant), o7.Driver if variant == 'o7' else o8.Driver)


@pytest.mark.parametrize('variant,policy', [('o3',0), ('o7',2), ('o8',-1)])
def test_reject_wrong_measured_route(variant, policy):
    with pytest.raises(ValueError):
        route(variant, policy)


@pytest.mark.parametrize('variant', ['o7','o8'])
@pytest.mark.parametrize('mode', ['conversion_only','compute_only','cold','steady_state'])
def test_timing_scope(variant, mode):
    c = timing_contract(variant, mode, 100)
    assert not c['gemm_cufunction_identical_between_policies']
    assert c['weight_preparation_identical'] and c['activation_preparation_identical']
    assert not c['new_CUDA_compilation'] and not c['production_default_changed']
    assert not c['failed_v126_packed_MX8_candidate_executed']
    assert c['stage_timing_inner_repeats']['total'] == (100 if mode == 'conversion_only' else 1)
    assert c['payload_norm_checked_after_every_call']


def test_original_reference_and_audits_preserved():
    source = (ROOT/'scripts/benchmark_o78_best_combo.py').read_text()
    assert 'checked_streaming(candidate' in source
    assert "receipt['baseline_cubin_sha256']" in source
    assert 'if policy == 2:' in source
    assert 'return super().run(case, policy' in source
    assert "SYMBOLS[policy]" in source
    assert 'full_v99_source_identity_equal=True' in source


@pytest.mark.parametrize('variant', ['o7', 'o8'])
@pytest.mark.parametrize('mode', ['conversion_only','compute_only','cold','steady_state'])
def test_analyzer_accepts_exact_route_and_rejects_false_identity(variant, mode):
    from analyze_o78_best_combo import validate_row
    from analyze_mx8_warp_lut import STAGES
    from benchmark_o78_best_combo import SYMBOLS, PREPARATIONS
    import copy
    name, selector = route(variant, 1)
    row = dict(variant=variant, candidate=1, mode=mode, bitwise_equal_v67=True,
        MSE_regression_passed=True, finite_fp32=True, metadata_exact=True,
        mse_vs_v67=0, max_abs_vs_v67=0,
        resources=dict(kernel_symbol=SYMBOLS[1], preparation=PREPARATIONS[variant],
            conversion_host_function=name, conversion_selector=selector),
        raw_ms={stage:[1.0]*200 for stage in STAGES[mode]},
        stage_summaries={stage:dict(median_ms=1.0,cv_percent=0.0) for stage in STAGES[mode]},
        **timing_contract(variant, mode, 100))
    validate_row(row, variant)
    for field, value in (('gemm_cufunction_identical_between_policies',True),
                         ('weight_preparation_identical',False), ('mse_vs_v67',1e-10)):
        bad = copy.deepcopy(row); bad[field] = value
        with pytest.raises(ValueError):
            validate_row(bad, variant)
    bad = copy.deepcopy(row); bad['resources']['conversion_selector'] = 1-selector
    with pytest.raises(ValueError):
        validate_row(bad, variant)
    bad = copy.deepcopy(row); bad['raw_ms'][STAGES[mode][0]].pop()
    with pytest.raises(ValueError):
        validate_row(bad, variant)
