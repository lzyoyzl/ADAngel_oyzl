"""Validate the O3 parser using the historical exact same v89 binary capture."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from analyze_o3_mma_phase_stalls import CODEGEN, analyze_capture, normalized_o3


@pytest.fixture(scope='module')
def inputs():
    directory = ROOT/'docs/evidence/a100_o378_roof_v90_ncu/reports/o378_roof_v90_ncu'
    prior = next(row for row in json.loads((directory/'analysis.json').read_text())['rows']
                 if row['variant'] == 'o3' and row['policy'] == 1)
    return ((CODEGEN/'o3_grouped_cta.sass').read_text(), (CODEGEN/'liveness.txt').read_text(),
            (directory/'o3_1_source_sass.csv').read_text(), prior)


def test_exact_323_instruction_role_and_dynamic_closure(inputs):
    r = analyze_capture(*inputs); s = r['integer_loop_static']; categories = r['consumer_categories']
    assert sum(s.values()) == 323
    assert s['high_times_16'] == s['weighted_integer_accumulate'] == 64
    assert sum(row['warp_instructions'] for row in categories.values()) == 88231936
    assert r['reason_samples']['wait'] == 4333
    assert 'partial_times_activation_factor' not in s
    assert not r['new_performance_result'] and not r['production_default_changed']


def test_operand_mismatch_and_false_sampling_totals_rejected(inputs):
    sass, live, source, prior = inputs
    wrong = source.replace('MOV R1, c[0x0][0x28]', 'MOV R2, c[0x0][0x28]', 1)
    with pytest.raises(ValueError, match='instruction mismatch'):
        analyze_capture(sass, live, wrong, prior)
    wrong_prior = deepcopy(prior); wrong_prior['pc_sampling']['reason_samples']['wait'] += 1
    with pytest.raises(ValueError, match='sampling total'):
        analyze_capture(sass, live, source, wrong_prior)
    assert normalized_o3('BSSY B0, 0x52c0') == normalized_o3('BSSY B0, 0x7fcdf529d9c0')
    assert normalized_o3('BSSY B0, 0x52c0') != normalized_o3('BSSY B1, 0x52c0')
