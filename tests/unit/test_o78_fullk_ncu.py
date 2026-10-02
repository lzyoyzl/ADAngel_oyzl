"""NCU identity/guard gates; actual four-capture evidence is verified separately."""
import ast
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from analyze_o78_fullk_ncu import analyze_capture
from analyze_roof_scale_ncu import analyze, validate_arithmetic_work, validate_fullk_integer_work


def test_o78_control_and_candidate_work_are_different():
    control = dict(IMMA=16777216, I2F=16777216, FFMA=16777216, FMUL=16777216)
    integer = dict(IMMA=16777216, I2F=524288, FMUL=1048576)
    validate_arithmetic_work(control, 59, False)
    validate_fullk_integer_work(integer)
    with pytest.raises(ValueError):
        validate_fullk_integer_work(control)
    with pytest.raises(ValueError):
        validate_arithmetic_work(integer, 59, False)


@pytest.mark.parametrize('variant,tune,symbol', [
    ('o7',54,'adangel_roof_o78_fullk_candidate'),
    ('o3',54,'adangel_roof_o78_fullk_candidate'),
    ('o8',59,'adangel_roof_o78_fullk_control'),
])
def test_fullk_arithmetic_rejects_unapproved_identity(variant,tune,symbol):
    with pytest.raises(ValueError):
        analyze('','',tune,variant,fullk_integer=True,expected_symbol=symbol)


def test_profile_with_any_fallback_cannot_claim_fullk_instruction_budget():
    receipt = dict(variant='o7',policy=1,expected_kernel='adangel_roof_o78_fullk_candidate',
                   guard=dict(integer_ctas=2047,fallback_ctas=1,invalid_ctas=0))
    with pytest.raises(ValueError,match='all 2048'):
        analyze_capture('','',receipt)


def test_profiler_selects_only_one_requested_policy_after_exact_warmups():
    source = (ROOT/'scripts/profile_o78_fullk_kernel.py').read_text()
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)
             and ast.unparse(n.func)=='driver.run']
    assert len(calls)==1
    assert [ast.literal_eval(n) for n in calls[0].args[-3:]]==[50,1,2]
    assert 'not_Event_performance' in source
    assert 'numerical_checks_passed=True' in source
    capture = (ROOT/'scripts/run_o78_fullk_ncu.py').read_text()
    assert "'--launch-skip', '50', '--launch-count', '1'" in capture
    assert "'--cache-control', 'all', '--clock-control', 'none'" in capture
    assert 'is_relative_to(ROOT)' in capture
