"""NCU identity/guard gates; actual four-capture evidence is verified separately."""
import ast
import hashlib
import json
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


def test_archived_four_captures_reproduce_every_metric_and_source_hash():
    report = ROOT/'docs/evidence/a100_o378_roof_v70_ncu/reports/o378_roof_v70_ncu'
    saved = json.loads((report/'analysis.json').read_text())
    assert len(saved['rows']) == 4
    for row in saved['rows']:
        stem = f"{row['variant']}_p{row['policy']}"
        receipt = json.loads((report/stem/'receipt.json').read_text())
        actual = analyze_capture((report/f'{stem}_raw.csv').read_text(),
                                 (report/f'{stem}_source_sass.csv').read_text(), receipt)
        assert actual == row
        assert receipt['filtered_launch_skip'] == 50 and receipt['filtered_launch_count'] == 1
        assert receipt['numerical_checks_passed'] and not receipt['production_default_changed']
        assert receipt['sample_id'] == 'layer_00_q_proj'
        assert receipt['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
        assert receipt['resources']['active_blocks_per_sm'] == 3
        if row['policy'] == 1:
            # Static local allocation includes the fallback path. It does not
            # imply the integer path executed local load/store instructions.
            assert row['opcodes'].get('LDL',0) == row['opcodes'].get('STL',0) == 0
            assert row['source_memory_work']['L2 Theoretical Sectors Local'] == 0
    for entry in saved['sources']:
        path = report/Path(entry['file']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']


def test_profile_numerical_outputs_match_unprofiled_v69_same_sample():
    report = ROOT/'docs/evidence/a100_o378_roof_v70_ncu/reports/o378_roof_v70_ncu'
    run = ROOT/'docs/evidence/a100_o378_roof_v69/runs/o378_roof_v69_trace24'
    rows = [json.loads(x) for x in (run/'results.jsonl').read_text().splitlines()]
    for variant in ('o7','o8'):
        for policy in (0,1):
            receipt = json.loads((report/f'{variant}_p{policy}'/'receipt.json').read_text())
            event = next(r for r in rows if r['variant']==variant and r['candidate']==policy
                         and r['mode']=='compute_only' and r['sample_id']=='layer_00_q_proj')
            assert receipt['mse_vs_paired_fp16'] == event['mse_vs_paired_fp16']
            assert receipt['mse_vs_best'] == event['mse_vs_best']
            assert receipt['max_abs_vs_best'] == event['max_abs_vs_best']
