"""Regression over retained raw v89 first/repeat evidence, not rounded tables."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'docs/evidence/a100_o378_roof_v89'
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_grouped_cta_retest import read_run, compare, summarize


@pytest.mark.parametrize('kind', ['o3', 'o78'])
def test_complete_pairs_mse_identity_and_unfiltered_repeat(kind):
    first = read_run(E / f'runs/o378_roof_v89_{kind}')
    retry = read_run(E / f'runs/o378_roof_v89_{kind}_retry')
    stored = json.loads((E / f'reports/o378_roof_v89_{kind}_retest_analysis.json').read_text())
    actual = compare(first, retry)
    assert first['args']['warmup'] == 50 and retry['args']['warmup'] == 1000
    assert stored['no_raw_filtering'] and stored['no_default_change']
    assert len(actual) == len(stored['records'])
    for a, b in zip(actual, stored['records']):
        assert (a['variant'], a['policy']) == (b['variant'], b['policy'])
        for phase in ('first', 'retry'):
            for field in ('median_ms', 'paired_speedup', 'median_cv_percent', 'max_cv_percent',
                          'median_mse', 'mean_mse', 'selected_cv_failed_records', 'any_stage_cv_failed_records'):
                assert a[phase][field] == pytest.approx(b[phase][field], rel=1e-10, abs=1e-12)
            assert a[phase]['paired_speedup_ci95'] == pytest.approx(b[phase]['paired_speedup_ci95'])
        assert not a['strict_cv_passed'] if a['retry']['any_stage_cv_failed_records'] else a['strict_cv_passed']


def test_four_mode_raw_output_and_cold_stage_failures_remain_visible():
    directory = E / 'runs/o378_roof_v89_o3_four'
    rows = [json.loads(s) for s in (directory / 'results.jsonl').read_text().splitlines()]
    result = summarize(rows, 1, 200)
    assert len(rows) == 192 and len(result) == 8
    assert all(x['bitwise_previous_best'] for x in result)
    for p in (0, 1):
        cold = [r for r in rows if r['mode'] == 'cold' and r['implementation'] == p]
        assert len(cold) == 24
        assert all(r['stage_summaries']['weight_conversion']['cv_percent'] >= 3 for r in cold)
    assert next(r for r in result if r['mode'] == 'cold' and r['policy'] == 1)['paired_speedup'] > 1
    conversion = next(r for r in result if r['mode'] == 'conversion_only' and r['policy'] == 1)
    assert conversion['paired_speedup_ci95'][0] < 1 < conversion['paired_speedup_ci95'][1]


def test_actual_same_entry_native_work_and_hot_local_load_difference():
    for kind, local in (('o3', (4, 0)), ('o78', (0, 2))):
        receipt = json.loads((E / f'reports/o378_roof_v89_{kind}_codegen/codegen.json').read_text())
        assert receipt['control_comparison']['passed']
        assert not receipt['changed_semantics'] and not receipt['production_default_changed']
        control = f'adangel_roof_{kind}_eight_chain_candidate'
        candidate = f'adangel_roof_{kind}_grouped_cta_candidate'
        for name, ldl in zip((control, candidate), local):
            entry, live = receipt['entries'][name], receipt['liveness'][name]
            assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma']
            assert live['allocated_gpr'] == 168
            loop = next(x for x in live['loops'] if x['kind'] == 'integer')['opcode_counts']
            assert loop['IMMA.16864.U4.S4'] == loop['IMMA.16864.S4.S4'] == 32
            assert loop['LDSM.16.M88.4'] == 16 and loop.get('LDL', 0) == ldl


def test_limited_sanitizers_pass_with_mapping_tail_checks():
    for tool in ('memcheck', 'synccheck', 'racecheck'):
        result = json.loads((E / f'runs/o378_roof_v89_o3_{tool}/validation.json').read_text())
        log = (E / f'reports/o378_roof_v89_o3_{tool}.log').read_text()
        assert result['passed'] and len(result['checks']) == 96 and len(result['rejected']) == 8
        assert [x['shape'] for x in result['grouped_coordinates_checks']] == [[512, 1024, 4096], [576, 384, 4096]]
        assert all(x['bitwise_v79'] and x['finite_fp32'] for x in result['grouped_coordinates_checks'])
        assert '0 errors, 0 warnings' in log if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors' in log
