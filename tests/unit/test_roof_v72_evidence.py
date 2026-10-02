"""Recompute v72 measurements and audit evidence without requiring CUDA."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
EVIDENCE = ROOT / 'docs/evidence/a100_o378_roof_v72'
BUILD = EVIDENCE / 'reports/o378_roof_v72_codegen'
RUN = EVIDENCE / 'runs/o378_roof_v72_screen'


def test_raw_events_summary_pairing_and_numerics():
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    rows = [json.loads(line) for line in (RUN / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 48
    for row in rows:
        assert row['bitwise_equal_v67'] and row['MSE_regression_passed'] and row['metadata_exact']
        assert row['mse_vs_v67'] == row['max_abs_vs_v67'] == 0
        assert row['raw_ms']['gemm'] == row['raw_ms']['total']
        assert len(row['raw_ms']['gemm']) == 200
        for stage, values in row['raw_ms'].items():
            assert stats(values) == pytest.approx(row['stage_summaries'][stage])
    result = summarize(rows, ('compute_only',))
    recorded = json.loads((RUN / 'summary.json').read_text())['records']
    assert result == recorded
    assert [r['selected_cv_failed_records'] for r in recorded] == [12, 9, 12, 11]
    assert all(r['paired_speedup_ci95'][0] < 1 < r['paired_speedup_ci95'][1] for r in recorded if r['candidate'] == 1)


def test_resources_dependencies_and_binary_identity():
    from analyze_o78_coefficient_codegen import analyze, SYMBOLS
    report = json.loads((BUILD / 'codegen.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(BUILD / 'o78_coefficient.cubin') == report['cubin_sha256']
    for source, sha in report['sources'].items():
        assert digest(ROOT / source) == sha
    assert report['control_opcode_counts_match_v67'] and report['control_instruction_count_match_v67']
    result = analyze((BUILD / 'o78_coefficient.sass').read_text())
    assert result == json.loads((BUILD / 'loop_analysis.json').read_text())
    loops = {r['symbol']: r for r in result['rows']}
    old, new = (loops[s] for s in SYMBOLS)
    assert (old['instructions'], new['instructions']) == (378, 373)
    assert len(old['mainloop_local_loads']) == 0 and len(new['mainloop_local_loads']) == 4
    assert new['dependency_counts']['independent_coefficient_multiplies'] == 64
    assert new['dependency_counts']['coefficient_based_accumulator_updates'] == 64
    assert old['dependency_counts']['partial_first_multiplies'] == 64
    for entry in report['entries'].values():
        assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma'] and entry['all_copies_bypass_l1']
    env = json.loads((RUN / 'environment.json').read_text())
    assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert not env['production_default_changed']
    assert env['control'] == 'v67_fullK_same_v69_preparation'
    assert env['resources']['0']['registers_per_thread'] == env['resources']['1']['registers_per_thread'] == 168
    assert env['resources']['0']['active_blocks_per_sm'] == env['resources']['1']['active_blocks_per_sm'] == 3


def test_gpu_checks_and_sanitizer_scope():
    for name in ('screen', 'memcheck', 'synccheck'):
        v = json.loads((EVIDENCE / f'runs/o378_roof_v72_{name}/validation.json').read_text())
        assert v['passed'] and v['count'] == 64 and v['edge_count'] == 12
        assert all(r['bitwise_equal_v67'] and r['semantic_tolerance_passed'] and r['finite_fp32'] for r in v['checks'])
        assert all(r['shape'][0] <= 128 and r['shape'][1] <= 256 and r['shape'][2] == 4096 for r in v['checks'])
    for name in ('memcheck', 'synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (BUILD / f'{name}.log').read_text()
