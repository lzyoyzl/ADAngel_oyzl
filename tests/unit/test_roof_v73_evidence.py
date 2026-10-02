"""Recompute v73 paired conversion/E2E results, code identity and GPU checks."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
EVIDENCE = ROOT / 'docs/evidence/a100_o378_roof_v73'
BUILD = EVIDENCE / 'reports/o378_roof_v73_codegen'


@pytest.mark.parametrize('run,samples', [('screen', 4), ('trace24', 24)])
def test_raw_pairing_timing_and_mse(run, samples):
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    from benchmark_o78_row_fused import base, timing_contract
    path = EVIDENCE / f'runs/o378_roof_v73_{run}'
    rows = [json.loads(line) for line in (path / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == samples * 2 * 4 * 2
    for row in rows:
        assert row['bitwise_equal_v67'] and row['MSE_regression_passed'] and row['metadata_exact']
        assert row['mse_vs_v67'] == row['max_abs_vs_v67'] == 0
        assert row['finite_fp32']
        for stage, values in row['raw_ms'].items():
            assert len(values) == 200
            assert stats(values) == pytest.approx(row['stage_summaries'][stage])
        if row['mode'] == 'conversion_only':
            w, a = (row['raw_ms'][s] for s in ('weight_conversion', 'activation_conversion'))
            assert row['raw_ms']['total'] == pytest.approx([x + y for x, y in zip(w, a)], rel=1e-6)
        if row['mode'] == 'compute_only':
            assert row['raw_ms']['gemm'] == row['raw_ms']['total']
        # Initial screen inherited the v69 preparation label; raw evidence is
        # preserved. The follow-up changes labels only, not timing/CUDA code.
        expected = timing_contract(row['mode'], 100, row['candidate']) if run == 'trace24' else base.timing_contract(row['mode'], 100)
        for key, value in expected.items():
            assert row[key] == value
    result = summarize(rows, base.MODES)
    assert result == json.loads((path / 'summary.json').read_text())['records']


def test_static_audit_and_unchanged_gemm_identity():
    from analyze_o78_row_fused_codegen import analyze
    result = analyze(BUILD, ROOT / 'docs/evidence/a100_o378_roof_v69/reports/o378_roof_v69_codegen')
    assert result == json.loads((BUILD / 'static_audit.json').read_text())
    assert len(result['original_control_encoded_identical']) == 15
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    build = json.loads((BUILD / 'build.json').read_text())
    for source, sha in build['sources'].items():
        assert digest(ROOT / source) == sha
    for run in ('screen', 'trace24'):
        env = json.loads((EVIDENCE / f'runs/o378_roof_v73_{run}/environment.json').read_text())
        assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
        assert not env['production_default_changed']
        assert env['gpu_preparation_build'] == build
        assert env['resources']['0'] == env['resources']['1'] == env['resources']['2']
        assert env['resources']['0']['kernel_symbol'] == 'adangel_roof_o78_fullk_candidate'
        original = json.loads((ROOT / 'docs/evidence/a100_o378_roof_v67/reports/o378_roof_v67_codegen/codegen.json').read_text())
        assert env['codegen'] == original
        for entry in original['entries'].values():
            assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma']


def test_gpu_validation_and_sanitizer_scope():
    for run in ('screen', 'trace24', 'memcheck', 'synccheck', 'racecheck'):
        data = json.loads((EVIDENCE / f'runs/o378_roof_v73_{run}/validation.json').read_text())
        assert data['passed'] and data['count'] == 64 and data['edge_count'] == 12
        assert all(r['bitwise_equal_v67'] and r['finite_fp32'] and r['semantic_tolerance_passed'] for r in data['checks'])
        assert all(r['shape'][0] <= 128 and r['shape'][1] <= 256 and r['shape'][2] == 4096 for r in data['checks'])
    for check in ('memcheck', 'synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (BUILD / f'{check}.log').read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (BUILD / 'racecheck.log').read_text()
