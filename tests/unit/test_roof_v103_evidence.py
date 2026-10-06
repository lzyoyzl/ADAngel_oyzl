import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_sparse_q4_feasibility import summarize, scalar_residual_model, REFERENCE_PROVENANCE

BASE = ROOT / 'docs/evidence/a100_o378_roof_v103/reports'
DATA = BASE / 'o378_roof_v103_sparse_feasibility'


def rows():
    return [json.loads(line) for line in (DATA / 'results.jsonl').read_text().splitlines()]


def test_complete_summary_reproduces_without_claiming_new_performance():
    result = summarize(rows())
    assert result == json.loads((DATA / 'summary.json').read_text())
    assert not any(result[k] for k in ('production_default_changed', 'pruning_performed',
        'new_MSE_measured', 'new_GEMM_measured', 'residual_format_or_kernel_implemented'))
    assert [r['scalar_model_median_ms'] for r in result['variants']] == pytest.approx(
        [.8665671657473075, .9263141581297609, 1.09394536380352])


def test_exact_integer_residual_histograms_reproduce_counts_and_models():
    for row in rows():
        s = row['statistics']
        assert s['shape'] == [4096, 4096] and s['weights'] == 4096**2
        assert s['decomposition_exact'] and s['ordered_metadata_valid']
        h = s['minimum_residual_nonzeros_per8_histogram']
        assert sum(h) == s['weights'] // 8
        nz = sum(i * value for i, value in enumerate(h))
        assert nz == s['minimum_residual_nonzeros']
        assert nz / s['weights'] == s['minimum_residual_fraction']
        assert sum(h[1:]) == s['chunks_requiring_residual']
        assert sum(s['active_pair_histogram']) == s['weights'] // 8
        assert row['model'] == scalar_residual_model(s['weights'], nz)


def test_existing_source_identity_and_code_receipts_are_exact():
    env = json.loads((DATA / 'environment.json').read_text())
    for name, digest in env['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    provenance_path = ROOT / REFERENCE_PROVENANCE
    assert hashlib.sha256(provenance_path.read_bytes()).hexdigest() == env['v99_provenance_sha256']
    old = {(r['sample_id'], r['variant']): r for r in
        map(json.loads, provenance_path.read_text().splitlines())}
    for r in rows():
        if r['variant'] == 'o3':
            assert r['provenance']['current_Q4_exact']
        else:
            assert r['provenance']['v99_source_exact']
            reference = old[r['sample_id'], r['variant']]
            assert reference['weight'] == r['provenance']['weight_source']
            assert reference['raw_sha256'] == r['raw_sha256']
    assert env['no_production_GEMM_or_candidate_kernel_launched']
    assert env['no_performance_timing'] and not env['quantization_semantics_changed']
    assert env['native_extension_sha256'] == (
        '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')


def test_full24_log_and_reported_unit_test_scope():
    assert (BASE / 'o378_roof_v103_sparse_feasibility.log').read_text().count(
        'exact sources and lossless pair statistics passed') == 24
    assert '10 passed' in (BASE / 'o378_roof_v103_unit_tests.log').read_text()
