"""Replay frozen v113 data evidence; no GPU timing or numerical acceptance."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from inspect_activation_high_sparsity import POLICIES, CORRECTION_BUDGET_MS, model, summarize
from analyze_sparse_q4_feasibility import REFERENCE_PROVENANCE

BASE = ROOT / 'docs/evidence/a100_o378_roof_v113'
DATA = BASE / 'reports/o378_roof_v113_activation_high_sparsity'


def rows():
    return [json.loads(line) for line in (DATA / 'results.jsonl').read_text().splitlines()]


def test_frozen_artifact_and_original_source_hashes():
    index = json.loads((BASE / 'index.json').read_text())
    env = json.loads((DATA / 'environment.json').read_text())
    assert env['git_commit'] == index['source_commit'] == (
        'fb82df383007856d3b2bf6c558923d4bbf672061')
    for name, digest in index['artifacts'].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest, name
    for name, digest in env['source_hashes'].items():
        original = subprocess.check_output(['git', 'show', env['git_commit'] + ':' + name], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == digest, name
    assert env['native_extension_sha256_before'] == env['native_extension_sha256_after'] == (
        '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')
    assert env['budget_predeclared_ms'] == CORRECTION_BUDGET_MS
    assert env['no_candidate_GEMM_or_timing'] and env['no_input_data_saved_or_changed']
    assert env['source_quantizer_reused']


def test_full24_exact_source_identity_and_no_policy_changes_to_source():
    env = json.loads((DATA / 'environment.json').read_text())
    path = ROOT / REFERENCE_PROVENANCE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == env['v99_provenance_sha256']
    refs = {(r['sample_id'], r['variant']): r for r in map(json.loads, path.read_text().splitlines())}
    observations = rows()
    ids = {r['sample_id'] for r in observations}
    assert ids == {f'layer_{layer:02d}_{proj}_proj'
                   for layer, proj in itertools.product((0, 6, 12, 18, 24, 31), ('q', 'k', 'v', 'o'))}
    for row in observations:
        if row['variant'] == 'o3':
            assert row['provenance']['original_A_int8']
        else:
            ref = refs[row['sample_id'], row['variant']]
            assert row['provenance']['v99_source_exact']
            assert row['provenance']['activation_source'] == ref['activation']
            assert row['raw_sha256'] == ref['raw_sha256']
    for sid, variant in itertools.product(ids, ('o7', 'o8')):
        selected = [r for r in observations if (r['sample_id'], r['variant']) == (sid, variant)]
        assert len(selected) == 2
        assert selected[0]['statistics']['source_fixed_sha256'] == selected[1]['statistics']['source_fixed_sha256']
        assert selected[0]['provenance'] == selected[1]['provenance']


def test_residual_histograms_reproduce_models_without_pruning():
    for row in rows():
        s = row['statistics']
        assert s['shape'] == [4096, 4096] and s['elements'] == 4096**2
        for key in ('decomposition_exact', 'ordered_metadata_valid', 'representation_exact',
                    'no_pruning', 'no_K_reordering'):
            assert s[key]
        assert -8 <= s['high_min'] <= s['high_max'] <= 7
        assert s['low_MMA_type'] == ('U4' if row['policy'] == 'ordinary' else 'S4')
        h = s['minimum_residual_nonzeros_per8_histogram']
        assert len(h) == 5 and sum(h) == s['elements'] // 8
        assert len(s['active_pair_histogram']) == 5
        assert sum(s['active_pair_histogram']) == s['elements'] // 8
        residual = sum(i * value for i, value in enumerate(h))
        assert residual == s['minimum_residual_nonzeros']
        assert residual / s['elements'] == s['minimum_residual_fraction']
        assert sum(h[1:]) / (s['elements'] // 8) == s['chunks_requiring_residual_fraction']
        assert row['model'] == model(s['elements'], residual, 4096)
        assert row['model']['mma_can_overlap_so_no_addition_to_mma_floor']
        assert row['model']['gate_is_investment_filter_not_impossibility_proof']


def test_complete_failed_investment_gate_not_recast_as_performance():
    result = summarize(rows())
    assert result == json.loads((DATA / 'summary.json').read_text())
    assert [(r['variant'], r['policy']) for r in result['variants']] == list(POLICIES)
    assert [r['data_gate_samples'] for r in result['variants']] == [0, 0, 0, 0, 4]
    assert not any(r['investment_gate_passed'] for r in result['variants'])
    assert [r['median_scalar_correction_model_ms'] for r in result['variants']] == pytest.approx(
        [.4702516417126346, .8803162595219333, .8002267402153928,
         .4819149986866299, .2588381402679275])
    accepted = [r for r in rows() if r['model']['data_gate_passed']]
    assert {(r['variant'], r['policy']) for r in accepted} == {('o8', 'balanced')}
    assert {r['sample_id'] for r in accepted} == {
        'layer_00_q_proj', 'layer_00_k_proj', 'layer_00_v_proj', 'layer_00_o_proj'}
    assert not any(result[key] for key in ('candidate_GEMM_launched', 'new_GEMM_measured',
        'new_MSE_measured', 'production_default_changed', 'quantization_semantics_changed',
        'sparse_kernel_implemented'))
    assert (BASE / 'reports/o378_roof_v113_activation_high_sparsity.log').read_text().count(
        'exact sources and five high-digit observations passed') == 24
