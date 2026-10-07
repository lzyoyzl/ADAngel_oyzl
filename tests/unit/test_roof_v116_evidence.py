"""Replay v116's data gate; no compilation, GPU launch or performance measurement."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from inspect_o3_atom_identity import (
    AOBS, OBS, observe, observe_activation, summarize, summarize_activation,
)

BASE = ROOT / 'docs/evidence/a100_o378_roof_v116'
O3 = BASE / 'reports/o378_roof_v116_atom_identity'
ALL3 = BASE / 'reports/o378_roof_v116_atom_identity_all3'
EXTENSION = '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_raw_artifacts_and_original_committed_sources():
    index = read(BASE / 'index.json')
    assert index['artifact_count'] == len(index['artifacts']) == 11
    for name, digest in index['artifacts'].items():
        assert sha(BASE / name) == digest, name
    for directory, key in ((O3, 'o3_source_commit'), (ALL3, 'all3_source_commit')):
        receipt = read(directory / 'environment.json')
        assert receipt['source_commit'] == index[key]
        for name, digest in receipt['source_hashes'].items():
            original = subprocess.check_output(
                ['git', 'show', index[key] + ':' + name], cwd=ROOT,
            )
            assert hashlib.sha256(original).hexdigest() == digest, name
        assert 'V12.8.93' in receipt['nvcc']
        assert receipt['cutlass_commit'] == 'db1c288993354c88e551c40c19a8fb93a774a241'
        assert receipt['formal_extension_sha256_before'] == receipt['formal_extension_sha256_after'] == EXTENSION
        assert receipt['snapshots_environment_sha256'] == sha(ROOT / OBS / 'environment.json')
        assert receipt['snapshots_results_sha256'] == sha(ROOT / OBS / 'results.jsonl')
    receipt = read(ALL3 / 'environment.json')
    assert receipt['activation_snapshots_environment_sha256'] == sha(ROOT / AOBS / 'environment.json')
    assert receipt['activation_snapshots_results_sha256'] == sha(ROOT / AOBS / 'results.jsonl')


def test_all24_O3_and_48_A_only_bounds_rederived_not_pass_label():
    original_O3 = observe()
    original_A = observe_activation()
    assert rows(O3 / 'results.jsonl') == original_O3
    assert rows(ALL3 / 'results.jsonl') == original_O3 + original_A
    assert len(original_O3) == 24 and len(original_A) == 48
    old = read(O3 / 'summary.json')
    assert 'not extended' in old.pop('O7_O8')
    assert old == summarize(original_O3)
    expected = summarize(original_O3)
    expected['O7_O8_necessary_condition_bounds'] = summarize_activation(original_A)
    assert read(ALL3 / 'summary.json') == expected
    assert expected['optimistic_loop_instruction_work_reduction_percent'] == pytest.approx(2.790560097394221)
    bounds = expected['O7_O8_necessary_condition_bounds']
    assert bounds[0]['optimistic_loop_instruction_work_reduction_percent_upper_bound'] == pytest.approx(0.5623096170583116)
    assert bounds[1]['optimistic_loop_instruction_work_reduction_percent_upper_bound'] == 0
    assert not expected['opportunity_gate'] and not any(r['opportunity_gate'] for r in bounds)


def test_exact_host_mapping_and_legacy_panel_label_preserved():
    initial = read(O3 / 'coordinates.json')
    current = read(ALL3 / 'coordinates.json')
    assert initial['passed'] and current['passed']
    assert not initial['gpu_execution'] and not current['gpu_execution']
    assert initial['outputs'] == current['outputs'] == 8192
    assert current['columns_per_native_atom'] == 8 and current['rows_per_native_atom'] == 16
    assert initial['atoms_per_thread'] == current['N8_panels_per_thread'] == 8
    assert current['native_M16_N8_atoms_per_thread'] == 16
    assert current['per_thread_outputs_per_native_M16_N8_atom'] == 4
    assert current['outputs_per_thread'] == 16 * 4 == 64
    assert current['O3_O78_TiledMMA_types_equal']


def test_no_new_kernel_performance_MSE_or_safety_claim():
    index = read(BASE / 'index.json')
    flags = ('candidate_GEMM_implemented', 'candidate_GPU_launched',
             'new_performance_result', 'new_MSE_result', 'production_default_changed')
    assert not any(index[key] for key in flags)
    for directory in (O3, ALL3):
        assert not any(read(directory / 'summary.json')[key] for key in flags)
        receipt = read(directory / 'environment.json')
        assert receipt['no_GPU_kernel_launched'] and receipt['no_new_quantization']
    for name in index['binary_files_not_committed']:
        assert not (BASE / name).exists()
    assert '7 passed' in (BASE / 'reports/o378_v116_atom_identity_unit.log').read_text()
