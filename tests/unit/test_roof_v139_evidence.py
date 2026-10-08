"""Frozen O7 joint conversion: replay all samples, actual SASS and exact scopes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
E = ROOT/'docs/evidence/a100_o378_roof_v139'
RUN = E/'runs/o378_v139_full24'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def test_exact_frozen_files_sources_and_retained_dependencies():
    index = read(E/'index.json')
    env = read(RUN/'environment.json')
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z', '--', str(E.relative_to(ROOT))],
                                         cwd=ROOT).decode().split('\0'))
    assert index['artifact_count'] == len(index['files'])
    for item in index['files']:
        path = E/item['path']
        assert path.relative_to(ROOT).as_posix() in tracked
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    for name, digest in index['retained_evidence_dependencies'].items():
        assert sha(ROOT/name) == digest
    assert index['artifact_bytes'] == sum(r['bytes'] for r in index['files'])
    assert sha(E/'analysis.json') == index['analysis_sha256']
    assert env['git_commit'] == index['runtime_source_commit']
    assert env['extension_sha256'] == index['formal_extension_sha256']
    assert not index['GEMM_modified'] and not index['production_default_changed']
    source = subprocess.check_output(['git', 'show', env['git_commit']+':scripts/benchmark_o7_conversion_combo.py'], cwd=ROOT)
    assert hashlib.sha256(source).hexdigest() == env['codegen']['runtime_source_sha256']
    assert not list(E.rglob('*.so')) and not list(E.rglob('*.cubin'))


def test_complete_Event_replay_sources_MSE_guards_and_native_routes():
    from freeze_o7_conversion_combo_evidence import validate_frozen
    from benchmark_o7_conversion_combo import timing_contract
    result = validate_frozen(E)
    assert result == read(E/'analysis.json')
    assert result['records'] == 576 and result['original_event_values'] == 345600
    assert result['samples'] == 24 and result['all_outputs_bitwise_equal']
    assert result['all_real_integer_path'] and not result['GEMM_speedup_claim']
    assert result['output_mse_vs_O5'] == pytest.approx(dict(
        median=.005536172273439442, mean=.005053635851002639))
    assert (result['validation_cases'], result['guard_edges'], result['decoder_words']) == (32, 9, 131072)
    for row in map(json.loads, (RUN/'results.jsonl').read_text().splitlines()):
        for key, value in timing_contract(row['mode'], 100).items():
            assert row[key] == value
        assert row['paired_fp16'] == 'o5'
        assert row['guard']['integer_ctas'] == row['guard']['ctas'] == 2048
        assert row['guard']['fallback_ctas'] == row['guard']['invalid_ctas'] == 0
    assert not result['conversion_audit']['frozen_v126_candidate_gate_passed']
    assert not result['conversion_audit']['failed_v126_packed_MX8_candidate_executed']
    assert result['conclusion']['no_new_GEMM_optimization']
