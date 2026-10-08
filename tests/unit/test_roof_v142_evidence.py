"""Recompute the complete v99 + best-conversion joint experiment on CPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
E = ROOT/'docs/evidence/a100_o378_roof_v142'


def test_frozen_artifacts_and_runtime_source():
    index = json.loads((E/'index.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    tracked = set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],
                                         cwd=ROOT).decode().split('\0'))
    assert len(index['artifacts']) == index['artifact_count']
    for name, digest in index['artifacts'].items():
        assert sha(E/name) == digest
        assert (E/name).relative_to(ROOT).as_posix() in tracked
    for name, digest in index['evidence_dependencies'].items():
        assert sha(ROOT/name) == digest
    assert sha(E/'analysis.json') == index['analysis_sha256']
    for variant in ('o7','o8'):
        env = json.loads((E/f'runs/o378_v142_{variant}_full24/environment.json').read_text())
        source = subprocess.check_output(['git','show',env['git_commit']+
                    ':scripts/benchmark_o78_best_combo.py'],cwd=ROOT)
        assert hashlib.sha256(source).hexdigest() == env['codegen']['runtime_source_sha256']
    assert not list(E.rglob('*.so')) and not list(E.rglob('*.cubin'))


def test_complete_joint_results_no_assumed_speedup():
    from freeze_o78_best_combo_evidence import verify
    result = verify(E)
    assert result == json.loads((E/'analysis.json').read_text())
    assert not result['production_default_changed'] and not result['new_CUDA_compilation']
    expected = {'o7': ('o5', .005536172273439442, .005053635851002639),
                'o8': ('o6', .004411084910985704, .00438137929907354)}
    for variant, row in result['runs'].items():
        assert row['samples'] == 24 and row['records'] == 576
        assert row['original_event_values'] == 345600
        assert row['all_outputs_bitwise_equal'] and row['no_filtering']
        assert row['output_mse']['reference'] == expected[variant][0]
        assert row['output_mse']['median'] == pytest.approx(expected[variant][1], rel=1e-12)
        assert row['output_mse']['mean'] == pytest.approx(expected[variant][2], rel=1e-12)
        assert row['limited_memcheck']['errors'] == 0
        # A valid experiment may have zero/negative gain or CV failures.
        # Never encode a desired performance outcome as an acceptance test.
        assert len(row['stages']) == 12
