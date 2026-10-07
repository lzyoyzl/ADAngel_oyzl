"""Replay frozen O3 NCU provenance/PC attribution; not a new GPU test."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from freeze_o3_warm_evidence import CAPTURE, replay

DIRECTORY = ROOT/'docs/evidence/a100_o378_roof_v136'


def test_all_raw_artifact_hashes_and_indexed_files_are_tracked():
    index = json.loads((DIRECTORY/'index.json').read_text())
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    assert index['artifact_count'] == len(index['files']) == 66
    assert index['artifact_bytes'] == sum(r['bytes'] for r in index['files'])
    for row in index['files']:
        path = DIRECTORY/row['path']
        assert path.stat().st_size == row['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        assert path.relative_to(ROOT).as_posix() in tracked
        assert path.suffix not in ('.so', '.cubin', '.ncu-rep')
    assert hashlib.sha256((DIRECTORY/'phase_analysis.json').read_bytes()).hexdigest() == index['phase_analysis_sha256']


def test_capture_identity_math_and_phase_analysis_replay_exactly():
    result, phase, receipt = replay(DIRECTORY)
    assert phase == json.loads((DIRECTORY/'phase_analysis.json').read_text())
    assert result['ncu_duration_ms'] == 0.37616000000000005
    assert result['mse_vs_previous_best'] == 0
    assert result['mse_vs_paired_fp16'] == 0.0003752320504872409
    assert result['completed_application_pass_receipts'] == 50
    assert phase['not_issued_samples'] == 10953
    mma = [v for k, v in phase['consumer_categories'].items() if k.startswith('mma_')]
    assert sum(v['reason_samples']['wait'] for v in mma) == 2280
    assert sum(v['reason_samples']['math'] for v in mma) == 2605
    assert phase['consumer_categories']['weighted_integer_accumulate']['reason_samples']['short_sb'] == 208
    assert receipt['gemm_binary_sha256'] == '89918880f8cad38f8234993ff07064d5c8bb6f2c5a4996c3d443d637836856bd'
    assert not result['production_default_changed'] and not result['new_performance_result']
    captured = json.loads((DIRECTORY/CAPTURE/'analysis.json').read_text())
    assert hashlib.sha256((ROOT/'scripts/run_o3_best_warm_ncu.py').read_bytes()).hexdigest() == captured['script_sha256']
