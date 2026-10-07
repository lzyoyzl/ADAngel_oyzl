"""Replay current-best O8 warm NCU evidence; not Event performance acceptance."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_o8_best_warm_ncu import analyze_capture, capture_command, verify_passes
from profile_o8_best_warm import SAMPLE, SYMBOL

BASE = ROOT / 'docs/evidence/a100_o378_roof_v114'
DATA = BASE / 'reports/o378_roof_v114_o8_warm_ncu'
COMMIT = 'e9c6556595875993bd1251e6f96084c2b84e7590'
EXTENSION = '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def passes():
    return [json.loads(p.read_text()) for p in sorted((DATA / 'pass_receipts').glob('pass_*/receipt.json'))]


def test_frozen_files_and_profile_source_at_original_commit():
    index = json.loads((BASE / 'index.json').read_text())
    assert index['source_commit'] == COMMIT
    assert len(index['artifacts']) == 58
    for name, sha in index['artifacts'].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == sha, name
    original = subprocess.check_output(['git', 'show', COMMIT + ':scripts/profile_o8_best_warm.py'], cwd=ROOT)
    assert hashlib.sha256(original).hexdigest() == passes()[0]['source_sha256']
    assert not index['new_GEMM_candidate'] and not index['production_default_changed']


def test_all50_completed_passes_exact_source_output_binary_and_resources():
    rows = passes()
    assert len(rows) == 50 and len({r['pid'] for r in rows}) == 50
    receipt = verify_passes(rows)
    refs = [json.loads(line) for line in (ROOT / 'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl').read_text().splitlines()]
    ref = next(r for r in refs if (r['sample_id'], r['variant']) == (SAMPLE, 'o8'))
    assert receipt['source_identity'] == ref
    v78 = json.loads((ROOT / 'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/codegen.json').read_text())
    assert receipt['gemm_binary_sha256'] == v78['cubin_sha256']
    assert receipt['resources']['registers_per_thread'] == 168
    assert receipt['resources']['active_blocks_per_sm'] == 3
    assert receipt['resources']['local_size_bytes'] == 0
    assert {k: receipt['guard'][k] for k in ('integer_ctas', 'fallback_ctas', 'invalid_ctas')} == dict(integer_ctas=2048, fallback_ctas=0, invalid_ctas=0)
    assert receipt['guard']['metadata_exact'] and receipt['guard']['group_squares_exact']
    assert receipt['guard']['additional_fp32_epilogue_range_guard']
    for r in rows:
        assert r['git_commit'] == COMMIT
        assert r['extension_sha256_before'] == r['extension_sha256_after'] == EXTENSION
        assert r['mse_vs_previous_fullK'] == 0
        assert r['mse_vs_paired_fp16'] == pytest.approx(0.00018277407059992296, rel=1e-12)


def test_capture_command_and_all_logged_replays_without_duplicate_capture():
    commands = json.loads((DATA / 'commands.json').read_text())
    assert len(commands) == 4
    cmd = commands[1]['command']
    assert cmd == capture_command(cmd[0], cmd[cmd.index('-o')+2], cmd[cmd.index('-o')+1],
                                 cmd[cmd.index('--validation')+1], cmd[cmd.index('--output')+1])
    log = (DATA / 'capture.log').read_text()
    assert [int(n) for n in re.findall(r'Application replay pass (\d+)', log)] == list(range(1, 51))
    assert log.count('warm O8 profile pass identity/output checks passed') == 50
    assert '==ERROR==' not in log and log.count('==PROF== Report:') == 1
    assert 'CURRENT BEST O8 WARM NCU IDENTITY/WORK CHECKS PASSED' in (BASE / 'reports/o378_roof_v114_o8_warm_ncu.log').read_text()


def test_raw_source_analysis_replayed_not_just_saved_PASS_or_old_profile():
    result = analyze_capture((DATA / 'o8_warm_raw.csv').read_text(encoding='utf-8-sig'),
                             (DATA / 'o8_warm_source_sass.csv').read_text(encoding='utf-8-sig'), passes())
    saved = json.loads((DATA / 'analysis.json').read_text())
    assert all(saved[k] == v for k, v in result.items())
    assert result['dynamic_instructions'] == 105521152
    assert result['opcodes']['IMMA'] == 16777216
    assert result['opcodes']['I2F'] == 524288
    assert result['opcodes']['FFMA'] == 0
    assert result['source_memory_work_by_opcode']['LDSM']['L1 Wavefronts Shared Excessive'] == 0
    assert result['source_memory_work_by_opcode']['LDS']['L1 Wavefronts Shared Excessive'] == 0
    assert result['source_memory_work']['L2 Theoretical Sectors Local'] == 0
    assert result['binding_modeled_resources'] == ['mma']
    assert result['optimistic_fixed_work_lower_bound_ms'] == pytest.approx(0.22034693984764905)
    assert result['pc_sampling']['not_issued_samples'] == 10557
    assert sum(result['pc_sampling']['reason_samples'].values()) == 10557
    assert 'not runtime fractions' in result['pc_sampling']['interpretation']
    assert not saved['new_performance_result'] and not saved['production_default_changed']
