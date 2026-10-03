"""Recompute the exact-identity LDSM/LDS diagnostic; never promote NCU timing."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
E = ROOT / 'docs/evidence/a100_o378_roof_v86/reports/o378_roof_v86_ncu'


@pytest.mark.parametrize('policy,dynamic', [(0, 105521152), (1, 104398848)])
def test_saved_capture_recomputes_with_exact_identity_and_work(policy, dynamic):
    from run_register_layout_ncu import analyze_capture

    saved = json.loads((E / 'analysis.json').read_text())
    assert not saved['new_performance_result'] and not saved['production_default_changed']
    for path, digest in saved['input_sha256'].items():
        assert hashlib.sha256((E / path).read_bytes()).hexdigest() == digest
    receipt = json.loads((E / f'o7_p{policy}/receipt.json').read_text())
    got = analyze_capture((E / f'o7_p{policy}_raw.csv').read_text(encoding='utf-8-sig'),
                          (E / f'o7_p{policy}_source_sass.csv').read_text(encoding='utf-8-sig'), receipt)
    assert got == next(r for r in saved['rows'] if r['policy'] == policy)
    assert got['dynamic_instructions'] == dynamic and got['static_fingerprint_verified']
    assert got['opcodes']['IMMA'] == 16777216
    assert got['opcodes']['I2F'] == 524288 and got['opcodes']['FMUL'] == 1048576
    assert got['opcodes'].get('LDL', 0) == got['opcodes'].get('STL', 0) == 0
    assert got['mse_vs_previous_fullk'] == 0
    assert got['mse_vs_paired_fp16'] == pytest.approx(0.00017169781117249843)
    assert got['optimistic_fixed_work_lower_bound_ms'] == pytest.approx(.22034693984764905)
    assert got['registers_per_thread'] == 168 and got['max_ctas_per_sm_from_launch_limits'] == 3
    assert receipt['sample_id'] == 'layer_00_q_proj' and receipt['shape'] == [4096] * 3
    assert receipt['guard']['integer_ctas'] == 2048 and not receipt['guard']['fallback_ctas']
    assert receipt['filtered_launch_skip'] == 50 and receipt['filtered_launch_count'] == 1
    assert receipt['git_commit'] == '0be91b30b28078631092ac0843d00a877f079f71'
    assert receipt['profiler_script_sha256'] == hashlib.sha256(
        (ROOT / 'scripts/profile_register_layout_kernel.py').read_bytes()).hexdigest()
    assert receipt['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    build = ROOT / 'docs/evidence/a100_o378_roof_v85/reports/o378_roof_v85_codegen_checked'
    assert receipt['gemm_codegen'] == json.loads((build / 'codegen.json').read_text())
    assert receipt['source_provenance']['weight']['group_size'] == 128
    assert receipt['source_provenance']['activation']['group_size'] == 128
    log = (E / f'o7_p{policy}.log').read_text()
    assert f'{50 if policy == 0 else 49} passes' in log
    assert 'REGISTER LAYOUT PROFILE NUMERICAL CHECKS PASSED' in log


def test_renaming_shared_load_did_not_reduce_payload_service_or_occupancy():
    a, b = json.loads((E / 'analysis.json').read_text())['rows']
    assert [a['policy'], b['policy']] == [0, 1]
    assert a['opcodes']['LDSM'] == b['opcodes']['LDS'] - a['opcodes']['LDS'] == 4194304
    assert b['opcodes']['LDSM'] == 0
    for row in (a, b):
        memory = row['source_memory_work_by_opcode']
        assert sum(memory.get(op, {}).get('L1 Wavefronts Shared', 0)
                   for op in ('LDS', 'LDSM')) == 22020096
        assert sum(memory.get(op, {}).get('L1 Wavefronts Shared Excessive', 0)
                   for op in ('LDS', 'LDSM')) == 0
        assert row['pc_sampling']['reason_share_percent']['wait'] > 35
        assert 'not causal producers' in row['pc_sampling']['interpretation']
    assert b['dynamic_instructions'] < a['dynamic_instructions']
    assert b['eligible_warps'] < a['eligible_warps']
    assert b['issue_active_percent'] < a['issue_active_percent']
    assert b['pc_sampling']['reason_share_percent']['short_sb'] > a['pc_sampling']['reason_share_percent']['short_sb']
    # The duration is diagnostic only; Event/MSE acceptance remains in v85.
    assert b['ncu_duration_ms'] > a['ncu_duration_ms']


@pytest.mark.parametrize('policy', [0, 1])
def test_corrupt_fingerprint_or_integer_work_is_rejected(policy):
    from run_register_layout_ncu import analyze_capture

    receipt = json.loads((E / f'o7_p{policy}/receipt.json').read_text())
    raw = (E / f'o7_p{policy}_raw.csv').read_text(encoding='utf-8-sig')
    src = (E / f'o7_p{policy}_source_sass.csv').read_text(encoding='utf-8-sig')
    for mutation in ('fallback', 'fingerprint'):
        bad = copy.deepcopy(receipt)
        if mutation == 'fallback':
            bad['guard']['fallback_ctas'] = 1
        else:
            bad['gemm_codegen']['entries'][bad['expected_kernel']]['opcode_counts']['IMMA'] += 1
        with pytest.raises(ValueError):
            analyze_capture(raw, src, bad)
