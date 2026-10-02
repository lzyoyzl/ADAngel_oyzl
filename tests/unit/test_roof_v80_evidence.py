"""Recompute best-kernel NCU and static chain evidence, not an Event speedup."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
E = ROOT / 'docs/evidence/a100_o378_roof_v80_ncu/reports'
NCU = E / 'o378_roof_v80_ncu'


@pytest.mark.parametrize('variant,dynamic', [('o3', 86802432), ('o7', 105521152), ('o8', 105521152)])
def test_exact_capture_identity_work_and_capacity(variant, dynamic):
    from run_eight_chain_ncu import analyze_capture
    saved = json.loads((NCU / 'analysis.json').read_text())
    for path, digest in saved['input_sha256'].items():
        assert hashlib.sha256((NCU / path).read_bytes()).hexdigest() == digest
    receipt = json.loads((NCU / variant / 'receipt.json').read_text())
    row = analyze_capture((NCU / f'{variant}_raw.csv').read_text(encoding='utf-8-sig'),
                          (NCU / f'{variant}_source_sass.csv').read_text(encoding='utf-8-sig'), receipt)
    assert row == next(r for r in saved['rows'] if r['variant'] == variant)
    assert not saved['new_performance_result'] and not saved['production_default_changed']
    assert row['static_fingerprint_verified'] and row['dynamic_instructions'] == dynamic
    assert row['opcodes']['IMMA'] == 16777216 and row['opcodes']['LDSM'] == 4194304
    assert row['opcodes']['I2F'] == 524288 and row['opcodes'].get('FFMA', 0) == 0
    assert row['optimistic_fixed_work_lower_bound_ms'] == pytest.approx(.22034693984764905)
    assert row['registers_per_thread'] == 168 and row['max_ctas_per_sm_from_launch_limits'] == 3
    assert row['mse_vs_previous_fullk'] == 0
    assert receipt['filtered_launch_skip'] == 50 and receipt['filtered_launch_count'] == 1
    assert receipt['shape'] == [4096, 4096, 4096] and receipt['sample_id'] == 'layer_00_q_proj'
    assert receipt['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    if variant != 'o3':
        work = row['source_memory_work_by_opcode']
        excessive = 'L1 Wavefronts Shared Excessive'
        assert row['source_memory_work'][excessive] == work['LDGSTS'][excessive] == 6422528
        assert work['LDS'][excessive] == work['LDSM'][excessive] == 0
        assert row['opcodes'].get('LDL', 0) == row['opcodes'].get('STL', 0) == 0


@pytest.mark.parametrize('variant,version,stem,symbol', [
    ('o3', 79, 'o3_eight_chain', 'adangel_roof_o3_eight_chain_candidate'),
    ('o78', 78, 'o78_eight_chain', 'adangel_roof_o78_eight_chain_candidate')])
def test_static_chains_recomputed_from_actual_cubin_disassembly(variant, version, stem, symbol):
    from inspect_eight_chain_schedule import trace
    from inspect_o78_register_liveness import analyze
    build = ROOT / f'docs/evidence/a100_o378_roof_v{version}/reports/o378_roof_v{version}_codegen'
    saved = json.loads((E / f'o378_roof_v80_schedule/{variant}/analysis.json').read_text())
    for path, digest in saved['input_sha256'].items():
        assert hashlib.sha256((build / path).read_bytes()).hexdigest() == digest
    assert saved['script_sha256'] == hashlib.sha256((ROOT / 'scripts/inspect_eight_chain_schedule.py').read_bytes()).hexdigest()
    live = analyze((build / 'liveness.txt').read_text(), symbol)
    got = json.loads(json.dumps(trace((build / f'{stem}.sass').read_text(), symbol, live)))
    assert all(got[key] == saved[key] for key in got)
    assert got['total_mma'] == 64 and got['chains_per_group'] == 16
    assert got['peak_started_not_finished_chains'] == 8
    assert 'not concurrent hardware execution' in got['scope']


@pytest.mark.parametrize('mutation', ['identity', 'fallback', 'fingerprint'])
def test_profile_acceptance_rejects_changed_identity_or_work(mutation):
    from run_eight_chain_ncu import analyze_capture
    receipt = copy.deepcopy(json.loads((NCU / 'o7/receipt.json').read_text()))
    if mutation == 'identity':
        receipt['expected_kernel'] += '_probe'
    elif mutation == 'fallback':
        receipt['guard']['fallback_ctas'] = 1
    else:
        receipt['gemm_codegen']['entries'][receipt['expected_kernel']]['opcode_counts']['IMMA'] += 1
    with pytest.raises(ValueError):
        analyze_capture((NCU / 'o7_raw.csv').read_text(), (NCU / 'o7_source_sass.csv').read_text(), receipt)
