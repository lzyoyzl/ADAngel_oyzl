"""Recompute the UNIT-scale diagnostic; never promote it to real O7/O8 results."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
EVIDENCE = ROOT / 'docs/evidence/a100_o378_roof_v77'
BUILD = EVIDENCE / 'reports/o378_roof_v77_codegen'
RUN = EVIDENCE / 'runs/o378_roof_v77_unit_scale'


def test_events_pairing_and_unit_reference_not_original_mse():
    from benchmark_a100_o1 import stats
    from benchmark_o78_unit_scale_probe import summarize
    rows = [json.loads(line) for line in (RUN / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 48
    assert {r['sample_id'] for r in rows} == {
        'layer_00_q_proj', 'layer_00_k_proj', 'layer_00_v_proj', 'layer_00_o_proj'}
    for row in rows:
        assert row['diagnostic_only'] and row['finite_fp32']
        assert row['mode'] == 'cached_UNIT_scale_core_NOT_original_compute_only'
        assert len(row['raw_ms']) == 200 and row['unit_reference_mse'] == 0
        assert stats(row['raw_ms']) == pytest.approx(row['summary'])
        assert 'mse_vs_o5' not in row and 'mse_vs_o6' not in row
    summary = json.loads((RUN / 'summary.json').read_text())
    assert summary['diagnostic_only'] and not summary['original_experiment_result']
    assert summarize(rows) == summary['records']
    assert [r['cv_failed'] for r in summary['records']] == [6, 8, 6, 7]
    assert summary['records'][1]['paired_ratio'] == pytest.approx(1.0763146060782713)
    assert summary['records'][3]['paired_ratio'] == pytest.approx(1.0844784280316082)


def test_source_hashes_and_same_entry_two_int4_paths():
    from probe_o78_unit_scale_codegen import unit_header
    from probe_roof_fullk_integer_codegen import static_entries
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((BUILD / 'codegen.json').read_text())
    assert receipt['diagnostic_only'] and not receipt['performance_candidate'] and not receipt['default_changed']
    for path, sha in receipt['sources'].items():
        assert digest(ROOT / path) == sha
    header = BUILD / 'o78_unit_scale_generated.cuh'
    assert digest(header) == receipt['generated_header_sha256']
    assert header.read_text() == unit_header((ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    entries = static_entries((BUILD / 'o78_unit_scale.sass').read_text(),
        r'^adangel_roof_o78_(?:fullk_(?:control|candidate)|unit_scale_diagnostic)$', set(receipt['entries']))
    assert entries == receipt['entries']
    assert receipt['control_opcode_counts_match'] and receipt['control_instructions_match']
    for symbol, entry in entries.items():
        assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma']
        assert entry['all_copies_bypass_l1']
        block = next(b for b in re.split(r'(?=\.visible \.entry )', (BUILD / 'o78_unit_scale.ptx').read_text())
                     if b.startswith('.visible .entry ' + symbol + '('))
        assert all(x in block for x in ('cp.async.cg.shared.global', '.s32.u4.s4.s32', '.s32.s4.s4.s32'))
    if (BUILD / 'o78_unit_scale.cubin').exists():
        assert digest(BUILD / 'o78_unit_scale.cubin') == receipt['cubin_sha256']


def test_register_liveness_and_resource_cost_not_a_strict_time_decomposition():
    from inspect_o78_register_liveness import analyze
    from probe_o78_unit_scale_codegen import CONTROL, SYMBOL
    receipt = json.loads((BUILD / 'codegen.json').read_text())
    text = (BUILD / 'liveness.txt').read_text()
    for symbol in (CONTROL, SYMBOL):
        assert analyze(text, symbol) == receipt['liveness'][symbol]
    loops = [next(x for x in receipt['liveness'][s]['loops'] if x['kind'] == 'integer') for s in (CONTROL, SYMBOL)]
    assert [x['static_instructions'] for x in loops] == [378, 314]
    assert [x['max_live_gpr'] for x in loops] == [160, 156]
    assert [sum(v for k, v in x['opcode_counts'].items() if k.startswith('IMAD')) for x in loops] == [154, 91]
    assert loops[0]['opcode_counts'].get('LDL.64', 0) == 0
    assert loops[1]['opcode_counts']['LDL.64'] == 1
    for loop in loops:
        assert loop['opcode_counts']['IMMA.16864.U4.S4'] == 32
        assert loop['opcode_counts']['IMMA.16864.S4.S4'] == 32
        assert loop['opcode_counts']['LDSM.16.M88.4'] == 16
    env = json.loads((RUN / 'environment.json').read_text())
    assert env['codegen']['diagnostic'] == receipt
    assert env['diagnostic_only'] and env['all_scales_replaced_by_one'] and env['no_filtering']
    assert env['not_an_original_experiment_speedup'] and not env['production_default_changed']
    assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    for policy in ('0', '1'):
        r = env['resources'][policy]
        assert r['registers_per_thread'] == 168 and r['active_blocks_per_sm'] == 3
        assert r['shared_memory_bytes'] == 34304
    assert [env['resources'][p]['local_size_bytes'] for p in ('0', '1')] == [8, 40]


def test_finite_sanitizer_scope_and_reject_nonunit_data():
    for suffix in ('unit_scale', 'memcheck', 'synccheck', 'racecheck'):
        validation = json.loads((EVIDENCE / f'runs/o378_roof_v77_{suffix}/validation.json').read_text())
        assert validation['passed'] and validation['count'] == 8 and validation['nonunit_rejected']
        assert validation['scope'] == 'unit_scales_only_small_MN_K4096_not_original_experiment_validation'
        for check in validation['checks']:
            assert check['exact_fp64_reference'] and check['finite_fp32'] and check['nondefault_stream']
            assert check['shape'][0] <= 128 and check['shape'][1] <= 256 and check['shape'][2] == 4096
    for name in ('memcheck', 'synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (BUILD / f'{name}.log').read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (BUILD / 'racecheck.log').read_text()
