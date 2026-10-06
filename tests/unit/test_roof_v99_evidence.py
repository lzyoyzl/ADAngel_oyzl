"""Replay exact v99 compiler evidence, not an assertion of runtime speedup."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from compare_a100_codegen import compare
from analyze_output_streaming import analyze_run
from inspect_o78_register_liveness import analyze
from probe_output_streaming_codegen import CONFIG, generated_header, prior_memory_evidence, store_audit, worth_runtime
from probe_roof_fullk_integer_codegen import static_entries

EVIDENCE = ROOT/'docs/evidence/a100_o378_roof_v99'


@pytest.mark.parametrize('kind', ('o3','o78'))
def test_hash_bound_compiler_receipt_and_predeclared_gate(kind):
    cfg = CONFIG[kind]
    directory = EVIDENCE/'reports'/f'o378_roof_v99_{kind}_codegen'
    receipt = json.loads((directory/'codegen.json').read_text())
    for name, digest in receipt['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    for name, digest in receipt['artifact_sha256'].items():
        if name.endswith('.cubin'):
            assert digest == receipt['cubin_sha256']
        else:
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    assert receipt['source_commit'] == '90dfbf7496919da1c34d095354eb51e982beb219'
    assert receipt['prior_source_memory'] == prior_memory_evidence(kind)
    assert (directory/(cfg['stem']+'_generated.cuh')).read_text() == generated_header(kind)
    assert not receipt['changed_semantics'] and not receipt['production_default_changed']
    assert receipt['partial_registers'] == 32 and receipt['threads'] == 128
    text = (directory/'liveness.txt').read_text()
    lives = {symbol: analyze(text, symbol) for symbol in (cfg['control'], cfg['symbol'])}
    assert lives == receipt['liveness']
    assert worth_runtime(lives[cfg['control']], lives[cfg['symbol']], receipt['stores'])
    assert receipt['worth_runtime_validation']
    for live in lives.values():
        assert live['allocated_gpr'] == 168
        loop = next(row for row in live['loops'] if row['kind'] == 'integer')
        assert loop['static_instructions'] == (323 if kind == 'o3' else 383)
        assert loop['opcode_counts']['LDSM.16.M88.4'] == 16
        assert not any(op.split('.')[0] in ('LDL','STL') for op in loop['opcode_counts'])


@pytest.mark.parametrize('kind', ('o3','o78'))
def test_exact_control_and_same_entry_streaming_native_int4(kind):
    cfg = CONFIG[kind]
    directory = EVIDENCE/'reports'/f'o378_roof_v99_{kind}_codegen'
    receipt = json.loads((directory/'codegen.json').read_text())
    old_root = 'a100_o378_roof_v89' if kind == 'o3' else 'a100_o378_roof_v78'
    old_sass = ROOT/'docs/evidence'/old_root/cfg['baseline']/(cfg['old_stem']+'.sass')
    sass = (directory/(cfg['stem']+'.sass')).read_text()
    assert compare(old_sass.read_text(), sass, '^'+re.escape(cfg['control'])+'$') == receipt['control_comparison']
    assert receipt['control_comparison']['passed']
    symbols = {cfg['control'], cfg['symbol']}
    assert static_entries(sass, '^(?:'+'|'.join(sorted(symbols))+')$', symbols) == receipt['entries']
    stores = {label: store_audit(sass, cfg['control'] if label == 'control' else cfg['symbol'])
              for label in ('control','candidate')}
    assert stores == receipt['stores']
    assert stores['control']['total'] == stores['candidate']['total'] == 192
    assert stores['control']['streaming'] == 0 and stores['candidate']['streaming'] == 96
    ptx = (directory/(cfg['stem']+'.ptx')).read_text()
    body = next(block for block in re.split(r'(?=\.visible \.entry )', ptx)
                if block.startswith('.visible .entry '+cfg['symbol']+'('))
    assert all(token in body for token in ('st.global.cs', 'cp.async.cg.shared.global',
        '.s32.u4.s4.s32', '.s32.s4.s4.s32', '.maxntid 128, 1, 1'))


@pytest.mark.parametrize('kind,records', (('o3',144), ('o78',288)))
def test_cached_complete24_raw_timing_output_and_mse_replay(kind, records):
    result = analyze_run(EVIDENCE/'runs'/f'o378_roof_v99_{kind}', kind)
    saved = json.loads((EVIDENCE/'reports/o378_v99_paired_analysis.json').read_text())['runs'][kind]
    assert result['records'] == records
    assert result['raw_sha256'] == saved['raw_sha256']
    assert result['summaries'] == saved['summaries']
    assert result['rounds'] == 3 and result['timing_scope'] == 'cached_compute_only'
    assert not result['raw_sequence_filtering']
    for row in result['summaries']:
        assert row['samples'] == 24 and row['records'] == 72
        assert row['bitwise_previous_best'] and row['no_filtering']


@pytest.mark.parametrize('kind', ('o3', 'o78'))
def test_gpu_validation_scope_and_actual_resources(kind):
    directory = EVIDENCE/'runs'/f'o378_roof_v99_{kind}'
    validation = json.loads((directory/'validation.json').read_text())
    assert validation['passed']
    assert len(validation['checks']) == (96 if kind == 'o3' else 64)
    assert all(row['finite_fp32'] and row['semantic_tolerance_passed'] for row in validation['checks'])
    assert len(validation['grouped_coordinates_checks']) == (2 if kind == 'o3' else 8)
    if kind == 'o3':
        assert len(validation['rejected']) == 8
        resource = json.loads((directory/'resources.json').read_text())
        pair = (resource['v89'], resource['v99'])
    else:
        assert validation['edge_count'] == len(validation['edge_checks']) == 12
        resource = json.loads((directory/'environment.json').read_text())['resources']
        pair = (resource['0'], resource['1'])
    assert all(row['registers_per_thread'] == 168 and row['active_blocks_per_sm'] == 3
               and row['threads'] == 128 for row in pair)
    assert pair[0]['integer_output_policy'] == 'default_wb'
    assert pair[1]['integer_output_policy'] == 'streaming_cs'


def test_four_mode_confirmation_raw_statistics_timing_and_mse():
    directory = EVIDENCE/'runs/o378_roof_v99_o78_four'
    result = analyze_run(directory, 'o78', four_modes=True)
    canonical = json.loads((EVIDENCE/'reports/o378_v99_complete_analysis.json').read_text())
    assert result == canonical['runs']['o78_four']
    assert hashlib.sha256((ROOT/'scripts/analyze_output_streaming.py').read_bytes()).hexdigest() == canonical['source_sha256']
    assert result['records'] == 384 and result['rounds'] == 1
    assert len(result['summaries']) == 16
    previous = {r['variant']: (r['median_mse'], r['mean_mse'])
                for r in canonical['runs']['o78']['summaries'] if r['policy'] == 1}
    for row in result['summaries']:
        assert (row['median_mse'], row['mean_mse']) == previous[row['variant']]
    rows = [json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
    for row in rows:
        mode = row['mode']
        assert row['timing_contract_version'] == 2
        assert row['weight_cached'] == (mode in ('compute_only', 'steady_state'))
        assert row['activation_prepared'] == (mode == 'compute_only')
        assert row['total_timing'] == ('sum_of_batched_stage_samples' if mode == 'conversion_only'
                                       else 'single_execution_cuda_event')
        for stage, inner in row['stage_timing_inner_repeats'].items():
            assert inner == (100 if 'conversion' in stage or
                            (stage == 'total' and mode == 'conversion_only') else 1)
        assert row['guard']['invalid_ctas'] == 0
        assert row['guard']['integer_ctas'] + row['guard']['fallback_ctas'] == row['guard']['ctas'] == 2048


def test_limited_memcheck_scope_and_unchanged_formal_extension():
    log = (EVIDENCE/'tmp/o378_v99_o78_memcheck.log').read_text()
    assert 'OUTPUT STREAMING VALIDATION PASSED' in log
    assert 'ERROR SUMMARY: 0 errors' in log
    receipt = json.loads((EVIDENCE/'runs/o378_roof_v99_o78_memcheck/validation.json').read_text())
    assert receipt['passed'] and receipt['scope'] == 'small_MN_full_K4096_not_4096cubed_sanitizer'
    assert len(receipt['checks']) == 64 and receipt['grouped_coordinate_count'] == 8
    expected = '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert (EVIDENCE/'tmp/o378_v99_formal_extension.sha256').read_text().split()[0] == expected
    for kind in ('o3', 'o78', 'o78_four'):
        environment = json.loads((EVIDENCE/'runs'/f'o378_roof_v99_{kind}'/'environment.json').read_text())
        assert not environment['production_default_changed']
        assert environment['extension_sha256'] == expected
