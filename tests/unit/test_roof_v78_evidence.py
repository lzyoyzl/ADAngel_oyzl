"""Recompute v78 paired timings, true-scale MSE and exact-entry ISA evidence."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
EVIDENCE = ROOT / 'docs/evidence/a100_o378_roof_v78'
BUILD = EVIDENCE / 'reports/o378_roof_v78_codegen'


@pytest.mark.parametrize('suffix,samples,rounds', [('screen', 4, 3), ('trace24', 24, 3)])
def test_cached_events_pairing_and_actual_quantized_output(suffix, samples, rounds):
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    from benchmark_o78_eight_chain_probe import timing_contract
    run = EVIDENCE / ('runs/o378_roof_v78_' + suffix)
    rows = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == samples * rounds * 2 * 2
    assert len({r['sample_id'] for r in rows}) == samples
    for r in rows:
        assert r['mode'] == 'compute_only'
        assert r['paired_fp16'] == {'o7': 'o5', 'o8': 'o6'}[r['variant']]
        assert r['bitwise_equal_v67'] and r['finite_fp32'] and r['MSE_regression_passed'] and r['metadata_exact']
        assert r['mse_vs_v67'] == r['max_abs_vs_v67'] == 0
        expected_fallback = 12 if (r['variant'], r['sample_id']) == ('o8', 'layer_24_o_proj') else 0
        assert r['guard']['fallback_ctas'] == expected_fallback
        assert r['guard']['integer_ctas'] == 2048 - expected_fallback
        assert r['guard']['invalid_ctas'] == 0
        assert r['raw_ms']['gemm'] == r['raw_ms']['total']
        assert set(r['raw_ms']) == {'gemm', 'total'}
        for stage, values in r['raw_ms'].items():
            assert len(values) == 200
            assert stats(values) == pytest.approx(r['stage_summaries'][stage])
        for key, value in timing_contract('compute_only', 100).items():
            assert r[key] == value
    saved = json.loads((run / 'summary.json').read_text())
    assert saved['no_filtering'] and not saved['production_default_changed']
    assert saved['records'] == summarize(rows, ('compute_only',))
    for variant in ('o7', 'o8'):
        old, new = [r for r in saved['records'] if r['variant'] == variant]
        assert old['median_mse'] == new['median_mse']
        assert old['mean_mse'] == new['mean_mse']
    env = json.loads((run / 'environment.json').read_text())
    assert env['source_quantization'] == 'original_FP16_direct_source_quantization_excluded'
    assert env['timing_scope'] == 'cached_compute_only_not_E2E'
    assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert env['resources']['0'] == env['resources']['2']
    for policy in ('0', '1'):
        resource = env['resources'][policy]
        assert resource['registers_per_thread'] == 168 and resource['active_blocks_per_sm'] == 3
        assert resource['shared_memory_bytes'] == 34304 and resource['cta_tile'] == [64, 128, 128]
        assert resource['local_size_bytes'] == (0 if policy == '1' else 8)


def test_same_entry_native_int4_and_merged_accumulator_inputs():
    from probe_roof_fullk_integer_codegen import static_entries
    from probe_o78_eight_chain_codegen import generated_header, CONTROL, SYMBOL
    from benchmark_o78_eight_chain_probe import merged_mma_counts
    from inspect_o78_register_liveness import analyze
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    r = json.loads((BUILD / 'codegen.json').read_text())
    for path, value in r['sources'].items():
        assert sha(ROOT / path) == value
    assert sha(BUILD / 'o78_eight_chain_generated.cuh') == r['generated_header_sha256']
    assert (BUILD / 'o78_eight_chain_generated.cuh').read_text() == generated_header(
        (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    assert r['control_opcode_counts_match'] and r['control_instructions_match']
    assert r['independent_partial_chains'] == 8 and r['logical_live_partial_registers'] == 32
    sass = (BUILD / 'o78_eight_chain.sass').read_text()
    entries = static_entries(sass, r'^adangel_roof_o78_(?:fullk_(?:candidate|control)|eight_chain_candidate)$', set(r['entries']))
    assert entries == r['entries']
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in entries.values())
    for symbol in (CONTROL, SYMBOL):
        live = analyze((BUILD / 'liveness.txt').read_text(), symbol)
        assert live == r['liveness'][symbol]
        counts = merged_mma_counts(sass, live, symbol)
        assert counts == dict(u4_total=32, u4_zero_c=0 if symbol == SYMBOL else 16, s4_total=32, s4_zero_c=16)
        loop = next(x for x in live['loops'] if x['kind'] == 'integer')
        assert loop['static_instructions'] == (383 if symbol == SYMBOL else 378)
        assert loop['max_live_gpr'] == (166 if symbol == SYMBOL else 160)
        assert loop['opcode_counts']['LDSM.16.M88.4'] == 16
        assert loop['opcode_counts']['LDGSTS.E.BYPASS.128'] == 10
        assert not any(op.startswith(('LDL', 'STL')) for op in loop['opcode_counts'])
    if (BUILD / 'o78_eight_chain.cubin').exists():
        assert sha(BUILD / 'o78_eight_chain.cubin') == r['cubin_sha256']


def test_finite_validation_and_sanitizer_scope():
    for suffix in ('screen', 'trace24', 'four24', 'memcheck', 'synccheck', 'racecheck'):
        v = json.loads((EVIDENCE / f'runs/o378_roof_v78_{suffix}/validation.json').read_text())
        assert v['passed'] and v['count'] == 64 and v['edge_count'] == 12
        assert all(c['bitwise_equal_v67'] and c['finite_fp32'] and c['semantic_tolerance_passed'] for c in v['checks'])
        assert all(c['shape'][0] <= 128 and c['shape'][1] <= 256 and c['shape'][2] == 4096 for c in v['checks'])
        assert any(c['pattern'] == 'wide_scale' and c['integer_ctas'] == 0
                   and c['fallback_ctas'] == c['ctas'] for c in v['checks'])
    for check in ('memcheck', 'synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (BUILD / f'{check}.log').read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (BUILD / 'racecheck.log').read_text()


def test_full_modes_include_online_preparation_and_direct_end_to_end_events():
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    from benchmark_o78_eight_chain_probe import timing_contract
    modes = ('conversion_only', 'compute_only', 'cold', 'steady_state')
    run = EVIDENCE / 'runs/o378_roof_v78_four24'
    rows = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 384 and len({r['sample_id'] for r in rows}) == 24
    assert {r['round'] for r in rows} == {0}
    for r in rows:
        assert r['bitwise_equal_v67'] and r['finite_fp32'] and r['MSE_regression_passed'] and r['metadata_exact']
        assert r['mse_vs_v67'] == r['max_abs_vs_v67'] == 0
        for stage, values in r['raw_ms'].items():
            assert len(values) == 200
            assert stats(values) == pytest.approx(r['stage_summaries'][stage])
        for key, value in timing_contract(r['mode'], 100).items():
            assert r[key] == value
        assert r['preparation_implementation'] == 'row_fused_conversion_factor_metadata'
        if r['mode'] in ('cold', 'steady_state'):
            assert r['stage_timing_inner_repeats']['total'] == 1
        if r['mode'] == 'steady_state':
            assert 'weight_conversion' not in r['raw_ms']
    recorded = json.loads((run / 'summary.json').read_text())
    assert recorded['records'] == summarize(rows, modes)
    assert recorded['no_filtering'] and not recorded['production_default_changed']
    env = json.loads((run / 'environment.json').read_text())
    assert env['timing_scope'] == 'all_four_modes'
    assert env['gpu_preparation_build'] == json.loads((EVIDENCE / 'runs/o378_roof_v78_trace24/environment.json').read_text())['gpu_preparation_build']
