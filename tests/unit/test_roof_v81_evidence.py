"""Verify a measured-hotspot change was rejected without hiding negative data."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
E = ROOT / 'docs/evidence/a100_o378_roof_v81'
B = E / 'reports/o378_roof_v81_codegen'


def test_all_paired_events_and_mse_recompute_without_filtering():
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    from benchmark_o78_warp_metadata import timing_contract
    run = E / 'runs/o378_roof_v81_screen'
    rows = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 48 and len({r['sample_id'] for r in rows}) == 4
    assert {r['round'] for r in rows} == {0, 1, 2}
    for r in rows:
        assert r['mode'] == 'compute_only' and r['paired_fp16'] == {'o7': 'o5', 'o8': 'o6'}[r['variant']]
        assert r['finite_fp32'] and r['metadata_exact'] and r['bitwise_equal_v67'] and r['MSE_regression_passed']
        assert r['mse_vs_v67'] == r['max_abs_vs_v67'] == 0
        assert r['guard']['integer_ctas'] == 2048 and r['guard']['fallback_ctas'] == r['guard']['invalid_ctas'] == 0
        assert set(r['raw_ms']) == {'gemm', 'total'} and r['raw_ms']['gemm'] == r['raw_ms']['total']
        for key, value in timing_contract('compute_only', 100).items():
            assert r[key] == value
        for stage, values in r['raw_ms'].items():
            assert len(values) == 200 and min(values) > 0
            assert stats(values) == pytest.approx(r['stage_summaries'][stage])
    saved = json.loads((run / 'summary.json').read_text())
    assert saved['no_filtering'] and not saved['production_default_changed']
    assert saved['records'] == summarize(rows, ('compute_only',))
    for variant in ('o7', 'o8'):
        old, new = [r for r in saved['records'] if r['variant'] == variant]
        assert old['median_mse'] == new['median_mse'] and old['mean_mse'] == new['mean_mse']
        assert old['selected_cv_failed_records'] == 12 and new['selected_cv_failed_records'] == 11
        assert new['paired_speedup_ci95'][0] <= 1 <= new['paired_speedup_ci95'][1]
    env = json.loads((run / 'environment.json').read_text())
    assert env['source_quantization'] == 'original_FP16_direct_source_quantization_excluded'
    assert env['timing_scope'] == 'cached_compute_only_not_E2E'
    assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert env['control'] == 'v78_eight_chain' and env['candidate'] == 'v81_full_warp_factor_copy'
    for policy in ('0', '1'):
        r = env['resources'][policy]
        assert r['registers_per_thread'] == 168 and r['active_blocks_per_sm'] == 3
        assert r['shared_memory_bytes'] == 34304 and r['local_size_bytes'] == 0


def test_same_entry_int4_copy_and_unchanged_control_machine_code():
    from compare_a100_codegen import compare
    from probe_o78_warp_metadata_codegen import generated_header, CONTROL, SYMBOL
    from probe_roof_fullk_integer_codegen import static_entries
    from inspect_o78_register_liveness import analyze
    from inspect_eight_chain_schedule import trace
    from benchmark_o78_eight_chain_probe import merged_mma_counts
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((B / 'codegen.json').read_text())
    for path, value in receipt['sources'].items():
        assert digest(ROOT / path) == value
    assert not receipt['production_default_changed'] and receipt['metadata_bytes_per_cta_group'] == 768
    assert digest(B / 'o78_warp_metadata_generated.cuh') == receipt['generated_header_sha256']
    assert (B / 'o78_warp_metadata_generated.cuh').read_text() == generated_header(
        (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    sass = (B / 'o78_warp_metadata.sass').read_text()
    old = (ROOT / 'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass').read_text()
    assert compare(old, sass, '^' + CONTROL + '$') == receipt['control_comparison']
    assert receipt['control_comparison']['passed']
    assert static_entries(sass, '^adangel_roof_o78_(?:eight_chain_candidate|warp_metadata_candidate)$',
                          {CONTROL, SYMBOL}) == receipt['entries']
    ptx = next(b for b in re.split(r'(?=\.visible \.entry )', (B / 'o78_warp_metadata.ptx').read_text())
               if b.startswith('.visible .entry ' + SYMBOL + '('))
    assert all(term in ptx for term in ('cp.async.ca.shared.global', 'cp.async.cg.shared.global',
                                      '.s32.u4.s4.s32', '.s32.s4.s4.s32'))
    for symbol, live_count, instructions, bypass in ((CONTROL, 166, 383, 10), (SYMBOL, 164, 388, 9)):
        entry = receipt['entries'][symbol]
        assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma']
        live = analyze((B / 'liveness.txt').read_text(), symbol)
        assert live == receipt['liveness'][symbol] and live['allocated_gpr'] == 168
        loop = next(item for item in live['loops'] if item['kind'] == 'integer')
        assert loop['max_live_gpr'] == live_count and loop['static_instructions'] == instructions
        assert loop['opcode_counts']['LDSM.16.M88.4'] == 16
        assert loop['opcode_counts']['LDGSTS.E.BYPASS.128'] == bypass
        assert loop['opcode_counts'].get('LDGSTS.E.64', 0) == (symbol == SYMBOL)
        assert not any(op.startswith(('LDL', 'STL')) for op in loop['opcode_counts'])
        assert merged_mma_counts(sass, live, symbol) == dict(u4_total=32, u4_zero_c=0, s4_total=32, s4_zero_c=16)
        assert trace(sass, symbol, live)['peak_started_not_finished_chains'] == 8
    if (B / 'o78_warp_metadata.cubin').exists():
        assert digest(B / 'o78_warp_metadata.cubin') == receipt['cubin_sha256']


def test_gpu_validation_and_limited_sanitizer_not_full_matrix_claim():
    for suffix in ('screen', 'memcheck', 'synccheck', 'racecheck'):
        r = json.loads((E / f'runs/o378_roof_v81_{suffix}/validation.json').read_text())
        assert r['passed'] and r['count'] == 64 and r['edge_count'] == 12
        assert r['scope'] == 'small_MN_full_K4096_not_4096cubed_sanitizer'
        assert all(c['bitwise_equal_v67'] and c['finite_fp32'] and c['semantic_tolerance_passed']
                   and c['nondefault_stream'] for c in r['checks'])
        assert {tuple(c['shape']) for c in r['checks']} == {(64, 128, 4096), (128, 256, 4096)}
        assert any(c['pattern'] == 'wide_scale' and c['integer_ctas'] == 0
                   and c['fallback_ctas'] == c['ctas'] for c in r['checks'])
    for name in ('memcheck', 'synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (B / f'{name}.log').read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (B / 'racecheck.log').read_text()


def test_ncu_recompute_proves_less_metadata_work_not_event_speedup():
    from run_eight_chain_ncu import analyze_capture
    ncu = E / 'reports/o378_roof_v81_ncu'
    saved = json.loads((ncu / 'analysis.json').read_text())
    for path, digest in saved['input_sha256'].items():
        assert hashlib.sha256((ncu / path).read_bytes()).hexdigest() == digest
    receipt = json.loads((ncu / 'o7/receipt.json').read_text())
    assert receipt['metadata_copy_candidate'] and receipt['expected_kernel'] == 'adangel_roof_o78_warp_metadata_candidate'
    row = analyze_capture((ncu / 'o7_raw.csv').read_text(), (ncu / 'o7_source_sass.csv').read_text(), receipt)
    assert saved['rows'] == [row] and not saved['new_performance_result'] and not saved['production_default_changed']
    assert row['source_memory_work']['L1 Wavefronts Shared Excessive'] == 0
    assert row['source_memory_work']['L2 Theoretical Sectors Global Excessive'] == 0
    assert row['source_memory_work']['L1 Wavefronts Shared'] == 30801920
    assert row['dynamic_instructions'] == 106823680
    assert row['opcodes']['IMMA'] == 16777216 and row['opcodes']['LDSM'] == 4194304
    assert row['mse_vs_previous_fullk'] == 0
    assert row['optimistic_fixed_work_lower_bound_ms'] == pytest.approx(.22034693984764905)
