"""Recompute the negative three-stage screen and preserve its safety limits."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
E = ROOT / 'docs/evidence/a100_o378_roof_v84'
B = E / 'reports/o378_roof_v84_codegen'
RUN = E / 'runs/o378_roof_v84_screen'


def test_all_events_mse_and_negative_paired_result_recompute():
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    from benchmark_o78_three_stage import timing_contract

    rows = [json.loads(line) for line in (RUN / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 48
    assert {r['sample_id'] for r in rows} == {
        'layer_00_q_proj', 'layer_00_k_proj', 'layer_00_v_proj', 'layer_00_o_proj'}
    assert {r['round'] for r in rows} == {0, 1, 2}
    for r in rows:
        assert r['mode'] == 'compute_only'
        assert r['paired_fp16'] == {'o7': 'o5', 'o8': 'o6'}[r['variant']]
        assert r['finite_fp32'] and r['metadata_exact'] and r['bitwise_equal_v67']
        assert r['MSE_regression_passed'] and r['mse_vs_v67'] == r['max_abs_vs_v67'] == 0
        assert r['guard']['ctas'] == r['guard']['integer_ctas'] == 2048
        assert r['guard']['fallback_ctas'] == r['guard']['invalid_ctas'] == 0
        assert set(r['raw_ms']) == {'gemm', 'total'}
        assert r['raw_ms']['gemm'] == r['raw_ms']['total']
        for key, value in timing_contract('compute_only', 100).items():
            assert r[key] == value
        for stage, times in r['raw_ms'].items():
            assert len(times) == 200 and min(times) > 0
            assert stats(times) == pytest.approx(r['stage_summaries'][stage])
    saved = json.loads((RUN / 'summary.json').read_text())
    assert saved['no_filtering'] and not saved['production_default_changed']
    assert saved['records'] == summarize(rows, ('compute_only',))
    for variant, failures in (('o7', 12), ('o8', 11)):
        old, new = [r for r in saved['records'] if r['variant'] == variant]
        assert old['median_mse'] == new['median_mse'] and old['mean_mse'] == new['mean_mse']
        assert old['selected_cv_failed_records'] == new['selected_cv_failed_records'] == failures
        assert new['paired_speedup'] < 1 and new['paired_speedup_ci95'][1] < 1


def test_provenance_resources_and_cached_only_contract():
    env = json.loads((RUN / 'environment.json').read_text())
    assert env['source_quantization'] == 'original_FP16_direct_source_quantization_excluded'
    assert env['timing_scope'] == 'cached_compute_only_not_E2E'
    assert not env['production_default_changed'] and env['no_filtering']
    assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert env['control'] == 'v78_two_stage_integer' and env['candidate'] == 'v84_three_stage_integer'
    for policy, stages, shared, local in (('0', 2, 34304, 0), ('1', 3, 51456, 8)):
        r = env['resources'][policy]
        assert r['cta_tile'] == [64, 128, 128] and r['pipeline_stages'] == stages
        assert r['threads'] == 128 and r['local_size_bytes'] == local
        assert r['registers_per_thread'] == 168 and r['shared_memory_bytes'] == shared
        assert r['active_blocks_per_sm'] == 3
    assert env['resources']['1']['fallback_pipeline_stages'] == 2
    provenance = [json.loads(line) for line in (RUN / 'source_provenance.jsonl').read_text().splitlines()]
    assert len(provenance) == 8 and len({(p['sample_id'], p['variant']) for p in provenance}) == 8
    for p in provenance:
        assert re.fullmatch('[0-9a-f]{64}', p['raw_sha256'])
        for name in ('activation', 'weight'):
            assert p[name]['group_size'] == 128 and p[name]['shape'] == [4096, 4096]


def test_same_entry_native_int4_waits_and_new_hot_spill_are_auditable():
    from benchmark_o78_eight_chain_probe import merged_mma_counts
    from compare_a100_codegen import compare
    from inspect_o78_register_liveness import analyze
    from probe_o78_three_stage_codegen import CONTROL, SYMBOL, generated_header
    from probe_roof_fullk_integer_codegen import static_entries

    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((B / 'codegen.json').read_text())
    assert receipt['source_commit'] == 'f709a77df1c6ca7b8caf637d2e6181090444ae13'
    assert not receipt['production_default_changed']
    for path, value in receipt['sources'].items():
        assert digest(ROOT / path) == value
    for path, value in receipt['artifact_sha256'].items():
        assert digest(B / path) == value
    assert digest(B / 'o78_three_stage_generated.cuh') == receipt['generated_header_sha256']
    assert (B / 'o78_three_stage_generated.cuh').read_text() == generated_header(
        (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    sass = (B / 'o78_three_stage.sass').read_text()
    old = (ROOT / 'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass').read_text()
    assert compare(old, sass, '^' + CONTROL + '$') == receipt['control_comparison']
    assert receipt['control_comparison']['passed']
    assert static_entries(sass, '^adangel_roof_o78_(?:eight_chain_candidate|three_stage_candidate)$',
                          {CONTROL, SYMBOL}) == receipt['entries']
    ptx = next(b for b in re.split(r'(?=\.visible \.entry )', (B / 'o78_three_stage.ptx').read_text())
               if b.startswith('.visible .entry ' + SYMBOL + '('))
    assert all(t in ptx for t in ('cp.async.cg.shared.global', '.s32.u4.s4.s32',
                                 '.s32.s4.s4.s32', 'cp.async.wait_group 1', 'cp.async.wait_group 0'))
    for symbol, instructions, fallback_count, local_loads in ((CONTROL, 383, 440, 0), (SYMBOL, 399, 453, 1)):
        actual = analyze((B / 'liveness.txt').read_text(), symbol)
        assert actual == receipt['liveness'][symbol]
        assert actual['allocated_gpr'] == 168
        loop = next(r for r in actual['loops'] if r['kind'] == 'integer')
        assert loop['static_instructions'] == instructions
        assert loop['opcode_counts']['LDSM.16.M88.4'] == 16
        assert loop['opcode_counts']['LDGSTS.E.BYPASS.128'] == 10
        assert sum(v for op, v in loop['opcode_counts'].items() if op.startswith('LDL')) == local_loads
        assert not any(op.startswith('STL') for op in loop['opcode_counts'])
        assert next(r for r in actual['loops'] if r['kind'] == 'fp32_fallback')['static_instructions'] == fallback_count
        assert merged_mma_counts(sass, actual, symbol) == {
            'u4_total': 32, 'u4_zero_c': 0, 's4_total': 32, 's4_zero_c': 16}
        entry = receipt['entries'][symbol]
        assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma']
        assert entry['all_copies_bypass_l1']


def test_limited_sanitizers_cover_four_modes_and_fallback():
    for path in (RUN, *(E / f'reports/o378_roof_v84_{name}' for name in ('memcheck', 'synccheck', 'racecheck'))):
        r = json.loads((path / 'validation.json').read_text())
        assert r['passed'] and r['count'] == len(r['checks']) == 64
        assert r['edge_count'] == len(r['edge_checks']) == 12
        assert r['scope'] == 'small_MN_full_K4096_not_4096cubed_sanitizer'
        assert {tuple(c['shape']) for c in r['checks']} == {(64, 128, 4096), (128, 256, 4096)}
        assert {c['mode'] for c in r['checks']} == {'conversion_only', 'compute_only', 'cold', 'steady_state'}
        for c in r['checks']:
            assert all(c[k] for k in ('bitwise_equal_v67', 'finite_fp32', 'semantic_tolerance_passed',
                                      'metadata_exact', 'nondefault_stream'))
            assert c['invalid_ctas'] == 0
            if c['pattern'] == 'wide_scale':
                assert c['fallback_ctas'] == c['ctas'] and c['integer_ctas'] == 0
        assert all(c['metadata_exact'] for c in r['edge_checks'])
        assert sum(c['invalid_ctas'] for c in r['edge_checks']) == 3
    for name in ('memcheck', 'synccheck'):
        log = (B / f'{name}.log').read_text()
        assert 'THREE STAGE INTEGER VALIDATION PASSED' in log and 'ERROR SUMMARY: 0 errors' in log
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (B / 'racecheck.log').read_text()
