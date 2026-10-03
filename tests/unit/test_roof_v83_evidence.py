"""Keep N256's negative screen, compiler limitation and safety scope auditable."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
E = ROOT / 'docs/evidence/a100_o378_roof_v83'
B = E / 'reports/o378_roof_v83_codegen'
RUN = E / 'runs/o378_roof_v83_screen'


def test_all_events_mse_and_negative_paired_result_recompute():
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    from benchmark_o78_n256_probe import contract

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
        # These are the original N128 guard cells, NOT the N256 launch grid.
        assert r['guard']['ctas'] == r['guard']['integer_ctas'] == 2048
        assert r['guard']['fallback_ctas'] == r['guard']['invalid_ctas'] == 0
        assert set(r['raw_ms']) == {'gemm', 'total'}
        assert r['raw_ms']['gemm'] == r['raw_ms']['total']
        for key, value in contract('compute_only', 100).items():
            assert r[key] == value
        for stage, times in r['raw_ms'].items():
            assert len(times) == 200 and min(times) > 0
            assert stats(times) == pytest.approx(r['stage_summaries'][stage])
    saved = json.loads((RUN / 'summary.json').read_text())
    assert saved['no_filtering'] and not saved['production_default_changed']
    assert saved['records'] == summarize(rows, ('compute_only',))
    for variant, failures in (('o7', 9), ('o8', 10)):
        old, new = [r for r in saved['records'] if r['variant'] == variant]
        assert old['median_mse'] == new['median_mse'] and old['mean_mse'] == new['mean_mse']
        assert old['selected_cv_failed_records'] == 12
        assert new['selected_cv_failed_records'] == failures
        assert new['paired_speedup'] < 1 and new['paired_speedup_ci95'][1] <= 1


def test_provenance_resources_and_cached_only_contract():
    env = json.loads((RUN / 'environment.json').read_text())
    assert env['source_quantization'] == 'original_FP16_direct_source_quantization_excluded'
    assert env['timing_scope'] == 'cached_compute_only_not_E2E'
    assert not env['production_default_changed'] and env['no_filtering']
    assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert env['control'] == 'v78_N128_same_v73_preparation'
    assert env['candidate'] == 'v83_N256_same_v73_preparation'
    for policy, n, regs, shared, ctas in (('0', 128, 168, 34304, 3), ('1', 256, 255, 51712, 2)):
        r = env['resources'][policy]
        assert r['cta_tile'] == [64, n, 128] and r['pipeline_stages'] == 2
        assert r['threads'] == 128 and r['local_size_bytes'] == 0
        assert r['registers_per_thread'] == regs and r['shared_memory_bytes'] == shared
        assert r['active_blocks_per_sm'] == ctas
    provenance = [json.loads(line) for line in (RUN / 'source_provenance.jsonl').read_text().splitlines()]
    assert len(provenance) == 8 and len({(p['sample_id'], p['variant']) for p in provenance}) == 8
    for p in provenance:
        assert re.fullmatch('[0-9a-f]{64}', p['raw_sha256'])
        for name in ('activation', 'weight'):
            assert p[name]['group_size'] == 128 and p[name]['shape'] == [4096, 4096]


def test_same_entry_int4_work_and_unavailable_liveness_are_not_hidden():
    from compare_a100_codegen import compare
    from probe_o78_n256_codegen import CONTROL, SYMBOL, generated_header, loops
    from probe_roof_fullk_integer_codegen import static_entries

    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((B / 'codegen.json').read_text())
    for path, value in receipt['sources'].items():
        assert digest(ROOT / path) == value
    for path, value in receipt['artifact_sha256'].items():
        assert digest(B / path) == value
    driver = json.loads((B / 'timing_driver.json').read_text())
    for path, value in driver['sources'].items():
        assert digest(ROOT / path) == value
    assert receipt['compiled_commit'] == '5187d7f' and receipt['audit_existing_artifacts']
    assert not receipt['liveness_available'] and not receipt['production_default_changed']
    assert (B / 'liveness.txt').read_text().strip() == "nvdisasm fatal   : Invalid register count : '255'"
    assert digest(B / 'o78_n256_generated.cuh') == receipt['generated_header_sha256']
    assert (B / 'o78_n256_generated.cuh').read_text() == generated_header(
        (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    sass = (B / 'o78_n256.sass').read_text()
    old = (ROOT / 'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass').read_text()
    assert compare(old, sass, '^' + CONTROL + '$') == receipt['control_comparison']
    assert receipt['control_comparison']['passed']
    assert static_entries(sass, '^' + SYMBOL + '$', {SYMBOL}) == receipt['entries']
    entry = receipt['entries'][SYMBOL]
    assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma']
    ptx = next(b for b in re.split(r'(?=\.visible \.entry )', (B / 'o78_n256.ptx').read_text())
               if b.startswith('.visible .entry ' + SYMBOL + '('))
    assert all(t in ptx for t in ('cp.async.cg.shared.global', '.s32.u4.s4.s32', '.s32.s4.s4.s32'))
    for symbol, mma, instructions, ldsm, copies in ((CONTROL, 64, 383, 16, 10), (SYMBOL, 128, 671, 24, 14)):
        actual = loops(sass, symbol, mma)
        assert actual == receipt['loop_counts'][symbol]
        integer = next(r for r in actual if r['kind'] == 'integer')
        assert integer['instruction_count'] == instructions
        assert integer['opcodes']['LDSM.16.M88.4'] == ldsm
        assert integer['opcodes']['LDGSTS.E.BYPASS.128'] == copies
        assert not any(op.startswith(('LDL', 'STL')) for op in integer['opcodes'])
    # The larger CTA halves the grid. This is a static work model, not NCU timing.
    assert 24 * 1024 / (16 * 2048) == .75
    assert 14 * 1024 / (10 * 2048) == .70


def test_limited_sanitizers_and_observed_mixed_fallback_rounding():
    for path in (RUN, *(E / f'reports/o378_roof_v83_{name}' for name in ('memcheck', 'synccheck', 'racecheck'))):
        r = json.loads((path / 'validation.json').read_text())
        assert r['passed'] and r['count'] == 16
        assert r['scope'] == 'compute_only_small_MN_full_K_not_all_shapes_or_four_modes'
        assert {tuple(c['shape']) for c in r['checks']} == {(128, 512, 4096)}
        assert all(c['finite_fp32'] and c['semantic_tolerance_passed'] and c['metadata_exact']
                   and c['nondefault_stream'] and c['invalid_ctas'] == 0 for c in r['checks'])
        for c in r['checks']:
            if not c['fallback_ctas']:
                assert c['bitwise_equal_v67']
        asymmetric = next(c for c in r['checks'] if (c['variant'], c['pattern'], c['policy']) == ('o8', 'wide_scale', 1))
        assert asymmetric['integer_ctas'] == 6 and asymmetric['fallback_ctas'] == 2
        # N256 expands the fallback region: tolerance passes, but bitwise is false.
        assert not asymmetric['bitwise_equal_v67']
    for name in ('memcheck', 'synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (B / f'{name}.log').read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (B / 'racecheck.log').read_text()
