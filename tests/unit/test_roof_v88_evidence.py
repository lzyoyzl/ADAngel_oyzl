"""Recompute CuTe-traversal measurements; never infer speed from reuse hints."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
E = ROOT / 'docs/evidence/a100_o378_roof_v88'
B = E / 'reports/o378_roof_v88_codegen'


@pytest.mark.parametrize('name,samples,rounds,modes', [
    ('screen', 4, 3, ('compute_only',)),
    ('confirm', 24, 3, ('compute_only',)),
])
def test_raw_pairing_stats_MSE_no_filtering(name, samples, rounds, modes):
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    from benchmark_o78_cute_traversal import timing_contract
    run = E / ('runs/o378_roof_v88_' + name)
    rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == samples * rounds * 2 * 2 * len(modes)
    assert len({r['sample_id'] for r in rows}) == samples
    assert {r['round'] for r in rows} == set(range(rounds))
    for r in rows:
        assert r['paired_fp16'] == {'o7': 'o5', 'o8': 'o6'}[r['variant']]
        assert all(r[k] for k in ('finite_fp32', 'metadata_exact', 'bitwise_equal_v67', 'MSE_regression_passed'))
        assert r['mse_vs_v67'] == r['max_abs_vs_v67'] == 0
        assert r['guard']['invalid_ctas'] == 0
        for key, value in timing_contract(r['mode'], 100).items(): assert r[key] == value
        for stage, times in r['raw_ms'].items():
            assert len(times) == 200 and min(times) > 0
            assert stats(times) == pytest.approx(r['stage_summaries'][stage])
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['records'] == summarize(rows, modes)
    assert summary['no_filtering'] and not summary['production_default_changed']
    for old, new in zip(summary['records'][::2], summary['records'][1::2]):
        assert old['median_mse'] == new['median_mse'] and old['mean_mse'] == new['mean_mse']
        assert old['selected_cv_failed_records'] > 0 and new['selected_cv_failed_records'] > 0


def test_control_identity_native_INT4_work_and_actual_encoded_traversal():
    from compare_a100_codegen import compare
    from inspect_o78_register_liveness import analyze
    from inspect_eight_chain_schedule import trace
    from benchmark_o78_cute_traversal import json_canonical
    from probe_o78_cute_traversal_codegen import CONTROL, SYMBOL, STEM, generated_header, loop_summary
    from probe_roof_fullk_integer_codegen import static_entries
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((B / 'codegen.json').read_text())
    assert receipt['source_commit'].startswith('866f48b')
    assert not receipt['production_default_changed'] and not receipt['changed_semantics']
    for path, digest in receipt['sources'].items(): assert sha(ROOT / path) == digest
    for path, digest in receipt['artifact_sha256'].items(): assert sha(B / path) == digest
    header = B / (STEM + '_generated.cuh')
    assert sha(header) == receipt['generated_header_sha256']
    assert header.read_text() == generated_header((ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    sass = (B / (STEM + '.sass')).read_text()
    old = (ROOT / 'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass').read_text()
    assert compare(old, sass, '^'+CONTROL+'$') == receipt['control_comparison']
    assert receipt['control_comparison']['passed']
    assert static_entries(sass, '^(?:'+CONTROL+'|'+SYMBOL+')$', {CONTROL, SYMBOL}) == receipt['entries']
    for symbol, count, reuse in ((CONTROL, 383, 25), (SYMBOL, 377, 27)):
        live = analyze((B / 'liveness.txt').read_text(), symbol)
        assert live == receipt['liveness'][symbol] and live['allocated_gpr'] == 168
        info = loop_summary(sass, symbol, live)
        assert info == receipt['loop_summary'][symbol]
        assert info['instructions'] == count and info['operand_reuse_markers'] == reuse
        assert info['mma_count'] == 64 and info['opcode_counts']['LDSM.16.M88.4'] == 16
        assert info['opcode_counts']['LDGSTS.E.BYPASS.128'] == 10
        assert not any(op.startswith(('LDL', 'STL')) for op in info['opcode_counts'])
        chain = trace(sass, symbol, live)
        assert json_canonical(chain) == receipt['schedule'][symbol]
        assert chain['chains_per_group'] == 16 and chain['peak_started_not_finished_chains'] == 8
        ptx = next(t for t in re.split(r'(?=\.visible \.entry )', (B / (STEM + '.ptx')).read_text())
                   if t.startswith('.visible .entry '+symbol+'('))
        assert all(t in ptx for t in ('cp.async.cg.shared.global', '.s32.u4.s4.s32', '.s32.s4.s4.s32'))


def test_preparation_resources_and_source_provenance():
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, samples in (('screen', 4), ('confirm', 24)):
        run = E / ('runs/o378_roof_v88_' + name)
        env = json.loads((run / 'environment.json').read_text())
        assert env['git_commit'].startswith('819ac75') and env['variants'] == ['o7', 'o8']
        assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
        assert env['source_quantization'] == 'original_FP16_direct_source_quantization_excluded'
        assert env['timing_scope'] == 'cached_compute_only_not_E2E'
        assert env['no_filtering'] and not env['production_default_changed']
        for policy in ('0', '1'):
            r = env['resources'][policy]
            assert r['cta_tile'] == [64,128,128] and r['pipeline_stages'] == 2 and r['threads'] == 128
            assert r['local_size_bytes'] == 0 and r['registers_per_thread'] == 168 and r['active_blocks_per_sm'] == 3
            assert r['shared_memory_bytes'] == 34304
        for path, digest in env['codegen']['cute_traversal']['runtime_source_sha256'].items():
            assert sha(ROOT / path) == digest
        preparation = json.loads((ROOT / 'docs/evidence/a100_o378_roof_v73/reports/o378_roof_v73_codegen/build.json').read_text())
        assert env['gpu_preparation_build'] == preparation
        sources = [json.loads(s) for s in (run / 'source_provenance.jsonl').read_text().splitlines()]
        assert len(sources) == 2 * samples
        assert len({(s['sample_id'], s['variant']) for s in sources}) == 2 * samples
        assert all(re.fullmatch('[0-9a-f]{64}', s['raw_sha256']) for s in sources)


def test_finite_validation_and_sanitizer_scope():
    for name in ('preflight', 'screen', 'confirm', 'memcheck', 'synccheck', 'racecheck'):
        r = json.loads((E / ('runs/o378_roof_v88_' + name) / 'validation.json').read_text())
        assert r['passed'] and r['count'] == len(r['checks']) == 64
        assert r['edge_count'] == len(r['edge_checks']) == 12
        assert r['scope'] == 'small_MN_full_K4096_not_4096cubed_sanitizer'
        assert {c['variant'] for c in r['checks']} == {'o7', 'o8'}
        assert {tuple(c['shape']) for c in r['checks']} == {(64,128,4096), (128,256,4096)}
        for c in r['checks']:
            assert all(c[k] for k in ('bitwise_equal_v67', 'finite_fp32', 'semantic_tolerance_passed',
                                     'metadata_exact', 'nondefault_stream'))
            if c['pattern'] == 'wide_scale': assert c['fallback_ctas'] == c['ctas']
        assert all(c['metadata_exact'] for c in r['edge_checks'])
    for tool in ('memcheck', 'synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (B / (tool + '.log')).read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (B / 'racecheck.log').read_text()
