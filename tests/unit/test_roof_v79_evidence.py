"""Recompute O3 eight-chain paired results; no inference from other backends."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
E = ROOT / 'docs/evidence/a100_o378_roof_v79'
B = E / 'reports/o378_roof_v79_codegen'


@pytest.mark.parametrize('suffix,samples,rounds,modes', [
    ('screen', 4, 3, ['compute_only']), ('trace24', 24, 3, ['compute_only']),
    ('four24', 24, 1, ['conversion_only', 'compute_only', 'cold', 'steady_state'])])
def test_timings_guard_cost_and_bitwise_mse(suffix, samples, rounds, modes):
    from benchmark_a100_o1 import stats
    from benchmark_o3_eight_chain_probe import summarize, CONTROL, CANDIDATE
    from roof_full_pipeline_probe import stage_contract
    run = E / f'runs/o378_roof_v79_{suffix}'
    rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
    ids = sorted({r['sample_id'] for r in rows})
    assert len(ids) == samples and len(rows) == samples * rounds * len(modes) * 2
    for r in rows:
        assert r['variant'] == 'o3' and r['paired_reference'] == 'o0'
        assert r['payload_bitwise'] and r['bitwise_equal_current_best']
        assert r['mse_vs_current_best'] == 0 and r['guard_status'] == 0
        assert r['stage_timing_inner_repeats'] == stage_contract(r['mode'], 100)
        assert r['weight_cached'] == (r['mode'] not in ('cold', 'conversion_only'))
        assert r['activation_prepared'] == (r['mode'] == 'compute_only')
        assert r['kernel']['weight_conversion_kernels'] == 2  # both use guard!
        assert r['kernel']['activation_conversion_kernels'] == 1
        assert r['kernel']['conversion_candidate'] == 2
        assert r['kernel']['kernel_symbol'] == (CONTROL if r['implementation'] == 0 else CANDIDATE)
        assert r['kernel']['guard_preparation_charged_to'] == 'weight_conversion'
        assert r['kernel']['guard_cached'] == r['weight_cached']
        assert not r['kernel']['production_default_changed']
        assert set(r['raw_ms']) == set(stage_contract(r['mode'], 100))
        for stage, values in r['raw_ms'].items():
            assert len(values) == 200 and min(values) > 0
            assert stats(values) == pytest.approx(r['stage_summaries'][stage])
        if r['mode'] == 'conversion_only':
            assert r['total_timing'] == 'sum_of_batched_stage_samples'
            assert r['raw_ms']['total'] == pytest.approx([
                w + a for w, a in zip(r['raw_ms']['weight_conversion'], r['raw_ms']['activation_conversion'])])
        else:
            assert r['total_timing'] == 'single_execution_cuda_event'
            assert r['stage_timing_inner_repeats']['total'] == 1
    saved = json.loads((run / 'summary.json').read_text())
    assert not saved['production_default_changed']
    assert saved['records'] == summarize(rows, ids, rounds, modes)
    if samples == 24:
        for s in saved['records']:
            assert s['median_mse_vs_paired_fp16'] == pytest.approx(.006653010287409885, rel=1e-13)
            assert s['mean_mse_vs_paired_fp16'] == pytest.approx(.007578847013302748, rel=1e-13)
    env = json.loads((run / 'environment.json').read_text())
    assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert not env['production_default_changed']
    resource = json.loads((run / 'resources.json').read_text())
    for key, local in (('device', 32), ('eight_chain', 24)):
        assert resource[key] == dict(registers_per_thread=168, local_size_bytes=local,
            threads=128, active_blocks_per_sm=3, shared_memory_bytes=50688)


def test_actual_entry_int4_schedule_and_liveness():
    from probe_o3_eight_chain_codegen import CONTROL, CANDIDATE, SENTINEL, generated_header
    from probe_roof_fullk_integer_codegen import static_entries
    from benchmark_o78_eight_chain_probe import merged_mma_counts
    from inspect_o78_register_liveness import analyze
    from compare_a100_codegen import compare
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    r = json.loads((B / 'codegen.json').read_text())
    for path, value in r['sources'].items():
        assert sha(ROOT / path) == value
    assert r['generated_header_sha256'] == sha(B / 'o3_eight_chain_generated.cuh')
    assert (B / 'o3_eight_chain_generated.cuh').read_text() == generated_header(
        (ROOT / 'csrc/sm80/o3_fullk_integer_probe.cuh').read_text())
    sass = (B / 'o3_eight_chain.sass').read_text()
    old = (ROOT / 'docs/evidence/a100_o378_roof_v61/reports/o378_roof_v61/device_factor.sass').read_text()
    assert compare(old, sass, '^' + CONTROL + '$') == r['control_comparison']
    assert compare(old, sass, '^' + SENTINEL + '$') == r['sentinel_comparison']
    entries = static_entries(sass, r'^adangel_roof_(?:device_factor_o3|o3_eight_chain_candidate|fullk_integer_o78)$',
                             {CONTROL, CANDIDATE, SENTINEL})
    assert entries == r['entries']
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] for e in entries.values())
    for symbol, instructions, live, loads in ((CONTROL, 315, 158, 7), (CANDIDATE, 319, 162, 4)):
        result = analyze((B / 'liveness.txt').read_text(), symbol)
        assert result == r['liveness'][symbol] and result['allocated_gpr'] == 168
        loop = next(l for l in result['loops'] if l['kind'] == 'integer')
        assert loop['static_instructions'] == instructions and loop['max_live_gpr'] == live
        assert loop['opcode_counts']['LDL'] == loads
        assert loop['opcode_counts']['LDSM.16.M88.4'] == 16
        assert loop['opcode_counts']['LDGSTS.E.BYPASS.128'] == 9
        assert merged_mma_counts(sass, result, symbol) == dict(
            u4_total=32, u4_zero_c=16 if symbol == CONTROL else 0, s4_total=32, s4_zero_c=16)
    if (B / 'o3_eight_chain.cubin').exists():
        assert sha(B / 'o3_eight_chain.cubin') == r['cubin_sha256']


def test_gpu_validation_and_limited_sanitizer():
    for suffix in ('screen', 'trace24', 'four24', 'memcheck', 'synccheck', 'racecheck'):
        r = json.loads((E / f'runs/o378_roof_v79_{suffix}/validation.json').read_text())
        assert r['passed'] and len(r['checks']) == 96 and len(r['rejected']) == 8
        assert all(c['bitwise_v61'] and c['semantic_tolerance_passed'] and c['finite_fp32']
                   and c['payload_exact'] and c['scale_exact'] for c in r['checks'])
        assert {c['status'] for c in r['checks']} == {0, 1}
        assert {tuple(c['shape']) for c in r['checks']} == {(64, 128, 4096), (128, 256, 4096)}
        assert any(c['pattern'] == 'mixed_tiles' and c['shape'][1] == 256 and c['status'] == 1 for c in r['checks'])
    for name in ('memcheck', 'synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (B / f'{name}.log').read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (B / 'racecheck.log').read_text()
