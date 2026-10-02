"""Recompute v76 negative screen, codegen and finite GPU safety evidence."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))
EVIDENCE = ROOT / 'docs/evidence/a100_o378_roof_v76'
BUILD = EVIDENCE / 'reports/o378_roof_v76_codegen'
RUN = EVIDENCE / 'runs/o378_roof_v76_screen_r1'


def test_all_raw_events_pairing_mse_and_cv_are_preserved():
    from benchmark_a100_o1 import stats
    from benchmark_o7_factor_table import summarize, timing_contract
    rows = [json.loads(line) for line in (RUN / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 24
    assert {r['sample_id'] for r in rows} == {
        'layer_00_q_proj', 'layer_00_k_proj', 'layer_00_v_proj', 'layer_00_o_proj'}
    for row in rows:
        assert row['bitwise_equal_v67'] and row['MSE_regression_passed'] and row['metadata_exact']
        assert row['finite_fp32'] and row['paired_fp16'] == 'o5'
        assert row['mse_vs_v67'] == row['max_abs_vs_v67'] == 0
        assert row['raw_ms']['gemm'] == row['raw_ms']['total']
        assert set(row['raw_ms']) == {'gemm', 'total'}
        for stage, values in row['raw_ms'].items():
            assert len(values) == 200
            assert stats(values) == pytest.approx(row['stage_summaries'][stage])
        for key, value in timing_contract('compute_only', 100).items():
            assert row[key] == value
        coverage = row['table_coverage']
        assert coverage['table_ctas'] == 2048
        assert coverage['original_integer_fallback_ctas'] == coverage['fp32_fallback_ctas'] == coverage['invalid_ctas'] == 0
        assert coverage['scope'] == 'host_prediction_from_checked_metadata_not_device_counter'
    recorded = json.loads((RUN / 'summary.json').read_text())
    assert recorded['no_filtering'] and not recorded['production_default_changed']
    assert summarize(rows, ('compute_only',)) == recorded['records']
    old, new = recorded['records']
    assert (old['selected_cv_failed_records'], new['selected_cv_failed_records']) == (12, 2)
    assert new['paired_speedup'] == pytest.approx(.760070941354999)
    assert new['paired_speedup_ci95'][1] < 1
    assert old['median_mse'] == new['median_mse']
    assert old['mean_mse'] == new['mean_mse']


def test_source_generated_layout_and_same_entry_int4_audit():
    from probe_o7_factor_table_codegen import table_header
    from probe_roof_fullk_integer_codegen import static_entries
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((BUILD / 'codegen.json').read_text())
    for path, sha in receipt['sources'].items():
        assert digest(ROOT / path) == sha
    assert digest(BUILD / 'o7_factor_table_generated.cuh') == receipt['generated_header_sha256']
    assert (BUILD / 'o7_factor_table_generated.cuh').read_text() == table_header(
        (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    assert receipt['control_opcode_counts_match_v67'] and receipt['control_instruction_count_match_v67']
    assert receipt['table_rows'] == 10 and receipt['shared_bytes'] == 43520
    entries = static_entries((BUILD / 'o7_factor_table.sass').read_text(),
        r'^adangel_roof_o(?:78_fullk_(?:candidate|control)|7_factor_table_candidate)$', set(receipt['entries']))
    assert entries == receipt['entries']
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in entries.values())
    # The complete cubin is retained in the SHA-checked local/server archive,
    # not required as a Git source dependency. Verify it when available.
    if (BUILD / 'o7_factor_table.cubin').exists():
        assert digest(BUILD / 'o7_factor_table.cubin') == receipt['cubin_sha256']


def test_mainloop_cost_and_runtime_resource_identity():
    from analyze_o7_factor_table_codegen import analyze, CONTROL, CANDIDATE
    analysis = analyze((BUILD / 'o7_factor_table.sass').read_text())
    assert analysis == json.loads((BUILD / 'loop_analysis.json').read_text())
    entries = {e['symbol']: e['loops'] for e in analysis['entries']}
    old = next(x for x in entries[CONTROL] if x['kind'] == 'original_integer')
    new = next(x for x in entries[CANDIDATE] if x['kind'] == 'factor_table')
    assert (old['instructions'], new['instructions']) == (378, 381)
    assert (old['imad_family'], new['imad_family']) == (154, 112)
    assert (old['scalar_shared_loads'], new['scalar_shared_loads']) == (15, 39)
    assert (old['scalar_shared_stores'], new['scalar_shared_stores']) == (0, 11)
    assert len(old['local_loads']) == 0 and len(new['local_loads']) == 4
    assert old['imma'] == new['imma'] == 64
    env = json.loads((RUN / 'environment.json').read_text())
    assert not env['production_default_changed'] and env['no_filtering']
    assert env['timing_scope'] == 'cached_compute_only_not_E2E'
    assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert env['resources']['0'] == env['resources']['2']
    for policy in ('0', '1'):
        assert env['resources'][policy]['registers_per_thread'] == 168
        assert env['resources'][policy]['active_blocks_per_sm'] == 3
    assert env['resources']['0']['shared_memory_bytes'] == 34304
    assert env['resources']['1']['shared_memory_bytes'] == 43520
    assert env['resources']['1']['local_size_bytes'] == 72
    original = json.loads((ROOT / 'docs/evidence/a100_o378_roof_v67/reports/o378_roof_v67_codegen/codegen.json').read_text())
    assert env['codegen']['v67'] == original
    prepare = json.loads((ROOT / 'docs/evidence/a100_o378_roof_v73/reports/o378_roof_v73_codegen/build.json').read_text())
    assert env['gpu_preparation_build'] == prepare


def test_validation_and_sanitizers_do_not_overclaim_shape_coverage():
    for suffix in ('screen_r1', 'memcheck', 'synccheck', 'racecheck'):
        checks = json.loads((EVIDENCE / f'runs/o378_roof_v76_{suffix}/validation.json').read_text())
        assert checks['passed'] and checks['count'] == 64 and checks['edge_count'] == 12
        assert all(c['bitwise_equal_v67'] and c['finite_fp32'] and c['semantic_tolerance_passed'] for c in checks['checks'])
        assert all(c['shape'][0] <= 128 and c['shape'][1] <= 256 and c['shape'][2] == 4096 for c in checks['checks'])
    for check in ('memcheck', 'synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (BUILD / f'{check}.log').read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (BUILD / 'racecheck.log').read_text()
