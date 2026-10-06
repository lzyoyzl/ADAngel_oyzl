"""Actual full24 raw timings/codegen for the rejected v91 candidate."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_grouped_cta_retest import read_run
from benchmark_o78_coefficient_probe import summarize
from compare_a100_codegen import compare
from probe_o78_address_remat_codegen import CONTROLS, SYMBOL, STEM, generated_header, opcode_count
from probe_roof_fullk_integer_codegen import static_entries

E = ROOT / 'docs/evidence/a100_o378_roof_v91'
C = E / 'reports/o378_roof_v91_codegen'
R = E / 'runs/o378_roof_v91_trace24'


def test_native_copy_work_and_exact_control_encodings():
    receipt = json.loads((C / 'codegen.json').read_text())
    for name, sha in receipt['artifact_sha256'].items():
        if name.endswith('.cubin'):
            continue  # Preserved in the complete archive, deliberately not Git.
        assert hashlib.sha256((C / name).read_bytes()).hexdigest() == sha
    for name, sha in receipt['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    assert (C / (STEM + '_generated.cuh')).read_text() == generated_header()
    symbols = {*CONTROLS, SYMBOL}
    sass = (C / (STEM + '.sass')).read_text()
    assert static_entries(sass, '^(?:' + '|'.join(sorted(symbols)) + ')$', symbols) == receipt['entries']
    prior = ROOT / 'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o78_codegen/o78_grouped_cta.sass'
    for symbol in CONTROLS:
        assert compare(prior.read_text(), sass, '^' + symbol + '$')['passed']
    for symbol in symbols:
        ptx = (C / (STEM + '.ptx')).read_text()
        block = next(b for b in re.split(r'(?=\.visible \.entry )', ptx)
                     if b.startswith('.visible .entry ' + symbol + '('))
        assert all(token in block for token in ('cp.async.cg.shared.global', '.s32.u4.s4.s32', '.s32.s4.s4.s32'))
    candidate = receipt['liveness'][SYMBOL]
    loop = next(l for l in candidate['loops'] if l['kind'] == 'integer')
    assert candidate['allocated_gpr'] == 168 and loop['max_live_gpr'] == 163
    assert loop['static_instructions'] == 379 and opcode_count(loop, 'LDL') == 0
    assert opcode_count(loop, 'STL') == 0
    assert receipt['worth_runtime_validation'] and not receipt['production_default_changed']


def test_full24_statistics_numerical_identity_and_negative_gain():
    run = read_run(R)  # Checks complete24, repeats, every raw mean/median/CV, MSE.
    rows = [json.loads(line) for line in (R / 'results.jsonl').read_text().splitlines()]
    stored = json.loads((R / 'summary.json').read_text())
    assert len(rows) == 288 and run['args']['warmup'] == 1000
    assert run['args']['rounds'] == 3 and run['args']['repeats'] == 200 and run['args']['inner'] == 100
    assert summarize(rows, ('compute_only',)) == stored['records']
    assert {r['mode'] for r in rows} == {'compute_only'}
    for r in rows:
        assert r['finite_fp32'] and r['MSE_regression_passed'] and r['metadata_exact']
        assert r['bitwise_equal_v67'] and r['mse_vs_v67'] == 0
        assert r['a_copy_address_rematerialization']
    new = [r for r in run['summary'] if r['policy'] == 1]
    assert all(r['paired_speedup_ci95'][1] < 1 for r in new)
    by_variant = {r['variant']: r for r in new}
    assert by_variant['o7']['paired_speedup'] == pytest.approx(0.9872068494904378)
    assert by_variant['o8']['paired_speedup'] == pytest.approx(0.9873683863537717)
    assert {r['variant']: r['selected_cv_failed_records'] for r in new} == {'o7': 1, 'o8': 1}
    fallback = {(r['sample_id'], r['variant'], r['guard']['fallback_ctas'])
                for r in rows if r['guard']['fallback_ctas']}
    assert fallback == {('layer_24_o_proj', 'o8', 12)}
    assert not stored['production_default_changed'] and stored['no_filtering']


def test_validation_scope_and_preserved_extension():
    check = json.loads((R / 'validation.json').read_text())
    assert check['passed'] and check['count'] == 64 and check['edge_count'] == 12
    assert check['grouped_coordinate_count'] == 8
    env = json.loads((R / 'environment.json').read_text())
    assert env['extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert not env['production_default_changed']
    assert env['codegen']['encoded_v78_runtime_control']['passed']
    assert all(c['finite_fp32'] and c['semantic_tolerance_passed'] for c in check['checks'])
