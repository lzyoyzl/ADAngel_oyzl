"""Replay v115 cost/data evidence; never run NVCC or a GPU candidate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_a100_codegen import compare
from probe_dp2a_recompose import data_gate, entries, isa_gate, SYMBOL as DOT
from probe_o3_dp2a_codegen import (CONTROL, SYMBOL, dot_liveness, gate,
                                 generated_header)
from probe_roof_fullk_integer_codegen import static_entries

BASE = ROOT / 'docs/evidence/a100_o378_roof_v115'
COST = BASE / 'reports/o378_roof_v115_dp2a_cost'
FULL = BASE / 'reports/o378_roof_v115_o3_dp2a_codegen'
OLD = ROOT / 'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen'
EXTENSION = '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def test_frozen_files_and_historical_sources_not_current_mutable_tree():
    index = json.loads((BASE / 'index.json').read_text())
    assert index['artifact_count'] == len(index['artifacts']) == 21
    for name, sha in index['artifacts'].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == sha, name
    assert not any(index[k] for k in ('candidate_GPU_launched', 'new_performance_result',
                                     'new_MSE_result', 'production_default_changed'))
    for directory, key in ((COST, 'scalar_source_commit'), (FULL, 'full_source_commit')):
        receipt = json.loads((directory / 'analysis.json').read_text())
        assert receipt['source_commit'] == index[key]
        for name, sha in receipt['source_hashes'].items():
            original = subprocess.check_output(['git', 'show', index[key] + ':' + name], cwd=ROOT)
            assert hashlib.sha256(original).hexdigest() == sha, name
        for name, sha in receipt['artifact_sha256'].items():
            if name.endswith('.cubin'):
                assert (directory / name).relative_to(BASE).as_posix() in index['binary_files_not_committed']
            else:
                assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == sha, name
        assert receipt['formal_extension_sha256_before'] == receipt['formal_extension_sha256_after'] == EXTENSION


def test_full24_original_factor_snapshots_and_scalar_lowering_replayed():
    rows, summary = data_gate()
    saved = json.loads((COST / 'analysis.json').read_text())
    assert rows == [json.loads(s) for s in (COST / 'data_rows.jsonl').read_text().splitlines()]
    assert len(rows) == 72 and summary == saved['data_summary']
    assert summary[0]['samples'] == 24
    assert summary[0]['mean_coverage'] == pytest.approx(0.8072916666666666)
    assert all('conditional' in x['coverage_kind'] for x in summary[1:])
    parsed = entries((COST / 'dp2a_recompose.sass').read_text())
    assert parsed == saved['entries']
    assert isa_gate(parsed[DOT]) == saved['isa_gate']
    assert saved['isa_gate']['passed'] and saved['proceed_with_O3_GEMM']
    assert not saved['candidate_GPU_launched'] and not saved['new_performance_result']


def test_complete_integer_hot_loop_gate_replayed_from_liveness_not_pass_label():
    receipt = json.loads((FULL / 'analysis.json').read_text())
    old = json.loads((OLD / 'codegen.json').read_text())['liveness'][CONTROL]
    live = dot_liveness((FULL / 'liveness.txt').read_text())
    assert live == receipt['liveness']
    actual = gate(old, live)
    assert actual == receipt['compile_gate']
    assert not actual['passed'] and not actual['hot_local_not_worse']
    assert not actual['work_growth_at_most5pct']
    assert live['allocated_gpr'] == 168
    assert live['loop']['static_instructions'] == 340
    assert actual['static_work_ratio'] == pytest.approx(340 / 323)
    ops = live['loop']['opcode_counts']
    assert ops['LDL'] == 3 and ops['PRMT'] == ops['IDP.2A.LO.S16.U8'] == 64
    assert ops['IMMA.16864.S4.S4'] == ops['IMMA.16864.U4.S4'] == 32
    assert ops['LDSM.16.M88.4'] == 16 and not ops.get('I2F')


def test_same_entry_native_math_and_original_control_encoding():
    receipt = json.loads((FULL / 'analysis.json').read_text())
    sass = (FULL / 'o3_dp2a.sass').read_text()
    symbols = {SYMBOL, CONTROL}
    parsed = static_entries(sass, '^(?:' + '|'.join(symbols) + ')$', symbols)
    assert parsed == receipt['entries']
    for r in parsed.values():
        assert r['native_u4_s4'] and r['native_s4_s4'] and not r['int8_mma']
        assert r['all_copies_bypass_l1']
    control = compare((OLD / 'o3_grouped_cta.sass').read_text(), sass, '^' + CONTROL + '$')
    assert control == receipt['control_comparison'] and control['passed']
    original_header = (OLD / 'o3_grouped_cta_generated.cuh').read_text()
    assert generated_header(original_header) == (FULL / 'o3_dp2a_generated.cuh').read_text()


def test_no_new_latency_mse_or_gpu_safety_claim():
    receipt = json.loads((FULL / 'analysis.json').read_text())
    assert not any(receipt[k] for k in ('candidate_GPU_launched', 'new_performance_result',
                                      'new_MSE_result', 'production_default_changed'))
    assert receipt['metadata_cost_must_count_in_Cold']
    assert '14 passed' in (BASE / 'reports/o378_v115_dp2a_unit.log').read_text()
    assert not list(BASE.rglob('*.cubin'))
