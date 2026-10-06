"""Frozen compiler-only evidence; do not mistake it for a GPU/MSE result."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_balanced_digits_codegen import summarize
from probe_o78_balanced_digits_codegen import CONTROL,SYMBOL

EVIDENCE = ROOT/'docs/evidence/a100_o378_roof_v100'


def test_exact_proof_native_instruction_budget_and_stop_gate():
    result = summarize(EVIDENCE/'reports/o378_roof_v100_codegen')
    saved = json.loads((EVIDENCE/'analysis.json').read_text())
    assert result == saved
    assert result['source_commit'] == '264cd712b77bd0f26aed767830535f60d33ad0a2'
    assert result['old_allocated_gpr'] == result['candidate_allocated_gpr'] == 168
    assert result['rows']['integer']['old_instructions'] == result['rows']['integer']['candidate_instructions'] == 383
    assert result['rows']['fp32_fallback']['old_instructions'] == result['rows']['fp32_fallback']['candidate_instructions'] == 440
    for row in result['rows'].values():
        assert row['old_native_u4s4'] == row['old_native_s4s4'] == 32
        assert row['candidate_native_s4s4'] == 64
        assert row['old_ldsm'] == row['candidate_ldsm'] == 16
        assert row['old_hot_local_instructions'] == row['candidate_hot_local_instructions'] == 0
    assert not result['runtime_gate']['worth_runtime_validation']
    assert result['runtime_gate']['integer_instructions_saved'] == 0
    assert not result['performance_measured'] and not result['MSE_measured']
    assert result['control_machine_encoding_passed']
    assert all(row['equal_except_low_MMA_signedness'] for row in result['sequence_checks'].values())
    assert not result['production_default_changed'] and not result['o3_changed']


def test_same_entry_PTX_signed_path_and_unchanged_old_unsigned_control():
    ptx = (EVIDENCE/'reports/o378_roof_v100_codegen/o78_balanced_digits.ptx').read_text()
    for symbol in (CONTROL,SYMBOL):
        body = next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
                    if b.startswith('.visible .entry '+symbol+'('))
        assert '.s32.s4.s4.s32' in body and 'cp.async.cg.shared.global' in body
        assert ('.s32.u4.s4.s32' in body) == (symbol == CONTROL)
