"""Exact payload recoding proof; compiler gating is not runtime acceptance."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from probe_o78_balanced_digits_codegen import (
    ATOM_NEW, ATOM_OLD, ROOT, balanced_pair, generated_headers, runtime_gate, scalar_proof)
from probe_o78_eight_chain_codegen import generated_header as eight_header


def test_all_valid_source_codes_and_safe_integer_domain():
    proof = scalar_proof()
    assert proof['source_codes']['o7']['valid_source_codes'] == 254
    assert proof['source_codes']['o8']['valid_source_codes'] == 64
    assert (proof['source_codes']['o7']['q_min'],proof['source_codes']['o7']['q_max']) == (-112,112)
    assert (proof['source_codes']['o8']['q_min'],proof['source_codes']['o8']['q_max']) == (-30,30)
    assert proof['o3_excluded'] and not proof['output_or_MSE_runtime_tested']
    for q in range(-128,120):
        lo, hi = balanced_pair(q)
        assert -8 <= lo <= 7 and -8 <= hi <= 7 and lo+16*hi == q
        for w in range(-8,8):
            assert lo*w + 16*hi*w == q*w
    for q in (-129,120,127,128,1.0):
        with pytest.raises(ValueError):
            balanced_pair(q)


def test_no_new_quantization_scale_or_group_arithmetic():
    headers = generated_headers()
    fallback = headers['o78_balanced_payload_generated_cuh']
    # Low now equals Signed, so restore only the locations that were unsigned.
    original = (ROOT/'csrc/sm80/o78_unsigned_payload_candidate.cuh').read_text()
    assert fallback == original.replace('o78_unsigned_payload_experiment','o78_balanced_payload_experiment').replace(
        'cutlass::uint4b_t','cutlass::int4b_t').replace(ATOM_OLD,ATOM_NEW)
    old_integer = eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    assert headers['o78_balanced_integer_generated_cuh'] == old_integer.replace(
        'o78_eight_chain_experiment','o78_balanced_integer_experiment').replace('O78::','O78Balanced::').replace(
        'cutlass::uint4b_t','cutlass::int4b_t').replace(ATOM_OLD,ATOM_NEW)
    converter = headers['o78_balanced_conversion_generated_cuh']
    restored = converter.replace('balanced_row_fused_probe','row_fused_probe').replace('unsigned(q+8)','unsigned(q)')
    assert restored == (ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    assert 'square+=unsigned(q*q)' in converter  # square/norm refer to original integer q
    assert ATOM_OLD not in fallback and ATOM_OLD not in headers['o78_balanced_integer_generated_cuh']


def test_unchanged_type_instruction_work_does_not_justify_GPU_timing():
    loop = dict(kind='integer',max_live_gpr=166,static_instructions=383,
                opcode_counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16})
    control = dict(allocated_gpr=168,loops=[loop])
    candidate = deepcopy(control)
    candidate['loops'][0]['opcode_counts'] = {'IMMA.16864.S4.S4':64,'LDSM.16.M88.4':16}
    assert not runtime_gate(control,candidate)['worth_runtime_validation']
    candidate['loops'][0]['static_instructions'] -= 8
    assert runtime_gate(control,candidate)['worth_runtime_validation']
    candidate['loops'][0]['opcode_counts']['LDL.64'] = 1
    assert not runtime_gate(control,candidate)['worth_runtime_validation']
    candidate['loops'][0]['opcode_counts'].pop('LDL.64')
    candidate['allocated_gpr'] = 169
    assert not runtime_gate(control,candidate)['worth_runtime_validation']
    candidate['allocated_gpr'] = 128
    candidate['loops'][0]['static_instructions'] = 383
    assert runtime_gate(control,candidate)['register_capacity_improved']


def test_O3_and_default_untouched_and_fallback_recoded_too():
    wrapper = (ROOT/'csrc/sm80/roof_o78_balanced_digits_probe.cu').read_text()
    assert 'O78Balanced::o3_body' in wrapper
    assert 'o78_balanced_integer_experiment::body' in wrapper
    assert 'adangel_roof_o3' not in wrapper and 'PYBIND11' not in wrapper
