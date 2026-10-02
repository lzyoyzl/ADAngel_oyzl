"""Fail-closed parser checks for static binary liveness diagnosis."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from inspect_o78_register_liveness import analyze, SYMBOL


def synthetic():
    lines = [f'//--------- .text.{SYMBOL} ---------', '.sectioninfo @"SHI_REGISTERS=168"']
    for index, offset in enumerate((0x100, 0x1000)):
        lines.append(f'.L_x_{index}:')
        for i in range(64):
            lines.append(f'/*{offset+16*i:04x}*/ IMMA.16864.U4.S4 R4, R8, R12, R4 ; // | 160 | 1 |')
        pc = offset + 16 * 64
        if index:
            lines.append(f'/*{pc:04x}*/ I2F R4, R4 ; // | 165 | 1 |')
            pc += 16
        lines.append(f'/*{pc:04x}*/ BRA `(.L_x_{index}) ; // | 90 | 1 |')
    lines.extend(['.L_tail:', '/*2000*/ BRA `(.L_tail); // | 1 | |'])
    return '\n'.join(lines)


def test_two_loops_and_register_gate():
    got = analyze(synthetic())
    assert got['allocated_gpr'] == 168 and got['function_max_live_gpr'] == 165
    assert got['register_only_four_cta_limit_per_thread'] == 128
    assert [x['max_live_gpr'] for x in got['loops']] == [160, 165]
    assert [x['kind'] for x in got['loops']] == ['integer', 'fp32_fallback']
    assert [x['static_instructions'] for x in got['loops']] == [65, 66]


@pytest.mark.parametrize('mutation', ['symbol', 'registers', 'liveness', 'label', 'mma', 'fallback'])
def test_missing_or_ambiguous_evidence_fails(mutation):
    text = synthetic()
    changes = {'symbol': (SYMBOL, 'probe'), 'registers': ('SHI_REGISTERS=', 'UNKNOWN='),
               'liveness': ('// |', '// ?'), 'label': ('`(.L_x_0)', '`(.L_missing)'),
               'mma': ('IMMA.', 'MMA.'), 'fallback': ('I2F R4', 'MOV R4')}
    text = text.replace(*changes[mutation])
    with pytest.raises(ValueError):
        analyze(text)
