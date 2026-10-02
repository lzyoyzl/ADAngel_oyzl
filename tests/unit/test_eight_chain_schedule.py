from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from inspect_eight_chain_schedule import trace


def fixture(width=8, mutation=None):
    ops = []
    for batch in range(16 // width):
        for stage, kind in enumerate(('S4', 'S4', 'U4', 'U4')):
            for chain in range(width):
                dest = 32 + chain * 4
                c = 'RZ' if stage == 0 else f'R{dest}'
                if stage == 2:
                    for i in range(4):
                        ops.append(f'SHF.L.U32 R{dest+i}, R{dest+i}, 0x4, RZ')
                ops.append(f'IMMA.16864.{kind}.S4 R{dest}, R0.ROW, R8.COL, {c}')
    if mutation:
        mutation(ops)
    lines = ['Function : exact'] + [f'/*{i*16:04x}*/ {op} ;' for i, op in enumerate(ops)]
    # Wrong-function instructions must never affect the exact entry.
    lines += ['Function : exact_suffix', '/*0000*/ IMMA.16864.U4.S4 R32, R0.ROW, R8.COL, RZ ;']
    live = dict(loops=[dict(kind='integer', begin_pc='0x0', end_pc=hex((len(ops)-1)*16))])
    return '\n'.join(lines), live


@pytest.mark.parametrize('width', [1, 4, 8, 16])
def test_actual_dependency_width_not_just_opcode_count(width):
    sass, live = fixture(width)
    result = trace(sass, 'exact', live)
    assert result['peak_started_not_finished_chains'] == width
    assert result['total_mma'] == 64 and result['chains_per_group'] == 16
    assert all([m['kind'] for m in c['mma']] == ['S4', 'S4', 'U4', 'U4'] for c in result['chains'])
    assert sum(result['active_count_histogram'].values()) == 64


@pytest.mark.parametrize('instruction', [
    'LDSM.16.M88.4 R32, [R2]',  # Clobbers all four C registers.
    'LDS.64 R32, [R2]',        # Clobbers only half of the C fragment.
    'MOV R32, R36',            # Mixes two different partial chains.
    'UNKNOWN R99, R32',        # Unknown register writer is not silently ignored.
])
def test_clobbered_mixed_or_unknown_dependency_is_rejected(instruction):
    sass, live = fixture(mutation=lambda ops: ops.insert(8, instruction))
    with pytest.raises(ValueError):
        trace(sass, 'exact', live)


def test_aliasing_move_and_in_place_math_preserve_chain():
    def move(ops):
        ops[8:8] = [f'MOV R{120+i}, R{32+i}' for i in range(4)]
        ops[12] = ops[12].replace(', R32', ', R120')
    sass, live = fixture(mutation=move)
    assert trace(sass, 'exact', live)['peak_started_not_finished_chains'] == 8


def test_incomplete_or_unsigned_start_rejected():
    for mutation in (lambda ops: ops.pop(), lambda ops: ops.__setitem__(0, ops[0].replace('.S4.S4', '.U4.S4'))):
        sass, live = fixture(mutation=mutation)
        with pytest.raises(ValueError):
            trace(sass, 'exact', live)


def test_nop_spelling_aliases_are_summed_not_overwritten():
    from run_eight_chain_ncu import normalized_counts
    result = normalized_counts({'NOP': 2, 'NOP;': 7, 'IMMA': 64})
    assert result == {'NOP': 9, 'IMMA': 64}
    assert sum(result.values()) == 73
