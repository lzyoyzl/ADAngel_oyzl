from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_o78_cute_traversal_codegen import generated_header, loop_summary
from probe_o78_eight_chain_codegen import generated_header as eight_header


def test_only_four_mma_phase_traversals_change():
    source = (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    control = eight_header(source)
    result = generated_header(source).replace('o78_cute_traversal_experiment', 'o78_eight_chain_experiment')
    marker = '      // Eight independent chains:'
    assert result.split(marker)[0] == control.split(marker)[0]
    tail = '          auto full_ni=nb*cute::_4{}+ni;'
    assert result.split(tail)[1] == control.split(tail)[1]
    assert result.count('cute::gemm(') == 4
    assert result.count('cute::copy(') == control.count('cute::copy(') == 6
    assert result.count('partial(i)*=16') == 1
    for atom, a, b in (('HA', 'h0', 'b0'), ('HA', 'h1', 'b1'), ('LA', 'a0', 'b0'), ('LA', 'a1', 'b1')):
        assert f'cute::gemm({atom}{{}},partial,{a}(cute::_,cute::_,cute::_0{{}}),' in result
        assert f'{b}(cute::_,cute::_,cute::_0{{}}),partial);' in result
    with pytest.raises(ValueError): generated_header(source.replace('auto mi', 'auto changed'))


def test_serpentine_changes_independent_output_order_not_integer_math():
    # Every coordinate is visited once per phase; individual K64 order stays fixed.
    old = [(m, n) for m in range(2) for n in range(4)]
    snake = [(1-m if n % 2 else m, n) for n in range(4) for m in range(2)]
    assert sorted(snake) == old and len(set(snake)) == 8
    rng = np.random.default_rng(20261003)
    a = rng.integers(-128, 128, (2, 128), dtype=np.int64)
    b = rng.integers(-8, 8, (4, 128), dtype=np.int64)
    def compute(order):
        out = np.zeros((2, 4), dtype=np.int64)
        for plane, half in (('high', 0), ('high', 1), ('low', 0), ('low', 1)):
            if plane == 'low' and half == 0: out *= 16
            for m, n in order:
                av = a[m, half*64:(half+1)*64]
                av = av // 16 if plane == 'high' else av & 15
                out[m, n] += av @ b[n, half*64:(half+1)*64]
        return out
    assert np.array_equal(compute(old), compute(snake))
    assert np.array_equal(compute(snake), a @ b.T)


def test_reuse_audit_uses_only_target_integer_loop():
    sass = '''Function : target
/*0010*/ IMMA.16864.S4.S4 R8, R0.reuse.ROW, R4.reuse.COL, RZ ;
/*0020*/ IMMA.16864.U4.S4 R8, R0.ROW, R4.COL, R8 ;
/*0030*/ IMMA.16864.S4.S4 R8, R0.reuse.ROW, R4.reuse.COL, RZ ;
Function : other
/*0010*/ IMMA.16864.S4.S4 R8, R0.reuse.ROW, R4.reuse.COL, RZ ;
'''
    live = dict(loops=[dict(kind='integer', begin_pc='0x10', end_pc='0x20', opcode_counts={})])
    result = loop_summary(sass, 'target', live)
    assert result['mma_count'] == 2 and result['operand_reuse_markers'] == 2
    assert 'not_measured' in result['scope']


def test_online_preparation_and_timing_contract_unchanged():
    from benchmark_o78_cute_traversal import timing_contract
    from benchmark_o78_eight_chain_probe import timing_contract as old
    for mode in ('conversion_only', 'compute_only', 'cold', 'steady_state'):
        result = timing_contract(mode, 100)
        assert not result.pop('new_preparation_or_layout')
        result['comparison'] = old(mode, 100)['comparison']
        assert result == old(mode, 100)


def test_json_histogram_roundtrip_keeps_all_schedule_values():
    from benchmark_o78_cute_traversal import json_canonical
    value = dict(active_count_histogram={0: 2, 8: 9}, mma=[dict(pc='0x1230', stage=1)])
    saved = dict(active_count_histogram={'0': 2, '8': 9}, mma=[dict(pc='0x1230', stage=1)])
    assert json_canonical(value) == saved
    saved['mma'][0]['stage'] = 2
    assert json_canonical(value) != saved
