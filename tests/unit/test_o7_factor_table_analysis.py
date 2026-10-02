from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_o7_factor_table_codegen import analyze, CONTROL, CANDIDATE


def synthetic():
    lines = []
    for symbol in (CONTROL, CANDIDATE):
        lines.append('Function : ' + symbol)
        for index, kind in enumerate(('integer', 'fp32', 'table') if symbol == CANDIDATE else ('integer', 'fp32')):
            lo = 0x100 + 0x1000 * index
            for i in range(64):
                lines.append(f'/*{lo+16*i:04x}*/ IMMA.16864.S4.S4 R4,R8,R12,R4;')
            text = {'integer': 'IMAD R4,R5,R6,R7', 'fp32': 'I2F R4,R4', 'table': 'STS [R4],R5'}[kind]
            lines.extend([f'/*{lo+1024:04x}*/ {text};', f'/*{lo+1040:04x}*/ BRA 0x{lo:x};'])
    return '\n'.join(lines)


def test_loop_cost_classification():
    rows = analyze(synthetic())['entries']
    assert len(rows) == 2
    assert [r['kind'] for r in rows[1]['loops']] == ['original_integer', 'fp32_fallback', 'factor_table']
    assert rows[1]['loops'][2]['scalar_shared_stores'] == 1
    assert rows[0]['loops'][0]['imad_family'] == 1
    assert all(r['instructions'] == 66 and r['imma'] == 64 for e in rows for r in e['loops'])


@pytest.mark.parametrize('a,b', [('STS', 'MOV'), (CONTROL, 'probe'), ('IMMA.', 'MMA.')])
def test_missing_ambiguous_or_unrelated_entry_rejected(a, b):
    with pytest.raises(ValueError):
        analyze(synthetic().replace(a, b))
