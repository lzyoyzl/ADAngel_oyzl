"""Recompute actual v90 captures; parser fixtures cannot establish these data."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_grouped_cta_ncu import analyze_capture

E = ROOT / 'docs/evidence/a100_o378_roof_v90_ncu/reports/o378_roof_v90_ncu'
EXPECTED = {
    'o3_0': (86802432, 647168, 32768),
    'o3_1': (88231936, 8192, 8192),
    'o7_0': (105521152, 0, 0),
    'o7_1': (105111552, 516096, 16384),
}


@pytest.mark.parametrize('tag', sorted(EXPECTED))
def test_real_capture_hash_fingerprint_work_and_output(tag):
    archive = json.loads((E / 'analysis.json').read_text())
    paths = [E / (tag + '_raw.csv'), E / (tag + '_source_sass.csv'), E / tag / 'receipt.json']
    for path in paths:
        relative = path.relative_to(E).as_posix()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == archive['input_sha256'][relative]
    receipt = json.loads(paths[2].read_text())
    result = analyze_capture(paths[0].read_text(encoding='utf-8-sig'),
                             paths[1].read_text(encoding='utf-8-sig'), receipt)
    expected = next(row for row in archive['rows']
                    if row['variant'] == receipt['variant'] and row['policy'] == receipt['policy'])
    assert result == expected
    assert (result['dynamic_instructions'], result['opcodes'].get('LDL', 0),
            result['opcodes'].get('STL', 0)) == EXPECTED[tag]
    assert result['opcodes']['IMMA'] == 16777216
    assert result['opcodes']['LDSM'] == 4194304
    assert result['optimistic_fixed_work_lower_bound_ms'] == pytest.approx(0.22034693984764905)
    assert result['mse_vs_previous_best'] == 0
    assert result['static_fingerprint_verified']
    assert receipt['numerical_checks_passed'] and receipt['bitwise_previous_best']


def test_complete_command_and_identity_archive_is_diagnostic_only():
    archive = json.loads((E / 'analysis.json').read_text())
    assert len(archive['rows']) == 4
    assert {row['variant'] for row in archive['rows']} == {'o3', 'o7'}
    assert not archive['new_performance_result'] and not archive['production_default_changed']
    commands = json.loads((E / 'commands.json').read_text())
    captures = [entry['command'] for entry in commands if '--set' in entry['command']]
    assert len(captures) == 4
    for command in captures:
        assert command[command.index('--set') + 1] == 'full'
        assert command[command.index('--clock-control') + 1] == 'none'
        assert command[command.index('--launch-count') + 1] == '1'
    for tag in ('o3_0', 'o3_1'):
        assert (E / tag / 'build/grouped_build.json').is_file()
