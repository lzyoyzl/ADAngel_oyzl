"""Parser/launch gates, not fabricated GPU evidence for renamed fixtures."""
import ast
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_roof_scale_ncu import analyze

E = ROOT / 'docs/evidence/a100_o378_roof_v80_ncu/reports/o378_roof_v80_ncu'


@pytest.mark.parametrize('variant', ['o3', 'o7', 'o8'])
def test_grouped_alias_preserves_exact_fullk_work_accounting(variant):
    # Deliberately renamed archived parser fixture. Actual v90 captures are
    # separately SHA/SASS checked; these substitutions are not GPU results.
    kind = 'o3' if variant == 'o3' else 'o78'
    old = f'adangel_roof_{kind}_eight_chain_candidate'
    new = f'adangel_roof_{kind}_grouped_cta_candidate'
    tune = 54 if variant == 'o3' else 59
    raw = (E / f'{variant}_raw.csv').read_text()
    source = (E / f'{variant}_source_sass.csv').read_text()
    reference = analyze(raw, source, tune, variant, True, True,
                        expected_symbol=old, fullk_integer=True)
    changed = analyze(raw.replace(old, new), source.replace(old, new),
                      tune, variant, True, True,
                      expected_symbol=new, fullk_integer=True)
    for key in ('opcodes', 'dynamic_instructions', 'source_memory_work',
                'pc_sampling', 'optimistic_fixed_work_lower_bound_ms'):
        assert changed[key] == reference[key]
    with pytest.raises(ValueError, match='full-K'):
        analyze(raw.replace(old, new), source.replace(old, new), tune, variant,
                expected_symbol=new, fullk_integer=False)


@pytest.mark.parametrize('variant,tune,symbol', [
    ('o3', 54, 'adangel_roof_o78_grouped_cta_candidate'),
    ('o7', 59, 'adangel_roof_o3_grouped_cta_candidate'),
    ('o8', 54, 'adangel_roof_o78_grouped_cta_candidate'),
    ('o3', 54, 'adangel_roof_o3_grouped_cta_candidate_probe'),
])
def test_wrong_variant_tune_or_probe_cannot_use_grouped_budget(variant, tune, symbol):
    with pytest.raises(ValueError):
        analyze('', '', tune, variant, fullk_integer=True, expected_symbol=symbol)


def test_single_requested_capture_uses_warmup_not_event_timing():
    profile = (ROOT / 'scripts/profile_grouped_cta_kernel.py').read_text()
    tree = ast.parse(profile)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and ast.unparse(node.func) in ('driver.run_four', 'driver.run')]
    target = [node for node in calls if any(isinstance(arg, ast.Attribute)
              and ast.unparse(arg) == 'a.policy' for arg in node.args)]
    assert len(target) == 2
    assert all([ast.literal_eval(arg) for arg in node.args[-3:]] == [50, 1, 2]
               for node in target)
    assert "filtered_launch_skip=51 if a.policy==0 else 50" in profile
    capture = (ROOT / 'scripts/run_grouped_cta_ncu.py').read_text()
    assert "'--clock-control','none'" in capture
    assert "'--launch-count','1'" in capture
    assert "if counts!=normalized_counts(expected['opcode_counts'])" in capture
    assert 'static_fingerprint_verified=True' in capture
    assert 'new_performance_result=False' in capture
