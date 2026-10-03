"""Check the one pipeline change before device compilation and runtime tests."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_o78_eight_chain_codegen import generated_header as old_header
from probe_o78_three_stage_codegen import generated_header, NEW_PIPELINE, OLD_PIPELINE


def test_generated_candidate_changes_storage_and_pipeline_not_math():
    source = (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    old, new = old_header(source), generated_header(source)
    assert 'sizeof(Storage)==51456' in new
    assert new.count(NEW_PIPELINE) == 1 and OLD_PIPELINE not in new
    start = '    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),ld0);'
    # All MMA, scale, accumulator and epilogue operations retain exact source.
    assert new[new.index(start):].replace('o78_three_stage_experiment', 'o78_eight_chain_experiment') == old[old.index(start):]
    with pytest.raises(ValueError):
        generated_header(source.replace('const int slot=group%2;', 'const int slot=1;'))


def test_three_slot_schedule_visibility_reuse_and_tail_drain():
    # Model oldest-completes semantics, not GPU execution. The compiled kernel
    # itself remains guarded for32 groups; extra lengths stress ring logic only.
    for groups in range(2, 36):
        slots, pending, completed = {}, [0, 1], set()
        slots[0], slots[1] = 0, 1
        consumed = []
        for g in range(groups):
            keep = 1 if g + 1 < groups else 0
            while len(pending) > keep:
                completed.add(pending.pop(0))
            assert g in completed and slots[g % 3] == g
            # CTA barrier completes all reads from group g-1, so this slot is
            # reusable before issuing group g+2, without overwriting g or g+1.
            if g + 2 < groups:
                slot = (g + 2) % 3
                assert slot not in slots or slots[slot] in consumed
                slots[slot] = g + 2
                pending.append(g + 2)
            consumed.append(g)
        assert consumed == list(range(groups)) and not pending


def test_guard_and_fallback_unchanged_and_full_modes_share_timer():
    wrapper = (ROOT / 'csrc/sm80/roof_o78_three_stage_probe.cu').read_text()
    assert '__launch_bounds__(128,3)' in wrapper
    assert 'status[blockIdx.y*(n/128)+blockIdx.x]' in wrapper
    assert 'if(flag>1u) return;' in wrapper
    assert 'true,true,6,false,false,true,false,2>' in wrapper
    driver = (ROOT / 'scripts/benchmark_o78_three_stage.py').read_text()
    assert 'class Driver(eight.Driver)' in driver and 'def run(' not in driver
    assert '51456' in driver and 'values[3] < 3' in driver
    assert 'fallback_pipeline_stages=2' in driver
