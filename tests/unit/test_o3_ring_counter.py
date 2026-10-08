from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_grouped_cta_codegen import generated_headers
from probe_o3_ring_counter_codegen import generated_header, schedule, cost_gate


@pytest.mark.parametrize('groups', (2, 3, 4, 31, 32, 33, 1024))
def test_recurrence_matches_modulo_and_does_not_overwrite_current_readers(groups):
    resident = {0: 0, 1: 1}
    for group, slot, future in schedule(groups):
        assert slot == group % 3 and resident[slot] == group
        if future is not None:
            target, write_slot = future
            assert (target, write_slot) == (group + 2, (group + 2) % 3)
            assert write_slot != slot
            assert write_slot not in resident or resident[write_slot] < group
            resident[write_slot] = target


def test_only_ring_selection_changes_no_math_or_layout_rewrite():
    text = generated_header()
    restored = text.replace('o3_ring_counter_experiment', 'o3_grouped_cta_experiment')
    restored = restored.replace('  int slot=0;\n  #pragma unroll 1\n  for(int group=0;group<groups;++group) {',
        '  for(int group=0;group<groups;++group) {\n    const int slot=group%3;')
    restored = restored.replace('prefetch(s,slot==0 ? 2 : slot-1,group+2,a,w,ws,m,n,k);',
        'prefetch(s,(group+2)%3,group+2,a,w,ws,m,n,k);')
    restored = restored.replace('    slot=slot==2 ? 0 : slot+1;\n', '')
    assert restored == generated_headers('o3')[0]
    assert '%3' not in text and text.count('for(int group=') == 1
    wrapper = (ROOT / 'csrc/sm80/roof_o3_ring_counter_probe.cu').read_text()
    original = (ROOT / 'csrc/sm80/roof_o3_stage_cycle_probe.cu').read_text()
    # Public signature, logical guard and original fallback remain identical.
    assert wrapper[wrapper.index('extern "C"'):].replace('ring_counter', 'stage_cycle') == original[original.index('extern "C"'):]


def test_cost_gate_does_not_confuse_static_change_with_performance():
    counts = {'IMMA.16864.S4.S4': 32, 'IMMA.16864.U4.S4': 32, 'LDSM.16.M88.4': 16,
              'LDGSTS.E.BYPASS.128': 9, 'LDGDEPBAR': 1, 'BAR.SYNC.DEFER_BLOCKING': 1}
    old = dict(allocated_gpr=168, loops=[dict(kind='integer', static_instructions=323, opcode_counts=counts)])
    new = deepcopy(old)
    new['loops'][0]['static_instructions'] = 310
    assert cost_gate(old, new)['passed']
    new['loops'][0]['static_instructions'] = 314
    assert not cost_gate(old, new)['passed']
    new['loops'][0]['static_instructions'] = 310
    new['allocated_gpr'] = 176
    assert not cost_gate(old, new)['passed']
    new['allocated_gpr'] = 168
    new['loops'][0]['opcode_counts']['LDL'] = 1
    assert not cost_gate(old, new)['passed']
    del new['loops'][0]['opcode_counts']['LDL']
    new['loops'][0]['opcode_counts']['IMMA.16864.U4.S4'] = 31
    assert not cost_gate(old, new)['passed']


def test_short_or_invalid_ring_rejected():
    for groups in (0, 1, -1, 32.5):
        with pytest.raises(ValueError):
            schedule(groups)
