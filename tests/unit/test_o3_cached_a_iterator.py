"""v134 retains the old predeclared gate and changes only A address production."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o3_cached_a_iterator_codegen import NEW,OLD,LIMITS,generated_header
from probe_o3_crossk_load_codegen import generated_header as crossk_header,ring_schedule


def test_only_address_block_changes_and_original_gate_is_not_relaxed():
    old=crossk_header();new=generated_header()
    assert new.replace(NEW,OLD).replace('o3_cached_a_iterator_experiment','o3_crossk_load_experiment')==old
    assert LIMITS['max_work_ratio']==1.05 and LIMITS['max_allocated_gpr']==168
    assert new.count('__shfl_sync')==1 # compile-time 2x2, prologue only
    assert new.index('__shfl_sync')<new.index('for(int group=0;group<groups;++group)')
    assert '__cvta_generic_to_shared(&src(mi))' in new
    assert 's.activation_factors' not in new
    assert 'SM80_16x8x64_S32U4S4S32_TN' in new and 'SM80_16x8x64_S32S4S4S32_TN' in new
    assert new.count('__syncthreads()')==old.count('__syncthreads()')
    assert len([event for event in ring_schedule() if event[0]=='barrier'])==32


def test_public_entry_still_guards_and_uses_original_fallback():
    s=(ROOT/'csrc/sm80/roof_o3_cached_a_iterator_probe.cu').read_text()
    for text in ('if(flag&6u) return','if(flag&1u)','o3_grouped_fallback::o3_body',
                 '__launch_bounds__(128,3)','o3_cached_a_iterator_experiment::body'):
        assert text in s
    assert 'int8' not in s.lower().replace('uint8_t','')
