"""v128 CPU source/ring/gate checks, not GPU performance or race acceptance."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o3_crossk_load_codegen import generated_header, ring_schedule, LIMITS


def test_same_ring_work_and_group_order():
    events=ring_schedule()
    for op in ('copy','load_a','barrier','mma_first_n','mma_last_n',
               'update_first_n','update_last_n','cache_factor'):
        assert [x[1] for x in events if x[0]==op]==list(range(32))
    ready={};loaded=set()
    for event in events:
        op,g,*rest=event
        if op=='copy':
            slot=rest[0]
            if slot in ready:
                old=ready[slot]
                assert events.index(('barrier',old+1))<events.index(event)
                assert events.index(('cache_factor',old,slot))<events.index(event)
            ready[slot]=g
        if op=='load_a':assert ready[rest[0]]==g;loaded.add(g)
        if op=='mma_first_n':assert g in loaded
        if op=='update_last_n' and g<31:
            assert events.index(('load_a',g+1,(g+1)%3))<events.index(event)
    assert [(x[1],x[2]) for x in events if x[0]=='wait'][-2:]==[(30,0),(31,0)]
    with pytest.raises(ValueError):ring_schedule(16)


def test_generation_contract():
    text=generated_header()
    assert 'namespace o3_crossk_load_experiment {' in text
    assert 'sizeof(Storage)==50688' in text
    assert text.count('load_a(0,a00,a01,h00,h01);')==1
    assert text.count('load_a((group+1)%3,a00,a01,h00,h01);')==1
    assert 'load_a(slot,a00,a01,h00,h01);' not in text
    assert text.count('__syncthreads();')==2 # prologue + 31 guarded transitions
    begin=text.index('auto factors=')
    cache=text.index('factors(vi,mi,ni)=s.factor',begin)
    barrier=text.index('__syncthreads();',begin)
    load=text.index('load_a((group+1)%3',begin)
    update=text.index('const int coefficient=factors(vi,mi,ni)',begin)
    assert cache<barrier<load<update
    assert 's.factor[' not in text[barrier:text.index('      } else {',barrier)]
    assert text.count('cute::gemm(')==4
    assert text.count('partial(i)*=16;')==1
    assert LIMITS['max_work_ratio']==1.05 and LIMITS['max_hot_local']==1
    wrapper=(ROOT/'csrc/sm80/roof_o3_crossk_load_probe.cu').read_text()
    assert 'if(flag&6u) return;' in wrapper and 'if(flag&1u)' in wrapper
    assert 'o3_grouped_fallback::o3_body' in wrapper
    assert '__launch_bounds__(128,3)' in wrapper
