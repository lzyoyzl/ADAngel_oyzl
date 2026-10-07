"""v127 host copy coverage and unchanged math; not a GPU synchronization proof."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_o78_activation_panel_codegen as p


def test_exact_nonoverlapping_all_group_copy_coverage():
    for m in (64,128,4096):
        for y in (0,m//64-1):
            seen=set()
            for t in range(128):
                for chunk in range(4):
                    g,r,src=p.mapping(t,chunk,m,y)
                    assert 0<=g<32 and r%4==0 and src%4==0
                    for j in range(4):
                        assert (g,r+j) not in seen
                        assert src+j==g*m+y*64+r+j
                        assert 0<=src+j<32*m
                        seen.add((g,r+j))
            assert seen=={(g,r) for g in range(32) for r in range(64)}
    assert 32*64*4==8192
    assert 8192+2*64*64*2+2*128*64+2*128*4==p.SHARED


def test_full_panel_never_rounded_and_math_unchanged():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    old=p.eight_header(source);new=p.generated_header(source)
    assert 'int activation_factors[32][64]' in new
    assert 's.activation_factors[slot]' not in new
    assert 's.activation_factors[group][cute::get<0>(coord)]' in new
    start='      // Eight independent chains:';end='  // Reuse the 64 INT32'
    assert old[old.index(start):old.index(end)].replace('s.activation_factors[slot]',
        's.activation_factors[group]')==new[new.index(start):new.index(end)]
    assert old[old.index(end):].replace('o78_eight_chain_experiment',
        'o78_activation_panel_experiment')==new[new.index(end):]
    assert new.count('preload_activation_factors(s,af,m);')==1
    assert new.index('preload_activation_factors(s,af,m);')<new.index('for(int group=0;group<Groups;++group)')
    loop=new[new.index('for(int group=0;group<Groups;++group)'):]
    assert loop.index('cp.async.wait_group 0;')<loop.index('__syncthreads();')<loop.index('s.activation_factors[group]')
    assert new.count('__syncthreads();')==old.count('__syncthreads();')
    kernel=(ROOT/'csrc/sm80/roof_o78_activation_panel_probe.cu').read_text()
    baseline=(ROOT/'csrc/sm80/roof_o78_eight_chain_probe.cu').read_text()
    assert kernel[kernel.index('  const uint32_t flag='):].replace('o78_activation_panel_experiment',
        'o78_eight_chain_experiment')==baseline[baseline.index('  const uint32_t flag='):]


def test_invalid_copy_coordinates_and_negative_gate():
    for args in ((128,0,64,0),(0,4,64,0),(-1,0,64,0),(0,0,63,0),(0,0,64,1)):
        with pytest.raises(ValueError):p.mapping(*args)
    def live(n,local=0):
        ops={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16,
             'LDGSTS.E.BYPASS.128':9,'BAR.SYNC':1,'LDL':local}
        return dict(allocated_gpr=168,loops=[dict(kind='integer',static_instructions=n,opcode_counts=ops)])
    resources={k:dict(active_blocks_per_sm=3) for k in ('control','candidate')}
    assert p.cost_gate(live(383),live(370),resources)['passed']
    assert not p.cost_gate(live(383),live(372),resources)['passed']
    assert not p.cost_gate(live(383),live(370,1),resources)['passed']
    resources['candidate']['active_blocks_per_sm']=2
    assert not p.cost_gate(live(383),live(370),resources)['passed']


def test_separate_small_spill_review_does_not_rewrite_failed_gate():
    from copy import deepcopy
    from benchmark_o78_activation_panel import resource_review
    gate=dict(passed=False,checks=dict(no_hot_local=False,allocation=True,
        meaningful_work_reduction=True,native_math=True,control_encoding_unchanged=True))
    original=deepcopy(gate)
    live=dict(loops=[dict(kind='integer',opcode_counts={'LDL':1,'STL':0})])
    resources=dict(candidate=dict(registers_per_thread=168,local_bytes=8,threads=128,active_blocks_per_sm=3))
    assert resource_review(gate,live,resources)['passed']
    assert gate==original and not gate['passed']
    for change in ('more_loads','hot_store','lower_residency','no_work_reduction'):
        g,l,r=deepcopy(gate),deepcopy(live),deepcopy(resources)
        if change=='more_loads':l['loops'][0]['opcode_counts']['LDL']=2
        elif change=='hot_store':l['loops'][0]['opcode_counts']['STL']=1
        elif change=='lower_residency':r['candidate']['active_blocks_per_sm']=2
        else:g['checks']['meaningful_work_reduction']=False
        assert not resource_review(g,l,r)['passed']
