"""O3 transpose semantics and immutable compile gates; no torch required."""
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_transposed_mma_codegen import cost_gate, LIMITS


def test_signed_unsigned_operand_swap_exact_dot_and_scale():
    rng=random.Random(124)
    for _ in range(300):
        a=[rng.randrange(-128,128) for _ in range(128)]
        w=[rng.randrange(-8,8) for _ in range(128)]
        low=[x&15 for x in a];high=[x//16 for x in a]
        original=sum(x*y for x,y in zip(a,w))
        transposed=16*sum(x*y for x,y in zip(w,high))+sum(x*y for x,y in zip(w,low))
        assert original==transposed
        f=1<<rng.randrange(0,10)
        assert original*f==transposed*f


def test_predeclared_gate_does_not_reward_only_scale_load_reduction():
    def info(ins,peak=166,**changes):
        ops={'IMMA.16864.S4.S4':32,'IMMA.16864.S4.U4':32,
             'LDSM.16.M88.4':16,'LDGSTS.E.BYPASS.128':9,'BAR.SYNC':1}
        ops.update(changes)
        return dict(allocated_gpr=168,loops=[dict(kind='integer',static_instructions=ins,
                    max_live_gpr=peak,opcode_counts=ops)])
    assert not cost_gate(info(323),info(320))['passed']
    assert cost_gate(info(323),info(313))['passed']
    assert cost_gate(info(323),info(323,150))['passed']
    assert not cost_gate(info(323),info(300,**{'LDL':1}))['passed']
    assert not cost_gate(info(323),info(300,**{'LDSM.16.M88.4':20}))['passed']
    assert LIMITS['min_hot_instruction_reduction']==.03


def test_no_extra_pipeline_or_transpose_buffer_or_guard_change():
    body=(ROOT/'csrc/sm80/o3_transposed_mma_probe.cuh').read_text()
    entry=(ROOT/'csrc/sm80/roof_o3_transposed_mma_probe.cu').read_text()
    assert 'o3_grouped_cta_experiment::prefetch(s,slot,group,a,w,metadata,m,n,k)' in body
    assert 'using Storage=o3_grouped_cta_experiment::Storage' in body
    assert 's.factor[slot][cute::get<0>(c)]' in body
    assert 'partial(i)*=16' in body and 'for(int group=0;group<32;++group)' in body
    assert 'tile.y*64+cute::get<1>(c)' in body and 'tile.x*128+cute::get<0>(c)' in body
    assert 'flag&6u' in entry and 'if(flag&1u)' in entry and 'o3_grouped_fallback::o3_body' in entry
    assert 'cudaMalloc' not in body and body.count('__syncthreads()')==1
