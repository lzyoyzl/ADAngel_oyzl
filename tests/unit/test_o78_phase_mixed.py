"""One distinct CTA heterogeneity mechanism, not a performance acceptance."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_o78_phase_mixed_codegen as probe
from probe_o78_eight_chain_codegen import generated_header,MERGED


def test_four_body_only_changes_output_independent_instruction_order():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    body=probe.generated_four(source).replace('o78_four_chain_experiment','o78_eight_chain_experiment')
    assert body.replace(probe.FOUR,MERGED)==generated_header(source)
    assert body.count('Groups=32;')==1 and body.count('__syncthreads();')==1
    assert 'acc(vi,mi,full_ni)+=partial(vi,ni)*coefficient;' in body
    assert body.count('cute::gemm(HA{}')==2 and body.count('cute::gemm(LA{}')==2


def loop(chains,work=383):
    return dict(kind='integer',begin_pc=hex(chains*100),static_instructions=work,
        opcode_counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,
            'LDSM.16.M88.4':16,'LDGSTS.E.BYPASS.128':10,'BAR.SYNC':1})


def test_gate_requires_actual_two_paths_not_an_identical_or_uniform_rename():
    a=loop(8);old=dict(loops=[a])
    four=loop(4);paths=[dict(peak_started_not_finished_chains=c,loop=l)
        for c,l in ((4,four),(8,a))]
    new=dict(allocated_gpr=168)
    assert probe.cost_gate(old,new,paths)['passed']
    assert not probe.cost_gate(old,new,paths[1:])['passed']
    assert not probe.cost_gate(old,dict(allocated_gpr=176),paths)['passed']
    four['opcode_counts']['LDL']=1
    assert not probe.cost_gate(old,new,paths)['passed']
    del four['opcode_counts']['LDL'];four['static_instructions']=430
    assert not probe.cost_gate(old,new,paths)['passed']


def test_fixed_ratio_and_uniform_dispatch_before_complete_k_body():
    src=(ROOT/'csrc/sm80/roof_o78_phase_mixed_probe.cu').read_text()
    assert '(blockIdx.y*gridDim.x+blockIdx.x)%3u==0u' in src
    assert 'threadIdx' not in src and 'if(flag>1u) return;' in src
    assert 'if(flag==1u)' in src and 'O78::o3_body<64,128,128' in src
    assert '__launch_bounds__(128,3)' in src and 'o78_four_chain_experiment::body' in src
    counts=[0,0]
    for y in range(64):
        for x in range(32):counts[(y*32+x)%3==0]+=1
    assert counts==[1365,683]  # No guarantee of grouping on a particular SM.
    assert probe.LIMITS['four_chain_fraction']==[1,3]


def test_mixed_liveness_rejects_missing_real_paths():
    text='//---------------- .text.'+probe.SYMBOL+'\nSHI_REGISTERS=168\n'
    text+='/*0000*/ MOV R0, RZ; // | 1 |\n'
    with pytest.raises(ValueError,match='exactly two integer'):
        probe.mixed_liveness(text)
