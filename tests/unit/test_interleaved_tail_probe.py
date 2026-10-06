from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_interleaved_tail_codegen import CONFIG, generated_header, ordering_summary, worth_runtime
from inspect_eight_chain_schedule import trace
from inspect_interleaved_tail_equivalence import equivalence


@pytest.mark.parametrize('kind',('o3','o78'))
def test_same_work_storage_groups_and_safe_B_overwrite(kind):
    text=generated_header(kind)
    assert 'cute::_4{},cute::_2{},cute::_4{}' in text
    assert text.count('partial(i)*=16;')==2
    assert text.count('cute::gemm(HA{}')==4 and text.count('cute::gemm(LA{}')==4
    assert text.count('load_b_half0(cute::_')==text.count('load_b_half1(cute::_')==2
    assert 'auto full_ni=cute::_4{}+ni;' in text
    assert 'cute::clear(p);' in text and 'cute::clear(partial);' in text
    assert 'SM80_16x8x64_S32U4S4S32_TN' in text and 'SM80_16x8x64_S32S4S4S32_TN' in text
    assert '__syncthreads();' in text and 'cp.async.commit_group' in text
    if kind=='o78':
        assert 'slot=group%2' in text and 's.activation_factors[slot][cute::get<0>(coord)]*' in text
    else:
        assert 'slot=group%3' in text and 'roof_grouped_cta::tile()' in text
        assert 'const int coefficient=s.factor[slot][cute::get<1>(coord)];' in text
    first_low=text.index('cute::gemm(LA{}')
    replace_B0=text.index('load_b_half0(cute::_1{})')
    tail_low=text.index('cute::gemm(LA{}',first_low+1)
    consume=text.index('acc(vi,mi,ni)+=partial')
    clear=text.index('cute::clear(p);')
    new_high=text.index('cute::gemm(HA{}',clear)
    replace_B1=text.index('load_b_half1(cute::_1{})')
    assert first_low<replace_B0<tail_low<consume<clear<new_high<replace_B1


def test_recycled_fragment_identity_for_all_signed_INT8_nibbles():
    #All possible INT8 values and S4 weight codes: no approximation/rounding.
    a=np.arange(-128,128,dtype=np.int32)[:,None]
    w=np.arange(-8,8,dtype=np.int32)[None,:]
    assert np.array_equal(16*(a>>4)*w+(a&15)*w,a*w)
    rng=np.random.default_rng(96)
    a=rng.integers(-128,128,(64,128),dtype=np.int32)
    w=rng.integers(-8,8,(128,128),dtype=np.int32)
    af=rng.integers(0,4,(64,1),dtype=np.int32)
    wf=rng.integers(0,4,(1,128),dtype=np.int32)
    output=np.zeros((64,128),dtype=np.int32)
    partial=((a[:,:64]>>4)@w[:64,:64].T+(a[:,64:]>>4)@w[:64,64:].T)*16
    partial+=(a[:,:64]&15)@w[:64,:64].T
    next_high=(a[:,:64]>>4)@w[64:,:64].T
    for column in range(64):
        partial[:,column]+=(a[:,64:]&15)@w[column,64:]
        output[:,column]=partial[:,column]*af[:,0]*wf[0,column]
        partial[:,column]=next_high[:,column]
    partial+=(a[:,64:]>>4)@w[64:,64:].T
    partial*=16
    partial+=(a[:,:64]&15)@w[64:,:64].T+(a[:,64:]&15)@w[64:,64:].T
    output[:,64:]=partial*af*wf[:,64:]
    assert np.array_equal(output,(a@w.T)*af*wf)


def test_order_gate_requires_real_overlap_without_extra_chains_or_loads():
    counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16}
    live=dict(allocated_gpr=168,loops=[dict(kind='integer',opcode_counts=counts)])
    order=dict(next_slice_starts_before_all_old_mma_finish=True,peak_started_not_finished_chains=8)
    assert worth_runtime(live,order)
    assert not worth_runtime(live,dict(order,next_slice_starts_before_all_old_mma_finish=False))
    assert not worth_runtime(live,dict(order,peak_started_not_finished_chains=9))
    assert not worth_runtime(dict(live,allocated_gpr=176),order)
    assert not worth_runtime(dict(live,loops=[dict(kind='integer',opcode_counts={**counts,'LDL.64':3})]),order)
    for kind in CONFIG:
        source=(ROOT/f'csrc/sm80/roof_{kind}_interleaved_tail_probe.cu').read_text()
        assert '__launch_bounds__(128,3)' in source and CONFIG[kind]['symbol'] in source
        runtime=(ROOT/f'scripts/benchmark_{kind}_interleaved_tail.py').read_text()
        assert 'full_sample_args()' in runtime and "if not receipt['worth_runtime_validation']" in runtime
        assert 'values[3]<3' in runtime


def test_order_summary_does_not_claim_measured_speedup():
    chains=[dict(start_pc=hex(i*16),end_pc=hex(512+i*16)) for i in range(16)]
    chains[8]['start_pc']='0x180'
    s=dict(chains=chains,schedule=[dict(chain=8,stage=1,ordinal=26)],peak_started_not_finished_chains=8)
    result=ordering_summary(s)
    assert result['first_next_slice_mma_ordinal']==26
    assert result['next_slice_starts_before_all_old_mma_finish']
    assert 'predicted speedup' in result['interpretation']


def test_exact_machine_dataflow_fixture_recycles_eight_slots_not_sixteen():
    ops=[]
    def mma(chain,kind,zero=False):
        r=32+4*chain
        ops.append(f'IMMA.16864.{kind}.S4 R{r}, R0.ROW, R8.COL, '+('RZ' if zero else f'R{r}'))
    def shift():
        for r in range(32,64):ops.append(f'SHF.L.U32 R{r}, R{r}, 0x4, RZ')
    for chain in range(8):mma(chain,'S4',True)
    for chain in range(8):mma(chain,'S4')
    shift()
    for chain in range(8):mma(chain,'U4')
    for chain in range(8):
        mma(chain,'U4')
        for vi in range(4):
            r=32+chain*4+vi
            ops.append(f'IMAD R{100+chain*4+vi}, R{r}, R20, R{100+chain*4+vi}')
        mma(chain,'S4',True)
    for chain in range(8):mma(chain,'S4')
    shift()
    for chain in range(8):mma(chain,'U4')
    for chain in range(8):mma(chain,'U4')
    sass='Function : recycled\n'+'\n'.join(f'/*{i*16:04x}*/ {op} ;' for i,op in enumerate(ops))
    live=dict(loops=[dict(kind='integer',begin_pc='0x0',end_pc=hex((len(ops)-1)*16))])
    result=trace(sass,'recycled',live)
    order=ordering_summary(result)
    assert result['total_mma']==64 and order['peak_started_not_finished_chains']==8
    assert order['first_next_slice_mma_ordinal']==26
    assert order['next_slice_starts_before_all_old_mma_finish']


def test_symbol_rename_never_hides_a_changed_encoded_instruction():
    encoded='/*0000*/ NOP; /* 0x0000000000000001 */\n/* 0x0000000000000002 */\n'
    before='Function : exact_old\n'+encoded
    after='Function : exact_new\n'+encoded
    assert equivalence(before+after,'exact_old','exact_new')['passed']
    bad=after.replace('0x0000000000000002','0x0000000000000003')
    assert not equivalence(before+bad,'exact_old','exact_new')['passed']
    with pytest.raises(ValueError):equivalence(before+before+after,'exact_old','exact_new')
