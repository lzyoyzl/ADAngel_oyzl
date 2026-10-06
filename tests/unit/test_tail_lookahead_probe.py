from pathlib import Path
import ast
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_tail_lookahead_codegen import CONFIG, generated_header, worth_runtime, schedule_summary
from inspect_eight_chain_schedule import trace


@pytest.mark.parametrize('kind',('o3','o78'))
def test_only_two_early_chains_same_quantization_copies_and_guard(kind):
    text=generated_header(kind)
    assert 'auto next_start=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_2{}));' in text
    assert 'decltype(mi)::value==0 && decltype(ni)::value<2' in text
    assert 'cute::copy(next_start(cute::_,ni),p);' in text
    assert text.index('load_b_half0(cute::_1{});')<text.index('cute::clear(next_start);')
    assert text.index('cute::clear(next_start);')<text.index('acc(vi,mi,ni)+=partial')
    assert text.count('partial(i)*=16;')==2
    assert 'cp.async.commit_group' in text and '__syncthreads();' in text
    if kind=='o78':
        assert 'slot=group%2' in text and 's.activation_factors[slot][cute::get<0>(coord)]*' in text
    else:
        assert 'slot=group%3' in text and 'roof_grouped_cta::tile()' in text
    wrapper=(ROOT/f'csrc/sm80/roof_{kind}_tail_lookahead_probe.cu').read_text()
    assert CONFIG[kind]['symbol'] in wrapper and '__launch_bounds__(128,3)' in wrapper


def test_integer_algebra_and_eight_extra_slot_transfer():
    rng=np.random.default_rng(97)
    a=rng.integers(-128,128,(64,128),dtype=np.int32)
    w=rng.integers(-8,8,(128,128),dtype=np.int32)
    low,high=a&15,a>>4
    p0=(high@w[:64].T)*16+low[:,:64]@w[:64,:64].T
    extra=high[:,:64]@w[64:72,:64].T
    out=np.zeros((64,128),dtype=np.int32)
    out[:,:64]=p0+low[:,64:]@w[:64,64:].T
    p1=high[:,:64]@w[64:,:64].T
    p1[:,:8]=extra
    p1+=high[:,64:]@w[64:,64:].T;p1*=16
    p1+=low@w[64:].T;out[:,64:]=p1
    assert np.array_equal(out,a@w.T)


def test_gate_requires_more_real_programmed_chains_not_only_source_slots():
    live=dict(allocated_gpr=168,loops=[dict(kind='integer',opcode_counts={
        'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16})])
    for count in (9,10):assert worth_runtime(live,dict(peak_started_not_finished_chains=count))
    for count in (8,11,16):assert not worth_runtime(live,dict(peak_started_not_finished_chains=count))
    assert not worth_runtime(dict(live,allocated_gpr=176),dict(peak_started_not_finished_chains=10))
    assert not worth_runtime(dict(live,loops=[dict(kind='integer',opcode_counts={
        **live['loops'][0]['opcode_counts'],'LDL.64':3})]),dict(peak_started_not_finished_chains=10))


@pytest.mark.parametrize('kind',('o3','o78'))
def test_runtime_gate_and_same_preparation_are_explicit(kind):
    source=(ROOT/f'scripts/benchmark_{kind}_tail_lookahead.py').read_text()
    ast.parse(source)
    assert 'full_sample_args()' in source
    assert "if not receipt['worth_runtime_validation']" in source
    assert 'values[0]>168' in source and 'values[3]<3' in source
    assert 'static_programmed_chain_peak' in source and 'partial_registers=' in source
    if kind=='o3':
        assert 'conversion_candidate=2' in source and 'preparation_identical=True' in source
    else:
        assert "default_gpu_build=Path('reports/o378_roof_v73_codegen')" in source
        assert 'new_preparation_or_layout=False' in source


def test_machine_fixture_ten_chains_with_native_old_work():
    ops=[]
    def mma(r,kind,zero=False):
        ops.append(f'IMMA.16864.{kind}.S4 R{r}, R0.ROW, R8.COL, '+('RZ' if zero else f'R{r}'))
    def shift():
        for r in range(32,64):ops.append(f'SHF.L.U32 R{r}, R{r}, 0x4, RZ')
    for r in range(32,64,4):mma(r,'S4',True)
    for r in range(32,64,4):mma(r,'S4')
    shift()
    for r in range(32,64,4):mma(r,'U4')
    mma(64,'S4',True);mma(68,'S4',True)
    for r in range(32,64,4):mma(r,'U4')
    for i in range(8):ops.append(f'MOV R{32+i}, R{64+i}')
    for r in range(40,64,4):mma(r,'S4',True)
    for r in range(32,64,4):mma(r,'S4')
    shift()
    for r in range(32,64,4):mma(r,'U4')
    for r in range(32,64,4):mma(r,'U4')
    sass='Function : prologue\n'+'\n'.join(f'/*{i*16:04x}*/ {op} ;' for i,op in enumerate(ops))
    live=dict(loops=[dict(kind='integer',begin_pc='0x0',end_pc=hex((len(ops)-1)*16))])
    result=trace(sass,'prologue',live);order=schedule_summary(result)
    assert result['total_mma']==64 and order['peak_started_not_finished_chains']==10
    assert order['ninth_started_chain_first_mma_ordinal']==25
    assert 'not logical N coordinates' in order['interpretation']
