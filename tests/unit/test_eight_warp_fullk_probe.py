from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_eight_warp_fullk_codegen import CONFIG, generated_header, generated_fallback, gate, analyze


@pytest.mark.parametrize('kind',('o3','o78'))
def test_fixed_geometry_keeps_math_and_all_copies(kind):
    text=generated_header(kind)
    assert 'using SliceMma=C::Mma;' in text
    assert 'auto full_ni=ni;' in text and 'nb*cute::_4{}+ni' not in text
    assert 'static_assert(decltype(cute::size(acc))::value==32);' in text
    assert 'cute::make_shape(cute::_4{},cute::_2{},cute::_4{})' in text
    assert text.count('partial(i)*=16;')==1
    assert 'cp.async.commit_group' in text and '__syncthreads();' in text
    assert text.count('chunk*4096')==2 and 'chunk*2048' not in text
    assert 'o1_static_for<0,1>([&](auto chunk)' in text
    assert 'o1_static_for<0,2>([&](auto chunk)' in text
    assert 'cute::make_shape(cute::_128{},cute::_64{}),cute::make_coord(nb,half)' in text
    if kind=='o3':
        assert 'roof_grouped_cta::tile()' in text and 'slot=group%3' in text
        assert 's.factor[slot][cute::get<1>(coord)]' in text
    else:
        assert 'slot=group%2' in text and 's.activation_factors[slot][cute::get<0>(coord)]*' in text
        assert 'roof_grouped_cta' not in text
    wrapper=(ROOT/f'csrc/sm80/roof_{kind}_eight_warp_fullk_probe.cu').read_text()
    assert CONFIG[kind]['symbol'] in wrapper and '__launch_bounds__(256,2)' in wrapper
    fallback=generated_fallback(kind)
    assert f'namespace {kind}_eight_warp_fallback' in fallback
    assert '(WN==2 || WN==4)' in fallback


def test_copy_coverage_256_threads_exactly_once_and_same_integer_algebra():
    for rows,count in ((64,1),(128,2)):
        offsets=[thread*16+chunk*4096 for thread in range(256) for chunk in range(count)]
        assert sorted(offsets)==list(range(0,rows*64,16))
    rng=np.random.default_rng(98)
    a=rng.integers(-128,128,(64,128),dtype=np.int32)
    w=rng.integers(-8,8,(128,128),dtype=np.int32)
    assert np.array_equal((a>>4)@w.T*16+(a&15)@w.T,a@w.T)


def fixture(count,threads,instructions=160,registers=120,local=0,ldsm=12):
    return dict(allocated_gpr=registers,cta_threads=threads,loops=[dict(kind='integer',
        static_instructions=instructions,opcode_counts={
        'IMMA.16864.S4.S4':count//2,'IMMA.16864.U4.S4':count//2,
        'LDSM.16.M88.4':ldsm,'LDL.64':local})])


def test_gate_weights_by_warps_not_deceptive_per_thread_reduction():
    old=fixture(64,128,instructions=323,registers=168,ldsm=16)
    assert gate(old,fixture(32,256,instructions=177))['passed']
    assert not gate(old,fixture(32,256,instructions=178))['passed']
    assert not gate(old,fixture(32,256,local=3))['passed']
    assert not gate(old,fixture(32,256,registers=132))['passed']
    assert not gate(old,fixture(32,256,ldsm=16))['passed']
    assert not gate(old,fixture(64,256))['passed']


def test_exact_symbol_and_32_mma_loop_liveness_parser():
    lines=['//---- .text.exact','// SHI_REGISTERS=120']
    pc=0
    for tag in ('integer','fallback'):
        lines.append(f'.L_{tag}:')
        ops=['IMMA.16864.S4.S4 R0, R8, R12, R0;']*16
        ops+=['IMMA.16864.U4.S4 R0, R8, R12, R0;']*16
        if tag=='fallback':ops+=['I2F R4, R0;']
        ops += [f'BRA `(.L_{tag});']
        for op in ops:
            lines.append(f'/*{pc:04x}*/ {op} // | 110 |');pc+=16
    result=analyze('\n'.join(lines),'exact')
    assert result['cta_threads']==256 and result['allocated_gpr']==120
    assert {x['kind'] for x in result['loops']}=={'integer','fp32_fallback'}
    with pytest.raises(ValueError,match='exact eight-warp'):analyze('\n'.join(lines),'other')
    with pytest.raises(ValueError,match='one complete32'):analyze('\n'.join(lines),'exact',64)
