from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_sixteen_chain_codegen import CONFIG, generated_header, worth_runtime


@pytest.mark.parametrize('kind',('o3','o78'))
def test_same_tile_work_and_semantics_more_independent_chains(kind):
    text=generated_header(kind)
    assert 'cute::_4{},cute::_2{},cute::_8{}' in text
    assert 'o1_static_for<0,64>([&](auto i) { partial(i)*=16; });' in text
    assert text.count('o1_static_for<0,8>([&](auto ni)')==5
    assert 'auto full_ni=ni;' in text
    assert 'using SliceMma=C::Mma;' in text
    assert 'cute::_128{},cute::_64{}),cute::make_coord(cute::_0{},half)' in text
    assert 'SM80_16x8x64_S32U4S4S32_TN' in text
    assert 'SM80_16x8x64_S32S4S4S32_TN' in text
    assert 'cute::size(acc))::value==64' in text or 'SAME 32-bit register slots' in text
    assert '__syncthreads();' in text and 'cp.async.commit_group' in text
    if kind=='o78':
        assert 's.activation_factors[slot][cute::get<0>(coord)]*' in text
        assert 's.weight_factors[slot][cute::get<1>(coord)]' in text
        assert 'slot=group%2' in text
    else:
        assert 'const int coefficient=s.factor[slot][cute::get<1>(coord)];' in text
        assert 'slot=group%3' in text and 'roof_grouped_cta::tile()' in text


def test_full_N_coordinate_and_integer_merge_identity():
    #16 physical MMA chains cover exactly the same16 logical atom outputs
    # as two successive groups of8. No floating-point reassociation needed.
    old={(mi,nb*4+ni,vi) for nb in range(2) for mi in range(2) for ni in range(4) for vi in range(4)}
    new={(mi,ni,vi) for mi in range(2) for ni in range(8) for vi in range(4)}
    assert old==new and len(new)==64
    rng=np.random.default_rng(92)
    a=rng.integers(-128,128,(64,128),dtype=np.int32)
    w=rng.integers(-8,8,(128,128),dtype=np.int32)
    high=a>>4; low=a&15
    merged=((high[:,:64]@w[:,:64].T)+(high[:,64:]@w[:,64:].T))*16
    merged+=low[:,:64]@w[:,:64].T; merged+=low[:,64:]@w[:,64:].T
    assert np.array_equal(merged,a@w.T)


def test_fixed_compile_gate_no_sweep_or_default_mutation():
    loop=dict(kind='integer',opcode_counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16})
    assert worth_runtime(dict(allocated_gpr=216,loops=[loop]))
    assert not worth_runtime(dict(allocated_gpr=249,loops=[loop]))
    assert not worth_runtime(dict(allocated_gpr=216,loops=[dict(loop,opcode_counts={**loop['opcode_counts'],'LDL.64':1})]))
    for kind in CONFIG:
        text=(ROOT/f'csrc/sm80/roof_{kind}_sixteen_chain_probe.cu').read_text()
        assert '__launch_bounds__(128,2)' in text
        assert CONFIG[kind]['symbol'] in text and 'if(flag' in text
        runtime=(ROOT/f'scripts/benchmark_{kind}_sixteen_chain.py').read_text()
        assert 'full_sample_args()' in runtime and "if not receipt['worth_runtime_validation']" in runtime
