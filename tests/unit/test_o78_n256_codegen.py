"""One N-reuse candidate: preserve source math/guard, no production dispatch."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o78_n256_codegen import generated_header


def test_128_accumulators_four_n64_slices_same_eight_chains():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    result=generated_header(source)
    assert 'sizeof(Storage)==51712' in result
    assert 'cute::size(acc))::value==128' in result
    assert 'o1_static_for<0,4>([&](auto nb)' in result
    assert 'cute::make_shape(cute::_4{},cute::_2{},cute::_4{})' in result
    assert 'blockIdx.x*128' not in result
    assert 'partial(vi,mi,ni)*coefficient' in result
    with pytest.raises(ValueError):generated_header(source.replace('blockIdx.x*128','blockIdx.x*127',1))


def test_combined_guard_never_silently_accepts_unsafe_half():
    source=(ROOT/'csrc/sm80/roof_o78_n256_probe.cu').read_text()
    assert 'status[index]|status[index+1]' in source and 'if(flag>1u) return' in source
    assert 'O78::o3_body<64,256,128' in source
    assert '__launch_bounds__(128,2)' in source
    for a in range(3):
        for b in range(3):
            assert ((a|b)==0) == (a==b==0)
            assert ((a|b)>1) == (a==2 or b==2)


def test_cached_driver_has_same_timer_and_exact_tile_guard():
    source=(ROOT/'csrc/sm80/roof_o78_n256_driver.cpp').read_text()
    assert 'n/tile_n,m/64,1,128,1,1' in source
    assert 'p->smem!=(tile_n==256?51712u:34304u)' in source
    assert 'Events events(repeats*2)' in source
    assert source.index('Events events(repeats*2)')<source.index('for(int i=0;i<warmup')
    script=(ROOT/'scripts/benchmark_o78_n256_probe.py').read_text()
    assert "mode!='compute_only'" in script and "if '--full-modes' in sys.argv" in script
    assert 'np.any(case.oracle[\'status_flat\']>1)' in script
    assert 'same_native_driver' in script
