"""Exact CPU SWAR/carry/packing and unchanged row/Event contracts, not GPU proof."""
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_nv6_swar_codegen as probe


def test_all_codes_and_byte_carry_bound():
    for code in range(256):
        assert probe.swar(code*0x01010101)==probe.reference(code*0x01010101)
        q=probe.scalar(code);sign=int(bool(code&32))
        assert ((abs(q)^(sign*127))+sign)<=128
    assert probe.swar(0x20202020)==probe.swar(0x21212121)==(0,0)
    assert probe.swar(0x3f3f3f3f)==(0xe2e2e2e2,3600)


def test_all_pairs_repeated_complemented_and_random():
    for lo in range(65536):
        for hi in (lo,lo^65535):
            word=lo|(hi<<16)
            assert probe.swar(word)==probe.reference(word)
    rng=random.Random(20261007)
    for _ in range(16384):
        word=rng.getrandbits(32)
        assert probe.swar(word)==probe.reference(word)
        q,_=probe.swar(word)
        for shift in (0,4):
            expect=sum(((q>>(8*j+shift))&15)<<(4*j) for j in range(4))
            assert probe.compact(q>>shift)==expect


def test_invalid_and_row_contract():
    import pytest
    for bad in (-1,1<<32,True,.5):
        with pytest.raises(ValueError):probe.swar(bad)
    old=(ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    new=probe.generated_header();marker='  const unsigned dest=dst_group*(128/Elements)+lane;'
    assert new[new.index(marker):]==old[old.index(marker):].replace(
        'namespace row_fused_probe','namespace nv6_row_swar_probe')
    assert 'square=unsigned(nv6_swar_probe::square_add(q,int(square)));' in new
    assert 'if constexpr(Format==Kind::Nv6)' in new


def test_event_guard_weight_and_scope_preserved():
    old=(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    new=probe.generated_host()
    a=old[old.index('extern "C" int roof_o78_row_fused_benchmark('):]
    b=new[new.index('extern "C" int roof_o78_nv6_swar_benchmark('):]
    assert a==b.replace('roof_o78_nv6_swar_benchmark','roof_o78_row_fused_benchmark').replace(
        'Nv6SwarOnline x(','RowFusedOnline x(')
    begin=old.index('    adangel_o78_prepare_cta_guard')
    end=old.index('\n  }',begin)
    assert old[begin:end] in new
    assert 'void weight(bool) {RowFusedOnline::weight(true);}' in new
    assert 'if(!candidate || variant!=8)' in new
    source=Path(probe.__file__).read_text()
    assert 'improvement>=.05 and dots==4' in source
    assert "b['registers']<=a['registers']" in source
    assert 'GEMM_modified=False' in source and 'production_default_changed=False' in source
