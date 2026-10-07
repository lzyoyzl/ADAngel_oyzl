"""CPU exact byte-lane, RNE and immutable generation checks, not GPU proof."""
from pathlib import Path
import random
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'python'))
import probe_mx8_swar_codegen as p


def test_all_codes_ties_sign_and_lane_carry():
    for c in range(256):
        assert p.swar(c*0x01010101)==p.reference(c*0x01010101)
        assert ((abs(p.scalar(c))^(127 if c&128 else 0))+bool(c&128))<=128
    assert [p.scalar(c) for c in (0x40,0x41,0x4c,0x52,0x56,0x59,0x5b)]==[0,1,2,2,4,4,6]
    assert p.scalar(0x7e)==112 and p.scalar(0xfe)==-112
    # These two numeric slots match unchecked legacy decoding, NOT valid FP8.
    assert p.scalar(0x7f)==120 and p.scalar(0xff)==-120


def test_pairs_random_packing_and_squares():
    table=[p.scalar(c) for c in range(256)]
    def oracle(word):
        q=[table[(word>>(8*j))&255] for j in range(4)]
        return sum((v&255)<<(8*j) for j,v in enumerate(q)),sum(v*v for v in q)
    for lo in range(65536):
        for hi in (lo,lo^65535):
            word=lo|(hi<<16)
            assert p.swar(word)==oracle(word)
    rng=random.Random(20261007)
    for _ in range(65536):
        word=rng.getrandbits(32);q,square=p.swar(word)
        assert (q,square)==oracle(word)
        for shift in (0,4):
            assert p.fp6.compact(q>>shift)==sum(((q>>(8*j+shift))&15)<<(4*j) for j in range(4))


def test_invalid_and_unchanged_row_and_event_contract():
    for value in (-1,1<<32,True,.5):
        with pytest.raises(ValueError):p.swar(value)
    old=(ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    header=p.generated_header();marker='  const unsigned dest=dst_group*(128/Elements)+lane;'
    assert header[header.index(marker):]==old[old.index(marker):].replace('row_fused_probe','mx8_row_swar_probe')
    assert 'if constexpr(Format==Kind::Mx8)' in header
    host=p.generated_host();raw=(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    body=raw[raw.index('extern "C" int roof_o78_row_fused_benchmark('):]
    candidate=host[host.index('extern "C" int roof_o78_mx8_swar_benchmark('):]
    assert body==candidate.replace('roof_o78_mx8_swar_benchmark','roof_o78_row_fused_benchmark').replace('Mx8SwarOnline x(','RowFusedOnline x(')
    assert 'void weight(bool) {Nv4SwarOnline::weight(true);}' in host
    assert 'if(candidate)packed_activation();else lookup_activation();' in host
    assert 'mx8_warp_lut_probe::adangel_sm80_row_warp_lut_metadata<Kind::Mx8,16>' in host
    assert 'mx8_row_swar_probe::adangel_sm80_mx8_swar_metadata<Kind::Mx8,16>' in host
    assert 'adangel_sm80_mx8_swar_metadata<Kind::Nv6' not in host
    assert 'vector_probe::mx8(' not in host
    marker='    adangel_o78_prepare_cta_guard'
    assert raw[raw.index(marker):raw.index('\n  }',raw.index(marker))] in host


def test_cpu_matches_independent_format_decoder():
    from adangel.quantization.mixed_formats import decode_scalar
    for c in range(256):
        if (c&127)==127:
            with pytest.raises(ValueError):decode_scalar(c,'e4m3')
        else:
            assert p.scalar(c)==round(decode_scalar(c,'e4m3')*.25)
