"""CPU proof of HiF4 packed RNE, sharing, carries and isolated timing scope."""
from pathlib import Path
import random
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_hif4_swar_codegen as probe


def test_scalar_range_ties_negative_zero_and_no_carry():
    for e8 in (0,1):
        for e4 in (0,1):
            for code in range(16):
                q=probe.scalar(code,e8,e4);sign=code>>3
                assert -7<=q<=7
                assert (abs(q)^(sign*7))+sign<=8
                assert probe.swar(code*0x11111111,e8,e4*3)==probe.reference(code*0x11111111,e8,e4*3)
    assert [probe.scalar(c,0,0) for c in range(8)]==[0,0,0,1,1,1,2,2]
    assert [probe.scalar(c,1,0) for c in range(8)]==[0,0,1,2,2,2,3,4]
    assert probe.swar(0x88888888,1,3)==(0,0)
    assert probe.swar(0xffffffff,1,3)==(0x99999999,392)


def test_million_patterns_all_microbits_vector_proof():
    from benchmark_o8_hif4_swar import exhaustive_reference
    word,e8,e4,packed,square=exhaustive_reference()
    lsb=np.uint32(0x11111111)
    magnitude=word&0x77777777
    half=((word>>1)&0x33333333)+((word&(word>>1))&lsb)
    quarter=((word>>2)&lsb)+(((word>>1)&(word|(word>>2)))&lsb)
    mask=(e4&1)*0xffff+((e4>>1)&1)*np.uint32(0xffff0000)
    hi=np.where(e8,magnitude,half);lo=np.where(e8,half,quarter)
    rounded=(hi&mask)|(lo&~mask);sign=(word>>3)&lsb
    assert np.array_equal(((rounded^(sign*7))+sign)^(sign<<3),packed)
    assert np.array_equal(sum(((rounded>>(4*j))&7)**2 for j in range(8)),square)
    rng=random.Random(20261008)
    for _ in range(8192):
        values=(rng.getrandbits(32),rng.randrange(2),rng.randrange(4))
        assert probe.swar(*values)==probe.reference(*values)


def test_metadata_byte_addressing_for_all_lanes():
    # Compare vector metadata reuse with the original per-four-element index.
    for lane in range(8):
        for j in range(2):
            i8=(lane*16+j*8)//8
            assert (lane//4,2*(lane%4)+j)==(i8//8,i8%8)
            for half in range(2):
                i4=(lane*16+j*8+half*4)//4
                assert (lane//2,4*(lane%2)+j*2+half)==(i4//8,i4%8)


def test_invalid_and_unchanged_metadata_events_and_activation():
    for values in ((-1,0,0),(1<<32,0,0),(True,0,0),(1,2,0),(1,0,4)):
        with pytest.raises(ValueError):probe.swar(*values)
    old=(ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    new=probe.generated_header();marker='  const unsigned dest=dst_group*(128/Elements)+lane;'
    assert new[new.index(marker):]==old[old.index(marker):].replace(
        'namespace row_fused_probe','namespace hif4_row_swar_probe')
    host=probe.generated_host();original=(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    a=original[original.index('extern "C" int roof_o78_row_fused_benchmark('):]
    b=host[host.index('extern "C" int roof_o78_hif4_swar_benchmark('):]
    assert a==b.replace('roof_o78_hif4_swar_benchmark','roof_o78_row_fused_benchmark').replace(
        'Hif4SwarOnline x(','RowFusedOnline x(')
    assert 'void activation(bool) {Nv6SwarOnline::activation(true);}' in host
    assert 'if(!candidate || variant!=8){RowFusedOnline::weight(true);return;}' in host
    assert probe.generated_files()['nv6_swar_prepare.cu']==probe.nv6.generated_host()


def test_runtime_contract_and_candidate_scope():
    from benchmark_o8_hif4_swar import timing_contract
    for mode in ('conversion_only','compute_only','cold','steady_state'):
        c=timing_contract(mode,100)
        assert c['gemm_cufunction_identical_between_policies'] and c['activation_preparation_identical']
        assert c['activation_preparation_implementation']=='v123_packed_FP6'
        assert c['payload_norm_checked_after_every_call']
        assert c['modified_stage']=='weight_payload_decode_and_exact_square_sum_only'
        assert c['stage_timing_inner_repeats']['total']==(100 if mode=='conversion_only' else 1)
    runtime=(ROOT/'scripts/benchmark_o8_hif4_swar.py').read_text()
    assert 'self.handles[0]=self.handles[1]' in runtime
    assert "for key in ('a','as','w','ws')" in runtime
    assert "for key in ('asq','wsq')" in runtime
    assert 'no small performance screen' in runtime
