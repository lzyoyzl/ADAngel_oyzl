from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_dp2a_recompose import LOW_BOUND,HIGH_BOUND,reference,entries,isa_gate,SYMBOL,CONTROL


def test_both_dot_ranges_fit_signed16_without_approximation():
    assert LOW_BOUND==15360 and HIGH_BOUND==8192
    for factor in (0,1,2,4,8,15):
        for low in range(-LOW_BOUND,LOW_BOUND+1):
            for high in (-HIGH_BOUND,-1,0,1,HIGH_BOUND):
                assert reference(low,high,factor,123)==123+(low+16*high)*factor


def test_high_sign_and_unsigned_byte_weights_exact():
    for high in range(-HIGH_BOUND,HIGH_BOUND+1):
        for factor in range(16):
            assert reference(-15360,high,factor)==(-15360+16*high)*factor


@pytest.mark.parametrize('args',[(15361,0,1),(0,-8193,1),(0,0,16),(1,1,1,(1<<31)-1)])
def test_range_and_prefix_guards_are_not_relaxed(args):
    with pytest.raises(ValueError):reference(*args)


def test_multi_idp4a_lowering_fails_closed():
    old={'opcode_counts':{'PRMT':1,'IDP.4A.U8.U8':1,'IDP.4A.S8.U8':1,'IMAD':1}}
    assert not isa_gate(old)['passed']
    native={'opcode_counts':{'PRMT':1,'IDP.2A.LO.S16.U8':1}}
    assert isa_gate(native)['passed']


def test_exact_symbol_predicate_and_similar_opcode_parsing():
    sass=f''' Function : {SYMBOL}
    /*0010*/ @P0 IDP.4A.S8.U8 R1, R2, R3, R4 ;
    /*0020*/ PRMT R1, R2, R3, 0x5410 ;
    Function : {CONTROL}
    /*0010*/ IMAD R1, R2, R3, R4 ;
    Function : {SYMBOL}_probe
    /*0010*/ IDP.2A.LO.S16.U8 R1, R2, R3, R4 ;
    '''
    parsed=entries(sass)
    assert parsed[SYMBOL]['opcode_counts']=={'IDP.4A.S8.U8':1,'PRMT':1}
    assert not isa_gate(parsed[SYMBOL])['passed']


def test_no_candidate_launch_or_production_binding_in_cost_gate():
    source=(ROOT/'scripts/probe_dp2a_recompose.py').read_text()
    cuda=(ROOT/'tests/cuda/probe_dp2a_recompose.cu').read_text()
    assert '<<<' not in cuda and 'dp2a.lo.s32.u32' in cuda
    assert 'torch' not in source and 'ctypes' not in source
    assert 'candidate_GPU_launched=False' in source and 'formal_extension_sha256_after' in source
