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


def test_actual_residency_review_and_runtime_contract():
    source=Path(probe.__file__).read_text()
    assert 'first_no_register_growth_gate_passed=first_gate' in source
    assert "runtime['control']['active_blocks_per_sm']==runtime['candidate']['active_blocks_per_sm']==8" in source
    assert "b['registers']<=32 and residency" in source
    runtime=(ROOT/'scripts/benchmark_o8_nv6_swar.py').read_text()
    for statement in ('self.handles[0]=self.handles[1]',"variants=('o8',)",
            'full24 only; no small performance screen',"case.group_squares_reference['asq']",
            'case.expected_payload[key].view(torch.uint8)','full_v99_source_identity_equal=True',
            'fn(self.handles[1],variant,1,0,sa,sw,state'):
        assert statement in runtime


def test_runtime_policy_mode_selection_with_mock(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import benchmark_o8_nv6_swar as runtime
    calls=[]
    def invoke(*args):
        calls.append(args)
        for i in range(len(args[-1])):args[-1][i]=1.
        if args[3]==0:
            for i in range(args[-4]):args[-1][3*args[-4]+i]=2.
        return 0
    tensor=SimpleNamespace(view=lambda _:None,cpu=lambda:SimpleNamespace(numpy=lambda:np.array([1])))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(uint8='u8',equal=lambda *_:True,
        cuda=SimpleNamespace(current_stream=lambda:SimpleNamespace(cuda_stream=7))))
    driver=object.__new__(runtime.Driver);driver.handles={0:11,1:11,2:22}
    driver.lib=SimpleNamespace(roof_o78_nv6_swar_benchmark=invoke)
    case=SimpleNamespace(variant='o8',oracle={'status_flat':np.array([0])},
        a_source=1,w_source=2,state_pointers=3,m=64,n=128,a_multiplier=4.,w_multiplier=1.,
        state={'y':'out','a':tensor,'as':tensor,'asq':tensor},
        expected_payload={'a':tensor,'as':tensor},group_squares_reference={'asq':np.array([1])})
    for mode in runtime.eight.base.MODES:
        for policy in (0,1,2):driver.run(case,policy,mode,50,2,100)
    assert [c[0] for c in calls]==[11,11,22]*4
    assert [c[2] for c in calls]==[0,1,0]*4
    assert all(c[-3]==100 for c in calls)
