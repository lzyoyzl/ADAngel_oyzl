"""Exact packed encoding/norm and scope gates; not GPU/performance proof."""
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_nv4_swar_codegen as probe


def test_all_codes_rne_negative_zero_and_nibble_carry_bound():
    for code in range(16):
        word=code*0x11111111
        assert probe.swar(word)==probe.reference(word)
        mag=abs(round((-.5 if code&8 else .5)*[0,1,2,3,4,6,8,12][code&7]))
        sign=int(bool(code&8))
        assert (mag^(sign*7))+sign<=8
    assert probe.swar(0x88888888)==probe.swar(0x99999999)==(0,0)
    assert probe.swar(0xffffffff)==(0xaaaaaaaa,8*36)


def test_all_four_nibble_patterns_repeat_and_complement():
    for lo in range(65536):
        for hi in (lo,lo^65535):
            word=lo|(hi<<16)
            assert probe.swar(word)==probe.reference(word)


def test_random_full_words_and_unsigned_limit():
    rng=random.Random(20261007)
    for _ in range(16384):
        word=rng.getrandbits(32)
        assert probe.swar(word)==probe.reference(word)
    import pytest
    for bad in (-1,1<<32):
        with pytest.raises(ValueError):probe.swar(bad)


def test_row_metadata_and_mapping_are_not_changed():
    old=(ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    new=probe.generated_header();marker='  const unsigned dest=dst_group*(128/Elements)+lane;'
    assert new[new.index(marker):]==old[old.index(marker):].replace(
        'namespace row_fused_probe','namespace nv4_row_swar_probe')
    assert 'const uint2 result=nv4_swar_probe::packed_word(words[w]);' in new
    assert 'low[w]=result.x;square+=result.y;' in new
    assert 'if(row>=rows) return;' in new
    assert '__constant__' not in new and 'lut_word' not in new


def test_event_body_and_activation_are_original_v73():
    old=(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    new=probe.generated_host()
    a=old[old.index('extern "C" int roof_o78_row_fused_benchmark('):]
    b=new[new.index('extern "C" int roof_o78_nv4_swar_benchmark('):]
    assert a==b.replace('roof_o78_nv4_swar_benchmark','roof_o78_row_fused_benchmark').replace(
        'Nv4SwarOnline x(','RowFusedOnline x(')
    assert 'void activation(bool) { RowFusedOnline::activation(true); }' in new
    assert 'if(!candidate || variant!=7)' in new


def test_scope_compile_gate_and_boolean_masks():
    source=Path(probe.__file__).read_text()
    helper=(ROOT/'csrc/sm80/nv4_swar_conversion.cuh').read_text()
    assert 'GEMM_modified=False' in source and 'production_default_changed=False' in source
    assert 'improvement>=.05' in source and "b['registers']<=a['registers']" in source
    assert "ops.get('POPC',0)==10" in source
    for lut in (0x24,0xb8,0xc0,0x98,0x20,0x40,0x80):
        assert 'truth<0x'+format(lut,'02x')+'>' in helper
    assert 'shfl' not in helper and 'constant__' not in helper


def test_runtime_has_identical_GEMM_full24_and_after_Event_payload_norm_checks():
    source=(ROOT/'scripts/benchmark_o7_nv4_swar.py').read_text()
    assert 'self.handles[0]=self.handles[1]' in source
    assert 'negative compile gate; do not launch candidate' in source
    assert 'full24 only; no small performance screen' in source
    assert "variants=('o7',)" in source
    assert "case.group_squares_reference['wsq']" in source
    assert 'case.expected_payload[key].view(torch.uint8)' in source
    assert 'full_v99_source_identity_equal=True' in source
    assert 'fn(self.handles[1],variant,1,0,sa,sw,state' in source


def test_runtime_event_selection_and_scope_with_mock(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import benchmark_o7_nv4_swar as runtime
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
    driver.lib=SimpleNamespace(roof_o78_nv4_swar_benchmark=invoke)
    case=SimpleNamespace(variant='o7',oracle={'status_flat':np.array([0])},
        a_source=1,w_source=2,state_pointers=3,m=64,n=128,a_multiplier=4.,w_multiplier=1.,
        state={'y':'out','w':tensor,'ws':tensor,'wsq':tensor},
        expected_payload={'w':tensor,'ws':tensor},group_squares_reference={'wsq':np.array([1])})
    for mode in runtime.eight.base.MODES:
        for policy in (0,1,2):driver.run(case,policy,mode,50,2,100)
    assert [c[0] for c in calls]==[11,11,22]*4
    assert [c[2] for c in calls]==[0,1,0]*4
    assert all(c[-3]==100 for c in calls)
