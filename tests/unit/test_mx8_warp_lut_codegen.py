"""Exact encoding and source gates, not GPU correctness or performance."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_mx8_warp_lut_codegen as probe


def test_all_valid_mx8_codes_rne_and_original_decoder():
    words=probe.lookup_words()
    for code in range(256):
        exp,mant=(code&127)>>3,code&7
        original=(1+mant/8)*2.**(exp-7) if exp else mant*2.**-9
        expected=round((-1 if code&128 else 1)*original*.25)
        magnitude=(words[(code&127)>>2]>>((code&3)*8))&255
        actual=-magnitude if code&128 else magnitude
        assert actual==expected==probe.legacy_q(code)
    assert probe.legacy_q(0)==probe.legacy_q(128)==0
    assert probe.legacy_q(126)==112 and probe.legacy_q(254)==-112
    # These preserve old unchecked numeric slots, NOT valid E4M3FN inputs.
    assert probe.legacy_q(127)==120 and probe.legacy_q(255)==-120


def test_every_lane_can_gather_every_positive_code_and_sign():
    own_words=probe.lookup_words()
    assert len(own_words)==32
    for lane in range(32):
        for shift in range(256):
            c=(lane+shift)&255
            word=own_words[(c&127)>>2]
            q=(word>>((c&3)*8))&255
            assert (-q if c&128 else q)==probe.legacy_q(c)


def test_row_mapping_guard_and_metadata_body_remain_exact():
    old=(ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    new=probe.generated_header()
    marker='  // One subwarp covers exactly one G128'
    tail=old[old.index(marker):].replace('namespace row_fused_probe','namespace mx8_warp_lut_probe')
    assert new[new.index(marker):]==tail
    assert 'if(row>=rows) return;' in new
    assert '__shfl_sync(0xffffffffu,own_word,(code&127u)>>2)' in new
    assert '__device__ const unsigned mx8_magnitude_words[32]' in new
    assert '__constant__' not in new
    assert new.count('q=mx8_lookup(c,lut_word);')==1


def test_original_event_timing_not_replaced_by_cheap_microbenchmark():
    old=(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    new=probe.generated_host()
    a=old[old.index('extern "C" int roof_o78_row_fused_benchmark('):]
    b=new[new.index('extern "C" int roof_o78_warp_lut_benchmark('):]
    assert a==b.replace('roof_o78_warp_lut_benchmark','roof_o78_row_fused_benchmark').replace('WarpLutOnline x(','RowFusedOnline x(')
    assert 'void weight(bool) { RowFusedOnline::weight(true); }' in new
    assert 'if(!candidate || variant!=7)' in new
    assert 'adangel_o78_prepare_cta_guard<<<' in new


def test_history_gates_are_explicit():
    source=Path(probe.__file__).read_text()
    assert 'GEMM_modified=False' in source
    assert 'production_default_changed=False' in source
    assert 'improvement>=.05' in source
    assert "ops.get('SHFL.IDX',0)>=16" in source
    assert "b['registers']<=a['registers']" in source


def test_runtime_compares_preparation_not_GEMM_and_requires_full24():
    path=ROOT/'scripts/benchmark_o7_warp_lut.py'
    source=path.read_text()
    assert 'self.handles[0]=self.handles[1]' in source
    assert 'negative compile gate; do not launch candidate' in source
    assert "variants=('o7',)" in source
    assert 'full24 only; no small performance screen' in source
    assert 'full_v99_source_identity_equal=True' in source
    assert "if any(row!=old[row['sample_id']] for row in rows)" in source


def test_runtime_uses_original_direct_and_amortized_timing(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import benchmark_o7_warp_lut as runtime
    calls=[]
    def invoke(*args):
        calls.append(args)
        for i in range(len(args[-1])):args[-1][i]=1.
        if args[3]==0:
            for i in range(args[-4]):args[-1][3*args[-4]+i]=2.
        return 0
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(
        current_stream=lambda:SimpleNamespace(cuda_stream=7))))
    driver=object.__new__(runtime.Driver)
    driver.handles={0:11,1:11,2:22}
    driver.lib=SimpleNamespace(roof_o78_warp_lut_benchmark=invoke)
    case=SimpleNamespace(variant='o7',oracle={'status_flat':np.array([0])},
        a_source=1,w_source=2,state_pointers=3,m=64,n=128,a_multiplier=4.,w_multiplier=1.,state={'y':'out'})
    for mode in runtime.eight.base.MODES:
        for policy in (0,1,2):driver.run(case,policy,mode,50,2,100)
    assert [c[0] for c in calls]==[11,11,22]*4
    assert [c[2] for c in calls]==[0,1,0]*4
    assert all(c[-3]==100 for c in calls)
