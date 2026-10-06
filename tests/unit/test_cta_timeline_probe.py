from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_cta_timeline import analyze
from probe_cta_timeline_codegen import instrumentation_gate


def records(intervals):
    return np.array([[[s,e,sm,sm]]*4 for s,e,sm in intervals],dtype=np.uint64)


def test_three_wave_slots_and_tail():
    # Two SMs, three slots each. First wave full, last wave one CTA/SM.
    a=records([(1,11,s) for s in (4,9) for _ in range(3)]+[(11,16,s) for s in (4,9)])
    r=analyze(a,sm_count=2)
    assert r['observed_SMs']==2 and r['observed_max_CTA_overlap_per_SM']==3
    assert r['globaltimer_span_ms']==15e-6
    assert r['after_last_start_ms']==5e-6
    assert r['tail_capacity_time_fraction']['1']==1
    assert r['tail_missing_capacity_equivalent_ms']==pytest.approx((5*2/3)/1e6)
    assert r['not_an_achievable_latency_bound']


def test_no_contiguous_smid_assumption_and_64bit_timer_precision():
    offset=2**60
    a=records([(offset+1,offset+8,900),(offset+1,offset+8,45)])
    r=analyze(a,sm_count=2)
    assert r['globaltimer_span_ms']==7e-6 and r['observed_SMs']==2


def test_all_warps_not_only_warp_zero():
    a=records([(5,20,4)])
    a[0,1,0]=2;a[0,2,1]=25
    assert analyze(a,sm_count=1)['globaltimer_span_ms']==23e-6


@pytest.mark.parametrize('failure',['missing','migration','overlap','SM_count'])
def test_invalid_capture_rejected(failure):
    a=records([(1,5,0)])
    if failure=='missing':a[0,1,0]=0
    if failure=='migration':a[0,1,3]=1
    if failure=='overlap':a=np.repeat(a,4,axis=0)
    if failure=='SM_count':a=np.concatenate([a,records([(1,5,4)])])
    with pytest.raises(ValueError):analyze(a,sm_count=1)


def test_instrumentation_does_not_add_sync_or_partial_payload():
    source=(ROOT/'csrc/sm80/roof_cta_timeline.cuh').read_text()
    assert '%globaltimer' in source and '%smid' in source
    assert 'y+size_t(m)*n' in source and 'threadIdx.x&31u' in source
    assert '__syncthreads' not in source and '__threadfence' not in source
    for name in ('o3','o78'):
        source=(ROOT/f'csrc/sm80/roof_{name}_cta_timeline_probe.cu').read_text()
        assert 'stamp<false>' in source and 'stamp<true>' in source
        assert '__launch_bounds__(128,3)' in source


def test_capacity_gate_rejects_new_spill_or_registers():
    loop=dict(kind='integer',static_instructions=380,opcode_counts={
        'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16,'BAR.SYNC':1})
    old=dict(allocated_gpr=168,loops=[loop])
    assert instrumentation_gate(old,old)['passed']
    assert not instrumentation_gate(old,dict(old,allocated_gpr=176))['passed']
    newloop=dict(loop,opcode_counts=dict(loop['opcode_counts'],LDL=1))
    assert not instrumentation_gate(old,dict(old,loops=[newloop]))['passed']


def test_guard_uint32_reduction_is_untimed_host_check():
    source=(ROOT/'scripts/profile_cta_timeline.py').read_text()
    assert 'np.any(status.cpu().numpy())' in source and 'status.any()' not in source
