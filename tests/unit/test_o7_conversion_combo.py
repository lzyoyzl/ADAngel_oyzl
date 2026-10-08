from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_o7_conversion_combo import route, timing_contract
from probe_mx8_swar_codegen import generated_host


def test_frozen_control_selected_never_failed_mx8_candidate():
    assert route(0)==('roof_o78_nv4_swar_benchmark',1)
    assert route(1)==('roof_o78_mx8_swar_benchmark',0)
    assert route(2)==('roof_o78_row_fused_benchmark',1)
    for bad in (-1,3):
        with pytest.raises(ValueError):route(bad)
    host=generated_host()
    assert 'if(candidate)packed_activation();else lookup_activation();' in host
    assert 'void weight(bool) {Nv4SwarOnline::weight(true);}' in host
    assert 'mx8_warp_lut_probe::adangel_sm80_row_warp_lut_metadata<Kind::Mx8,16>' in host


@pytest.mark.parametrize('mode',('conversion_only','compute_only','cold','steady_state'))
def test_fixed_timing_and_scope(mode):
    c=timing_contract(mode,100)
    assert c['gemm_cufunction_identical_between_policies']
    assert c['weight_preparation_identical'] and c['payload_norm_checked_after_every_call']
    assert not c['failed_v126_packed_MX8_candidate_executed']
    assert c['stage_timing_inner_repeats']['total']==(100 if mode=='conversion_only' else 1)
    assert c['modified_stage']=='activation_conversion_selection_only'


def test_exact_event_body_reused():
    old=(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    old=old[old.index('extern "C" int roof_o78_row_fused_benchmark('):]
    host=generated_host();actual=host[host.index('extern "C" int roof_o78_mx8_swar_benchmark('):]
    assert actual.replace('roof_o78_mx8_swar_benchmark','roof_o78_row_fused_benchmark').replace(
        'Mx8SwarOnline x(','RowFusedOnline x(')==old
