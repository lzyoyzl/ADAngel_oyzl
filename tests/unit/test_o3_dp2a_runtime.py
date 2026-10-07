import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_o3_dp2a import reviewed,timing_check,stage_contract,CONTROL,SYMBOL


def receipt():
    return json.loads((ROOT/'docs/evidence/a100_o378_roof_v115/reports/o378_roof_v115_o3_dp2a_codegen/analysis.json').read_text())


def test_review_preserves_original_failure_and_exact_candidate():
    r=receipt();old=copy.deepcopy(r)
    out=reviewed(r)
    assert r==old and not out['original_gate']['passed']
    assert all(out['checks'].values())
    for kind in ('cubin','stores','math','work','gate'):
        bad=copy.deepcopy(r)
        if kind=='cubin':bad['cubin_sha256']='0'*64
        if kind=='stores':bad['liveness']['loop']['opcode_counts']['STL']=1
        if kind=='math':bad['liveness']['loop']['opcode_counts']['IMMA.16832.S8.S8']=1
        if kind=='work':bad['liveness']['loop']['static_instructions']=341
        if kind=='gate':bad['compile_gate']['passed']=True
        with pytest.raises(ValueError):reviewed(bad)


def test_pack_in_weight_path_and_all_allocations_outside_timing():
    text=(ROOT/'csrc/sm80/roof_o3_dp2a_driver.cpp').read_text()
    weight=text[text.index('auto cvw='):text.index('auto cva=')]
    assert 'guard_args' in weight and 'pack_args' in weight and 'if(policy)' in weight
    assert 'auto effective_meta=policy?joined:meta' in text
    assert text.index('Events direct')<text.index('cvw();cva();')
    assert 'if(weight)batch(cvw,wb,0)' in text and 'times[stage*repeats+i]/=inner' in text
    assert 'cuMemAlloc' not in text and 'cuCtxSynchronize' not in text
    py=(ROOT/'scripts/benchmark_o3_dp2a.py').read_text()
    assert 'pack_metadata_reference(meta.cpu().numpy())' in py
    assert 'three_paths_in_one_launch=True' in py and 'full_sample_args()' in py
    assert "('--warmup','1000')" in py and "('--rounds','3')" in py


def test_extra_pack_is_required_by_timing_contract_not_hidden():
    for symbol,kernels in ((CONTROL,2),(SYMBOL,3)):
        r=dict(stage_timing_inner_repeats=stage_contract('conversion_only',100),
            weight_cached=False,activation_prepared=False,
            kernel=dict(kernel_symbol=symbol,weight_conversion_kernels=kernels,activation_conversion_kernels=1),
            timings_ms=dict(weight_conversion=[.01,.01],activation_conversion=[.02,.02],total=[.03,.03]))
        timing_check(r,'conversion_only',100,2,1)
        r['kernel']['weight_conversion_kernels']+=1
        with pytest.raises(AssertionError):timing_check(r,'conversion_only',100,2,1)
