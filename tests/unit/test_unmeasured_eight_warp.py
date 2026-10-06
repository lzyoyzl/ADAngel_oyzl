from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'python'))
from validate_unmeasured_eight_warp import resources_gate,model_rationale,summarize


def test_actual_resources_not_launch_bounds_gate():
    rows={(k,p):dict(threads=(128,256)[p],local_size_bytes=0,active_blocks_per_sm=(3,2)[p])
          for k in ('o3','o78') for p in (0,1)}
    assert resources_gate(rows)
    rows['o78',1]['active_blocks_per_sm']=1
    assert not resources_gate(rows)
    with pytest.raises(ValueError):resources_gate({})


def test_no_static_gate_rewrite_or_speedup_claim():
    r=model_rationale()
    assert not r['v98_original_compile_gate_changed']
    assert r['v98_original_heuristic_failed'] and r['previous_candidate_GPU_launches']==0
    assert r['shared_capacity_estimate_ms']['o3']>.22
    assert r['no_new_optimization_or_parameter_scan']
    with pytest.raises(ValueError):summarize([])


def test_full24_pairing_and_MSE_aggregation():
    rows=[dict(sample_id=f'sample_{s}',variant=v,round=r,policy=p,
        summary=dict(median_ms=1 if p==0 else .8,cv_percent=0),
        bitwise_equal_control=True,finite_fp32=True,mse_vs_reference=.002)
        for s in range(24) for v in ('o3','o7','o8') for r in range(3) for p in (0,1)]
    out=summarize(rows)
    assert out['records']==432 and len(out['summary'])==6
    assert all(x['paired_speedup']==1.25 and x['median_MSE']==.002
               for x in out['summary'] if x['policy']==1)
    assert not out['original_v98_compile_gate_changed']
    with pytest.raises(ValueError):summarize(rows[:-1])


def test_pointer_ABI_order_matches_exact_o78_entry():
    from types import SimpleNamespace
    from benchmark_unmeasured_eight_warp import tensor_pointers_o78
    names=('a','w','as','ws','af','wf','ab','wb','status','y')
    c=SimpleNamespace(state={key:i for i,key in enumerate(names)})
    assert tensor_pointers_o78(c)==tuple(range(10))


def test_native_public_payload_and_o0_interface_contract():
    source=(ROOT/'scripts/validate_unmeasured_eight_warp.py').read_text()
    assert "base['packed_activation_g128_major']" not in source
    assert "base['packed_weight_g128_major']" not in source
    assert "reshape(2,m,32,64).permute(0,2,1,3).contiguous()" in source
    assert "split_int8_to_packed_int4(x.A_int8)" in source
    assert "base['converted_weight'],x.W_q4" in source
    runner=(ROOT/'scripts/benchmark_unmeasured_eight_warp.py').read_text()
    assert "native.benchmark('o0'" not in runner
    assert "native.benchmark_o0(" in runner
