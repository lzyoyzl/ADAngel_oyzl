from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
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
