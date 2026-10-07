"""INT16 is a guarded storage optimization, never another quantization."""
import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from inspect_partial_handoff import observe, capacity_model


def test_exact_bound_boundary_and_old_guard():
    a = np.ones((128,32), dtype=np.int64) * 32767
    w = np.ones((256,32), dtype=np.int64) * 32767
    # S4 norm cannot exceed8192: use a product exactly crossing the bound
    # between adjacent possible integer norms, not an invalid S4 norm.
    w[:] = 8192
    a[:] = 32767**2 // 8192
    old = np.array([[0,1],[0,0]], dtype=np.uint32)
    assert observe(a,w,old)['narrow_integer_ctas'] == 3
    a[0,0] += 1
    result = observe(a,w,old)
    assert result['narrow_integer_ctas'] == 2
    assert result['wide_integer_ctas'] == 1
    assert result['fp32_fallback_ctas'] == 1
    json.dumps(result, allow_nan=False)


def test_zero_worst_case_and_invalid_guard():
    a = np.zeros((64,32), dtype=np.uint32)
    w = np.full((128,32),8192,dtype=np.uint32)
    old = np.zeros((1,1), dtype=np.uint32)
    assert observe(a,w,old)['data_gate_passed']
    a[:] = 2097152
    assert not observe(a,w,old)['data_gate_passed']
    old[:] = 2
    with pytest.raises(ValueError):
        observe(a,w,old)
    with pytest.raises(ValueError):
        observe(a.astype(np.float32),w,old)


def test_capacity_is_not_a_speedup_claim():
    m = capacity_model()
    assert m['candidate_shared_reservation_bytes'] == 51712
    assert m['handoff_int16_extra_bytes'] == 2**31
    assert m['handoff_int32_service_floor_ms'] == 2*m['handoff_int16_service_floor_ms']
    assert m['actual_candidate_performance_not_measured']
