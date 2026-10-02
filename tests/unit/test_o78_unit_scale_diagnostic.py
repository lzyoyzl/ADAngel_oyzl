from pathlib import Path
import sys

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from probe_o78_unit_scale_codegen import unit_header
from benchmark_o78_unit_scale_probe import require_unit_values, summarize


def test_only_coefficient_semantics_change_not_payload_or_mma():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    generated=unit_header(source)
    assert 'constexpr int coefficient=1' in generated
    assert generated.count('cute::gemm(')==source.count('cute::gemm(')==4
    assert generated.split('__device__ __forceinline__ void body',1)[0].split('void prefetch',1)[1] == source.split('__device__ __forceinline__ void body',1)[0].split('void prefetch',1)[1]
    assert generated.replace('o78_unit_scale_diagnostic','o78_fullk_integer_experiment').split('// Reuse the 64 INT32',1)[1] == source.split('// Reuse the 64 INT32',1)[1]
    with pytest.raises(ValueError): unit_header(source.replace('const int coefficient=', 'const auto coefficient='))


def test_unit_scale_integer_and_fp32_exactness_bounds():
    assert 4096*128*8 == 4194304 < 2**24 < 2**31
    for a in range(-128,128):
        u=a & 255
        high=((u>>4)^8)-8
        for w in range(-8,8):
            assert a*w == (u&15)*w+16*high*w
            assert int(float(a*w*4096))==a*w*4096


def test_nonunit_input_or_fallback_rejected():
    require_unit_values([np.ones((32,64)),np.ones((32,128))],np.zeros(2))
    for arrays,status in [([],np.zeros(1)),([np.array([2])],np.zeros(1)),
                          ([np.array([1])],np.ones(1)),([np.array([np.nan])],np.zeros(1))]:
        with pytest.raises(ValueError,match='rejects non-unit'): require_unit_values(arrays,status)


def test_diagnostic_summary_does_not_claim_original_speedup():
    rows=[]
    for v in ('o7','o8'):
        for p in (0,1):
            rows.append(dict(payload_origin=v,sample_id='x',round=0,policy=p,unit_reference_mse=0,
                summary=dict(median_ms=1 if p else 2,cv_percent=5)))
    s=summarize(rows)
    assert s[1]['paired_ratio']==2 and s[1]['cv_failed']==1
    assert all('NOT_original' in r['interpretation'] for r in s)
    with pytest.raises(ValueError):summarize(rows[:-1])
