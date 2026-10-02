from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from probe_o78_unit_scale_codegen import unit_header


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
