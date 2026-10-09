"""CPU release checks: accepted bodies, native-only dispatch, unchanged SM120."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))


def test_frozen_accepted_bodies():
    from generate_sm80_production import contents
    for name, expected in contents().items():
        assert (ROOT/'csrc/sm80/production_generated'/name).read_text() == expected


def test_native_production_entry_and_guards():
    host = (ROOT/'csrc/sm80/production_benchmark.cuh').read_text()
    gemm = (ROOT/'csrc/sm80/production_gemm.cu').read_text()
    assert 'status.max().item<int>()<=1' in host
    assert 'state[14].max().item<int>()<=1' in host
    assert 'integer_guard_in_conversion_timing' in host
    assert 'api::o3_prepare' in host and 'api::mixed_convert' in host
    assert 'adangel_sm80_o3_fullk_grouped' in gemm
    assert 'adangel_sm80_o78_fullk_streaming' in gemm
    assert 'if(flag==1u)' in gemm and 'if(flag&1u)' in gemm
    assert 'reports/' not in host and 'ctypes' not in host


def test_dispatch_retains_explicit_legacy():
    host = (ROOT/'csrc/sm80/o1_o3.cu').read_text()
    mixed = (ROOT/'csrc/sm80/mixed_benchmark.cuh').read_text()
    assert 'return production::o3(' in host
    assert 'if(implementation=="legacy")implementation="production"' in host
    assert 'return production::mixed(' in mixed
    assert 'implementation=="legacy"' in mixed
    # Canonical SM120 dispatch is deliberately not redirected to A100.
    assert 'adangel._sm120' in (ROOT/'python/adangel/ops/extension.py').read_text()
