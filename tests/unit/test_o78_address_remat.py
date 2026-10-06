from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_o78_address_remat_codegen import OLD, NEW, address_offsets, generated_header
from probe_grouped_cta_codegen import generated_headers


def test_only_a_address_expressions_change():
    expected = generated_headers('o78')[0]
    actual = generated_header()
    assert actual.replace(NEW, OLD).replace('o78_address_remat_experiment', 'o78_grouped_cta_experiment') == expected
    assert actual.count('roof_address_remat::copy16<false>') == 1
    assert actual.count('roof_address_remat::copy16<true>') == 1
    assert 's.activation_factors' in actual and 's.weight_factors' in actual
    assert 'cute::gemm(HA{}' in actual and 'cute::gemm(LA{}' in actual


@pytest.mark.parametrize('m,k,y', [(256,4096,0),(576,4096,8),(4096,4096,63),
                                   (1048577,4096,16384),(33554433,4096,524287)])
def test_original_pointer_component_arithmetic_is_exact(m,k,y):
    for group in range(32):
        for thread in range(128):
            for chunk in range(2):
                original, changed = address_offsets(group,m,k,y,thread,chunk)
                assert original == changed
                assert all(value % 16 == 0 for value in changed)
    # Large component sums must not be accidentally truncated to uint32.
    if m > 1048576:
        assert address_offsets(31,m,k,y,127,1)[1][1] > 2**32


def test_helper_keeps_async_copy_width_cache_and_synchronization():
    helper = (ROOT / 'csrc/sm80/roof_cp_async_offset.cuh').read_text()
    assert helper.count('cp.async.cg.shared.global [%0],[address],16;') == 2
    assert 'mad.wide.u32' in helper and 'if constexpr(High)' in helper
    assert 'memory' in helper and '__cvta_generic_to_shared' in helper
    for forbidden in ('ld.param', 'bar.sync', 'wait_group', '__syncthreads', 'cudaMalloc'):
        assert forbidden not in helper
    entry = (ROOT / 'csrc/sm80/roof_o78_address_remat_probe.cu').read_text()
    assert '__launch_bounds__(128,3)' in entry
    assert 'status[tile.y*(n/128)+tile.x]' in entry
    assert 'o78_grouped_fallback::o3_body' in entry
    assert 'o78_address_remat_experiment::body' in entry
