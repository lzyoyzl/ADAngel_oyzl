from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_o78_address_remat_codegen import (
    OLD, NEW, address_offsets, generated_header, opcode_count, runtime_justified,
)
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


def test_spill_gate_counts_width_suffixes_and_only_integer_loop():
    old = dict(allocated_gpr=168, loops=[
        dict(kind='integer', opcode_counts={'LDL.64': 2, 'LDL': 1, 'LDS.U': 8}),
        dict(kind='fp32_fallback', opcode_counts={'LDL.128': 20}),
    ])
    candidate = dict(allocated_gpr=168, loops=[
        dict(kind='integer', opcode_counts={'LDL.64': 1, 'LDS.U': 8}),
        dict(kind='fp32_fallback', opcode_counts={'LDL.128': 99}),
    ])
    assert opcode_count(old['loops'][0], 'LDL') == 3
    assert runtime_justified(candidate, old)
    assert not runtime_justified(old, old)
    assert not runtime_justified(dict(candidate, allocated_gpr=169), old)


def test_paired_runtime_preserves_preparation_and_rejects_negative_compile_gate():
    from benchmark_o78_address_remat import timing_contract
    from benchmark_o78_eight_chain_probe import timing_contract as previous
    for mode in ('conversion_only', 'compute_only', 'cold', 'steady_state'):
        current = timing_contract(mode, 100)
        assert current.pop('a_copy_address_rematerialization')
        assert current.pop('cta_order_group_m') == 8
        assert current.pop('new_preparation_or_layout') is False
        current['comparison'] = previous(mode, 100)['comparison']
        assert current == previous(mode, 100)
    source = (ROOT / 'scripts/benchmark_o78_address_remat.py').read_text()
    assert "if not receipt['worth_runtime_validation']" in source
    assert "if not encoded_control['passed']" in source
    assert 'full_sample_args()' in source and 'validation_fn=validate' in source
    assert 'self.handles[0] = self.handles[1]' in source
    assert 'default_gpu_build=Path(\'reports/o378_roof_v73_codegen\')' in source
