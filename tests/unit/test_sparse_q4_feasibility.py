import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from analyze_sparse_q4_feasibility import (
    paired_decomposition, weight_statistics, scalar_residual_model, summarize, VALID_METADATA,
)


def test_exhaustive_zero_patterns_minimal_residual_and_valid_metadata():
    for mask in range(256):
        group = np.array([-3 if i & 1 else 6 for i in range(8)], dtype=np.int8)
        group *= np.array([(mask >> i) & 1 for i in range(8)], dtype=np.int8)
        q = np.tile(group, (2, 16))
        main, residual, metadata, counts = paired_decomposition(q)
        assert np.array_equal(main.astype(np.int16) + residual, q)
        assert set(metadata.ravel().tolist()) <= VALID_METADATA
        minimum = min(sum(counts[0, 0, i] for i in range(4) if i not in selected)
                      for selected in itertools.combinations(range(4), 2))
        assert np.count_nonzero(residual[0, :8]) == minimum
        assert np.count_nonzero(main.reshape(2, 16, 4, 2).any(-1), axis=-1).max() <= 2


def test_exact_integer_dot_and_chunk_independence():
    rng = np.random.default_rng(20261007)
    q = rng.integers(-8, 8, (7, 256), dtype=np.int8)
    a = rng.integers(-128, 128, (5, 256), dtype=np.int8).astype(np.int64)
    main, residual, _, _ = paired_decomposition(q)
    np.testing.assert_array_equal(a @ q.astype(np.int64).T,
                                  a @ main.astype(np.int64).T + a @ residual.astype(np.int64).T)
    assert weight_statistics(q, 1) == weight_statistics(q, 64)


def test_zero_and_dense_limits():
    z = weight_statistics(np.zeros((1, 128), dtype=np.int8))
    assert z['minimum_residual_fraction'] == 0 and z['g128_without_residual_fraction'] == 1
    full = weight_statistics(np.ones((1, 128), dtype=np.int8))
    assert full['minimum_residual_fraction'] == .5
    assert full['chunks_requiring_residual_fraction'] == 1
    assert full['minimum_residual_nonzeros_per8_histogram'] == [0, 0, 0, 0, 16]


@pytest.mark.parametrize('q', [np.ones((1, 8), dtype=np.int8),
    np.ones((1, 128), dtype=np.float32), np.ones((1, 128), dtype=np.int8) * 8,
    np.empty((0, 128), dtype=np.int8), np.ones(128, dtype=np.int8)])
def test_invalid_inputs_fail_closed(q):
    with pytest.raises(ValueError):
        paired_decomposition(q)


def test_scalar_model_not_kernel_roof_or_speedup():
    model = scalar_residual_model(4096**2, 4096**2 // 2)
    assert model['scalar_correction_only_optimistic_ms'] == pytest.approx(1.762774762)
    assert model['ideal_four_mac_vector_instruction_ms'] == pytest.approx(
        model['scalar_correction_only_optimistic_ms'] / 4)
    assert not model['measured_kernel_peak'] and not model['measured_speedup']
    with pytest.raises(ValueError):
        scalar_residual_model(5, 6)


def test_summary_requires_complete_full24():
    with pytest.raises(ValueError):
        summarize([])
