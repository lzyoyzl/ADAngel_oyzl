"""Offline exact-metadata and unfiltered timing contracts for independent v68."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("benchmark_o78_fullk_gpu_prepare", ROOT / "scripts/benchmark_o78_fullk_gpu_prepare.py")
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)
from o78_fullk_integer_metadata import prepare_fullk_metadata


def test_dual_timing_modes_preserve_cached_weight_and_single_direct_execution():
    for mode in probe.MODES:
        r = probe.timing_contract(mode, 100)
        assert r["stage_timing_inner_repeats"]["total"] == (100 if mode == "conversion_only" else 1)
        assert r["weight_cached"] == (mode in ("compute_only", "steady_state"))
        assert r["activation_prepared"] == (mode == "compute_only")
        assert r["total_timing"] == ("sum_of_batched_stage_samples" if mode == "conversion_only" else "single_execution_cuda_event")
        if "gemm" in r["stage_timing_inner_repeats"]:
            assert r["stage_timing_inner_repeats"]["gemm"] == 1
    with pytest.raises(ValueError):
        probe.timing_contract("cold", 1)


def test_conversion_total_sums_each_corresponding_sample_not_stage_medians():
    raw = np.array([[1, 3, 100], [100, 3, 1], [0, 0, 0], [101, 6, 101]], dtype=float)
    result = probe.normalize_timings("conversion_only", raw, 3)
    assert result["total"] == [101, 6, 101]
    assert np.median(result["total"]) != np.median(result["weight_conversion"]) + np.median(result["activation_conversion"])
    raw[3, 0] = 100
    with pytest.raises(ValueError, match="corresponding batch"):
        probe.normalize_timings("conversion_only", raw, 3)


def test_direct_total_is_not_replaced_with_stage_sum_and_all_values_retained():
    raw = np.array([[1, 1], [2, 2], [3, 3], [7, 50]], dtype=float)
    result = probe.normalize_timings("cold", raw, 2)
    assert result["total"] == [7, 50]
    assert len(result["total"]) == 2  # 50 is an outlier, not silently filtered.
    with pytest.raises(ValueError, match="must equal GEMM"):
        probe.normalize_timings("compute_only", raw, 2)
    raw[3] = raw[2]
    assert probe.normalize_timings("compute_only", raw, 2) == {"gemm": [3, 3], "total": [3, 3]}


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_selected_invalid_timings_reject(bad):
    raw = np.ones((4, 2)); raw[3, 0] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        probe.normalize_timings("compute_only", raw, 2)


def test_gpu_norm_saturation_matches_exact_host_and_rejection_mask():
    cpu = prepare_fullk_metadata([[127, 127]], [[56, 56]], [[2**80, 0]], [[1, 1]], activation_kind="ue8m0", weight_kind="e4m3")
    arrays = {}
    for side, prefix in (("activation", "a"), ("weight", "w")):
        expected = probe.expected_operand_gpu(cpu[side])
        for field, suffix in (("factors", "f"), ("bases", "b"), ("norms", "n"), ("maxima", "m"), ("row_status", "st")):
            arrays[prefix + suffix] = expected[field].copy()
    arrays["status"] = cpu["status_flat"].copy()
    result = probe.check_metadata_arrays(cpu, arrays)
    assert arrays["an"].tolist() == [probe.NORM_LIMIT]
    assert result["fallback_ctas"] == 1 and result["integer_ctas"] == 0
    arrays["status"].fill(0)
    with pytest.raises(ValueError, match="CTA guard"):
        probe.check_metadata_arrays(cpu, arrays)


def test_gpu_base_comparison_is_bitwise_not_allclose():
    cpu = prepare_fullk_metadata([[127]], [[56]], [[0]], [[0]], activation_kind="ue8m0", weight_kind="e4m3")
    arrays = {}
    for side, prefix in (("activation", "a"), ("weight", "w")):
        expected = probe.expected_operand_gpu(cpu[side])
        for field, suffix in (("factors", "f"), ("bases", "b"), ("norms", "n"), ("maxima", "m"), ("row_status", "st")):
            arrays[prefix + suffix] = expected[field].copy()
    arrays["status"] = cpu["status_flat"].copy()
    arrays["ab"] = np.nextafter(arrays["ab"], np.float32(2))
    with pytest.raises(ValueError, match="oracle mismatch: ab"):
        probe.check_metadata_arrays(cpu, arrays)


def test_factor_overflow_norm_sentinel_even_if_payload_is_zero():
    cpu = prepare_fullk_metadata([[127, 159]], [[56, 56]], [[0, 0]], [[0, 0]], activation_kind="ue8m0", weight_kind="e4m3")
    expected = probe.expected_operand_gpu(cpu["activation"])
    assert cpu["activation"]["norm2"] == [0]
    assert expected["norms"].tolist() == [probe.NORM_LIMIT]
    assert expected["maxima"].tolist() == [2**31 - 1]
    assert expected["row_status"].tolist() == [1]


def test_v68_rejects_extreme_activation_base_even_when_integer_bound_is_safe():
    cpu = prepare_fullk_metadata([[127]], [[56]], [[0]], [[0]], activation_kind="ue8m0", weight_kind="e4m3", activation_base_multiplier=2.0**97)
    assert cpu["status_flat"].tolist() == [0]
    expected = probe.expected_operand_gpu(cpu["activation"], activation=True)
    assert expected["row_status"].tolist() == [1]
    assert expected["bases"].tolist() == [1]
    assert not expected["factors"].any()
    assert probe.expected_v68_guard(cpu).tolist() == [1]


def test_v68_rejects_combined_base_exponents_without_changing_payload_scale():
    cpu = prepare_fullk_metadata([[127]], [[56]], [[1]], [[1]], activation_kind="ue8m0", weight_kind="e4m3", activation_base_multiplier=2.0**48, weight_base_multiplier=2.0**47)
    assert cpu["status_flat"].tolist() == [0]
    assert probe.expected_operand_gpu(cpu["activation"], activation=True)["row_status"].tolist() == [0]
    assert probe.expected_v68_guard(cpu).tolist() == [1]


def test_summary_requires_complete_paired_four_mode_coverage():
    with pytest.raises(ValueError, match="empty"):
        probe.summarize([])
    with pytest.raises(ValueError, match="incomplete"):
        probe.summarize([dict(sample_id="s0", variant="o7", mode="compute_only", round=0, candidate=0)])
