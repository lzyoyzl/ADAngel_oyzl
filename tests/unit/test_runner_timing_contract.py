import pytest

from adangel.benchmark.runner import _validate_benchmark_payload


def _payload(mode: str, inner_repeats: int = 100) -> dict:
    total_method = (
        "batched_cuda_event_average"
        if mode == "conversion_only"
        else "direct_single_path"
    )
    return {
        "output": object(),
        "timings_ms": {"total": [1.0, 1.0, 1.0]},
        "timing_method": {
            "strategy": "conversion_amortized_end_to_end_direct",
            "conversion_inner_repeats": inner_repeats,
            "mode_total_timing": total_method,
        },
    }


@pytest.mark.parametrize(
    "mode",
    ["conversion_only", "compute_only", "cold", "steady_state"],
)
def test_validate_benchmark_payload_accepts_dual_track_contract(mode):
    _validate_benchmark_payload(_payload(mode), 3, mode, 100)


def test_validate_benchmark_payload_rejects_direct_conversion_only_total():
    payload = _payload("conversion_only")
    payload["timing_method"]["mode_total_timing"] = "direct_single_path"
    with pytest.raises(RuntimeError, match="wrong total timing method"):
        _validate_benchmark_payload(payload, 3, "conversion_only", 100)


def test_validate_benchmark_payload_rejects_inner_repeat_mismatch():
    with pytest.raises(RuntimeError, match="does not match config"):
        _validate_benchmark_payload(_payload("cold", inner_repeats=50), 3, "cold", 100)
