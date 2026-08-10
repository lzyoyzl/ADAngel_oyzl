import pytest

from adangel.ops.extension import require_variant


def _inputs():
    import torch

    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4
    from adangel.trace.schema import PreparedInputs

    generator = torch.Generator().manual_seed(2026)
    activation = torch.randn((128, 64), generator=generator, dtype=torch.float16)
    weight = torch.randn((128, 64), generator=generator, dtype=torch.float16)
    a_int8, a_scale = quantize_int8_per_row(activation)
    w_mxfp4, w_scale = quantize_mxfp4(weight)
    return PreparedInputs(
        "o0_native_small",
        a_int8.cuda(),
        a_scale.cuda(),
        w_mxfp4.cuda(),
        w_scale.cuda(),
    )


@pytest.mark.sm120
def test_o0_native_matches_semantic_reference():
    import torch

    from adangel.ops.dispatch import run_o0
    from adangel.reference import run_o0_reference

    inputs = _inputs()
    expected = run_o0_reference(inputs)
    actual = run_o0(inputs, mode="compute_only", backend="native")
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual["converted_activation"], expected["converted_activation"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["converted_weight"], expected["converted_weight"], rtol=0, atol=0
    )
    torch.testing.assert_close(actual["output"], expected["output"], rtol=1e-3, atol=1e-3)
    assert actual["output"].dtype == torch.float32
    assert torch.isfinite(actual["output"]).all()
    assert dict(actual["kernel"])["tensor_core"] is True


@pytest.mark.sm120
@pytest.mark.parametrize(
    ("mode", "expected_stages"),
    [
        ("conversion_only", {"weight_conversion", "activation_conversion", "total"}),
        ("compute_only", {"gemm", "total"}),
        ("cold", {"weight_conversion", "activation_conversion", "gemm", "total"}),
        ("steady_state", {"activation_conversion", "gemm", "total"}),
    ],
)
def test_o0_native_timing_modes(mode, expected_stages):
    import torch

    from adangel.reference import run_o0_reference

    inputs = _inputs()
    native = require_variant("o0")
    result = native.benchmark(
        "o0", mode, inputs.A_int8, inputs.A_scale, inputs.W_mxfp4, inputs.W_scale, 2, 3
    )
    torch.cuda.synchronize()
    timings = dict(result["timings_ms"])
    assert set(timings) == expected_stages
    assert all(len(values) == 3 and all(value > 0 for value in values) for values in timings.values())
    timing_method = dict(result["timing_method"])
    assert timing_method["strategy"] == "conversion_amortized_end_to_end_direct"
    assert timing_method["conversion_inner_repeats"] == 100
    assert timing_method["mode_total_timing"] == (
        "batched_cuda_event_average"
        if mode == "conversion_only"
        else "direct_single_path"
    )
    torch.testing.assert_close(
        result["output"], run_o0_reference(inputs)["output"], rtol=1e-3, atol=1e-3
    )
