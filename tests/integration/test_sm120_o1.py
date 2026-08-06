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
    cpu = PreparedInputs("o1_native_small", a_int8, a_scale, w_mxfp4, w_scale)
    cuda = PreparedInputs(
        cpu.sample_id,
        cpu.A_int8.cuda(),
        cpu.A_scale.cuda(),
        cpu.W_mxfp4.cuda(),
        cpu.W_scale.cuda(),
    )
    return cpu, cuda


@pytest.mark.sm120
def test_o1_native_matches_semantic_reference():
    import torch

    from adangel.ops.dispatch import run_o1
    from adangel.reference import run_o1_reference

    cpu_inputs, cuda_inputs = _inputs()
    expected = run_o1_reference(cpu_inputs)
    actual = run_o1(cuda_inputs, mode="compute_only", backend="native")
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual["converted_weight"].cpu(), expected["converted_weight"], rtol=0, atol=0
    )
    assert actual["converted_activation"] is None
    torch.testing.assert_close(actual["output"].cpu(), expected["output"], rtol=1e-3, atol=1e-3)
    assert actual["output"].dtype == torch.float32
    assert torch.isfinite(actual["output"]).all()
    kernel = dict(actual["kernel"])
    assert kernel["tensor_core"] is True
    assert kernel["library"] == "CUDA WMMA"
    assert kernel["mma_family"] == "IMMA"
    assert kernel["mma_api"] == "nvcuda::wmma"
    assert kernel["mma_shape"] == "m16n16k16"
    assert kernel["implementation"] == "fused_tiled"
    assert kernel["kernel_symbol"] == "adangel_o1_fused_tiled"
    assert kernel["compute_type"] == "S8xS8_TO_S32"
    assert kernel["partial_dtype"] == "int32"
    assert kernel["accumulation_dtype"] == "fp32"
    assert kernel["output_dtype"] == "fp32"
    assert kernel["group_size"] == 32
    assert kernel["global_partial_buffer"] is False
    assert kernel["output_stores_per_element"] == 1


@pytest.mark.sm120
@pytest.mark.parametrize(
    ("mode", "expected_stages"),
    [
        ("conversion_only", {"weight_conversion", "total"}),
        ("compute_only", {"gemm", "total"}),
        ("cold", {"weight_conversion", "gemm", "total"}),
        ("steady_state", {"gemm", "total"}),
    ],
)
def test_o1_native_timing_modes(mode, expected_stages):
    import torch

    from adangel.reference import run_o1_reference

    cpu_inputs, cuda_inputs = _inputs()
    expected = run_o1_reference(cpu_inputs)
    native = require_variant("o1")
    result = native.benchmark(
        "o1",
        mode,
        cuda_inputs.A_int8,
        cuda_inputs.A_scale,
        cuda_inputs.W_mxfp4,
        cuda_inputs.W_scale,
        2,
        3,
    )
    torch.cuda.synchronize()
    timings = dict(result["timings_ms"])
    assert set(timings) == expected_stages
    assert all(
        len(values) == 3 and all(value > 0 for value in values)
        for values in timings.values()
    )
    torch.testing.assert_close(result["output"].cpu(), expected["output"], rtol=1e-3, atol=1e-3)
