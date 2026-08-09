import pytest

from adangel.ops.extension import require_variant


def _inputs(m=128, n=128, k=256, pattern="random"):
    import torch

    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4
    from adangel.trace.schema import PreparedInputs

    generator = torch.Generator().manual_seed(2026)
    if pattern == "random":
        activation = torch.randn((m, k), generator=generator, dtype=torch.float16)
        a_int8, a_scale = quantize_int8_per_row(activation)
    else:
        if pattern == "zero":
            a_int8 = torch.zeros((m, k), dtype=torch.int8)
        elif pattern == "alternating":
            values = torch.tensor([-127, 127], dtype=torch.int8).repeat(k // 2)
            a_int8 = values.repeat(m, 1)
        elif pattern == "saturated":
            a_int8 = torch.full((m, k), 127, dtype=torch.int8)
            a_int8[:, 1::2] = -127
        else:
            raise ValueError(pattern)
        a_scale = torch.linspace(2.0**-8, 2.0**4, m, dtype=torch.float32)
    weight = torch.randn((n, k), generator=generator, dtype=torch.float16)
    w_mxfp4, w_scale = quantize_mxfp4(weight)
    cpu = PreparedInputs("o2_native_small", a_int8, a_scale, w_mxfp4, w_scale)
    cuda = PreparedInputs(
        cpu.sample_id,
        cpu.A_int8.cuda(),
        cpu.A_scale.cuda(),
        cpu.W_mxfp4.cuda(),
        cpu.W_scale.cuda(),
    )
    return cpu, cuda


def _assert_matches_reference(cpu_inputs, cuda_inputs):
    import torch

    from adangel.ops.dispatch import run_o2
    from adangel.reference import run_o2_reference

    expected = run_o2_reference(cpu_inputs)
    actual = run_o2(cuda_inputs, mode="compute_only", backend="native")
    torch.cuda.synchronize()
    actual_values, actual_scales = actual["converted_activation"]
    expected_values, expected_scales = expected["converted_activation"]
    torch.testing.assert_close(actual_values.cpu(), expected_values, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales.cpu(), expected_scales, rtol=0, atol=0)
    assert actual["converted_weight"] is None
    torch.testing.assert_close(actual["output"].cpu(), expected["output"], rtol=1e-3, atol=1e-3)
    assert actual["output"].dtype == torch.float32
    assert torch.isfinite(actual["output"]).all()
    return actual


@pytest.mark.sm120
def test_o2_native_matches_semantic_reference():
    cpu_inputs, cuda_inputs = _inputs()
    actual = _assert_matches_reference(cpu_inputs, cuda_inputs)
    kernel = dict(actual["kernel"])
    assert kernel["library"] == "CUTLASS"
    assert kernel["implementation"] == "cutlass_sm120_mxf4_tma_warp_specialized"
    assert kernel["data_movement"] == "TMA"
    assert kernel["kernel_schedule"] == "cooperative_warp_specialized"
    assert kernel["stage_count_policy"] == "StageCountAutoCarveout"
    assert tuple(kernel["cta_tile"]) == (128, 128, 256)
    assert tuple(kernel["cluster"]) == (1, 1, 1)
    assert kernel["tensor_core"] is True
    assert kernel["mma_family"] == "MXFP4_BLOCK_SCALED"
    assert kernel["mma_shape"] == "m16n8k64"
    assert kernel["input_dtype"] == "mxfp4_e2m1"
    assert kernel["scale_dtype"] == "ue8m0"
    assert kernel["scale_vector_size"] == 32
    assert kernel["accumulation_dtype"] == "fp32"
    assert kernel["output_dtype"] == "fp32"
    assert kernel["weight_scale_layout_repack"] is True
    assert kernel["weight_scale_repack_timing_method"] == "batched_cuda_event_average"
    assert kernel["weight_scale_repack_timing_isolated"] is True
    assert kernel["weight_scale_repack_inner_repeats"] == 100
    assert kernel["total_timing_semantics"] == "direct_single_weight_scale_repack"
    assert kernel["global_partial_buffer"] is False
    assert kernel["output_stores_per_element"] == 1


@pytest.mark.sm120
@pytest.mark.parametrize("pattern", ["zero", "alternating", "saturated"])
def test_o2_activation_edge_patterns(pattern):
    cpu_inputs, cuda_inputs = _inputs(pattern=pattern)
    _assert_matches_reference(cpu_inputs, cuda_inputs)


@pytest.mark.sm120
def test_o2_cutlass_handles_mn_residue():
    cpu_inputs, cuda_inputs = _inputs(m=132, n=132, k=256)
    _assert_matches_reference(cpu_inputs, cuda_inputs)


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
def test_o2_native_timing_modes(mode, expected_stages):
    import torch

    from adangel.reference import run_o2_reference

    cpu_inputs, cuda_inputs = _inputs()
    native = require_variant("o2")
    result = native.benchmark(
        "o2",
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
    expected = run_o2_reference(cpu_inputs)
    torch.testing.assert_close(result["output"].cpu(), expected["output"], rtol=1e-3, atol=1e-3)


@pytest.mark.sm120
def test_o2_rejects_ue8m0_nan_scale():
    cpu_inputs, cuda_inputs = _inputs()
    del cpu_inputs
    cuda_inputs.W_scale[0, 0] = 255
    native = require_variant("o2")
    with pytest.raises(RuntimeError, match="UE8M0 NaN"):
        native.run_o2(
            cuda_inputs.A_int8,
            cuda_inputs.A_scale,
            cuda_inputs.W_mxfp4,
            cuda_inputs.W_scale,
            "compute_only",
        )


@pytest.mark.sm120
def test_o2_rejects_nonfinite_activation_scale():
    cpu_inputs, cuda_inputs = _inputs()
    del cpu_inputs
    cuda_inputs.A_scale[0] = float("nan")
    native = require_variant("o2")
    with pytest.raises(RuntimeError, match="A_scale contains NaN/Inf"):
        native.run_o2(
            cuda_inputs.A_int8,
            cuda_inputs.A_scale,
            cuda_inputs.W_mxfp4,
            cuda_inputs.W_scale,
            "compute_only",
        )
