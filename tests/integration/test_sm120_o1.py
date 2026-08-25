import pytest

from adangel.ops.extension import require_variant


def _inputs(m=128, n=128, k=64):
    import torch

    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4
    from adangel.trace.schema import PreparedInputs

    generator = torch.Generator().manual_seed(2026)
    activation = torch.randn((m, k), generator=generator, dtype=torch.float16)
    weight = torch.randn((n, k), generator=generator, dtype=torch.float16)
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


def _pattern_inputs(pattern, m=16, n=8, k=96):
    import torch

    from adangel.trace.schema import PreparedInputs

    if pattern == "zero":
        a_int8 = torch.zeros((m, k), dtype=torch.int8)
        w_mxfp4 = torch.zeros((n, k // 2), dtype=torch.uint8)
    elif pattern == "alternating":
        columns = torch.arange(k)
        a_int8 = torch.where(columns % 2 == 0, 127, -127).to(torch.int8).repeat(m, 1)
        # Low nibble +6, high nibble -6.
        w_mxfp4 = torch.full((n, k // 2), 0xF7, dtype=torch.uint8)
    elif pattern == "saturated":
        a_int8 = torch.full((m, k), 127, dtype=torch.int8)
        w_mxfp4 = torch.full((n, k // 2), 0x77, dtype=torch.uint8)
    else:
        raise ValueError(pattern)
    a_scale = torch.linspace(0.25, 2.0, m, dtype=torch.float32)
    rows = torch.arange(n, dtype=torch.int64)[:, None]
    groups = torch.arange(k // 32, dtype=torch.int64)[None, :]
    w_scale = (124 + (rows + groups) % 7).to(torch.uint8).contiguous()
    cpu = PreparedInputs("o1_pattern", a_int8, a_scale, w_mxfp4, w_scale)
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
    assert kernel["library"] == "CUTLASS CuTe + CUDA"
    assert kernel["mma_family"] == "IMMA"
    assert kernel["mma_api"] == "cute::MMA_Atom"
    assert kernel["mma_atom"] == "SM80_16x8x32_S32S8S8S32_TN"
    assert kernel["mma_shape"] == "m16n8k32"
    assert kernel["implementation"] == "tma_warp_specialized_register_partial"
    assert kernel["implementation_key"] == "register_64x32"
    assert kernel["kernel_symbol"] == "adangel_o1_register_partial_64x32"
    assert kernel["partial_storage"] == "register"
    assert kernel["shared_partial_redistribution"] is False
    assert kernel["production_selected"] is True
    assert tuple(kernel["cta_tile"]) == (64, 32, 32)
    assert kernel["data_movement"] == "TMA"
    assert kernel["kernel_schedule"] == "cooperative_warp_specialized"
    assert kernel["pipeline_stages"] == 3
    assert kernel["producer_warps"] == 1
    assert kernel["consumer_warps"] == 8
    assert tuple(kernel["tma_operands"]) == ("A_int8", "W_int8")
    assert kernel["compute_type"] == "S8xS8_TO_S32"
    assert kernel["partial_dtype"] == "int32"
    assert kernel["accumulation_dtype"] == "fp32"
    assert kernel["output_dtype"] == "fp32"
    assert kernel["group_size"] == 32
    assert kernel["global_partial_buffer"] is False
    assert kernel["output_stores_per_element"] == 1


@pytest.mark.sm120
def test_o1_tma_zero_fills_partial_mn_tiles():
    import torch

    from adangel.ops.dispatch import run_o1
    from adangel.reference import run_o1_reference

    cpu_inputs, cuda_inputs = _inputs(m=68, n=36, k=64)
    expected = run_o1_reference(cpu_inputs)
    actual = run_o1(cuda_inputs, mode="compute_only", backend="native")
    torch.cuda.synchronize()
    torch.testing.assert_close(actual["output"].cpu(), expected["output"], rtol=1e-3, atol=1e-3)


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
    timing_method = dict(result["timing_method"])
    assert timing_method["strategy"] == "conversion_amortized_end_to_end_direct"
    assert timing_method["conversion_inner_repeats"] == 100
    assert timing_method["mode_total_timing"] == (
        "batched_cuda_event_average"
        if mode == "conversion_only"
        else "direct_single_path"
    )
    torch.testing.assert_close(result["output"].cpu(), expected["output"], rtol=1e-3, atol=1e-3)


@pytest.mark.sm120
@pytest.mark.parametrize(
    "implementation",
    [
        "register_64x32",
        "register_64x32_scale_shared",
        "register_64x32_k64_scale_shared",
        "register_128x128",
    ],
)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(16, 8, 32), (64, 32, 64), (68, 36, 96)],
)
def test_o1_register_partial_candidates_match_reference(implementation, m, n, k):
    import torch

    from adangel.ops.extension import require_variant
    from adangel.reference import run_o1_reference

    cpu_inputs, cuda_inputs = _inputs(m=m, n=n, k=k)
    expected = run_o1_reference(cpu_inputs)
    native = require_variant("o1")
    result = native._benchmark_o1_impl(
        implementation,
        "compute_only",
        cuda_inputs.A_int8,
        cuda_inputs.A_scale,
        cuda_inputs.W_mxfp4,
        cuda_inputs.W_scale,
        0,
        1,
        100,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        result["converted_weight"].cpu(), expected["converted_weight"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        result["output"].cpu(), expected["output"], rtol=1e-3, atol=1e-3
    )
    kernel = dict(result["kernel"])
    assert kernel["implementation_key"] == implementation
    scale_shared = implementation in {
        "register_64x32_scale_shared",
        "register_64x32_k64_scale_shared",
    }
    pipeline_k64 = implementation == "register_64x32_k64_scale_shared"
    assert kernel["implementation"] == (
        "tma_warp_specialized_register_partial_scale_shared"
        if scale_shared
        else "tma_warp_specialized_register_partial"
    )
    assert kernel["library"] == "CUTLASS CuTe + CUDA"
    assert kernel["mma_api"] == "cute::MMA_Atom"
    assert kernel["mma_atom"] == "SM80_16x8x32_S32S8S8S32_TN"
    assert kernel["mma_shape"] == "m16n8k32"
    assert kernel["partial_storage"] == "register"
    assert kernel["shared_partial_redistribution"] is False
    assert kernel["global_partial_buffer"] is False
    assert kernel["output_stores_per_element"] == 1
    assert kernel["production_selected"] is (implementation == "register_64x32")
    assert kernel["column_scale_load_scope"] == (
        "cta_once_per_column_group" if scale_shared else "consumer_warp"
    )
    assert kernel["fp32_accumulation_op"] == (
        "fma_rn" if scale_shared else "mul_then_add_rn"
    )
    assert kernel["groups_per_pipeline_stage"] == (2 if pipeline_k64 else 1)
    assert kernel["pipeline_tile_k"] == (64 if pipeline_k64 else 32)


@pytest.mark.sm120
@pytest.mark.parametrize(
    "implementation",
    [
        "register_64x32",
        "register_64x32_scale_shared",
        "register_64x32_k64_scale_shared",
        "register_128x128",
    ],
)
@pytest.mark.parametrize("pattern", ["zero", "alternating", "saturated"])
def test_o1_register_partial_scale_coordinate_patterns(implementation, pattern):
    import torch

    from adangel.ops.extension import require_variant
    from adangel.reference import run_o1_reference

    cpu_inputs, cuda_inputs = _pattern_inputs(pattern)
    expected = run_o1_reference(cpu_inputs)
    native = require_variant("o1")
    result = native._benchmark_o1_impl(
        implementation,
        "compute_only",
        cuda_inputs.A_int8,
        cuda_inputs.A_scale,
        cuda_inputs.W_mxfp4,
        cuda_inputs.W_scale,
        0,
        1,
        100,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        result["output"].cpu(), expected["output"], rtol=1e-3, atol=1e-3
    )
    assert result["output"].dtype == torch.float32
    assert torch.isfinite(result["output"]).all()
