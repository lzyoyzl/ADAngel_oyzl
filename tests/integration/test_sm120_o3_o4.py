import pytest


def _inputs(m=32, n=16, k=128, seed=2026):
    import torch

    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import mxfp4_to_q4_packed, quantize_mxfp4
    from adangel.trace.schema import PreparedInputs

    generator = torch.Generator().manual_seed(seed)
    activation = torch.randn((m, k), dtype=torch.float16, generator=generator)
    weight = torch.randn((n, k), dtype=torch.float16, generator=generator)
    a_int8, a_scale = quantize_int8_per_row(activation)
    w_k32, scale_k32 = quantize_mxfp4(weight, group_size=32)
    w_g128, scale_g128 = quantize_mxfp4(weight, group_size=128)
    w_q4 = mxfp4_to_q4_packed(w_g128)
    cpu = PreparedInputs(
        "o3_o4_native_small", a_int8, a_scale, w_k32, scale_k32,
        w_g128, scale_g128, w_q4,
    )
    cuda = PreparedInputs(
        cpu.sample_id,
        cpu.A_int8.cuda(), cpu.A_scale.cuda(), cpu.W_mxfp4.cuda(), cpu.W_scale.cuda(),
        cpu.W_mxfp4_g128.cuda(), cpu.W_scale_g128.cuda(), cpu.W_q4.cuda(),
    )
    return cpu, cuda


@pytest.mark.sm120
@pytest.mark.parametrize("variant", ["o3", "o4"])
def test_arbitrary_bit_backend_matches_semantic_reference(variant):
    import torch

    from adangel.ops.dispatch import run_o3, run_o4
    from adangel.reference import run_o3_reference, run_o4_reference

    cpu, cuda = _inputs()
    reference = run_o3_reference(cpu) if variant == "o3" else run_o4_reference(cpu)
    result = run_o3(cuda, mode="compute_only") if variant == "o3" else run_o4(cuda, mode="compute_only")
    torch.cuda.synchronize()
    torch.testing.assert_close(result["output"].cpu(), reference["output"], rtol=1e-3, atol=1e-3)
    assert result["output"].dtype == torch.float32
    assert torch.isfinite(result["output"]).all()
    kernel = dict(result["kernel"])
    assert kernel["tensor_core"] is True
    assert kernel["data_movement"] == "TMA"
    assert kernel["kernel_schedule"] == "cooperative_warp_specialized"
    assert tuple(kernel["cta_tile"]) == (
        (128, 16, 128) if variant == "o3" else (64, 64, 512)
    )
    assert kernel["pipeline_stages"] == 2
    assert kernel["groups_per_pipeline_stage"] == (1 if variant == "o3" else 4)
    assert kernel["partial_storage"] == "register"
    assert kernel["global_partial_buffer"] is False
    assert kernel["output_stores_per_element"] == 1
    assert kernel["group_size"] == 128
    if variant == "o3":
        assert kernel["mma_family"] == "IMMA_INT4"
        assert tuple(kernel["mma_atoms"]) == (
            "SM80_16x8x64_S32U4S4S32_TN",
            "SM80_16x8x64_S32S4S4S32_TN",
        )
    else:
        assert kernel["mma_family"] == "BMMA"
        assert kernel["mma_atom"] == "SM80_16x8x128_S32U1U1S32_TN_ANDPOPC"
        assert kernel["logical_mma_per_group"] == 32
        assert kernel["bmma_accumulator_chains"] == 2
        assert kernel["b_fragment_cached"] is True
        assert tuple(kernel["activation_bit_weights"]) == (1, 2, 4, 8, 16, 32, 64, -128)
        assert tuple(kernel["weight_bit_weights"]) == (1, 2, 4, -8)


@pytest.mark.sm120
@pytest.mark.parametrize("variant", ["o3", "o4"])
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("conversion_only", {"weight_conversion", "activation_conversion", "total"}),
        ("compute_only", {"gemm", "total"}),
        ("cold", {"weight_conversion", "activation_conversion", "gemm", "total"}),
        ("steady_state", {"activation_conversion", "gemm", "total"}),
    ],
)
def test_arbitrary_bit_timing_contract(variant, mode, expected):
    import torch

    from adangel.ops.extension import require_variant

    _, inputs = _inputs()
    native = require_variant(variant)
    result = native.benchmark(
        variant, mode, inputs.A_int8, inputs.A_scale,
        inputs.W_mxfp4_g128, inputs.W_scale_g128, 1, 3, 100,
    )
    torch.cuda.synchronize()
    assert set(result["timings_ms"]) == expected
    assert all(len(values) == 3 for values in result["timings_ms"].values())
    timing = dict(result["timing_method"])
    assert timing["strategy"] == "conversion_amortized_end_to_end_direct"
    assert timing["mode_total_timing"] == (
        "batched_cuda_event_average" if mode == "conversion_only" else "direct_single_path"
    )


@pytest.mark.sm120
@pytest.mark.parametrize("variant", ["o3", "o4"])
def test_arbitrary_bit_twos_complement_extremes(variant):
    import torch

    from adangel.ops.dispatch import run_o3, run_o4
    from adangel.reference import run_o3_reference, run_o4_reference
    from adangel.trace.schema import PreparedInputs

    m, n, k = 17, 9, 128
    columns = torch.arange(k)
    a = torch.where(columns % 4 == 0, -128, torch.where(columns % 2 == 0, -1, 127))
    a = a.to(torch.int8).repeat(m, 1).contiguous()
    # E2M1 private values alternate -6, -1, 0, +6.  These exercise both
    # two's-complement sign planes while remaining reachable by the Q4 map.
    q = torch.tensor([-6, -1, 0, 6], dtype=torch.int8).repeat(k // 4)
    q = q.repeat(n, 1).contiguous()
    raw = q.to(torch.int16) & 0xF
    q4 = (raw[:, 0::2].to(torch.uint8) | (raw[:, 1::2].to(torch.uint8) << 4)).contiguous()
    scales = torch.arange(n, dtype=torch.int64)[:, None]
    scale_g128 = (124 + scales % 7).to(torch.uint8).contiguous()
    e2m1_codes = torch.tensor([15, 10, 0, 7], dtype=torch.uint8).repeat(k // 4)
    e2m1_codes = e2m1_codes.repeat(n, 1)
    w_g128 = (
        e2m1_codes[:, 0::2] | (e2m1_codes[:, 1::2] << 4)
    ).contiguous()
    cpu = PreparedInputs(
        "extreme", a, torch.linspace(0.25, 2.0, m),
        torch.zeros((n, k // 2), dtype=torch.uint8),
        torch.full((n, k // 32), 127, dtype=torch.uint8),
        w_g128, scale_g128, q4,
    )
    cuda = PreparedInputs(
        cpu.sample_id, cpu.A_int8.cuda(), cpu.A_scale.cuda(),
        cpu.W_mxfp4.cuda(), cpu.W_scale.cuda(), cpu.W_mxfp4_g128.cuda(),
        cpu.W_scale_g128.cuda(), cpu.W_q4.cuda(),
    )
    expected = run_o3_reference(cpu) if variant == "o3" else run_o4_reference(cpu)
    actual = run_o3(cuda, mode="compute_only") if variant == "o3" else run_o4(cuda, mode="compute_only")
    torch.cuda.synchronize()
    torch.testing.assert_close(actual["output"].cpu(), expected["output"], rtol=1e-3, atol=1e-3)


@pytest.mark.sm120
@pytest.mark.parametrize(
    "implementation",
    (
        "n16_k128",
        "n16_k256",
        "n32_k128",
        "n16_k128_dual",
        "n32_k256_dual",
        "n16_k128_swizzle",
        "n32_k128_swizzle",
        "m64_n16_k128",
        "m64_n32_k128",
        "m64_n16_k128_cute_ldsm",
        "n16_k128_cute_ldsm",
        "n16_k128_mrep2_cute_ldsm",
        "n16_k128_ldsm_scale_broadcast",
        "n16_k128_ldsm_factor_row_scale",
        "n32_k128_cute_ldsm",
        "n16_k128_ldsm_swizzle",
        "n16_k128_ldsm_split_chains",
    ),
)
def test_o3_internal_optimization_candidates_match_production(implementation):
    import torch

    from adangel.ops.extension import require_variant

    _, inputs = _inputs(m=33, n=35, k=256)
    native = require_variant("o3")
    baseline = native._benchmark_o3_impl(
        "n16_k128", "compute_only", inputs.A_int8, inputs.A_scale,
        inputs.W_mxfp4_g128, inputs.W_scale_g128, 1, 1, 100,
    )
    candidate = native._benchmark_o3_impl(
        implementation, "compute_only", inputs.A_int8, inputs.A_scale,
        inputs.W_mxfp4_g128, inputs.W_scale_g128, 1, 1, 100,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(candidate["output"], baseline["output"], rtol=1e-3, atol=1e-3)
    kernel = dict(candidate["kernel"])
    assert kernel["implementation_key"] == implementation
    assert kernel["partial_storage"] == "register"


@pytest.mark.sm120
@pytest.mark.parametrize(
    "implementation",
    (
        "n64_k256",
        "n64_k256_split2",
        "n64_k256_cache_b",
        "n64_k256_split2_cache_b",
        "m64_n64_k512",
        "m64_n64_k512_optimized",
    ),
)
def test_o4_internal_optimization_candidates_match_production(implementation):
    import torch

    from adangel.ops.extension import require_variant

    _, inputs = _inputs(m=67, n=69, k=512)
    native = require_variant("o4")
    baseline = native._benchmark_o4_impl(
        "n64_k256", "compute_only", inputs.A_int8, inputs.A_scale,
        inputs.W_mxfp4_g128, inputs.W_scale_g128, 1, 1, 100,
    )
    candidate = native._benchmark_o4_impl(
        implementation, "compute_only", inputs.A_int8, inputs.A_scale,
        inputs.W_mxfp4_g128, inputs.W_scale_g128, 1, 1, 100,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(candidate["output"], baseline["output"], rtol=1e-3, atol=1e-3)
    kernel = dict(candidate["kernel"])
    assert kernel["implementation_key"] == implementation
    assert kernel["partial_storage"] == "register"
