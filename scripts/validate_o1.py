#!/usr/bin/env python3
"""Validate the O1 native backend on synthetic data; no model or trace is required."""

from __future__ import annotations

import argparse
import json
import statistics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--conversion-inner-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--implementation",
        choices=(
            "production",
            "shared_partial",
            "register_64x32",
            "register_64x32_scale_shared",
            "register_64x32_k64_scale_shared",
            "register_128x32_k64_scale_shared",
            "register_128x128",
        ),
        default="production",
    )
    parser.add_argument("--max-compute-ms", type=float)
    parser.add_argument("--max-cv-percent", type=float, default=3.0)
    return parser.parse_args()


def semantic_reference(inputs):
    """Scalable O1 reference whose FP32 K32 dot is exact for the bounded integer inputs."""
    import torch

    from adangel.quantization.mxfp4 import (
        GROUP_SIZE,
        decode_ue8m0_tensor,
        mxfp4_to_int8_base,
    )

    w_int8 = mxfp4_to_int8_base(inputs.W_mxfp4)
    m, k = inputs.A_int8.shape
    n = w_int8.shape[0]
    output = torch.zeros((m, n), dtype=torch.float32, device=inputs.A_int8.device)
    w_scales = decode_ue8m0_tensor(inputs.W_scale)
    for group in range(k // GROUP_SIZE):
        sl = slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)
        # Products and every K32 sum fit exactly in FP32 for INT8 x O1 base values.
        partial = inputs.A_int8[:, sl].float() @ w_int8[:, sl].float().transpose(0, 1)
        scale = inputs.A_scale[:, None] * w_scales[:, group][None, :] * 0.5
        output.add_(partial * scale)
    return output, w_int8


def summarize(values):
    numbers = sorted(float(value) for value in values)

    def percentile(q):
        position = (len(numbers) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(numbers) - 1)
        fraction = position - lower
        return numbers[lower] * (1.0 - fraction) + numbers[upper] * fraction

    mean = statistics.fmean(numbers)
    return {
        "median_ms": statistics.median(numbers),
        "p5_ms": percentile(0.05),
        "p95_ms": percentile(0.95),
        "iqr_ms": percentile(0.75) - percentile(0.25),
        "cv_percent": 0.0 if mean == 0.0 else statistics.pstdev(numbers) / mean * 100.0,
        "samples": len(numbers),
    }


def main() -> int:
    args = parse_args()
    if min(args.m, args.n, args.k) <= 0 or args.k % 32:
        raise SystemExit("M/N/K must be positive and K must be divisible by 32")
    if args.m % 4 or args.n % 4:
        raise SystemExit("M and N must be divisible by 4 for the regular-order INT8 TC path")
    if args.warmup < 0 or args.repeats <= 0 or args.conversion_inner_repeats <= 1:
        raise SystemExit(
            "warmup must be non-negative, repeats must be positive, and "
            "conversion-inner-repeats must exceed one"
        )

    import torch

    from adangel.ops.extension import require_variant
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4
    from adangel.trace.schema import PreparedInputs

    native = require_variant("o1")
    generator = torch.Generator().manual_seed(args.seed)
    activation = torch.randn((args.m, args.k), generator=generator, dtype=torch.float16)
    weight = torch.randn((args.n, args.k), generator=generator, dtype=torch.float16)
    a_int8, a_scale = quantize_int8_per_row(activation)
    w_mxfp4, w_scale = quantize_mxfp4(weight)
    inputs = PreparedInputs(
        "synthetic_o1",
        a_int8.cuda(),
        a_scale.cuda(),
        w_mxfp4.cuda(),
        w_scale.cuda(),
    )

    expected_output, expected_weight = semantic_reference(inputs)

    def benchmark(mode, warmup, repeats):
        if args.implementation == "production":
            return native.benchmark(
                "o1",
                mode,
                inputs.A_int8,
                inputs.A_scale,
                inputs.W_mxfp4,
                inputs.W_scale,
                warmup,
                repeats,
                args.conversion_inner_repeats,
            )
        return native._benchmark_o1_impl(
            args.implementation,
            mode,
            inputs.A_int8,
            inputs.A_scale,
            inputs.W_mxfp4,
            inputs.W_scale,
            warmup,
            repeats,
            args.conversion_inner_repeats,
        )

    actual = benchmark("compute_only", 0, 1)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual["converted_weight"], expected_weight, rtol=0, atol=0)
    if actual["converted_activation"] is not None:
        raise RuntimeError("O1 must preserve A_int8 and must not return a converted activation")
    torch.testing.assert_close(actual["output"], expected_output, rtol=1e-3, atol=1e-3)
    if actual["output"].dtype != torch.float32 or not torch.isfinite(actual["output"]).all():
        raise RuntimeError("O1 output must be finite FP32")
    max_abs_error = float((actual["output"] - expected_output).abs().max().item())

    kernel = dict(actual["kernel"])
    common_metadata = {
        "tensor_core": True,
        "mma_family": "IMMA",
        "data_movement": "TMA",
        "kernel_schedule": "cooperative_warp_specialized",
        "pipeline_stages": 3,
        "producer_warps": 1,
        "tma_operands": ("A_int8", "W_int8"),
        "compute_type": "S8xS8_TO_S32",
        "input_dtype": "int8",
        "partial_dtype": "int32",
        "accumulation_dtype": "fp32",
        "output_dtype": "fp32",
        "group_size": 32,
        "global_partial_buffer": False,
        "output_stores_per_element": 1,
    }
    selected_key = kernel.get("implementation_key")
    if args.implementation != "production" and selected_key != args.implementation:
        raise RuntimeError(
            f"requested {args.implementation}, native metadata selected {selected_key}"
        )
    if selected_key == "shared_partial":
        implementation_metadata = {
            "library": "CUTLASS CuTe + CUDA WMMA",
            "mma_api": "nvcuda::wmma",
            "mma_shape": "m16n16k16",
            "implementation": "tma_warp_specialized_shared_partial",
            "kernel_symbol": "adangel_o1_shared_partial_baseline",
            "consumer_warps": 8,
            "partial_storage": "shared_memory",
            "shared_partial_redistribution": True,
        }
    else:
        is_128 = selected_key == "register_128x128"
        is_128x32_k64 = selected_key == "register_128x32_k64_scale_shared"
        scale_shared = selected_key in {
            "register_64x32_scale_shared",
            "register_64x32_k64_scale_shared",
            "register_128x32_k64_scale_shared",
        }
        pipeline_k64 = selected_key in {
            "register_64x32_k64_scale_shared",
            "register_128x32_k64_scale_shared",
        }
        implementation_metadata = {
            "library": "CUTLASS CuTe + CUDA",
            "mma_api": "cute::MMA_Atom",
            "mma_atom": "SM80_16x8x32_S32S8S8S32_TN",
            "mma_shape": "m16n8k32",
            "implementation": (
                "tma_warp_specialized_register_partial_scale_shared"
                if scale_shared
                else "tma_warp_specialized_register_partial"
            ),
            "kernel_symbol": (
                "adangel_o1_register_partial_128x32_k64_scale_shared"
                if is_128x32_k64
                else (
                    "adangel_o1_register_partial_64x32_k64_scale_shared"
                    if pipeline_k64
                    else (
                        "adangel_o1_register_partial_64x32_scale_shared"
                        if scale_shared
                        else (
                            "adangel_o1_register_partial_128x128"
                            if is_128
                            else "adangel_o1_register_partial_64x32"
                        )
                    )
                )
            ),
            "consumer_warps": 16 if (is_128 or is_128x32_k64) else 8,
            "cta_tile": (
                (128, 128, 32)
                if is_128
                else (
                    (128, 32, 64)
                    if is_128x32_k64
                    else (64, 32, 64 if pipeline_k64 else 32)
                )
            ),
            "partial_storage": "register",
            "shared_partial_redistribution": False,
            "column_scale_load_scope": (
                "cta_once_per_column_group" if scale_shared else "consumer_warp"
            ),
            "column_scale_storage": (
                "stage_local_shared_fp32" if scale_shared else "warp_register"
            ),
            "fp32_accumulation_op": "fma_rn" if scale_shared else "mul_then_add_rn",
            "groups_per_pipeline_stage": 2 if pipeline_k64 else 1,
            "pipeline_tile_k": 64 if pipeline_k64 else 32,
        }
    for key, value in {**common_metadata, **implementation_metadata}.items():
        if kernel.get(key) != value:
            raise RuntimeError(f"kernel metadata {key}: expected {value!r}, got {kernel.get(key)!r}")

    expected_stages = {
        "conversion_only": {"weight_conversion", "total"},
        "compute_only": {"gemm", "total"},
        "cold": {"weight_conversion", "gemm", "total"},
        "steady_state": {"gemm", "total"},
    }
    modes = {}
    timing_methods = {}
    for mode, stages in expected_stages.items():
        payload = benchmark(mode, args.warmup, args.repeats)
        torch.cuda.synchronize()
        timings = dict(payload["timings_ms"])
        if set(timings) != stages:
            raise RuntimeError(f"{mode}: expected stages {sorted(stages)}, got {sorted(timings)}")
        torch.testing.assert_close(payload["output"], expected_output, rtol=1e-3, atol=1e-3)
        timing_methods[mode] = dict(payload["timing_method"])
        modes[mode] = {stage: summarize(values) for stage, values in timings.items()}

    formal_shape = (args.m, args.n, args.k) == (4096, 4096, 4096)
    compute = modes["compute_only"]["gemm"]
    violations = []
    if (
        formal_shape
        and args.max_compute_ms is not None
        and compute["median_ms"] > args.max_compute_ms
    ):
        violations.append(
            f"compute-only median {compute['median_ms']:.6f} ms exceeds "
            f"{args.max_compute_ms:.6f} ms"
        )
    if formal_shape and compute["cv_percent"] >= args.max_cv_percent:
        violations.append(
            f"compute-only CV {compute['cv_percent']:.3f}% must be below "
            f"{args.max_cv_percent:.3f}%"
        )

    report = {
        "passed": not violations,
        "requested_implementation": args.implementation,
        "shape": [args.m, args.n, args.k],
        "output_dtype": str(actual["output"].dtype),
        "max_abs_error": max_abs_error,
        "kernel": kernel,
        "timing_method_by_mode": timing_methods,
        "modes": modes,
        "performance_gate": {
            "applied": formal_shape,
            "max_compute_ms": args.max_compute_ms,
            "max_cv_percent": args.max_cv_percent,
            "violations": violations,
        },
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
