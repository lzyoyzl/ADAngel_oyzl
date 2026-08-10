#!/usr/bin/env python3
"""Validate the O2 native backend on synthetic data; no model or trace is required."""

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
    parser.add_argument("--max-cv-percent", type=float, default=3.0)
    return parser.parse_args()


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
    if min(args.m, args.n, args.k) <= 0 or args.k % 64:
        raise SystemExit("M/N/K must be positive and K must be divisible by 64")
    if args.m % 4 or args.n % 4:
        raise SystemExit("M and N must be divisible by 4 for the FP32 row-major epilogue")
    if args.warmup < 0 or args.repeats <= 0 or args.conversion_inner_repeats <= 1:
        raise SystemExit(
            "warmup must be non-negative, repeats must be positive, and "
            "conversion-inner-repeats must exceed one"
        )

    import torch

    from adangel.ops.dispatch import run_o2
    from adangel.ops.extension import require_variant
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4
    from adangel.reference import run_o2_reference
    from adangel.trace.schema import PreparedInputs

    native = require_variant("o2")
    generator = torch.Generator().manual_seed(args.seed)
    activation = torch.randn((args.m, args.k), generator=generator, dtype=torch.float16)
    weight = torch.randn((args.n, args.k), generator=generator, dtype=torch.float16)
    a_int8, a_scale = quantize_int8_per_row(activation)
    w_mxfp4, w_scale = quantize_mxfp4(weight)
    inputs = PreparedInputs(
        "synthetic_o2",
        a_int8.cuda(),
        a_scale.cuda(),
        w_mxfp4.cuda(),
        w_scale.cuda(),
    )

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = run_o2_reference(inputs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    actual = run_o2(inputs, mode="compute_only", backend="native")
    torch.cuda.synchronize()
    actual_values, actual_scales = actual["converted_activation"]
    expected_values, expected_scales = expected["converted_activation"]
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)
    if actual["converted_weight"] is not None:
        raise RuntimeError("O2 must keep W_mxfp4 and return no converted weight")
    torch.testing.assert_close(actual["output"], expected["output"], rtol=1e-3, atol=1e-3)
    if actual["output"].dtype != torch.float32 or not torch.isfinite(actual["output"]).all():
        raise RuntimeError("O2 output must be finite FP32")
    max_abs_error = float((actual["output"] - expected["output"]).abs().max().item())

    layout = dict(native.verify_layout())
    if not layout["passed"] or layout["max_abs_error"] > 1.0e-3:
        raise RuntimeError(f"single-warp MXFP4 scale-layout probe failed: {layout}")

    kernel = dict(actual["kernel"])
    required_metadata = {
        "library": "CUTLASS",
        "cutlass_commit": "db1c288993354c88e551c40c19a8fb93a774a241",
        "implementation": "cutlass_sm120_mxf4_tma_warp_specialized",
        "data_movement": "TMA",
        "kernel_schedule": "cooperative_warp_specialized",
        "stage_count_policy": "StageCountAutoCarveout",
        "cta_tile": (128, 128, 256),
        "cluster": (1, 1, 1),
        "tensor_core": True,
        "mma_family": "MXFP4_BLOCK_SCALED",
        "mma_shape": "m16n8k64",
        "input_dtype": "mxfp4_e2m1",
        "scale_dtype": "ue8m0",
        "scale_vector_size": 32,
        "accumulation_dtype": "fp32",
        "output_dtype": "fp32",
        "weight_scale_layout_repack": True,
        "weight_scale_repack_timing_method": "batched_cuda_event_average",
        "weight_scale_repack_timing_isolated": True,
        "weight_scale_repack_inner_repeats": 100,
        "activation_conversion_timing_method": "batched_cuda_event_average",
        "activation_conversion_timing_isolated": True,
        "activation_conversion_inner_repeats": 100,
        "total_timing_semantics": "conversion_only_amortized_cold_steady_direct",
        "global_partial_buffer": False,
        "output_stores_per_element": 1,
    }
    for key, value in required_metadata.items():
        observed = kernel.get(key)
        if isinstance(value, tuple):
            observed = tuple(observed)
        if observed != value:
            raise RuntimeError(f"kernel metadata {key}: expected {value!r}, got {observed!r}")

    expected_stages = {
        "conversion_only": {"weight_conversion", "activation_conversion", "total"},
        "compute_only": {"gemm", "total"},
        "cold": {"weight_conversion", "activation_conversion", "gemm", "total"},
        "steady_state": {"activation_conversion", "gemm", "total"},
    }
    modes = {}
    timing_methods = {}
    for mode, stages in expected_stages.items():
        payload = native.benchmark(
            "o2",
            mode,
            inputs.A_int8,
            inputs.A_scale,
            inputs.W_mxfp4,
            inputs.W_scale,
            args.warmup,
            args.repeats,
            args.conversion_inner_repeats,
        )
        torch.cuda.synchronize()
        timings = dict(payload["timings_ms"])
        if set(timings) != stages:
            raise RuntimeError(f"{mode}: expected stages {sorted(stages)}, got {sorted(timings)}")
        torch.testing.assert_close(payload["output"], expected["output"], rtol=1e-3, atol=1e-3)
        timing_methods[mode] = dict(payload["timing_method"])
        modes[mode] = {stage: summarize(values) for stage, values in timings.items()}

    formal_shape = (args.m, args.n, args.k) == (4096, 4096, 4096)
    violations = []
    if formal_shape:
        for mode, stages in modes.items():
            for stage, summary in stages.items():
                if summary["cv_percent"] >= args.max_cv_percent:
                    violations.append(
                        f"{mode}/{stage} CV {summary['cv_percent']:.3f}% must be below "
                        f"{args.max_cv_percent:.3f}%"
                    )

    report = {
        "passed": not violations,
        "shape": [args.m, args.n, args.k],
        "output_dtype": str(actual["output"].dtype),
        "max_abs_error": max_abs_error,
        "layout_probe": layout,
        "kernel": kernel,
        "timing_method_by_mode": timing_methods,
        "modes": modes,
        "performance_gate": {
            "applied": formal_shape,
            "max_cv_percent": args.max_cv_percent,
            "absolute_latency_limit_ms": None,
            "violations": violations,
        },
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
