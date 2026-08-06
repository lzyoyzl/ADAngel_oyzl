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
    parser.add_argument("--seed", type=int, default=2026)
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


def main() -> int:
    args = parse_args()
    if min(args.m, args.n, args.k) <= 0 or args.k % 64:
        raise SystemExit("M/N/K must be positive and K must be divisible by 64")
    if args.m % 4 or args.n % 4:
        raise SystemExit("M and N must be divisible by 4 for the regular-order INT8 TC path")
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("warmup must be non-negative and repeats must be positive")

    import torch

    from adangel.ops.dispatch import run_o1
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
    actual = run_o1(inputs, mode="compute_only", backend="native")
    torch.cuda.synchronize()
    torch.testing.assert_close(actual["converted_weight"], expected_weight, rtol=0, atol=0)
    if actual["converted_activation"] is not None:
        raise RuntimeError("O1 must preserve A_int8 and must not return a converted activation")
    torch.testing.assert_close(actual["output"], expected_output, rtol=1e-3, atol=1e-3)
    if actual["output"].dtype != torch.float32 or not torch.isfinite(actual["output"]).all():
        raise RuntimeError("O1 output must be finite FP32")
    max_abs_error = float((actual["output"] - expected_output).abs().max().item())

    kernel = dict(actual["kernel"])
    required_metadata = {
        "library": "CUDA WMMA",
        "tensor_core": True,
        "mma_family": "IMMA",
        "mma_api": "nvcuda::wmma",
        "mma_shape": "m16n16k16",
        "implementation": "fused_tiled",
        "kernel_symbol": "adangel_o1_fused_tiled",
        "compute_type": "S8xS8_TO_S32",
        "input_dtype": "int8",
        "partial_dtype": "int32",
        "accumulation_dtype": "fp32",
        "output_dtype": "fp32",
        "group_size": 32,
        "global_partial_buffer": False,
        "output_stores_per_element": 1,
    }
    for key, value in required_metadata.items():
        if kernel.get(key) != value:
            raise RuntimeError(f"kernel metadata {key}: expected {value!r}, got {kernel.get(key)!r}")

    expected_stages = {
        "conversion_only": {"weight_conversion", "total"},
        "compute_only": {"gemm", "total"},
        "cold": {"weight_conversion", "gemm", "total"},
        "steady_state": {"gemm", "total"},
    }
    modes = {}
    for mode, stages in expected_stages.items():
        payload = native.benchmark(
            "o1",
            mode,
            inputs.A_int8,
            inputs.A_scale,
            inputs.W_mxfp4,
            inputs.W_scale,
            args.warmup,
            args.repeats,
        )
        torch.cuda.synchronize()
        timings = dict(payload["timings_ms"])
        if set(timings) != stages:
            raise RuntimeError(f"{mode}: expected stages {sorted(stages)}, got {sorted(timings)}")
        torch.testing.assert_close(payload["output"], expected_output, rtol=1e-3, atol=1e-3)
        modes[mode] = {
            stage: {
                "median_ms": statistics.median(float(value) for value in values),
                "samples": len(values),
            }
            for stage, values in timings.items()
        }

    report = {
        "passed": True,
        "shape": [args.m, args.n, args.k],
        "output_dtype": str(actual["output"].dtype),
        "max_abs_error": max_abs_error,
        "kernel": kernel,
        "modes": modes,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
