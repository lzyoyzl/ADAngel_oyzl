#!/usr/bin/env python3
"""Validate the O0 native backend on synthetic data; no model or trace is required."""

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


def main() -> int:
    args = parse_args()
    if min(args.m, args.n, args.k) <= 0 or args.k % 64:
        raise SystemExit("M/N/K must be positive and K must be divisible by 64")
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("warmup must be non-negative and repeats must be positive")

    import torch

    from adangel.ops.dispatch import run_o0
    from adangel.ops.extension import require_variant
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4
    from adangel.reference import run_o0_reference
    from adangel.trace.schema import PreparedInputs

    native = require_variant("o0")
    generator = torch.Generator().manual_seed(args.seed)
    activation = torch.randn((args.m, args.k), generator=generator, dtype=torch.float16)
    weight = torch.randn((args.n, args.k), generator=generator, dtype=torch.float16)
    a_int8, a_scale = quantize_int8_per_row(activation)
    w_mxfp4, w_scale = quantize_mxfp4(weight)
    inputs = PreparedInputs(
        "synthetic_o0",
        a_int8.cuda(),
        a_scale.cuda(),
        w_mxfp4.cuda(),
        w_scale.cuda(),
    )

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
    max_abs_error = float((actual["output"] - expected["output"]).abs().max().item())

    expected_stages = {
        "conversion_only": {"weight_conversion", "activation_conversion", "total"},
        "compute_only": {"gemm", "total"},
        "cold": {"weight_conversion", "activation_conversion", "gemm", "total"},
        "steady_state": {"activation_conversion", "gemm", "total"},
    }
    modes = {}
    for mode, stages in expected_stages.items():
        payload = native.benchmark(
            "o0",
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
        torch.testing.assert_close(payload["output"], expected["output"], rtol=1e-3, atol=1e-3)
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
        "kernel": dict(actual["kernel"]),
        "modes": modes,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
