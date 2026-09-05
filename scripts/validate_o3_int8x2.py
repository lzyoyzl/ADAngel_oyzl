#!/usr/bin/env python3
"""Validate the diagnostic O3 Split implementation that uses two INT8 MMA paths."""

from __future__ import annotations

import argparse
import json
import statistics


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--conversion-inner-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-cv-percent", type=float, default=3.0)
    return parser.parse_args()


def summarize(values):
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": mean,
        "cv_percent": 0.0 if mean == 0 else statistics.pstdev(values) / mean * 100.0,
        "samples": len(values),
    }


def mse64(actual, reference):
    return float((actual.double() - reference.double()).square().mean().item())


def main() -> int:
    args = arguments()
    if (
        min(args.m, args.n, args.k) <= 0
        or args.m % 128
        or args.n % 64
        or args.k % 128
    ):
        raise SystemExit("diagnostic requires M%128=0, N%64=0, K%128=0")

    import torch

    from adangel.ops.extension import require_variant
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import (
        decode_ue8m0_tensor,
        mxfp4_to_q4_packed,
        quantize_mxfp4,
        unpack_int4_tensor,
    )

    generator = torch.Generator().manual_seed(args.seed)
    activation = torch.randn(
        (args.m, args.k), dtype=torch.float16, generator=generator
    )
    weight = torch.randn(
        (args.n, args.k), dtype=torch.float16, generator=generator
    )
    a_int8, a_scale = quantize_int8_per_row(activation)
    w_k32, w_scale_k32 = quantize_mxfp4(weight, group_size=32)
    w_g128, w_scale_g128 = quantize_mxfp4(weight, group_size=128)
    q4_packed = mxfp4_to_q4_packed(w_g128)
    q4 = unpack_int4_tensor(q4_packed)

    raw = a_int8.to(torch.int16) & 0xFF
    expected_low = (raw & 0xF).to(torch.int8)
    high_nibble = raw >> 4
    expected_high = torch.where(
        high_nibble >= 8, high_nibble - 16, high_nibble
    ).to(torch.int8)

    a_cuda = a_int8.cuda()
    a_scale_cuda = a_scale.cuda()
    w_g128_cuda = w_g128.cuda()
    w_scale_g128_cuda = w_scale_g128.cuda()
    q4_cuda = q4.cuda()
    decoded_scale = decode_ue8m0_tensor(w_scale_g128_cuda)
    expected = torch.zeros(
        (args.m, args.n), device="cuda", dtype=torch.float32
    )
    for group in range(args.k // 128):
        sl = slice(group * 128, (group + 1) * 128)
        partial = a_cuda[:, sl].float() @ q4_cuda[:, sl].float().T
        expected.add_(
            partial
            * (a_scale_cuda[:, None] * decoded_scale[:, group][None, :])
        )

    native = require_variant("o3")
    expected_stages = {
        "conversion_only": {"weight_conversion", "activation_conversion", "total"},
        "compute_only": {"gemm", "total"},
        "cold": {"weight_conversion", "activation_conversion", "gemm", "total"},
        "steady_state": {"activation_conversion", "gemm", "total"},
    }
    modes = {}
    payload = None
    for mode, stages in expected_stages.items():
        payload = native._benchmark_o3_split_int8x2(
            mode,
            a_cuda,
            a_scale_cuda,
            w_g128_cuda,
            w_scale_g128_cuda,
            args.warmup,
            args.repeats,
            args.conversion_inner_repeats,
        )
        timings = dict(payload["timings_ms"])
        if set(timings) != stages:
            raise RuntimeError(f"{mode}: unexpected timing stages {sorted(timings)}")
        torch.testing.assert_close(payload["output"], expected, rtol=1e-3, atol=1e-3)
        modes[mode] = {name: summarize(values) for name, values in timings.items()}

    assert payload is not None
    converted_low, converted_high = payload["converted_activation"]
    torch.testing.assert_close(
        payload["converted_weight"].cpu(), q4, rtol=0, atol=0
    )
    torch.testing.assert_close(converted_low.cpu(), expected_low, rtol=0, atol=0)
    torch.testing.assert_close(converted_high.cpu(), expected_high, rtol=0, atol=0)

    o0 = native.benchmark(
        "o0",
        "compute_only",
        a_cuda,
        a_scale_cuda,
        w_k32.cuda(),
        w_scale_k32.cuda(),
        1,
        1,
        args.conversion_inner_repeats,
    )
    torch.cuda.synchronize()
    kernel = dict(payload["kernel"])
    required = {
        "diagnostic_only": True,
        "production_selected": False,
        "o3_requirement_compliant": False,
        "tensor_core": True,
        "mma_family": "IMMA_INT8_X2",
        "ptx_mma_semantics": "S8xS8_low_and_S8xS8_high",
        "data_movement": "TMA",
        "kernel_schedule": "cooperative_warp_specialized",
        "partial_storage": "register",
        "group_size": 128,
    }
    for key, value in required.items():
        if kernel.get(key) != value:
            raise RuntimeError(f"kernel metadata {key}: {kernel.get(key)!r} != {value!r}")

    violations = []
    if (args.m, args.n, args.k) == (4096, 4096, 4096):
        for mode, stages in modes.items():
            for stage, stats in stages.items():
                if stats["cv_percent"] >= args.max_cv_percent:
                    violations.append(
                        f"{mode}/{stage} CV={stats['cv_percent']:.3f}%"
                    )
    result = {
        "passed": not violations,
        "shape": [args.m, args.n, args.k],
        "output_dtype": str(payload["output"].dtype),
        "max_abs_error_vs_o3_semantic_reference": float(
            (payload["output"] - expected).abs().max().item()
        ),
        "mse_vs_o3_semantic_reference": mse64(payload["output"], expected),
        "mse_vs_o0": mse64(payload["output"], o0["output"]),
        "kernel": kernel,
        "modes": modes,
        "violations": violations,
    }
    print(json.dumps(result, indent=2))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
