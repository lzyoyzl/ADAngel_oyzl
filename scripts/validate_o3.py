#!/usr/bin/env python3
"""Validate the O3 Split INT4 Tensor Core backend without a model trace."""

from __future__ import annotations

import argparse
import json
import statistics


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--conversion-inner-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-cv-percent", type=float, default=3.0)
    parser.add_argument(
        "--implementation",
        default="production",
        choices=(
            "production",
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
            "n16_k128_ldsm_biased_high_u4",
            "n32_k128_cute_ldsm",
            "n16_k128_ldsm_swizzle",
            "n16_k128_ldsm_split_chains",
        ),
    )
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


def main() -> int:
    args = arguments()
    if min(args.m, args.n, args.k) <= 0 or args.k % 128:
        raise SystemExit("M/N/K must be positive and K must be divisible by 128")
    import torch

    from adangel.ops.extension import require_variant
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import (
        decode_ue8m0_tensor,
        mxfp4_to_q4_packed,
        quantize_mxfp4,
        unpack_int4_tensor,
    )

    generator = torch.Generator().manual_seed(args.seed)
    activation = torch.randn((args.m, args.k), dtype=torch.float16, generator=generator)
    weight = torch.randn((args.n, args.k), dtype=torch.float16, generator=generator)
    a_int8, a_scale = quantize_int8_per_row(activation)
    w_g128, w_scale = quantize_mxfp4(weight, group_size=128)
    expected_q4 = mxfp4_to_q4_packed(w_g128)
    expected_split = split_int8_to_packed_int4(a_int8)
    if args.implementation == "n16_k128_ldsm_biased_high_u4":
        expected_split = expected_split.clone()
        expected_split[args.m :] ^= 0x88
    q4 = unpack_int4_tensor(expected_q4).cuda()
    a_cuda, a_scale_cuda = a_int8.cuda(), a_scale.cuda()
    scale_cuda = w_scale.cuda()
    expected = torch.zeros((args.m, args.n), device="cuda", dtype=torch.float32)
    decoded = decode_ue8m0_tensor(scale_cuda)
    for group in range(args.k // 128):
        sl = slice(group * 128, (group + 1) * 128)
        partial = a_cuda[:, sl].float() @ q4[:, sl].float().T
        expected.add_(partial * (a_scale_cuda[:, None] * decoded[:, group][None, :]))

    native = require_variant("o3")
    modes = {}
    payload = None
    expected_stages = {
        "conversion_only": {"weight_conversion", "activation_conversion", "total"},
        "compute_only": {"gemm", "total"},
        "cold": {"weight_conversion", "activation_conversion", "gemm", "total"},
        "steady_state": {"activation_conversion", "gemm", "total"},
    }
    for mode, stages in expected_stages.items():
        payload = native._benchmark_o3_impl(
            args.implementation, mode, a_cuda, a_scale_cuda, w_g128.cuda(), scale_cuda,
            args.warmup, args.repeats, args.conversion_inner_repeats,
        )
        timings = dict(payload["timings_ms"])
        if set(timings) != stages:
            raise RuntimeError(f"{mode}: unexpected timing stages {sorted(timings)}")
        torch.testing.assert_close(payload["output"], expected, rtol=1e-3, atol=1e-3)
        modes[mode] = {name: summarize(values) for name, values in timings.items()}

    torch.testing.assert_close(payload["converted_weight"].cpu(), expected_q4, rtol=0, atol=0)
    torch.testing.assert_close(payload["converted_activation"].cpu(), expected_split, rtol=0, atol=0)
    kernel = dict(payload["kernel"])
    required = {
        "tensor_core": True,
        "mma_family": "IMMA_INT4",
        "data_movement": "TMA",
        "kernel_schedule": "cooperative_warp_specialized",
        "partial_storage": "register",
        "group_size": 128,
    }
    for key, value in required.items():
        if kernel.get(key) != value:
            raise RuntimeError(f"kernel metadata {key}: {kernel.get(key)!r} != {value!r}")
    max_error = float((payload["output"] - expected).abs().max().item())
    violations = []
    if (args.m, args.n, args.k) == (4096, 4096, 4096):
        for mode, stages in modes.items():
            for stage, stats in stages.items():
                if stats["cv_percent"] >= args.max_cv_percent:
                    violations.append(f"{mode}/{stage} CV={stats['cv_percent']:.3f}%")
    print(json.dumps({
        "passed": not violations,
        "shape": [args.m, args.n, args.k],
        "implementation_requested": args.implementation,
        "output_dtype": str(payload["output"].dtype),
        "max_abs_error": max_error,
        "kernel": kernel,
        "modes": modes,
        "violations": violations,
    }, indent=2))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
